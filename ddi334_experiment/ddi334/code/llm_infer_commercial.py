#!/usr/bin/env python3
"""DDI-334 상용 LLM(GPT-4o / DeepSeek V4 / Claude Sonnet 5 / Gemini 3 Pro) S2 추론+평가
(V2 binary-with-R 재사용).

기존 llm_infer.py의 infer_v2와 동일한 프롬프트(DDI334Prompt.build_v2)/평가(evaluate)를
그대로 재사용하되, vLLM 로컬 서버 대신 실제 상용 API를 호출한다. 4개 provider 전부
OpenAI 호환 chat.completions 엔드포인트를 제공해서 클라이언트 코드는 공용.

- P(Yes) 추출 방식은 provider별 logprobs 신뢰도에 따라 다름 (PROVIDERS[.]["logprobs"] 참조):
  * True  (openai만): top_logprobs에서 Yes/No 확률을 softmax 정규화 -> 연속확률.
    AUROC/AUPRC/AP@50 전부 정상 계산됨.
  * False (deepseek/anthropic/gemini): 하드 Yes/No 텍스트만 사용 -> p_yes는 0.0/1.0.
    DeepSeek는 logprobs=True를 보내도 실제로는 선택토큰=0.0/나머지=-9999.0 더미값만 반환
    (deepseek-ai/DeepSeek-V3.2-Exp #49 등 커뮤니티에서도 반복 보고된 플랫폼 한계, V4도 동일 확인,
    2026-08). Anthropic/Gemini는애초에 logprobs 미지원. 이 3개는 프로젝트 기존 관례
    (gen_resulttable.py 상단 주석: "LLM ROC-AUC == accuracy 하드라벨 -> balanced-acc" /
    "AP@50은 하드라벨이라 N/A")를 그대로 따라 Acc/F1을 주 지표로, AP@50은 N/A, AUROC/AUPRC는
    balanced-acc 각주와 함께 참고치로만 보고한다.
- 캐시: results/{outdir}/llm_cache/commercial/{provider}_{model}_{mode}_{dataset}_{split}.jsonl 에
  완료된 (d1,d2,t)별 응답을 append-only로 저장 -> 재실행 시 이미 낸 task는 API 재호출(과금) 없이 skip.
- DeepSeek는 non-thinking 모드로 고정 호출 (thinking이 기본값이라 명시 안 하면 reasoning 켜짐).

Usage (vllm-llm env, 사용할 provider의 *_API_KEY 필요):
  python llm_infer_commercial.py --provider openai    --dataset ddibn --split S2 --mode real  --label gpt4o_zs_real       --outdir ../results/commercial_llm
  python llm_infer_commercial.py --provider deepseek   --dataset ddibn --split S2 --mode ideal --label deepseekv4_zs_ideal --outdir ../results/commercial_llm
  python llm_infer_commercial.py --provider anthropic  --dataset ddibn --split S2 --mode real  --label claude_zs_real      --outdir ../results/commercial_llm
  python llm_infer_commercial.py --provider gemini     --dataset ddibn --split S2 --mode ideal --label gemini_zs_ideal     --outdir ../results/commercial_llm

  # 실제 API 호출 없이 비용만 미리 계산:
  python llm_infer_commercial.py --provider openai --dataset ddibn --split S2 --mode real --dry-run
"""
import argparse, functools, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Event

import anthropic
import numpy as np
from openai import OpenAI

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'code'))
from llm_prompts import DDI334Prompt
from llm_infer import load_rows, evaluate, save_results

# provider -> (base_url, api_key env var, logprobs 신뢰 가능 여부, json_schema 강제 지원, reasoning_effort).
# None base_url = OpenAI 공식 엔드포인트. 다른 3개는 각 사 OpenAI-호환 엔드포인트.
PROVIDERS = {
    "openai":    {"base_url": None,                                            "api_key_env": "OPENAI_API_KEY",    "logprobs": True,  "json_schema": True,  "reasoning_effort": None},
    "deepseek":  {"base_url": "https://api.deepseek.com/v1",                   "api_key_env": "DEEPSEEK_API_KEY",  "logprobs": False, "json_schema": False, "reasoning_effort": None},  # response_format=json_schema "unavailable now" (확인됨, 2026-08)
    "anthropic": {"base_url": None,                                            "api_key_env": "ANTHROPIC_API_KEY", "logprobs": False, "json_schema": False, "reasoning_effort": None},  # claude-sonnet-5, 네이티브 anthropic SDK(call_one_anthropic) 전용 경로 사용 -> base_url/json_schema 플래그 미사용. thinking=disabled 필수(라이브 검증됨, 2026-08-21)
    "gemini":    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "api_key_env": "GEMINI_API_KEY", "logprobs": False, "json_schema": True, "reasoning_effort": "minimal"},  # gemini-3-flash-preview, 라이브 검증됨(2026-08-21): minimal=thinking 토큰 0
}
# 대략적 가격($/1M token, 2026-08 기준) — dry-run 비용 추정용. 실제 청구와는 오차 있을 수 있음.
PRICE_PER_M = {
    "openai":    {"in": 2.50, "out": 10.00},
    "deepseek":  {"in": 0.44, "out": 1.32},   # peak 기준(보수적 상한); off-peak면 더 저렴
    "anthropic": {"in": 2.00, "out": 10.00},  # Claude Sonnet 5
    "gemini":    {"in": 0.50, "out": 3.00},   # Gemini 3 Flash (minimal reasoning)
}


def build_tasks(rows):
    """rows(list of (d1,d2,vec,pol)) -> [(row_idx, d1, d2, type_id), ...] (vec==1인 type만, infer_v2와 동일)."""
    tasks = []
    for r, (d1, d2, vec, pol) in enumerate(rows):
        for t in np.where(vec == 1)[0].tolist():
            tasks.append((r, d1, d2, t))
    return tasks


def load_cache(cache_path):
    done = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[(o["d1"], o["d2"], o["t"])] = o["p_yes"]
    return done


# vLLM guided_choice와 동일한 효과: 답을 {"answer":"Yes"} / {"answer":"No"}로 강제.
# "##Answer:" 같은 프리픽스를 모델이 못 쓰게 막아서, max_tokens 부족으로 답 전에 잘리는
# 문제(2026-08-21 GPT-4o 전체 재실행 원인)를 원천 차단한다.
DDI_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ddi_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["Yes", "No"]}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


def _parse_hard_label(content):
    """JSON 파싱 우선, 실패하면 텍스트에서 단어경계 yes/no 검색. 둘 다 실패하면 None(파싱 실패)."""
    content = content or ""
    try:
        obj = json.loads(content)
        ans = str(obj.get("answer", "")).strip().lower()
        if ans.startswith("yes"):
            return 1.0
        if ans.startswith("no"):
            return 0.0
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    m = re.search(r'\b(yes|no)\b', content, re.IGNORECASE)
    if m:
        return 1.0 if m.group(1).lower() == "yes" else 0.0
    return None


def call_one(client, model, sys_t, usr, extra_body=None, use_logprobs=True, use_json_schema=True,
             reasoning_effort=None):
    """P(Yes) 추출.
    use_json_schema=True: response_format=json_schema(strict, enum=[Yes,No])로 답을
      {"answer":"Yes"|"No"}로 강제 -> 자유 텍스트 프리픽스로 토큰을 낭비할 수 없음.
    use_logprobs=True(openai만): 생성된 모든 위치를 훑어 "선택된 토큰이 yes/no로 시작하는
      위치"를 찾고, 그 위치의 top_logprobs로 softmax 정규화 -> 연속확률
      (position 고정 가정 안 함 -> JSON 스키마 토큰화가 모델/프롬프트별로 달라져도 안전).
    use_logprobs=False(deepseek/anthropic/gemini): 하드 0.0/1.0. 파싱 완전 실패시 None
      반환(호출부에서 실패로 집계, 침묵 편향 방지 위해 0.0으로 임의 대체하지 않음).
    reasoning_effort: Gemini 3 계열은 기본이 hidden thinking 토큰을 씀(과금 대상) ->
      'minimal'로 강제하면 thinking 토큰 0 확인됨(2026-08-21 라이브 검증, Pro는 LOW가
      최저라 완전 차단 불가하지만 Flash는 minimal로 완전히 꺼짐)."""
    msgs = [{"role": "system", "content": sys_t}, {"role": "user", "content": usr}]
    kwargs = dict(model=model, messages=msgs, temperature=0.0, max_tokens=100,
                  extra_body=extra_body or {})
    if use_json_schema:
        kwargs["response_format"] = DDI_ANSWER_SCHEMA
    if use_logprobs:
        kwargs.update(logprobs=True, top_logprobs=20)
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = client.chat.completions.create(**kwargs)
    ch = resp.choices[0]

    if use_logprobs and ch.logprobs and ch.logprobs.content:
        for tokinfo in ch.logprobs.content:
            chosen = tokinfo.token.strip().lower()
            if chosen.startswith("yes") or chosen.startswith("no"):
                lp_yes = lp_no = None
                for tl in tokinfo.top_logprobs:
                    tok = tl.token.strip().lower()
                    if lp_yes is None and tok.startswith("yes"):
                        lp_yes = tl.logprob
                    if lp_no is None and tok.startswith("no"):
                        lp_no = tl.logprob
                if lp_yes is not None and lp_no is not None:
                    mx = max(lp_yes, lp_no)
                    ey, en = np.exp(lp_yes - mx), np.exp(lp_no - mx)
                    return float(ey / (ey + en))
                if lp_yes is not None:
                    return float(np.exp(lp_yes))
                if lp_no is not None:
                    return float(1.0 - np.exp(lp_no))
                break

    hard = _parse_hard_label(ch.message.content)
    if hard is None:
        raise ValueError(f"파싱 실패: content={ch.message.content!r}")
    return hard


# Claude 전용 스키마(OpenAI json_schema 래퍼 없이 순수 schema만 -> output_config 형식).
_ANSWER_SCHEMA_PLAIN = DDI_ANSWER_SCHEMA["json_schema"]["schema"]


def call_one_anthropic(client, model, sys_t, usr):
    """Claude 전용 경로 (anthropic SDK 네이티브 /v1/messages, OpenAI 클라이언트 아님).
    OpenAI 호환 레이어는 strict 스키마 준수가 보장 안 된다는 공식 경고가 있어
    (docs: "For guaranteed schema conformance ... use the native Claude API") 네이티브
    output_config를 직접 씀.
    thinking={"type":"disabled"} 필수: 안 하면 claude-sonnet-5가 가끔 예측 불가능하게
    자동으로 reasoning을 켜서 max_tokens를 전부 thinking에 쓰고 답 자체를 못 낼 수 있음
    (2026-08-21 라이브 검증: 8건 중 2건이 이렇게 실패, disabled로 8/8 해결됨)."""
    resp = client.messages.create(
        model=model, max_tokens=50, system=sys_t,
        messages=[{"role": "user", "content": usr}],
        thinking={"type": "disabled"},
        extra_body={"output_config": {"format": {"type": "json_schema", "schema": _ANSWER_SCHEMA_PLAIN}}},
    )
    text = next((b.text for b in resp.content if b.type == "text"), None)
    hard = _parse_hard_label(text)
    if hard is None:
        raise ValueError(f"파싱 실패: content={resp.content!r}")
    return hard


def infer_commercial(client, model, pb, rows, mode, cache_path, workers=8, call_fn=None, extra_body=None,
                      use_logprobs=True, use_json_schema=True, reasoning_effort=None):
    call_fn = call_fn or functools.partial(call_one, extra_body=extra_body, use_logprobs=use_logprobs,
                                            use_json_schema=use_json_schema, reasoning_effort=reasoning_effort)
    N = len(pb.typenames)
    pred = np.zeros((len(rows), N))
    lab = np.zeros((len(rows), N + 1))
    for r, (d1, d2, vec, pol) in enumerate(rows):
        lab[r, :N] = vec
        lab[r, -1] = pol

    done = load_cache(cache_path)
    tasks = build_tasks(rows)
    todo = [(r, d1, d2, t) for (r, d1, d2, t) in tasks if (d1, d2, t) not in done]
    for (r, d1, d2, t) in tasks:
        if (d1, d2, t) in done:
            pred[r, t] = done[(d1, d2, t)]

    print(f"[commercial] total_tasks={len(tasks)} cached={len(tasks)-len(todo)} todo={len(todo)}", flush=True)

    # 재시도해도 절대 성공 못하는 영구 에러(쿼터 소진/인증 실패 등) 시그니처.
    # 이런 에러는 backoff sleep 없이 즉시 포기하고, 전체 실행도 즉시 중단한다
    # (남은 task를 계속 던지면 시간만 낭비하고 rate-limit만 더 유발함 -> 과금은 안 되지만 비효율적).
    PERMANENT_ERRS = ("insufficient_quota", "credit_balance_exhausted", "invalid_api_key",
                       "authentication", "account_deactivated",
                       "credit balance is too low",           # Anthropic 크레딧 소진 (2026-08-21 확인)
                       "exceeded your current quota",          # Gemini RESOURCE_EXHAUSTED(일일 한도, 2026-08-21 확인)
                       "RESOURCE_EXHAUSTED")

    fails = [0]; ndone = [0]
    lock = Lock()
    abort = Event()
    cache_f = open(cache_path, "a")

    def one(task):
        r, d1, d2, t = task
        if abort.is_set():
            return
        sys_t, usr = pb.build_v2(d1, d2, t, mode=mode)
        for attempt in range(3):
            try:
                p_yes = call_fn(client, model, sys_t, usr)
                pred[r, t] = p_yes
                with lock:
                    cache_f.write(json.dumps({"d1": d1, "d2": d2, "t": t, "p_yes": p_yes}) + "\n")
                    cache_f.flush()
                    ndone[0] += 1
                    if ndone[0] % 500 == 0:
                        print(f"  [commercial] {ndone[0]}/{len(todo)} 완료", flush=True)
                return
            except Exception as e:
                permanent = any(sig in str(e) for sig in PERMANENT_ERRS)
                if permanent:
                    with lock:
                        if not abort.is_set():
                            print(f"[commercial] 영구 에러 감지 -> 즉시 중단: {e}", flush=True)
                        abort.set()
                        fails[0] += 1
                    return
                if attempt == 2:
                    with lock:
                        fails[0] += 1
                        if fails[0] <= 10:
                            print(f"  [commercial r{r} t{t}] FAIL {e}", flush=True)
                else:
                    time.sleep(2 * (attempt + 1))

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, todo))
    cache_f.close()
    if abort.is_set():
        raise SystemExit(f"[commercial] 영구 에러로 중단됨 ({fails[0]}건). API 키/크레딧 상태를 확인 후 재실행하세요 "
                          f"(캐시된 성공분은 재실행 시 자동 재사용됨).")
    if fails[0]:
        print(f"[commercial] 총 {fails[0]}/{len(todo)} task 실패 (재실행하면 캐시 skip하고 실패분만 재시도)", flush=True)
    return pred, lab


def dry_run_cost(provider, pb, rows, mode):
    """API 호출 없이, 실제 프롬프트를 tiktoken으로 인코딩해 예상 비용만 출력."""
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    tasks = build_tasks(rows)
    total_in = 0
    for (r, d1, d2, t) in tasks:
        sys_t, usr = pb.build_v2(d1, d2, t, mode=mode)
        total_in += len(enc.encode(sys_t + "\n\n" + usr))
    total_out = len(tasks) * 3  # max_tokens=3
    price = PRICE_PER_M[provider]
    cost = total_in / 1e6 * price["in"] + total_out / 1e6 * price["out"]
    print(f"[dry-run] provider={provider} mode={mode} tasks={len(tasks)} "
          f"in_tok={total_in} out_tok(est)={total_out} -> est_cost=${cost:.2f}")
    return cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=list(PROVIDERS))
    ap.add_argument("--model", default="")  # 비면 provider 기본값(openai=gpt-4o, deepseek=deepseek-chat)
    ap.add_argument("--dataset", default="ddibn")
    ap.add_argument("--split", default="S2")
    ap.add_argument("--mode", default="real", choices=["real", "ideal"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    default_model = {"openai": "gpt-4o", "deepseek": "deepseek-v4-flash",
                      "anthropic": "claude-sonnet-5", "gemini": "gemini-3-flash-preview"}[a.provider]
    model = a.model or default_model

    pb = DDI334Prompt(a.dataset)
    rows = load_rows(a.dataset, a.split, a.limit)
    print(f"[{a.provider}/{model}/{a.dataset}/{a.split}/{a.mode}] rows={len(rows)}")

    if a.dry_run:
        dry_run_cost(a.provider, pb, rows, a.mode)
        return

    cfg = PROVIDERS[a.provider]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise SystemExit(f"환경변수 {cfg['api_key_env']} 가 설정되어 있지 않습니다. "
                          f"export {cfg['api_key_env']}=... 후 다시 실행하세요.")
    if a.provider == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
    else:
        client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    label = a.label or f"commercial_{a.provider}_{model.replace('/', '_')}_{a.mode}"
    rdir = a.outdir or os.path.join(HERE, 'results', a.dataset)
    rpdir = os.path.join(rdir, 'raw_preds'); os.makedirs(rpdir, exist_ok=True)
    cache_dir = os.path.join(rdir, 'llm_cache', 'commercial'); os.makedirs(cache_dir, exist_ok=True)
    # 캐시 키는 --label이 아니라 실제 API 설정(provider/model/mode/split/dataset)에 묶는다.
    # -> 스모크 테스트(--label smoketest_...)에서 이미 낸 (d1,d2,t) 응답을 --label이 다른
    #    본 실행(--label commercial_...)이 그대로 재사용해 중복 과금을 피한다.
    cache_key = f"{a.provider}_{model.replace('/', '_')}_{a.mode}_{a.dataset}"
    cache_path = os.path.join(cache_dir, f"{cache_key}_{a.split}.jsonl")

    # DeepSeek V4는 기본이 thinking(reasoning) 모드 -> logprobs 요청이 조용히 무시됨.
    # non-thinking으로 명시 고정 (DeepSeek API "Thinking Mode" 문서, 2026-08 기준).
    extra_body = {"thinking": {"type": "disabled"}} if a.provider == "deepseek" else None

    call_fn = call_one_anthropic if a.provider == "anthropic" else None

    t0 = time.time()
    pred, lab = infer_commercial(client, model, pb, rows, a.mode, cache_path,
                                  workers=a.workers, call_fn=call_fn, extra_body=extra_body,
                                  use_logprobs=cfg["logprobs"], use_json_schema=cfg["json_schema"],
                                  reasoning_effort=cfg["reasoning_effort"])
    metrics = evaluate(pred, lab)
    wall = time.time() - t0
    print(f"[RESULT] {metrics} | wall {wall:.0f}s")
    save_results(a.dataset, label, a.split, metrics, wall, outdir=a.outdir or None)
    np.savez(os.path.join(rpdir, f'{label}_{a.split}.npz'), pred=pred, lab=lab)
    print(f"[SAVED] summary.csv + llm_metrics/{label}_{a.split}.json + raw_preds/{label}_{a.split}.npz")


if __name__ == '__main__':
    main()
