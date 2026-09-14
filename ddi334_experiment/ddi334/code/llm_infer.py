#!/usr/bin/env python3
"""DDI-334 LLM 셀 추론+평가 (V1 multi-label JSON / V2 binary-with-R).

- 입력 프롬프트: llm_prompts.DDI334Prompt (02 §3.1)
- queried side effect(vec=1)만 질의 (전 실험과 동일)
- 평가: train.py evaluate와 동일한 per-type ROC-AUC/PR-AUC/Acc (+ V1은 macro-F1)
- vLLM OpenAI 서버 (vllm-llm env에서 실행)

Usage (vllm-llm env):
  python llm_infer.py --version v1 --dataset ddibn --model Qwen/Qwen2.5-3B-Instruct \
      --gpu 3 --split S0 --limit 16        # smoke
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
from openai import OpenAI
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from llm_prompts import DDI334Prompt

PORT = 12341
CACHE = os.path.join(HERE, 'results', 'llm_cache')
os.makedirs(CACHE, exist_ok=True)


def start_server(model, gpu, merge_system, mem_util=0.85):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["VLLM_USE_V1"] = "0"
    if merge_system:
        env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"; env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--host", "127.0.0.1", "--port", str(PORT),
           "--gpu-memory-utilization", str(mem_util), "--enforce-eager",
           "--max-model-len", "4096" if merge_system else "8192", "--disable-log-requests"]
    log = open(os.path.join(CACHE, f"server_{model.replace('/','_')}.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=log), log


def wait_ready(timeout=600):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3); return True
        except Exception:
            time.sleep(5)
    return False


def load_rows(ds, split, limit):
    rows = []
    for line in open(os.path.join(HERE, 'data', ds, f'test_{split}.txt')):
        p = line.split()
        if len(p) != 4:
            continue
        vec = np.array([int(x) for x in p[2].split(',')], dtype=np.int64)
        rows.append((int(p[0]), int(p[1]), vec, int(p[3])))
        if limit and len(rows) >= limit:
            break
    return rows


def infer_v1(client, model, pb, rows, merge, workers=24):
    """동시 요청(ThreadPoolExecutor) -> vLLM 배치 처리. (직렬은 Running:1로 너무 느림)"""
    from concurrent.futures import ThreadPoolExecutor
    N = len(pb.typenames)
    pred = np.zeros((len(rows), N)); lab = np.zeros((len(rows), N + 1))
    fails = [0]

    def one(r):
        d1, d2, vec, pol = rows[r]
        qids = np.where(vec == 1)[0].tolist()
        lab[r, :N] = vec; lab[r, -1] = pol
        sys_t, usr, schema = pb.build_v1(d1, d2, qids)
        msgs = ([{"role": "user", "content": f"{sys_t}\n\n{usr}"}] if merge
                else [{"role": "system", "content": sys_t}, {"role": "user", "content": usr}])
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.0,
                                                      max_tokens=1024, seed=123,  # K_max=128 -> 최악 JSON ~824tok, 512는 잘림
                                                      extra_body={"guided_json": schema})
                obj = json.loads(resp.choices[0].message.content)
                for t in qids:
                    pred[r, t] = 1.0 if obj.get(str(t)) else 0.0
                return
            except Exception as e:
                if attempt == 1:
                    fails[0] += 1
                    if fails[0] <= 5:
                        print(f"  [v1 row{r}] FAIL {e}")
                else:
                    time.sleep(1)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, range(len(rows))))
    if fails[0]:
        print(f"  [v1] 총 {fails[0]}/{len(rows)} 행 실패")
    return pred, lab


def infer_v2(client, model, pb, rows, merge, mode='ideal', workers=24):
    """동시 요청(ThreadPoolExecutor) -> vLLM 배치. 직렬은 Running:1이라 GPU idle+극저속.
    각 (pair, t)별 Yes/No 1건. temperature=0 결정적이므로 병렬해도 결과 불변.
    mode: ideal / ideal_genes / real (정보 가용성 ablation)."""
    from concurrent.futures import ThreadPoolExecutor
    N = len(pb.typenames)
    pred = np.zeros((len(rows), N)); lab = np.zeros((len(rows), N + 1))
    tasks = []
    for r, (d1, d2, vec, pol) in enumerate(rows):
        lab[r, :N] = vec; lab[r, -1] = pol
        for t in np.where(vec == 1)[0].tolist():
            tasks.append((r, d1, d2, t))
    fails = [0]; done = [0]

    def one(task):
        r, d1, d2, t = task
        sys_t, usr = pb.build_v2(d1, d2, t, mode=mode)
        msgs = ([{"role": "user", "content": f"{sys_t}\n\n{usr}"}] if merge
                else [{"role": "system", "content": sys_t}, {"role": "user", "content": usr}])
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.0,
                                                      max_tokens=3, seed=123, logprobs=True, top_logprobs=20,
                                                      extra_body={"guided_choice": ["Yes", "No"]})
                ch = resp.choices[0]
                # Yes·No 둘 다 logprob 추출 -> softmax over {Yes,No} 정규화 = 인코더 sigmoid 대응.
                lp_yes = lp_no = None
                if ch.logprobs and ch.logprobs.content:
                    for tl in ch.logprobs.content[0].top_logprobs:
                        tok = tl.token.strip().lower()
                        if lp_yes is None and tok.startswith("yes"): lp_yes = tl.logprob
                        if lp_no is None and tok.startswith("no"):  lp_no = tl.logprob
                if lp_yes is not None and lp_no is not None:
                    mx = max(lp_yes, lp_no)
                    ey, en = np.exp(lp_yes - mx), np.exp(lp_no - mx)
                    p_yes = float(ey / (ey + en))           # P(Yes | Yes,No)
                elif lp_yes is not None:
                    p_yes = float(np.exp(lp_yes))           # No가 top-k 밖 -> Yes 압도
                elif lp_no is not None:
                    p_yes = float(1.0 - np.exp(lp_no))      # Yes가 top-k 밖 -> No 압도
                else:
                    p_yes = 1.0 if ch.message.content.strip().lower().startswith("yes") else 0.0
                pred[r, t] = p_yes
                done[0] += 1
                if done[0] % 2000 == 0:
                    print(f"  [v2] {done[0]}/{len(tasks)} 질의 완료", flush=True)
                return
            except Exception as e:
                if attempt == 1:
                    fails[0] += 1
                    if fails[0] <= 5:
                        print(f"  [v2 row{r} t{t}] FAIL {e}", flush=True)
                else:
                    time.sleep(1)

    print(f"  [v2] 총 {len(tasks)} 질의 ({len(rows)} pair) 병렬 시작 (workers={workers})", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, tasks))
    if fails[0]:
        print(f"  [v2] 총 {fails[0]}/{len(tasks)} 질의 실패", flush=True)
    return pred, lab


def evaluate(pred, lab):
    """train.py evaluate와 동일 per-type AUROC/PR-AUC/accuracy/macro-F1 + P@50/P@(N/2)."""
    N = pred.shape[1]; roc, prc, acc, f1, p50, prn = [], [], [], [], [], []
    for j in range(N):
        where = np.where(lab[:, j] == 1)[0]
        if len(where) == 0:
            continue
        pc = pred[where, j]; lc = lab[where, j] * lab[where, -1]
        if 0 < lc.sum() < len(lc):
            phard = (pc > 0.5).astype(float)
            roc.append(roc_auc_score(lc, pc)); prc.append(average_precision_score(lc, pc))
            acc.append(accuracy_score(lc, phard))
            f1.append(f1_score(lc, phard, zero_division=0))
            lc_sorted = lc[np.argsort(-pc)]   # 점수 내림차순 -> 상위 k개 양성비율 (train.py와 동일)
            k50 = min(50, lc_sorted.shape[0]); krn = max(1, lc_sorted.shape[0] // 2)
            p50.append(float(lc_sorted[:k50].mean())); prn.append(float(lc_sorted[:krn].mean()))
    m = lambda x: float(np.mean(x)) if x else 0.0
    return {"AUC-ROC": m(roc), "PR-AUC": m(prc), "accuracy": m(acc), "macro-F1": m(f1),
            "P@50": m(p50), "P@N2": m(prn), "n_types": len(roc)}


SUMMARY_COLS = ["best_epoch", "cell", "checkpoint_metric", "dataset", "n_params", "seed",
                "split", "test_AUC-ROC", "test_PR-AUC", "test_accuracy",
                "val_AUC-ROC", "val_PR-AUC", "val_accuracy", "wall_time_sec"]


def save_results(ds, cell_label, split, metrics, wall, outdir=None):
    """인코더 summary.csv와 동일 컬럼으로 append (LLM은 val/epoch 공란). F1·n_types는 별도 json."""
    import csv
    rdir = outdir or os.path.join(HERE, 'results', ds)
    os.makedirs(os.path.join(rdir, 'llm_metrics'), exist_ok=True)
    summ = os.path.join(rdir, 'summary.csv')
    row = {c: "" for c in SUMMARY_COLS}
    row.update({"cell": cell_label, "checkpoint_metric": "AUC-ROC(LLM-zs)", "dataset": ds,
                "split": split, "wall_time_sec": round(wall, 1),
                "test_AUC-ROC": round(metrics["AUC-ROC"], 6),
                "test_PR-AUC": round(metrics["PR-AUC"], 6),
                "test_accuracy": round(metrics["accuracy"], 6)})
    wh = not os.path.exists(summ) or os.path.getsize(summ) == 0
    with open(summ, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if wh:
            w.writeheader()
        w.writerow(row)
    json.dump({**metrics, "cell": cell_label, "dataset": ds, "split": split, "wall_sec": wall},
              open(os.path.join(rdir, 'llm_metrics', f'{cell_label}_{split}.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v1", "v2"])
    ap.add_argument("--dataset", default="ddibn")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--split", default="S0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="")    # summary.csv cell 라벨 (예: 15_v1zs_qwen)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)  # 공유 GPU면 낮춤
    ap.add_argument("--port", type=int, default=12341)  # 병렬 잡 충돌 방지: 잡마다 고유 포트
    ap.add_argument("--merge-system", action="store_true")  # gemma류
    ap.add_argument("--mode", default="ideal", choices=["ideal", "ideal_genes", "real"])  # V2 정보 ablation
    ap.add_argument("--workers", type=int, default=24)  # 동시 요청 수 (작은 모델은 64+로 GPU 채움)
    ap.add_argument("--outdir", default="")  # 결과 저장 폴더 override (예: results/ddibn_acc)
    a = ap.parse_args()
    global PORT
    PORT = a.port

    pb = DDI334Prompt(a.dataset)
    rows = load_rows(a.dataset, a.split, a.limit)
    print(f"[{a.version}/{a.dataset}/{a.split}] rows={len(rows)} model={a.model}")

    proc, log = start_server(a.model, a.gpu, a.merge_system, a.gpu_mem_util)
    try:
        if not wait_ready():
            print("서버 기동 실패"); return
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        t0 = time.time()
        if a.version == "v1":
            pred, lab = infer_v1(client, a.model, pb, rows, a.merge_system, workers=a.workers)
        else:
            pred, lab = infer_v2(client, a.model, pb, rows, a.merge_system, mode=a.mode, workers=a.workers)
        metrics = evaluate(pred, lab)
        wall = time.time() - t0
        print(f"[RESULT] {metrics} | wall {wall:.0f}s")
        label = a.label or f"{a.version}_{a.model.replace('/','_')}"
        save_results(a.dataset, label, a.split, metrics, wall, outdir=a.outdir or None)
        # raw 예측 저장 (분포 분석용; 인코더 raw_preds와 동일 위치/형식).
        # V2: pred=p_yes[행,타입], V1: pred=0/1[행,타입], lab=multihot+polarity[행,N+1].
        rdir = a.outdir or os.path.join(HERE, 'results', a.dataset)
        rpdir = os.path.join(rdir, 'raw_preds'); os.makedirs(rpdir, exist_ok=True)
        np.savez(os.path.join(rpdir, f'{label}_{a.split}.npz'), pred=pred, lab=lab)
        print(f"[SAVED] summary.csv + llm_metrics/{label}_{a.split}.json + raw_preds/{label}_{a.split}.npz")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        log.close()


if __name__ == '__main__':
    main()
