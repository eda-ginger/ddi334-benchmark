#!/usr/bin/env python3
"""V1(multi-label JSON, 한번에) vs V2(binary Yes/No, 하나씩) 답 분포 probe.
라이브 gemma 서버(port 12512) 재사용. test_S0에서 양성/음성 쌍 샘플 -> 같은 (pair,type)에
두 방식 질의 -> 예측분포 수집(실제값, 지어내지 않음). 결과 probe_dist.json.
Run: micromamba run -n vllm-llm python probe_dist.py
"""
import json, os, numpy as np
from openai import OpenAI
from llm_prompts import DDI334Prompt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(HERE)
PORT = int(os.environ.get("PORT", "12512"))
MODEL = "google/gemma-2-2b-it"
MERGE = True            # gemma = merge-system
N_POS, N_NEG = 40, 40   # 샘플 쌍 수 (polarity별)
MAX_Q = 8               # 쌍당 질의 타입 상한 (서버 부담 제한)
client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
pb = DDI334Prompt("ddibn")


def msgs_of(sys_t, usr):
    return ([{"role": "user", "content": f"{sys_t}\n\n{usr}"}] if MERGE
            else [{"role": "system", "content": sys_t}, {"role": "user", "content": usr}])


def v1_pred(d1, d2, qids):
    s, u, sc = pb.build_v1(d1, d2, qids)
    try:
        r = client.chat.completions.create(model=MODEL, messages=msgs_of(s, u), temperature=0.0,
                                            max_tokens=512, seed=123, extra_body={"guided_json": sc})
        obj = json.loads(r.choices[0].message.content)
        return {t: (1 if obj.get(str(t)) else 0) for t in qids}
    except Exception as e:
        return {t: None for t in qids}


def v2_pred(d1, d2, t):
    s, u = pb.build_v2(d1, d2, t)
    try:
        r = client.chat.completions.create(model=MODEL, messages=msgs_of(s, u), temperature=0.0,
                                           max_tokens=3, seed=123, logprobs=True, top_logprobs=20,
                                           extra_body={"guided_choice": ["Yes", "No"]})
        ch = r.choices[0]; p_yes = None
        if ch.logprobs and ch.logprobs.content:
            for tl in ch.logprobs.content[0].top_logprobs:
                if tl.token.strip().lower().startswith("yes"):
                    p_yes = float(np.exp(tl.logprob)); break
        if p_yes is None:
            p_yes = 1.0 if ch.message.content.strip().lower().startswith("yes") else 0.0
        return p_yes
    except Exception:
        return None


def main():
    rows = [l.split() for l in open("ddibn/test_S0.txt")]
    pos = [r for r in rows if r[3] == '1'][:N_POS]
    neg = [r for r in rows if r[3] == '0'][:N_NEG]
    out = {"v1_yes": [], "v2_pyes": [], "true": []}  # (pair,type) 단위, true=pol
    for grp, pol in [(pos, 1), (neg, 0)]:
        for r in grp:
            d1, d2 = int(r[0]), int(r[1])
            qids = [i for i, x in enumerate(r[2].split(',')) if x == '1'][:MAX_Q]
            if not qids:
                continue
            v1 = v1_pred(d1, d2, qids)
            for t in qids:
                py = v2_pred(d1, d2, t)
                if v1[t] is None or py is None:
                    continue
                out["v1_yes"].append(v1[t]); out["v2_pyes"].append(py); out["true"].append(pol)
        print(f"[probe] pol={pol} done, 누적 {len(out['true'])} 질의", flush=True)
    json.dump(out, open("probe_dist.json", "w"))
    n = len(out["true"]); ty = np.array(out["true"]); v1y = np.array(out["v1_yes"]); v2y = (np.array(out["v2_pyes"]) > 0.5).astype(int)
    print(f"[probe] 총 {n} 질의")
    print(f"  V1 예측-Yes율: 전체 {v1y.mean():.3f} | 진짜양성 {v1y[ty==1].mean():.3f} | 진짜음성 {v1y[ty==0].mean():.3f}")
    print(f"  V2 예측-Yes율: 전체 {v2y.mean():.3f} | 진짜양성 {v2y[ty==1].mean():.3f} | 진짜음성 {v2y[ty==0].mean():.3f}")


if __name__ == "__main__":
    main()
