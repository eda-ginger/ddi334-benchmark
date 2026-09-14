#!/usr/bin/env python3
"""V1-ZS 실제 출력 분포 진단 — 모델이 true/false를 어떻게 찍는지 확인.
Usage (vllm-llm): python diag_v1.py --model google/gemma-2-2b-it --gpu 5 --merge-system --limit 16
"""
import argparse, json, os, sys, time
import numpy as np
from openai import OpenAI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_infer as LI
from llm_prompts import DDI334Prompt

a = argparse.ArgumentParser()
a.add_argument("--model", required=True); a.add_argument("--gpu", default="5")
a.add_argument("--dataset", default="ddibn"); a.add_argument("--split", default="S0")
a.add_argument("--limit", type=int, default=16); a.add_argument("--merge-system", action="store_true")
A = a.parse_args()

pb = DDI334Prompt(A.dataset)
rows = LI.load_rows(A.dataset, A.split, A.limit)
proc, log = LI.start_server(A.model, A.gpu, A.merge_system, 0.4)
try:
    if not LI.wait_ready():
        print("서버 실패"); sys.exit(1)
    cl = OpenAI(base_url=f"http://127.0.0.1:{LI.PORT}/v1", api_key="x")
    n_true = n_false = n_pred = 0
    pos_true = pos_n = neg_true = neg_n = 0  # 양성쌍 vs 음성쌍에서의 true율
    for r, (d1, d2, vec, pol) in enumerate(rows):
        qids = np.where(vec == 1)[0].tolist()
        st, us, sc = pb.build_v1(d1, d2, qids)
        msgs = ([{"role": "user", "content": f"{st}\n\n{us}"}] if A.merge_system
                else [{"role": "system", "content": st}, {"role": "user", "content": us}])
        resp = cl.chat.completions.create(model=A.model, messages=msgs, temperature=0.0,
                                          max_tokens=512, seed=123, extra_body={"guided_json": sc})
        obj = json.loads(resp.choices[0].message.content)
        t = sum(1 for v in obj.values() if v); f = len(obj) - t
        n_true += t; n_false += f; n_pred += len(obj)
        if pol == 1: pos_true += t; pos_n += len(obj)
        else: neg_true += t; neg_n += len(obj)
        if r < 3:
            print(f"[row{r} pol={pol} 후보{len(qids)}] true={t} false={f} | 예시출력: {dict(list(obj.items())[:6])}")
    print(f"\n=== 분포 (총 {n_pred} 예측) ===")
    print(f"true={n_true} ({100*n_true/n_pred:.1f}%) | false={n_false} ({100*n_false/n_pred:.1f}%)")
    print(f"양성쌍에서 true율: {100*pos_true/max(pos_n,1):.1f}% ({pos_true}/{pos_n})")
    print(f"음성쌍에서 true율: {100*neg_true/max(neg_n,1):.1f}% ({neg_true}/{neg_n})")
    print("(양성=음성=낮으면 '거의 다 false'. AUC 50 원인 확인)")
finally:
    proc.terminate()
    try: proc.wait(timeout=15)
    except Exception: proc.kill()
    log.close()
