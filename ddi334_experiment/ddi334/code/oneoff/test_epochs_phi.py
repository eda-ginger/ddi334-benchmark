#!/usr/bin/env python3
"""Phi real/ideal의 미측정 (epoch,split) test 셀만 채우는 일회성 러너.
- 한 서버에 real ep1/2/3 + ideal ep1/2/3 어댑터를 올리고, --cells 로 지정된 셀만 test.
- 저장: v2ft_ep{e}_{tag}_{sp}.json  (ep3-고정과 동일 네이밍 규약, e=1/2)
- 기존 파일 있으면 skip (idempotent).
run 예: python code/oneoff/test_epochs_phi.py --model microsoft/Phi-3.5-mini-instruct \
  --realtag microsoft_Phi-3.5-mini-instruct_real_seed42 --idealtag microsoft_Phi-3.5-mini-instruct_ideal_seed42 \
  --gpu 0 --port 12600 --mem 0.5 --cells real:S2:2,ideal:S2:2,real:S1:1,real:S1:2,ideal:S1:2
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # code/
import numpy as np
from openai import OpenAI
import llm_infer as LI
import run_ft_cv_v2 as R
from llm_prompts import DDI334Prompt

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--realtag", required=True)
ap.add_argument("--idealtag", required=True)
ap.add_argument("--gpu", default="0")
ap.add_argument("--port", type=int, default=12600)
ap.add_argument("--mem", type=float, default=0.5)
ap.add_argument("--cells", required=True)  # "real:S2:2,ideal:S1:2,..."  (mode:split:epoch)
a = ap.parse_args()
LI.PORT = a.port

FT = os.path.join(R.OUTDIR, "..", "..", "ft_v2")  # results/ft_v2
ckpt = {"real": os.path.join("results", "ft_v2", a.realtag),
        "ideal": os.path.join("results", "ft_v2", a.idealtag)}
tag = {"real": a.realtag, "ideal": a.idealtag}

adapters = []
for mode in ("real", "ideal"):
    for e in (1, 2, 3):
        p = os.path.join(ckpt[mode], f"epoch_{e}")
        if os.path.isdir(p):
            adapters.append((f"{mode}_ep{e}", p))
print(f"[test_epochs] {len(adapters)} 어댑터 serving GPU{a.gpu}: {[n for n,_ in adapters]}", flush=True)

pb = DDI334Prompt("ddibn")
proc, log = R.start_lora_server(a.model, a.gpu, adapters, False, a.mem, a.port)
try:
    if not LI.wait_ready(timeout=1800):
        print("서버 기동 실패", flush=True); sys.exit(1)
    client = OpenAI(base_url=f"http://127.0.0.1:{a.port}/v1", api_key="x")
    rpdir = os.path.join(R.OUTDIR, "raw_preds"); os.makedirs(rpdir, exist_ok=True)
    for cell in a.cells.split(","):
        mode, sp, ep = cell.split(":")
        outj = os.path.join(R.OUTDIR, "llm_metrics", f"v2ft_ep{ep}_{tag[mode]}_{sp}.json")
        if os.path.exists(outj):
            print(f"[{cell}] skip (이미 있음)", flush=True); continue
        trows = R.load_rows("ddibn", "test", sp)
        t0 = time.time()
        pred, lab = LI.infer_v2(client, f"{mode}_ep{ep}", pb, trows, False, mode=mode)
        m = LI.evaluate(pred, lab)
        print(f"[{cell}] AUROC={m['AUC-ROC']:.4f} F1={m['macro-F1']:.4f} | {time.time()-t0:.0f}s", flush=True)
        LI.save_results("ddibn", f"v2ft_ep{ep}_{tag[mode]}", sp, m, 0, outdir=R.OUTDIR)
        np.savez(os.path.join(rpdir, f"v2ft_ep{ep}_{tag[mode]}_{sp}.npz"), pred=pred, lab=lab)
    print("[test_epochs] ALL DONE", flush=True)
finally:
    proc.terminate()
    try: proc.wait(timeout=30)
    except Exception: proc.kill()
    log.close()
