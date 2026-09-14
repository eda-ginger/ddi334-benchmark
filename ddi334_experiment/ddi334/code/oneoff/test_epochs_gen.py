#!/usr/bin/env python3
"""일반화 epoch-fill 러너: 임의 (mode,tag)의 미측정 (epoch,split) test cell만 채움.
test_epochs_phi.py의 일반화판 — real/ideal뿐 아니라 ideal_genes(genes)도 지원.
- --tags "mode=tag,..." 로 각 mode별 어댑터 tag 지정 (예: real=..._real_seed42,ideal_genes=..._ideal_genes_seed42)
- --cells "mode:split:epoch,..." 로 채울 cell 지정 (mode는 --tags의 key와 일치, 프롬프트도 그 mode로)
- 저장: v2ft_ep{e}_{tag}_{sp}.json (ep3-고정과 동일 네이밍). 기존 있으면 skip.
run 예: python code/oneoff/test_epochs_gen.py --model google/gemma-2-2b-it --merge \
  --tags real=google_gemma-2-2b-it_real_seed42,ideal_genes=google_gemma-2-2b-it_ideal_genes_seed42 \
  --gpu 3 --port 12720 --mem 0.4 --cells real:S2:2,ideal_genes:S2:1
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
ap.add_argument("--tags", required=True)   # "real=<tag>,ideal_genes=<tag>"
ap.add_argument("--gpu", default="0")
ap.add_argument("--port", type=int, default=12720)
ap.add_argument("--mem", type=float, default=0.4)
ap.add_argument("--merge", action="store_true")
ap.add_argument("--cells", required=True)  # "real:S2:2,ideal_genes:S2:1"
a = ap.parse_args()
LI.PORT = a.port

tagmap = dict(kv.split("=") for kv in a.tags.split(","))
adapters = []
for mode, tag in tagmap.items():
    for e in (1, 2, 3):
        p = os.path.join("results", "ft_v2", tag, f"epoch_{e}")
        if os.path.isdir(p):
            adapters.append((f"{mode}_ep{e}", p))
print(f"[test_epochs_gen] merge={a.merge} {len(adapters)} 어댑터 serving GPU{a.gpu}: {[n for n,_ in adapters]}", flush=True)

pb = DDI334Prompt("ddibn")
proc, log = R.start_lora_server(a.model, a.gpu, adapters, a.merge, a.mem, a.port)
try:
    if not LI.wait_ready(timeout=1800):
        print("서버 기동 실패", flush=True); sys.exit(1)
    client = OpenAI(base_url=f"http://127.0.0.1:{a.port}/v1", api_key="x")
    rpdir = os.path.join(R.OUTDIR, "raw_preds"); os.makedirs(rpdir, exist_ok=True)
    for cell in a.cells.split(","):
        mode, sp, ep = cell.split(":")
        tag = tagmap[mode]
        outj = os.path.join(R.OUTDIR, "llm_metrics", f"v2ft_ep{ep}_{tag}_{sp}.json")
        if os.path.exists(outj):
            print(f"[{cell}] skip (이미 있음)", flush=True); continue
        trows = R.load_rows("ddibn", "test", sp)
        t0 = time.time()
        pred, lab = LI.infer_v2(client, f"{mode}_ep{ep}", pb, trows, a.merge, mode=mode)
        m = LI.evaluate(pred, lab)
        print(f"[{cell}] AUROC={m['AUC-ROC']:.4f} F1={m['macro-F1']:.4f} | {time.time()-t0:.0f}s", flush=True)
        LI.save_results("ddibn", f"v2ft_ep{ep}_{tag}", sp, m, 0, outdir=R.OUTDIR)
        np.savez(os.path.join(rpdir, f"v2ft_ep{ep}_{tag}_{sp}.npz"), pred=pred, lab=lab)
    print("[test_epochs_gen] ALL DONE", flush=True)
finally:
    proc.terminate()
    try: proc.wait(timeout=30)
    except Exception: proc.kill()
    log.close()
