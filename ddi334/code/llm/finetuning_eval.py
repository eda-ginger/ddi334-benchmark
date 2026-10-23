#!/usr/bin/env python3
"""DDI-334 V2(binary Yes/No) FT per-split best-checkpoint eval.

finetuning_train.py (adapters epoch_1..N) -> vLLM LoRA serving:
  1) runs infer_v2(mode) for each epoch's adapter over the full val_S0/S1/S2 -> best epoch per split (by val AUROC, consistent with the encoders)
  2) evaluates test_{split} with each split's best-epoch adapter -> saves it
Output: results/main_experiments/13-18_llm/finetuning/llm_metrics/ft_{tag}_{split}.json  (+ _cv.json selection record)

Usage (vllm-llm):
  python finetuning_eval.py --tag microsoft_Phi-3.5-mini-instruct_existing_drug_seed42 \
     --model microsoft/Phi-3.5-mini-instruct --mode existing_drug \
     --ckpt results/ft_checkpoints/microsoft_Phi-3.5-mini-instruct_existing_drug_seed42 --gpu 0 --epochs 3
"""
import argparse, json, os, sys, time
import numpy as np
from openai import OpenAI
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, 'code', 'build', 'document'))
import zeroshot as LI
from instruction import DDI334Prompt

SPLITS_DIR = os.path.join(HERE, "data", "splits")
DATASET_TAG = "ddibn"   # constant kept only for result filename / summary.csv schema compatibility (not used to build paths)
OUTDIR = os.path.join(HERE, "results", "main_experiments", "13-18_llm", "finetuning")
SPLITS = ("S0", "S1", "S2")


def load_rows(prefix, split, limit=0):
    rows = []
    for line in open(os.path.join(SPLITS_DIR, f"{prefix}_{split}.txt")):
        p = line.split()
        if len(p) != 4:
            continue
        vec = np.array([int(x) for x in p[2].split(',')], dtype=np.int64)
        rows.append((int(p[0]), int(p[1]), vec, int(p[3])))
        if limit and len(rows) >= limit:
            break
    return rows


def start_lora_server(base, gpu, adapters, merge, mem, port):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["VLLM_USE_V1"] = "0"
    if merge:
        env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"; env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", base,
           "--host", "127.0.0.1", "--port", str(port), "--gpu-memory-utilization", str(mem),
           "--enforce-eager", "--max-model-len", "4096" if merge else "8192",
           "--disable-log-requests", "--enable-lora", "--max-lora-rank", "16",
           "--max-loras", "1", "--max-cpu-loras", str(max(2, len(adapters)))]
    cmd += ["--lora-modules"] + [f"{name}={path}" for name, path in adapters]
    # Include the port in the log filename -- avoids file collisions when multiple instances of the same base model are launched concurrently on different GPUs/ports
    log = open(os.path.join(HERE, "results", "llm_cache", f"loraserver_v2_{base.replace('/','_')}_{port}.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=log), log


import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["novel_drug", "existing_drug"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--gpu-mem-util", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=12390)
    ap.add_argument("--merge-system", action="store_true")
    ap.add_argument("--policy", default="both", choices=["ep3", "best", "both"])
    a = ap.parse_args()
    LI.PORT = a.port
    os.makedirs(os.path.join(OUTDIR, "llm_metrics"), exist_ok=True)

    eps = [e for e in range(1, a.epochs + 1) if os.path.isdir(os.path.join(a.ckpt, f"epoch_{e}"))]
    adapters = [(f"ep{e}", os.path.join(a.ckpt, f"epoch_{e}")) for e in eps]
    print(f"[ftcv_v2/{a.tag}] LoRA-serving {len(adapters)} epoch adapters (GPU{a.gpu}, mode={a.mode})", flush=True)
    pb = DDI334Prompt()
    proc, log = start_lora_server(a.model, a.gpu, adapters, a.merge_system, a.gpu_mem_util, a.port)
    try:
        if not LI.wait_ready(timeout=1200):
            print("server failed to start"); return
        client = OpenAI(base_url=f"http://127.0.0.1:{a.port}/v1", api_key="x")
        rpdir = os.path.join(OUTDIR, "raw_preds"); os.makedirs(rpdir, exist_ok=True)

        # ===== policy=ep3: skip the val sweep, test only with the fixed ep3 adapter (fast) =====
        if a.policy == "ep3":
            if 3 not in eps:
                print("no ep3 adapter -> aborting", flush=True); return
            for sp in SPLITS:
                e3json = os.path.join(OUTDIR, "llm_metrics", f"ft_ep3_{a.tag}_{sp}.json")
                if os.path.exists(e3json):
                    print(f"  [TEST ep3 {sp}] skip", flush=True); continue
                trows = load_rows("test", sp)
                t0 = time.time()
                pred, lab = LI.infer_v2(client, "ep3", pb, trows, a.merge_system, mode=a.mode)
                m = LI.evaluate(pred, lab)
                print(f"  [TEST ep3 {sp}]: {m} | {time.time()-t0:.0f}s", flush=True)
                LI.save_results(DATASET_TAG, f"ft_ep3_{a.tag}", sp, m, 0, outdir=OUTDIR)
                np.savez(os.path.join(rpdir, f"ft_ep3_{a.tag}_{sp}.npz"), pred=pred, lab=lab)
            open(os.path.join(OUTDIR, "llm_metrics", f"ft_{a.tag}_ep3DONE"), "w").close()
            print(f"[ftcv_v2/{a.tag}] ep3 DONE", flush=True)
            return

        # progress file: saved immediately after each (split,epoch) val slot -> resumable even if it dies (slots already done are skipped)
        prog_path = os.path.join(OUTDIR, "llm_metrics", f"ft_{a.tag}_cv.json")
        prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"tag": a.tag, "mode": a.mode, "selected": {}, "val_curve": {}}
        valcurve = {sp: {int(k): v for k, v in prog["val_curve"].get(sp, {}).items()} for sp in SPLITS}
        sel = prog.get("selected", {})

        def save_prog():
            json.dump({"tag": a.tag, "mode": a.mode, "selected": sel,
                       "val_curve": {sp: {str(e): valcurve[sp][e] for e in valcurve[sp]} for sp in SPLITS}},
                      open(prog_path, "w"), indent=1)

        for sp in SPLITS:
            vrows = load_rows("valid", sp)
            print(f"  [{sp}] val rows={len(vrows)} (FULL)", flush=True)
            for e in eps:
                if e in valcurve[sp]:   # slot already evaluated -> skip (resume)
                    print(f"  [{sp}] ep{e} skip (already evaluated, AUROC={valcurve[sp][e]['AUC-ROC']})", flush=True)
                    continue
                pred, lab = LI.infer_v2(client, f"ep{e}", pb, vrows, a.merge_system, mode=a.mode)
                m = LI.evaluate(pred, lab)
                valcurve[sp][e] = {"AUC-ROC": round(m["AUC-ROC"], 6), "PR-AUC": round(m["PR-AUC"], 6),
                                   "accuracy": round(m["accuracy"], 6), "macro-F1": round(m["macro-F1"], 6)}
                print(f"  [{sp}] ep{e} val_AUROC={m['AUC-ROC']:.4f} val_acc={m['accuracy']:.4f} val_f1={m['macro-F1']:.4f}", flush=True)
                save_prog()   # save immediately after every slot
            best = max(valcurve[sp].items(), key=lambda kv: kv[1]["AUC-ROC"])
            sel[sp] = {"epoch": best[0], "val_AUROC": best[1]["AUC-ROC"]}
            save_prog()
            print(f"  [{sp}] -> best epoch {best[0]} (val_AUROC {best[1]['AUC-ROC']:.4f})", flush=True)
        # test: save results for both policies -- best-epoch (highest val among ep1~N) and fixed ep3
        rpdir = os.path.join(OUTDIR, "raw_preds"); os.makedirs(rpdir, exist_ok=True)
        EP3 = 3
        for sp in SPLITS:
            trows = None
            # --- best-epoch policy ---
            be = sel[sp]["epoch"]
            bjson = os.path.join(OUTDIR, "llm_metrics", f"ft_{a.tag}_{sp}.json")
            mb = None
            if os.path.exists(bjson):
                mb = json.load(open(bjson)); print(f"  [TEST best {sp}] skip", flush=True)
            else:
                trows = load_rows("test", sp)
                pred, lab = LI.infer_v2(client, f"ep{be}", pb, trows, a.merge_system, mode=a.mode)
                mb = LI.evaluate(pred, lab)
                print(f"  [TEST best {sp}] ep{be}: AUROC={mb['AUC-ROC']:.4f}", flush=True)
                LI.save_results(DATASET_TAG, f"ft_{a.tag}", sp, mb, 0, outdir=OUTDIR)
                np.savez(os.path.join(rpdir, f"ft_{a.tag}_{sp}.npz"), pred=pred, lab=lab)
            # --- fixed-ep3 policy (only if an ep3 adapter exists) ---
            e3json = os.path.join(OUTDIR, "llm_metrics", f"ft_ep3_{a.tag}_{sp}.json")
            if os.path.exists(e3json):
                print(f"  [TEST ep3 {sp}] skip", flush=True)
            elif EP3 not in eps:
                print(f"  [TEST ep3 {sp}] no ep3 adapter -> skip", flush=True)
            elif be == EP3 and mb is not None:
                LI.save_results(DATASET_TAG, f"ft_ep3_{a.tag}", sp, mb, 0, outdir=OUTDIR)
                print(f"  [TEST ep3 {sp}] = reusing best(ep3)", flush=True)
            else:
                if trows is None: trows = load_rows("test", sp)
                pred, lab = LI.infer_v2(client, f"ep{EP3}", pb, trows, a.merge_system, mode=a.mode)
                m3 = LI.evaluate(pred, lab)
                print(f"  [TEST ep3 {sp}]: AUROC={m3['AUC-ROC']:.4f}", flush=True)
                LI.save_results(DATASET_TAG, f"ft_ep3_{a.tag}", sp, m3, 0, outdir=OUTDIR)
                np.savez(os.path.join(rpdir, f"ft_ep3_{a.tag}_{sp}.npz"), pred=pred, lab=lab)
        save_prog()
        open(os.path.join(OUTDIR, "llm_metrics", f"ft_{a.tag}_bestDONE"), "w").close()
        print(f"[ftcv_v2/{a.tag}] DONE. best epoch: " + ", ".join(f"{sp}:ep{sel[sp]['epoch']}" for sp in SPLITS), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
