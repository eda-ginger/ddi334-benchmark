#!/usr/bin/env python3
"""DDI-334 V2(binary Yes/No) FT per-split best-checkpoint eval. run_ft_cv.py의 V2판.

ft_train_v2(어댑터 epoch_1..N) -> vLLM LoRA serving:
  1) 각 epoch 어댑터를 full val_S0/S1/S2로 infer_v2(mode) -> split별 best epoch (val AUROC 기준, 인코더 일관)
  2) split별 best epoch 어댑터로 test_{split} 평가 -> 저장
결과: results/ddibn_ablation/ft/llm_metrics/v2ft_{tag}_{split}.json  (+ _cv.json 선택기록)

Usage (vllm-llm):
  python run_ft_cv_v2.py --tag microsoft_Phi-3.5-mini-instruct_ideal_seed42 \
     --model microsoft/Phi-3.5-mini-instruct --mode ideal \
     --ckpt results/ft_v2/microsoft_Phi-3.5-mini-instruct_ideal_seed42 --gpu 0 --epochs 5
"""
import argparse, json, os, sys, time
import numpy as np
from openai import OpenAI
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import llm_infer as LI
from llm_prompts import DDI334Prompt

OUTDIR = os.path.join(HERE, "results", "ddibn_ablation", "ft")
SPLITS = ("S0", "S1", "S2")


def load_rows(ds, prefix, split, limit=0):
    rows = []
    for line in open(os.path.join(HERE, 'data', ds, f"{prefix}_{split}.txt")):
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
    # 로그 파일명에 port 포함 — 같은 base model을 다른 GPU/port로 동시에 여러 개 띄울 때 파일 충돌 방지
    log = open(os.path.join(HERE, "results", "llm_cache", f"loraserver_v2_{base.replace('/','_')}_{port}.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=log), log


import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["real", "ideal", "ideal_genes"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="ddibn")
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
    print(f"[ftcv_v2/{a.tag}] {len(adapters)} epoch 어댑터 LoRA serving (GPU{a.gpu}, mode={a.mode})", flush=True)
    pb = DDI334Prompt(a.dataset)
    proc, log = start_lora_server(a.model, a.gpu, adapters, a.merge_system, a.gpu_mem_util, a.port)
    try:
        if not LI.wait_ready(timeout=1200):
            print("서버 기동 실패"); return
        client = OpenAI(base_url=f"http://127.0.0.1:{a.port}/v1", api_key="x")
        rpdir = os.path.join(OUTDIR, "raw_preds"); os.makedirs(rpdir, exist_ok=True)

        # ===== policy=ep3: val 스윕 생략, ep3 고정 어댑터로 test만 (빠름) =====
        if a.policy == "ep3":
            if 3 not in eps:
                print("ep3 어댑터 없음 -> 중단", flush=True); return
            for sp in SPLITS:
                e3json = os.path.join(OUTDIR, "llm_metrics", f"v2ft_ep3_{a.tag}_{sp}.json")
                if os.path.exists(e3json):
                    print(f"  [TEST ep3 {sp}] skip", flush=True); continue
                trows = load_rows(a.dataset, "test", sp)
                t0 = time.time()
                pred, lab = LI.infer_v2(client, "ep3", pb, trows, a.merge_system, mode=a.mode)
                m = LI.evaluate(pred, lab)
                print(f"  [TEST ep3 {sp}]: {m} | {time.time()-t0:.0f}s", flush=True)
                LI.save_results(a.dataset, f"v2ft_ep3_{a.tag}", sp, m, 0, outdir=OUTDIR)
                np.savez(os.path.join(rpdir, f"v2ft_ep3_{a.tag}_{sp}.npz"), pred=pred, lab=lab)
            open(os.path.join(OUTDIR, "llm_metrics", f"v2ft_{a.tag}_ep3DONE"), "w").close()
            print(f"[ftcv_v2/{a.tag}] ep3 DONE", flush=True)
            return

        # progress 파일: (split,epoch) val 슬롯마다 즉시 저장 -> 죽어도 resume(이미 한 슬롯 skip)
        prog_path = os.path.join(OUTDIR, "llm_metrics", f"v2ft_{a.tag}_cv.json")
        prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"tag": a.tag, "mode": a.mode, "selected": {}, "val_curve": {}}
        valcurve = {sp: {int(k): v for k, v in prog["val_curve"].get(sp, {}).items()} for sp in SPLITS}
        sel = prog.get("selected", {})

        def save_prog():
            json.dump({"tag": a.tag, "mode": a.mode, "selected": sel,
                       "val_curve": {sp: {str(e): valcurve[sp][e] for e in valcurve[sp]} for sp in SPLITS}},
                      open(prog_path, "w"), indent=1)

        for sp in SPLITS:
            vrows = load_rows(a.dataset, "valid", sp)
            print(f"  [{sp}] val rows={len(vrows)} (FULL)", flush=True)
            for e in eps:
                if e in valcurve[sp]:   # 이미 평가된 슬롯 -> skip (resume)
                    print(f"  [{sp}] ep{e} skip (이미 평가됨 AUROC={valcurve[sp][e]['AUC-ROC']})", flush=True)
                    continue
                pred, lab = LI.infer_v2(client, f"ep{e}", pb, vrows, a.merge_system, mode=a.mode)
                m = LI.evaluate(pred, lab)
                valcurve[sp][e] = {"AUC-ROC": round(m["AUC-ROC"], 6), "PR-AUC": round(m["PR-AUC"], 6),
                                   "accuracy": round(m["accuracy"], 6), "macro-F1": round(m["macro-F1"], 6)}
                print(f"  [{sp}] ep{e} val_AUROC={m['AUC-ROC']:.4f} val_acc={m['accuracy']:.4f} val_f1={m['macro-F1']:.4f}", flush=True)
                save_prog()   # 슬롯마다 즉시 저장
            best = max(valcurve[sp].items(), key=lambda kv: kv[1]["AUC-ROC"])
            sel[sp] = {"epoch": best[0], "val_AUROC": best[1]["AUC-ROC"]}
            save_prog()
            print(f"  [{sp}] -> best epoch {best[0]} (val_AUROC {best[1]['AUC-ROC']:.4f})", flush=True)
        # test: best-epoch(ep1~N val 최고) + ep3 고정 두 정책 모두 저장
        rpdir = os.path.join(OUTDIR, "raw_preds"); os.makedirs(rpdir, exist_ok=True)
        EP3 = 3
        for sp in SPLITS:
            trows = None
            # --- best-epoch 정책 ---
            be = sel[sp]["epoch"]
            bjson = os.path.join(OUTDIR, "llm_metrics", f"v2ft_{a.tag}_{sp}.json")
            mb = None
            if os.path.exists(bjson):
                mb = json.load(open(bjson)); print(f"  [TEST best {sp}] skip", flush=True)
            else:
                trows = load_rows(a.dataset, "test", sp)
                pred, lab = LI.infer_v2(client, f"ep{be}", pb, trows, a.merge_system, mode=a.mode)
                mb = LI.evaluate(pred, lab)
                print(f"  [TEST best {sp}] ep{be}: AUROC={mb['AUC-ROC']:.4f}", flush=True)
                LI.save_results(a.dataset, f"v2ft_{a.tag}", sp, mb, 0, outdir=OUTDIR)
                np.savez(os.path.join(rpdir, f"v2ft_{a.tag}_{sp}.npz"), pred=pred, lab=lab)
            # --- ep3 고정 정책 (ep3 어댑터 있을 때만) ---
            e3json = os.path.join(OUTDIR, "llm_metrics", f"v2ft_ep3_{a.tag}_{sp}.json")
            if os.path.exists(e3json):
                print(f"  [TEST ep3 {sp}] skip", flush=True)
            elif EP3 not in eps:
                print(f"  [TEST ep3 {sp}] ep3 어댑터 없음 -> skip", flush=True)
            elif be == EP3 and mb is not None:
                LI.save_results(a.dataset, f"v2ft_ep3_{a.tag}", sp, mb, 0, outdir=OUTDIR)
                print(f"  [TEST ep3 {sp}] = best(ep3) 재사용", flush=True)
            else:
                if trows is None: trows = load_rows(a.dataset, "test", sp)
                pred, lab = LI.infer_v2(client, f"ep{EP3}", pb, trows, a.merge_system, mode=a.mode)
                m3 = LI.evaluate(pred, lab)
                print(f"  [TEST ep3 {sp}]: AUROC={m3['AUC-ROC']:.4f}", flush=True)
                LI.save_results(a.dataset, f"v2ft_ep3_{a.tag}", sp, m3, 0, outdir=OUTDIR)
                np.savez(os.path.join(rpdir, f"v2ft_ep3_{a.tag}_{sp}.npz"), pred=pred, lab=lab)
        save_prog()
        open(os.path.join(OUTDIR, "llm_metrics", f"v2ft_{a.tag}_bestDONE"), "w").close()
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
