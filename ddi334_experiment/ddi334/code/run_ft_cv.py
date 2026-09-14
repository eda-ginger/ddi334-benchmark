#!/usr/bin/env python3
"""DDI-334 LLM FT per-split best-checkpoint eval (인코더 방식 일관).

한 번 학습(ft_train_v1, 10 epoch 어댑터)한 결과를 vLLM LoRA serving으로 평가:
  1) 각 epoch 어댑터를 val_S0/S1/S2로 평가 -> split별 best epoch 선택 (val accuracy 기준)
  2) split별 best epoch 어댑터로 test_{split} 평가 -> 결과 저장 (results/ddibn_acc/llm_metrics/)
어댑터 직접 serving (merge 안 함). 결과는 split별로 자기 best epoch에서 나옴.

Usage (vllm-llm):
  python run_ft_cv.py --tag qwen --model Qwen/Qwen2.5-3B-Instruct \
      --ckpt results/ft_v1/ddibn_Qwen_Qwen2.5-3B-Instruct_seed42 --gpu 0 --epochs 10 \
      --gpu-mem-util 0.5 [--merge-system]
"""
import argparse, json, os, subprocess, sys, time, glob
import numpy as np
from openai import OpenAI
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import llm_infer as LI
from llm_prompts import DDI334Prompt

OUTDIR = os.path.join(HERE, "results", "ddibn_acc")
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
    # vLLM --lora-modules 는 nargs='+': 반드시 한 플래그에 전부 (반복하면 마지막만 등록됨)
    cmd += ["--lora-modules"] + [f"{name}={path}" for name, path in adapters]
    log = open(os.path.join(HERE, "results", "llm_cache", f"loraserver_{base.replace('/','_')}.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=log), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="ddibn")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--gpu-mem-util", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=12390)
    ap.add_argument("--val-limit", type=int, default=0)  # 0=full val (인코더와 동일 기준). >0은 디버그용 subset
    ap.add_argument("--merge-system", action="store_true")
    a = ap.parse_args()
    LI.PORT = a.port

    eps = [e for e in range(1, a.epochs + 1) if os.path.isdir(os.path.join(a.ckpt, f"epoch_{e}"))]
    adapters = [(f"ep{e}", os.path.join(a.ckpt, f"epoch_{e}")) for e in eps]
    print(f"[ft_cv/{a.tag}] {len(adapters)} epoch 어댑터 -> LoRA serving (GPU{a.gpu})", flush=True)
    pb = DDI334Prompt(a.dataset)
    proc, log = start_lora_server(a.model, a.gpu, adapters, a.merge_system, a.gpu_mem_util, a.port)
    try:
        if not LI.wait_ready(timeout=900):
            print("서버 기동 실패"); return
        client = OpenAI(base_url=f"http://127.0.0.1:{a.port}/v1", api_key="x")
        # 1) val로 split별 best epoch 선택
        sel = {}   # split -> {'epoch':e, 'val_acc':..}
        valcurve = {}
        for sp in SPLITS:
            vrows = load_rows(a.dataset, "valid", sp, a.val_limit)
            print(f"  [{sp}] val rows={len(vrows)} ({'FULL' if not a.val_limit else f'limit {a.val_limit}'})", flush=True)
            best = None
            valcurve[sp] = {}
            for e in eps:
                pred, lab = LI.infer_v1(client, f"ep{e}", pb, vrows, a.merge_system)
                m = LI.evaluate(pred, lab)
                acc = m["accuracy"]
                # 선택은 accuracy 기준. 단 hard-label로 계산 가능한 지표(acc+F1)는 매 epoch 저장.
                # 확률 지표(AUC-ROC/PR-AUC)는 LLM hard label에선 무의미 -> 제외.
                valcurve[sp][e] = {"accuracy": round(m["accuracy"], 6), "macro-F1": round(m["macro-F1"], 6)}
                print(f"  [{sp}] ep{e} val_acc={acc:.4f} val_f1={m['macro-F1']:.4f}", flush=True)
                if best is None or acc > best[1]:
                    best = (e, acc)
            sel[sp] = {"epoch": best[0], "val_acc": round(best[1], 6),
                       "val_f1": valcurve[sp][best[0]]["macro-F1"]}
            print(f"  [{sp}] -> best epoch {best[0]} (val_acc {best[1]:.4f}, val_f1 {sel[sp]['val_f1']:.4f})", flush=True)
        # 2) split별 best epoch으로 test 평가
        for sp in SPLITS:
            e = sel[sp]["epoch"]
            trows = load_rows(a.dataset, "test", sp)
            t0 = time.time()
            pred, lab = LI.infer_v1(client, f"ep{e}", pb, trows, a.merge_system)
            m = LI.evaluate(pred, lab)
            wall = time.time() - t0
            print(f"  [TEST {sp}] ep{e}: {m} | {wall:.0f}s", flush=True)
            LI.save_results(a.dataset, f"v1ft_{a.tag}", sp, m, wall, outdir=OUTDIR)
        # 선택 기록 저장
        json.dump({"tag": a.tag, "selected": sel, "val_curve": valcurve},
                  open(os.path.join(OUTDIR, "llm_metrics", f"v1ft_{a.tag}_cv.json"), "w"), indent=1)
        print(f"[ft_cv/{a.tag}] DONE. split별 best epoch: "
              + ", ".join(f"{sp}:ep{sel[sp]['epoch']}" for sp in SPLITS), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
