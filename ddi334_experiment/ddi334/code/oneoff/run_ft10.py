#!/usr/bin/env python3
"""DDI-334 LLM FT 10-epoch + per-split best-checkpoint (6 모델). GPU0 순차.

각 모델: ft_train_v1(10 epoch 어댑터 저장) -> run_ft_cv(vLLM LoRA serving, split별 best epoch 선택 -> test).
결과 -> results/ddibn_acc/llm_metrics/v1ft_{tag}_{split}.json (기존 3e는 ft3e_archive 보관됨).
skip-done: epoch_10 어댑터 있으면 train skip, v1ft_{tag}_cv.json 있으면 eval skip.

Run: GPU=0 nohup micromamba run -n base python run_ft10.py > results/run_logs/ft10_orch.log 2>&1 &
"""
import os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
GPU = os.environ.get("GPU", "0")
VLLM = "micromamba run -n vllm-llm python"
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
ACC = os.path.join(HERE, "results", "ddibn_acc")

# (tag, model, eval mem-util(LoRA serving=base full model), merge_system)
MODELS = [
    ("qwen",     "Qwen/Qwen2.5-3B-Instruct",               "0.40", False),
    ("phi",      "microsoft/Phi-3.5-mini-instruct",        "0.45", False),
    ("gemma",    "google/gemma-2-2b-it",                   "0.40", True),
    ("llama8b",  "meta-llama/Meta-Llama-3.1-8B-Instruct",  "0.50", False),
    ("qwen14b",  "Qwen/Qwen2.5-14B-Instruct",              "0.65", False),
    ("gemma27b", "google/gemma-2-27b-it",                  "0.85", True),
]


_sel = [s.strip() for s in os.environ.get("FT_MODELS", "").split(",") if s.strip()]
if _sel:
    MODELS = [m for m in MODELS if m[0] in _sel]


def ckpt(model):
    return os.path.join(HERE, "results", "ft_v1", f"ddibn_{model.replace('/','_')}_seed42")


def sh(cmd, logname):
    lf = open(os.path.join(LOGDIR, logname), "w")
    rc = subprocess.call(cmd, shell=True, stdout=lf, stderr=lf, executable="/bin/bash")
    lf.close(); return rc


def main():
    print(f"[ft10] GPU{GPU} | {len(MODELS)} 모델 10e + per-split eval", flush=True)
    for tag, model, mem, ms in MODELS:
        ck = ckpt(model)
        flag = "--merge-system" if ms else ""
        # 1) 10e 학습 (epoch_10 어댑터 없으면)
        if not os.path.isdir(os.path.join(ck, "epoch_10")):
            print(f"[train] {tag} 10epoch", flush=True)
            rc = sh(f"{VLLM} ft_train_v1.py --model {model} --dataset ddibn --gpu {GPU} --seed 42 --epochs 10",
                    f"ft10_{tag}_train.log")
            print(f"[train] {tag} rc={rc} epoch10={'OK' if os.path.isdir(os.path.join(ck,'epoch_10')) else 'FAIL'}", flush=True)
        else:
            print(f"[skip ] {tag} 학습 (epoch_10 존재)", flush=True)
        if not os.path.isdir(os.path.join(ck, "epoch_10")):
            print(f"[ERR  ] {tag} epoch_10 없음 -> eval skip", flush=True); continue
        # 2) per-split best eval (cv json 없으면)
        if os.path.exists(os.path.join(ACC, "llm_metrics", f"v1ft_{tag}_cv.json")):
            print(f"[skip ] {tag} eval (cv json 존재)", flush=True); continue
        print(f"[eval ] {tag} per-split best (LoRA serving)", flush=True)
        rc = sh(f"{VLLM} run_ft_cv.py --tag {tag} --model {model} --ckpt {ck} "
                f"--gpu {GPU} --epochs 10 --port {12390+int(GPU)} --gpu-mem-util {mem} {flag}",
                f"ft10_{tag}_eval.log")
        print(f"[eval ] {tag} rc={rc}", flush=True)
    print("[ft10] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
