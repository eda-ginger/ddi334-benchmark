#!/usr/bin/env python3
"""DDI-334 큰 모델 QLoRA FT 추가 (Llama-8B/Qwen-14B/Gemma-27B). ZS/FT × 크기 그리드 완성용.

- GPU0 순차 (ollama idle, KGE 체인은 1/3/5라 분리). 각 모델: ft_train_v1(QLoRA) -> merge -> llm_infer eval 3split.
- 결과 -> results/ddibn_acc/llm_metrics/ (label v1ft_{tag}), 기존 미변경.
- skip-done: merged 존재 시 학습 skip, json 존재 시 eval skip.

Run: GPU=0 nohup micromamba run -n base python run_ft_big.py > results/run_logs/ftbig_orch.log 2>&1 &
"""
import os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
GPU = os.environ.get("GPU", "0")
OUTDIR = os.path.join(HERE, "results", "ddibn_acc")
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
VLLM = "micromamba run -n vllm-llm python"
PORT = 12370 + int(GPU)

# (tag, model, eval mem-util, merge_system)
MODELS = [
    ("llama8b",  "meta-llama/Meta-Llama-3.1-8B-Instruct", "0.35", False),
    ("qwen14b",  "Qwen/Qwen2.5-14B-Instruct",             "0.55", False),
    ("gemma27b", "google/gemma-2-27b-it",                 "0.75", True),
]
SPLITS = ("S0", "S1", "S2")


def merged_path(model):
    return os.path.join(HERE, "results", "ft_v1", f"ddibn_{model.replace('/','_')}_seed42_merged")


def eval_done(tag, sp):
    return os.path.exists(os.path.join(OUTDIR, "llm_metrics", f"v1ft_{tag}_{sp}.json"))


def sh(cmd, logname):
    lf = open(os.path.join(LOGDIR, logname), "w")
    rc = subprocess.call(cmd, shell=True, stdout=lf, stderr=lf, executable="/bin/bash")
    lf.close()
    return rc


def main():
    print(f"[ftbig] GPU{GPU} port{PORT} -> {OUTDIR}", flush=True)
    for tag, model, mem, ms in MODELS:
        merged = merged_path(model)
        flag = "--merge-system" if ms else ""
        # 1) QLoRA FT (merged 없으면)
        if not os.path.isdir(merged):
            print(f"[train] {tag} QLoRA FT", flush=True)
            rc = sh(f"{VLLM} ft_train_v1.py --model {model} --dataset ddibn --gpu {GPU} --seed 42",
                    f"ftbig_{tag}_train.log")
            print(f"[train] {tag} rc={rc} merged={'OK' if os.path.isdir(merged) else 'FAIL'}", flush=True)
        else:
            print(f"[skip ] {tag} merged 존재", flush=True)
        if not os.path.isdir(merged):
            print(f"[ERR  ] {tag} merged 없음 -> eval skip", flush=True); continue
        # 2) eval 3 split
        for sp in SPLITS:
            if eval_done(tag, sp):
                print(f"[skip ] {tag} {sp} eval", flush=True); continue
            print(f"[eval ] {tag} {sp}", flush=True)
            rc = sh(f"{VLLM} llm_infer.py --version v1 --dataset ddibn --model {merged} "
                    f"--gpu {GPU} --port {PORT} --split {sp} --label v1ft_{tag} "
                    f"--gpu-mem-util {mem} --outdir {OUTDIR} {flag}", f"ftbig_{tag}_{sp}.log")
            print(f"[eval ] {tag} {sp} rc={rc}", flush=True)
    print("[ftbig] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
