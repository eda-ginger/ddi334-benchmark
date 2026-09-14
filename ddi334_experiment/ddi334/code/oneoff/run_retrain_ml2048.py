#!/usr/bin/env python3
"""완전 픽스 재학습: 학습 max_len 768->2048 (train 12% 절단 해소) + eval max_tokens 1024.
기존 보존: 어댑터는 results/ft_v1_ml2048/ (기존 ft_v1/ 불변), 결과는 v1ftN_{tag} 라벨 (기존 v1ft_ 불변).
다 검증되면 그때 표에 반영. 현재 08 표는 미팅용으로 그대로 유지.
Run: GPU=4 nohup micromamba run -n base python run_retrain_ml2048.py > results/run_logs/retrain_ml2048.log 2>&1 &
"""
import os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
GPU = os.environ.get("GPU", "4")
VLLM = "micromamba run -n vllm-llm python"
ROOT = "ft_v1_ml2048"
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
# (tag, model, eval mem-util, merge_system)
MODELS = [
    ("qwen", "Qwen/Qwen2.5-3B-Instruct",        "0.40", False),
    ("phi",  "microsoft/Phi-3.5-mini-instruct", "0.45", False),
    ("phi4", "microsoft/phi-4",                 "0.60", False),  # Phi 대형(14B), llama 대체
]
MODELS = [m for m in MODELS if not os.environ.get("RT_MODELS") or m[0] in os.environ["RT_MODELS"].split(",")]


def ckpt(model):
    return os.path.join(HERE, "results", ROOT, f"ddibn_{model.replace('/','_')}_seed42")


def sh(cmd, logname):
    lf = open(os.path.join(LOGDIR, logname), "w")
    print(f"[run] {cmd}", flush=True)
    rc = subprocess.call(cmd, shell=True, stdout=lf, stderr=lf, executable="/bin/bash")
    lf.close(); return rc


def main():
    print(f"[retrain] GPU{GPU} | {len(MODELS)} 모델 max_len2048 재학습 + 1024 재eval -> {ROOT}/ + v1ftN_", flush=True)
    for tag, model, mem, ms in MODELS:
        ck = ckpt(model)
        # 1) max_len=2048 재학습 (신규 어댑터 없을 때만)
        if not os.path.isdir(os.path.join(ck, "epoch_10")):
            print(f"[train] {tag} (max_len 2048) -> {ROOT}/", flush=True)
            rc = sh(f"{VLLM} ft_train_v1.py --model {model} --dataset ddibn --gpu {GPU} "
                    f"--seed 42 --epochs 10 --max_len 2048 --ckpt_root {ROOT}",
                    f"retrain_{tag}_train.log")
            print(f"[train] {tag} rc={rc} epoch10={'OK' if os.path.isdir(os.path.join(ck,'epoch_10')) else 'FAIL'}", flush=True)
        else:
            print(f"[skip ] {tag} 학습 (신규 epoch_10 존재)", flush=True)
        if not os.path.isdir(os.path.join(ck, "epoch_10")):
            print(f"[ERR  ] {tag} epoch_10 없음 -> eval skip", flush=True); continue
        # 2) 1024 재eval (llm_infer max_tokens=1024 적용됨) -> v1ftN_ 라벨 (기존 v1ft_ 보존)
        flag = "--merge-system" if ms else ""
        port = 12390 + int(GPU)
        print(f"[eval ] v1ftN_{tag} (max_tokens 1024)", flush=True)
        rc = sh(f"{VLLM} run_ft_cv.py --tag v1ftN_{tag} --model {model} --ckpt {ck} "
                f"--gpu {GPU} --epochs 10 --port {port} --gpu-mem-util {mem} {flag}",
                f"retrain_{tag}_eval.log")
        print(f"[eval ] {tag} rc={rc}", flush=True)
    print("[retrain] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
