#!/usr/bin/env python3
"""DDI-334 ddibn 큰 모델 Zero-shot 추가 실험 (LLMDDI 인벤토리 대형 3종).

- Llama-3.1-8B / Qwen2.5-14B / Gemma2-27B, V1 multi-label JSON ZS (FT 아님)
- 결과는 canonical 폴더 results/ddibn_acc/llm_metrics/ 에 저장 (--outdir)
- GPU 1개에서 순차 실행 (인코더가 ~4GB만 쓰므로 공유 가능). skip-done 재시작 안전.

Run (백그라운드):
  GPU=3 nohup micromamba run -n base python run_zs_big.py > results/run_logs/zsbig_orch.log 2>&1 &
"""
import os, subprocess, time, json

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
GPU = os.environ.get("GPU", "3")
OUTDIR = os.path.join(HERE, "results", "ddibn_acc")
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
VLLM = "micromamba run -n vllm-llm python"

# (label, model, mem-util, merge_system)
MODELS = [
    ("v1zs_llama8b",  "meta-llama/Meta-Llama-3.1-8B-Instruct", "0.30", False),
    ("v1zs_qwen14b",  "Qwen/Qwen2.5-14B-Instruct",             "0.50", False),
    ("v1zs_gemma27b", "google/gemma-2-27b-it",                 "0.72", True),
]
SPLITS = ("S0", "S1", "S2")
PORT = 12360 + int(GPU)


def done(label, sp):
    f = os.path.join(OUTDIR, "llm_metrics", f"{label}_{sp}.json")
    return os.path.exists(f)


def main():
    print(f"[zsbig] GPU{GPU} port{PORT} -> {OUTDIR}", flush=True)
    for label, model, mem, ms in MODELS:
        for sp in SPLITS:
            if done(label, sp):
                print(f"[skip ] {label} {sp}", flush=True); continue
            flag = "--merge-system" if ms else ""
            cmd = (f"{VLLM} llm_infer.py --version v1 --dataset ddibn --model {model} "
                   f"--gpu {GPU} --port {PORT} --split {sp} --label {label} "
                   f"--gpu-mem-util {mem} --outdir {OUTDIR} {flag}")
            lf = open(os.path.join(LOGDIR, f"zsbig_{label}_{sp}.log"), "w")
            print(f"[run  ] {label} {sp} (mem {mem})", flush=True)
            rc = subprocess.call(cmd, shell=True, stdout=lf, stderr=lf, executable="/bin/bash")
            lf.close()
            print(f"[done ] {label} {sp} rc={rc}", flush=True)
    print("[zsbig] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
