#!/usr/bin/env python3
"""DDI-334 V2 (Binary-with-R) Zero-shot — 작은 모델 3종.

V1(multi-label JSON) ZS가 chance(~50)라, V2(쌍+타입 r마다 Yes/No 이진질문)로 재시도.
- vec=1(그 쌍의 pos/neg 후보)만 질문 (209개 전부 X) -> infer_v2.
- 작은 모델만(빠른 추론): gemma2-2B / qwen2.5-3B / phi-3.5-mini.
- 결과 results/ddibn_acc/llm_metrics/v2zs_{tag}_{split}.json (V1 ZS와 분리, 같은 폴더).
- 빈 GPU [0,1] 큐 분산. skip-done(json 존재). split 순서 S0->S2->S1(싼것 먼저).

Run: nohup micromamba run -n base python run_v2zs.py > results/run_logs/v2zs_orch.log 2>&1 &
"""
import os, subprocess, time
from collections import deque

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(HERE)
OUTDIR = os.path.join(HERE, "results", "ddibn_acc")
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
VLLM = "micromamba run -n vllm-llm python"
GPUS = [int(g) for g in os.environ.get("GPUS", "0,1").split(",") if g.strip()]
# (label, model, mem-util, merge_system)
MODELS = [   # 소형 -> 대형 (3 패밀리 × 2 크기)
    ("v2zs_gemma",    "google/gemma-2-2b-it",            "0.40", True),
    ("v2zs_qwen",     "Qwen/Qwen2.5-3B-Instruct",        "0.40", False),
    ("v2zs_phi",      "microsoft/Phi-3.5-mini-instruct", "0.45", False),
    ("v2zs_phi4",     "microsoft/phi-4",                 "0.60", False),
    ("v2zs_qwen14b",  "Qwen/Qwen2.5-14B-Instruct",       "0.65", False),
    ("v2zs_gemma27b", "google/gemma-2-27b-it",           "0.85", True),
]
SPLITS = ("S0", "S2", "S1")   # 싼것(S2 1만)→S0(3만)→S1(8만) 순


def done(label, sp):
    return os.path.exists(os.path.join(OUTDIR, "llm_metrics", f"{label}_{sp}.json"))


def cmd(label, model, mem, ms, sp, gpu):
    flag = "--merge-system" if ms else ""
    port = 12500 + gpu * 10 + SPLITS.index(sp)
    # llm_infer.py가 --gpu 값으로 CUDA_VISIBLE_DEVICES를 내부 설정하므로 실제 gpu 인덱스를 --gpu로 넘김
    return (f"{VLLM} code/llm_infer.py --version v2 --dataset ddibn "
            f"--model {model} --split {sp} --gpu {gpu} --label {label} --gpu-mem-util {mem} "
            f"--port {port} --outdir {OUTDIR} {flag}")


def main():
    jobs = deque((lb, md, mm, ms, sp) for lb, md, mm, ms in MODELS for sp in SPLITS)
    print(f"[v2zs] {len(jobs)} jobs (3 model x 3 split) -> GPU {GPUS}", flush=True)
    running = {}
    while jobs or running:
        free = [g for g in GPUS if g not in running]
        while free and jobs:
            lb, md, mm, ms, sp = jobs[0]
            if done(lb, sp):
                jobs.popleft(); print(f"[skip ] {lb} {sp}", flush=True); continue
            jobs.popleft(); gpu = free.pop(0)
            lf = open(os.path.join(LOGDIR, f"v2zs_{lb}_{sp}.log"), "w")
            p = subprocess.Popen(cmd(lb, md, mm, ms, sp, gpu), shell=True,
                                 stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (f"{lb}_{sp}", p, lf); print(f"[start] {lb} {sp} -> GPU{gpu}", flush=True)
            time.sleep(15)   # micromamba lock 경합 방지: 동시 기동 사이 시차
        time.sleep(30)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); print(f"[done ] {name} rc={p.returncode}", flush=True)
                del running[gpu]
    print("[v2zs] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
