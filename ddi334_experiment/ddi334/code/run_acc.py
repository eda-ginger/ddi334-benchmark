#!/usr/bin/env python3
"""DDI-334 ddibn 인코더 42 config 재실행 (accuracy-primary).

- 신규 폴더 results/ddibn_acc/ 에 저장 (기존 results/ddibn = ROC-AUC 체크포인트 + LLM, 그대로 보존)
- code/train.py 가 체크포인트=accuracy, 전 지표(Acc/F1/ROC/PR/P@50/P@(N/2)) + raw .npz 저장
- GPU 1,3,5 3-way 병렬 (GPU4=gemma LLM+v5 보존 / GPU0=ollama 타인 / GPU2 금지). skip-done 재시작 안전.

Run (백그라운드):
  nohup micromamba run -n base python run_acc.py > results/run_logs/acc_orch.log 2>&1 &
"""
import os, subprocess, time, csv
from collections import deque

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(HERE)
GPUS = [1, 3, 5]
SEEDS = [0, 42, 124]
CELLS = [f"{i:02d}" for i in range(1, 15)]
RDIR = os.path.join(HERE, "results", "ddibn_acc")
SUMMARY = os.path.join(RDIR, "summary.csv")
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
DDIBENCH = "micromamba run -n DDIBench python"


def done_keys():
    k = set()
    if os.path.exists(SUMMARY):
        try:
            for r in csv.DictReader(open(SUMMARY)):
                k.add((str(r.get("cell")), str(r.get("seed")), str(r.get("split"))))
        except Exception:
            pass
    return k


def is_done(cell, seed):
    k = done_keys()
    return all((cell, str(seed), sp) in k for sp in ("S0", "S1", "S2"))


def cmd(cell, seed, gpu):
    return (f"CUDA_VISIBLE_DEVICES={gpu} {DDIBENCH} code/train.py "
            f"--cell {cell} --dataset ddibn --seed {seed} --gpu 0 --result_dir {RDIR}")


def main():
    jobs = deque((c, s) for c in CELLS for s in SEEDS)
    total = len(jobs)
    print(f"[acc] {total} jobs -> {RDIR} | GPU {GPUS}", flush=True)
    running = {}
    done = 0
    while jobs or running:
        free = [g for g in GPUS if g not in running]
        while free and jobs:
            c, s = jobs[0]
            if is_done(c, s):
                jobs.popleft(); done += 1
                print(f"[skip ] cell{c} s{s} ({done}/{total})", flush=True); continue
            jobs.popleft(); gpu = free.pop(0)
            name = f"cell{c}_s{s}"
            lf = open(os.path.join(LOGDIR, f"acc_{name}.log"), "w")
            p = subprocess.Popen(cmd(c, s, gpu), shell=True, stdout=lf, stderr=lf,
                                 executable="/bin/bash")
            running[gpu] = (name, p, lf)
            print(f"[start] {name} -> GPU{gpu} ({done}/{total} done)", flush=True)
        time.sleep(20)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); done += 1
                print(f"[done ] {name} GPU{gpu} rc={p.returncode} ({done}/{total})", flush=True)
                del running[gpu]
    print(f"[acc] ALL DONE ({done}/{total})", flush=True)


if __name__ == "__main__":
    main()
