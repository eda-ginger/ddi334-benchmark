#!/usr/bin/env python3
"""DDI-334 ddibn 인코더 42 config 재실행 (--apk): 재현 검증 + P@50/P@(N/2) + raw 점수.

- 같은 seed/HP로 재학습 -> summary_apk.csv (기존 summary.csv는 보존, 사후 reproducibility 비교용)
- raw 점수는 results/ddibn/raw_preds/*.npz 에 저장 (향후 지표 추가 시 재학습 불필요)
- GPU 1, 5 두 스트림 병렬 (LLM phi/gemma 가 점유한 GPU3/4, v5 GPU4 와 충돌 회피). skip-done 재시작 안전.

Run (백그라운드):
  nohup micromamba run -n base python rerun_apk.py > results/run_logs/apk_orch.log 2>&1 &
"""
import os, subprocess, time, csv
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
GPUS = [1, 5]
SEEDS = [0, 42, 124]
CELLS = [f"{i:02d}" for i in range(1, 15)]
SUMMARY = os.path.join(HERE, "results", "ddibn", "summary_apk.csv")
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
    return (f"CUDA_VISIBLE_DEVICES={gpu} {DDIBENCH} train.py "
            f"--cell {cell} --dataset ddibn --seed {seed} --gpu 0 --apk")


def main():
    jobs = deque((c, s) for c in CELLS for s in SEEDS)
    total = len(jobs)
    print(f"[apk] {total} jobs -> GPU {GPUS}", flush=True)
    running = {}   # gpu -> (name, popen, logf)
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
            lf = open(os.path.join(LOGDIR, f"apk_{name}.log"), "w")
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
    print(f"[apk] ALL DONE ({done}/{total})", flush=True)


if __name__ == "__main__":
    main()
