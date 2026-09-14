#!/usr/bin/env python3
"""DDI-334 ddibn 나머지 4 KGE 학습 (rotate/distmult/complex/transh) — cell08 노드초기값 비교용.

- transe는 이미 학습됨(제외). tdc도 제외(ddibn만).
- 각 KGE: baseline.py -c {kge}.yaml -> extract_embeddings.py --result_dir {kge}/
- GPU 1/3/5 분산(인코더 종료로 가용). skip-done = {kge}/ 에 *.npy 존재.

Run: nohup micromamba run -n base python run_kge4.py > results/run_logs/kge4_orch.log 2>&1 &
"""
import os, subprocess, time, glob
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
BASE = "/home/rudwls2717/Latex/experiments/v5_ddi-bench/kge_hetionet"
KGEDIR = os.path.join(HERE, "kge", "ddibn")
GPUS = [1, 3, 5]
KGES = ["rotate", "distmult", "complex", "transh"]
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
RUN = "micromamba run -n DDIBench python3"


def done(kge):
    return bool(glob.glob(os.path.join(KGEDIR, kge, "*.npy")))


def cmd(kge, gpu):
    cfg = os.path.join(KGEDIR, "configs", f"{kge}.yaml")
    out = os.path.join(KGEDIR, kge)
    return (f"CUDA_VISIBLE_DEVICES={gpu} {RUN} {BASE}/src/baseline.py -c {cfg} && "
            f"CUDA_VISIBLE_DEVICES={gpu} {RUN} {BASE}/src/extract_embeddings.py --result_dir {out}")


def main():
    jobs = deque(KGES)
    total = len(jobs)
    print(f"[kge4] {total} KGE -> GPU {GPUS}", flush=True)
    running = {}
    fin = 0
    while jobs or running:
        free = [g for g in GPUS if g not in running]
        while free and jobs:
            kge = jobs[0]
            if done(kge):
                jobs.popleft(); fin += 1
                print(f"[skip ] {kge} 이미 학습됨 ({fin}/{total})", flush=True); continue
            jobs.popleft(); gpu = free.pop(0)
            lf = open(os.path.join(LOGDIR, f"kge4_{kge}.log"), "w")
            p = subprocess.Popen(cmd(kge, gpu), shell=True, stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (kge, p, lf)
            print(f"[start] {kge} -> GPU{gpu}", flush=True)
        time.sleep(30)
        for gpu, (kge, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); fin += 1
                print(f"[done ] {kge} GPU{gpu} rc={p.returncode} ({fin}/{total})", flush=True)
                del running[gpu]
    print(f"[kge4] ALL DONE ({fin}/{total})", flush=True)


if __name__ == "__main__":
    main()
