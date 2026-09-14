#!/usr/bin/env python3
"""Case-3 신약 시나리오 KGE 학습 (transe/rotate/complex) — novel 약물 고립 KG에서.

Case-1과 동일 HP, data.kg만 case3/kg_bio_334_case3.txt(novel 64개 edge 제거=고립).
각 KGE: baseline.py -c {kge}_case3.yaml -> extract_embeddings.py --result_dir ddibn_case3/{kge}/
GPU 3/5 분산 (Case-1 KGE 종료로 가용). skip-done = ddibn_case3/{kge}/ 에 *.npy 존재.
가설: novel 고립이라 cell08 S1/S2에서 KGE 종류 무관(complex 우위 사라짐) 기대.

Run: nohup micromamba run -n base python run_kge_case3.py > results/run_logs/kgec3_orch.log 2>&1 &
"""
import os, subprocess, time, glob
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
BASE = "/home/rudwls2717/Latex/experiments/v5_ddi-bench/kge_hetionet"
KGEDIR = os.path.join(HERE, "kge", "ddibn")          # configs 위치
OUTDIR = os.path.join(HERE, "kge", "ddibn_case3")    # case3 산출물
GPUS = [3, 5]
KGES = ["complex", "rotate", "transe"]               # 긴 것(complex 700ep) 먼저
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
RUN = "micromamba run -n DDIBench python3"


def done(kge):
    return bool(glob.glob(os.path.join(OUTDIR, kge, "*ent_*.npy")))


def cmd(kge, gpu):
    cfg = os.path.join(KGEDIR, "configs", f"{kge}_case3.yaml")
    out = os.path.join(OUTDIR, kge)
    return (f"CUDA_VISIBLE_DEVICES={gpu} {RUN} {BASE}/src/baseline.py -c {cfg} && "
            f"CUDA_VISIBLE_DEVICES={gpu} {RUN} {BASE}/src/extract_embeddings.py --result_dir {out}")


def main():
    jobs = deque(KGES); total = len(jobs)
    print(f"[kgec3] {total} KGE (case3 novel-고립) -> GPU {GPUS}", flush=True)
    running = {}; fin = 0
    while jobs or running:
        free = [g for g in GPUS if g not in running]
        while free and jobs:
            kge = jobs[0]
            if done(kge):
                jobs.popleft(); fin += 1
                print(f"[skip ] {kge} 이미 학습됨 ({fin}/{total})", flush=True); continue
            jobs.popleft(); gpu = free.pop(0)
            lf = open(os.path.join(LOGDIR, f"kgec3_{kge}.log"), "w")
            p = subprocess.Popen(cmd(kge, gpu), shell=True, stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (kge, p, lf)
            print(f"[start] {kge} -> GPU{gpu}", flush=True)
        time.sleep(30)
        for gpu, (kge, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); fin += 1
                print(f"[done ] {kge} GPU{gpu} rc={p.returncode} ({fin}/{total})", flush=True)
                del running[gpu]
    print("[kgec3] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
