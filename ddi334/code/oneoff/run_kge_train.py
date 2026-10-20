#!/usr/bin/env python3
"""DDI-334 KGE 학습 — existing_drug(ddibn) / novel_drug(ddibn_novel_drug) 공용 디스패처.
run_kge4.py + run_kge_case3.py를 --novel-drug 플래그로 통합(두 스크립트의 디스패치 로직
—GPU 큐, skip-if-done—이 동일했고 KGE 목록/출력 폴더/config 접미사만 달랐음).

--novel-drug 없음 (기본, existing_drug): transe는 이미 학습됨(제외) -> rotate/distmult/complex/transh.
  각 KGE: baseline.py -c {kge}.yaml -> extract_embeddings.py --result_dir kge/ddibn/{kge}/
  GPU 1/3/5 분산(인코더 종료로 가용). skip-done = kge/ddibn/{kge}/ 에 *.npy 존재.
--novel-drug: novel 약물 고립 KG에서 재학습 (HP는 existing_drug와 동일, data.kg/save.path만 다름).
  각 KGE: baseline.py -c {kge}_novel_drug.yaml -> extract_embeddings.py --result_dir kge/ddibn_novel_drug/{kge}/
  GPU 3/5 분산. skip-done = kge/ddibn_novel_drug/{kge}/ 에 *ent_*.npy 존재.
  가설(novel_drug): novel 고립이라 cell08 S1/S2에서 KGE 종류 무관(complex 우위 사라짐) 기대.

Run: nohup micromamba run -n base python run_kge_train.py [--novel-drug] \
     > results/run_logs/kge_train.log 2>&1 &
"""
import argparse, os, subprocess, time, glob
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
BASE = "/home/internal-server-user/Latex/experiments/v5_ddi-bench/kge_hetionet"
RUN = "micromamba run -n DDIBench python3"
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--novel-drug", action="store_true")
a = ap.parse_args()
NOVEL = a.novel_drug

CFGDIR = os.path.join(HERE, "kge", "ddibn", "configs")   # config 위치는 existing_drug/novel_drug 공용
OUTDIR = os.path.join(HERE, "kge", "ddibn_novel_drug") if NOVEL else os.path.join(HERE, "kge", "ddibn")
GPUS = [3, 5] if NOVEL else [1, 3, 5]
KGES = ["complex", "rotate", "transe"] if NOVEL else ["rotate", "distmult", "complex", "transh"]
CFG_SUFFIX = "_novel_drug" if NOVEL else ""
DONE_GLOB = "*ent_*.npy" if NOVEL else "*.npy"
TAG = "kge_novel_drug" if NOVEL else "kge_existing_drug"


def done(kge):
    return bool(glob.glob(os.path.join(OUTDIR, kge, DONE_GLOB)))


def cmd(kge, gpu):
    cfg = os.path.join(CFGDIR, f"{kge}{CFG_SUFFIX}.yaml")
    out = os.path.join(OUTDIR, kge)
    return (f"CUDA_VISIBLE_DEVICES={gpu} {RUN} {BASE}/src/baseline.py -c {cfg} && "
            f"CUDA_VISIBLE_DEVICES={gpu} {RUN} {BASE}/src/extract_embeddings.py --result_dir {out}")


def main():
    jobs = deque(KGES)
    total = len(jobs)
    print(f"[{TAG}] {total} KGE -> GPU {GPUS}", flush=True)
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
            lf = open(os.path.join(LOGDIR, f"{TAG}_{kge}.log"), "w")
            p = subprocess.Popen(cmd(kge, gpu), shell=True, stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (kge, p, lf)
            print(f"[start] {kge} -> GPU{gpu}", flush=True)
        time.sleep(30)
        for gpu, (kge, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); fin += 1
                print(f"[done ] {kge} GPU{gpu} rc={p.returncode} ({fin}/{total})", flush=True)
                del running[gpu]
    print(f"[{TAG}] ALL DONE ({fin}/{total})", flush=True)


if __name__ == "__main__":
    main()
