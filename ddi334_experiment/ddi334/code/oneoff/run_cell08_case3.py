#!/usr/bin/env python3
"""Case-3 신약 시나리오 cell08 KGE 비교 (transe/rotate/complex).

CASE3=1 -> train.py가 kge/ddibn_case3/{kge}/ 임베딩(novel 고립=random) 사용.
DDI 데이터/splits는 Case-1과 동일(ddibn), KG 임베딩만 case3로 교체.
결과 -> results/kge_compare_case3/{kge}/ (Case-1의 kge_compare와 분리).
GPU당 1 job(KGE 학습과 컴퓨트 공유 방지). job_done이면 skip.

Run: nohup micromamba run -n base python run_cell08_case3.py > results/run_logs/cell08_case3.log 2>&1 &
"""
import os, subprocess, time, csv, collections
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
CMP = os.path.join(HERE, "results", "kge_compare_case3")
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
DDIBENCH = "micromamba run -n DDIBench python"
SEEDS = [0, 42, 124]
GPUS = [3, 5]
_ALL_KGE = ["transe", "rotate", "complex"]   # Case-3 후보 (사용자 지정)


def _has_npy(k):
    d = os.path.join(HERE, "kge", "ddibn_case3", k)
    return os.path.isdir(d) and any("ent_" in f and f.endswith(".npy") for f in os.listdir(d))


KGES = [k for k in _ALL_KGE if _has_npy(k)]


def splits_done(summary, cell):
    d = collections.defaultdict(set)
    try:
        for r in csv.DictReader(open(summary)):
            if r.get("cell") == cell:
                d[r["seed"]].add(r["split"])
    except Exception:
        pass
    return d


def job_done(summary, cell, seed):
    return {"S0", "S1", "S2"} <= splits_done(summary, cell).get(str(seed), set())


def run_queue(jobs):
    jq = deque(jobs); running = {}
    while jq or running:
        free = [g for g in GPUS if g not in running]
        while free and jq:
            name, tmpl = jq.popleft(); gpu = free.pop(0)
            lf = open(os.path.join(LOGDIR, f"cell08c3_{name}.log"), "w")
            p = subprocess.Popen(tmpl.replace("{gpu}", str(gpu)), shell=True,
                                 stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (name, p, lf); print(f"start {name} -> GPU{gpu}", flush=True)
        time.sleep(20)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); print(f"done {name} rc={p.returncode}", flush=True)
                del running[gpu]


def main():
    print(f"[cell08_case3] KGES(npy 있는것)={KGES}", flush=True)
    jobs = []
    for k in KGES:
        for s in SEEDS:
            if job_done(os.path.join(CMP, k, "summary.csv"), "08", s):
                print(f"skip c08c3_{k}_s{s} (done)", flush=True); continue
            # CASE3=1 -> train.py가 ddibn_case3/{k} 임베딩 사용
            cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} CASE3=1 KGE_TYPE={k} {DDIBENCH} train.py "
                   f"--cell 08 --dataset ddibn --seed {s} --gpu 0 --result_dir {os.path.join(CMP, k)}")
            jobs.append((f"c08c3_{k}_s{s}", cmd))
    print(f"[cell08_case3] {len(jobs)} jobs -> GPU{GPUS}", flush=True)
    run_queue(jobs)
    print("[cell08_case3] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
