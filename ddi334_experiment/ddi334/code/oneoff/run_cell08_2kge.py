#!/usr/bin/env python3
"""끝난 2 KGE(distmult/rotate)로 cell08 먼저 실행 (complex/transh 완주 대기 중 병행).

GPU3/5(메모리 79GB 여유, KGE와 컴퓨트만 공유)에 cell08 x {distmult,rotate} x 3seed = 6run 분산.
결과 -> results/kge_compare/{kge}/ (run_kge_chain Phase2와 동일 경로 -> 나중에 chain이 job_done으로 skip).
GPU당 1 job씩만(KGE와 컴퓨트 공유라 과포화 방지). job_done이면 skip.

Run: nohup micromamba run -n base python run_cell08_2kge.py > results/run_logs/cell08_2kge.log 2>&1 &
"""
import os, subprocess, time, csv, collections
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
CMP = os.path.join(HERE, "results", "kge_compare")
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
DDIBENCH = "micromamba run -n DDIBench python"
SEEDS = [0, 42, 124]
GPUS = [3, 5]                       # 메모리 79GB 여유, KGE 학습과 공유
# npy(임베딩 학습완료)가 있는 KGE만 cell08 실행. job_done이면 자동 skip.
_ALL_KGE = ["distmult", "rotate", "transh", "complex"]


def _has_npy(k):
    d = os.path.join(HERE, "kge", "ddibn", k)
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
            lf = open(os.path.join(LOGDIR, f"cell08_{name}.log"), "w")
            p = subprocess.Popen(tmpl.replace("{gpu}", str(gpu)), shell=True,
                                 stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (name, p, lf); print(f"start {name} -> GPU{gpu}", flush=True)
        time.sleep(20)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); print(f"done {name} rc={p.returncode}", flush=True)
                del running[gpu]


def main():
    jobs = []
    for k in KGES:
        for s in SEEDS:
            if job_done(os.path.join(CMP, k, "summary.csv"), "08", s):
                print(f"skip c08_{k}_s{s} (done)", flush=True); continue
            cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} KGE_TYPE={k} {DDIBENCH} train.py "
                   f"--cell 08 --dataset ddibn --seed {s} --gpu 0 --result_dir {os.path.join(CMP, k)}")
            jobs.append((f"c08_{k}_s{s}", cmd))
    print(f"[cell08_2kge] {len(jobs)} jobs -> GPU{GPUS}", flush=True)
    run_queue(jobs)
    print("[cell08_2kge] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
