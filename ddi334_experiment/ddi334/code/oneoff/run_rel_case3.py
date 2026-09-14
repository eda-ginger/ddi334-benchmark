#!/usr/bin/env python3
"""Case-3 신약 시나리오 REL 본체: cell 09/10/11 (transe 고립 KGE init).

- KGE init = case3 transe (novel 고립 -> random). CASE3=1로 train.py가 case3 임베딩+case3 triples(subgraph) 사용.
- 평가 = train.py 기본(split별 val accuracy best epoch). 3 seed.
- 결과 -> results/ddibn_case3/ (Case-1 results/ddibn_acc/와 분리).
- cell11(가벼움) -> cell09(R-GCN subgraph) -> cell10(GT subgraph, 최장 ~63h) 순.
- 의존: kge/ddibn_case3/transe/*ent_*.npy 있어야 시작 (없으면 대기).

Run: GPUS="3,5" nohup micromamba run -n base python run_rel_case3.py > results/run_logs/rel_case3.log 2>&1 &
"""
import os, subprocess, time, csv, collections
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
OUT = os.path.join(HERE, "results", "ddibn_case3")
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
DDIBENCH = "micromamba run -n DDIBench python"
SEEDS = [0, 42, 124]
CELLS = ["11", "09", "10"]            # 가벼운 것 -> 무거운 것(cell10 최장)
GPUS = [int(g) for g in os.environ.get("GPUS", "3,5").split(",") if g.strip()]
TRANSE_NPY = os.path.join(HERE, "kge", "ddibn_case3", "transe")


def transe_ready():
    return os.path.isdir(TRANSE_NPY) and any("ent_" in f and f.endswith(".npy") for f in os.listdir(TRANSE_NPY))


def job_done(cell, seed):
    s = os.path.join(OUT, "summary.csv")
    done = collections.defaultdict(set)
    try:
        for r in csv.DictReader(open(s)):
            if r.get("cell") == cell:
                done[r["seed"]].add(r["split"])
    except Exception:
        pass
    return {"S0", "S1", "S2"} <= done.get(str(seed), set())


def run_queue(jobs):
    jq = deque(jobs); running = {}
    while jq or running:
        free = [g for g in GPUS if g not in running]
        while free and jq:
            name, tmpl = jq.popleft(); gpu = free.pop(0)
            lf = open(os.path.join(LOGDIR, f"relc3_{name}.log"), "w")
            p = subprocess.Popen(tmpl.replace("{gpu}", str(gpu)), shell=True,
                                 stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (name, p, lf); print(f"start {name} -> GPU{gpu}", flush=True)
        time.sleep(30)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); print(f"done {name} rc={p.returncode}", flush=True)
                del running[gpu]


def main():
    print(f"[rel_case3] transe(case3) KGE 대기...", flush=True)
    while not transe_ready():
        time.sleep(300)
    print(f"[rel_case3] transe 준비됨. cells={CELLS} seeds={SEEDS} GPU={GPUS}", flush=True)
    jobs = []
    for c in CELLS:
        for s in SEEDS:
            if job_done(c, s):
                print(f"skip cell{c}_s{s} (done)", flush=True); continue
            cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} CASE3=1 KGE_TYPE=transe {DDIBENCH} train.py "
                   f"--cell {c} --dataset ddibn --seed {s} --gpu 0 --result_dir {OUT}")
            jobs.append((f"c{c}_s{s}", cmd))
    print(f"[rel_case3] {len(jobs)} jobs", flush=True)
    run_queue(jobs)
    print("[rel_case3] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
