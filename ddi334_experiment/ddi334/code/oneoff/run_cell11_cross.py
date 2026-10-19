#!/usr/bin/env python3
"""cell11(R-GCN on whole DDI graph, 약물 노드만) 노드 초기화 비교 (2026-10-19 추가).

cell09 KG subgraph(bio+DDI 혼합, 유전자/질병 포함)와 달리 cell11은 순수 약물-약물 DDI
그래프(1710 drug node)만 사용한다. molfp(Morgan FP, 1024d)는 train.py의 cell11 기본값이 되어
results/encoders_main/에 이미 있음(TransE 비교값은 results/encoders_novel_drug/). 여기서는
molgraph(분자그래프 원시 원자피처 sum-pool, 55d, frozen) 초기화(NODE_INIT=molgraph)만 추가 실행.

결과 -> results/ddi_network_molgraph_init/ (summary.csv).

Run: nohup micromamba run -n base python run_cell11_cross.py > results/run_logs/cell11_cross.log 2>&1 &
"""
import os, subprocess, time, csv, collections
from collections import deque

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(HERE)
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
DDIBENCH = "micromamba run -n DDIBench python"
SEEDS = [0, 42, 124]
GPUS = [int(g) for g in os.environ.get("GPUS", "0,1,2,3,4,5").split(",") if g.strip()]

COMBOS = {
    "molgraph": {"NODE_INIT": "molgraph"},
}


def out_dir(combo):
    return os.path.join(HERE, "results", "ddi_network_molgraph_init")


def job_done(combo, seed):
    s = os.path.join(out_dir(combo), "summary.csv")
    done = collections.defaultdict(set)
    try:
        for r in csv.DictReader(open(s)):
            if r.get("cell") == "11":
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
            lf = open(os.path.join(LOGDIR, f"cell11cross_{name}.log"), "w")
            p = subprocess.Popen(tmpl.replace("{gpu}", str(gpu)), shell=True,
                                 stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (name, p, lf); print(f"start {name} -> GPU{gpu}", flush=True)
        time.sleep(15)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); print(f"done {name} rc={p.returncode}", flush=True)
                del running[gpu]


def main():
    jobs = []
    for combo, envs in COMBOS.items():
        od = out_dir(combo); os.makedirs(od, exist_ok=True)
        env_str = " ".join(f"{k}={v}" for k, v in envs.items())
        for s in SEEDS:
            if job_done(combo, s):
                print(f"skip {combo}_s{s} (done)", flush=True); continue
            cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} {env_str} {DDIBENCH} code/train.py "
                   f"--cell 11 --dataset ddibn --seed {s} --gpu 0 --result_dir {od}")
            jobs.append((f"{combo}_s{s}", cmd))
    print(f"[cell11_cross] {len(jobs)} jobs, GPUs={GPUS}", flush=True)
    run_queue(jobs)
    print("[cell11_cross] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
