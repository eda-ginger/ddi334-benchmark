#!/usr/bin/env python3
"""Existing/Novel 교차 실험 (cell09 R-GCN subgraph만, Appendix J, 2026-10-19 추가).

기존 A(existing emb + existing subgraph)/D(novel emb + novel subgraph)는 이미
results/encoders_main, results/encoders_novel_drug에 있음. 여기서는 2개 신규 조합만 수행:
  B: existing subgraph + novel(case3)   TransE 임베딩   (EMB_SRC=case3,  SUBGRAPH_SRC=case1)
  C: novel(case3) subgraph + existing   TransE 임베딩   (EMB_SRC=case1,  SUBGRAPH_SRC=case3)
(molfp 초기화로 같은 교차를 보는 E/F 변형은 검토 후 보류 — cell11 쪽 molfp/molgraph 비교로 대체됨.)

결과 -> results/appendix_j_embedding_subgraph_cross/{novel_embedding_existing_subgraph,
existing_embedding_novel_subgraph}/ (B, C 각각 summary.csv). GPU 제약 없음(사용자 지시, 2026-10-19) —
서버의 6개 GPU 모두 후보로 사용(다른 사용자 프로세스가 있어도 메모리 여유로 공유 가능).

Run: nohup micromamba run -n base python run_rel_cross.py > results/run_logs/rel_cross.log 2>&1 &
"""
import os, subprocess, time, csv, collections
from collections import deque

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(HERE)
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
DDIBENCH = "micromamba run -n DDIBench python"
SEEDS = [0, 42, 124]
GPUS = [int(g) for g in os.environ.get("GPUS", "0,1,2,3,4,5").split(",") if g.strip()]

COMBOS = {
    "B": {"EMB_SRC": "case3", "SUBGRAPH_SRC": "case1"},
    "C": {"EMB_SRC": "case1", "SUBGRAPH_SRC": "case3"},
}

COMBO_DIR = {
    "B": "novel_embedding_existing_subgraph",
    "C": "existing_embedding_novel_subgraph",
}


def out_dir(combo):
    return os.path.join(HERE, "results", "appendix_j_embedding_subgraph_cross", COMBO_DIR[combo])


def job_done(combo, seed):
    s = os.path.join(out_dir(combo), "summary.csv")
    done = collections.defaultdict(set)
    try:
        for r in csv.DictReader(open(s)):
            if r.get("cell") == "09":
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
            lf = open(os.path.join(LOGDIR, f"relcross_{name}.log"), "w")
            p = subprocess.Popen(tmpl.replace("{gpu}", str(gpu)), shell=True,
                                 stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (name, p, lf); print(f"start {name} -> GPU{gpu}", flush=True)
        time.sleep(30)
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
                print(f"skip combo{combo}_s{s} (done)", flush=True); continue
            cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} {env_str} {DDIBENCH} code/train.py "
                   f"--cell 09 --dataset ddibn --seed {s} --gpu 0 --result_dir {od}")
            jobs.append((f"{combo}_s{s}", cmd))
    print(f"[rel_cross] {len(jobs)} jobs, GPUs={GPUS}", flush=True)
    run_queue(jobs)
    print("[rel_cross] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
