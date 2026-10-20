#!/usr/bin/env python3
"""Novel drug 시나리오 cell08/09/10/11 실행 공용 디스패처
(run_cell08_case3.py + run_rel_case3.py를 --mode로 통합 — 두 스크립트의 job_done/run_queue
디스패치 로직은 동일했고, job 목록을 만드는 방식(대상 cell 고정 vs KGE_TYPE 고정)만 달랐음).

NOVEL_DRUG=1 -> train.py가 kge/ddibn_novel_drug/{kge}/ 임베딩(novel 고립=random) 사용
(cell09/10은 NOVEL_DRUG=1이면 subgraph도 함께 novel로 교체).

--mode cell08 (cell08 KGE 비교, transe/rotate/complex 중 npy 있는 것만):
  결과 -> results/kge_comparison/novel_drug/{kge}/ (existing_drug의 kge_comparison/existing_drug와 분리).
  GPU당 1 job(KGE 학습과 컴퓨트 공유 방지).
--mode rel (기본값, cell 09/10/11 본체, transe 고립 KGE init만 사용):
  결과 -> results/encoders_novel_drug/ (existing_drug의 results/encoders_main/와 분리).
  cell11(가벼움) -> cell09(R-GCN subgraph) -> cell10(GT subgraph, 최장 ~63h) 순.
  의존: kge/ddibn_novel_drug/transe/*ent_*.npy 있어야 시작(없으면 대기).

Run:
  nohup micromamba run -n base python run_novel_drug_cells.py --mode cell08 \
     > results/run_logs/novel_drug_cell08.log 2>&1 &
  GPUS="3,5" nohup micromamba run -n base python run_novel_drug_cells.py --mode rel \
     > results/run_logs/novel_drug_rel.log 2>&1 &
"""
import argparse, os, subprocess, time, csv, collections
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
DDIBENCH = "micromamba run -n DDIBench python"
SEEDS = [0, 42, 124]

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="rel", choices=["cell08", "rel"])
a = ap.parse_args()


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


def run_queue(jobs, gpus, tag, sleep_s):
    jq = deque(jobs); running = {}
    while jq or running:
        free = [g for g in gpus if g not in running]
        while free and jq:
            name, tmpl = jq.popleft(); gpu = free.pop(0)
            lf = open(os.path.join(LOGDIR, f"{tag}_{name}.log"), "w")
            p = subprocess.Popen(tmpl.replace("{gpu}", str(gpu)), shell=True,
                                 stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (name, p, lf); print(f"start {name} -> GPU{gpu}", flush=True)
        time.sleep(sleep_s)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); print(f"done {name} rc={p.returncode}", flush=True)
                del running[gpu]


def has_npy(d):
    return os.path.isdir(d) and any("ent_" in f and f.endswith(".npy") for f in os.listdir(d))


def main_cell08():
    GPUS = [3, 5]
    CMP = os.path.join(HERE, "results", "kge_comparison", "novel_drug")
    _ALL_KGE = ["transe", "rotate", "complex"]
    KGES = [k for k in _ALL_KGE if has_npy(os.path.join(HERE, "kge", "ddibn_novel_drug", k))]
    print(f"[novel_drug/cell08] KGES(npy 있는것)={KGES}", flush=True)
    jobs = []
    for k in KGES:
        for s in SEEDS:
            if job_done(os.path.join(CMP, k, "summary.csv"), "08", s):
                print(f"skip c08_{k}_s{s} (done)", flush=True); continue
            cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} NOVEL_DRUG=1 KGE_TYPE={k} {DDIBENCH} train.py "
                   f"--cell 08 --dataset ddibn --seed {s} --gpu 0 --result_dir {os.path.join(CMP, k)}")
            jobs.append((f"c08_{k}_s{s}", cmd))
    print(f"[novel_drug/cell08] {len(jobs)} jobs -> GPU{GPUS}", flush=True)
    run_queue(jobs, GPUS, "novel_drug_cell08", 20)
    print("[novel_drug/cell08] ALL DONE", flush=True)


def main_rel():
    OUT = os.path.join(HERE, "results", "encoders_novel_drug")
    CELLS = ["11", "09", "10"]            # 가벼운 것 -> 무거운 것(cell10 최장)
    GPUS = [int(g) for g in os.environ.get("GPUS", "3,5").split(",") if g.strip()]
    TRANSE_NPY = os.path.join(HERE, "kge", "ddibn_novel_drug", "transe")
    print("[novel_drug/rel] transe(novel_drug) KGE 대기...", flush=True)
    while not has_npy(TRANSE_NPY):
        time.sleep(300)
    print(f"[novel_drug/rel] transe 준비됨. cells={CELLS} seeds={SEEDS} GPU={GPUS}", flush=True)
    jobs = []
    for c in CELLS:
        for s in SEEDS:
            if job_done(os.path.join(OUT, "summary.csv"), c, s):
                print(f"skip cell{c}_s{s} (done)", flush=True); continue
            cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} NOVEL_DRUG=1 KGE_TYPE=transe {DDIBENCH} train.py "
                   f"--cell {c} --dataset ddibn --seed {s} --gpu 0 --result_dir {OUT}")
            jobs.append((f"c{c}_s{s}", cmd))
    print(f"[novel_drug/rel] {len(jobs)} jobs", flush=True)
    run_queue(jobs, GPUS, "novel_drug_rel", 30)
    print("[novel_drug/rel] ALL DONE", flush=True)


if __name__ == "__main__":
    (main_cell08 if a.mode == "cell08" else main_rel)()
