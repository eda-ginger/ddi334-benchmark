#!/usr/bin/env python3
"""DDI-334 KGE 비교 -> 09/10/11 재실행 체인 (기존 결과 절대 미변경, 전부 신규 폴더).

Phase1: 4 KGE(run_kge4.py) 학습 완료까지 대기 (kge/ddibn/{kge}/*ent_*.npy).
Phase2: cell08 x {rotate,distmult,complex,transh} x 3seed -> results/kge_compare/{kge}/  (KGE_TYPE env)
        * transe cell08은 이미 results/ddibn_acc/ 에 있음 (유지, 재실행 X)
Phase3: cell08 S0 accuracy로 최고 KGE 선택 (transe 포함 5종 비교) -> results/kge_compare/best.json
Phase4: best가 transe가 아니면 cell 09/10/11 x best x 3seed -> results/ddibn_kge_{best}/  (KGE_TYPE env)
        * 기존 ddibn_acc 의 transe-init 09/10/11 은 유지
Phase5: results/kge_compare/CHAIN_DONE 마커 + 요약.

GPU 1/3/5 (KGE 학습 종료 후 가용). skip-done 재시작 안전.
Run: nohup micromamba run -n base python run_kge_chain.py > results/run_logs/chain_orch.log 2>&1 &
"""
import os, subprocess, time, glob, csv, collections, statistics as st, json
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
KGEDIR = os.path.join(HERE, "kge", "ddibn")
ACC = os.path.join(HERE, "results", "ddibn_acc")
CMP = os.path.join(HERE, "results", "kge_compare")
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
GPUS = [1, 3, 5]
KGES = ["rotate", "distmult", "complex", "transh"]
SEEDS = [0, 42, 124]
DDIBENCH = "micromamba run -n DDIBench python"


def log(m):
    print(f"[chain] {m}", flush=True)


def kge_trained(k):
    return bool(glob.glob(os.path.join(KGEDIR, k, "*ent_*.npy")))


def splits_done(summary, cell):
    if not os.path.exists(summary):
        return {}
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
    """jobs: list of (name, cmd_template{gpu}). GPU 1/3/5 분산 실행."""
    jq = deque(jobs)
    running = {}
    while jq or running:
        free = [g for g in GPUS if g not in running]
        while free and jq:
            name, tmpl = jq.popleft()
            gpu = free.pop(0)
            lf = open(os.path.join(LOGDIR, f"chain_{name}.log"), "w")
            p = subprocess.Popen(tmpl.replace("{gpu}", str(gpu)), shell=True,
                                 stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (name, p, lf)
            log(f"start {name} -> GPU{gpu}")
        time.sleep(20)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); log(f"done {name} rc={p.returncode}")
                del running[gpu]


def cell08_s0_acc(summary):
    """summary.csv의 cell08 S0 accuracy 평균(%)."""
    vals = []
    if os.path.exists(summary):
        for r in csv.DictReader(open(summary)):
            if r.get("cell") == "08" and r.get("split") == "S0" and r.get("test_accuracy"):
                vals.append(float(r["test_accuracy"]) * 100)
    return st.mean(vals) if vals else None


def main():
    # Phase 1
    log("Phase1: KGE 4종 학습 완료 대기")
    while not all(kge_trained(k) for k in KGES):
        time.sleep(300)
    log("Phase1 완료: 4 KGE npy 확인")

    # Phase 2: cell08 x kge x seed
    log("Phase2: cell08 x 4KGE x 3seed -> kge_compare")
    jobs = []
    for k in KGES:
        for s in SEEDS:
            if job_done(os.path.join(CMP, k, "summary.csv"), "08", s):
                continue
            cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} KGE_TYPE={k} {DDIBENCH} train.py "
                   f"--cell 08 --dataset ddibn --seed {s} --gpu 0 --result_dir {os.path.join(CMP, k)}")
            jobs.append((f"c08_{k}_s{s}", cmd))
    run_queue(jobs)

    # Phase 3: 최고 KGE 선택 (cell08 S0 accuracy)
    log("Phase3: 최고 KGE 선택")
    scores = {"transe": cell08_s0_acc(os.path.join(ACC, "summary.csv"))}
    for k in KGES:
        scores[k] = cell08_s0_acc(os.path.join(CMP, k, "summary.csv"))
    valid = {k: v for k, v in scores.items() if v is not None}
    best = max(valid, key=valid.get)
    json.dump({"scores_cell08_S0_acc": scores, "best": best}, open(os.path.join(CMP, "best.json"), "w"), indent=1)
    log(f"cell08 S0 acc: {scores} -> best={best}")

    # Phase 4: best로 09/10/11 (transe면 이미 ddibn_acc에 있음)
    if best == "transe":
        log("best=transe -> 09/10/11 이미 ddibn_acc에 존재, 재실행 불필요")
    else:
        outdir = os.path.join(HERE, "results", f"ddibn_kge_{best}")
        log(f"Phase4: cell 09/10/11 x {best} x 3seed -> {outdir}")
        jobs = []
        for c in ("09", "10", "11"):
            for s in SEEDS:
                if job_done(os.path.join(outdir, "summary.csv"), c, s):
                    continue
                cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} KGE_TYPE={best} {DDIBENCH} train.py "
                       f"--cell {c} --dataset ddibn --seed {s} --gpu 0 --result_dir {outdir}")
                jobs.append((f"c{c}_{best}_s{s}", cmd))
        run_queue(jobs)

    open(os.path.join(CMP, "CHAIN_DONE"), "w").write(f"best={best}\n")
    log(f"CHAIN DONE. best={best}")


if __name__ == "__main__":
    main()
