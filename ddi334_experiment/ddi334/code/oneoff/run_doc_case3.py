"""Case-3 DOC: cell13(BioBERT frozen) × 2 변형 × 3 seed.

변형 (DOC_INPUT env):
  - smiles_c3    : BioBERT(SMILES 문자열)        -> results/ddibn_case3_doc_smiles/
  - rdkitdesc_c3 : BioBERT(RDKit 30-feature JSON) -> results/ddibn_case3_doc_rdkitdesc/
인코더/평가는 Case-1 cell13과 동일(BioBERT frozen + split별 val-best). 입력 텍스트만 구조유도.
cell13은 frozen feature라 가벼움(분 단위). 의존: biobert_smiles.pt / biobert_rdkitdesc.pt.

Run: GPUS="3,5" nohup micromamba run -n base python run_doc_case3.py > results/run_logs/doc_case3.log 2>&1 &
"""
import os, subprocess, time, csv, collections
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PRECOMP = os.path.join(HERE, "ddibn", "precompute")
LOGDIR = os.path.join(HERE, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
DDIBENCH = "micromamba run -n DDIBench python"
SEEDS = [0, 42, 124]
GPUS = [int(g) for g in os.environ.get("GPUS", "3,5").split(",") if g.strip()]
VARIANTS = [   # (DOC_INPUT, pt파일, result_dir)
    ("smiles_c3", "biobert_smiles.pt", os.path.join(HERE, "results", "ddibn_case3_doc_smiles")),
    ("rdkitdesc_c3", "biobert_rdkitdesc.pt", os.path.join(HERE, "results", "ddibn_case3_doc_rdkitdesc")),
]


def job_done(out, seed):
    s = os.path.join(out, "summary.csv")
    done = collections.defaultdict(set)
    try:
        for r in csv.DictReader(open(s)):
            if r.get("cell") == "13":
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
            lf = open(os.path.join(LOGDIR, f"docc3_{name}.log"), "w")
            p = subprocess.Popen(tmpl.replace("{gpu}", str(gpu)), shell=True,
                                 stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (name, p, lf); print(f"start {name} -> GPU{gpu}", flush=True)
        time.sleep(15)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); print(f"done {name} rc={p.returncode}", flush=True)
                del running[gpu]


def main():
    # precompute 대기
    for _, pt, _ in VARIANTS:
        while not os.path.exists(os.path.join(PRECOMP, pt)):
            print(f"[doc_case3] {pt} 대기...", flush=True); time.sleep(60)
    print(f"[doc_case3] biobert .pt 준비됨. variants={[v[0] for v in VARIANTS]} GPU={GPUS}", flush=True)
    jobs = []
    for inp, pt, out in VARIANTS:
        for s in SEEDS:
            if job_done(out, s):
                print(f"skip {inp}_s{s} (done)", flush=True); continue
            cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} DOC_INPUT={inp} {DDIBENCH} train.py "
                   f"--cell 13 --dataset ddibn --seed {s} --gpu 0 --result_dir {out}")
            jobs.append((f"{inp}_s{s}", cmd))
    print(f"[doc_case3] {len(jobs)} jobs", flush=True)
    run_queue(jobs)
    print("[doc_case3] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
