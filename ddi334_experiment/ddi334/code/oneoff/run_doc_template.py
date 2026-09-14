"""DOC 템플릿 재설계: cell13(frozen text encoder) × 3 tier(real/genes/ideal) × 3 seed × N backbone.

2026-07-27 미팅 피드백 반영 — LLM(§5) real/genes/ideal 템플릿과 통일한 cell13 설계.
`05_실험설계(DDI334).md` §8 참조. cell14(구 drug name 전용)는 흡수·폐기, 이 트랙에서 실행 안 함.
2026-07-27 확장(§8.5): DOC_MODEL(backbone) 축 추가 — biobert/pubmedbert/scibert 비교.
입력: llm_prompts.py DDI334Prompt.drug_text() 템플릿 -> build_biobert_template.py --tag {model}로
      사전 생성한 {model}_{real,genes,ideal}.pt. 인코더/평가는 기존 cell13과 동일(frozen + split별 val-best).
cell13은 frozen feature라 가벼움(분 단위).

MODELS env(comma, 기본 "biobert")로 대상 backbone 선택. biobert는 result_dir이 ddibn_doc_{tier}
(기존 관례 유지), 그 외는 ddibn_doc_{model}_{tier}.

선행 조건: build_biobert_template.py --tag {model} 실행 완료({model}_real/genes/ideal.pt 존재).

Run: GPUS="3,5" MODELS="pubmedbert,scibert" nohup micromamba run -n base python run_doc_template.py \
     > results/run_logs/doc_template.log 2>&1 &
"""
import os, subprocess, time, csv, collections
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
CODE_DIR = os.path.dirname(HERE)                      # ddi334/code
DDI334_DIR = os.path.dirname(CODE_DIR)                 # ddi334
DATASET_PRECOMP = os.path.join(DDI334_DIR, 'data', 'ddibn', 'precompute')  # train.py가 실제 읽는 경로
LOGDIR = os.path.join(DDI334_DIR, "results", "run_logs"); os.makedirs(LOGDIR, exist_ok=True)
DDIBENCH = "micromamba run -n DDIBench python"
SEEDS = [0, 42, 124]
GPUS = [int(g) for g in os.environ.get("GPUS", "3,5").split(",") if g.strip()]
SKIP = set(s for s in os.environ.get("SKIP", "").split(",") if s)   # 예: 다른 프로세스가 이미 돌리는 중인 "{model}_{tier}_s{seed}"
MODELS = [m for m in os.environ.get("MODELS", "biobert").split(",") if m]
TIERS = ["real", "genes", "ideal"]


def result_dir(model, tier):
    if model == "biobert":
        return os.path.join(DDI334_DIR, "results", f"ddibn_doc_{tier}")   # 기존 관례 유지
    return os.path.join(DDI334_DIR, "results", f"ddibn_doc_{model}_{tier}")


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
            lf = open(os.path.join(LOGDIR, f"doctpl_{name}.log"), "w")
            p = subprocess.Popen(tmpl.replace("{gpu}", str(gpu)), shell=True,
                                 stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (name, p, lf); print(f"start {name} -> GPU{gpu}", flush=True)
        time.sleep(15)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); print(f"done {name} rc={p.returncode}", flush=True)
                del running[gpu]


def main():
    for model in MODELS:
        for tier in TIERS:
            path = os.path.join(DATASET_PRECOMP, f"{model}_{tier}.pt")
            if not os.path.exists(path):
                print(f"[doc_template] 필요한 precompute 없음: {path}\n"
                      f"  먼저 실행: micromamba run -n DDIBench python ../build_biobert_template.py --tag {model}",
                      flush=True)
                return
    print(f"[doc_template] MODELS={MODELS} 전체 precompute 확인됨. GPU={GPUS}", flush=True)
    jobs = []
    for model in MODELS:
        for tier in TIERS:
            out = result_dir(model, tier)
            for s in SEEDS:
                key = f"{model}_{tier}_s{s}"
                if key in SKIP:
                    print(f"skip {key} (already running elsewhere)", flush=True); continue
                if job_done(out, s):
                    print(f"skip {key} (done)", flush=True); continue
                cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} DOC_INPUT={tier} DOC_MODEL={model} {DDIBENCH} ../train.py "
                       f"--cell 13 --dataset ddibn --seed {s} --gpu 0 --result_dir {out}")
                jobs.append((key, cmd))
    print(f"[doc_template] {len(jobs)} jobs", flush=True)
    run_queue(jobs)
    print("[doc_template] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
