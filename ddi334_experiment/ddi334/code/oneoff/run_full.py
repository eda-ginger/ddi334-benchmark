#!/usr/bin/env python3
"""DDI-334 ddibn 전체 실험 — GPU 작업큐 오케스트레이터.

빈 GPU에 다음 작업을 즉시 배정(웨이브 배리어 없음) -> 효율적.
  - 인코더 01-14 x 3 seed (DDIBench env, train.py)      -> results/ddibn/summary.csv
  - LLM V1-ZS / V1-FT x 3 모델 (vllm-llm env)            -> 같은 summary.csv (append)
GPU: 가용 {1,3,5}만 (0=ollama full / 2=금지 / 4=내 v5 실험 compute점유). 공유 안전 위해
vLLM mem-util 0.4. 평가/저장은 인코더와 동일 형식.

Run (백그라운드):
  nohup micromamba run -n base python run_full.py > results/run_logs/orchestrator.log 2>&1 &
  (env 무관 — 각 작업이 micromamba run 으로 자기 env 호출)
"""
import os, subprocess, time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
GPUS_FILE = os.path.join(HERE, "gpus.txt")   # 매 루프 재읽기 (모니터링이 갱신) — 동적 GPU 풀
DEFAULT_GPUS = [1, 3, 5]
SUMMARY = os.path.join(HERE, "results", "ddibn", "summary.csv")
SEEDS = [0, 42, 124]


def read_gpus():
    try:
        toks = open(GPUS_FILE).read().split()
        g = [int(x) for x in toks if x.strip().isdigit() and int(x) != 2]  # GPU2 영구 금지
        return g or DEFAULT_GPUS
    except Exception:
        return DEFAULT_GPUS


def _summary_keys():
    """완료 판정용: summary.csv의 (cell, seed, split) 집합."""
    import csv as _csv
    keys = set()
    if os.path.exists(SUMMARY):
        try:
            for r in _csv.DictReader(open(SUMMARY)):
                keys.add((str(r.get("cell", "")), str(r.get("seed", "")), str(r.get("split", ""))))
        except Exception:
            pass
    return keys


def is_done(name):
    """enc_{cell}_s{seed} -> 3 split / llm_{tag} -> v1zs+v1ft 각 3 split 다 있으면 완료."""
    k = _summary_keys()
    if name.startswith("enc_"):
        cell, seed = name[4:].split("_s")
        return all((cell, seed, sp) in k for sp in ("S0", "S1", "S2"))
    if name.startswith("llm_"):
        tag = name[4:]
        return all((f"{pre}_{tag}", "", sp) in k
                   for pre in ("v1zs", "v1ft") for sp in ("S0", "S1", "S2"))
    return False
ENC_CELLS = [f"{i:02d}" for i in range(1, 15)]
LLM = [("qwen", "Qwen/Qwen2.5-3B-Instruct", False),
       ("phi", "microsoft/Phi-3.5-mini-instruct", False),
       ("gemma", "google/gemma-2-2b-it", True)]
DDIBENCH = "micromamba run -n DDIBench python"
VLLM = "micromamba run -n vllm-llm python"
MEM = "0.4"
LOGDIR = os.path.join(HERE, "results", "run_logs")
os.makedirs(LOGDIR, exist_ok=True)


def enc_cmd(cell, seed):
    return f"CUDA_VISIBLE_DEVICES={{gpu}} {DDIBENCH} train.py --cell {cell} --dataset ddibn --seed {seed} --gpu 0"


def llm_cmd(tag, path, ms):
    # NOTE: ft_train_v1.py / llm_infer.py 는 --gpu 값으로 CUDA_VISIBLE_DEVICES를 직접 설정.
    #       따라서 CUDA_VISIBLE_DEVICES 프리픽스 없이 --gpu 에 실제 물리 인덱스를 넘긴다.
    flag = "--merge-system" if ms else ""
    sane = path.replace("/", "_")
    merged = f"results/ft_v1/ddibn_{sane}_seed42_merged"
    parts = []
    port = "$((12341+{gpu}))"   # 병렬 LLM 잡 포트 충돌 방지 (GPU별 고유)
    # V1 Zero-shot (3 splits)
    for sp in ("S0", "S1", "S2"):
        parts.append(f"{VLLM} llm_infer.py --version v1 --dataset ddibn "
                     f"--model {path} --gpu {{gpu}} --port {port} --split {sp} --label v1zs_{tag} --gpu-mem-util {MEM} {flag}")
    # V1 QLoRA FT (train+merge) -> eval 3 splits
    parts.append(f"{VLLM} ft_train_v1.py --model {path} --dataset ddibn --gpu {{gpu}} --seed 42")
    for sp in ("S0", "S1", "S2"):
        parts.append(f"{VLLM} llm_infer.py --version v1 --dataset ddibn "
                     f"--model {merged} --gpu {{gpu}} --port {port} --split {sp} --label v1ft_{tag} --gpu-mem-util {MEM} {flag}")
    return " && ".join(parts)


def main():
    jobs = deque()
    for c in ENC_CELLS:
        for s in SEEDS:
            jobs.append((f"enc_{c}_s{s}", enc_cmd(c, s)))
    for tag, path, ms in LLM:
        jobs.append((f"llm_{tag}", llm_cmd(tag, path, ms)))
    total = len(jobs)
    print(f"[orchestrator] {total} jobs | GPU pool <- {GPUS_FILE} (동적), 매 루프 재확인", flush=True)

    running = {}   # gpu -> (name, popen, logf)
    done = 0
    while jobs or running:
        desired = read_gpus()                                  # 매 루프 가용 GPU 재읽기
        free = [g for g in desired if g not in running]        # 새로 빈/추가된 GPU 자동 반영
        while free and jobs:
            name, tmpl = jobs[0]
            if is_done(name):                                  # 이미 완료(재시작 안전) -> skip
                jobs.popleft(); done += 1
                print(f"[skip ] {name} 이미 완료 ({done}/{total})", flush=True); continue
            jobs.popleft(); gpu = free.pop(0)
            cmd = tmpl.replace("{gpu}", str(gpu))
            lf = open(os.path.join(LOGDIR, f"{name}.log"), "w")
            p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=lf, executable="/bin/bash")
            running[gpu] = (name, p, lf)
            print(f"[start] {name} -> GPU{gpu} (pool={desired}, {done}/{total} done)", flush=True)
        time.sleep(20)
        for gpu, (name, p, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close(); done += 1
                print(f"[done ] {name} GPU{gpu} rc={p.returncode} ({done}/{total})", flush=True)
                del running[gpu]                               # free는 다음 루프 desired-running으로 재계산
    print(f"[orchestrator] ALL DONE ({done}/{total})", flush=True)


if __name__ == "__main__":
    main()
