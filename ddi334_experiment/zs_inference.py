"""V5 DDI-Bench — LLM Zero-Shot Inference (Cells 15-17).

86-class multi-class prediction via guided_choice(["0".."85"]).
Drug input: Name + SMILES (설계문서 §3.1 프롬프트 준용).
vLLM serve + OpenAI client 패턴 (gs_ddi_eval_port.py 재사용).

Usage:
    python experiment/zs_inference.py --cell 15 --gpu 0

Cells:
    15  Phi-3.5-mini-instruct   (2.7B)
    16  Qwen2.5-3B-Instruct     (3B)   [or 14B if 3B unavailable]
    17  gemma-2-2b-it            (2B)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from tqdm import tqdm

# ── 경로 상수 ──────────────────────────────────────────────────────────────
EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR       = EXPERIMENT_DIR / "data"
RESULTS_DIR    = EXPERIMENT_DIR / "results_drugbank"
SUMMARY_CSV    = RESULTS_DIR / "summary.csv"
ZS_CACHE_DIR   = RESULTS_DIR / "zs_cache"

LATEX_DIR      = Path("/home/rudwls2717/Latex")
NAMES_JSON     = LATEX_DIR / "data/embeddings/drugbank_names.json"
DESC_JSON      = LATEX_DIR / "data/descriptions/drugbank.json"
SMILES_JSON    = LATEX_DIR / "data/smiles/drugbank_id2smiles.json"
DDI_LABELS_JSON = EXPERIMENT_DIR.parent / "ddi_type_descriptions_ryu86.json"
SPLIT_FILES    = {
    "S0": (DATA_DIR / "valid_S0.txt", DATA_DIR / "test_S0.txt"),
    "S1": (DATA_DIR / "valid_S1.txt", DATA_DIR / "test_S1.txt"),
    "S2": (DATA_DIR / "valid_S2.txt", DATA_DIR / "test_S2.txt"),
}

CELL_MODELS = {
    "15": "microsoft/Phi-3.5-mini-instruct",
    "16": "Qwen/Qwen2.5-3B-Instruct",
    "17": "google/gemma-2-2b-it",
}
CELL_PARAMS = {
    "15": 2_700_000_000,
    "16": 3_000_000_000,
    "17": 2_000_000_000,
}
# Gemma2 계열은 system role 미지원 → user 메시지로 merge
MERGE_SYSTEM_CELLS = {"17"}

NUM_LABELS = 86
GUIDED_CHOICES = [str(i) for i in range(NUM_LABELS)]
VLLM_PORT = 12340  # train.py와 충돌 없는 포트
BATCH_SIZE = 20
NUM_WORKERS = 6


# ── 프롬프트 빌더 ──────────────────────────────────────────────────────────
def build_system(label_map: dict) -> str:
    label_lines = "\n".join(
        f"{i}: {label_map[str(i)].replace('#Drug1', 'Drug1').replace('#Drug2', 'Drug2')}"
        for i in range(NUM_LABELS)
    )
    return (
        "You are a pharmacology expert. Given the following information on "
        "two drugs, predict their interaction type by selecting the single "
        "best label:\n\n"
        f"{label_lines}\n\n"
        "Output only the integer (0-85) after ##Answer: <integer>"
    )


def build_user(n1: str, s1: str, n2: str, s2: str) -> str:
    return (
        f"Drug 1\n  Name: {n1}\n  SMILES: {s1}\n\n"
        f"Drug 2\n  Name: {n2}\n  SMILES: {s2}\n\n"
        "##Answer:"
    )


def build_messages(n1, s1, n2, s2, system_text: str, merge_system: bool):
    user_text = build_user(n1, s1, n2, s2)
    if merge_system:
        return [{"role": "user", "content": f"{system_text}\n\n{user_text}"}]
    return [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]


# ── 데이터 로더 ────────────────────────────────────────────────────────────
def load_pairs(txt_path: Path) -> list[tuple[int, int, int]]:
    pairs = []
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 3:
                pairs.append((int(p[0]), int(p[1]), int(p[2])))
    return pairs


# ── vLLM 서버 관리 (gs_ddi_eval_port.py 패턴 재사용) ──────────────────────
def start_vllm_server(model_path: str, gpu: str, merge_system: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["VLLM_USE_V1"] = "0"  # 안정성 (reference: reference_vllm_v1_legacy)
    if merge_system:
        env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
        # Gemma-2 sliding_window=4096; allow override and cap max_model_len
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    max_len = "4096" if merge_system else "8192"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
        "--max-model-len", max_len,
        "--disable-log-requests",
    ]
    log_path = ZS_CACHE_DIR / f"vllm_server_{model_path.replace('/', '_')}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    return proc, log_file


def wait_server_ready(timeout=300) -> bool:
    import urllib.request
    url = f"http://127.0.0.1:{VLLM_PORT}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(5)
    return False


def kill_server(proc: subprocess.Popen, log_file):
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    log_file.close()
    time.sleep(3)


# ── 단일 샘플 inference (gs_ddi_eval_port.py classify() 패턴) ──────────────
def classify_one(client: OpenAI, model_path: str, msgs: list, max_retry: int = 2) -> int:
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model_path,
                messages=msgs,
                temperature=0.0,
                max_tokens=5,
                seed=123,
                extra_body={"guided_choice": GUIDED_CHOICES},
            )
            txt = resp.choices[0].message.content.strip()
            return int(txt)
        except (ValueError, TypeError):
            return -1
        except Exception:
            time.sleep(2 ** attempt)
    return -1


def classify_batch(client, model_path, batch_msgs, batch_labels):
    preds = []
    for msgs in batch_msgs:
        preds.append(classify_one(client, model_path, msgs))
    return preds, batch_labels


# ── 메트릭 계산 ────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
    if not valid:
        return dict(acc=0.0, kappa=0.0, macro_f1=0.0, loss=-1.0, parse_rate=0.0)
    yt, yp = zip(*valid)
    return dict(
        acc=float(accuracy_score(yt, yp)),
        kappa=float(cohen_kappa_score(yt, yp)),
        macro_f1=float(f1_score(yt, yp, average="macro", zero_division=0)),
        loss=-1.0,  # ZS는 loss 없음
        parse_rate=len(valid) / len(y_true),
    )


# ── summary.csv append ──────────────────────────────────────────────────────
SUMMARY_HEADER = [
    "best_epoch", "cell", "checkpoint_metric", "n_params", "seed",
    "split", "test_acc", "test_kappa", "test_loss", "test_macro_f1",
    "val_acc", "val_kappa", "val_loss", "val_macro_f1", "wall_time_sec",
]


def append_summary(row: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists() or SUMMARY_CSV.stat().st_size == 0
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELL_MODELS))
    ap.add_argument("--gpu",  required=True)
    ap.add_argument("--model", default=None, help="모델 경로 override")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    model_path  = args.model or CELL_MODELS[args.cell]
    n_params    = CELL_PARAMS[args.cell]
    merge_system = args.cell in MERGE_SYSTEM_CELLS
    ZS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # placeholder: 이미 완료된 경우 skip (run_zero_shot_orig.sh 패턴)
    done_flag = ZS_CACHE_DIR / f"cell{args.cell}_done.flag"
    if done_flag.exists():
        print(f"[skip] cell {args.cell} already completed ({done_flag})")
        return

    # 데이터 로드
    names_raw  = json.load(open(NAMES_JSON))["drug2name"]  # 1700 entries (10개 미포함)
    desc_raw   = json.load(open(DESC_JSON))                # 1710 entries (설명문)
    smiles_map = json.load(open(SMILES_JSON))              # 1710 entries
    label_map  = json.load(open(DDI_LABELS_JSON))          # 86 DDI types

    # biobert.py 동일 패턴: name 없으면 description fallback (1710 전체 커버)
    name_map = {int(k): v["name"] for k, v in names_raw.items()}

    def get_name(did: int) -> str:
        if did in name_map and name_map[did]:
            return name_map[did]
        return f"Drug DB{did}"  # description 없는 10개 약물: DrugBank ID로 대체

    system_text = build_system(label_map)

    # vLLM 서버 기동
    print(f"[cell {args.cell}] Starting vLLM server: {model_path} on GPU {args.gpu}")
    server_proc, server_log = start_vllm_server(model_path, args.gpu, merge_system)
    if not wait_server_ready(timeout=300):
        kill_server(server_proc, server_log)
        raise RuntimeError("vLLM server did not start within 300s")
    print(f"  server ready on port {VLLM_PORT}")

    client = OpenAI(base_url=f"http://127.0.0.1:{VLLM_PORT}/v1", api_key="dummy", timeout=60)

    t_total_start = time.time()

    try:
        for split_name, (valid_txt, test_txt) in SPLIT_FILES.items():
            t_split_start = time.time()  # split별 순수 추론 시간
            cache_pkl = ZS_CACHE_DIR / f"cell{args.cell}_{split_name}_preds.pkl"
            if cache_pkl.exists():
                print(f"  [{split_name}] cache found, loading...")
                with open(cache_pkl, "rb") as f:
                    y_true, y_pred = pickle.load(f)
            else:
                pairs = load_pairs(test_txt)
                print(f"  [{split_name}] {len(pairs)} test pairs")

                # 메시지 빌드
                all_msgs = []
                y_true = []
                for h, t, r in pairs:
                    n1 = get_name(h)
                    n2 = get_name(t)
                    s1 = smiles_map.get(str(h), "")
                    s2 = smiles_map.get(str(t), "")
                    all_msgs.append(build_messages(n1, s1, n2, s2, system_text, merge_system))
                    y_true.append(r)

                # 배치 병렬 inference (ThreadPoolExecutor, gs_ddi_eval_port.py 패턴)
                bs = args.batch_size
                batches = [
                    (all_msgs[i:i+bs], y_true[i:i+bs])
                    for i in range(0, len(all_msgs), bs)
                ]
                y_pred = [None] * len(all_msgs)

                with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                    futs = {
                        ex.submit(classify_batch, client, model_path, bm, bl): idx
                        for idx, (bm, bl) in enumerate(batches)
                    }
                    for f in tqdm(as_completed(futs), total=len(batches), desc=f"  [{split_name}]"):
                        idx = futs[f]
                        preds, _ = f.result()
                        start = idx * bs
                        y_pred[start:start + len(preds)] = preds

                # 캐시 저장
                with open(cache_pkl, "wb") as f:
                    pickle.dump((y_true, y_pred), f)

            # 메트릭 계산
            m = compute_metrics(y_true, y_pred)
            split_wall = time.time() - t_split_start  # 이 split 순수 추론 시간
            parse_fail = sum(1 for p in y_pred if p == -1)
            pairs_per_sec = len(y_pred) / max(split_wall, 1)
            print(f"  [{split_name}] acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} "
                  f"kappa={m['kappa']:.4f} parse_fail={parse_fail}/{len(y_pred)} "
                  f"wall={split_wall:.0f}s ({pairs_per_sec:.1f} pairs/s)")

            append_summary({
                "best_epoch":        0,
                "cell":              f"{int(args.cell):02d}",
                "checkpoint_metric": "zs",
                "n_params":          n_params,
                "seed":              0,
                "split":             split_name,
                "test_acc":          round(m["acc"],    6),
                "test_kappa":        round(m["kappa"],  6),
                "test_loss":         -1.0,
                "test_macro_f1":     round(m["macro_f1"], 6),
                "val_acc":           -1.0,
                "val_kappa":         -1.0,
                "val_loss":          -1.0,
                "val_macro_f1":      -1.0,
                "wall_time_sec":     round(split_wall, 1),
            })

    finally:
        kill_server(server_proc, server_log)

    # 완료 flag
    done_flag.touch()
    total_wall = time.time() - t_total_start
    print(f"[cell {args.cell}] Done. Total inference wall={total_wall:.0f}s (server startup excluded)")


if __name__ == "__main__":
    main()
