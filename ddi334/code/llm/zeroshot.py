#!/usr/bin/env python3
"""DDI-334 LLM cell inference+evaluation (V2 binary-with-R).

- Input prompt: instruction.DDI334Prompt (02 section 3.1)
- Only the queried side effect (vec=1) is asked about (same as earlier experiments)
- Evaluation: same per-type ROC-AUC/PR-AUC/Acc as train.py's evaluate
- vLLM OpenAI server (run in the vllm-llm env)

Usage (vllm-llm env):
  python zeroshot.py --model Qwen/Qwen2.5-3B-Instruct \
      --gpu 3 --split S0 --limit 16        # smoke
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
from openai import OpenAI
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS_DIR = os.path.join(HERE, 'data', 'splits')
DATASET_TAG = 'ddibn'   # constant kept only for result filename / summary.csv schema compatibility (not used to build paths)
sys.path.insert(0, os.path.join(HERE, 'code', 'build', 'document'))
from instruction import DDI334Prompt

PORT = 12341
CACHE = os.path.join(HERE, 'results', 'llm_cache')
os.makedirs(CACHE, exist_ok=True)


def start_server(model, gpu, merge_system, mem_util=0.85):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["VLLM_USE_V1"] = "0"
    if merge_system:
        env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"; env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--host", "127.0.0.1", "--port", str(PORT),
           "--gpu-memory-utilization", str(mem_util), "--enforce-eager",
           "--max-model-len", "4096" if merge_system else "8192", "--disable-log-requests"]
    log = open(os.path.join(CACHE, f"server_{model.replace('/','_')}.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=log), log


def wait_ready(timeout=600):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3); return True
        except Exception:
            time.sleep(5)
    return False


def load_rows(split, limit):
    rows = []
    for line in open(os.path.join(SPLITS_DIR, f'test_{split}.txt')):
        p = line.split()
        if len(p) != 4:
            continue
        vec = np.array([int(x) for x in p[2].split(',')], dtype=np.int64)
        rows.append((int(p[0]), int(p[1]), vec, int(p[3])))
        if limit and len(rows) >= limit:
            break
    return rows


def infer_v2(client, model, pb, rows, merge, mode='existing_drug', workers=24):
    """Concurrent requests (ThreadPoolExecutor) -> vLLM batches. Serial requests leave
    Running:1, so the GPU sits idle and it's extremely slow.
    One Yes/No query per (pair, t). temperature=0 is deterministic, so parallelizing
    doesn't change the result.
    mode: existing_drug / novel_drug (information-availability ablation)."""
    from concurrent.futures import ThreadPoolExecutor
    N = len(pb.typenames)
    pred = np.zeros((len(rows), N)); lab = np.zeros((len(rows), N + 1))
    tasks = []
    for r, (d1, d2, vec, pol) in enumerate(rows):
        lab[r, :N] = vec; lab[r, -1] = pol
        for t in np.where(vec == 1)[0].tolist():
            tasks.append((r, d1, d2, t))
    fails = [0]; done = [0]

    def one(task):
        r, d1, d2, t = task
        sys_t, usr = pb.build_v2(d1, d2, t, mode=mode)
        msgs = ([{"role": "user", "content": f"{sys_t}\n\n{usr}"}] if merge
                else [{"role": "system", "content": sys_t}, {"role": "user", "content": usr}])
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.0,
                                                      max_tokens=3, seed=123, logprobs=True, top_logprobs=20,
                                                      extra_body={"guided_choice": ["Yes", "No"]})
                ch = resp.choices[0]
                # Extract logprobs for both Yes and No -> normalize via softmax over {Yes,No} (analogous to the encoders' sigmoid).
                lp_yes = lp_no = None
                if ch.logprobs and ch.logprobs.content:
                    for tl in ch.logprobs.content[0].top_logprobs:
                        tok = tl.token.strip().lower()
                        if lp_yes is None and tok.startswith("yes"): lp_yes = tl.logprob
                        if lp_no is None and tok.startswith("no"):  lp_no = tl.logprob
                if lp_yes is not None and lp_no is not None:
                    mx = max(lp_yes, lp_no)
                    ey, en = np.exp(lp_yes - mx), np.exp(lp_no - mx)
                    p_yes = float(ey / (ey + en))           # P(Yes | Yes,No)
                elif lp_yes is not None:
                    p_yes = float(np.exp(lp_yes))           # No fell outside top-k -> Yes dominates
                elif lp_no is not None:
                    p_yes = float(1.0 - np.exp(lp_no))      # Yes fell outside top-k -> No dominates
                else:
                    p_yes = 1.0 if ch.message.content.strip().lower().startswith("yes") else 0.0
                pred[r, t] = p_yes
                done[0] += 1
                if done[0] % 2000 == 0:
                    print(f"  [v2] {done[0]}/{len(tasks)} queries done", flush=True)
                return
            except Exception as e:
                if attempt == 1:
                    fails[0] += 1
                    if fails[0] <= 5:
                        print(f"  [v2 row{r} t{t}] FAIL {e}", flush=True)
                else:
                    time.sleep(1)

    print(f"  [v2] starting {len(tasks)} queries total ({len(rows)} pairs) in parallel (workers={workers})", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, tasks))
    if fails[0]:
        print(f"  [v2] {fails[0]}/{len(tasks)} queries failed total", flush=True)
    return pred, lab


def evaluate(pred, lab):
    """Same per-type AUROC/PR-AUC/accuracy/macro-F1 + P@50/P@(N/2) as train.py's evaluate."""
    N = pred.shape[1]; roc, prc, acc, f1, p50, prn = [], [], [], [], [], []
    for j in range(N):
        where = np.where(lab[:, j] == 1)[0]
        if len(where) == 0:
            continue
        pc = pred[where, j]; lc = lab[where, j] * lab[where, -1]
        if 0 < lc.sum() < len(lc):
            phard = (pc > 0.5).astype(float)
            roc.append(roc_auc_score(lc, pc)); prc.append(average_precision_score(lc, pc))
            acc.append(accuracy_score(lc, phard))
            f1.append(f1_score(lc, phard, zero_division=0))
            lc_sorted = lc[np.argsort(-pc)]   # descending score -> fraction of positives among the top k (same as train.py)
            k50 = min(50, lc_sorted.shape[0]); krn = max(1, lc_sorted.shape[0] // 2)
            p50.append(float(lc_sorted[:k50].mean())); prn.append(float(lc_sorted[:krn].mean()))
    m = lambda x: float(np.mean(x)) if x else 0.0
    return {"AUC-ROC": m(roc), "PR-AUC": m(prc), "accuracy": m(acc), "macro-F1": m(f1),
            "P@50": m(p50), "P@N2": m(prn), "n_types": len(roc)}


SUMMARY_COLS = ["best_epoch", "cell", "checkpoint_metric", "dataset", "n_params", "seed",
                "split", "test_AUC-ROC", "test_PR-AUC", "test_accuracy",
                "val_AUC-ROC", "val_PR-AUC", "val_accuracy", "wall_time_sec"]


def save_results(ds, cell_label, split, metrics, wall, outdir=None):
    """Appends with the same columns as the encoders' summary.csv (val/epoch are left blank for LLMs). F1/n_types go in a separate json."""
    import csv
    rdir = outdir or os.path.join(HERE, 'results', ds)
    os.makedirs(os.path.join(rdir, 'llm_metrics'), exist_ok=True)
    summ = os.path.join(rdir, 'summary.csv')
    row = {c: "" for c in SUMMARY_COLS}
    row.update({"cell": cell_label, "checkpoint_metric": "AUC-ROC(LLM-zs)", "dataset": ds,
                "split": split, "wall_time_sec": round(wall, 1),
                "test_AUC-ROC": round(metrics["AUC-ROC"], 6),
                "test_PR-AUC": round(metrics["PR-AUC"], 6),
                "test_accuracy": round(metrics["accuracy"], 6)})
    wh = not os.path.exists(summ) or os.path.getsize(summ) == 0
    with open(summ, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if wh:
            w.writeheader()
        w.writerow(row)
    json.dump({**metrics, "cell": cell_label, "dataset": ds, "split": split, "wall_sec": wall},
              open(os.path.join(rdir, 'llm_metrics', f'{cell_label}_{split}.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--split", default="S0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="")    # summary.csv cell label (e.g. 15_zs_qwen)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)  # lower this on a shared GPU
    ap.add_argument("--port", type=int, default=12341)  # avoid collisions between parallel jobs: each job gets its own port
    ap.add_argument("--merge-system", action="store_true")  # for gemma-family models
    ap.add_argument("--mode", default="existing_drug", choices=["existing_drug", "novel_drug"])  # V2 information ablation
    ap.add_argument("--workers", type=int, default=24)  # number of concurrent requests (use 64+ for small models to saturate the GPU)
    ap.add_argument("--outdir", default="")  # override the results output folder (e.g. results/main_experiments/01-12_encoders)
    a = ap.parse_args()
    global PORT
    PORT = a.port

    pb = DDI334Prompt()
    rows = load_rows(a.split, a.limit)
    print(f"[{DATASET_TAG}/{a.split}] rows={len(rows)} model={a.model}")

    proc, log = start_server(a.model, a.gpu, a.merge_system, a.gpu_mem_util)
    try:
        if not wait_ready():
            print("server failed to start"); return
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        t0 = time.time()
        pred, lab = infer_v2(client, a.model, pb, rows, a.merge_system, mode=a.mode, workers=a.workers)
        metrics = evaluate(pred, lab)
        wall = time.time() - t0
        print(f"[RESULT] {metrics} | wall {wall:.0f}s")
        label = a.label or f"v2_{a.model.replace('/','_')}"
        save_results(DATASET_TAG, label, a.split, metrics, wall, outdir=a.outdir or None)
        # Save raw predictions (for distribution analysis; same location/format as the encoders' raw_preds). pred=p_yes[row,type], lab=multihot+polarity[row,N+1].
        rdir = a.outdir or os.path.join(HERE, 'results', DATASET_TAG)
        rpdir = os.path.join(rdir, 'raw_preds'); os.makedirs(rpdir, exist_ok=True)
        np.savez(os.path.join(rpdir, f'{label}_{a.split}.npz'), pred=pred, lab=lab)
        print(f"[SAVED] summary.csv + llm_metrics/{label}_{a.split}.json + raw_preds/{label}_{a.split}.npz")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        log.close()


if __name__ == '__main__':
    main()
