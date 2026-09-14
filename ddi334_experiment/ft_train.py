"""V5 DDI-Bench — LLM Fine-Tuning (Cells 15-17 FT).

QLoRA SFT (4-bit + LoRA) for 86-class DDI prediction.
Follows LLMDDI fine-tuning pattern (peft + bitsandbytes).
Simplified system prompt (no 86-class listing) since FT model learns the mapping.

Usage:
    python experiment/ft_train.py --cell 15 --gpu 0
    python experiment/ft_train.py --cell 15 --gpu 0 --epochs 3 --seed 42

After training:
    python experiment/ft_eval.py --cell 15 --gpu 0

Cells:
    15  Phi-3.5-mini-instruct   (2.7B)
    16  Qwen2.5-3B-Instruct     (3B)
    17  gemma-2-2b-it            (2B)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import Dataset
from tqdm import tqdm

# ── 경로 상수 ──────────────────────────────────────────────────────────────
EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR       = EXPERIMENT_DIR / "data"
RESULTS_DIR    = EXPERIMENT_DIR / "results_drugbank"
SUMMARY_CSV    = RESULTS_DIR / "summary.csv"
FT_DIR         = RESULTS_DIR / "ft_checkpoints"

LATEX_DIR      = Path("/home/rudwls2717/Latex")
NAMES_JSON     = LATEX_DIR / "data/embeddings/drugbank_names.json"
SMILES_JSON    = LATEX_DIR / "data/smiles/drugbank_id2smiles.json"
SPLIT_FILES    = {
    "S0": (DATA_DIR / "valid_S0.txt", DATA_DIR / "test_S0.txt"),
    "S1": (DATA_DIR / "valid_S1.txt", DATA_DIR / "test_S1.txt"),
    "S2": (DATA_DIR / "valid_S2.txt", DATA_DIR / "test_S2.txt"),
}
TRAIN_TXT      = DATA_DIR / "train.txt"

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
MERGE_SYSTEM_CELLS = {"17"}  # Gemma2: no system role


# ── 프롬프트 빌더 (FT용: 클래스 목록 없이 간결하게) ──────────────────────
SYSTEM_PROMPT = (
    "You are a pharmacology expert. Given two drugs (name and SMILES), "
    "predict their drug-drug interaction type as a single integer (0-85). "
    "Output only the integer."
)


def build_user(n1: str, s1: str, n2: str, s2: str) -> str:
    return (
        f"Drug 1\n  Name: {n1}\n  SMILES: {s1}\n\n"
        f"Drug 2\n  Name: {n2}\n  SMILES: {s2}\n\n"
        "##Answer:"
    )


# ── 데이터 로더 ────────────────────────────────────────────────────────────
def load_pairs(txt_path: Path) -> list[tuple[int, int, int]]:
    pairs = []
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 3:
                pairs.append((int(p[0]), int(p[1]), int(p[2])))
    return pairs


def load_resources():
    names_raw  = json.load(open(NAMES_JSON))["drug2name"]
    smiles_map = json.load(open(SMILES_JSON))
    name_map = {int(k): v["name"] for k, v in names_raw.items()}

    def get_name(did: int) -> str:
        if did in name_map and name_map[did]:
            return name_map[did]
        return f"Drug DB{did}"

    return get_name, smiles_map


# ── SFT Dataset ────────────────────────────────────────────────────────────
class DDISFTDataset(Dataset):
    """Each sample: (prompt_text, label_str). Tokenized in collator."""
    def __init__(self, pairs, get_name, smiles_map, merge_system: bool):
        self.items = []
        for h, t, r in pairs:
            n1 = get_name(h)
            n2 = get_name(t)
            s1 = smiles_map.get(str(h), "")
            s2 = smiles_map.get(str(t), "")
            user_text = build_user(n1, s1, n2, s2)
            if merge_system:
                prompt = f"{SYSTEM_PROMPT}\n\n{user_text}"
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_text},
                ]
            self.items.append((messages, str(r)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_collator(tokenizer, max_len: int = 512):
    """Tokenize and create labels with -100 masking on prompt tokens."""
    def collate(batch):
        all_input_ids = []
        all_labels    = []
        all_attn      = []

        for messages, label_str in batch:
            # Prompt part: apply chat template up to assistant turn start
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Full text = prompt + answer + eos
            answer_text = label_str
            full_text   = prompt_text + answer_text + tokenizer.eos_token

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]

            if len(full_ids) > max_len:
                full_ids   = full_ids[:max_len]
                prompt_ids = prompt_ids[:max_len]

            labels = [-100] * len(full_ids)
            # answer starts at len(prompt_ids), unmask it
            for i in range(len(prompt_ids), len(full_ids)):
                labels[i] = full_ids[i]

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_attn.append([1] * len(full_ids))

        # Pad to max length in batch
        max_batch_len = max(len(x) for x in all_input_ids)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        def pad_seq(seq, pad_val, target):
            return seq + [pad_val] * (target - len(seq))

        input_ids = torch.tensor([pad_seq(x, pad_id, max_batch_len) for x in all_input_ids])
        labels    = torch.tensor([pad_seq(x, -100,   max_batch_len) for x in all_labels])
        attn_mask = torch.tensor([pad_seq(x, 0,      max_batch_len) for x in all_attn])
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

    return collate


# ── 메트릭 ─────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
    if not valid:
        return dict(acc=0.0, kappa=0.0, macro_f1=0.0, parse_rate=0.0)
    yt, yp = zip(*valid)
    return dict(
        acc=float(accuracy_score(yt, yp)),
        kappa=float(cohen_kappa_score(yt, yp)),
        macro_f1=float(f1_score(yt, yp, average="macro", zero_division=0)),
        parse_rate=len(valid) / len(y_true),
    )


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


# ── Eval loop (greedy generate) ────────────────────────────────────────────
@torch.no_grad()
def eval_split(model, tokenizer, pairs, get_name, smiles_map,
               merge_system: bool, batch_size: int = 16, max_new_tokens: int = 4):
    model.eval()
    y_true, y_pred = [], []

    for i in tqdm(range(0, len(pairs), batch_size), desc="  eval", leave=False):
        batch = pairs[i:i + batch_size]
        prompts = []
        for h, t, r in batch:
            n1, n2 = get_name(h), get_name(t)
            s1, s2 = smiles_map.get(str(h), ""), smiles_map.get(str(t), "")
            user_text = build_user(n1, s1, n2, s2)
            if merge_system:
                messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_text}"}]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_text},
                ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            y_true.append(r)

        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for j, seq in enumerate(out):
            gen = seq[enc["input_ids"].shape[1]:]
            txt = tokenizer.decode(gen, skip_special_tokens=True).strip()
            try:
                pred = int(txt.split()[0])
                if 0 <= pred <= 85:
                    y_pred.append(pred)
                else:
                    y_pred.append(-1)
            except (ValueError, IndexError):
                y_pred.append(-1)

    return y_true, y_pred


# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell",    required=True, choices=list(CELL_MODELS))
    ap.add_argument("--gpu",     required=True)
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--epochs",  type=int,   default=3)
    ap.add_argument("--lr",      type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_acc",   type=int, default=8)
    ap.add_argument("--lora_r",     type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_len",    type=int, default=512)
    ap.add_argument("--eval_batch", type=int, default=16)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_path   = CELL_MODELS[args.cell]
    n_params     = CELL_PARAMS[args.cell]
    merge_system = args.cell in MERGE_SYSTEM_CELLS
    ckpt_dir     = FT_DIR / f"cell{int(args.cell):02d}_seed{args.seed}"
    FT_DIR.mkdir(parents=True, exist_ok=True)

    done_flag = FT_DIR / f"cell{int(args.cell):02d}_seed{args.seed}_done.flag"
    if done_flag.exists():
        print(f"[skip] cell {args.cell} seed {args.seed} FT already done")
        return

    print(f"[cell {args.cell} FT] model={model_path}  seed={args.seed}  epochs={args.epochs}")

    # ── 모델·토크나이저 로드 ─────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("  loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("  loading model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    # ── LoRA 설정 ────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 데이터 로드 ──────────────────────────────────────────────────────
    get_name, smiles_map = load_resources()
    train_pairs = load_pairs(TRAIN_TXT)
    val_pairs   = {k: load_pairs(v[0]) for k, v in SPLIT_FILES.items()}
    test_pairs  = {k: load_pairs(v[1]) for k, v in SPLIT_FILES.items()}

    print(f"  train={len(train_pairs)}, val S0={len(val_pairs['S0'])}")

    train_ds   = DDISFTDataset(train_pairs, get_name, smiles_map, merge_system)
    collator   = make_collator(tokenizer, max_len=args.max_len)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=4, pin_memory=True,
    )

    # ── Optimizer ────────────────────────────────────────────────────────
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = min(100, total_steps // 10)

    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    decay_sched  = LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                            total_iters=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, decay_sched],
                             milestones=[warmup_steps])

    # ── 학습 루프 ────────────────────────────────────────────────────────
    best_val_f1  = -1.0
    best_epoch   = 0
    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"  epoch {epoch}/{args.epochs}")):
            batch = {k: v.cuda() for k, v in batch.items()}
            out   = model(**batch)
            loss  = out.loss / args.grad_acc
            loss.backward()
            total_loss += out.loss.item()

            if (step + 1) % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"  epoch {epoch}: loss={avg_loss:.4f}")

        # Val eval on S0
        print(f"  evaluating val S0 (epoch {epoch})...")
        yt, yp = eval_split(model, tokenizer, val_pairs["S0"], get_name, smiles_map,
                            merge_system, batch_size=args.eval_batch)
        val_m = compute_metrics(yt, yp)
        print(f"  val S0: acc={val_m['acc']:.4f}  macro_f1={val_m['macro_f1']:.4f}  "
              f"parse={val_m['parse_rate']:.3f}")

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_epoch  = epoch
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"  *** best checkpoint saved (epoch {epoch} val_f1={best_val_f1:.4f}) ***")

    train_wall = time.time() - t_train_start
    print(f"  training done. best_epoch={best_epoch}  val_f1={best_val_f1:.4f}  wall={train_wall:.0f}s")

    # ── 테스트 평가 (best checkpoint) ───────────────────────────────────
    print("  loading best checkpoint for test eval...")
    from peft import PeftModel
    del model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = False
    ft_model = PeftModel.from_pretrained(base_model, str(ckpt_dir))
    ft_model.eval()

    cm = f"ft_ep{best_epoch}"

    for split_name in ["S0", "S1", "S2"]:
        t_split = time.time()

        # Val metrics
        yt_val, yp_val = eval_split(ft_model, tokenizer, val_pairs[split_name], get_name,
                                    smiles_map, merge_system, batch_size=args.eval_batch)
        val_m = compute_metrics(yt_val, yp_val)

        # Test metrics
        yt_tst, yp_tst = eval_split(ft_model, tokenizer, test_pairs[split_name], get_name,
                                    smiles_map, merge_system, batch_size=args.eval_batch)
        tst_m = compute_metrics(yt_tst, yp_tst)
        split_wall = time.time() - t_split

        print(f"  [{split_name}] test acc={tst_m['acc']:.4f} f1={tst_m['macro_f1']:.4f} "
              f"kappa={tst_m['kappa']:.4f}  val acc={val_m['acc']:.4f} f1={val_m['macro_f1']:.4f}")

        append_summary({
            "best_epoch":        best_epoch,
            "cell":              f"{int(args.cell):02d}",
            "checkpoint_metric": cm,
            "n_params":          n_params,
            "seed":              args.seed,
            "split":             split_name,
            "test_acc":          round(tst_m["acc"],      6),
            "test_kappa":        round(tst_m["kappa"],    6),
            "test_loss":         -1.0,
            "test_macro_f1":     round(tst_m["macro_f1"], 6),
            "val_acc":           round(val_m["acc"],      6),
            "val_kappa":         round(val_m["kappa"],    6),
            "val_loss":          -1.0,
            "val_macro_f1":      round(val_m["macro_f1"], 6),
            "wall_time_sec":     round(split_wall, 1),
        })

    done_flag.touch()
    print(f"[cell {args.cell} FT] seed {args.seed} complete. Total wall={time.time()-t_train_start:.0f}s")


if __name__ == "__main__":
    main()
