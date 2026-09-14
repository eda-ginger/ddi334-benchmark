#!/usr/bin/env python3
"""DDI-334 V1 (multi-label JSON) QLoRA fine-tune.

기존 ../ft_train.py(multi-class FT)는 보존. 이 파일은 TWOSIDES V1용:
  - 프롬프트: llm_prompts.DDI334Prompt.build_v1 (name+SMILES + queried 후보)
  - SFT 타깃: JSON {id: true/false}  (양성쌍 -> queried 전부 true / 음성쌍 -> 전부 false)
  - QLoRA 설정은 ft_train.py와 동일 (4bit nf4, LoRA r16/a32 q,k,v,o, AdamW 2e-4, 3ep, seed42)
학습 후 어댑터 저장 -> 평가는 `llm_infer.py --model <ckpt> --version v1` 로.

Usage (vllm-llm 또는 peft 가능 env):
  python ft_train_v1.py --model Qwen/Qwen2.5-3B-Instruct --dataset ddibn --gpu 3 \
      [--epochs 3 --seed 42 --limit 0]
"""
import argparse, json, os, sys, time, random
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from llm_prompts import DDI334Prompt, V1_SYSTEM

MERGE_SYSTEM_MODELS = ("gemma",)   # gemma류 system role 미지원


def is_merge(model):
    return any(k in model.lower() for k in MERGE_SYSTEM_MODELS)


def load_rows(ds, split, limit=0):
    rows = []
    for line in open(os.path.join(HERE, 'data', ds, f'{split}.txt')):
        p = line.split()
        if len(p) != 4:
            continue
        vec = [i for i, x in enumerate(p[2].split(',')) if x == '1']
        rows.append((int(p[0]), int(p[1]), vec, int(p[3])))
        if limit and len(rows) >= limit:
            break
    return rows


class V1SFTDataset(torch.utils.data.Dataset):
    """(messages, label_json_str). 양성->queried 전부 true / 음성->전부 false."""
    def __init__(self, rows, pb, merge):
        self.items = []
        for d1, d2, qids, pol in rows:
            if not qids:
                continue
            sys_t, usr, _ = pb.build_v1(d1, d2, qids)
            msgs = ([{"role": "user", "content": f"{sys_t}\n\n{usr}"}] if merge
                    else [{"role": "system", "content": sys_t}, {"role": "user", "content": usr}])
            label = json.dumps({str(t): (pol == 1) for t in qids})
            self.items.append((msgs, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(tok, max_len=768):
    def collate(batch):
        ii, ll, aa = [], [], []
        for msgs, label in batch:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            full = prompt + label + tok.eos_token
            pids = tok(prompt, add_special_tokens=False)["input_ids"]
            fids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
            pids = pids[:max_len]
            lab = [-100] * len(fids)
            for i in range(len(pids), len(fids)):
                lab[i] = fids[i]
            ii.append(fids); ll.append(lab); aa.append([1] * len(fids))
        m = max(len(x) for x in ii)
        pad = tok.pad_token_id or tok.eos_token_id
        f = lambda s, v: s + [v] * (m - len(s))
        return {"input_ids": torch.tensor([f(x, pad) for x in ii]),
                "attention_mask": torch.tensor([f(x, 0) for x in aa]),
                "labels": torch.tensor([f(x, -100) for x in ll])}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ddibn")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_acc", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--ckpt_root", default="ft_v1")  # 출력 어댑터 루트 (재학습은 별도 폴더로 기존 보존)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    merge = is_merge(a.model)
    ckpt = os.path.join(HERE, 'results', a.ckpt_root,
                        f"{a.dataset}_{a.model.replace('/','_')}_seed{a.seed}")
    os.makedirs(ckpt, exist_ok=True)
    print(f"[V1-FT] model={a.model} ds={a.dataset} seed={a.seed} merge_system={merge} -> {ckpt}")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    pb = DDI334Prompt(a.dataset)
    rows = load_rows(a.dataset, 'train', a.limit)
    ds = V1SFTDataset(rows, pb, merge)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                                         collate_fn=make_collator(tok, a.max_len),
                                         num_workers=4, pin_memory=True)
    print(f"  train samples={len(ds)}")

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    opt = AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(loader) * a.epochs
    warm = min(100, total // 10)
    sched = SequentialLR(opt, [LinearLR(opt, 0.1, 1.0, warm),
                               LinearLR(opt, 1.0, 0.1, total - warm)], [warm])

    t0 = time.time()
    # epoch마다 LoRA 어댑터 저장 -> 평가단계에서 split별 best epoch 선택 (인코더 per-split 방식과 일관).
    # merge 안 함: vLLM LoRA serving(--enable-lora)으로 어댑터 직접 평가.
    for ep in range(1, a.epochs + 1):
        model.train(); opt.zero_grad(); tot = 0.0
        for step, b in enumerate(loader):
            b = {k: v.cuda() for k, v in b.items()}
            loss = model(**b).loss / a.grad_acc
            loss.backward(); tot += loss.item() * a.grad_acc
            if (step + 1) % a.grad_acc == 0:
                opt.step(); sched.step(); opt.zero_grad()
        ep_dir = os.path.join(ckpt, f"epoch_{ep}")
        model.save_pretrained(ep_dir); tok.save_pretrained(ep_dir)
        print(f"  epoch {ep}/{a.epochs} loss={tot/len(loader):.4f} -> {ep_dir}", flush=True)
    print(f"[DONE] {a.epochs} epoch 어댑터 저장 -> {ckpt}/epoch_1..{a.epochs} | wall {time.time()-t0:.0f}s")
    print(f"  평가: run_ft_cv.py (vLLM LoRA serving, split별 best epoch 선택)")


if __name__ == '__main__':
    main()
