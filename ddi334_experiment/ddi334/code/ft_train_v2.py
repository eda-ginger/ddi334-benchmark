#!/usr/bin/env python3
"""DDI-334 V2 (binary Yes/No) QLoRA fine-tune. ft_train_v1의 V2판.

- 학습 데이터: data/prompt/{mode}/train.jsonl (이미 렌더됨; 각 줄 {d1,d2,type,y,user})
  target = " Yes" if y==1 else " No"  (V2 binary-with-R)
- mode: real / ideal (Case-3 / Case-1)
- QLoRA 설정 ft_train_v1과 동일 (4bit nf4, LoRA r16/a32 q/k/v/o, AdamW 2e-4, 5ep, seed42)
- epoch마다 어댑터 저장 + (resume용) optimizer/scheduler/rng 저장
- 평가: run_ft_cv_v2.py (vLLM LoRA serving, split별 best epoch)

Usage (vllm-llm 또는 peft env):
  python ft_train_v2.py --model microsoft/Phi-3.5-mini-instruct --mode ideal --gpu 0 \
      [--epochs 5 --seed 42 --resume]
"""
import argparse, json, os, sys, time, random
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from llm_prompts import V2_SYSTEM

MERGE_SYSTEM_MODELS = ("gemma",)


def is_merge(model):
    return any(k in model.lower() for k in MERGE_SYSTEM_MODELS)


def load_examples(mode, limit=0):
    """렌더된 train.jsonl -> [(user, target)]. target=' Yes'/' No'."""
    path = os.path.join(HERE, "data", "prompt", mode, "train.jsonl")
    items = []
    for line in open(path):
        o = json.loads(line)
        items.append((o["user"], " Yes" if int(o["y"]) == 1 else " No"))
        if limit and len(items) >= limit:
            break
    return items


class V2SFTDataset(torch.utils.data.Dataset):
    def __init__(self, items, merge):
        self.items = items
        self.merge = merge

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(tok, merge, max_len=1024):
    def collate(batch):
        ii, ll, aa = [], [], []
        for user, target in batch:
            msgs = ([{"role": "user", "content": f"{V2_SYSTEM}\n\n{user}"}] if merge
                    else [{"role": "system", "content": V2_SYSTEM}, {"role": "user", "content": user}])
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            full = prompt + target + tok.eos_token
            pids = tok(prompt, add_special_tokens=False)["input_ids"][:max_len]
            fids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
            lab = [-100] * len(fids)
            for j in range(len(pids), len(fids)):
                lab[j] = fids[j]
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
    ap.add_argument("--mode", required=True, choices=["real", "ideal", "ideal_genes"])
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_acc", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--ckpt_root", default="ft_v2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    merge = is_merge(a.model)
    tag = f"{a.model.replace('/','_')}_{a.mode}_seed{a.seed}"
    ckpt = os.path.join(HERE, 'results', a.ckpt_root, tag)
    os.makedirs(ckpt, exist_ok=True)
    print(f"[V2-FT] model={a.model} mode={a.mode} seed={a.seed} merge={merge} ep={a.epochs} -> {ckpt}", flush=True)

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

    items = load_examples(a.mode, a.limit)
    ds = V2SFTDataset(items, merge)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                                         collate_fn=make_collator(tok, merge, a.max_len),
                                         num_workers=4, pin_memory=True)
    print(f"  train examples={len(ds)} (mode={a.mode})", flush=True)

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    opt = AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(loader) * a.epochs
    warm = min(100, total // 10)
    sched = SequentialLR(opt, [LinearLR(opt, 0.1, 1.0, warm),
                               LinearLR(opt, 1.0, 0.1, total - warm)], [warm])

    start_ep = 1
    state_path = os.path.join(ckpt, "trainer_state.pt")
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu")
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"]); start_ep = st["epoch"] + 1
        # 마지막 epoch 어댑터 로드
        from peft import PeftModel
        last = os.path.join(ckpt, f"epoch_{st['epoch']}")
        if os.path.exists(last):
            model.load_adapter(last, adapter_name="default", is_trainable=True)
        print(f"  [resume] epoch {st['epoch']} 이후부터 (start={start_ep})", flush=True)

    t0 = time.time()
    for ep in range(start_ep, a.epochs + 1):
        model.train(); opt.zero_grad(); tot = 0.0
        for step, b in enumerate(loader):
            b = {k: v.cuda() for k, v in b.items()}
            loss = model(**b).loss / a.grad_acc
            loss.backward(); tot += loss.item() * a.grad_acc
            if (step + 1) % a.grad_acc == 0:
                opt.step(); sched.step(); opt.zero_grad()
        ep_dir = os.path.join(ckpt, f"epoch_{ep}")
        model.save_pretrained(ep_dir); tok.save_pretrained(ep_dir)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": torch.get_rng_state(), "epoch": ep}, state_path)
        print(f"  epoch {ep}/{a.epochs} loss={tot/len(loader):.4f} -> {ep_dir} | {time.time()-t0:.0f}s", flush=True)
    print(f"[DONE] {a.epochs}ep 어댑터 저장 -> {ckpt}/epoch_1..{a.epochs} | wall {time.time()-t0:.0f}s", flush=True)
    print(f"  평가: run_ft_cv_v2.py --tag {tag} --mode {a.mode}", flush=True)


if __name__ == '__main__':
    main()
