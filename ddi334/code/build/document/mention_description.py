#!/usr/bin/env python3
"""DDI-334 Mention/Description (Cell 12, SciBERT frozen) — Table 2's
"Mention†/Description†" row, built as one two-step pipeline.

Step 1 (raw facts): for each drug (db_id), extract the 3 leakage/noise-free
HetioNet neighbor types by name:
  - Target genes : Compound-binds-Gene (CbG, kg_bio_334 rel 6) = direct binding target (LLMDDI convention)
  - Indications  : Compound-treats/palliates-Disease (CtD/CpD, rel 5/10)
  - Pharmacologic class : Compound-(class)-PharmacologicClass
  (Side Effect is excluded since it shares the same vocabulary as the prediction label -> leakage.
   up/down-regulate (indirect gene) relations are excluded too.)
  -> data/document/bio_profile_text.json
     { "<db_id>": {"genes":[..], "indications":[..], "pharm_class":[..]} }
  (also the raw-fact source for llm/instruction.py's "Existing drug" template)

Step 2 (embedding): build the drug-name (Mention) + Step-1 facts
(Description) template text via llm/instruction.py's DDI334Prompt.drug_text()
(single source, shared with the LLM prompts), then take a frozen BERT
backbone's [CLS] embedding of it. Backbone selectable via --tag
(scibert = main-text Table 2/D.1 setting; biobert/pubmedbert = Appendix C.4
comparison only). hidden_dim read from each model's own config (no
hardcoding -- dimension differs per model).
  novel_drug    : SMILES only
  existing_drug : + Name + Target genes + Indications + Pharmacologic class
  -> data/document/{tag}_{novel_drug,existing_drug}.pt
     {'embeddings': {int db_id: Tensor[dim]}, 'hidden_dim':dim, 'model_name':..., 'input_type':...}
     (train/train.py's load_biobert_embeddings reads from here)

Run: micromamba run -n DDIBench python build/document/mention_description.py --tag scibert [--device cuda:0]
     micromamba run -n DDIBench python build/document/mention_description.py --tag biobert
     micromamba run -n DDIBench python build/document/mention_description.py --tag pubmedbert
"""
import argparse
import json
import os
import time
from collections import defaultdict

import torch

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.dirname(DDI334_DIR)
V5_DIR = os.path.dirname(EXP_DIR)
DOC_DIR = os.path.join(DDI334_DIR, 'data', 'document')
BIO_PROFILE = os.path.join(DOC_DIR, 'bio_profile_text.json')
MAX_TOKENS = 512
TIERS = ['novel_drug', 'existing_drug']
DRUG_MAX = 1710
GENE_REL = '6'          # CbG (binds) = direct target

MODEL_PRESETS = {
    'biobert': 'dmis-lab/biobert-base-cased-v1.2',
    'pubmedbert': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    'scibert': 'allenai/scibert_scivocab_uncased',
}


def build_bio_profile_text():
    """Step 1: HetioNet neighbor names -> data/document/bio_profile_text.json."""
    e2i = json.load(open(os.path.join(V5_DIR, 'kge_hetionet', 'data', 'entity_drug.json')))
    i2e = {v: k for k, v in e2i.items()}
    nm = {int(k): v for k, v in json.load(open(os.path.join(DDI334_DIR, 'data', 'relation', 'hetionet_names.json'))).items()}

    genes = defaultdict(list); dis = defaultdict(list); cls = defaultdict(list)
    for line in open(os.path.join(DDI334_DIR, 'data', 'relation', 'kge', 'kg_bio_334.txt')):
        a, b, r = line.split(); a, b = int(a), int(b)
        for d, e in ((a, b), (b, a)):
            if d < DRUG_MAX and e >= DRUG_MAX and e in nm:
                t = i2e[e].split('::')[0]
                if t == 'Gene' and r == GENE_REL:
                    genes[d].append(nm[e])
                elif t == 'Disease':
                    dis[d].append(nm[e])
                elif t == 'Pharmacologic Class':
                    cls[d].append(nm[e])

    drug_map = json.load(open(os.path.join(DDI334_DIR, 'data', 'splits', 'drug_map.json')))
    dbids = sorted(set(drug_map.values()))
    out = {}
    for d in dbids:
        out[str(d)] = {"genes": sorted(genes[d]), "indications": sorted(dis[d]), "pharm_class": sorted(cls[d])}

    os.makedirs(DOC_DIR, exist_ok=True)
    json.dump(out, open(BIO_PROFILE, 'w'), ensure_ascii=False, indent=0)
    ng = sum(1 for d in dbids if genes[d]); nd = sum(1 for d in dbids if dis[d]); nc = sum(1 for d in dbids if cls[d])
    print(f"[1/2] bio_profile_text.json: {len(dbids)} drugs -> {BIO_PROFILE}")
    print(f"      have genes {ng}, indications {nd}, pharm_class {nc}")


def build_bert_template(args):
    """Step 2: drug_text() template -> frozen [CLS] embedding."""
    model_name = args.model_name or MODEL_PRESETS[args.tag]

    from instruction import DDI334Prompt   # same folder (build/document/)

    pb = DDI334Prompt()   # typenames is not used by drug_text
    drug_ids = sorted(int(k) for k in json.load(open(BIO_PROFILE)).keys())
    print(f'[2/2] Target drugs: {len(drug_ids)}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Loading {model_name} on {device}...')
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    hidden_dim = model.config.hidden_size   # the actual per-model dimension (never hardcoded)
    print(f'  hidden_dim={hidden_dim}')

    for tier in TIERS:
        texts = [pb.drug_text(d, tier) for d in drug_ids]
        embeddings = {}
        t0 = time.time()
        n = len(drug_ids)
        with torch.no_grad():
            for i in range(0, n, args.batch_size):
                batch_ids = drug_ids[i:i + args.batch_size]
                batch_texts = texts[i:i + args.batch_size]
                tok = tokenizer(batch_texts, padding=True, truncation=True,
                                 max_length=MAX_TOKENS, return_tensors='pt')
                tok = {k: v.to(device) for k, v in tok.items()}
                out = model(**tok)
                cls = out.last_hidden_state[:, 0, :].detach().clone().cpu().float()
                for did, emb in zip(batch_ids, cls):
                    embeddings[int(did)] = emb
            print(f'  [{tier}] {n}/{n} wall={time.time()-t0:.0f}s', flush=True)

        payload = {
            'embeddings': embeddings,
            'hidden_dim': hidden_dim,
            'model_name': model_name,
            'input_type': tier,
            'fallback_count': 0,
        }
        os.makedirs(DOC_DIR, exist_ok=True)
        out_pt = os.path.join(DOC_DIR, f'{args.tag}_{tier}.pt')
        torch.save(payload, out_pt)
        print(f'Saved: {out_pt} ({len(embeddings)} drugs, dim={hidden_dim})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='scibert', choices=list(MODEL_PRESETS.keys()),
                     help='output filename prefix ({tag}_{tier}.pt), also selects the preset model')
    ap.add_argument('--model_name', default=None, help='optional: specify an arbitrary HF model instead of a preset')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=16)
    args = ap.parse_args()

    build_bio_profile_text()
    build_bert_template(args)


if __name__ == '__main__':
    main()
