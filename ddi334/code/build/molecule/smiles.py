#!/usr/bin/env python3
"""DDI-334 SMILES representation (Cell 01 1D-CNN, Cell 02 ChemBERTa) — Table 2's
"SMILES" row covers both, so both are built here.

Step 1: filter the shared v5 drug_smiles.json (1,710 drugs, outside this repo)
        down to the DDI-334 334-drug cohort -> data/molecule/drug_smiles.json
        (Cell 01 reads this directly; also the input to Step 2 and to
        fingerprint.py / mol_graph.py)
Step 2: ChemBERTa (DeepChem/ChemBERTa-77M-MLM, hidden=384) [CLS] frozen
        embedding of each drug's SMILES -> data/molecule/chemberta.pt
        {'embeddings': {int db_id: Tensor[384]}, 'hidden_dim': 384, 'model_name': ...}

Step 2 is ported from the shared v5 framework's precompute/scripts/chemberta.py
(which computed this for the full 1,710-drug population before a separate
filtering step cut it down to 334); this version computes directly for the
334-drug SMILES file instead, since per-drug embeddings don't depend on the
rest of the population -- same numbers, fully self-contained in this repo.

Run: micromamba run -n DDIBench python build/molecule/smiles.py [--device cuda:0] [--batch_size 64]
"""
import argparse
import json
import os
import time

import torch

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.dirname(DDI334_DIR)
SMILES_V5 = os.path.join(EXP_DIR, 'data', 'drug_smiles.json')
MOL_DIR = os.path.join(DDI334_DIR, 'data', 'molecule')
SMILES_JSON = os.path.join(MOL_DIR, 'drug_smiles.json')
OUT_PT = os.path.join(MOL_DIR, 'chemberta.pt')
MODEL_NAME = 'DeepChem/ChemBERTa-77M-MLM'


def db334():
    return set(int(x) for x in json.load(open(os.path.join(DDI334_DIR, 'data', 'splits', 'id_maps.json')))['db_ids'])


def filter_smiles():
    ids = db334()
    print(f"Loaded 334 db_ids: {len(ids)}")
    os.makedirs(MOL_DIR, exist_ok=True)
    smi = {k: v for k, v in json.load(open(SMILES_V5)).items() if int(k) in ids}
    json.dump(smi, open(SMILES_JSON, 'w'))
    print(f"  drug_smiles.json: {len(smi)}")
    return smi


def build_chemberta(args):
    from transformers import AutoTokenizer, AutoModel

    smiles_map = {int(k): v for k, v in json.load(open(SMILES_JSON)).items()}
    drug_ids = sorted(smiles_map.keys())
    print(f'Target drugs: {len(drug_ids)}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Loading {MODEL_NAME} on {device}...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    hidden_dim = model.config.hidden_size
    print(f'  hidden_dim={hidden_dim}')

    embeddings = {}
    t0 = time.time()
    n = len(drug_ids)

    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            batch_ids = drug_ids[i:i + args.batch_size]
            batch_smi = [smiles_map.get(did, 'C') or 'C' for did in batch_ids]  # fallback: methane

            tok = tokenizer(batch_smi, padding=True, truncation=True,
                            max_length=256, return_tensors='pt')
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(**tok)
            cls = out.last_hidden_state[:, 0, :].detach().clone().cpu().float()

            for did, emb in zip(batch_ids, cls):
                embeddings[int(did)] = emb

            if (i // args.batch_size) % 10 == 0 or i + args.batch_size >= n:
                print(f'  [{i + len(batch_ids)}/{n}]  wall={time.time()-t0:.0f}s', flush=True)

    torch.save({
        'embeddings': embeddings,
        'hidden_dim': hidden_dim,
        'model_name': MODEL_NAME,
    }, OUT_PT)
    print(f'Saved: {OUT_PT}  ({len(embeddings)} drugs, dim={hidden_dim}, wall={time.time()-t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    filter_smiles()
    build_chemberta(args)


if __name__ == '__main__':
    main()
