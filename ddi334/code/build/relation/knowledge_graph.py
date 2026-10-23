#!/usr/bin/env python3
"""
DDI-334 Knowledge graph (Cell 09 KGE whole-graph, Cell 10/11 subgraph) — Table 2's
"Knowledge graph" row covers all three, so all three are built here.

Confirmed design decisions:
  - Keep all of HetioNet (bio relations + the non-334-drug DrugBank DDI)
  - Our 334 drugs' interactions = DDI-334 TWOSIDES **train** edges only (untyped, rel 109)
  - DDI edges touching a cold drug (Dn) are automatically 0 (train.txt has only Dk x Dk) -> S1/S2 inductive

Part 1 (Cell 09 KGE embeddings, frozen): KGE training reuses the existing
`kge_hetionet/baseline.py` (PyKEEN, AstraZeneca HP). This script generates that
training's input (ddi_train), per-model configs, and the run script.

Inputs:
  - kge_hetionet/data/KG.txt              HetioNet bio (rel 86-108)  <- config's data.kg (shared, 80/10/10 split)
  - kge_hetionet/data/train.txt           DrugBank DDI (rel 0-85)    <- non-334 only, included in ddi_train
  - ddi334/data/splits/train.txt           DDI-334 TWOSIDES train     <- 334-drug interactions (rel 109)
  - ddi334/data/splits/id_maps.json       334 db_ids

Outputs (ddi334/data/relation/kge/):
  existing_drug/ddi_train.txt                      (non-334 DrugBank + 334 ts-train rel109; all are KGE train rows)
  existing_drug/configs/{transe,rotate,distmult,complex,transh}.yaml
  run_all.sh                                        (5 training runs: 5 models)

Part 2 (Cell 10/11 subgraph extraction): combines kg_bio_334.txt (Part 1's output)
with the 334 TWOSIDES train edges into data/relation/kg_triples.npy [E,3] (h,t,r).
train.py's SubgraphExtractor pulls each drug pair's surrounding k-hop subgraph from
this file. (node init=TransE comes from Part 1's kge/existing_drug/transe/.npy)

Relation scheme: see relation_map.json (bio 0-22, TWOSIDES = 23+type_idx).

Run: python3 build/relation/knowledge_graph.py
"""
import os
import json
import numpy as np

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.dirname(DDI334_DIR)
V5_DIR = os.path.dirname(EXP_DIR)
KGE_DIR = os.path.join(V5_DIR, 'kge_hetionet')
KG_TXT = os.path.join(KGE_DIR, 'data', 'KG.txt')              # bio (shared)
DRUGBANK_DDI = os.path.join(KGE_DIR, 'data', 'train.txt')     # DrugBank DDI (rel 0-85)

OUT_KGE = os.path.join(DDI334_DIR, 'data', 'relation', 'kge')
# Clean renumbering: remove DrugBank (0-85) -> bio becomes 0-22, TWOSIDES becomes 23+. No empty slots.
BIO_ORIG_BASE = 86   # original HetioNet bio relation starting id (86-108) -> shift by -86 to get 0-22
BIO_N = 23           # number of bio relations (86-108)
TS_REL_BASE = BIO_N  # TWOSIDES typed relations = 23 + type_idx
COND = 'existing_drug'   # this script only generates the existing_drug (full KG) condition; novel_drug is prep_novel_drug_kg.py

# 5 KGE models' AstraZeneca HP (as-is from kge_hetionet/config/baseline/hetionet/*.yaml)
MODELS = {
    'transe':   dict(name='TransE',   dim=304, lr=0.02,  epoch=500, neg=61),
    'rotate':   dict(name='RotatE',   dim=512, lr=0.03,  epoch=500, neg=41),
    'distmult': dict(name='DistMult', dim=80,  lr=0.02,  epoch=400, neg=41),
    'complex':  dict(name='ComplEx',  dim=272, lr=0.03,  epoch=700, neg=91),
    'transh':   dict(name='TransH',   dim=480, lr=0.005, epoch=800, neg=1),
}


def load_334():
    idm = json.load(open(os.path.join(DDI334_DIR, 'data', 'splits', 'id_maps.json')))
    return set(int(x) for x in idm['db_ids'])


def filter_bio_kg(db334):
    """From HetioNet bio (KG.txt), **remove edges touching a non-334-drug node**.
    Drug nodes = id 0-1709 (node2id). Removes the non-334 drugs (1,376 of them) from
    the KG, keeping the 334 drugs plus non-drug nodes/structure (genes, diseases,
    etc.). DrugBank DDI is not used."""
    non334_drugs = set(range(1710)) - db334   # non-334 drug ids
    kept = []
    for line in open(KG_TXT):
        p = line.split()
        if len(p) != 3:
            continue
        h, t, r = int(p[0]), int(p[1]), int(p[2])
        if h in non334_drugs or t in non334_drugs:
            continue
        assert BIO_ORIG_BASE <= r < BIO_ORIG_BASE + BIO_N, f"bio rel out of range: {r}"
        kept.append(f"{h} {t} {r - BIO_ORIG_BASE}")   # renumber 86-108 -> 0-22
    return kept


def ts_train_edges():
    """DDI-334 data/splits/train.txt positive pairs -> typed edges.
    For each active type, emits 'h t (109+type_idx)' (expanding the multi-hot vector).
    Per-type relations, same convention as DrugBank."""
    path = os.path.join(DDI334_DIR, 'data', 'splits', 'train.txt')
    edges = []
    rels = set()
    n_types = 0
    for line in open(path):
        p = line.split()
        if p[-1] != '1':       # positives only
            continue
        h, t = int(p[0]), int(p[1])
        vec = p[2].split(',')
        n_types = len(vec)
        for ti, v in enumerate(vec):
            if v == '1':
                r = TS_REL_BASE + ti       # 23 + type_idx
                edges.append(f"{h} {t} {r}")
                rels.add(r)
    return edges, len(rels), n_types


def write_config(mkey, m, n_relations):
    cfg_dir = os.path.join(OUT_KGE, COND, 'configs')
    os.makedirs(cfg_dir, exist_ok=True)
    ddi_train = os.path.join(OUT_KGE, COND, 'ddi_train.txt')
    bio_kg = os.path.join(OUT_KGE, 'kg_bio_334.txt')
    save_path = os.path.join(OUT_KGE, COND, mkey)
    txt = f"""type: baseline
# DDI-334 ({COND}) KGE: HetioNet bio edges (334 drug nodes only, rel 0-22) + 334-drug TWOSIDES train split (typed rel 23+). No DrugBank DDI edges.
# Relation renumbering: see relation_map.json. n_relations={n_relations} (23 bio + TWOSIDES type count)
# HP: AstraZeneca/kgem-in-drug-discovery (kge_hetionet config as-is)
dataset: emergnn
n_relations: {n_relations}

data:
  kg: {bio_kg}
  ddi_train: {ddi_train}

model:
  name: {m['name']}
  embedding_dim: {m['dim']}

optimizer:
  class: Adagrad
  lr: {m['lr']}

train:
  loss_function: MarginRankingLoss
  num_epoch: {m['epoch']}
  num_negative: {m['neg']}
  create_inverse: False

save:
  path: {save_path}

seed: 42
"""
    with open(os.path.join(cfg_dir, f'{mkey}.yaml'), 'w') as f:
        f.write(txt)


def main():
    db334 = load_334()
    os.makedirs(OUT_KGE, exist_ok=True)
    print(f"[1] Loaded 334 db_ids: {len(db334)}")

    # Shared bio KG: from HetioNet bio, keep only the 334 drug nodes (remove non-334-drug edges)
    bio = filter_bio_kg(db334)
    bio_path = os.path.join(OUT_KGE, 'kg_bio_334.txt')
    with open(bio_path, 'w') as f:
        f.write('\n'.join(bio) + '\n')
    print(f"[2] bio KG (334 drug nodes only): {len(bio):,} edges (removed non-334-drug edges from the original 1,690,693) -> {bio_path}")

    os.makedirs(os.path.join(OUT_KGE, COND), exist_ok=True)
    ts, n_ts_rel, n_types = ts_train_edges()   # typed TWOSIDES only (no DrugBank)
    ddi_train_path = os.path.join(OUT_KGE, COND, 'ddi_train.txt')
    with open(ddi_train_path, 'w') as f:
        f.write('\n'.join(ts) + '\n')
    n_relations = BIO_N + n_types   # bio 0-22 + TWOSIDES 23..23+n_types-1
    ts_drugs, ts_rels = set(), set()
    for e in ts:
        a, b, r = e.split(); ts_drugs.add(int(a)); ts_drugs.add(int(b)); ts_rels.add(int(r))
    print(f"[3] TWOSIDES typed edges {len(ts):,} | types seen {n_ts_rel}/{n_types} "
          f"(rel {min(ts_rels)}-{max(ts_rels)}) | drugs {len(ts_drugs)} | n_relations={n_relations} -> {ddi_train_path}")
    # Record the renumbering (to avoid mistakes later)
    rel_map = {
        'note': 'Relation renumbering (after removing DrugBank DDI 0-85). Both KGE and R-GCN use these ids.',
        'bio': {'orig_hetionet_rel': '86-108', 'new_rel': '0-22', 'count': BIO_N,
                'rule': 'new = orig - 86'},
        'twosides': {'new_rel': f'{TS_REL_BASE}-{TS_REL_BASE + n_types - 1}',
                     'rule': f'new = {TS_REL_BASE} + dense_type_idx (0..{n_types-1})',
                     'n_types': n_types},
        'drugbank_ddi': 'REMOVED (originally 0-85, no longer used)',
        'n_relations': n_relations,
    }
    with open(os.path.join(OUT_KGE, COND, 'relation_map.json'), 'w') as f:
        json.dump(rel_map, f, indent=2, ensure_ascii=False)
    for mkey, m in MODELS.items():
        write_config(mkey, m, n_relations)
    print(f"[4] Generated {len(MODELS)} configs")

    # run_all.sh
    run = ['#!/bin/bash', '# DDI-334 KGE training (5 models). Usage: bash run_all.sh [GPU]',
           'set -e', 'GPU=${1:-0}',
           f'BASE="{KGE_DIR}"', f'KGE="{OUT_KGE}"', '']
    for mkey in MODELS:
        cfg = os.path.join(OUT_KGE, COND, 'configs', f'{mkey}.yaml')
        res = os.path.join(OUT_KGE, COND, mkey)
        run.append(f'echo "=== {COND}/{mkey} ==="')
        run.append(f'CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "{cfg}"')
        run.append(f'CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "{res}"')
        run.append('')
    run_path = os.path.join(OUT_KGE, 'run_all.sh')
    with open(run_path, 'w') as f:
        f.write('\n'.join(run))
    os.chmod(run_path, 0o755)
    print(f"[5] Run script: {run_path}")

    # -- Cell 10/11: KG triples for subgraph extraction (bio + TWOSIDES train) --
    triples = []
    for line in open(bio_path):
        h, t, r = line.split()
        triples.append([int(h), int(t), int(r)])
    for line in open(ddi_train_path):
        h, t, r = line.split()
        triples.append([int(h), int(t), int(r)])
    kg = np.array(triples, dtype=np.int64)
    kg_triples_path = os.path.join(DDI334_DIR, 'data', 'relation', 'kg_triples.npy')
    np.save(kg_triples_path, kg)
    print(f"[6] kg_triples.npy {kg.shape} (bio+TWOSIDES, rel 0-{kg[:,2].max()}) -> {kg_triples_path}")

    print(f"\n[DONE] KGE inputs/configs + kg_triples.npy complete. KGE training: bash {run_path} [GPU]")


if __name__ == '__main__':
    main()
