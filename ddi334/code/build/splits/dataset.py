#!/usr/bin/env python3
"""
DDI-334 Dataset Builder -- source to final in one pass (reproducible)
================================================================
Deterministically generates the raw data -> final training dataset in a single
script. No intermediate outputs are left on disk, and every random draw is
seeded with seed=42.

[Input sources]
  - meeting/ddi334_tdc/*.csv                     TDC TWOSIDES raw (CID, 1308 types)
  - original_repo/.../twosides_cluster/*.txt     DDI-Bench TWOSIDES (209 types, filtered)
  - original_repo/.../drugbank_cluster/*.txt     DrugBank DDI (for determining db_only)
  - original_repo/.../initial/drugbank/id2smiles.json   cluster-split SMILES
  - kge_hetionet/data/twosides.csv               for determining TWOSIDES connectivity (type-agnostic)
  - data/smiles/twosides_drugbank_mapping.csv    cid<->drugbank<->int_id(db_id)

[Outputs]  (all under ddi334/data/splits/)
  - id_maps.json          cid<->db<->local mapping for the 334 drugs (single source of truth)
  - cluster_split.json    Dk(270)/Dn(64), gamma=0.25 seed=42 (reproducibility reference)
  - train/valid/test_{S0,S1,S2}.txt   final dataset (209 types, pos+neg 1:1 crossed, db space)
  Row format: "h t y0,...,y(N-1) p"   (p=1 positive / p=0 negative)

[Pipeline]  (entirely in memory, no intermediate directories)
  1. id_maps   : fix the cid/db/local mapping for the 334 drugs
  2. cluster   : Tanimoto (gamma=0.25) union-find -> Dk/Dn (seed=42)
  3. neither   : all pairs - TWOSIDES-connected - DrugBank-connected  (negative pool)
  4. per version : collect positives -> split into Dk/Dn (8:1:1/5:5/5:5) -> negatives 1:1 -> write output

Run: micromamba run -n DDIBench python build/splits/dataset.py
"""
import os
import csv
import json
import random
from itertools import combinations
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# -- Paths (this script lives inside ddi334/) --------------------------------
DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.dirname(DDI334_DIR)                 # .../experiment
V5_DIR = os.path.dirname(EXP_DIR)                     # .../v5_ddi-bench
LATEX_ROOT = os.path.dirname(os.path.dirname(V5_DIR)) # .../Latex
REPO = os.path.join(V5_DIR, 'original_repo', 'DDI_Ben', 'DDI_Ben')

MEETING_TDC = os.path.join(LATEX_ROOT, 'meeting', 'ddi334_tdc')
MEETING_DDIBN = os.path.join(LATEX_ROOT, 'meeting', 'ddi334_ddibn')
TWOSIDES_CLUSTER = os.path.join(REPO, 'data', 'twosides_cluster')
DRUGBANK_CLUSTER = os.path.join(REPO, 'data', 'drugbank_cluster')
ID2SMILES = os.path.join(REPO, 'data', 'initial', 'drugbank', 'id2smiles.json')
TWOSIDES_CSV = os.path.join(V5_DIR, 'kge_hetionet', 'data', 'twosides.csv')
MAPPING_CSV = os.path.join(LATEX_ROOT, 'data', 'smiles', 'twosides_drugbank_mapping.csv')

# -- Constants ----------------------------------------------------------------
SEED = 42
GAMMA = 0.25          # Tanimoto cutoff (per the 05 experiment-design doc)
TARGET_DK = 270       # confirmed in the 05 experiment-design doc
N_DRUGS = 334
TDC_SPLITS = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']
OUT_SPLITS = TDC_SPLITS


def canon(a, b):
    return (a, b) if a < b else (b, a)


# ═════════════════════════════════════════════════════════════════════════════
# 1. id_maps -- cid <-> db_id <-> local_id for the 334 drugs  (single source of truth)
# ═════════════════════════════════════════════════════════════════════════════
def build_id_maps():
    # 334 CIDs = drugs appearing across all meeting/ddi334_tdc splits (same definition as prep_tdc)
    all_cids = set()
    for s in TDC_SPLITS:
        with open(os.path.join(MEETING_TDC, f'{s}.csv')) as f:
            r = csv.DictReader(f)
            for row in r:
                all_cids.add(row['ID1']); all_cids.add(row['ID2'])
    sorted_cids = sorted(all_cids)
    assert len(sorted_cids) == N_DRUGS, f"CID {len(sorted_cids)} != {N_DRUGS}"

    # cid -> db_id (int_id) (mapping csv)
    cid2db = {}
    with open(MAPPING_CSV) as f:
        for row in csv.DictReader(f):
            cid2db[row['cid']] = int(row['int_id'])
    for c in sorted_cids:
        assert c in cid2db, f"CID missing from mapping: {c}"

    cid2local = {c: i for i, c in enumerate(sorted_cids)}
    local2db = {i: cid2db[c] for i, c in enumerate(sorted_cids)}
    db_ids = sorted(local2db.values())
    assert len(set(db_ids)) == N_DRUGS

    id_maps = {
        'note': '334 DDI-334 drugs. cid<->db_id(int_id)<->local_id(0-333). The final dataset uses the db_id space.',
        'cid_sorted': sorted_cids,
        'cid2db': cid2db if False else {c: cid2db[c] for c in sorted_cids},
        'cid2local': cid2local,
        'local2db': {str(k): v for k, v in local2db.items()},
        'db_ids': db_ids,
    }
    with open(os.path.join(DDI334_DIR, 'data', 'splits', 'id_maps.json'), 'w') as f:
        json.dump(id_maps, f, indent=0)
    print(f"[1] id_maps.json: 334 drugs (cid/db/local). db_id range {db_ids[0]}~{db_ids[-1]}")
    return set(db_ids), {c: cid2db[c] for c in sorted_cids}


# ═════════════════════════════════════════════════════════════════════════════
# 2. cluster split (Tanimoto gamma=0.25, union-find, seed=42) -> Dk/Dn
# ═════════════════════════════════════════════════════════════════════════════
def build_cluster_split(db_set):
    id2smiles = json.load(open(ID2SMILES))
    db_ids = sorted(db_set)
    fps = {}
    for d in db_ids:
        smi = id2smiles.get(str(d))
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            print(f"  [WARN] SMILES failed for db_id {d}")
            continue
        fps[d] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    fp_list = [(d, fps[d]) for d in db_ids if d in fps]
    edges = defaultdict(set)
    for i in range(len(fp_list)):
        u, ufp = fp_list[i]
        sims = DataStructs.BulkTanimotoSimilarity(ufp, [v for _, v in fp_list[i+1:]])
        for off, sim in enumerate(sims):
            if sim > GAMMA:
                v = fp_list[i + 1 + off][0]
                edges[u].add(v); edges[v].add(u)

    parent = {d: d for d in db_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py
    for u, nbrs in edges.items():
        for v in nbrs:
            union(u, v)
    groups = defaultdict(list)
    for d in db_ids:
        groups[find(d)].append(d)
    clusters = sorted([sorted(v) for v in groups.values()], key=lambda c: (-len(c), c[0]))

    def assign(clist):
        rng = np.random.RandomState(SEED)
        idx = list(range(len(clist))); rng.shuffle(idx)
        dk, dn = set(), set()
        for i in idx:
            c = clist[i]
            (dk if len(dk) + len(c) <= TARGET_DK else dn).update(c)
        return dk, dn
    dk, dn = assign(clusters)
    if len(dk) != TARGET_DK:
        dk2, dn2 = assign(sorted(clusters, key=lambda c: (len(c), c[0])))
        if abs(len(dk2) - TARGET_DK) < abs(len(dk) - TARGET_DK):
            dk, dn = dk2, dn2
    assert len(dk) + len(dn) == N_DRUGS
    with open(os.path.join(DDI334_DIR, 'data', 'splits', 'cluster_split.json'), 'w') as f:
        json.dump({'gamma': GAMMA, 'seed': SEED,
                   'Dk': sorted(dk), 'Dn': sorted(dn)}, f, indent=0)
    print(f"[2] cluster_split.json: Dk={len(dk)} / Dn={len(dn)} (gamma={GAMMA}, seed={SEED})")
    return dk, dn


# ═════════════════════════════════════════════════════════════════════════════
# 3. neither pool (db space, type-agnostic) = all pairs - TWOSIDES-connected - DrugBank-connected
# ═════════════════════════════════════════════════════════════════════════════
def build_neither(db_set, cid2db):
    # TWOSIDES connectivity (raw TDC, type-agnostic)
    ts_pairs = set()
    with open(TWOSIDES_CSV) as f:
        f.readline()
        for line in f:
            p = line.split(',', 2)
            if len(p) < 2:
                continue
            d1, d2 = cid2db.get(p[0]), cid2db.get(p[1])
            if d1 is not None and d2 is not None and d1 != d2:
                ts_pairs.add(canon(d1, d2))
    # DrugBank connectivity
    db_pairs = set()
    for s in TDC_SPLITS:
        path = os.path.join(DRUGBANK_CLUSTER, f'{s}.txt')
        if not os.path.exists(path):
            continue
        for line in open(path):
            parts = line.split()
            if len(parts) < 2:
                continue
            d1, d2 = int(parts[0]), int(parts[1])
            if d1 in db_set and d2 in db_set and d1 != d2:
                db_pairs.add(canon(d1, d2))
    all_pairs = set(canon(a, b) for a, b in combinations(sorted(db_set), 2))
    neither = all_pairs - ts_pairs - db_pairs
    db_only = db_pairs - ts_pairs
    print(f"[3] all {len(all_pairs)} | ts-positive {len(ts_pairs)} | "
          f"db_only {len(db_only)} | neither {len(neither)}")
    return neither


# ═════════════════════════════════════════════════════════════════════════════
# 4. collect positives (db space, multi-hot)
# ═════════════════════════════════════════════════════════════════════════════
def collect_ddibn_positives(db_set):
    """meeting/ddi334_ddibn -> {(db1,db2): 209-hot}.

    Raw meeting format: "db1 db2 209-hot 1" (db_id space, all positive).
    Just as tdc comes from meeting/ddi334_tdc, ddibn also treats the raw meeting
    data as its single source.
    """
    N = 209
    pair_vec = {}
    for s in TDC_SPLITS:
        path = os.path.join(MEETING_DDIBN, f'{s}.txt')
        if not os.path.exists(path):
            continue
        for line in open(path):
            parts = line.rstrip('\n').split(' ')
            if len(parts) < 3:
                continue
            d1, d2 = int(parts[0]), int(parts[1])
            if d1 not in db_set or d2 not in db_set or d1 == d2:
                continue
            vec = [int(x) for x in parts[2].split(',')]
            assert len(vec) == N
            if sum(vec) == 0:
                continue
            key = canon(d1, d2)
            pair_vec[key] = [a | b for a, b in zip(pair_vec[key], vec)] if key in pair_vec else vec
    return N, pair_vec


# ═════════════════════════════════════════════════════════════════════════════
# 5. split assignment + negatives 1:1 + output
# ═════════════════════════════════════════════════════════════════════════════
def split_pairs(pair_vec, dk, dn):
    dk_dk, dk_dn, dn_dn = [], [], []
    for (d1, d2), vec in pair_vec.items():
        a, b = d1 in dk, d2 in dk
        if a and b:
            dk_dk.append((d1, d2, vec))
        elif a or b:
            dk_dn.append((d1, d2, vec))
        else:
            dn_dn.append((d1, d2, vec))
    def sp811(items):
        r = np.random.RandomState(SEED); idx = list(range(len(items))); r.shuffle(idx)
        nv = len(items)//10; nt = len(items)//10; ntr = len(items)-nv-nt
        return ([items[i] for i in idx[:ntr]],
                [items[i] for i in idx[ntr:ntr+nv]],
                [items[i] for i in idx[ntr+nv:]])
    def sp55(items):
        r = np.random.RandomState(SEED); idx = list(range(len(items))); r.shuffle(idx)
        h = len(items)//2
        return [items[i] for i in idx[:h]], [items[i] for i in idx[h:]]
    tr, vs0, ts0 = sp811(dk_dk)
    vs1, ts1 = sp55(dk_dn)
    vs2, ts2 = sp55(dn_dn)
    return {'train': tr, 'valid_S0': vs0, 'test_S0': ts0,
            'valid_S1': vs1, 'test_S1': ts1, 'valid_S2': vs2, 'test_S2': ts2}


def add_negatives_and_write(name, splits, neither, dk, dn, n_types, type_map):
    out_dir = os.path.join(DDI334_DIR, 'data', 'splits')
    os.makedirs(out_dir, exist_ok=True)

    # neither bucket
    dk_dk = [p for p in neither if p[0] in dk and p[1] in dk]
    dk_dn = [p for p in neither if (p[0] in dk) != (p[1] in dk)]
    dn_dn = [p for p in neither if p[0] in dn and p[1] in dn]
    rng = random.Random(SEED)
    rng.shuffle(dk_dk); rng.shuffle(dk_dn); rng.shuffle(dn_dn)
    bucket_of = {'train': dk_dk, 'valid_S0': dk_dk, 'test_S0': dk_dk,
                 'valid_S1': dk_dn, 'test_S1': dk_dn,
                 'valid_S2': dn_dn, 'test_S2': dn_dn}

    print(f"  [{name}] buckets Dk×Dk={len(dk_dk)} Dk×Dn={len(dk_dn)} Dn×Dn={len(dn_dn)}")
    for s in OUT_SPLITS:
        rows = splits[s]
        bucket = bucket_of[s]
        n = len(rows)
        if n <= len(bucket):
            negs = [bucket.pop() for _ in range(n)]
        else:
            print(f"    [WARN] {name}/{s}: not enough neither pairs ({len(bucket)}<{n}) -> topping up with sampling-with-replacement")
            negs = bucket[:]; bucket.clear()
            while len(negs) < n:
                negs.append(negs[rng.randrange(len(negs))])
        with open(os.path.join(out_dir, f'{s}.txt'), 'w') as f:
            for (d1, d2, vec), (n1, n2) in zip(rows, negs):
                vs = ','.join(map(str, vec))
                f.write(f"{d1} {d2} {vs} 1\n")
                f.write(f"{n1} {n2} {vs} 0\n")
        print(f"    {s}.txt: {n} pos + {n} neg")

    # identity drug_map (db space) + type_map
    db_ids = sorted(dk | dn)
    with open(os.path.join(out_dir, 'drug_map.json'), 'w') as f:
        json.dump({str(d): d for d in db_ids}, f)
    if type_map is not None:
        with open(os.path.join(out_dir, 'type_map.json'), 'w') as f:
            json.dump(type_map, f)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump({'n_types': n_types, 'space': 'db_id', 'neg_ratio': '1:1',
                   'gamma': GAMMA, 'seed': SEED}, f)


def main():
    print("DDI-334 Dataset Builder (source -> final, reproducible)\n")
    db_set, cid2db = build_id_maps()
    dk, dn = build_cluster_split(db_set)
    neither = build_neither(db_set, cid2db)

    print("\n[4-5] ddibn (209 type)")
    N1, pos1 = collect_ddibn_positives(db_set)
    print(f"  positives: {len(pos1)} pairs")
    sp1 = split_pairs(pos1, dk, dn)
    add_negatives_and_write('ddibn', sp1, set(neither), dk, dn, N1, None)

    print(f"\n[DONE] ddi334/data/splits/ (N={N1}) + id_maps.json + cluster_split.json")


if __name__ == '__main__':
    main()
