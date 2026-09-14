#!/usr/bin/env python3
"""
DDI-334 Dataset Builder — 소스에서 최종까지 한 번에 (재현 가능)
================================================================
원본 데이터 -> 최종 학습 데이터셋을 단일 스크립트로 결정적으로 생성한다.
중간 산출물을 디스크에 남기지 않으며, 모든 무작위는 seed=42로 고정한다.

[입력 소스]
  - meeting/ddi334_tdc/*.csv                     TDC TWOSIDES 원본 (CID, 1308 type)
  - original_repo/.../twosides_cluster/*.txt     DDI-Bench TWOSIDES (209 type, 필터본)
  - original_repo/.../drugbank_cluster/*.txt     DrugBank DDI (db_only 판정용)
  - original_repo/.../initial/drugbank/id2smiles.json   cluster split SMILES
  - kge_hetionet/data/twosides.csv               TWOSIDES 연결(type-무관) 판정용
  - data/smiles/twosides_drugbank_mapping.csv    cid<->drugbank<->int_id(db_id)

[산출물]  (모두 ddi334/ 아래)
  - id_maps.json          334 약물의 cid<->db<->local 매핑 (단일 진실 소스)
  - cluster_split.json    Dk(270)/Dn(64), gamma=0.25 seed=42 (재현 기준)
  - ddibn/                최종 ddibn 데이터셋 (209 type, pos+neg 1:1 교차, db 공간)
  - tdc/                  최종 tdc 데이터셋 (1308 type, 동일)
  각 데이터셋 행 포맷: "h t y0,...,y(N-1) p"   (p=1 positive / p=0 negative)

[파이프라인]  (전부 메모리상, 중간 디렉토리 없음)
  1. id_maps   : 334 약물 cid/db/local 매핑 확정
  2. cluster   : Tanimoto(gamma=0.25) union-find -> Dk/Dn (seed=42)
  3. neither   : 전체쌍 - TWOSIDES연결 - DrugBank연결  (negative pool)
  4. per ver   : positive 수집 -> Dk/Dn split(8:1:1/5:5/5:5) -> negative 1:1 -> 출력

Run: micromamba run -n DDIBench python build_dataset.py
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

# ── 경로 (스크립트는 ddi334/ 안에 있음) ──────────────────────────────────────
DDI334_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# ── 상수 ─────────────────────────────────────────────────────────────────────
SEED = 42
GAMMA = 0.25          # Tanimoto cutoff (05 문서)
TARGET_DK = 270       # 05 문서 확정
N_DRUGS = 334
TDC_SPLITS = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']
OUT_SPLITS = TDC_SPLITS


def canon(a, b):
    return (a, b) if a < b else (b, a)


# ═════════════════════════════════════════════════════════════════════════════
# 1. id_maps — 334 약물의 cid <-> db_id <-> local_id  (단일 진실 소스)
# ═════════════════════════════════════════════════════════════════════════════
def build_id_maps():
    # 334 CID = meeting/ddi334_tdc 전 split에 등장하는 약물 (prep_tdc와 동일 정의)
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
        assert c in cid2db, f"mapping에 없는 CID: {c}"

    cid2local = {c: i for i, c in enumerate(sorted_cids)}
    local2db = {i: cid2db[c] for i, c in enumerate(sorted_cids)}
    db_ids = sorted(local2db.values())
    assert len(set(db_ids)) == N_DRUGS

    id_maps = {
        'note': '334 DDI-334 drugs. cid<->db_id(int_id)<->local_id(0-333). 최종 데이터셋은 db_id 공간 사용.',
        'cid_sorted': sorted_cids,
        'cid2db': cid2db if False else {c: cid2db[c] for c in sorted_cids},
        'cid2local': cid2local,
        'local2db': {str(k): v for k, v in local2db.items()},
        'db_ids': db_ids,
    }
    with open(os.path.join(DDI334_DIR, 'meta', 'id_maps.json'), 'w') as f:
        json.dump(id_maps, f, indent=0)
    print(f"[1] id_maps.json: 334 약물 (cid/db/local). db_id 범위 {db_ids[0]}~{db_ids[-1]}")
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
            print(f"  [WARN] SMILES 실패 db_id {d}")
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
    with open(os.path.join(DDI334_DIR, 'meta', 'cluster_split.json'), 'w') as f:
        json.dump({'gamma': GAMMA, 'seed': SEED,
                   'Dk': sorted(dk), 'Dn': sorted(dn)}, f, indent=0)
    print(f"[2] cluster_split.json: Dk={len(dk)} / Dn={len(dn)} (gamma={GAMMA}, seed={SEED})")
    return dk, dn


# ═════════════════════════════════════════════════════════════════════════════
# 3. neither pool (db 공간, type-무관) = 전체 - TWOSIDES연결 - DrugBank연결
# ═════════════════════════════════════════════════════════════════════════════
def build_neither(db_set, cid2db):
    # TWOSIDES 연결 (TDC 원본, type-무관)
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
    # DrugBank 연결
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
    print(f"[3] 전체 {len(all_pairs)} | ts-positive {len(ts_pairs)} | "
          f"db_only {len(db_only)} | neither {len(neither)}")
    return neither


# ═════════════════════════════════════════════════════════════════════════════
# 4. positive 수집 (db 공간, multi-hot)
# ═════════════════════════════════════════════════════════════════════════════
def collect_ddibn_positives(db_set):
    """meeting/ddi334_ddibn -> {(db1,db2): 209-hot}.

    meeting 원본 포맷: "db1 db2 209-hot 1" (db_id 공간, 전부 positive).
    tdc가 meeting/ddi334_tdc에서 오듯, ddibn도 meeting 원본을 단일 소스로.
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


def collect_tdc_positives(db_set, cid2db):
    """meeting/ddi334_tdc CSV -> {(db1,db2): 1308-hot} + type_map(raw Y->dense)."""
    # type_map: 334내 모든 Y 수집 -> dense
    all_y = set()
    pair_rawy = defaultdict(set)
    for s in TDC_SPLITS:
        with open(os.path.join(MEETING_TDC, f'{s}.csv')) as f:
            for row in csv.DictReader(f):
                d1, d2 = cid2db.get(row['ID1']), cid2db.get(row['ID2'])
                if d1 is None or d2 is None or d1 == d2:
                    continue
                y = int(row['Y'])
                all_y.add(y)
                pair_rawy[canon(d1, d2)].add(y)
    sorted_y = sorted(all_y)
    raw2dense = {y: i for i, y in enumerate(sorted_y)}
    N = len(sorted_y)
    pair_vec = {}
    for key, ys in pair_rawy.items():
        vec = [0] * N
        for y in ys:
            vec[raw2dense[y]] = 1
        pair_vec[key] = vec
    type_map = {str(y): raw2dense[y] for y in sorted_y}
    return N, pair_vec, type_map


# ═════════════════════════════════════════════════════════════════════════════
# 5. split 배정 + negative 1:1 + 출력
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
    out_dir = os.path.join(DDI334_DIR, 'data', name)
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
            print(f"    [WARN] {name}/{s}: neither 부족({len(bucket)}<{n}) -> 복원추출 보충")
            negs = bucket[:]; bucket.clear()
            while len(negs) < n:
                negs.append(negs[rng.randrange(len(negs))])
        with open(os.path.join(out_dir, f'{s}.txt'), 'w') as f:
            for (d1, d2, vec), (n1, n2) in zip(rows, negs):
                vs = ','.join(map(str, vec))
                f.write(f"{d1} {d2} {vs} 1\n")
                f.write(f"{n1} {n2} {vs} 0\n")
        print(f"    {s}.txt: {n} pos + {n} neg")

    # identity drug_map (db 공간) + type_map
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

    print("\n[4-5] tdc (1308 type)")
    N2, pos2, tmap2 = collect_tdc_positives(db_set, cid2db)
    print(f"  positives: {len(pos2)} pairs / types: {N2}")
    sp2 = split_pairs(pos2, dk, dn)
    add_negatives_and_write('tdc', sp2, set(neither), dk, dn, N2, tmap2)

    print(f"\n[DONE] ddi334/ddibn (N={N1}), ddi334/tdc (N={N2}) + id_maps.json + cluster_split.json")


if __name__ == '__main__':
    main()
