#!/usr/bin/env python3
"""
DDI-334 KGE용 KG 빌더 (option b) — 데이터셋별(ddibn/tdc) 따로
==============================================================
사용자 확정 설계:
  - HetioNet 전체 유지 (bio 관계 + 비-334 약물의 DrugBank DDI)
  - 우리 334 약물의 상호작용 = DDI-334 TWOSIDES **train** 엣지만 (untyped, rel 109)
  - cold 약물(Dn)에 닿는 DDI 엣지는 자동 0 (train.txt엔 Dk×Dk만) → S1/S2 inductive

KGE 학습은 기존 `kge_hetionet/baseline.py`(PyKEEN, AstraZeneca HP) 재활용.
이 스크립트는 그 학습에 들어갈 입력(ddi_train)과 모델별 config, run 스크립트를 생성한다.

입력:
  - kge_hetionet/data/KG.txt              HetioNet bio (rel 86-108)  ← config의 data.kg (공유, 80/10/10 split)
  - kge_hetionet/data/train.txt           DrugBank DDI (rel 0-85)    ← 비-334만 추려 ddi_train에 포함
  - ddi334/{ddibn,tdc}/train.txt          DDI-334 TWOSIDES train     ← 334 상호작용(rel 109)
  - ddi334/id_maps.json                   334 db_id

출력 (ddi334/kge/):
  ddibn/ddi_train.txt , tdc/ddi_train.txt          (비334 DrugBank + 334 ts-train rel109; 전부 KGE train행)
  {ddibn,tdc}/configs/{transe,rotate,distmult,complex,transh}.yaml
  run_all.sh                                        (10 학습: 5 model × 2 dataset)

Run: python3 build_kg.py
"""
import os
import json

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.dirname(DDI334_DIR)
V5_DIR = os.path.dirname(EXP_DIR)
KGE_DIR = os.path.join(V5_DIR, 'kge_hetionet')
KG_TXT = os.path.join(KGE_DIR, 'data', 'KG.txt')              # bio (공유)
DRUGBANK_DDI = os.path.join(KGE_DIR, 'data', 'train.txt')     # DrugBank DDI (rel 0-85)

OUT_KGE = os.path.join(DDI334_DIR, 'data', 'kge')
# 재번호(깔끔): DrugBank(0-85) 제거 -> bio를 0-22로, TWOSIDES를 23+로. 빈 슬롯 없음.
BIO_ORIG_BASE = 86   # 원본 HetioNet bio 관계 시작 id (86-108) -> -86 시프트해 0-22
BIO_N = 23           # bio 관계 수 (86-108)
TS_REL_BASE = BIO_N  # TWOSIDES typed 관계 = 23 + type_idx
DATASETS = ['ddibn', 'tdc']

# 5종 KGE AstraZeneca HP (kge_hetionet/config/baseline/hetionet/*.yaml 그대로)
MODELS = {
    'transe':   dict(name='TransE',   dim=304, lr=0.02,  epoch=500, neg=61),
    'rotate':   dict(name='RotatE',   dim=512, lr=0.03,  epoch=500, neg=41),
    'distmult': dict(name='DistMult', dim=80,  lr=0.02,  epoch=400, neg=41),
    'complex':  dict(name='ComplEx',  dim=272, lr=0.03,  epoch=700, neg=91),
    'transh':   dict(name='TransH',   dim=480, lr=0.005, epoch=800, neg=1),
}


def load_334():
    idm = json.load(open(os.path.join(DDI334_DIR, 'meta', 'id_maps.json')))
    return set(int(x) for x in idm['db_ids'])


def filter_bio_kg(db334):
    """HetioNet bio(KG.txt)에서 **비-334 약물 노드에 닿는 엣지 제거**.
    약물 노드 = id 0-1709 (node2id). 비-334 약물(1376개)을 KG에서 빼고,
    334 약물 + 유전자/질병 등 비-약물 노드·구조는 유지. DrugBank DDI는 안 씀."""
    non334_drugs = set(range(1710)) - db334   # 비-334 약물 id
    kept = []
    for line in open(KG_TXT):
        p = line.split()
        if len(p) != 3:
            continue
        h, t, r = int(p[0]), int(p[1]), int(p[2])
        if h in non334_drugs or t in non334_drugs:
            continue
        assert BIO_ORIG_BASE <= r < BIO_ORIG_BASE + BIO_N, f"bio rel 범위 벗어남: {r}"
        kept.append(f"{h} {t} {r - BIO_ORIG_BASE}")   # 86-108 -> 0-22 재번호
    return kept


def ts_train_edges(dataset):
    """DDI-334 {dataset}/train.txt positive 쌍 -> typed edges.
    각 활성 type마다 'h t (109+type_idx)' (multi-hot 펼침). DrugBank처럼 type별 관계."""
    path = os.path.join(DDI334_DIR, 'data', dataset, 'train.txt')
    edges = []
    rels = set()
    n_types = 0
    for line in open(path):
        p = line.split()
        if p[-1] != '1':       # positive만
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


def write_config(dataset, mkey, m, n_relations):
    cfg_dir = os.path.join(OUT_KGE, dataset, 'configs')
    os.makedirs(cfg_dir, exist_ok=True)
    ddi_train = os.path.join(OUT_KGE, dataset, 'ddi_train.txt')
    bio_kg = os.path.join(OUT_KGE, 'kg_bio_334.txt')
    save_path = os.path.join(OUT_KGE, dataset, mkey)
    txt = f"""type: baseline
# DDI-334 ({dataset}) KGE: HetioNet bio(334 약물 노드만, rel 0-22) + 334 TWOSIDES train(typed rel 23+). DrugBank DDI 없음.
# 관계 재번호: relation_map.json 참조. n_relations={n_relations} (bio 23 + TWOSIDES type 수)
# HP: AstraZeneca/kgem-in-drug-discovery (kge_hetionet/config 그대로)
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
    print(f"[1] 334 db_id 로드: {len(db334)}")

    # 공유 bio KG: HetioNet bio에서 334 약물 노드만 유지 (비-334 약물 엣지 제거)
    bio = filter_bio_kg(db334)
    bio_path = os.path.join(OUT_KGE, 'kg_bio_334.txt')
    with open(bio_path, 'w') as f:
        f.write('\n'.join(bio) + '\n')
    print(f"[2] bio KG (334 약물 노드만): {len(bio):,} 엣지 (원본 1,690,693에서 비-334 약물 엣지 제거) -> {bio_path}")

    for ds in DATASETS:
        os.makedirs(os.path.join(OUT_KGE, ds), exist_ok=True)
        ts, n_ts_rel, n_types = ts_train_edges(ds)   # typed TWOSIDES only (DrugBank 없음)
        ddi_train_path = os.path.join(OUT_KGE, ds, 'ddi_train.txt')
        with open(ddi_train_path, 'w') as f:
            f.write('\n'.join(ts) + '\n')
        n_relations = BIO_N + n_types   # bio 0-22 + TWOSIDES 23..23+n_types-1
        ts_drugs, ts_rels = set(), set()
        for e in ts:
            a, b, r = e.split(); ts_drugs.add(int(a)); ts_drugs.add(int(b)); ts_rels.add(int(r))
        print(f"[3:{ds}] TWOSIDES typed 엣지 {len(ts):,} | 등장 type {n_ts_rel}/{n_types}종 "
              f"(rel {min(ts_rels)}-{max(ts_rels)}) | 약물 {len(ts_drugs)} | n_relations={n_relations} -> {ddi_train_path}")
        # 재번호 기록 (다음에 실수 방지)
        rel_map = {
            'note': '관계 재번호 (DrugBank DDI 0-85 제거 후). KGE/R-GCN 모두 이 id 사용.',
            'bio': {'orig_hetionet_rel': '86-108', 'new_rel': '0-22', 'count': BIO_N,
                    'rule': 'new = orig - 86'},
            'twosides': {'new_rel': f'{TS_REL_BASE}-{TS_REL_BASE + n_types - 1}',
                         'rule': f'new = {TS_REL_BASE} + dense_type_idx (0..{n_types-1})',
                         'n_types': n_types},
            'drugbank_ddi': 'REMOVED (원래 0-85, 더 이상 사용 안 함)',
            'n_relations': n_relations,
        }
        with open(os.path.join(OUT_KGE, ds, 'relation_map.json'), 'w') as f:
            json.dump(rel_map, f, indent=2, ensure_ascii=False)
        for mkey, m in MODELS.items():
            write_config(ds, mkey, m, n_relations)
    print(f"[4] configs 생성: {len(DATASETS)}×{len(MODELS)} = {len(DATASETS)*len(MODELS)}개")

    # run_all.sh
    run = ['#!/bin/bash', '# DDI-334 KGE 학습 (5 model × 2 dataset). Usage: bash run_all.sh [GPU]',
           'set -e', 'GPU=${1:-0}',
           f'BASE="{KGE_DIR}"', f'KGE="{OUT_KGE}"', '']
    for ds in DATASETS:
        for mkey in MODELS:
            cfg = os.path.join(OUT_KGE, ds, 'configs', f'{mkey}.yaml')
            res = os.path.join(OUT_KGE, ds, mkey)
            run.append(f'echo "=== {ds}/{mkey} ==="')
            run.append(f'CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "{cfg}"')
            run.append(f'CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "{res}"')
            run.append('')
    run_path = os.path.join(OUT_KGE, 'run_all.sh')
    with open(run_path, 'w') as f:
        f.write('\n'.join(run))
    os.chmod(run_path, 0o755)
    print(f"[5] run 스크립트: {run_path}")
    print(f"\n[DONE] KGE 입력·config 생성 완료. 학습: bash {run_path} [GPU]")


if __name__ == '__main__':
    main()
