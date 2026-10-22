#!/usr/bin/env python3
"""
DDI-334 drug_smiles.json 334-필터: data/molecule/의 첫 단계
====================================================================================
공유 v5 drug_smiles.json(1,710드럭, 이 repo 밖의 raw 소스)을 DDI-334 334드럭으로
필터링해 data/molecule/drug_smiles.json을 만든다. chemberta.py/schnet_coords.py가
그 다음 단계로 이 파일을 읽어 각자의 임베딩/좌표를 처음부터 계산한다(별도 외부
precompute 풀에서 자르는 방식이 아님 — per-drug 계산이라 334개만 돌려도 동일).

Cell12(DOC)의 existing_drug/novel_drug tier 임베딩은 이 스크립트가 아니라
build/document/bert_template.py가 bio_profile_text.json에서 직접 생성한다
(이 스크립트가 다루는 '334 필터링'과 무관 — 처음부터 334드럭 전용으로 만들어짐).

TransE 등 KGE 임베딩은 kge/existing_drug/{type}/*.npy에서 train.py가 직접 읽으므로
(load_kge_embeddings의 fallback 경로) 별도 복사 단계가 필요 없다.

Run: micromamba run -n DDIBench python build/molecule/precompute.py
"""
import os
import json

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR = os.path.dirname(DDI334_DIR)
SMILES = os.path.join(EXP_DIR, 'data', 'drug_smiles.json')


def db334():
    return set(int(x) for x in json.load(open(os.path.join(DDI334_DIR, 'data', 'splits', 'id_maps.json')))['db_ids'])


def main():
    ids = db334()
    print(f"334 db_id 로드: {len(ids)}")
    mol_dir = os.path.join(DDI334_DIR, 'data', 'molecule')
    os.makedirs(mol_dir, exist_ok=True)

    smi = {k: v for k, v in json.load(open(SMILES)).items() if int(k) in ids}
    json.dump(smi, open(os.path.join(mol_dir, 'drug_smiles.json'), 'w'))
    print(f"  drug_smiles.json: {len(smi)} (MOL 런타임 graph/CNN/FP, chemberta.py/schnet_coords.py의 입력)")

    print("[DONE] data/molecule/drug_smiles.json 생성 -> chemberta.py, schnet_coords.py 실행 가능")


if __name__ == '__main__':
    main()
