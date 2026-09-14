#!/usr/bin/env python3
"""
DDI-334 precompute 정리: MOL/REL/DOC 전처리 파일을 각 {ds}/precompute/에 모음
====================================================================================
공유 v5 per-drug feature(chemberta/biobert/schnet, db_id 키, 데이터셋 무관)를
**334 약물로 필터링**해 ddibn/tdc 각 precompute에 넣어 자기완결적으로 만든다.

이미 데이터셋별로 있는 것: fp_pca.pt, bio_pca.pt (Dk fit), kg_triples.npy, ddi_whole.pt
이 스크립트가 채우는 것 (334 필터):
  MOL: drug_smiles.json(런타임 graph/CNN/FP용), schnet_coords.pt+schnet_drop.json(Cell02), chemberta.pt(Cell06)
  DOC: biobert_description.pt(Cell13), biobert_drugname.pt(Cell14)
  REL: transe_ent.npy (kge/{ds}/transe 학습 완료 시 복사; Cell08/09/10/11 node init)

Run: micromamba run -n DDIBench python assemble_precompute.py
"""
import os
import json
import shutil
import numpy as np
import torch

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.dirname(DDI334_DIR)
PC_V5 = os.path.join(EXP_DIR, 'data', 'precompute')
SMILES = os.path.join(EXP_DIR, 'data', 'drug_smiles.json')
KGE = os.path.join(DDI334_DIR, 'data', 'kge')
DATASETS = ['ddibn', 'tdc']


def db334():
    return set(int(x) for x in json.load(open(os.path.join(DDI334_DIR, 'meta', 'id_maps.json')))['db_ids'])


def filter_emb_pt(src_name, ids, out_dirs, has_embeddings=True):
    """payload['embeddings'] (or raw dict) {db_id: tensor} -> 334만 남겨 저장."""
    payload = torch.load(os.path.join(PC_V5, src_name), map_location='cpu', weights_only=False)
    if has_embeddings:
        emb = payload['embeddings']
        filt = {k: v for k, v in emb.items() if int(k) in ids}
        new = dict(payload); new['embeddings'] = filt; new['n_drugs'] = len(filt)
    else:   # raw dict {db_id: value} (schnet_coords)
        filt = {k: v for k, v in payload.items() if int(k) in ids}
        new = filt
    for od in out_dirs:
        torch.save(new, os.path.join(od, src_name))
    return len(filt)


def main():
    ids = db334()
    print(f"334 db_id 로드: {len(ids)}")
    out_dirs = [os.path.join(DDI334_DIR, 'data', ds, 'precompute') for ds in DATASETS]
    for od in out_dirs:
        os.makedirs(od, exist_ok=True)

    # drug_smiles.json (334)
    smi = {k: v for k, v in json.load(open(SMILES)).items() if int(k) in ids}
    for od in out_dirs:
        json.dump(smi, open(os.path.join(od, 'drug_smiles.json'), 'w'))
    print(f"  drug_smiles.json: {len(smi)} (MOL 런타임 graph/CNN/FP)")

    # MOL: chemberta, schnet
    print(f"  chemberta.pt: {filter_emb_pt('chemberta.pt', ids, out_dirs)} (MOL Cell06)")
    print(f"  schnet_coords.pt: {filter_emb_pt('schnet_coords.pt', ids, out_dirs, has_embeddings=False)} (MOL Cell02)")
    # schnet_drop 불필요: 334 약물 전부 3D 좌표 있음 (failed 0) -> 생성 안 함

    # DOC: biobert
    print(f"  biobert_description.pt: {filter_emb_pt('biobert_description.pt', ids, out_dirs)} (DOC Cell13)")
    print(f"  biobert_drugname.pt: {filter_emb_pt('biobert_drugname.pt', ids, out_dirs)} (DOC Cell14)")

    # REL: TransE node init (kge/{ds}/transe 완료 시 복사)
    for ds in DATASETS:
        tdir = os.path.join(KGE, ds, 'transe')
        npys = [f for f in os.listdir(tdir)] if os.path.isdir(tdir) else []
        npys = [f for f in npys if f.endswith('.npy') and 'ent_' in f]
        if npys:
            src = os.path.join(tdir, sorted(npys)[-1])
            dst = os.path.join(DDI334_DIR, 'data', ds, 'precompute', 'transe_ent.npy')
            shutil.copy(src, dst)
            print(f"  [{ds}] transe_ent.npy 복사 ({os.path.basename(src)})")
        else:
            print(f"  [{ds}] TransE 미완료 — transe_ent.npy 나중에 복사 필요")

    print("[DONE] MOL/REL/DOC precompute를 각 {ds}/precompute/에 정리")


if __name__ == '__main__':
    main()
