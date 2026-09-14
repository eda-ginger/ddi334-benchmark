#!/usr/bin/env python3
"""
DDI-334 REL 그래프 사전 제작 (Cell 09/10 KG subgraph, Cell 11 DDI whole graph)
=============================================================================
재번호 KG(bio 0-22, TWOSIDES 23+)에 맞춰 09/10/11이 쓸 그래프를 데이터셋별
precompute에 만들어 둔다. (node init=TransE는 별도 kge/{ds}/transe/.npy)

출력 (ddi334/{ddibn,tdc}/precompute/):
  kg_triples.npy   [E,3] (h, t, r)  — Cell 09/10 subgraph 추출용 KG (v5 load_hetionet_triples 순서)
                   = bio(kg_bio_334, rel 0-22) + TWOSIDES train(ddi_train, rel 23+)
  ddi_whole.pt     Cell 11 (Decagon) 용 전체 DDI 그래프
                   edge_index[2,E], edge_type[E] (TWOSIDES type_idx 0..N-1 + inverse),
                   num_types, num_relations(=2*num_types)

관계 체계: relation_map.json 참조 (bio 0-22, TWOSIDES = 23+type_idx).
Cell 11은 DDI-only라 type_idx(0..N-1)로 재매핑(rel-23) + inverse.

Run: micromamba run -n DDIBench python build_rel_graphs.py
"""
import os
import json
import numpy as np
import torch

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KGE = os.path.join(DDI334_DIR, 'data', 'kge')
KG_BIO = os.path.join(KGE, 'kg_bio_334.txt')
DATASETS = ['ddibn', 'tdc']
BIO_N = 23   # TWOSIDES rel = BIO_N + type_idx


def main():
    for ds in DATASETS:
        out = os.path.join(DDI334_DIR, 'data', ds, 'precompute')
        os.makedirs(out, exist_ok=True)
        n_types = json.load(open(os.path.join(KGE, ds, 'relation_map.json')))['twosides']['n_types']

        # ── Cell 09/10: subgraph 추출용 KG triples (bio + TWOSIDES train) ──
        triples = []
        for line in open(KG_BIO):                       # bio, rel 0-22
            h, t, r = line.split()
            triples.append([int(h), int(t), int(r)])    # (h, t, r) — v5 순서
        ddi_path = os.path.join(KGE, ds, 'ddi_train.txt')
        for line in open(ddi_path):                     # TWOSIDES typed, rel 23+
            h, t, r = line.split()
            triples.append([int(h), int(t), int(r)])
        kg = np.array(triples, dtype=np.int64)
        np.save(os.path.join(out, 'kg_triples.npy'), kg)

        # ── Cell 11: DDI whole graph (TWOSIDES train, type_idx + inverse) ──
        src, dst, etype = [], [], []
        for line in open(ddi_path):
            h, t, r = line.split()
            h, t, tyi = int(h), int(t), int(r) - BIO_N   # rel 23+ -> type_idx 0..N-1
            src.append(h); dst.append(t); etype.append(tyi)
            src.append(t); dst.append(h); etype.append(tyi + n_types)   # inverse 관계
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(etype, dtype=torch.long)
        torch.save({'edge_index': edge_index, 'edge_type': edge_type,
                    'num_types': n_types, 'num_relations': 2 * n_types,
                    'note': 'Cell 11 Decagon DDI graph. node=db_id. rel=type_idx(0..N-1)+inverse(N..2N-1)'},
                   os.path.join(out, 'ddi_whole.pt'))

        print(f"[{ds}] kg_triples.npy {kg.shape} (bio+TWOSIDES, rel 0-{kg[:,2].max()}) | "
              f"ddi_whole.pt edges={edge_index.shape[1]} num_relations={2*n_types} -> {out}")

    print("[DONE] 09/10 KG triples + 11 DDI whole graph 생성 (데이터셋별 precompute)")


if __name__ == '__main__':
    main()
