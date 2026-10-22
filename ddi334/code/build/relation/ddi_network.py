#!/usr/bin/env python3
"""
DDI-334 DDI network whole graph (Cell 08, Decagon R-GCN)
=============================================================================
TWOSIDES train-split 엣지로 Cell 08(DDI network R-GCN)이 쓸 전체 DDI 그래프를
data/relation/ddi_network.pt로 만든다. node=db_id, 외부 생물학적 관계는 포함
하지 않음 (DDI network는 DDI 연결 자체만 사용 — knowledge_graph.py의 지식그래프와는
별개). node init(Morgan FP 기본 / TransE)은 train.py의 NODE_INIT env가 선택.

출력: ddi334/data/relation/ddi_network.pt
  edge_index[2,E], edge_type[E] (TWOSIDES type_idx 0..N-1 + inverse),
  num_types, num_relations(=2*num_types)

관계 체계: relation_map.json 참조 (bio 0-22, TWOSIDES = 23+type_idx).
Cell 08은 DDI-only라 type_idx(0..N-1)로 재매핑(rel-23) + inverse.

Run: micromamba run -n DDIBench python build/relation/ddi_network.py
"""
import os
import json
import torch

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REL_DIR = os.path.join(DDI334_DIR, 'data', 'relation')
KGE = os.path.join(REL_DIR, 'kge')
BIO_N = 23   # TWOSIDES rel = BIO_N + type_idx


def main():
    n_types = json.load(open(os.path.join(KGE, 'existing_drug', 'relation_map.json')))['twosides']['n_types']
    ddi_path = os.path.join(KGE, 'existing_drug', 'ddi_train.txt')

    src, dst, etype = [], [], []
    for line in open(ddi_path):
        h, t, r = line.split()
        h, t, tyi = int(h), int(t), int(r) - BIO_N   # rel 23+ -> type_idx 0..N-1
        src.append(h); dst.append(t); etype.append(tyi)
        src.append(t); dst.append(h); etype.append(tyi + n_types)   # inverse 관계
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)

    out_path = os.path.join(REL_DIR, 'ddi_network.pt')
    torch.save({'edge_index': edge_index, 'edge_type': edge_type,
                'num_types': n_types, 'num_relations': 2 * n_types,
                'note': 'Cell 08 Decagon DDI graph. node=db_id. rel=type_idx(0..N-1)+inverse(N..2N-1)'},
               out_path)
    print(f"ddi_network.pt edges={edge_index.shape[1]} num_relations={2*n_types} -> {out_path}")


if __name__ == '__main__':
    main()
