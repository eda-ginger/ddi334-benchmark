#!/usr/bin/env python3
"""
DDI-334 Knowledge graph triples (Cell 10/11 subgraph extraction source)
=============================================================================
재번호 KG(bio 0-22, TWOSIDES 23+)에 맞춰 Cell 10/11의 R-GCN/Graph Transformer
subgraph encoder가 쓸 전체 KG triples를 data/relation/kg_triples.npy로 만든다.
train.py의 SubgraphExtractor가 이 파일에서 각 drug pair 주변 k-hop subgraph를
뽑아낸다. (node init=TransE는 별도 kge/existing_drug/transe/.npy, kg.py 참조)

출력: ddi334/data/relation/kg_triples.npy
  [E,3] (h, t, r)  (v5 load_hetionet_triples 순서)
  = bio(kg_bio_334, rel 0-22) + TWOSIDES train(ddi_train, rel 23+)

관계 체계: relation_map.json 참조 (bio 0-22, TWOSIDES = 23+type_idx).

Run: micromamba run -n DDIBench python build/relation/kg_triples.py
"""
import os
import numpy as np

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REL_DIR = os.path.join(DDI334_DIR, 'data', 'relation')
KGE = os.path.join(REL_DIR, 'kge')
KG_BIO = os.path.join(KGE, 'kg_bio_334.txt')


def main():
    triples = []
    for line in open(KG_BIO):                       # bio, rel 0-22
        h, t, r = line.split()
        triples.append([int(h), int(t), int(r)])    # (h, t, r) — v5 순서
    ddi_path = os.path.join(KGE, 'existing_drug', 'ddi_train.txt')
    for line in open(ddi_path):                     # TWOSIDES typed, rel 23+
        h, t, r = line.split()
        triples.append([int(h), int(t), int(r)])
    kg = np.array(triples, dtype=np.int64)
    out_path = os.path.join(REL_DIR, 'kg_triples.npy')
    np.save(out_path, kg)
    print(f"kg_triples.npy {kg.shape} (bio+TWOSIDES, rel 0-{kg[:,2].max()}) -> {out_path}")


if __name__ == '__main__':
    main()
