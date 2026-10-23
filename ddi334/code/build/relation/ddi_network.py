#!/usr/bin/env python3
"""
DDI-334 DDI network whole graph (Cell 08, Decagon R-GCN)
=============================================================================
Builds the whole DDI graph that Cell 08 (DDI network R-GCN) uses, from the
TWOSIDES train-split edges, into data/relation/ddi_network.pt. node=db_id; no
external biological relations are included (the DDI network uses only the DDI
connections themselves -- separate from knowledge_graph.py's knowledge graph).
Node init (Morgan FP default / TransE) is selected by train.py's NODE_INIT env.

Output: ddi334/data/relation/ddi_network.pt
  edge_index[2,E], edge_type[E] (TWOSIDES type_idx 0..N-1 + inverse),
  num_types, num_relations(=2*num_types)

Relation scheme: see relation_map.json (bio 0-22, TWOSIDES = 23+type_idx).
Cell 08 is DDI-only, so relations are remapped to type_idx (0..N-1) (rel-23) + inverse.

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
        src.append(t); dst.append(h); etype.append(tyi + n_types)   # inverse relation
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
