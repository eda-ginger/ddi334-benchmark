#!/usr/bin/env python3
"""DDI-334 Fingerprint representation (Cell 03, MLP).

Source: data/molecule/drug_smiles.json (334 drugs, built by smiles.py)
Method: Morgan fingerprint, radius=2, 1024 bits (RDKit GetMorganFingerprintAsBitVect)
Output: data/molecule/fingerprint.pt
        {'embeddings': {int db_id: Tensor[1024]}}

Was previously computed inline in train/train.py on every run (cheap RDKit
call, no caching) -- moved here so every Molecule input representation has
one build script producing one cached file, the same way Relation's inputs
do (bio_profile.py, ddi_network.py, kg.py/kg_triples.py).

Run: micromamba run -n DDIBench python build/molecule/fingerprint.py
"""
import json
import os

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MOL_DIR = os.path.join(DDI334_DIR, 'data', 'molecule')
SMILES_JSON = os.path.join(MOL_DIR, 'drug_smiles.json')
OUT_PT = os.path.join(MOL_DIR, 'fingerprint.pt')


def main():
    smiles_map = {int(k): v for k, v in json.load(open(SMILES_JSON)).items()}
    embeddings = {}
    for did, smi in smiles_map.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
        else:
            fp = np.zeros(1024, dtype=np.float32)
        embeddings[did] = torch.from_numpy(fp)

    torch.save({'embeddings': embeddings}, OUT_PT)
    print(f'Saved: {OUT_PT}  ({len(embeddings)} drugs, dim=1024)')


if __name__ == '__main__':
    main()
