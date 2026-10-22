#!/usr/bin/env python3
"""DDI-334 Molecular graph representation (Cell 04/05 2D, Cell 06 3D) — Table 2's
"Molecular graph" row covers both scopes, so both are built here.

2D (Cell 04 GAT, Cell 05 Graph Transformer): TIGER-style graph (67-d atom
   features + bond edges + shortest-path info up to distance 8), built by
   train/encoders.py's mol_to_graph_tiger() -- imported from here so the
   logic stays identical to what train/train.py itself uses, rather than a
   duplicated copy that could drift.
   -> data/molecule/mol_graph_2d.pt  {int db_id: {'x','edge_index',
      'sp_edge_index','sp_value','sp_edge_rel'}}
   Was previously computed inline in train/train.py on every run (cheap
   RDKit call, no caching); moved here for the same reason as fingerprint.py.

3D (Cell 06 SchNet): ETKDGv3 + MMFF94 (UFF fallback for metal complexes),
   timeout 300s. Ported from the shared v5 framework's
   precompute/scripts/schnet_coords.py (full 1,710-drug population before a
   separate filtering step); this version computes directly for the 334-drug
   SMILES file instead -- same numbers, fully self-contained here.
   -> data/molecule/schnet_coords.pt  {int db_id: {'z','pos','opt'}}
   -> data/molecule/schnet_drop.json  failed/timed-out db_ids (not shipped
      in this repo: all 334 DDI-334 drugs succeed, so this file is empty)

Run: micromamba run -n DDIBench python build/molecule/mol_graph.py [--dim 2d|3d|both] [--resume]
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOL_DIR = os.path.join(DDI334_DIR, 'data', 'molecule')
SMILES_JSON = os.path.join(MOL_DIR, 'drug_smiles.json')
OUT_2D = os.path.join(MOL_DIR, 'mol_graph_2d.pt')
OUT_3D = os.path.join(MOL_DIR, 'schnet_coords.pt')
OUT_3D_DROP = os.path.join(MOL_DIR, 'schnet_drop.json')


def build_2d():
    sys.path.insert(0, os.path.join(CODE_DIR, 'train'))
    from encoders import mol_to_graph_tiger

    smiles_map = {int(k): v for k, v in json.load(open(SMILES_JSON)).items()}
    graphs = {did: mol_to_graph_tiger(smi) for did, smi in smiles_map.items()}
    torch.save(graphs, OUT_2D)
    print(f'Saved: {OUT_2D}  ({len(graphs)} drugs, TIGER 67-d atom features)')


# ── 3D (SchNet) ──────────────────────────────────────────────────────────
class TimeoutError_(Exception):
    pass


def _alarm(signum, frame):
    raise TimeoutError_('mol timeout')


def smiles_to_3d(smiles: str, max_retries: int = 10, seed: int = 42,
                 timeout_sec: int = 300):
    """ETKDGv3 + MMFF94 (UFF fallback). Returns ({'z','pos','opt'}, 'ok') or (None, reason)."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_sec)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, 'parse_fail'
        mol = Chem.AddHs(mol)

        ps = AllChem.ETKDGv3()
        ps.useRandomCoords = True
        ps.maxAttempts = max_retries * 100
        ps.numThreads = 1
        ps.enforceChirality = False
        ps.useSmallRingTorsions = True

        embedded = False
        for r in range(max_retries):
            ps.randomSeed = seed + r
            if AllChem.EmbedMolecule(mol, ps) != -1:
                embedded = True
                break
        if not embedded:
            return None, 'embed_fail'

        opt_method = None
        try:
            res = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            opt_method = 'MMFF'
        except Exception:
            res = -1

        if res == -1:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
                opt_method = 'UFF'
            except Exception:
                return None, 'optimize_fail'

        conf = mol.GetConformer()
        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
        pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                       dtype=np.float32)
        return {'z': z, 'pos': pos, 'opt': opt_method}, 'ok'

    except TimeoutError_:
        return None, 'timeout'
    except Exception as e:
        return None, f'exception:{type(e).__name__}'
    finally:
        signal.alarm(0)


def build_3d(resume=False, max_retries=10, seed=42, timeout_sec=300):
    smiles_map = {int(k): v for k, v in json.load(open(SMILES_JSON)).items()}
    drug_ids = sorted(smiles_map.keys())
    print(f'Target drugs: {len(drug_ids)}')

    coords = {}
    if resume and os.path.exists(OUT_3D):
        coords = torch.load(OUT_3D, weights_only=False)
        print(f'Resume: loaded {len(coords)} existing entries')

    drops = {}
    opt_counts = {'MMFF': 0, 'UFF': 0}
    t0 = time.time()
    todo = [did for did in drug_ids if did not in coords]
    print(f'To process: {len(todo)} drugs')

    for i, did in enumerate(todo):
        smi = smiles_map.get(did)
        c, reason = smiles_to_3d(smi, max_retries=max_retries, seed=seed, timeout_sec=timeout_sec)
        if c is None:
            drops[did] = reason
        else:
            coords[did] = c
            opt_counts[c['opt']] = opt_counts.get(c['opt'], 0) + 1

        if (i + 1) % 100 == 0 or (i + 1) == len(todo):
            print(f'  [{i+1}/{len(todo)}] ok={len(coords)}  fail={len(drops)}  '
                  f'wall={time.time()-t0:.0f}s', flush=True)

    torch.save(coords, OUT_3D)
    if drops:
        json.dump({'failed': drops, 'fail_count': len(drops), 'success_count': len(coords),
                   'total_target': len(drug_ids), 'opt_counts': opt_counts},
                  open(OUT_3D_DROP, 'w'), indent=2)

    print(f'Saved: {OUT_3D}  (success={len(coords)} failed={len(drops)} opt={opt_counts} '
          f'wall={time.time()-t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', default='both', choices=['2d', '3d', 'both'])
    ap.add_argument('--resume', action='store_true', help='3D only: skip already-completed drugs')
    ap.add_argument('--max_retries', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--timeout_sec', type=int, default=300)
    args = ap.parse_args()

    if args.dim in ('2d', 'both'):
        build_2d()
    if args.dim in ('3d', 'both'):
        build_3d(resume=args.resume, max_retries=args.max_retries, seed=args.seed,
                  timeout_sec=args.timeout_sec)


if __name__ == '__main__':
    main()
