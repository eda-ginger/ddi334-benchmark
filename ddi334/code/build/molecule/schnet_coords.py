#!/usr/bin/env python3
"""SchNet 3D coordinates precompute for Cell 06 (DDI-334).

Source : data/molecule/drug_smiles.json  {str db_id: smiles} (334 drugs,
         already filtered to the DDI-334 cohort by precompute.py)
3D method: ETKDGv3 + MMFF94 (UFF fallback for metal complexes), timeout 300s.
Output : data/molecule/schnet_coords.pt  {int db_id: {'z':..., 'pos':..., 'opt':...}}
         data/molecule/schnet_drop.json  failed/timed-out db_ids + reasons
         (not shipped in this repo: all 334 DDI-334 drugs succeed, so the
         file would be empty -- see precompute.py's comment)

Ported from the shared v5 framework's precompute/scripts/schnet_coords.py
(which computed this for the full 1,710-drug population before a separate
filtering step cut it down to 334); this version computes directly for the
334-drug SMILES file instead, since per-drug 3D coordinates don't depend on
the rest of the population -- same numbers, fully self-contained here.

Run: micromamba run -n DDIBench python build/molecule/schnet_coords.py [--resume]
"""
import argparse
import json
import os
import signal
import time

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MOL_DIR = os.path.join(DDI334_DIR, 'data', 'molecule')
SMILES_JSON = os.path.join(MOL_DIR, 'drug_smiles.json')
OUT_PT = os.path.join(MOL_DIR, 'schnet_coords.pt')
OUT_DROP = os.path.join(MOL_DIR, 'schnet_drop.json')


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

        # MMFF94 first; fallback to UFF for metal complexes
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true', help='Skip already-completed drugs')
    ap.add_argument('--max_retries', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--timeout_sec', type=int, default=300)
    args = ap.parse_args()

    smiles_map = {int(k): v for k, v in json.load(open(SMILES_JSON)).items()}
    drug_ids = sorted(smiles_map.keys())
    print(f'Target drugs: {len(drug_ids)}')

    coords = {}
    if args.resume and os.path.exists(OUT_PT):
        coords = torch.load(OUT_PT, weights_only=False)
        print(f'Resume: loaded {len(coords)} existing entries')

    drops = {}   # db_id -> reason
    opt_counts = {'MMFF': 0, 'UFF': 0}
    t0 = time.time()

    todo = [did for did in drug_ids if did not in coords]
    print(f'To process: {len(todo)} drugs')

    for i, did in enumerate(todo):
        smi = smiles_map.get(did)
        c, reason = smiles_to_3d(smi, max_retries=args.max_retries,
                                  seed=args.seed, timeout_sec=args.timeout_sec)
        if c is None:
            drops[did] = reason
        else:
            coords[did] = c
            opt_counts[c['opt']] = opt_counts.get(c['opt'], 0) + 1

        if (i + 1) % 100 == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(todo)}] ok={len(coords)}  fail={len(drops)}  '
                  f'wall={elapsed:.0f}s', flush=True)

    os.makedirs(MOL_DIR, exist_ok=True)
    torch.save(coords, OUT_PT)

    if drops:
        json.dump({
            'failed': drops, 'fail_count': len(drops), 'success_count': len(coords),
            'total_target': len(drug_ids), 'opt_counts': opt_counts,
        }, open(OUT_DROP, 'w'), indent=2)

    print(f'\nDone.')
    print(f'  success : {len(coords)}')
    print(f'  failed  : {len(drops)}  {drops}')
    print(f'  opt     : {opt_counts}')
    print(f'  wall    : {time.time()-t0:.0f}s')
    print(f'  saved   : {OUT_PT}')


if __name__ == '__main__':
    main()
