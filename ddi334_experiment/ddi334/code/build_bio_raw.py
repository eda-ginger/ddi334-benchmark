#!/usr/bin/env python3
"""
DDI-334 Cell 15 (REL) precompute: raw HetioNet bio-neighbor multi-hot vector
=============================================================================
Cell 12 (InteractionJaccardEncoder) reduces this same raw profile through
Jaccard similarity + train-only PCA before the MLP head. Cell 15 instead
feeds the raw multi-hot vector directly into a 2-layer MLP, mirroring Cell 04
(MorganFPEncoder)'s raw-fingerprint-to-MLP pattern for the Molecule modality
(user decision, appendix.md B.3, 2026-09-08).

No similarity/PCA step, so unlike bio_pca.pt this raw profile has no
train-only-fit leakage concern (each drug's feature is computed independently
of the drug-pair split, same reasoning as the raw Morgan fingerprint used by
Cell 04) — one shared file suffices for both ddibn and tdc.

Reuses precompute_pca.py's bio_profile_hetionet() logic exactly (same feature
index / neighbor extraction), just skips the Jaccard+PCA step and saves the
raw binary matrix instead.

Run: micromamba run -n DDIBench python build_bio_raw.py
"""
import os
import json
import numpy as np
import torch

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V5_DIR = os.path.dirname(os.path.dirname(DDI334_DIR))
KG_BIO_334 = os.path.join(DDI334_DIR, 'data', 'kge', 'kg_bio_334.txt')
DRUG_MAX = 1710
DATASETS = ['ddibn', 'tdc']
SIDE_EFFECT_REL = '21'    # Compound-causes-Side Effect: same vocabulary as the DDI-334
                           # prediction label (adverse events) -> leakage, must be excluded.
                           # Already excluded for this reason in build_bio_profile.py (LLM
                           # prompt text) and data/prompt/ANALYSIS.md; precompute_pca.py's
                           # bio_profile_hetionet() (Cell 12) does NOT exclude it -- found
                           # 2026-09-08 while building this raw-vector cell (Cell 15).
GENE_REL_DIRECT = '6'     # Compound-binds-Gene (CbG) = direct target only. Other Gene-typed
                           # relations (7=upregulates, 18=downregulates) are indirect and are
                           # already excluded for that reason in build_bio_profile.py's
                           # GENE_REL='6' + "up/down-regulate(간접 gene)도 제외" comment;
                           # bio_profile_hetionet() did not apply this narrowing either.
ALLOWED_TYPES = {'Gene', 'Disease'}   # target gene (direct binding) + indication only.
                           # Pharmacologic Class and Compound are excluded (user decision,
                           # 2026-09-08): Compound neighbors turned out to be connected via
                           # CrC (Compound-resembles-Compound), a structural-similarity edge,
                           # which CLAUDE.md's modality rule assigns to Molecule ("KG
                           # containing only molecular similarity edges = Molecule"), not
                           # Relation -- verified directly against the original HetioNet SIF
                           # edge file. Pharmacologic Class was dropped too, to keep the
                           # profile aligned with the target-only focus of the cited
                           # literature (DDIMDL/MDF-SA-DDI's target+enzyme+pathway design).


def load_all_ids():
    idm = json.load(open(os.path.join(DDI334_DIR, 'meta', 'id_maps.json')))
    return sorted(int(x) for x in idm['db_ids'])              # 334


def bio_profile_hetionet(all_ids):
    """Same neighbor extraction as precompute_pca.py's bio_profile_hetionet(), except:
    - Side Effect neighbors (relation 21) are excluded (label leakage).
    - Gene neighbors are restricted to relation 6 (direct target binding); indirect
      up/downregulation relations are excluded (matches build_bio_profile.py's GENE_REL)."""
    e2i = json.load(open(os.path.join(V5_DIR, 'kge_hetionet', 'data', 'entity_drug.json')))
    i2e = {v: k for k, v in e2i.items()}
    drug_set = set(all_ids)
    neighbors = {d: set() for d in all_ids}
    for line in open(KG_BIO_334):
        p = line.split()
        if len(p) != 3:
            continue
        h, t, r = int(p[0]), int(p[1]), p[2]
        if r == SIDE_EFFECT_REL:
            continue
        for a, b in ((h, t), (t, h)):
            if a in drug_set and b >= DRUG_MAX:
                etype = i2e.get(b, '?').split('::')[0]
                if etype not in ALLOWED_TYPES:
                    continue
                if etype == 'Gene' and r != GENE_REL_DIRECT:
                    continue
                neighbors[a].add(b)
    feats = sorted(set().union(*neighbors.values())) if neighbors else []
    fidx = {e: i for i, e in enumerate(feats)}
    M = np.zeros((len(all_ids), len(feats)), dtype=np.uint8)
    for i, d in enumerate(all_ids):
        for e in neighbors[d]:
            M[i, fidx[e]] = 1
    return M, len(feats)


def main():
    all_ids = load_all_ids()
    M, n_feat = bio_profile_hetionet(all_ids)
    print(f"raw bio profile: {M.shape} (feat={n_feat} non-drug HetioNet entities)")
    empty = int((M.sum(1) == 0).sum())
    print(f"  {empty}/{len(all_ids)} drugs with zero HetioNet bio neighbors (all-zero vector)")

    emb = {d: torch.from_numpy(M[i].astype(np.float32)) for i, d in enumerate(all_ids)}
    for ds in DATASETS:
        out_dir = os.path.join(DDI334_DIR, 'data', ds, 'precompute')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'bio_raw.pt')
        torch.save({'embeddings': emb, 'input_dim': n_feat, 'n_all': len(all_ids),
                    'source': 'hetionet_bio_neighbors_raw',
                    'method': 'bio_raw_multihot_no_similarity_no_pca'}, out_path)
        print(f"  -> {out_path}")


if __name__ == '__main__':
    main()
