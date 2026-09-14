#!/usr/bin/env python3
"""
DDI-334 전용 PCA 전처리 (Cell 05 FP, Cell 12 bio) — 누수 방지 재계산
=====================================================================
기존 precompute의 fp_pca/bio_pca는 v5/1710 DrugBank split(train 1347)로 fit돼
DDI-334의 cold(Dn) 약물이 PCA basis에 들어가 inductive 누수가 있었다.

이 스크립트는 KGE와 동일 원칙으로 **데이터셋별(ddibn/tdc)** 재계산:
  - Jaccard 기준 + PCA fit = 해당 데이터셋 **train.txt 등장 약물만** (cold Dn 제외)
  - 모든 334 약물을 그 basis로 transform
  - 데이터셋별 train 약물이 다를 수 있어 따로 보관

방법은 기존 scripts/{fp_pca,bio_pca}.py 그대로(train-only Jaccard+PCA), fit set만 교체.
출력: ddi334/precompute/{ddibn,tdc}/{fp_pca.pt, bio_pca.pt}  (334 db_id 키)

Run: micromamba run -n DDIBench python precompute_pca.py
"""
import os
import json
import time
import numpy as np
import torch
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

DDI334_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.dirname(DDI334_DIR)
SMILES_JSON = os.path.join(EXP_DIR, 'data', 'drug_smiles.json')
KG_BIO_334 = os.path.join(DDI334_DIR, 'data', 'kge', 'kg_bio_334.txt')   # HetioNet bio (334 약물 노드만)
DATASETS = ['ddibn', 'tdc']               # 데이터셋별 train 약물로 fit, 폴더 안에 저장
DRUG_MAX = 1710        # entity id < 1710 = 약물, >= 1710 = 비-약물(gene/disease/...)
FP_COMPONENTS = 128
BIO_COMPONENTS = 384   # min(384, n_fit)로 자동 클램프
RADIUS, N_BITS = 2, 1024


def jaccard(A, B):
    A = A.astype(np.float32); B = B.astype(np.float32)
    inter = A @ B.T
    union = A.sum(1, keepdims=True) + B.sum(1, keepdims=True).T - inter
    return (inter / np.maximum(union, 1e-12)).astype(np.float32)


def load_all_ids():
    idm = json.load(open(os.path.join(DDI334_DIR, 'meta', 'id_maps.json')))
    return sorted(int(x) for x in idm['db_ids'])              # 334


def load_dk():
    """기준(reference/fit) 약물 = Dk(known) 270. cold(Dn 64)만 제외.
    cluster split이 ddibn/tdc 공통이라 두 데이터셋 동일."""
    split = json.load(open(os.path.join(DDI334_DIR, 'meta', 'cluster_split.json')))
    return sorted(int(x) for x in split['Dk'])


def bio_profile_hetionet(all_ids):
    """Cell 12 bio profile = HetioNet bio 이웃 (DrugBank 속성 대신).
    각 334 약물이 bio 관계로 연결된 **비-약물 entity(gene/disease/side-effect 등)** 집합 -> binary.
    DDIMDL 방식(약물 feature 프로파일 -> Jaccard -> PCA)을 HetioNet으로 적용. (CrC 등 drug-drug 제외)"""
    drug_set = set(all_ids)
    neighbors = {d: set() for d in all_ids}
    for line in open(KG_BIO_334):
        p = line.split()
        if len(p) != 3:
            continue
        h, t = int(p[0]), int(p[1])
        if h in drug_set and t >= DRUG_MAX:        # drug -> 비약물 entity
            neighbors[h].add(t)
        elif t in drug_set and h >= DRUG_MAX:
            neighbors[t].add(h)
    feats = sorted(set().union(*neighbors.values())) if neighbors else []
    fidx = {e: i for i, e in enumerate(feats)}
    M = np.zeros((len(all_ids), len(feats)), dtype=np.uint8)
    for i, d in enumerate(all_ids):
        for e in neighbors[d]:
            M[i, fidx[e]] = 1
    return M, len(feats)


def smiles_to_fp(smi):
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return np.zeros(N_BITS, dtype=np.uint8)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    arr = np.zeros(N_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def run_pca(M_all, all_ids, fit_ids, n_components, tag, zero_out_empty=False):
    id2idx = {d: i for i, d in enumerate(all_ids)}
    fit_idx = [id2idx[d] for d in fit_ids]
    M_fit = M_all[fit_idx]
    J_all = jaccard(M_all, M_fit)            # [334 × 270]
    J_fit = J_all[fit_idx]                    # [270 × 270]
    n_comp = min(n_components, len(fit_ids))
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(J_fit)
    evr = float(pca.explained_variance_ratio_.sum())
    V = pca.transform(J_all).astype(np.float32)
    zero_profile = 0
    if zero_out_empty:
        zmask = (M_all.sum(1) == 0)
        V[zmask] = 0.0
        zero_profile = int(zmask.sum())
    print(f"  [{tag}] dim={n_comp} EVR={evr:.4f} fit={len(fit_ids)} all={len(all_ids)} "
          f"zero_profile={zero_profile}")
    emb = {d: torch.from_numpy(V[id2idx[d]]) for d in all_ids}
    return emb, n_comp, evr, zero_profile


def main():
    t0 = time.time()
    all_ids = load_all_ids()
    smiles = {int(k): v for k, v in json.load(open(SMILES_JSON)).items()}
    fp_all = np.zeros((len(all_ids), N_BITS), dtype=np.uint8)
    for i, d in enumerate(all_ids):
        fp_all[i] = smiles_to_fp(smiles.get(d))
    # Cell 12 bio = HetioNet bio 이웃 프로파일 (DrugBank 속성 대신)
    bio_all, n_bio_feat = bio_profile_hetionet(all_ids)
    print(f"bio profile (HetioNet 이웃): {bio_all.shape} (feat={n_bio_feat} 비약물 entity)")

    fit_ids = load_dk()   # Dk 270 (cold Dn 제외), ddibn/tdc 공통
    # DDIMDL 방식: PCA n_components = 기준 약물 수(=Dk 270) 전체 유지 (축소가 아닌 decorrelation)
    n_comp = len(fit_ids)
    print(f"all(334)={len(all_ids)}  fit/Dk={len(fit_ids)}  PCA dim={n_comp} (DDIMDL: 기준약물 수 전체)")

    for ds in DATASETS:
        out_dir = os.path.join(DDI334_DIR, 'data', ds, 'precompute')
        os.makedirs(out_dir, exist_ok=True)

        # Cell 05: Morgan FP Jaccard -> PCA
        emb, dim, evr, _ = run_pca(fp_all, all_ids, fit_ids, n_comp, 'fp')
        torch.save({'embeddings': emb, 'input_dim': dim, 'n_all': len(all_ids),
                    'n_train': len(fit_ids), 'evr': evr,
                    'method': f'fp_jaccard_pca_DDIMDL_Dk_ddi334_{ds}'},
                   os.path.join(out_dir, 'fp_pca.pt'))

        # Cell 12: bio profile Jaccard -> PCA
        emb, dim, evr, zp = run_pca(bio_all, all_ids, fit_ids, n_comp, 'bio', zero_out_empty=True)
        torch.save({'embeddings': emb, 'input_dim': dim, 'n_all': len(all_ids),
                    'n_train': len(fit_ids), 'zero_profile': zp, 'evr': evr,
                    'source': 'hetionet_bio_neighbors',
                    'method': f'bio_jaccard_pca_DDIMDL_Dk_hetionet_{ds}'},
                   os.path.join(out_dir, 'bio_pca.pt'))
        print(f"  [{ds}] -> {out_dir}/{{fp_pca,bio_pca}}.pt (fp dim={dim} 동일)")

    print(f"\n[DONE] 데이터셋별 precompute 생성 (fit=Dk {len(fit_ids)}, dim {n_comp}) wall={time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
