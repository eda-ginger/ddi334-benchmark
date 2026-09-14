# Data sources and build pipeline

The files under `ddi334/data/` in this repository (train/valid/test
splits, KG triples, id maps) are the exact outputs used for the results
reported in the paper. This document describes where the upstream raw
data came from and how to regenerate everything from scratch.

## Required directory layout

The build scripts (`build_dataset.py`, `build_kg.py`) resolve their
input paths relative to a fixed sibling-folder layout that mirrors how
this project's working repository is organized:

```
<some root>/
  ddi334_experiment/        <- this repository
    ddi334/code/build_dataset.py, build_kg.py, ...
  original_repo/
    DDI_Ben/
      DDI_Ben/data/{twosides_cluster,drugbank_cluster,initial}/...
      EmerGNN/DrugBank/data/KG.txt
  kge_hetionet/
    data/{KG.txt, train.txt, twosides.csv, drugbank.tab,
          hetionet_edges.sif.gz, hetionet_nodes.tsv}
  meeting/
    ddi334_tdc/*.csv
```

`original_repo/` and `kge_hetionet/` are **not included** in this
repository (they are third-party / regenerable raw data, not our
code). To rerun the build scripts as-is, recreate this layout
yourself — or adapt the hardcoded paths near the top of
`build_dataset.py` / `build_kg.py` to point at wherever you put the
raw sources.

## Upstream sources

| Source | Used for | Where to get it |
|---|---|---|
| **DDI-Bench** (Shen et al., *Bioinformatics* 2025, "Benchmarking drug-drug interaction prediction methods: a perspective of distribution changes") | TWOSIDES 209-type cluster split, DrugBank DDI cluster split, DrugBank SMILES (`initial/drugbank/id2smiles.json`) | `git clone https://github.com/LARS-research/DDI-Bench` -> place its `DDI_Ben/` folder at `original_repo/DDI_Ben/DDI_Ben/` |
| **EmerGNN** (same authors) | HetioNet non-DDI KG edges (`EmerGNN/DrugBank/data/KG.txt`) | `git clone https://github.com/LARS-research/EmerGNN` -> place at `original_repo/DDI_Ben/EmerGNN/` (per the DDI-Bench repo's own README, EmerGNN is set up as a sibling with its own environment) |
| **TDC TWOSIDES** (1,308-type version) | `meeting/ddi334_tdc/*.csv` | Therapeutics Data Commons, `pytdc` package (pinned `pytdc==0.4.1` in the DDI-Bench repo's `requirements.txt`). Load via `tdc.multi_pred.DDI(name="TWOSIDES")` and export train/valid/test splits as CSV (`ID1,ID2,Y,Side Effect Name` columns). [TODO: 정확히 어떤 TDC API 호출/스크립트로 저장했는지 재확인 필요] |
| **DrugBank** full XML | Drug descriptions, SMILES cross-check | Requires a DrugBank academic license (https://go.drugbank.com/releases/latest) — cannot be redistributed here. This project used `drugbank_5.1.12_full.xml`. |
| **HetioNet raw files** (`kge_hetionet/data/hetionet_edges.sif.gz`, `hetionet_nodes.tsv`, `drugbank.tab`, `twosides.csv`) | Bio-relation KG for the Relation-modality encoders; base for the KGE training in `kge_hetionet/` | [TODO: 정확한 다운로드 출처 재확인 필요 — Hetionet 프로젝트(https://het.io/) 자체 릴리스인지, DDI-Bench/EmerGNN 번들에 포함된 건지 확인 후 채우기] |

## Build pipeline

1. **`ddi334/code/build_dataset.py`** (env: `ddibench`, seed=42, fully
   deterministic, no intermediate files written to disk):
   1. Resolve the 334 drugs at the DrugBank x TWOSIDES intersection
      and their canonical cid/db/local id mapping -> `id_maps.json`
   2. Tanimoto (Morgan fingerprint) + union-find clustering, gamma=0.25
      -> `cluster_split.json` (Dk=270 known / Dn=64 novel drugs)
   3. Build the "neither TWOSIDES-connected nor DrugBank-connected"
      negative pool from all `C(334, 2)` pairs
   4. For each dataset version (`ddibn` 209-type, `tdc` 1,308-type):
      collect positives, split by Dk/Dn membership into S0/S1/S2,
      cross negatives 1:1 -> writes `ddibn/` and `tdc/` (train/valid/
      test `*.txt`, format `h t y0,...,y(N-1) p`)

2. **`ddi334/code/build_kg.py`** (produces KGE training inputs under
   `ddi334/data/kge/`):
   - Combines HetioNet bio edges (`kge_hetionet/data/KG.txt`, non-DDI
     relations) with the non-334-drug subset of DrugBank DDI
     (`kge_hetionet/data/train.txt`) and the 334-drug TWOSIDES
     **train**-split edges only (so Dn/cold drugs have zero DDI edges
     into the KG, which is what makes S1/S2 inductive)
   - KGE models (TransE/RotatE/DistMult/ComplEx/TransH) are then
     trained with `kge_hetionet/src/baseline.py` (PyKEEN, hyperparameters
     from the AstraZeneca DRKG baseline configs)

3. Encoder-specific precomputation (`ddi334/code/build_bio_profile.py`,
   `build_bio_raw.py`, `build_biobert_template.py`,
   `assemble_precompute.py`, ...) turns the above into the per-cell
   input caches consumed by `train.py` (not shipped in this repo —
   see the main README's "What is / isn't included" section).

## Verification

`build_dataset.py`'s output was checked by SHA256 against the version
actually used to produce the paper's results (see the project's
internal experiment log, `09_실험전체기록(DDI334).md` section 1) — the
same deterministic build (seed=42) should reproduce identical files.
