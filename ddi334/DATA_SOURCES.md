# Data sources and build pipeline

The files under `ddi334/data/` in this repository (train/valid/test
splits, KG triples, id maps) are the exact outputs used for the results
reported in the paper. This document describes where the upstream raw
data came from and how to regenerate everything from scratch.

## Required directory layout

The build scripts (`dataset.py`, `knowledge_graph.py`, and most other
`build_*.py` files) hardcode the sibling-folder layout of the *original*
internal server working tree they were developed in, which nested this
repository's `ddi334/` one level deeper (inside an `experiment/` wrapper
that sat alongside `original_repo/`, `kge_hetionet/`, and `meeting/`).
That extra wrapper level was removed from this public repository for
clarity, so these scripts' hardcoded relative paths to
`original_repo/`, `kge_hetionet/`, and `meeting/` no longer resolve
as-is even if you had copies of that raw data:

```
<original internal layout, for reference only>
  v5_ddi-bench/
    experiment/
      ddi334/code/build/splits/dataset.py, build/relation/knowledge_graph.py, ...   <- this repo's ddi334/
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
code) — these build scripts are included as documentation of exactly
how the shipped `ddi334/data/` files were produced, not as something
you are expected to re-run. To actually re-run one, recreate the
layout above and adapt the hardcoded path constants near the top of
the script (`DDI334_DIR`/`EXP_DIR`/`V5_DIR`/`LATEX_ROOT` or similar) to
point at wherever you put the raw sources.

## Upstream sources

| Source | Used for | Where to get it |
|---|---|---|
| **DDI-Bench** (Shen et al., *Bioinformatics* 2025, "Benchmarking drug-drug interaction prediction methods: a perspective of distribution changes") | TWOSIDES 209-type cluster split, DrugBank DDI cluster split, DrugBank SMILES (`initial/drugbank/id2smiles.json`) | `git clone https://github.com/LARS-research/DDI-Bench` -> place its `DDI_Ben/` folder at `original_repo/DDI_Ben/DDI_Ben/` |
| **EmerGNN** (same authors) | HetioNet non-DDI KG edges (`EmerGNN/DrugBank/data/KG.txt`) | `git clone https://github.com/LARS-research/EmerGNN` -> place at `original_repo/DDI_Ben/EmerGNN/` (per the DDI-Bench repo's own README, EmerGNN is set up as a sibling with its own environment) |
| **TDC TWOSIDES** (1,308-type version) | `meeting/ddi334_tdc/*.csv` | Therapeutics Data Commons, standard `pytdc` API (pinned `pytdc==0.4.1` in the DDI-Bench repo's `requirements.txt`): `from tdc.multi_pred import DDI; data = DDI(name="TWOSIDES")`, then export the default train/valid/test split to CSV (`ID1,ID2,Y,Side Effect Name` columns). No custom preprocessing beyond TDC's own split. |
| **DrugBank** full XML | Drug descriptions, SMILES cross-check | Requires a DrugBank academic license (https://go.drugbank.com/releases/latest) — cannot be redistributed here. This project used `drugbank_5.1.12_full.xml`. |
| **HetioNet raw files** (`kge_hetionet/data/hetionet_edges.sif.gz`, `hetionet_nodes.tsv`, `drugbank.tab`, `twosides.csv`) | Bio-relation KG for the Relation-modality encoders; base for the KGE training in `kge_hetionet/` | Downloaded directly from the official HetioNet project's GitHub repository (`hetio/hetionet`, v1.0 data release, https://het.io/). Note: `drugbank.tab` and `twosides.csv` here are the raw DrugBank/TWOSIDES cross-reference tables HetioNet ships alongside its graph, not the same files as the DrugBank XML / TDC TWOSIDES CSVs above. |

### Two name-lookup files with no build script

`data/splits/drug_names.json` (db_id -> drug name, 333/334) and
`data/relation/hetionet_names.json` (entity_id -> name, 3,117 non-drug
HetioNet entities) are not pipeline outputs — they are themselves raw
lookup tables lifted directly from the two licensed/upstream sources
above (DrugBank's name field for `drug_names.json`, `hetionet_nodes.tsv`
for `hetionet_names.json`), keyed by this project's own db_id / entity_id
numbering. Since the DrugBank XML and HetioNet node file are not
redistributed in this repository either (see the table above), there is
no extraction script to ship: these two JSON files are included as-is,
the same way the split/triple files in `data/` are included as the exact
outputs actually used, without the unredistributable raw inputs behind
them.

## Build pipeline

1. **`ddi334/code/build/splits/dataset.py`** (env: `ddibench`, seed=42, fully
   deterministic, no intermediate files written to disk):
   1. Resolve the 334 drugs at the DrugBank x TWOSIDES intersection
      and their canonical cid/db/local id mapping -> `id_maps.json`
   2. Tanimoto (Morgan fingerprint) + union-find clustering, gamma=0.25
      -> `cluster_split.json` (Dk=270 known / Dn=64 novel drugs)
   3. Build the "neither TWOSIDES-connected nor DrugBank-connected"
      negative pool from all `C(334, 2)` pairs
   4. Collect ddibn (209-type) positives, split by Dk/Dn membership
      into S0/S1/S2, cross negatives 1:1 -> writes `data/splits/` (train/
      valid/test `*.txt`, format `h t y0,...,y(N-1) p`). The raw TDC
      TWOSIDES CSVs (`meeting/ddi334_tdc/*.csv`, 1,308-type) are only
      used upstream, in step 1, to define the 334-drug universe; this
      repository does not ship or use a separate 1,308-type dataset.

2. **`ddi334/code/build/relation/knowledge_graph.py`** (Table 2's "Knowledge
   graph" row, Cell 09/10/11 — produces both the KGE training inputs under
   `ddi334/data/relation/kge/` and, as its last step, `kg_triples.npy`, the
   KG triples array Cell 10/11's subgraph extractor reads from):
   - Combines HetioNet bio edges (`kge_hetionet/data/KG.txt`, non-DDI
     relations) with the non-334-drug subset of DrugBank DDI
     (`kge_hetionet/data/train.txt`) and the 334-drug TWOSIDES
     **train**-split edges only (so Dn/cold drugs have zero DDI edges
     into the KG, which is what makes S1/S2 inductive)
   - KGE models (TransE/RotatE/DistMult/ComplEx/TransH) are then
     trained with `kge_hetionet/src/baseline.py` (PyKEEN, hyperparameters
     from the AstraZeneca DRKG baseline configs)
   - `run_all.sh` and `configs/*.yaml` under `data/relation/kge/` are frozen
     snapshots of the actual training run (not regenerated for this
     repository's folder layout), so they still show the pre-rename
     folder name `ddibn` (now `existing_drug` / `novel_drug` here) and
     a generic `/path/to/internal_server/...` stand-in for the original
     internal server's absolute paths — reference only, not meant to
     be re-run as-is

3. Encoder-specific precomputation turns the above into the per-cell input
   caches consumed by `train/train.py` (not shipped in this repo — see the
   main README's "What is / isn't included" section):
   - `build/relation/ddi_network.py` (Cell 08 whole DDI graph),
     `build/relation/bio_profile.py` (Cell 07 raw bio-neighbor vector)
   - `build/document/mention_description.py` (genes/indications/pharm-class
     text source, then Cell 12's frozen BERT embeddings of it)
   - `build/molecule/smiles.py` (filters the shared drug_smiles.json down
     to the 334-drug cohort, then builds Cell 02's ChemBERTa embeddings),
     `build/molecule/fingerprint.py` (Cell 03 Morgan fingerprint),
     `build/molecule/mol_graph.py` (Cell 04/05 2D graph and Cell 06 SchNet
     3D coordinates) — these compute their outputs directly for the 334
     drugs rather than filtering a precomputed pool, so they are fully
     reproducible from this repository alone (unlike step 2's KGE
     training, which depends on the external `kge_hetionet/src/baseline.py`)

4. `data/splits/typenames.json` (209 side-effect names, e.g. `{"0": "aplasia
   pure red cell", ...}`, used to render LLM prompts in human-readable form
   instead of bare type indices) has no corresponding script in this
   repository, since its derivation script required the raw TDC TWOSIDES
   CSVs (`meeting/ddi334_tdc/*.csv`, 1,308-type, named) that are not
   included here either — so the script could not be re-run as shipped
   and was removed rather than kept as dead code. For provenance: our
   209-type DDI-Bench-based positive-pair pattern was matched column-by-
   column against the TDC 1,308-type positive-pair pattern on the 15,471
   drug pairs the two datasets share; every one of the 209 types matched
   exactly one TDC column (no ambiguous or unmatched types), so each
   type's name was taken directly from that TDC column's label.

## Verification

`dataset.py`'s output was checked by SHA256 against the version
actually used to produce the paper's results — the same deterministic
build (seed=42) should reproduce identical files.
