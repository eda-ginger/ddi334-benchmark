# DDI-334 Benchmark

**Repository**: https://github.com/eda-ginger/ddi334-benchmark

Code and results for the DDI-334 experiments reported in:

> [TODO: citation information (authors, journal, year, DOI) — to be finalized after submission/acceptance]

DDI-334 is a 334-drug benchmark (TWOSIDES x DrugBank overlap) used to
compare how Molecule / Relation / Document information sources, and
LLM-based zero-shot / fine-tuned prediction, perform on drug-drug
interaction prediction under transductive (S0) and inductive (S1, S2)
splits.

This repository contains only the code and result summaries that
produced the numbers, tables, and figures in the final manuscript — it
is not the full experiment history of the underlying project.

## Contents

- [Structure](#structure)
- [What is / isn't included](#what-is--isnt-included)
- [Data](#data)
- [Reproducing a result](#reproducing-a-result)
- [Environment](#environment)
- [License](#license)

## Structure

```
ddi334_experiment/
  models/          model/encoder definitions (GAT, SchNet, GT, MLP, CNN,
                   ChemBERTa, BioBERT, KGE-based encoders, ...)
  dataio/          dataset loading
  ddi334/
    code/              main trainer (`train.py`, cells 01-15) and all
                       DDI-334-specific scripts (dataset build, KG build,
                       LLM prompt rendering/inference, result aggregation,
                       cross-evaluation ablations, ...). See below.
    data/              train/valid/test split files (S0/S1/S2), drug/type
                       maps, KG triples used for the Relation modality
    meta/              cluster split definition, id maps, type names
    results/           summary metrics (per-class scores, configs used,
                       loss curves, LLM ZS/FT metrics, KGE comparison
                       results, commercial-LLM comparison)
    DATA_SOURCES.md    upstream data sources + build pipeline
environment/
  README.md, *_requirements.txt   exact pip environments used
```

`models/` and `dataio/` are imported by `ddi334/code/train.py` (added to
`sys.path` there) — they are shared, not a separate pipeline.

`ddi334/results/` is organized by which part of the paper it backs:

```
results/
  encoders_main/                        Table 1/D.1: cells 01-15, Existing
                                         drug condition (the default for
                                         every cell; Molecule/Bio-profile
                                         cells have no Novel-drug variant)
  encoders_novel_drug/                  Section 5.2/Figure: cells 08-11
                                         under the Novel drug condition
  appendix_j_embedding_subgraph_cross/  Appendix J: TransE embedding and
    novel_embedding_existing_subgraph/    KG subgraph crossed independently
    existing_embedding_novel_subgraph/    for Knowledge graph (R-GCN)
  document_backbone_comparison/         Appendix C.1: BioBERT vs PubMedBERT
    {biobert,pubmedbert,scibert}/         vs SciBERT, each under both
      {existing_drug,novel_drug}/         drug conditions
  kge_comparison/                       Appendix KGE comparison figure
    existing_drug/                        (TransE/RotatE/ComplEx/DistMult/
    novel_drug/                           TransH): link-reconstruction vs
    novel_drug_embedding_only/            downstream AUROC. novel_drug_embed
                                           ding_only isolates the embedding
                                           from the graph-structure effect
                                           (Section 6.4-style ablation)
  llm/
    finetuning/                         Section 5: QLoRA FT, 6 models x
                                         Existing/Novel drug condition x
                                         3 epochs
    zeroshot/                           Section 5: zero-shot, 6 models x
                                         Existing/Novel
    zeroshot_s2_reproducibility_check/  reproducibility check for the
                                         Unseen-Unseen zero-shot numbers
  ddi_network_molgraph_init/            robustness check for DDI network
                                         (R-GCN) with a different node
                                         initialization (not reported in
                                         the paper; see Reproducing a
                                         result below)
  commercial_llm/                       Appendix I: commercial LLM comparison
  random_baseline.json                  random-prediction reference line
```

Inside each leaf folder, filenames use the codebase's internal mode
names `novel_drug` / `existing_drug` for the Existing/Novel drug
condition (and `S0`/`S1`/`S2` = Seen-Seen/Seen-Unseen/Unseen-Unseen).

## What is / isn't included

**Included**: everything needed to verify every number, table, and
figure reported in the paper — summary metrics (csv/json), the exact
run configs, loss curves, and the train/val/test split membership
files used for S0 (transductive) / S1 / S2 (inductive) evaluation.

**Not included** (regenerable, not shipped due to size):
- Model checkpoints and LoRA adapter weights
- Per-sample raw prediction arrays (`raw_preds/*.npz`) — only the
  aggregated metrics computed from them are kept
- Precomputed encoder embedding caches (SchNet coordinates, ChemBERTa/
  BioBERT embeddings, TransE/KGE embeddings, Morgan-fingerprint/
  pooled-atom-feature node-init caches, rendered LLM prompt jsonl
  files) — derived from the raw external data below and regenerable
  with the scripts in `ddi334/code/` (the fingerprint/atom-feature
  caches are rebuilt automatically on first run of `train.py`)
- Training/inference logs

## Data

The DDI-334 splits and KG triples in `ddi334/data/` are the exact
files used for the paper's results. The upstream raw sources
(DDI-Bench/EmerGNN, TDC TWOSIDES, DrugBank, HetioNet) are third-party
and are not redistributed here.

See [`ddi334/DATA_SOURCES.md`](ddi334_experiment/ddi334/DATA_SOURCES.md)
for exact sources, the required sibling-folder layout, and the build
pipeline (`build_dataset.py` -> `build_kg.py` -> per-encoder
precomputation).

## Reproducing a result

```bash
cd ddi334_experiment/ddi334/code

# example: cell04 (SchNet), DDI-334 (ddibn), seed 42
python train.py --cell 04 --dataset ddibn --gpu 0 --seed 42
```

`--checkpoint_metric` is not a flag here — checkpoint selection (accuracy,
per-split, patience=10 early stopping, as described in the paper's
Appendix E.1) is fixed in `train.py` and is not configurable.

**Existing / Novel drug conditions** (biological-relation availability for
Unseen drugs, Section 4 of the paper) are selected with env vars, not CLI
flags:

```bash
# Existing drug condition (default)
python train.py --cell 09 --dataset ddibn --gpu 0 --seed 42

# Novel drug condition (Unseen drugs' biological relations removed)
CASE3=1 python train.py --cell 09 --dataset ddibn --gpu 0 --seed 42
```

**DDI network (R-GCN), cell 11** — the paper reports this cell initialized
with a 1,024-bit Morgan fingerprint (4.66M parameters). This is the
*default* for cell 11 (`NODE_INIT` defaults to `molfp` there, unlike cells
09/10 where it defaults to `transe`):

```bash
python train.py --cell 11 --dataset ddibn --gpu 0 --seed 42
# equivalent to: NODE_INIT=molfp python train.py --cell 11 ...

# the earlier TransE-initialized version of this cell (not used in the
# paper) is reproduced with:
NODE_INIT=transe python train.py --cell 11 --dataset ddibn --gpu 0 --seed 42
```

**Appendix J** (crossing the Existing/Novel condition of the TransE
embedding independently from that of the KG subgraph, for Knowledge graph
(R-GCN)/(Graph Transformer), cells 09/10) uses two more env vars that
override `CASE3` when set:

```bash
# e.g. Novel subgraph + Existing TransE embedding
EMB_SRC=case1 SUBGRAPH_SRC=case3 python train.py --cell 09 --dataset ddibn --gpu 0 --seed 42
```

`NODE_INIT=molgraph` (sum-pooled raw atom features instead of a fingerprint
or TransE) is also available for cells 09/10/11 as a further robustness
check; it is not used for any number reported in the paper.

See `ddi334/code/` for dataset build (`build_dataset.py`, `build_kg.py`,
`build_bio_profile.py`, ...), LLM prompt rendering/inference
(`render_prompts.py`, `llm_infer.py`),
fine-tuning (`ft_train.py`, `run_ft_cv.py`), and the batch
launchers used for the ablations above (`oneoff/run_rel_case3.py`,
`oneoff/run_rel_cross.py`, `oneoff/run_cell11_cross.py`).

## Environment

Three separate environments were used (PyG/DGL/vLLM version conflicts
made a single environment impractical): `DDIBench` (review-encoder
training), `dglke` (KGE training), `vllm-llm` (LLM zero-shot/fine-tune).
See [`environment/README.md`](environment/README.md) for exact
versions and setup instructions.

## License

MIT — see [LICENSE](LICENSE).
