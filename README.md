# DDI-334 Benchmark

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
ddi334/
  code/              see code/README.md for the cell-numbering table
    build/             data precompute, one script per Table 2 input
                       representation — splits/ molecule/ relation/ document/
    train/             cells 01-12 trainer (train.py, encoders.py,
                       ddi_model.py, dataset.py)
    llm/               cells 13-18 zero-shot/fine-tuning runners
                       (zeroshot.py, finetuning_train.py, finetuning_eval.py)
  data/                precomputed inputs — splits/ molecule/ relation/ document/
  results/           see results/README.md for the folder-to-figure/table map
  DATA_SOURCES.md      upstream data sources + build pipeline
environment/
  README.md, *_requirements.txt
```

### Paper correspondence

- Cell numbers (01-18) vs. the manuscript's **Table 2**:
  [`ddi334/code/README.md`](ddi334/code/README.md)
- Results folder vs. paper figure/table:
  [`ddi334/results/README.md`](ddi334/results/README.md)

Filenames/columns use `existing_drug`/`novel_drug` and `S0`/`S1`/`S2`.
See [REPRODUCING.md](REPRODUCING.md) for exactly how each folder maps
to a command.

## What is / isn't included

**Included**: everything needed to verify every number, table, and
figure in the paper — summary metrics, exact run configs, loss curves,
and the S0/S1/S2 split membership files.

**Not included** (regenerable with `ddi334/code/build/`, not shipped due
to size): model checkpoints/LoRA adapters, per-sample raw prediction
arrays, precomputed embedding caches (SchNet, ChemBERTa/BioBERT, Morgan
fingerprints, TransE/KGE, 2D graphs, rendered LLM prompts), training/
inference logs.

## Data

The DDI-334 splits and KG triples in `ddi334/data/` are the exact
files used for the paper's results. The upstream raw sources
(DDI-Bench/EmerGNN, TDC TWOSIDES, DrugBank, HetioNet) are third-party
and are not redistributed here.

See [`ddi334/DATA_SOURCES.md`](ddi334/DATA_SOURCES.md) for exact sources,
the required sibling-folder layout, and the build pipeline.

## Reproducing a result

```bash
cd ddi334/code
python train/train.py --cell 01 --gpu 0 --seed 42   # example: SMILES 1D-CNN, seed 42
```

See [REPRODUCING.md](REPRODUCING.md) for the per-cell/per-appendix
command table (Existing/Novel drug, KGE comparison, LLM zero-shot/
fine-tuning, etc.).

## Environment

Three separate environments were used (PyG/DGL/vLLM version conflicts
made a single environment impractical): `DDIBench` (review-encoder
training), `dglke` (KGE training), `vllm-llm` (LLM zero-shot/fine-tune).
See [`environment/README.md`](environment/README.md) for exact
versions and setup instructions.

## License

MIT — see [LICENSE](LICENSE).
