# DDI-334 Benchmark

**Repository**: https://github.com/eda-ginger/ddi334-benchmark

Code and results for the DDI-334 experiments reported in:

> [TODO: 논문 인용 정보 (저자, 저널, 연도, DOI) — 투고/억셉 후 확정]

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
  configs/         top-level experiment configs
  train.py         main trainer for the 14 review-encoder cells (+ cell15)
  ft_train.py      fine-tuning entry point
  zs_inference.py  zero-shot LLM inference entry point
  ddi334/
    code/              DDI-334-specific scripts (dataset build, KG build,
                       LLM prompt rendering/inference, result aggregation, ...)
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
  BioBERT embeddings, TransE/KGE embeddings, rendered LLM prompt
  jsonl files) — derived from the raw external data below and
  regenerable with the scripts in `ddi334/code/`
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
cd ddi334_experiment

# example: cell04 (SchNet), DDI-334 (ddibn), seed 42
python train.py --cell 04 --dataset ddibn --gpu 0 --seed 42
```

See `ddi334/code/` for dataset build (`build_dataset.py`, `build_kg.py`,
`build_bio_profile.py`, ...), LLM prompt rendering/inference
(`render_prompts.py`, `llm_infer.py`, `llm_infer_commercial.py`), and
fine-tuning (`ft_train_v1.py`, `ft_train_v2.py`, `run_ft_cv_v2.py`).

## Environment

Three separate environments were used (PyG/DGL/vLLM version conflicts
made a single environment impractical): `ddibench` (review-encoder
training), `dglke` (KGE training), `vllm-llm` (LLM zero-shot/fine-tune).
See [`environment/README.md`](environment/README.md) for exact
versions and setup instructions.

## License

MIT — see [LICENSE](LICENSE).
