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
ddi334/
  code/
    build/           dataset/KG/precompute build scripts, grouped by modality
                     like `data/` itself:
                       splits/    `dataset.py`
                       molecule/  `smiles.py`, `fingerprint.py`,
                                  `mol_graph.py`
                       relation/  `knowledge_graph.py`, `ddi_network.py`,
                                  `bio_profile.py`
                       document/  `mention_description.py`, `instruction.py`
                                  (the latter used by `llm/`, see below)
    train/           main trainer (`train.py`, cells 01-12 — numbered to
                     match the paper's Table 2 exactly, see "Paper
                     correspondence" below) and the shared model/dataset
                     modules it imports (`encoders.py`, `ddi_model.py`,
                     `dataset.py`)
    llm/             zero-shot inference (`zeroshot.py`) and fine-tuning
                     (`finetuning_train.py`, `finetuning_eval.py`) for
                     cells 13-18 — both built on `build/document/
                     instruction.py`'s prompts (Table 2's "Instruction" row)
  data/              organized by modality:
                       splits/    train/valid/test split files (S0/S1/S2),
                                  cluster-split/id-map/type-name/drug-name
                                  metadata
                       molecule/  SMILES, SchNet coordinates, ChemBERTa embeddings
                       relation/  KG triples, DDI network graph (`ddi_network.pt`),
                                  bio profile vector (`bio_profile.pt`), KGE
                                  embeddings (`relation/kge/`)
                       document/  BioBERT/PubMedBERT/SciBERT embeddings,
                                  bio profile text source (`bio_profile_text.json`)
  results/           summary metrics (per-class scores, configs used,
                     loss curves, LLM ZS/FT metrics, KGE comparison
                     results, commercial-LLM comparison)
  DATA_SOURCES.md    upstream data sources + build pipeline
environment/
  README.md, *_requirements.txt   exact pip environments used
```

### Paper correspondence

`train/train.py --cell` numbers are the same 01-12 as the manuscript's **Table 2**
(not an internal numbering scheme that needs translating):

| # | Modality | Input | Model |
|---|---|---|---|
| 01 | Molecule | SMILES | 1D-CNN |
| 02 | Molecule | SMILES | ChemBERTa (Frozen) |
| 03 | Molecule | Fingerprint | MLP |
| 04 | Molecule | Molecular graph (2D) | GAT |
| 05 | Molecule | Molecular graph (2D) | Graph Transformer |
| 06 | Molecule | Molecular graph (3D) | SchNet |
| 07 | Relation | Bio profile | MLP |
| 08 | Relation | DDI network (Whole graph) | R-GCN |
| 09 | Relation | Knowledge graph (Whole graph) | KGE (Frozen) |
| 10 | Relation | Knowledge graph (Subgraph) | R-GCN |
| 11 | Relation | Knowledge graph (Subgraph) | Graph Transformer |
| 12 | Document | Mention/Description | SciBERT (Frozen) |

The paper's Instruction (LLM) configurations, Table 2 #13-18, are not
renumbered the same way — `results/main_experiments/13-18_llm/` and
`results/commercial_llm/` already label every file by the actual HF model
id or a short nickname (`google/gemma-2-2b-it`, `qwen3b`, ...), which
identifies the model more directly than a number would:

| # | Model |
|---|---|
| 13 | Gemma2-2B |
| 14 | Gemma2-27B |
| 15 | Qwen2.5-3B |
| 16 | Qwen2.5-14B |
| 17 | Phi-3.5-mini |
| 18 | Phi-4 |

`ddi334/results/` is organized by which part of the paper it backs:

| Folder | Paper reference |
|---|---|
| `main_experiments/01-12_encoders/` | Figure 6, Table D.1, Appendix G (AUPRC/AP@50 are extra columns, not separate runs) |
| `main_experiments/13-18_llm/{zeroshot,finetuning}/` | Figure 6, Instruction #13-18 |
| `main_experiments/commercial_llm/` | Figure 7, Appendix I (Tables I.1, I.2) |
| `main_experiments/random_baseline.json` | Figure 6 dashed line |
| `appendix/c_document_backbone_comparison/{biobert,pubmedbert,scibert}/` | Appendix C.4, Table C.1 |
| `appendix/f_kge_comparison/` | Appendix F, Figure F.1 |
| `appendix/h_embedding_subgraph_cross/` | Appendix H, Table H.1 |

Filenames/columns use `existing_drug`/`novel_drug` (or `existing`/`novel`)
for the Existing/Novel drug condition, and `S0`/`S1`/`S2` for
Seen-Seen/Seen-Unseen/Unseen-Unseen. See [REPRODUCING.md](REPRODUCING.md)
for exactly how each folder maps to a command.

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
