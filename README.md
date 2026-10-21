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
  code/              main trainer (`train.py`, cells 01-12 — numbered to
                     match the paper's Table 2 exactly, see "Paper
                     correspondence" below), the shared model/dataset
                     modules it imports (`encoders.py`, `ddi_model.py`,
                     `dataset.py`), and all DDI-334-specific scripts
                     (dataset build, KG build, LLM prompt
                     rendering/inference, fine-tuning, ...). See below.
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

`train.py --cell` numbers are the same 01-12 as the manuscript's **Table 2**
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

(Two configurations tried during development — Jaccard+PCA-reduced Molecule
fingerprint/Relation bio-profile variants, and a drug-name-only Document
cell superseded by cell 12's existing_drug/novel_drug tiers — are not
reported in the paper and are not in this repository at all, so the
numbering above has no gaps.)

The paper's Instruction (LLM) configurations, Table 2 #13-18, are not
renumbered the same way — `results/llm/` and `results/commercial_llm/`
already label every file by the actual HF model id or a short nickname
(`google/gemma-2-2b-it`, `qwen3b`, ...), which identifies the model more
directly than a number would:

| # | Model |
|---|---|
| 13 | Gemma2-2B |
| 14 | Gemma2-27B |
| 15 | Qwen2.5-3B |
| 16 | Qwen2.5-14B |
| 17 | Phi-3.5-mini |
| 18 | Phi-4 |

`ddi334/results/` is organized by which part of the paper it backs:

```
results/
  encoders_main/                        Figure 6 (Existing drug / "Single
                                         setting" markers) + Table D.1:
                                         cells 01-12, Existing drug
                                         condition (the default for every
                                         cell; Molecule/Bio-profile cells
                                         have no Novel-drug variant).
                                         Cell 12 here is SciBERT +
                                         existing_drug (DOC_MODEL/DOC_INPUT
                                         defaults), the same run as
                                         document_backbone_comparison/
                                         scibert/existing_drug/ below
  encoders_novel_drug/                  Figure 6 (Novel drug / open-circle
                                         markers) + Section 5.2: cells
                                         08, 10, 11 under the Novel drug
                                         condition (cell 09's Novel-drug
                                         run lives in kge_comparison/
                                         novel_drug/transe/ instead, see
                                         below)
  appendix_h_embedding_subgraph_cross/  Appendix H (Table H.1): TransE
    novel_embedding_existing_subgraph/    embedding and KG subgraph
    existing_embedding_novel_subgraph/    crossed independently for
                                           Knowledge graph (subgraph R-GCN,
                                           cell 10)
  document_backbone_comparison/         Appendix C.4 (Table C.1): BioBERT
    {biobert,pubmedbert,scibert}/         vs PubMedBERT vs SciBERT, each
      {existing_drug,novel_drug}/         under both drug conditions
  kge_comparison/                       Appendix F (Figure F.1): TransE/
    existing_drug/                        RotatE/ComplEx link-reconstruction
    novel_drug/                           (Hits@10) vs downstream DDI
    novel_drug_embedding_only/            prediction (AUROC). novel_drug_
                                           embedding_only isolates the
                                           embedding from the
                                           graph-structure effect
  llm/
    finetuning/                         Figure 6 (Instruction "F" markers):
                                         QLoRA FT, 6 models x Existing/
                                         Novel drug condition x 3 epochs
    zeroshot/                           Figure 6 (Instruction "Z" markers):
                                         zero-shot, 6 models x Existing/
                                         Novel
    zeroshot_s2_reproducibility_check/  reproducibility check for the
                                         Unseen-Unseen zero-shot numbers
  ddi_network_molgraph_init/            robustness check for DDI network
                                         (R-GCN, cell 08) with a different
                                         node initialization (not reported
                                         in the paper; see Reproducing a
                                         result below)
  commercial_llm/                       Figure 7 + Appendix I (Tables I.1,
                                         I.2): local vs commercial LLM
                                         comparison
  random_baseline.json                  random-prediction reference line
                                         (Figure 6 dashed line)
```

Appendix G (AUPRC/AP@50/zero-shot response-distribution supplementary
figures) is not a separate results folder — it's the same `test_PR-AUC`/
`test_P@50`/`test_P@N2` columns already in every `summary.csv` above,
recomputed from the same runs as Figure 6/7.

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

See [`ddi334/DATA_SOURCES.md`](ddi334/DATA_SOURCES.md)
for exact sources, the required sibling-folder layout, and the build
pipeline (`build_dataset.py` -> `build_kg.py` -> per-encoder
precomputation).

## Reproducing a result

There is no batch-launcher script in this repository — every command
below is a single, self-contained invocation. The full set of results in
a given `results/` folder is just that command repeated across the listed
cells/models/seeds; how you parallelize that across your own GPUs is up
to you (we used a handful of small internal dispatcher scripts tuned to
our specific shared server, which is why they aren't included here).

```bash
cd ddi334/code

# example: cell01 (SMILES 1D-CNN), seed 42
python train.py --cell 01 --gpu 0 --seed 42
```

`--checkpoint_metric` is not a flag here — checkpoint selection (accuracy,
per-split, patience=10 early stopping, as described in the paper's
Appendix E.1) is fixed in `train.py` and is not configurable. One call of
`train.py` trains once and evaluates all three splits (S0/S1/S2) itself —
there is no `--split` flag.

**`results/encoders_main/`** (Figure 6 + Table D.1): the command above,
repeated over every cell (01-12) and seed:

```bash
for cell in 01 02 03 04 05 06 07 08 09 10 11 12; do
  for seed in 0 42 124; do
    python train.py --cell $cell --gpu 0 --seed $seed \
      --result_dir ../results/encoders_main
  done
done
```

**Existing / Novel drug conditions** (biological-relation availability for
Unseen drugs, Section 4 of the paper) are selected with env vars, not CLI
flags:

```bash
# Existing drug condition (default)
python train.py --cell 10 --gpu 0 --seed 42

# Novel drug condition (Unseen drugs' biological relations removed)
NOVEL_DRUG=1 python train.py --cell 10 --gpu 0 --seed 42
```

**`results/encoders_novel_drug/`** (Section 5.2 figure): cells 08, 10, 11
under `NOVEL_DRUG=1`, 3 seeds each (cell 09's Novel-drug run additionally
needs `KGE_TYPE=transe` and writes to
`results/kge_comparison/novel_drug/transe/` instead — see the KGE
comparison note below):

```bash
for cell in 08 10 11; do
  for seed in 0 42 124; do
    NOVEL_DRUG=1 python train.py --cell $cell --gpu 0 --seed $seed \
      --result_dir ../results/encoders_novel_drug
  done
done
```

**DDI network (R-GCN), cell 08** — the paper reports this cell initialized
with a 1,024-bit Morgan fingerprint (4.66M parameters). This is the
*default* for cell 08 (`NODE_INIT` defaults to `molfp` there, unlike cells
10/11 where it defaults to `transe`):

```bash
python train.py --cell 08 --gpu 0 --seed 42
# equivalent to: NODE_INIT=molfp python train.py --cell 08 ...

# the earlier TransE-initialized version of this cell (not used in the
# paper) is reproduced with:
NODE_INIT=transe python train.py --cell 08 --gpu 0 --seed 42
```

`results/ddi_network_molgraph_init/` (not used for any number in the
paper) is the same cell with `NODE_INIT=molgraph` (sum-pooled raw atom
features instead of a fingerprint or TransE), 3 seeds.

**`results/appendix_h_embedding_subgraph_cross/`** (Appendix H, Table H.1:
crossing the Existing/Novel condition of the TransE embedding independently
from that of the KG subgraph, Knowledge graph subgraph R-GCN, cell 10 only)
uses two more env vars that override `NOVEL_DRUG` when set, 3 seeds each:

```bash
# novel_embedding_existing_subgraph/
EMB_SRC=novel_drug SUBGRAPH_SRC=existing_drug python train.py --cell 10 --gpu 0 --seed 42
# existing_embedding_novel_subgraph/
EMB_SRC=existing_drug SUBGRAPH_SRC=novel_drug python train.py --cell 10 --gpu 0 --seed 42
```

**KGE comparison (Appendix F)**: only the TransE entity embeddings are
included in this repository (for both drug conditions), since they are
what cells 08-11 use by default and what cell 08's and the Appendix H
cross-evaluation's reported numbers were produced with. The RotatE/ComplEx
comparison numbers in `results/kge_comparison/` are the exact values used
for the paper's Figure F.1, but retraining those two KGE types from this
repository (`KGE_TYPE=rotate ...`) is not supported since their trained
embeddings are not included (RotatE's in particular exceeds GitHub's
100MB file-size limit). DistMult/TransH were trained during development
but are not reported anywhere in the paper, so they are not in this
repository at all. `results/kge_comparison/{existing_drug,novel_drug}/transe/`
is reproduced the same way as `encoders_main`/`encoders_novel_drug` cell 09
above; `novel_drug_embedding_only/` additionally needs `ABLNR=1` (reuses the
existing-drug TransE embedding as-is, with only the Novel-drug rows replaced
by a random vector, so no KGE retraining is involved).

**`results/document_backbone_comparison/{biobert,pubmedbert,scibert}/{existing_drug,novel_drug}/`**
(Appendix C.4, Table C.1): cell 12 with `DOC_MODEL` selecting the backbone,
3 seeds each. `scibert` is the default (`DOC_MODEL`/`DOC_INPUT` need no
override for `existing_drug`) and is also the main-text Figure 6 result
(and Table D.1's parameter count) — (`encoders_main`'s cell 12 above is
this exact run); `biobert`/`pubmedbert` are only used for this Appendix
C.4 comparison. Each backbone's embeddings must first be built once with
`build_biobert_template.py --tag {biobert,pubmedbert,scibert}` (reads
`data/document/bio_profile_text.json`, writes into `data/document/`):

```bash
DOC_MODEL=pubmedbert DOC_INPUT=novel_drug python train.py --cell 12 --gpu 0 --seed 42 \
  --result_dir ../results/document_backbone_comparison/pubmedbert/novel_drug
```

**`results/llm/zeroshot/`** (Figure 6, Instruction #13-18, "Z" markers): `llm_infer.py` for each of the 6
models (`google/gemma-2-2b-it`, `Qwen/Qwen2.5-3B-Instruct`,
`microsoft/Phi-3.5-mini-instruct`, `microsoft/phi-4`,
`Qwen/Qwen2.5-14B-Instruct`, `google/gemma-2-27b-it`) x 2 drug conditions
x 3 splits (temperature=0, seed hardcoded, so one call per split/mode/model
is deterministic and sufficient — no repeated seeds):

```bash
python llm_infer.py --version v2 --model google/gemma-2-2b-it \
  --gpu 0 --split S0 --mode existing_drug --outdir ../results/llm/zeroshot
```

`results/llm/zeroshot_s2_reproducibility_check/` is the same command for
the S2 split only, re-run into a separate output directory as a
determinism check (see `Reproducibility` discussion in the paper).

**`results/llm/finetuning/`** (Figure 6, Instruction #13-18, "F" markers): QLoRA fine-tune (`ft_train.py`,
3 epochs, seed 42) then evaluate (`run_ft_cv.py`, which itself evaluates
all 3 splits and selects the best epoch per split by validation AUROC),
for each of the same 6 models x 2 drug conditions:

```bash
python ft_train.py --model google/gemma-2-2b-it --mode existing_drug --gpu 0 --epochs 3
python run_ft_cv.py --tag google_gemma-2-2b-it_existing_drug_seed42 \
  --model google/gemma-2-2b-it --mode existing_drug \
  --ckpt ../results/ft_checkpoints/google_gemma-2-2b-it_existing_drug_seed42 --gpu 0 --epochs 3
```

See `ddi334/code/` for dataset build (`build_dataset.py`, `build_kg.py`,
`build_bio_profile.py`, ...) and prompt rendering (`llm_prompts.py`).

## Environment

Three separate environments were used (PyG/DGL/vLLM version conflicts
made a single environment impractical): `DDIBench` (review-encoder
training), `dglke` (KGE training), `vllm-llm` (LLM zero-shot/fine-tune).
See [`environment/README.md`](environment/README.md) for exact
versions and setup instructions.

## License

MIT — see [LICENSE](LICENSE).
