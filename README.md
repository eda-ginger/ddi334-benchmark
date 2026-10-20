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
  code/              main trainer (`train.py`, cells 01-15 excluding
                     05/12, two Jaccard+PCA variants that were tried
                     but are not reported in the paper), the shared
                     model/dataset modules it imports (`encoders.py`,
                     `ddi_model.py`, `dataset.py`), and all
                     DDI-334-specific scripts (dataset build, KG build,
                     LLM prompt rendering/inference, fine-tuning, ...).
                     See below.
  data/              train/valid/test split files (S0/S1/S2), KG
                     triples, cluster-split/id-map/type-name metadata,
                     and the precomputed encoder inputs under
                     `ddibn/precompute/`
  results/           summary metrics (per-class scores, configs used,
                     loss curves, LLM ZS/FT metrics, KGE comparison
                     results, commercial-LLM comparison)
  DATA_SOURCES.md    upstream data sources + build pipeline
environment/
  README.md, *_requirements.txt   exact pip environments used
```

`ddi334/results/` is organized by which part of the paper it backs:

```
results/
  encoders_main/                        Table 1/D.1: cells 01-15 excluding
                                         05/12 (see code/ note above),
                                         Existing drug condition (the
                                         default for every cell;
                                         Molecule/Bio-profile cells have
                                         no Novel-drug variant)
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

# example: cell04 (SchNet), DDI-334 (ddibn), seed 42
python train.py --cell 04 --dataset ddibn --gpu 0 --seed 42
```

`--checkpoint_metric` is not a flag here — checkpoint selection (accuracy,
per-split, patience=10 early stopping, as described in the paper's
Appendix E.1) is fixed in `train.py` and is not configurable. One call of
`train.py` trains once and evaluates all three splits (S0/S1/S2) itself —
there is no `--split` flag.

**`results/encoders_main/`** (Table 1/D.1): the command above, repeated
over every reported cell and seed:

```bash
for cell in 01 02 03 04 06 07 08 09 10 11 13 14 15; do
  for seed in 0 42 124; do
    python train.py --cell $cell --dataset ddibn --gpu 0 --seed $seed \
      --result_dir ../results/encoders_main
  done
done
```

**Existing / Novel drug conditions** (biological-relation availability for
Unseen drugs, Section 4 of the paper) are selected with env vars, not CLI
flags:

```bash
# Existing drug condition (default)
python train.py --cell 09 --dataset ddibn --gpu 0 --seed 42

# Novel drug condition (Unseen drugs' biological relations removed)
NOVEL_DRUG=1 python train.py --cell 09 --dataset ddibn --gpu 0 --seed 42
```

**`results/encoders_novel_drug/`** (Section 5.2 figure): cells 08-11 under
`NOVEL_DRUG=1`, 3 seeds each (cell 08 additionally needs `KGE_TYPE=transe`
and writes to `results/kge_comparison/novel_drug/transe/` instead — see
the KGE comparison note below):

```bash
for cell in 09 10 11; do
  for seed in 0 42 124; do
    NOVEL_DRUG=1 python train.py --cell $cell --dataset ddibn --gpu 0 --seed $seed \
      --result_dir ../results/encoders_novel_drug
  done
done
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

`results/ddi_network_molgraph_init/` (not used for any number in the
paper) is the same cell with `NODE_INIT=molgraph` (sum-pooled raw atom
features instead of a fingerprint or TransE), 3 seeds.

**`results/appendix_j_embedding_subgraph_cross/`** (Appendix J: crossing
the Existing/Novel condition of the TransE embedding independently from
that of the KG subgraph, Knowledge graph (R-GCN), cell 09 only) uses two
more env vars that override `NOVEL_DRUG` when set, 3 seeds each:

```bash
# novel_embedding_existing_subgraph/
EMB_SRC=novel_drug SUBGRAPH_SRC=existing_drug python train.py --cell 09 --dataset ddibn --gpu 0 --seed 42
# existing_embedding_novel_subgraph/
EMB_SRC=existing_drug SUBGRAPH_SRC=novel_drug python train.py --cell 09 --dataset ddibn --gpu 0 --seed 42
```

**KGE comparison (Appendix)**: only the TransE entity embeddings are
included in this repository (for both drug conditions), since they are
what `cell08/09/10/11` use by default and what cell 11's and the
Appendix-J cross-evaluation's reported numbers were produced with. The
RotatE/ComplEx/DistMult/TransH comparison numbers in
`results/kge_comparison/` are the exact values used for the paper's KGE
comparison figure, but retraining those four KGE types from this
repository (`KGE_TYPE=rotate ...`) is not supported since their trained
embeddings are not included (RotatE's in particular exceeds GitHub's
100MB file-size limit). `results/kge_comparison/{existing_drug,novel_drug}/transe/`
is reproduced the same way as `encoders_main`/`encoders_novel_drug` cell 08
above; `novel_drug_embedding_only/` additionally needs `ABLNR=1` (reuses the
existing-drug TransE embedding as-is, with only the Novel-drug rows replaced
by a random vector, so no KGE retraining is involved).

**`results/document_backbone_comparison/{biobert,pubmedbert,scibert}/{existing_drug,novel_drug}/`**
(Appendix C.1): cell 13 with `DOC_MODEL` selecting the backbone, 3 seeds
each. Each backbone's embeddings must first be built once with
`build_biobert_template.py --tag {biobert,pubmedbert,scibert}` (reads
`data/bio_profile_llm.json`, writes into `data/ddibn/precompute/`):

```bash
DOC_MODEL=pubmedbert DOC_INPUT=novel_drug python train.py --cell 13 --dataset ddibn --gpu 0 --seed 42 \
  --result_dir ../results/document_backbone_comparison/pubmedbert/novel_drug
```

**`results/llm/zeroshot/`** (Section 5): `llm_infer.py` for each of the 6
models (`google/gemma-2-2b-it`, `Qwen/Qwen2.5-3B-Instruct`,
`microsoft/Phi-3.5-mini-instruct`, `microsoft/phi-4`,
`Qwen/Qwen2.5-14B-Instruct`, `google/gemma-2-27b-it`) x 2 drug conditions
x 3 splits (temperature=0, seed hardcoded, so one call per split/mode/model
is deterministic and sufficient — no repeated seeds):

```bash
python llm_infer.py --version v2 --dataset ddibn --model google/gemma-2-2b-it \
  --gpu 0 --split S0 --mode existing_drug --outdir ../results/llm/zeroshot
```

`results/llm/zeroshot_s2_reproducibility_check/` is the same command for
the S2 split only, re-run into a separate output directory as a
determinism check (see `Reproducibility` discussion in the paper).

**`results/llm/finetuning/`** (Section 5): QLoRA fine-tune (`ft_train.py`,
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
