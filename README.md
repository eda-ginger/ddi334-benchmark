# [TODO: 논문 제목 / 저장소 이름 확정 필요]

Code and results for the DDI-334 experiments in:

> [TODO: 논문 인용 정보 (저자, 저널, 연도, DOI) — 투고/억셉 후 확정]

This repository contains the encoder/LLM experiment code, experiment
design artifacts, and result summaries for the DDI-334 dataset (334
drugs, TWOSIDES x DrugBank overlap) reported in the paper. It does not
include the full experiment history of the project — only the code and
results that produced the numbers, tables, and figures in the final
manuscript.

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
    code/          DDI-334-specific scripts (dataset build, KG build,
                   LLM prompt rendering/inference, result aggregation, ...)
    data/          train/valid/test split files (S0/S1/S2), drug/type
                   maps, KG triples used for the Relation modality
    meta/          cluster split definition, id maps, type names
    results/       summary metrics (per-class scores, configs used,
                   loss curves, LLM ZS/FT metrics, KGE comparison
                   results, commercial-LLM comparison)
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
  jsonl files) — these are derived from the raw external data below
  and can be regenerated with the scripts in `ddi334/code/`
- Training/inference logs

## Data

The DDI-334 splits and KG triples in `ddi334/data/` are the exact
files used for the paper's results. The upstream raw sources
(DrugBank, DRKG, TWOSIDES) are third-party and are not redistributed
here.

[TODO: 원본 데이터 다운로드/전처리 절차 문서화 — 어떤 소스에서 어떤 스크립트로
`ddi334/data/`가 만들어지는지, 정확한 순서를 다음 세션에서 정리]

## Reproducing a result

```bash
# example: cell04 (SchNet), DDI-334 (ddibn), seed 42
python train.py --cell 04 --dataset ddibn --gpu 0 --seed 42
```

See `ddi334/code/` for dataset build (`build_dataset.py`, `build_kg.py`,
`build_bio_profile.py`, ...), LLM prompt rendering/inference
(`render_prompts.py`, `llm_infer.py`, `llm_infer_commercial.py`), and
fine-tuning (`ft_train_v1.py`, `ft_train_v2.py`, `run_ft_cv_v2.py`).

## Environment

[TODO: conda/pip 환경 명세 정리 필요]

## License

[TODO: 라이선스 결정 필요]
