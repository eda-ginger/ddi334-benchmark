# Reproducing a result

No batch launcher — each row below is one self-contained command, repeated
across the listed seeds/models/conditions. Run from `ddi334/code`.

**Molecule cells (01-06)** need `data/molecule/` built first, once:
`build/molecule/smiles.py` -> `fingerprint.py` -> `mol_graph.py`.

**LLM fine-tuning (13-18)** needs `data/prompt/` built first, once:
`build/document/instruction.py render` (not shipped — rendered prompts
are ~500MB). Zero-shot (`zeroshot.py`) and `finetuning_eval.py` render
prompts on the fly and don't need this step.

| Results folder | Cells | Command | Notes |
|---|---|---|---|
| `main_experiments/01-12_encoders/` | 01-12 | `train/train.py --cell NN --gpu 0 --seed S` | seeds 0/42/124. No `--split` or `--checkpoint_metric` flag — fixed per Appendix E.1 |
| — Existing/Novel drug condition | 09,10,11 | add `NOVEL_DRUG=1` | only these 3 cells differ; everything else is `condition=single` |
| `appendix/h_embedding_subgraph_cross/` | 10 | `EMB_SRC=... SUBGRAPH_SRC=existing_drug\|novel_drug` | Table H.1 |
| `appendix/f_kge_comparison/` | 09-11 | `KGE_TYPE=transe\|rotate\|complex` | only TransE embeddings are shipped (RotatE/ComplEx too large); `novel_drug_embedding_only/transe/` needs `ABLNR=1` (Figure F.1's "TransE Existing drug" point) |
| `appendix/c_document_backbone_comparison/` | 12 | `DOC_MODEL=scibert\|biobert\|pubmedbert` | build each backbone once: `build/document/mention_description.py --tag <name>` |
| `main_experiments/13-18_llm/zeroshot/` | 13-18 | `llm/zeroshot.py --model <hf_id> --split S0\|S1\|S2 --mode existing_drug\|novel_drug` | 6 models x 2 conditions x 3 splits, temperature=0 (deterministic, no seed sweep); add `--merge-system` for the 2 gemma models (vLLM's gemma-2 chat template rejects a separate system role) |
| `main_experiments/13-18_llm/finetuning/` | 13-18 | `llm/finetuning_train.py --model <hf_id> --mode ... --epochs 3` then `llm/finetuning_eval.py --tag <tag> ... --merge-system` (gemma models only; `finetuning_train.py` auto-detects, no flag needed there) | eval picks the best epoch per split by validation AUROC; only that checkpoint's metrics + the per-epoch `_cv.json` curve are shipped |
| `main_experiments/commercial_llm/` | Figure 7, Appendix I | no command — manual, one-off | commercial models were queried through their web chat UI (not the API), on a fixed 30-question subset of the S2 test set (`web_manual_questions_subset30.json`, ground truth included). `web_manual_questions_{gpt,claude,gemini}_subset30.txt` are the raw transcripts; `summary.csv` is the scored accuracy/F1 per model/condition. The open-source rows in `summary.csv` are the same zero-shot models from `13-18_llm/zeroshot/`, re-scored on this same 30-question subset for a matched comparison |

Example:
```bash
cd ddi334/code
python train/train.py --cell 01 --gpu 0 --seed 42
python train/train.py --cell 10 --gpu 0 --seed 42 --result_dir ../results/main_experiments/01-12_encoders
NOVEL_DRUG=1 python train/train.py --cell 10 --gpu 0 --seed 42
python llm/zeroshot.py --model google/gemma-2-2b-it --gpu 0 --split S0 --mode existing_drug --merge-system
```

Reproducing the shipped `main_experiments/01-12_encoders/` layout exactly
(Existing and Novel runs merged into one folder, with an `_existing`/
`_novel`/`_single` filename suffix and `condition` column) is a manual
relabeling step — `train.py` itself has no "condition" concept in its own
output naming. Running Existing/Novel into two separate `--result_dir`s is
equivalent if you don't need the identical layout.

See `ddi334/code/build/` (one script per Table 2 input representation,
grouped by modality like `data/`) for how every precomputed input was
built, and `ddi334/code/llm/` for the zero-shot/fine-tuning runners.
