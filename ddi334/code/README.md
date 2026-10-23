# ddi334/code

```
build/             data precompute, one script per Table 2 input
                    representation -- splits/ molecule/ relation/ document/
train/              cells 01-12 trainer (train.py, encoders.py,
                    ddi_model.py, dataset.py)
llm/                cells 13-18 zero-shot/fine-tuning runners
                    (zeroshot.py, finetuning_train.py, finetuning_eval.py)
```

## Cell numbering = manuscript Table 2

`train/train.py --cell` and the LLM scripts' model choice are the same
01-18 as the manuscript's **Table 2**:

| # | Modality | Input Representation | Input Usage | Model |
|---|---|---|---|---|
| 01 | Molecule | SMILES | Direct | 1D-CNN |
| 02 | Molecule | SMILES | Direct | ChemBERTa (Frozen) |
| 03 | Molecule | Fingerprint | Direct | MLP |
| 04 | Molecule | Molecular graph | 2D molecular graph | GAT |
| 05 | Molecule | Molecular graph | 2D molecular graph | Graph Transformer |
| 06 | Molecule | Molecular graph | 3D molecular graph | SchNet |
| 07 | Relation | Bio profile | Direct | MLP |
| 08 | Relation | DDI network | Whole Graph | R-GCN |
| 09 | Relation | Knowledge graph | Whole Graph | KGE (Frozen) |
| 10 | Relation | Knowledge graph | Subgraph | R-GCN |
| 11 | Relation | Knowledge graph | Subgraph | Graph Transformer |
| 12 | Document | Mention / Description | User Only | SciBERT (Frozen) |
| 13 | Document | Instruction | System + User + Question | Gemma2-2B (Zero-shot / Fine-tuning) |
| 14 | Document | Instruction | System + User + Question | Gemma2-27B (Zero-shot / Fine-tuning) |
| 15 | Document | Instruction | System + User + Question | Qwen2.5-3B (Zero-shot / Fine-tuning) |
| 16 | Document | Instruction | System + User + Question | Qwen2.5-14B (Zero-shot / Fine-tuning) |
| 17 | Document | Instruction | System + User + Question | Phi-3.5-mini (Zero-shot / Fine-tuning) |
| 18 | Document | Instruction | System + User + Question | Phi-4 (Zero-shot / Fine-tuning) |

See [`REPRODUCING.md`](../../REPRODUCING.md) for the exact command per
cell/appendix, and [`../DATA_SOURCES.md`](../DATA_SOURCES.md) for how
each cell's input data was built.
