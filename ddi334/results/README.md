# ddi334/results

Summary metrics, run configs, and loss curves -- organized by which
part of the manuscript each folder backs.

| Folder | Paper reference |
|---|---|
| `main_experiments/01-12_encoders/` | Figure 6, Table D.1, Appendix G (AUPRC/AP@50 are extra columns, not separate runs) |
| `main_experiments/13-18_llm/{zeroshot,finetuning}/` | Figure 6, Instruction #13-18 |
| `main_experiments/commercial_llm/` | Figure 7, Appendix I (Tables I.1, I.2) |
| `main_experiments/random_baseline.json` | Figure 6 dashed line |
| `appendix/c_document_backbone_comparison/{biobert,pubmedbert,scibert}/` | Appendix C.4, Table C.1 |
| `appendix/f_kge_comparison/` | Appendix F, Figure F.1 |
| `appendix/h_embedding_subgraph_cross/` | Appendix H, Table H.1 |

Filenames/columns use `existing_drug`/`novel_drug` and `S0`/`S1`/`S2`.

See [`REPRODUCING.md`](../../REPRODUCING.md) for exactly how each
folder maps to a command, and [`../code/README.md`](../code/README.md)
for the cell numbering (01-18) referenced above.
