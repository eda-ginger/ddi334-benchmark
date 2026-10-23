# DDI-334 KGE Training Results Log

Training log for the KGE entity embeddings used as node input by the Relation cells
(08 RGCN-whole / 09 TransE / 10 RGCN-sub / 11 GT-KG) (ddibn dataset).

## KG composition
- **bio KG**: `kg_bio_334.txt` = the HetioNet bio edges touching only the 334 drugs, **1,547,425 edges**, rel 0-22 (23 relations)
- **TWOSIDES train (typed)**: `existing_drug/ddi_train.txt` -- **130,233 edges**, rel 23+ (type_idx)
- **No DrugBank DDI** (to prevent cold-drug leakage)
- Relation renumbering: `existing_drug/relation_map.json` (bio 0-22, TWOSIDES 23+)
- n_relations: **232** | entity space: **34124** (drug db_id + bio HetioNet id)

## HP (AstraZeneca / kgem-in-drug-discovery, shared across all models)
- optimizer Adagrad lr=0.02 | loss MarginRankingLoss | num_epoch 500 | num_negative 61
- create_inverse=False | seed=42
- embedding_dim: **TransE 304 / RotatE 512 / DistMult 80 / ComplEx 272 / TransH 480**

## Results table

This table was written right after the TransE training run, so RotatE/DistMult/ComplEx/TransH
are still marked "pending" below, but RotatE and ComplEx were in fact trained afterward and
reported in Appendix F (link recovery vs. downstream DDI prediction comparison). DistMult/TransH
are not reported anywhere in the manuscript, so they were removed from `results/kge_comparison/`
and `data/relation/kge/existing_drug/` (this table wasn't updated to reflect that, 2026-10-21).

| Dataset | Model | Status | loss(start->end) | adj. MRR idx | geo. rank idx | hits@10 | Embedding health | Output |
|---|---|---|---|---|---|---|---|---|
| ddibn | TransE | Done (2026-06-16, ~4h, GPU0) | 0.241->0.044 | 0.906 | 0.986 | 0.192 | (34124,304) finite, unit norm, all 334 drug rows nonzero, drug-pair cos 0.21+/-0.12 (not collapsed) | `existing_drug/transe/hetionet_transe_ent_304d.npy` |
| ddibn | RotatE | Done (reported in Appendix F) | | | | | | `results/kge_comparison/existing_drug/rotate/` |
| ddibn | ComplEx | Done (reported in Appendix F) | | | | | | `results/kge_comparison/existing_drug/complex/` |

## Health criteria
- **loss convergence**: late-stage loss plateaus (TransE ddibn 0.044)
- **adj. MRR index / geo. rank index**: closer to 1 is better relative to random (absolute hits@1 being low is expected given 34k entities and many relations)
- **Embeddings**: finite and not collapsed (drug-pair cosine mean should not be close to 1). TransE normalizes to unit norm by default, so row norm=1.0 / std=0 is normal (not collapse).

## Notes
- Cells 08/09/10/11 all use this embedding as a **frozen** node input (08 was also unified to frozen on 2026-06-16).
- `train.py`'s `load_kge_embeddings()` reads `kge/existing_drug/{type}/*.npy` directly (no separate copy step).
