# DDI-334 KGE 학습 결과 기록

REL 셀(08 TransE / 09 RGCN-sub / 10 GT-KG / 11 RGCN-whole)의 노드 입력으로 쓰는
KGE 엔티티 임베딩 학습 기록 (ddibn 데이터셋).

## KG 구성
- **bio KG**: `kg_bio_334.txt` = HetioNet bio 엣지 중 334 약물만 닿는 것, **1,547,425 edges**, rel 0-22 (23종)
- **TWOSIDES train (typed)**: `existing_drug/ddi_train.txt` — **130,233 edges**, rel 23+ (type_idx)
- **DrugBank DDI 없음** (cold-drug 누수 방지)
- 관계 재번호: `existing_drug/relation_map.json` (bio 0-22, TWOSIDES 23+)
- n_relations: **232** | entity 공간: **34124** (drug db_id + bio HetioNet id)

## HP (AstraZeneca / kgem-in-drug-discovery, 전 모델 공통)
- optimizer Adagrad lr=0.02 | loss MarginRankingLoss | num_epoch 500 | num_negative 61
- create_inverse=False | seed=42
- embedding_dim: **TransE 304 / RotatE 512 / DistMult 80 / ComplEx 272 / TransH 480**

## 결과 표

이 표는 TransE 학습 직후 작성된 것이라 RotatE/DistMult/ComplEx/TransH는 "대기"로 남아있지만,
실제로는 RotatE/ComplEx까지 학습되어 Appendix F(링크 복원 vs 후속 DDI 예측 비교)에 보고되었다.
DistMult/TransH는 원고에 전혀 보고되지 않아 `results/kge_comparison/`과 `data/relation/kge/
existing_drug/`에서 제거되었다 (업데이트 당시 미반영, 2026-10-21).

| 데이터셋 | 모델 | 상태 | loss(초→말) | adj. MRR idx | geo. rank idx | hits@10 | 임베딩 건강도 | 산출물 |
|---|---|---|---|---|---|---|---|---|
| ddibn | TransE | ✅ 완료 (2026-06-16, ~4h, GPU0) | 0.241→0.044 | 0.906 | 0.986 | 0.192 | (34124,304) finite, 단위norm, 334약물행 all-nonzero, 약물쌍 cos 0.21±0.12 (collapse X) | `existing_drug/transe/hetionet_transe_ent_304d.npy` |
| ddibn | RotatE | ✅ 완료 (Appendix F에 보고) | | | | | | `results/kge_comparison/existing_drug/rotate/` |
| ddibn | ComplEx | ✅ 완료 (Appendix F에 보고) | | | | | | `results/kge_comparison/existing_drug/complex/` |

## 판정 기준 (건강도)
- **loss 수렴**: 말기 loss가 plateau (TransE ddibn 0.044)
- **adj. MRR index / geo. rank index**: 1에 가까울수록 랜덤 대비 우수 (절대 hits@1은 34k entity·다관계라 낮은 게 정상)
- **임베딩**: finite + collapse 아님(약물쌍 cosine mean이 1에 가깝지 않을 것). TransE는 단위 norm 정규화가 기본이라 행 norm=1.0 / std=0은 정상(collapse 아님).

## 비고
- Cell 08/09/10/11 모두 이 임베딩을 **frozen** 노드 입력으로 사용 (11도 frozen 통일, 2026-06-16).
- `train.py`의 `load_kge_embeddings()`가 `kge/existing_drug/{type}/*.npy`를 직접 읽는다 (별도 복사 단계 없음).
