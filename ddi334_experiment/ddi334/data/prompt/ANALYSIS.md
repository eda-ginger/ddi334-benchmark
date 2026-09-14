# data/prompt 분석 보고서 (V2 Ideal/Real, ddibn)

LLM 추론 전 사전 분석. 프롬프트=`build_v2(mode)` 렌더본, system=`_system.txt`(상수).
3개 모드(정보 가용성 ablation):
- **real**: SMILES만
- **ideal_genes**: Name + SMILES + Target genes
- **ideal**: Name + SMILES + Target genes + Indications + Pharmacologic class (5필드 동일포맷, 결측 `(none)`)
→ ideal vs ideal_genes = indications+class 순효과 / ideal_genes vs real = genes 순효과.

## 1. 규모 · 라벨
| split | (쌍,타입) | y=1 | y=0 |
|---|---|---|---|
| train | 260,466 | 130,233 | 130,233 |
| valid_S0 | 32,580 | 16,290 | 16,290 |
| test_S0 | 34,018 | 17,009 | 17,009 |
| valid_S1 | 83,614 | 41,807 | 41,807 |
| test_S1 | 86,742 | 43,371 | 43,371 |
| valid_S2 | 10,930 | 5,465 | 5,465 |
| test_S2 | 11,176 | 5,588 | 5,588 |

총 519,526 × (ideal/real). **양/음 1:1**, 빈 프롬프트 0.

## 2. 프롬프트 길이 (토큰, Qwen2.5, test_S0)
| mode | 중앙 | 90%ile | 최대 |
|---|---|---|---|
| real | 113 | 185 | 289 |
| ideal_genes | 231 | 344 | 546 |
| ideal | 278 | 394 | 597 |

## 3. (none) 결측 — 항목별

### 3-1. 약물 단위 (334개)
| 필드 | 보유 약물 | **(none) 약물** | 보유시 중앙 개수 |
|---|---|---|---|
| Target genes | 286 | **48 (14%)** | 8 |
| Indications | 151 | **183 (55%)** | 1 |
| Pharmacologic class | 172 | **162 (49%)** | 1 |

- 3종 **전부 (none)** 약물: 47/334 (14%) (무기물/단순분자 등) → Ideal ≈ Real+이름.

### 3-2. 쌍 단위 — 둘다보유 / 한쪽만 none / 둘다 none (test_S0)
| 필드 | 둘다보유 | 한쪽만 none | 둘다 none |
|---|---|---|---|
| Target genes | 71% | 26% | **3%** |
| Indications | 17% | 49% | **34%** |
| Pharmacologic class | 23% | 50% | **27%** |
*(S1/S2 거의 동일: genes 둘다none 1~3%, indications 둘다none ~30%)*

### 3-3. 결측 근본 원인 (실증)
- 우리가 쓰는 3종(Gene/Disease/Pharm Class) 엔티티 **이름 매핑 100% (실패 0)** → (none)은 전부 **KG 엣지 부재**.
- indications=(none) 약물 = KG 내 Disease 엣지 **0개** (보유 약물은 엣지수=indications수 일치).
- HetioNet 약물-비약물 엣지 분포: Gene **11,408** vs Disease **342** / Pharm Class **250** (Side Effect 36,849은 누수로 제외) → disease/class 연결이 원래 희박한 게 결측 원인.

## 4. LLM 입력 사전 분석 포인트
- **Ideal vs Real = 정보 가용성 단일변수** (system·질문 동일, user만 다름).
- **Target genes**(직접표적/binds)에 CYP 대사효소 포함 → DDI(CYP매개) 핵심 신호. (none) 14%로 커버 높음.
- **Indications·Pharmacologic class는 결측 ~50%**(쌍의 ~30%가 둘다 none) → 실질 Ideal 신호는 SMILES+genes 중심, indications/class는 있을 때만 보너스.
- **누수 차단**: Side Effect(=라벨 어휘) 제외, gene은 직접표적만(간접 제외).
- **질문 익명화**(Drug 1/Drug 2): Real 이름부재와 일관, 약물명 암기 차단.
