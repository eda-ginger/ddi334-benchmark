#!/usr/bin/env python3
"""DDI-334 LLM 프롬프트용 bio profile precompute (Ideal 입력).

각 약물(db_id)의 HetioNet 이웃 중 누수/노이즈 없는 3종만 이름으로 추출:
  - Target genes : Compound-binds-Gene (CbG, kg_bio_334 rel 6) = 직접 결합 표적 (LLMDDI식)
  - Indications  : Compound-treats/palliates-Disease (CtD/CpD, rel 5/10)
  - Pharmacologic class : Compound-(class)-PharmacologicClass
(Side Effect는 예측 라벨과 동일 어휘라 누수 -> 제외. up/down-regulate(간접 gene)도 제외.)

입력: data/kge/kg_bio_334.txt, kge_hetionet/data/entity_drug.json, meta/hetionet_names.json
출력: meta/bio_profile_llm.json  { "<db_id>": {"genes":[..], "indications":[..], "pharm_class":[..]} }
Run: micromamba run -n DDIBench python code/build_bio_profile.py
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ddi334/
EXP = os.path.dirname(HERE)
V5 = os.path.dirname(EXP)
DRUG_MAX = 1710
GENE_REL = '6'          # CbG (binds) = 직접 표적

e2i = json.load(open(os.path.join(V5, 'kge_hetionet', 'data', 'entity_drug.json')))
i2e = {v: k for k, v in e2i.items()}
nm = {int(k): v for k, v in json.load(open(os.path.join(HERE, 'meta', 'hetionet_names.json'))).items()}

genes = defaultdict(list); dis = defaultdict(list); cls = defaultdict(list)
for line in open(os.path.join(HERE, 'data', 'kge', 'kg_bio_334.txt')):
    a, b, r = line.split(); a, b = int(a), int(b)
    for d, e in ((a, b), (b, a)):
        if d < DRUG_MAX and e >= DRUG_MAX and e in nm:
            t = i2e[e].split('::')[0]
            if t == 'Gene' and r == GENE_REL:
                genes[d].append(nm[e])
            elif t == 'Disease':
                dis[d].append(nm[e])
            elif t == 'Pharmacologic Class':
                cls[d].append(nm[e])

drug_map = json.load(open(os.path.join(HERE, 'data', 'ddibn', 'drug_map.json')))
dbids = sorted(set(drug_map.values()))
out = {}
for d in dbids:
    out[str(d)] = {"genes": sorted(genes[d]), "indications": sorted(dis[d]), "pharm_class": sorted(cls[d])}

path = os.path.join(HERE, 'meta', 'bio_profile_llm.json')
json.dump(out, open(path, 'w'), ensure_ascii=False, indent=0)
ng = sum(1 for d in dbids if genes[d]); nd = sum(1 for d in dbids if dis[d]); nc = sum(1 for d in dbids if cls[d])
print(f"[bio_profile] {len(dbids)} 약물 -> {path}")
print(f"  genes 보유 {ng}, indications {nd}, pharm_class {nc}")
