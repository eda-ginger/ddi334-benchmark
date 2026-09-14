#!/usr/bin/env python3
"""Case-3 KG 전처리: 신약(Dn) 목록 + 신약-고립 bio KG + kg_triples_case3. (GPU 불필요, 재현용)
출력: case3/{novel_drugs.json, kg_bio_334_case3.txt, kg_triples_case3.npy}"""
import numpy as np, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); DD = os.path.join(HERE, "..")
def drugs(fn):
    s=set()
    for line in open(fn):
        p=line.split()
        if len(p)==4: s.add(int(p[0])); s.add(int(p[1]))
    return s
train=drugs(os.path.join(DD,"ddibn/train.txt"))
novel=sorted((drugs(os.path.join(DD,"ddibn/test_S1.txt"))|drugs(os.path.join(DD,"ddibn/test_S2.txt")))-train)
json.dump({"novel_drug_ids":novel,"n_novel":len(novel),"n_train_drugs":len(train),
           "note":"Case-3 신약=cold cluster Dn. entity_id==drug_id."}, open(os.path.join(HERE,"novel_drugs.json"),"w"),indent=1)
nov=set(novel); kept=removed=0
with open(os.path.join(DD,"kge/kg_bio_334.txt")) as f, open(os.path.join(HERE,"kg_bio_334_case3.txt"),"w") as g:
    for line in f:
        h,t,r=line.split(); h,t=int(h),int(t)
        if (h in nov) or (t in nov): removed+=1
        else: g.write(line); kept+=1
trip=[[int(x) for x in l.split()] for l in open(os.path.join(HERE,"kg_bio_334_case3.txt"))]
nb=len(trip)
trip+=[[int(x) for x in l.split()] for l in open(os.path.join(DD,"kge/ddibn/ddi_train.txt"))]
np.save(os.path.join(HERE,"kg_triples_case3.npy"), np.array(trip,dtype=np.int64))
print(f"신약 {len(novel)} / known {len(train)} | bio 제거 {removed} 유지 {kept} | kg_triples {len(trip)} (bio {nb}+ddi {len(trip)-nb})")
