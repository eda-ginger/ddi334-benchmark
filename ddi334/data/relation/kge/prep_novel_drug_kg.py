#!/usr/bin/env python3
"""Novel drug KG preprocessing: novel-drug (Dn) list + novel-drug-isolated bio KG + kg_triples_novel_drug. (No GPU needed, for reproduction.)
Output: kge/{novel_drugs.json, kg_bio_334_novel_drug.txt, kg_triples_novel_drug.npy}"""
import numpy as np, json, os
HERE = os.path.dirname(os.path.abspath(__file__))              # data/relation/kge
DATA_DIR = os.path.dirname(os.path.dirname(HERE))               # data/
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
def drugs(fn):
    s=set()
    for line in open(fn):
        p=line.split()
        if len(p)==4: s.add(int(p[0])); s.add(int(p[1]))
    return s
train=drugs(os.path.join(SPLITS_DIR,"train.txt"))
novel=sorted((drugs(os.path.join(SPLITS_DIR,"test_S1.txt"))|drugs(os.path.join(SPLITS_DIR,"test_S2.txt")))-train)
json.dump({"novel_drug_ids":novel,"n_novel":len(novel),"n_train_drugs":len(train),
           "note":"novel_drug = cold-cluster Dn. entity_id==drug_id."}, open(os.path.join(HERE,"novel_drugs.json"),"w"),indent=1)
nov=set(novel); kept=removed=0
with open(os.path.join(HERE,"kg_bio_334.txt")) as f, open(os.path.join(HERE,"kg_bio_334_novel_drug.txt"),"w") as g:
    for line in f:
        h,t,r=line.split(); h,t=int(h),int(t)
        if (h in nov) or (t in nov): removed+=1
        else: g.write(line); kept+=1
trip=[[int(x) for x in l.split()] for l in open(os.path.join(HERE,"kg_bio_334_novel_drug.txt"))]
nb=len(trip)
trip+=[[int(x) for x in l.split()] for l in open(os.path.join(HERE,"existing_drug","ddi_train.txt"))]
np.save(os.path.join(HERE,"kg_triples_novel_drug.npy"), np.array(trip,dtype=np.int64))
print(f"novel {len(novel)} / known {len(train)} | bio removed {removed} kept {kept} | kg_triples {len(trip)} (bio {nb}+ddi {len(trip)-nb})")
