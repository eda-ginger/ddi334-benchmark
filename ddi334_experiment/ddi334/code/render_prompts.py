#!/usr/bin/env python3
"""V2 Ideal/Real 프롬프트를 미리 렌더 -> data/prompt/{ideal,real}/{split}.jsonl.

각 줄: {"d1","d2","type","y","user"}  (y=polarity: 1 양성쌍 / 0 음성쌍)
system 프롬프트는 상수라 data/prompt/_system.txt 에 1회 저장.
build_v2(mode=ideal/real)와 동일 로직 -> 재생성 가능(gitignore 대상).
입력 데이터(meta/bio_profile_llm.json 등)는 meta/에 유지, 산출 프롬프트만 data/에.
Run: micromamba run -n DDIBench python code/render_prompts.py [ddibn]
"""
import json
import os
import sys
from llm_prompts import DDI334Prompt, V2_SYSTEM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ddi334/
SPLITS = ["train", "valid_S0", "test_S0", "valid_S1", "test_S1", "valid_S2", "test_S2"]


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "ddibn"
    pb = DDI334Prompt(ds)
    out = os.path.join(HERE, "data", "prompt")
    MODES = ("ideal", "ideal_genes", "real")
    for m in MODES:
        os.makedirs(os.path.join(out, m), exist_ok=True)
    open(os.path.join(out, "_system.txt"), "w").write(V2_SYSTEM + "\n")
    total = 0
    for sp in SPLITS:
        path = os.path.join(HERE, "data", ds, f"{sp}.txt")
        if not os.path.exists(path):
            continue
        fh = {m: open(os.path.join(out, m, f"{sp}.jsonl"), "w") for m in MODES}
        n = 0
        for line in open(path):
            p = line.split()
            if len(p) < 4:
                continue
            d1, d2, pol = int(p[0]), int(p[1]), int(p[3])
            qids = [i for i, x in enumerate(p[2].split(',')) if x == '1']
            for t in qids:
                for m in MODES:
                    _, u = pb.build_v2(d1, d2, t, mode=m)
                    fh[m].write(json.dumps({"d1": d1, "d2": d2, "type": t, "y": pol, "user": u}, ensure_ascii=False) + "\n")
                n += 1
        for f in fh.values():
            f.close()
        total += n
        print(f"  {sp}: {n} prompts", flush=True)
    print(f"[render] {ds} 총 {total} (쌍,타입) x ideal/real -> {out}/")


if __name__ == "__main__":
    main()
