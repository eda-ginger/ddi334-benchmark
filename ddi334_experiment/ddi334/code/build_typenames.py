#!/usr/bin/env python3
"""DDI-334 TWOSIDES side-effect 이름 매핑 생성 (LLM 프롬프트용).

원천 (DDI-Ben 기준 + TDC 이름):
  - tdc: Y코드 = dense index(0-1307), 원본 TDC CSV(meeting/ddi334_tdc)의 'Side Effect Name'
         → tdc index i = Y i → 이름 (직통).
  - ddibn: DDI_Ben cluster 209-hot이라 이름 없음 → ddibn↔tdc 공유 약물쌍에서
           컬럼 패턴(어느 쌍이 그 부작용을 갖는지)이 동일한 tdc 컬럼을 찾아 정렬.
           (검증: 209/209 정확 1:1, 모호 0, 미매칭 0 — shared 15,471쌍 기준)

출력 (ddi334/):
  twosides_typenames_ddibn.json  {"0": "aplasia pure red cell", ..., "208": ...}
  twosides_typenames_tdc.json    {"0": ..., ..., "1307": ...}

Run: micromamba run -n DDIBench python build_typenames.py
"""
import csv, glob, os, json
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
MEETING_TDC = os.path.join(LATEX_ROOT, 'meeting', 'ddi334_tdc')
SPLITS = ['train', 'valid_S0', 'valid_S1', 'valid_S2', 'test_S0', 'test_S1', 'test_S2']


def y2name():
    m = {}
    for f in glob.glob(os.path.join(MEETING_TDC, '*.csv')):
        for row in csv.DictReader(open(f)):
            m[int(row['Y'])] = row['Side Effect Name']
    return m


def load_pos(ds):
    pv = {}
    for sp in SPLITS:
        for line in open(os.path.join(HERE, 'data', ds, f'{sp}.txt')):
            p = line.split()
            if len(p) == 4 and p[3] == '1':
                a, b = int(p[0]), int(p[1])
                key = (a, b) if a < b else (b, a)
                pv[key] = [int(x) for x in p[2].split(',')]
    return pv


def main():
    names = y2name()
    sorted_y = sorted(names.keys())   # 1308 present Y codes (0-1316, 9 gaps)
    print(f"TDC Y->name: {len(names)} (Y {sorted_y[0]}~{sorted_y[-1]}, dense=rank)")

    # tdc dense index i = rank of i-th smallest present Y (build_dataset raw2dense)
    tdc_map = {str(i): names[sorted_y[i]] for i in range(len(sorted_y))}
    json.dump(tdc_map, open(os.path.join(HERE, 'meta', 'twosides_typenames_tdc.json'), 'w'),
              ensure_ascii=False, indent=0)
    print(f"[tdc] {len(tdc_map)} -> twosides_typenames_tdc.json")

    # ddibn: column-pattern match against tdc on shared pairs
    ddibn, tdc = load_pos('ddibn'), load_pos('tdc')
    shared = sorted(set(ddibn) & set(tdc))
    A = np.array([ddibn[k] for k in shared])   # [P,209]
    B = np.array([tdc[k] for k in shared])      # [P,1308]
    ddibn_map = {}
    ambig, unmatched = [], []
    for i in range(A.shape[1]):
        eq = np.where((B == A[:, i][:, None]).all(axis=0))[0]
        if len(eq) == 1:
            ddibn_map[str(i)] = names[sorted_y[int(eq[0])]]   # tdc dense col -> Y -> name
        elif len(eq) > 1:
            ambig.append((i, [int(x) for x in eq]))
        else:
            unmatched.append(i)
    assert not ambig and not unmatched, f"ambig={ambig} unmatched={unmatched}"
    json.dump(ddibn_map, open(os.path.join(HERE, 'meta', 'twosides_typenames_ddibn.json'), 'w'),
              ensure_ascii=False, indent=0)
    print(f"[ddibn] {len(ddibn_map)} -> twosides_typenames_ddibn.json "
          f"(shared {len(shared)} pairs, 209/209 exact 1:1)")
    print("  sample:", {k: ddibn_map[k] for k in ['0', '1', '2', '3', '4']})


if __name__ == '__main__':
    main()
