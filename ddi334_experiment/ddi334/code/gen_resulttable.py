#!/usr/bin/env python3
"""DDI-334 ddibn accuracy-primary 결과표 -> meeting/08_실험결과표(DDI334).md 생성.

- 인코더: results/ddibn_acc/summary.csv (체크포인트=accuracy, 3 seed mean+-std)
          지표 = Accuracy(primary) / macro-F1 / ROC-AUC / PR-AUC / AP@50(=P@50)  [P@(N/2)는 raw에만]
- LLM   : results/ddibn/llm_metrics/*.json 이관 (Acc/ROC/PR/F1; AP@50은 하드라벨이라 N/A "-")
          * LLM은 val-체크포인트가 없어 체크포인트 지표와 무관 -> 기존 추론 결과 그대로 유효
          * LLM ROC-AUC == accuracy (하드라벨 -> balanced-acc) : 각주
- Random: results/random_baseline.json (학습 없이 무작위 예측, 참조선 ~50)
값은 % (mean+-std). Run: micromamba run -n DDIBench python gen_resulttable.py
"""
import csv, collections, os, json, statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data", "ddibn")
ENC_SUMM = os.path.join(HERE, "results", "ddibn_acc", "summary.csv")
LLM_DIR = os.path.join(HERE, "results", "ddibn_acc", "llm_metrics")   # LLM도 canonical 폴더에 통합
RAND = os.path.join(HERE, "results", "random_baseline.json")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "meeting", "08_실험결과표(DDI334).md"))

META = {  # cell -> (modality, input, encoder)  — 표기는 03_실험결과표(DrugBank)와 통일
    "01": ("MOL", "mol graph", "GAT + SAGPool"),
    "02": ("MOL", "mol graph (3D)", "SchNet"),
    "03": ("MOL", "mol graph", "Graph Transformer"),
    "04": ("MOL", "fingerprint", "MLP"),
    "05": ("MOL", "fingerprint", "Jaccard+PCA"),
    "06": ("MOL", "smiles", "ChemBERTa (frozen)"),
    "07": ("MOL", "smiles", "1D-CNN"),
    "08": ("REL", "KG whole", "KGE frozen"),
    "09": ("REL", "KG subgraph", "R-GCN + KGE"),
    "10": ("REL", "KG subgraph", "GT + KGE"),
    "11": ("REL", "DDI network", "R-GCN (DDI whole)"),
    "12": ("REL", "Bio profile", "BP -> PCA"),
    "13": ("DOC", "description", "BioBERT (frozen)"),
    "14": ("DOC", "drug name", "BioBERT (frozen)"),
    "v1zs_phi": ("DOC", "template", "Phi-3.5-3.8B (ZS)"),
    "v1zs_qwen": ("DOC", "template", "Qwen2.5-3B (ZS)"),
    "v1zs_gemma": ("DOC", "template", "Gemma2-2B (ZS)"),
    "v1zs_phi4": ("DOC", "template", "Phi-4-14B (ZS)"),
    "v1zs_qwen14b": ("DOC", "template", "Qwen2.5-14B (ZS)"),
    "v1zs_gemma27b": ("DOC", "template", "Gemma2-27B (ZS)"),
    "v1ft_phi": ("DOC", "template", "Phi-3.5-3.8B (FT)"),
    "v1ft_qwen": ("DOC", "template", "Qwen2.5-3B (FT)"),
    "v1ft_gemma": ("DOC", "template", "Gemma2-2B (FT)"),
    "v1ft_phi4": ("DOC", "template", "Phi-4-14B (FT)"),
    "v1ft_qwen14b": ("DOC", "template", "Qwen2.5-14B (FT)"),
    "v1ft_gemma27b": ("DOC", "template", "Gemma2-27B (FT)"),
}
ENC_ORDER = [f"{i:02d}" for i in range(1, 15)]
# 모델 파라미터 수(B) — LLM 행을 크기 오름차순 정렬용
LLM_SIZE = {"v1zs_gemma": 2.0, "v1zs_qwen": 3.0, "v1zs_phi": 3.8, "v1zs_phi4": 13.9,
            "v1zs_qwen14b": 14.0, "v1zs_gemma27b": 27.0,
            "v1ft_gemma": 2.0, "v1ft_qwen": 3.0, "v1ft_phi": 3.8,
            "v1ft_phi4": 13.9, "v1ft_qwen14b": 14.0, "v1ft_gemma27b": 27.0}
# 표에서 제외할 LLM 셀 (행 자체를 숨김). 현재 없음 —
# qwen14b FT는 garbage 0.5가 삭제+필터로 빠져 행은 '—'(미실행)로 표시됨 (행은 유지).
LLM_EXCLUDE = set()
_zs = sorted([k for k in LLM_SIZE if k.startswith("v1zs") and k not in LLM_EXCLUDE], key=lambda k: LLM_SIZE[k])
_ft = sorted([k for k in LLM_SIZE if k.startswith("v1ft") and k not in LLM_EXCLUDE], key=lambda k: LLM_SIZE[k])
LLM_ORDER = _zs + _ft   # ZS(작은->큰) 그다음 FT(작은->큰)
SPLITS = ("S0", "S1", "S2")
# 표시 지표: summary 컬럼키 -> 표 헤더 약칭. 표가 길어 둘로 분할.
_PAPER = " (paper)"   # paper에서 실제 제시하는 지표 표식 (일반 텍스트)
METRICS = [("test_accuracy", "Acc"), ("test_macro-F1", "F1"),
           ("test_AUC-ROC", f"AUROC{_PAPER}"), ("test_PR-AUC", f"AUPRC{_PAPER}"), ("test_P@50", f"AP@50{_PAPER}")]
METRICS_T1 = [("test_accuracy", "Acc"), ("test_macro-F1", "F1")]
METRICS_T2 = [("test_AUC-ROC", f"AUROC{_PAPER}"), ("test_PR-AUC", f"AUPRC{_PAPER}"), ("test_P@50", f"AP@50{_PAPER}")]


def sub(std, dec=1):
    """std를 작은 회색 첨자로 (칸 폭 절약)."""
    return f"<sub style='font-size:78%;color:#999'>±{std:.{dec}f}</sub>"


def fmt(vals):
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{st.mean(vals):.2f}{sub(st.pstdev(vals), 2)}"


def dataset_stats():
    """ddibn train/valid/test의 triplet ((d1,d2,type)) 수 집계 (양성/음성 별도, 보고 단위=triplet)."""
    def interactions(fn):
        pos = neg = 0
        for line in open(fn):
            p = line.split()
            if len(p) != 4:
                continue
            b = sum(int(x) for x in p[2].split(','))
            if int(p[3]) == 1:
                pos += b
            else:
                neg += b
        return pos, neg
    files = {"train": "train.txt", "S0v": "valid_S0.txt", "S0t": "test_S0.txt",
             "S1v": "valid_S1.txt", "S1t": "test_S1.txt", "S2v": "valid_S2.txt", "S2t": "test_S2.txt"}
    pos, neg = {}, {}
    for k, v in files.items():
        fp = os.path.join(DATA_DIR, v)
        if os.path.exists(fp):
            pos[k], neg[k] = interactions(fp)
        else:
            pos[k] = neg[k] = None
    return pos, neg


def load_encoders():
    """cell -> split -> metrickey -> [seed별 값(%)]"""
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    if not os.path.exists(ENC_SUMM):
        return agg
    for r in csv.DictReader(open(ENC_SUMM)):
        c, sp = r.get("cell"), r.get("split")
        if sp not in SPLITS:
            continue
        for key, _ in METRICS:
            if r.get(key):
                agg[c][sp][key].append(float(r[key]) * 100)
    return agg


def load_llm():
    """label -> split -> {Acc/ROC/PR/F1}(%) (단일 run)"""
    d = collections.defaultdict(dict)
    override = {}   # v1ftN_(재학습) -> v1ft_ override 모음
    if not os.path.isdir(LLM_DIR):
        return d
    for fn in os.listdir(LLM_DIR):
        if not fn.endswith(".json") or fn.endswith("_cv.json"):
            continue   # _cv.json은 selected/val_curve 구조라 결과 json 아님 (skip)
        m = json.load(open(os.path.join(LLM_DIR, fn)))
        if "cell" not in m or "split" not in m:
            continue   # 결과 포맷 아닌 파일 방어
        label, sp = m["cell"], m["split"]
        if m.get("accuracy") == 0.5 and m.get("macro-F1") == 0:
            continue   # degenerate 0.5/0.0 = vLLM 연결실패 artifact (실제 점수 아님) -> 미반영
        md = {"test_accuracy": m["accuracy"] * 100, "test_macro-F1": m["macro-F1"] * 100,
              "test_AUC-ROC": m["AUC-ROC"] * 100, "test_PR-AUC": m["PR-AUC"] * 100}
        if label.startswith("v1ftN_"):   # 재학습(max_len2048+1024) 결과 -> v1ft_ 행에 덮어씀(완료 시 우선)
            override.setdefault(label.replace("v1ftN_", "v1ft_", 1), {})[sp] = md
        else:
            d[label][sp] = md
    for lbl, spd in override.items():   # 재학습 결과가 기존 v1ft_(구버전)를 override
        for sp, md in spd.items():
            d[lbl][sp] = md
    return d


# cell08 KGE 노드초기값 비교: (KGE_TYPE, summary.csv 경로). transe=기존 cell08, 나머지=kge_compare/{k}/
KGE_COMPARE = [
    ("transe", ENC_SUMM),
    ("distmult", os.path.join(HERE, "results", "kge_compare", "distmult", "summary.csv")),
    ("rotate", os.path.join(HERE, "results", "kge_compare", "rotate", "summary.csv")),
    ("complex", os.path.join(HERE, "results", "kge_compare", "complex", "summary.csv")),
    ("transh", os.path.join(HERE, "results", "kge_compare", "transh", "summary.csv")),
]


def load_cell08(path):
    """summary.csv에서 cell08만 -> split -> metrickey -> [seed별 %]. seed 집합도 반환.
    test_* 지표 + val_accuracy(선택 기준) 둘 다 로드 (기존 train.py 컨벤션: val로 선택, test로 보고)."""
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    seeds = set()
    if not os.path.exists(path):
        return agg, seeds
    for r in csv.DictReader(open(path)):
        if r.get("cell") != "08" or r.get("split") not in SPLITS:
            continue
        seeds.add(r.get("seed"))
        for key, _ in METRICS:
            if r.get(key):
                agg[r["split"]][key].append(float(r[key]) * 100)
        if r.get("val_accuracy"):   # 선택 기준 (체크포인트와 동일)
            agg[r["split"]]["val_accuracy"].append(float(r["val_accuracy"]) * 100)
    return agg, seeds


# Real 신약 시나리오 cell08 비교 (novel 고립 KG로 학습한 KGE). 후보 transe/rotate/complex.
KGE_COMPARE_CASE3 = [
    ("transe", os.path.join(HERE, "results", "kge_compare_case3", "transe", "summary.csv")),
    ("rotate", os.path.join(HERE, "results", "kge_compare_case3", "rotate", "summary.csv")),
    ("complex", os.path.join(HERE, "results", "kge_compare_case3", "complex", "summary.csv")),
]

# Ablation: KGE 재학습 없이 신약 64개 임베딩 행만 랜덤 교체 (S1 향상 원인 단일변수 검증).
ABLATION_TRANSE = os.path.join(HERE, "results", "kge_compare_ablnr", "transe", "summary.csv")
CASE3_TRANSE = os.path.join(HERE, "results", "kge_compare_case3", "transe", "summary.csv")


def _render_kge_table(L, kge_list, dim_subdir, title, intro, tail):
    """KGE 비교 표 1개 렌더. 각 (split×지표 S0/S1) 열 최고값(test mean) 굵게+음영, S2 생략."""
    DIV = "border-left:2px solid #555"
    data = {k: load_cell08(p) for k, p in kge_list}
    best_cell = {}
    for key, _ in METRICS:
        for sp in SPLITS:
            if sp == "S2":
                continue
            means = {k: st.mean(a[sp][key]) for k, (a, _) in data.items() if a[sp][key]}
            if means:
                mx = max(means.values())
                best_cell[(key, sp)] = {k for k, v in means.items() if abs(v - mx) < 1e-9}

    def kge_dim(k):
        d = os.path.join(HERE, "data", "kge", dim_subdir, k)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if "ent_" in f and f.endswith("d.npy"):
                    return f.split("_ent_")[-1].replace(".npy", "")
        return "—"

    def row_tds(k, agg, empty=False):
        s = ""
        for key, _ in METRICS:
            for i, sp in enumerate(SPLITS):
                cs = f' style="{DIV}"' if i == 0 else ""
                if empty:
                    val = "—"
                else:
                    val = fmt(agg[sp][key])
                    if k in best_cell.get((key, sp), set()):
                        cs = f' style="{DIV};background:#ffe0e0"' if i == 0 else ' style="background:#ffe0e0"'
                        val = f"<b>{val}</b>"
                s += f"<td{cs}>{val}</td>"
        return s

    L.append(f"### {title}")
    L.append("")
    L.append(intro)
    L.append("")
    L.append('<table border="1" cellspacing="0" cellpadding="5" '
             'style="border-collapse:collapse;text-align:center;white-space:nowrap">')
    h1 = ('<tr style="background:#e8e8e8"><th rowspan="2">KGE</th><th rowspan="2">dim</th>'
          '<th rowspan="2">seeds</th>')
    for _, ab in METRICS:
        h1 += f'<th colspan="3" style="{DIV}">{ab}</th>'
    h1 += "</tr>"
    h2 = '<tr style="background:#f3f3f3">'
    for _, ab in METRICS:
        for i, sp in enumerate(SPLITS):
            h2 += f'<th style="{DIV}">{sp}</th>' if i == 0 else f'<th>{sp}</th>'
    h2 += "</tr>"
    L.append(h1); L.append(h2)
    for k, p in kge_list:
        agg, seeds = data[k]
        nseed = len(seeds)
        nseed_txt = f"{nseed}" if nseed else "0 (대기)"
        L.append(f"<tr><td>{k}</td><td>{kge_dim(k)}</td><td>{nseed_txt}</td>"
                 f"{row_tds(k, agg, empty=(nseed == 0))}</tr>")
    L.append("</table>")
    L.append("")
    done = [k for k, (a, sd) in data.items() if len(sd) > 0]
    pend = [k for k, (a, sd) in data.items() if len(sd) == 0]
    note = f"> 완료: {', '.join(done) if done else '없음'}"
    if pend:
        note += f" / 대기(KGE 학습중): {', '.join(pend)}"
    note += tail
    L.append(note)
    L.append("")


def load_cell(path, cell):
    """summary.csv에서 특정 cell -> split -> metrickey -> [seed별 %]. seeds도."""
    agg = collections.defaultdict(lambda: collections.defaultdict(list)); seeds = set()
    if not os.path.exists(path):
        return agg, seeds
    for r in csv.DictReader(open(path)):
        if r.get("cell") == cell and r.get("split") in SPLITS:
            seeds.add(r.get("seed"))
            for key, _ in METRICS:
                if r.get(key):
                    agg[r["split"]][key].append(float(r[key]) * 100)
    return agg, seeds


def _c3(*p):
    return os.path.join(HERE, "results", *p)
# Ideal/Real 짝 비교 행: (#, 모달, 입력, 인코더, Case, summary경로, summary내 cell).
# 같은 실험을 Ideal(정상)/Real(신약) 연달아 둬서 S1/S2 차이를 바로 비교.
CASE3_COMPARE_ROWS = [
    ("08", "REL", "KG whole", "KGE frozen (transe)", "Ideal", ENC_SUMM, "08"),
    ("08", "REL", "KG whole", "KGE frozen (transe)", "Real", _c3("kge_compare_case3", "transe", "summary.csv"), "08"),
    ("09", "REL", "KG subgraph", "R-GCN + KGE", "Ideal", ENC_SUMM, "09"),
    ("09", "REL", "KG subgraph", "R-GCN + KGE", "Real", _c3("ddibn_case3", "summary.csv"), "09"),
    ("10", "REL", "KG subgraph", "GT + KGE", "Ideal", ENC_SUMM, "10"),
    ("10", "REL", "KG subgraph", "GT + KGE", "Real", _c3("ddibn_case3", "summary.csv"), "10"),
    ("11", "REL", "DDI network", "R-GCN (DDI whole)", "Ideal", ENC_SUMM, "11"),
    ("11", "REL", "DDI network", "R-GCN (DDI whole)", "Real", _c3("ddibn_case3", "summary.csv"), "11"),
    ("13", "DOC", "description", "BioBERT (frozen)", "Ideal", ENC_SUMM, "13"),
    ("14", "DOC", "drug name", "BioBERT (frozen)", "Ideal", ENC_SUMM, "14"),
    ("13", "DOC", "SMILES", "BioBERT (frozen)", "Real", _c3("ddibn_case3_doc_smiles", "summary.csv"), "13"),
    ("13", "DOC", "RDKit 30-desc", "BioBERT (frozen)", "Real", _c3("ddibn_case3_doc_rdkitdesc", "summary.csv"), "13"),
]


def zs_prompt_compare_section(L):
    """ZS 프롬프트 방식 비교: V1(multi-label JSON) vs V2(binary Yes/No). 작은 모델 3종. Acc + F1."""
    DIV = "border-left:2px solid #555"
    TAGS = [("gemma", "Gemma2-2B"), ("qwen", "Qwen2.5-3B"), ("phi", "Phi-3.5-mini")]

    def zs(ver, tag, sp, metric):
        f = os.path.join(LLM_DIR, f"{ver}zs_{tag}_{sp}.json")
        if os.path.exists(f):
            return json.load(open(f)).get(metric)
        return None

    L.append("## ZS 프롬프트 방식 비교 (multi-label JSON vs binary Yes/No)")
    L.append("")
    L.append("> 기존 ZS(**V1**, 한 번에 multi-label JSON)는 전 모델 chance(~50). "
             "**V2**(쌍+타입별 binary Yes/No)로 같은 후보(vec=1)를 하나씩 질문 → 프롬프트 형식이 ZS 성능을 살리는지 비교. "
             "값 = test % (단일 run). 작은 모델 3종. **V1 대비 V2↑ = '형식이 ZS 병목'이었음.**")
    L.append("")
    L.append('<table border="1" cellspacing="0" cellpadding="5" '
             'style="border-collapse:collapse;text-align:center;white-space:nowrap">')
    METR = [("Acc", "accuracy"), ("F1", "macro-F1")]
    h1 = ('<tr style="background:#e8e8e8"><th rowspan="2" style="text-align:left">모델</th>'
          '<th rowspan="2">방식</th>')
    for ab, _ in METR:
        h1 += f'<th colspan="3" style="{DIV}">{ab}</th>'
    h1 += "</tr>"
    h2 = '<tr style="background:#f3f3f3">'
    for _m in METR:
        for i, sp in enumerate(SPLITS):
            h2 += f'<th style="{DIV}">{sp}</th>' if i == 0 else f'<th>{sp}</th>'
    h2 += "</tr>"
    L.append(h1); L.append(h2)
    for tag, name in TAGS:
        for ver, vlabel in [("v1", "V1 (JSON)"), ("v2", "V2 (binary)")]:
            rowbg = ' style="background:#eef5ff"' if ver == "v2" else ""   # V2 행 옅은 음영
            s = ""
            for _, metric in METR:
                for i, sp in enumerate(SPLITS):
                    cs = f' style="{DIV}"' if i == 0 else ""
                    v = zs(ver, tag, sp, metric)
                    s += f"<td{cs}>{('%.2f' % (v*100)) if v is not None else '—'}</td>"
            L.append(f'<tr{rowbg}><td style="text-align:left">{name}</td><td><b>{vlabel}</b></td>{s}</tr>')
    L.append("</table>")
    L.append("> V2 미완 셀은 '—' (채워지는 중). 결과 v2zs_*.json (V1 v1zs_*와 분리 저장).")
    L.append("")
    # 답 분포 plot (probe 실측: gemma-2-2b, S0 샘플 458 질의)
    fig = "fig_zs_answer_dist.png"
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "meeting", fig)) or True:
        L.append(f"![ZS 답 분포]({fig})")
        L.append("")
        L.append("> 분포 (gemma-2-2b, S0 80쌍/458질의 실측): **V1=Yes 5%**(양성·음성 무차별), **V2=Yes 0%**(전부 No). "
                 "둘 다 양성/음성 못 가림(AUROC~0.5) → 형식이 분포만 바꿀 뿐 무신호, FT 필요.")
        L.append("")


def case3_section(L):
    """신약 개발 시나리오 (Real) — Ideal과 같은 실험을 나란히 비교. 표 1과 동일 스타일 + Case 열."""
    DIV = "border-left:2px solid #555"
    L.append("## 신약 개발 시나리오 (Real)")
    L.append("")
    L.append("> **신약 = 알려진 상호작용/문헌 없이 화학 구조만 아는 약물.** novel 64개를 KG에서 **고립**(bio edge 제거 → KGE random)시키고, "
             "DOC는 **구조에서만 유도**(SMILES / RDKit 30-desc)로 제한. **MOL은 그대로**(신약도 구조 known)라 Ideal 재사용(표 1 참조). "
             "DDI 데이터/splits·인코더·평가(split별 val-best)는 Ideal과 동일 — **입력(REL KG, DOC 텍스트)만 신약 조건으로 교체**. "
             "값 = test % (3 seed mean±std). **같은 실험의 Ideal(정상)/Real(신약) 행을 나란히** 둬서 S1/S2(novel) 차이 = 신약 조건 영향.")
    L.append("")
    L.append('<table border="1" cellspacing="0" cellpadding="5" '
             'style="border-collapse:collapse;text-align:center;white-space:nowrap">')
    h1 = ('<tr style="background:#e8e8e8"><th rowspan="2">#</th><th rowspan="2">모달리티</th>'
          '<th rowspan="2">입력 타입</th><th rowspan="2" style="text-align:left">인코더</th>'
          '<th rowspan="2">Case</th>')
    for _, ab in METRICS:
        h1 += f'<th colspan="3" style="{DIV}">{ab}</th>'
    h1 += "</tr>"
    h2 = '<tr style="background:#f3f3f3">'
    for _, ab in METRICS:
        for i, sp in enumerate(SPLITS):
            h2 += f'<th style="{DIV}">{sp}</th>' if i == 0 else f'<th>{sp}</th>'
    h2 += "</tr>"
    L.append(h1); L.append(h2)
    for cid, mod, inp, name, case, path, cell in CASE3_COMPARE_ROWS:
        agg, seeds = load_cell(path, cell)
        nseed = len(seeds)
        rowbg = ' style="background:#eef5ff"' if case == "Real" else ""   # Real 행 옅은 음영
        s = ""
        for key, _ in METRICS:
            for i, sp in enumerate(SPLITS):
                cs = f' style="{DIV}"' if i == 0 else ""
                s += f"<td{cs}>{('—' if not nseed else fmt(agg[sp][key]))}</td>"
        L.append(f'<tr{rowbg}><td>{cid}</td><td>{mod}</td><td>{inp}</td>'
                 f'<td style="text-align:left">{name}</td><td><b>{case}</b></td>{s}</tr>')
    L.append("</table>")
    L.append("")
    L.append("> 파란 행 = Real(신약). REL(08-11): novel을 KG에서 고립(bio edge 제거)시켜 KGE에서 random init. "
             "DOC(13): 구조유도 SMILES는 신약도 유효 / RDKit-desc는 BioBERT 임베딩 뭉침(cos 0.986)으로 약함=encoding 한계. "
             "cell08 KGE 종류별 상세는 아래 표 B 참조. (Real 미완 셀은 '—', 채워지는 중)")
    L.append("")
    # DOC 입력 텍스트 예시 (파일에서 직접 로드 — 드리프트 방지)
    ex_id = "156"
    try:
        sm_ex = json.load(open(os.path.join(HERE, "data", "ddibn", "precompute", "drug_smiles.json"))).get(ex_id, "")
    except Exception:
        sm_ex = ""
    try:
        rd_ex = json.load(open(os.path.join(HERE, "data", "case3", "rdkit_desc.json"))).get(ex_id, "")
    except Exception:
        rd_ex = ""
    try:
        dj = json.load(open("/home/rudwls2717/Latex/data/descriptions/drugbank.json"))
        ds_ex = dj.get(ex_id) or dj.get(int(ex_id)) or ""
        if len(ds_ex) > 350:
            ds_ex = ds_ex[:350].rstrip() + " ..."
    except Exception:
        ds_ex = "(description 소스 미접근)"
    L.append("### DOC 입력 텍스트 예시 — 약물 156 (Carbamazepine)")
    L.append("")
    L.append("BioBERT에 실제 들어가는 입력. **같은 약물, 3가지 DOC 형식**:")
    L.append("")
    L.append("**① SMILES (Real):**")
    L.append("```\n" + sm_ex + "\n```")
    L.append("**② RDKit 30-desc (Real):**")
    L.append("```json\n" + rd_ex + "\n```")
    L.append("**③ description (Ideal):**")
    L.append("```\n" + ds_ex + "\n```")
    L.append("> **①②③ 비교**: ② RDKit-desc는 모든 약물이 **동일 키 틀**(`{\"SMILES\":..,\"MW\":..}`) + 숫자만 달라 BioBERT 임베딩이 뭉침(cos 0.986). "
             "③ description은 자연어지만 약물군/적응증 위주라 **상호작용·구조와 직결 X** (신약 전이 약함, S2 chance↓). "
             "① SMILES는 **구조 그 자체**라 약물마다 다르고 분자에 grounded → **신약에도 유효**(S2 최고).")
    L.append("")


def kge_compare_section(L):
    """cell08 노드초기값(KGE) 비교 — Ideal(정상 KG) + Real(신약 고립 KG) 두 표."""
    L.append("## KGE 노드 초기값 비교 (cell08, REL · KG whole · KGE frozen)")
    L.append("")
    L.append("> cell08(KG 전체 entity 임베딩 frozen lookup)에서 **노드 초기값 KGE만 교체**해 비교. "
             "값 = test % (3 seed mean±std). **각 (split×지표) 열 최고값(평균)을 굵게+음영** (동률 모두), "
             "**S2는 전 KGE chance(~50%)라 생략**. cell08은 인코더라 AUROC/AUPRC/AP@50도 유효.")
    L.append("")
    _render_kge_table(
        L, KGE_COMPARE, "ddibn",
        "표 A. Ideal (정상 KG — cold 약물도 bio edge 보유)",
        "> 일반 시나리오: cold 약물도 KG의 bio 관계(target/pathway 등)로 임베딩 학습됨(58/64).",
        ". 굵게=S0/S1 각 지표 열 최고값. cell 09/10/11 진행 방식은 사용자와 결정.")
    _render_kge_table(
        L, KGE_COMPARE_CASE3, "ddibn_case3",
        "표 B. Real (신약 시나리오 — novel 약물 KG 고립)",
        "> 신약 시나리오: novel 64개를 KG에서 고립(bio edge 제거) -> KGE에서 random init 유지. "
        "DDI 데이터/splits는 Ideal과 동일, KG 임베딩만 교체. REL은 transe 사용.",
        ". 굵게=S0/S1 각 지표 열 최고값. (Ideal 표 A와 대조용 데이터 보존.)")
    ablation_compare_section(L)


def ablation_compare_section(L):
    """표 B-1: 신약 임베딩 제거가 S1 향상의 원인인지 단일변수로 검증 (transe, 3 seed)."""
    def ms(path, sp, key):
        agg, _ = load_cell08(path)
        v = agg.get(sp, {}).get(key, [])
        return sum(v) / len(v) if v else None

    def g(x):
        return f"{x:.1f}" if x is not None else "—"

    rows = [
        ("Ideal",                 "Ideal에서 학습된 임베딩 그대로",        ENC_SUMM),
        ("Ideal (+신약 고립 임베딩)", "신약만 Real의 고립 임베딩으로 교체 (재학습 X)", ABLATION_TRANSE),
        ("Real",                  "KG 고립 후 KGE 전체 재학습",          CASE3_TRANSE),
    ]
    b_a = ms(ENC_SUMM, "S1", "test_accuracy")
    b_f = ms(ENC_SUMM, "S1", "test_macro-F1")

    L.append("### 표 B-1. \"신약 효과\"와 \"기존 약물 임베딩 효과\"를 분리 (transe, 3 seed)")
    L.append("")
    L.append("> **무엇을 했나**: KGE를 **재학습하지 않고**, Ideal(정상 KG) 임베딩에서 "
             "**신약 64행만 Real(신약 고립 KG)의 임베딩으로 교체**. 기존 약물·비약물 엔티티·DDI 데이터·splits는 전부 Ideal과 동일. "
             "저장 `results/kge_compare_ablnr/transe/` (Ideal·Real과 분리).")
    L.append("> **목적**: 중간 트랙(신약만 고립값)을 끼우면, 두 비교가 각각 **단일 변수**가 됨 — "
             "Ideal↔중간은 **신약 임베딩만** 다르고(신약 효과), 중간↔Real은 신약이 동일하므로 **기존 약물 임베딩만** 다름(기존 효과). "
             "(cell08은 약물 행만 예측에 사용 → 비약물 엔티티 차이는 무관.)")
    L.append("")
    L.append("| 시나리오 | 신약(novel 64) 처리 | S0 acc | S0 F1 | S1 acc | S1 F1 | S2 acc | S2 F1 | S1 Δ (Ideal 대비) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    s1a_by = {}
    for name, how, path in rows:
        s0a, s0f = ms(path, "S0", "test_accuracy"), ms(path, "S0", "test_macro-F1")
        s1a, s1f = ms(path, "S1", "test_accuracy"), ms(path, "S1", "test_macro-F1")
        s2a, s2f = ms(path, "S2", "test_accuracy"), ms(path, "S2", "test_macro-F1")
        s1a_by[name] = s1a
        if name == "Ideal" or b_a is None or s1a is None:
            d = "기준" if name == "Ideal" else "—"
        else:
            d = f"acc {s1a - b_a:+.1f}pt / F1 {s1f - b_f:+.1f}pt"
        L.append(f"| {name} | {how} | {g(s0a)} | {g(s0f)} | **{g(s1a)}** | **{g(s1f)}** | {g(s2a)} | {g(s2f)} | {d} |")
    L.append("")
    L.append("> **S2**(두 약물 모두 신약)는 세 시나리오 모두 chance(acc≈50, F1≈0) — S2에선 양쪽 다 신약이라 "
             "어떤 처리를 해도 쓸 수 있는 임베딩 정보가 없음. 효과는 **S1에서만** 나타남. (S0도 세 트랙 동일 → 차이는 순수 S1 효과.)")
    L.append("")
    mid = s1a_by.get("Ideal (+신약 고립 임베딩)")
    ide = s1a_by.get("Ideal")
    rea = s1a_by.get("Real")
    L.append("**해석 (두 효과 분리, S1 accuracy 기준)**")
    if ide is not None and mid is not None:
        dn = mid - ide
        L.append(f"- **(A) 신약 효과** [Ideal ↔ 중간]: 기존·비약물 동일, 신약만 *학습값→고립값*. "
                 f"S1 **{dn:+.1f}pt** → 신약의 학습 임베딩이 고립(미학습) 임베딩보다 "
                 f"{'예측에 해로웠다' if dn > 0 else '나았다'}(모델이 신약 정보를 버리고 기존 짝 약물에 의지).")
    else:
        L.append("- **(A) 신약 효과** [Ideal ↔ 중간]: (중간 트랙 결과 집계 중)")
    if mid is not None and rea is not None:
        dk = rea - mid
        verdict = ("**full-KG 기존 임베딩이 더 좋음 → 신약 bio edge가 기존 약물 임베딩을 향상시켰다(확정)**"
                   if dk < -0.3 else
                   "두 트랙 차이 미미 → 기존 임베딩 품질 효과는 작음(아까 Real<중간 격차는 신약값/frame 차이였음)"
                   if abs(dk) <= 0.3 else
                   "축소-KG 기존 임베딩이 오히려 더 좋음(예상과 반대 — 추가 점검 필요)")
        L.append(f"- **(B) 기존 약물 임베딩 효과** [중간 ↔ Real]: 신약 동일(둘 다 고립값), 기존만 *full-KG→축소-KG*. "
                 f"S1 **{dk:+.1f}pt** → {verdict}.")
    else:
        L.append("- **(B) 기존 약물 임베딩 효과** [중간 ↔ Real]: (Real/중간 결과 집계 중)")
    L.append("")


def main():
    enc = load_encoders()
    llm = load_llm()
    rand = json.load(open(RAND)) if os.path.exists(RAND) else {}

    n_enc = sum(1 for c in ENC_ORDER if enc.get(c))
    n_llm = sum(1 for c in LLM_ORDER if llm.get(c))

    L = []
    L.append("# DDI-334 (ddibn) 실험 결과표 — accuracy-primary")
    L.append("")
    L.append("> 데이터셋 **ddibn** (DDI-Bench 필터본, 209 side-effect types). "
             "프로토콜 DDI-Bench (per-type, negative-sampling link prediction). "
             "**Primary = Accuracy** (인코더+LLM 공정비교 위해 동일 지표 + 선택=보고 일관). "
             "인코더 체크포인트 = val Accuracy. 값 = test %, 인코더는 3 seed mean±std / LLM은 1 run.")
    L.append("> Split: **S0** transductive / **S1** inductive(1 cold) / **S2** inductive(2 cold). "
             "지표: Acc / macro-F1 / AUROC / AUPRC / AP@50(=상위 50 정밀도). (paper 표기: AUROC=ROC-AUC, AUPRC=PR-AUC)")
    L.append(f"> 진행: 인코더 {n_enc}/14 셀, LLM {n_llm}/12 셀 집계.")
    L.append("")

    # 데이터셋 통계 (triplet 단위, 양성/음성)
    pos, neg = dataset_stats()
    def cm(x):
        return f"{x:,}" if x is not None else "—"
    def rowvals(d):
        tot = sum(v for v in d.values() if v is not None)
        return (f"| {cm(d['train'])} | {cm(d['S0v'])} / {cm(d['S0t'])} | {cm(d['S1v'])} / {cm(d['S1t'])} | "
                f"{cm(d['S2v'])} / {cm(d['S2t'])} | {tot:,} |")
    grand = sum(v for v in list(pos.values()) + list(neg.values()) if v is not None)
    L.append("## 데이터셋 통계 (DDI-334 ddibn) — **triplet ((d1,d2,type)) 수**")
    L.append("")
    L.append("| | Train | S0 val / test | S1 val / test | S2 val / test | 합계 |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| 양성 triplet {rowvals(pos)}")
    L.append(f"| 음성 triplet {rowvals(neg)}")
    L.append(f"| **합계** {rowvals({k: (pos[k] or 0) + (neg[k] or 0) if pos[k] is not None else None for k in pos})}")
    L.append("")
    L.append("> 약물 334개, side-effect type 209개, negative 1:1(쌍 단위 샘플링 — 음성 쌍은 짝 양성의 멀티핫을 상속). "
             "수치는 **triplet ((d1,d2,type)) 수**이며(양성 triplet = 실제 상호작용), 파일은 약물쌍+멀티핫으로 저장되나 "
             "학습(BCE per-type)·평가(type별 평균)는 triplet 단위로 수행됨. "
             "참고: LLM eval 로그의 'val rows'는 LLM 질의 단위인 약물쌍 수(쌍당 여러 triplet을 한 번에 질의).")
    L.append("")

    # ── getter 정의 (kind별) ──
    enc_get = lambda c, sp, key: fmt(enc.get(c, {}).get(sp, {}).get(key, []))

    def llm_get(c, sp, key):
        # LLM V1은 하드라벨(true/false) -> 확률/랭킹 없음.
        # ROC(=balanced acc 재포장) / PR(2점 degenerate) / AP@50(랭킹 없음)은 정의 불가 -> "—"
        # 하드라벨에 정직한 Acc·macro-F1만 보고.
        if key in ("test_AUC-ROC", "test_PR-AUC", "test_P@50"):
            return "—"
        v = llm.get(c, {}).get(sp, {}).get(key)
        return f"{v:.2f}" if v is not None else "—"

    def rand_get(c, sp, key):
        k = {"test_accuracy": "acc", "test_macro-F1": "f1", "test_AUC-ROC": "roc",
             "test_PR-AUC": "pr", "test_P@50": "p50"}[key]
        if sp in rand and k in rand[sp]:
            m, s = rand[sp][k]
            return f"{m:.2f}{sub(s, 2)}"   # Random은 50 근처 미세차가 의미있어 2자리 (mean+std)
        return "—"

    # 행 순서: 인코더(01-14) -> Random(참조선) -> LLM(15-20)
    ROWS = [(f"{i:02d}", c, enc_get) for i, c in enumerate(ENC_ORDER, 1)]
    ROWS.append(("ref", "__rand__", rand_get))     # LLM 위에 배치, 음영+이탤릭으로 구분
    ROWS += [(f"{i:02d}", c, llm_get) for i, c in enumerate(LLM_ORDER, 15)]

    DIV = "border-left:2px solid #555"   # 지표 블록 구분선

    # 인코더(01-14)만 대상으로 각 (지표×split) 열 1등/2등(mean) 셀 강조 (LLM/Random 제외)
    # best_enc[(key,sp)] = {'1': {1등 cell들}, '2': {2등 cell들}} — 동률은 같은 등수로 묶고 다음 값이 2등.
    best_enc = {}
    for key, _ in METRICS:
        for sp in SPLITS:
            means = {c: st.mean(enc[c][sp][key]) for c in ENC_ORDER
                     if enc.get(c, {}).get(sp, {}).get(key)}
            if means:
                uniq = sorted(set(means.values()), reverse=True)   # 서로 다른 값 내림차순
                rk = {}
                for n in range(min(2, len(uniq))):   # 1·2등 (동률은 같은 등수)
                    rk[str(n + 1)] = {c for c, v in means.items() if abs(v - uniq[n]) < 1e-9}
                best_enc[(key, sp)] = rk

    def render(metrics, title):
        L.append(f"### {title}")
        L.append("")
        L.append('<table border="1" cellspacing="0" cellpadding="5" '
                 'style="border-collapse:collapse;text-align:center;white-space:nowrap">')
        # 헤더 1행: 지표명 colspan=3
        h1 = ('<tr style="background:#e8e8e8">'
              '<th rowspan="2">#</th><th rowspan="2">모달리티</th>'
              '<th rowspan="2">입력 타입</th><th rowspan="2" style="text-align:left">인코더</th>')
        for _, ab in metrics:
            h1 += f'<th colspan="3" style="{DIV}">{ab}</th>'
        h1 += "</tr>"
        # 헤더 2행: split
        h2 = '<tr style="background:#f3f3f3">'
        for _, ab in metrics:
            for i, sp in enumerate(SPLITS):
                h2 += f'<th style="{DIV}">{sp}</th>' if i == 0 else f'<th>{sp}</th>'
        h2 += "</tr>"
        L.append(h1); L.append(h2)

        def tds(getter, c, is_enc=False):
            s = ""
            for key, _ in metrics:
                for i, sp in enumerate(SPLITS):
                    stl = f' style="{DIV}"' if i == 0 else ""
                    val = getter(c, sp, key)
                    # 인코더 행: 그 (지표×split) 열 1등=빨강 / 2등=파랑 / 3등=초록 (전부 굵게)
                    bc = best_enc.get((key, sp), {})
                    bg = None
                    if is_enc:
                        if c in bc.get('1', set()):
                            bg = "#ffd6d6"
                        elif c in bc.get('2', set()):
                            bg = "#d6e6ff"
                        elif c in bc.get('3', set()):
                            bg = "#d9f0d9"
                    if bg:
                        stl = f' style="{DIV};background:{bg}"' if i == 0 else f' style="background:{bg}"'
                        val = f"<b>{val}</b>"
                    s += f"<td{stl}>{val}</td>"
            return s

        for cid, c, getter in ROWS:
            if c == "__rand__":   # 참조선 행: 음영 + 이탤릭
                L.append(f'<tr style="background:#fff3cd;font-style:italic">'
                         f'<td>{cid}</td><td>(ref)</td><td>random</td>'
                         f'<td style="text-align:left">Random (no training)</td>{tds(getter, None)}</tr>')
            else:
                mod, inp, name = META.get(c, ("?", "?", c))
                L.append(f"<tr><td>{cid}</td><td>{mod}</td><td>{inp}</td>"
                         f'<td style="text-align:left">{name}</td>{tds(getter, c, is_enc=(getter is enc_get))}</tr>')
        L.append("</table>")
        L.append("")

    render(METRICS, "표 1. 전 지표 통합 (Acc·macro-F1 = 전 모델 / AUROC·AUPRC·AP@50 = 인코더 전용, LLM은 확률 없어 '—')")
    L.append("> 색: <b style='background:#ffd6d6'>&nbsp;빨강=1등&nbsp;</b> / <b style='background:#d6e6ff'>&nbsp;파랑=2등&nbsp;</b> "
             "(각 split×지표 열에서 **인코더 14개 중** mean 기준, 동률은 같은 등수). LLM·Random은 비대상.")
    L.append("")

    zs_prompt_compare_section(L)

    case3_section(L)

    kge_compare_section(L)

    L.append("## 비고")
    L.append("- **Primary = Accuracy**: DDI-Bench repo 동작 및 인코더·LLM 공정비교(동일 지표)와 정합. "
             "체크포인트도 val Accuracy로 선택(선택=보고 일관). 구 ROC-AUC 체크포인트 결과는 `results/ddibn/` 보관.")
    L.append("- **LLM은 하드라벨(true/false) 출력 -> 확률/랭킹 없음**. 따라서 **AUROC·AUPRC·AP@50 모두 정의 불가('—')**: "
             "ROC는 balanced-accuracy의 재포장(실측 Acc와 소수점까지 동일), PR은 2점 degenerate. "
             "하드라벨에 정직한 **Accuracy·macro-F1만** 보고. (LLM은 val-체크포인트 없어 재실행 불필요)")
    L.append("- **Random (no training)**: 학습 없이 각 약물쌍·부작용마다 0~1 난수를 찍어 예측한 기준선(3 seed 평균). "
             "아무 정보 없이 찍으면 모든 지표가 ~50이 나오므로, 점수가 이와 비슷하면 '사실상 무작위 수준'이라는 뜻이다.")
    L.append("- 지표 계산: AUROC/AUPRC/Acc = 원본 DDI-Bench `trainer.py`와 동일, AP@50 = Decagon식 top-50 정밀도. "
             "P@(N/2)=EmerGNN R-Precision은 raw(`summary.csv`)에만 보관.")
    L.append("- raw: 인코더 `results/ddibn_acc/` (summary.csv + per_class + raw_preds.npz), LLM `results/ddibn_acc/llm_metrics/`.")
    L.append("")

    # ── 실험별 저장 위치 (분리 원칙: 신규는 별도 폴더, 기존은 읽기 전용) ──
    L.append("## 실험별 저장 위치")
    L.append("")
    L.append("> 원칙: **신규 실험은 별도 폴더에 저장하고 기존 결과는 덮어쓰지 않는다** (기존은 읽기 전용).")
    L.append("")
    L.append("| 실험 | 저장 위치 | 비고 |")
    L.append("|---|---|---|")
    L.append("| 인코더 14셀 (cell01–14) + transe cell08 | `results/ddibn_acc/` | summary.csv + per_class + configs + raw_preds.npz (3 seed) |")
    L.append("| **cell08 KGE 노드초기값 비교** (distmult/rotate/complex/transh) | `results/kge_compare/{kge}/` | transe 제외 4종, cell08 × 3 seed (run_cell08_2kge.py / run_kge_chain.py) |")
    L.append("| KGE 임베딩 4종 | `kge/ddibn/{kge}/hetionet_{kge}_ent_*.npy` | distmult 80d / rotate 512d / complex 700ep / transh 800ep |")
    L.append("| best KGE 선택 결과 | `results/kge_compare/best.json` | cell08 S0 accuracy 기준 5종(transe 포함) 비교 |")
    L.append("| best KGE로 cell 09/10/11 | `results/ddibn_kge_{best}/` | best≠transe일 때만 (run_kge_chain Phase4) |")
    L.append("| LLM FT 10epoch (full-val, per-split best) | `results/ddibn_acc/llm_metrics/v1ft_{tag}_{split}.json` + `_cv.json` | cv.json = split별 best epoch + val_curve(epoch별 acc/F1) |")
    L.append("| LLM ZS | `results/ddibn_acc/llm_metrics/v1zs_{tag}_{split}.json` | zero-shot |")
    L.append("| 구 LLM FT 3epoch | `results/ft3e_archive/` | 이전 3ep 결과 별도 보관 |")
    L.append("| Random baseline | `results/random_baseline.json` | 3 seed |")
    L.append("| 실행 로그 | `results/run_logs/` | ft10_*, qwen_fulleval, kge4_*, cell08_*, chain_* |")
    L.append("")
    L.append("> 체크포인트 진행 마커: KGE 4종 완주+체인 완료 시 `results/kge_compare/CHAIN_DONE`.")
    L.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[written] {OUT}  (인코더 {n_enc}/14, LLM {n_llm}/6, random {'O' if rand else 'X'})")


if __name__ == "__main__":
    main()
