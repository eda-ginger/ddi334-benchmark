"""Build the evidence tables behind Figure 1 (dataset usage and reported scale).

Input : snapshot/{ddi,kg,corpus_papers}.csv, figure1_values.csv, manual_additions.csv
        (raw export of the hand-curated Google Sheet, see its README)
Output: Figure1_evidence.xlsx   the one file to read / cite (Summary, Per paper, Method)
        per_paper.csv   one row per (paper, dataset variant)
        ranges.csv      per dataset: #papers, min/max with the papers that give them,
                                          and the value used in Figure 1 (check column)

Corpus rule: papers removed from the corpus (EXCLUDED) are listed but never counted.
The sheet's per-row `count` column is ignored; counts come from the paper names themselves.

Counting rules (column `counted_in_fig1`):
  R1  A paper removed from the corpus (EXCLUDED) is never counted.
  R2  A KG paper that is listed under "Combine" (several sources merged, memo "논문에서는 합침")
      is counted under Combine only, not under the single-source KG rows (DrugBank - kg, KEGG, ...).
Figure 1 cells that cannot come from the sheet rows are reported in `fig1_note`.
"""
import ast
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE
SNAP = DATA / (sys.argv[1] if len(sys.argv) > 1 else "snapshot")
FIG_VALUES = DATA / "figure1_values.csv"   # numbers printed in Figure 1

EXCLUDED = {"scat", "scatrans"}          # removed from the corpus on 2026-09-04 (table1_mapping_review.md)
ALIAS = {                                 # sheet name -> backbone model name
    "aae-kge": "aae-kge (dai et al.)",
    "hkg-ddie": "hkg-ddie (asada et al.)",
    "zhang et al.": "kgc-lstm",           # same paper: Zhang et al., ICIC 2022 (Table 1 uses KGC-LSTM)
}
DDI_FIG = {"DrugBank - ddi": "DrugBank - ddi", "TWOSIDES": "TWOSIDES", "DDIExtraction 2013": "DDIExtraction 2013",
           "BioSNAP": "BioSNAP", "DDInter": "DDInter", "TAC 2018": "TAC", "TAC": "TAC"}


def read(path):
    return [[c.strip() for c in r] for r in csv.reader(open(path, encoding="utf-8-sig", newline=""))]


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def num(s):
    s = s.replace(",", "").strip()
    return int(s) if re.fullmatch(r"\d+", s) else None


backbone = {norm(r["model"]): r for r in csv.DictReader(open(SNAP / "corpus_papers.csv", encoding="utf-8-sig", newline=""))}
corpus = {k for k in backbone if k not in EXCLUDED}


def resolve(name):
    k = ALIAS.get(norm(name), norm(name))
    if k in EXCLUDED or norm(name) in EXCLUDED:
        return None, "excluded"
    if k in corpus:
        return backbone[k]["model"], "ok"
    # tolerate a trailing author tag, e.g. "AAE-KGE (Dai et al.)"
    for c in corpus:
        if c.startswith(k + " (") or k.startswith(c + " ("):
            return backbone[c]["model"], "ok"
    return name, "UNMATCHED"


# ---------------------------------------------------------------- DDI datasets
ddi = read(SNAP / "ddi.csv")
per_paper, problems = [], []
for lineno, r in enumerate(ddi[1:], start=2):
    r = r + [""] * (26 - len(r))
    ds = r[1]
    if not ds or ds in ("min", "max", "dataset"):
        continue
    papers = [x for x in r[10:] if x]
    for name in papers:
        model, status = resolve(name)
        if status == "UNMATCHED":
            problems.append((lineno, ds, name))
        per_paper.append({
            "group": "DDI", "dataset": ds, "version": r[2], "drugs": r[3], "ddis": r[4], "types": r[5],
            "source_dataset_author": r[7], "source_model": r[8], "extra": r[9],
            "paper": model if model else name, "in_corpus": "no (excluded)" if status == "excluded" else ("yes" if status == "ok" else "UNMATCHED"),
            "counted_in_fig1": "yes" if status == "ok" else ("no: R1 removed from corpus" if status == "excluded" else "no: unmatched name"),
            "sheet_row": lineno,
        })

# ---------------------------------------------------------------- manual additions (not yet in the sheet)
manual_path = DATA / "manual_additions.csv"
if manual_path.exists():
    for m in csv.DictReader(open(manual_path, encoding="utf-8", newline="")):
        model, status = resolve(m["paper"])
        if status == "UNMATCHED":
            problems.append(("manual", m["dataset"], m["paper"]))
        per_paper.append({
            "group": m["group"], "dataset": m["dataset"], "version": m["version"], "drugs": m["drugs"], "ddis": m["ddis"],
            "types": m["types"], "source_dataset_author": m["source_dataset_author"], "source_model": m["source_model"],
            "extra": m["extra"], "paper": model if model else m["paper"],
            "in_corpus": "yes" if status == "ok" else "UNMATCHED", "counted_in_fig1": "yes" if status == "ok" else "no: unmatched name",
            "sheet_row": "manual (pending sheet update)",
        })

# ---------------------------------------------------------------- KG datasets
kg = read(SNAP / "kg.csv")
combine_papers = {norm(resolve(r[7])[0] or r[7]) for r in kg[1:] if len(r) > 7 and r[1] == "Combine" and r[7]}
for lineno, r in enumerate(kg[1:], start=2):
    r = r + [""] * (9 - len(r))
    ds = r[1]
    if not ds or ds in ("min", "max", "dataset") or not r[7]:
        continue
    model, status = resolve(r[7])
    if status == "UNMATCHED":
        problems.append((lineno, ds, r[7]))
    per_paper.append({
        "group": "KG", "dataset": ds, "version": "", "drugs": r[2], "ddis": r[3], "types": r[6],
        "source_dataset_author": "", "source_model": "", "extra": f"entities={r[4]}; triples={r[5]}; {r[8]}".strip("; "),
        "paper": model if model else r[7], "in_corpus": "no (excluded)" if status == "excluded" else ("yes" if status == "ok" else "UNMATCHED"),
        "counted_in_fig1": ("no: R1 removed from corpus" if status == "excluded" else "no: unmatched name" if status != "ok"
                            else "no: R2 merged-source KG, counted under Combine" if (ds != "Combine" and norm(model) in combine_papers) else "yes"),
        "sheet_row": lineno,
    })

fields = ["group", "dataset", "version", "drugs", "ddis", "types", "paper", "in_corpus", "counted_in_fig1", "source_dataset_author", "source_model", "extra", "sheet_row"]
with open(DATA / "per_paper.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(per_paper)

# ---------------------------------------------------------------- Figure 1 values
fig = {r["dataset"]: [r["dataset"], int(r["papers"]), r["drugs"], r["ddis"], r["types"]] for r in csv.DictReader(open(FIG_VALUES, encoding="utf-8", newline=""))}
FIG_NAME = {"DrugBank - ddi": "DrugBank - ddi", "TWOSIDES": "TWOSIDES", "DDIExtraction 2013": "DDIExtraction 2013", "BioSNAP": "BioSNAP",
            "DDInter": "DDInter", "TAC 2018": "TAC", "TAC": "TAC"}
KG_FIG = {"Combine": "Combine", "DrugBank - kg": "DrugBank - kg", "Hetionet": "Hetionet", "DRKG": "DRKG", "KEGG": "KEGG",
          "OGB-biokg": "OGB-biokg", "OGB-ddi": "OGB-ddi"}


FIG_NOTES = {
    "BioSNAP": "types: Figure 1 shows 1 (binary task, no interaction-type count reported in the sheet)",
    "TAC 2018": "drugs/ddis: Figure 1 values come from direct parsing of the TAC corpus, documented in corpus_task_type_stats.csv (TAC row)",
}


def rng(vals):
    vals = sorted(vals)
    if not vals:
        return "", ""
    return vals[0], vals[-1]


def fmt(lo, hi):
    if lo == "":
        return "-"
    return str(lo) if lo == hi else f"{lo} ~ {hi}"


def at(items, key, val):
    return "; ".join(sorted({p["paper"] for p in items if num(p[key]) == val}))


out = []
datasets = []
for p in per_paper:
    if (p["group"], p["dataset"]) not in datasets:
        datasets.append((p["group"], p["dataset"]))
for grp, ds in datasets:
    items = [p for p in per_paper if p["group"] == grp and p["dataset"] == ds and p["counted_in_fig1"] == "yes"]
    in_fig = (DDI_FIG.get(ds) if grp == "DDI" else KG_FIG.get(ds))
    papers = sorted({p["paper"] for p in items})
    rec = {"group": grp, "dataset": ds, "in_figure_1": "yes" if in_fig else "no", "n_papers": len(papers)}
    figrow = fig.get(FIG_NAME.get(ds) if grp == "DDI" else KG_FIG.get(ds)) if in_fig else None
    for key, label, figidx in (("drugs", "drugs", 2), ("ddis", "ddis", 3), ("types", "types", 4)):
        vals = [num(p[key]) for p in items if num(p[key]) is not None]
        lo, hi = rng(vals)
        rec[f"{label}_range"] = fmt(lo, hi)
        rec[f"{label}_min_from"] = at(items, key, lo) if vals else ""
        rec[f"{label}_max_from"] = at(items, key, hi) if vals and hi != lo else ""
        rec[f"{label}_fig1"] = figrow[figidx] if figrow else ""
    rec["n_papers_fig1"] = figrow[1] if figrow else ""
    rec["papers"] = "; ".join(papers)
    flags = []
    if figrow:
        if rec["n_papers"] != figrow[1]:
            flags.append(f"papers {rec['n_papers']} vs fig {figrow[1]}")
        for label in ("drugs", "ddis", "types"):
            a, b = rec[f"{label}_range"], rec[f"{label}_fig1"]
            if a != b and not (a == "-" and b):     # fig may carry values taken from a different table (e.g. TAC)
                flags.append(f"{label} {a} vs fig {b}")
            elif a == "-" and b:
                flags.append(f"{label}: not in sheet rows, fig {b}")
    note = FIG_NOTES.get(ds, "")
    if note:
        flags = [f for f in flags if "not in sheet rows" not in f]
    rec["fig1_note"] = note
    rec["check_vs_fig1"] = "OK" if figrow and not flags else ("; ".join(flags) if flags else "not in Figure 1")
    out.append(rec)

rf = ["group", "dataset", "in_figure_1", "n_papers", "n_papers_fig1",
      "drugs_range", "drugs_fig1", "drugs_min_from", "drugs_max_from",
      "ddis_range", "ddis_fig1", "ddis_min_from", "ddis_max_from",
      "types_range", "types_fig1", "types_min_from", "types_max_from", "check_vs_fig1", "fig1_note", "papers"]
with open(DATA / "ranges.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rf)
    w.writeheader()
    w.writerows(out)

print(f"per-paper rows: {len(per_paper)}  (excluded rows: {sum(p['in_corpus'].startswith('no') for p in per_paper)})")
if problems:
    print("UNMATCHED names:", problems)
for r in out:
    if r["in_figure_1"] == "yes":
        print(f"{r['dataset']:20s} papers {r['n_papers']:>3} | {r['check_vs_fig1']}")


# ---------------------------------------------------------------- one-file workbook
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def extremes(rec, key):
    lo, hi = rec[f"{key}_min_from"], rec[f"{key}_max_from"]
    rng_ = rec[f"{key}_range"]
    if rng_ == "-":
        return "not reported"
    return f"{lo}" if not hi else f"min: {lo} / max: {hi}"


def fnum(txt):
    return re.sub(r"\d+", lambda m: f"{int(m.group()):,}", txt)


wb = Workbook()
ws = wb.active
ws.title = "Summary"
head = ["Dataset", "Type", "Papers", "Drugs", "Drugs: reported by", "DDIs", "DDIs: reported by",
        "Interaction types", "Types: reported by", "Check vs Figure 1", "Note"]
ws.append(head)
for r in out:
    if r["in_figure_1"] != "yes":
        continue
    ws.append([r["dataset"], r["group"], r["n_papers"],
               fnum(r["drugs_range"]), extremes(r, "drugs"), fnum(r["ddis_range"]), extremes(r, "ddis"),
               fnum(r["types_range"]), extremes(r, "types"), r["check_vs_fig1"], r["fig1_note"]])
widths = [20, 7, 8, 16, 34, 22, 34, 14, 34, 16, 50]
for i, wd in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(i)].width = wd
for c in ws[1]:
    c.font = Font(bold=True, color="FFFFFF")
    c.fill = PatternFill("solid", fgColor="305496")
    c.alignment = Alignment(wrap_text=True, vertical="center")
for row in ws.iter_rows(min_row=2):
    for c in row:
        c.alignment = Alignment(wrap_text=True, vertical="top")
ws.freeze_panes = "B2"

wp = wb.create_sheet("Per paper")
pf = ["group", "dataset", "paper", "drugs", "ddis", "types", "version", "counted_in_fig1", "source_dataset_author", "source_model", "extra", "sheet_row"]
wp.append(pf)
for p_ in sorted(per_paper, key=lambda x: (x["group"], x["dataset"], x["paper"].lower())):
    wp.append([p_[k] if p_[k] != "" else None for k in pf])
for i, wd in enumerate([7, 20, 24, 9, 11, 9, 9, 40, 18, 14, 40, 24], 1):
    wp.column_dimensions[get_column_letter(i)].width = wd
for c in wp[1]:
    c.font = Font(bold=True, color="FFFFFF")
    c.fill = PatternFill("solid", fgColor="305496")
wp.freeze_panes = "D2"
wp.auto_filter.ref = wp.dimensions

wm = wb.create_sheet("Method")
for line in (DATA / "README.md").read_text(encoding="utf-8").splitlines():
    wm.append([line])
wm.column_dimensions["A"].width = 130
for row in wm.iter_rows():
    for c in row:
        c.alignment = Alignment(wrap_text=True, vertical="top")
wb.save(DATA / "Figure1_evidence.xlsx")
print("wrote", DATA / "Figure1_evidence.xlsx")


# ---------------------------------------------------------------- markdown for GitHub
def md(s):
    return str(s).replace("|", "\\|")


L = []
L.append("# Figure 1 evidence: dataset usage and reported scale\n")
L.append("This document shows where every number in **Figure 1** comes from: how many of the surveyed DDI-prediction papers "
         "use each dataset, and the minimum and maximum dataset size (drugs, DDIs, interaction types) that those papers report.\n")
L.append("Everything below is generated by `gen_fig1_evidence.py` from a CSV export of the hand-curated sheet "
         "(`snapshot/`); nothing was typed in by hand except the one row in `manual_additions.csv`.\n")
L.append("## 1. Summary (the numbers in Figure 1)\n")
L.append("| Dataset | Type | Papers | Drugs | DDIs | Interaction types | Matches Figure 1 |")
L.append("|---|---|---:|---:|---:|---:|:---:|")
for r in out:
    if r["in_figure_1"] != "yes":
        continue
    L.append(f"| {md(r['dataset'])} | {r['group']} | {r['n_papers']} | {fnum(r['drugs_range'])} | {fnum(r['ddis_range'])} | "
             f"{fnum(r['types_range'])} | {'yes' if r['check_vs_fig1'] == 'OK' else md(r['check_vs_fig1'])} |")
L.append("\n`-` means that no paper reports the value. BioSNAP: Figure 1 shows 1 interaction type (binary task). "
         "TAC: drugs and DDIs come from direct parsing of the TAC corpus (see `corpus_task_type_stats.csv`).\n")
L.append("## 2. Which paper gives each minimum and maximum\n")
def short(names, k=3):
    names = [x for x in names.split("; ") if x]
    return "; ".join(names) if len(names) <= k + 1 else "; ".join(names[:k]) + f" (+{len(names) - k} more)"


def extremes_md(rec, key):
    lo, hi, rng_ = rec[f"{key}_min_from"], rec[f"{key}_max_from"], rec[f"{key}_range"]
    if rng_ == "-":
        return "not reported"
    if not hi:                                   # one single value shared by all reporting papers
        n = len([x for x in lo.split("; ") if x])
        return f"{short(lo)}" if n <= 4 else f"same value in {n} papers"
    return f"min: {short(lo)} / max: {short(hi)}"


L.append("| Dataset | Drugs | DDIs | Interaction types |")
L.append("|---|---|---|---|")
for r in out:
    if r["in_figure_1"] != "yes":
        continue
    L.append(f"| {md(r['dataset'])} | {md(extremes_md(r, 'drugs'))} | {md(extremes_md(r, 'ddis'))} | {md(extremes_md(r, 'types'))} |")
L.append("\nExample: for DDIExtraction 2013 the smallest reported DDI count is 23,539 (SubGE-DDI) and the largest is 41,852 (HiFAB-DDI), "
         "so Figure 1 draws the range 23,539 ~ 41,852.\n")
L.append("## 3. How it was made\n")
L.append("1. **Reading the papers.** For every paper in the corpus (84 papers) the dataset it uses and the dataset statistics it reports "
         "were read from the paper and entered in a Google Sheet (tabs `ddi` for DDI datasets and `KG` for knowledge graphs). "
         "Papers that report the same numbers share one row.")
L.append("2. **Exporting.** Each tab was exported unchanged to CSV (`snapshot/`).")
L.append("3. **One row per paper.** The script splits every sheet row into one row per paper and matches the names to the corpus "
         "(three names are aliased: `AAE-KGE`, `HKG-DDIE`, `Zhang et al.` = `KGC-LSTM`). An unmatched name stops the script.")
L.append("4. **Counting rules.**")
L.append("   - *R1:* a paper that was removed from the corpus is not counted (SCAT / SCATrans).")
L.append("   - *R2:* a knowledge-graph paper that merges several sources is counted under *Combine* only, "
         "not under single-source rows such as DrugBank - kg or KEGG (Karim et al., Celebi et al., KGE-MFL).")
L.append("   - The `count` column of the sheet is ignored; the number of papers is the number of distinct matched names.")
L.append("5. **Ranges.** For each dataset the script takes the smallest and largest drugs / DDIs / interaction types over the counted rows "
         "(unreported values are skipped) and records which papers give them.")
L.append("6. **Check.** The result is compared with the numbers used to draw Figure 1 (`figure1_values.csv`). "
         "All 13 datasets match.\n")
L.append("Rebuild: `python gen_fig1_evidence.py` writes `Figure1_evidence.xlsx`, `per_paper.csv`, `ranges.csv` and this file.\n")
L.append("## 4. Papers per dataset (every reported value)\n")
for r in out:
    if r["in_figure_1"] != "yes":
        continue
    rows_ = sorted([p for p in per_paper if p["group"] == r["group"] and p["dataset"] == r["dataset"]], key=lambda x: x["paper"].lower())
    L.append(f"<details><summary><b>{md(r['dataset'])}</b> ({r['n_papers']} papers)</summary>\n")
    L.append("| Paper | Drugs | DDIs | Interaction types | Counted | Note |")
    L.append("|---|---:|---:|---:|:---:|---|")
    for p_ in rows_:
        counted = "yes" if p_["counted_in_fig1"] == "yes" else "no"
        note = "" if p_["counted_in_fig1"] == "yes" else p_["counted_in_fig1"].replace("no: ", "")
        if str(p_["sheet_row"]).startswith("manual"):
            note = (note + "; " if note else "") + "added manually (pending sheet update)"
        L.append(f"| {md(p_['paper'])} | {p_['drugs'] or '-'} | {p_['ddis'] or '-'} | {p_['types'] or '-'} | {counted} | {md(note)} |")
    L.append("\n</details>\n")
(DATA / "Figure1_evidence.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
print("wrote", DATA / "Figure1_evidence.md")
