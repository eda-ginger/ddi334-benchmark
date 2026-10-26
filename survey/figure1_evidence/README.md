# Figure 1 evidence tables

Evidence for the dataset usage counts and the reported-scale ranges in Figure 1.

**Start with `Figure1_evidence.xlsx`** (one file, three sheets): *Summary* (one row per dataset, with the paper(s)
that give each minimum / maximum), *Per paper* (every reported value), *Method* (this text).
The CSV files are the same content in plain text, plus the raw sheet export.

| file | content |
|---|---|
| `Figure1_evidence.xlsx` | the evidence table to read and cite |
| `per_paper.csv`, `ranges.csv` | plain-text versions of the *Per paper* and *Summary* sheets |
| `snapshot/` | unchanged CSV export of the hand-curated Google Sheet everything is built from |
| `manual_additions.csv` | rows not yet in the sheet (see below) |
| `gen_fig1_evidence.py` | script that rebuilds all of the above from the snapshot |

## How the tables were built

1. **Source.** Every paper was read and its dataset statistics were entered by hand into the
   `ddi` and `KG` tabs of the sheet (one row per distinct reported value; the papers that reported it
   are listed on that row). The sheet is exported tab by tab (`export?format=csv&gid=...`) into
   `sheet_snapshot_<date>/` without modification.
2. **Expansion.** `gen_fig1_evidence.py` expands each sheet row into one row per paper and matches the
   paper names to the 84-paper corpus (`sheet1_backbone`). A sheet name that differs from the corpus name is
   resolved through a small alias table (`AAE-KGE`, `HKG-DDIE`, `Zhang et al. = KGC-LSTM`);
   any name that cannot be matched stops the check with an `UNMATCHED` error.
3. **Counting rules.**
   - R1: papers removed from the corpus are never counted (SCAT/SCATrans, removed on 2026-09-04 because
     no axis of Table 1 could be formed from its Doc2Vec / Graph2Vec components).
   - R2: a KG paper that merges several sources (listed under *Combine*, memo "논문에서는 합침") is counted
     under *Combine* only, not under single-source KG rows such as *DrugBank - kg* or *KEGG*.
   - The sheet's own `count` column is ignored; the number of papers is the number of distinct matched names.
4. **Ranges.** For each dataset, min and max of drugs, DDIs, and interaction types over the counted rows
   (unreported values, written `-`, are skipped), together with the papers at the extremes.
5. **Check.** The result is compared with the numbers hard-coded in the Figure 1 script
   (`figure1_values.csv`); `check_vs_fig1` is `OK` when they agree.

Rows that are not yet in the sheet can be added in `manual_additions.csv`; they are merged into the per-paper
table with `sheet_row = manual (pending sheet update)` and removed once the sheet contains them
(currently one row: ACDGNN, DrugBank - ddi, Ryu et al. 1,704 drugs / 191,400 DDIs / 86 types).

Figure 1 values that cannot be derived from the sheet rows are noted in `fig1_note`
(BioSNAP: one interaction type because the task is binary; TAC: drugs and DDIs from direct parsing of the TAC corpus,
described in `corpus_task_type_stats.csv`).

Reproduce: `python gen_fig1_evidence.py [snapshot_dir_name]` (default: `snapshot`).
