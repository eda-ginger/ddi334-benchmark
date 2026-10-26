# Figure 1: statistics behind the figure

Figure 1 of the manuscript shows, for each dataset used by the surveyed DDI-prediction papers, how many papers use it
and the smallest and largest dataset size those papers report (drugs, DDIs, interaction types).

## Summary (the numbers in Figure 1)

| Dataset | Papers | Drugs | DDIs | Interaction types |
|---|---:|---:|---:|---:|
| DrugBank - ddi | 55 | 567 ~ 2,004 | 17,119 ~ 556,757 | 65 ~ 175 |
| TWOSIDES | 35 | 182 ~ 1,345 | 346 ~ 4,651,131 | 40 ~ 1,318 |
| DDIExtraction 2013 | 15 | 17,805 ~ 18,502 | 23,539 ~ 41,852 | 4 |
| BioSNAP | 8 | 957 ~ 1,514 | 20,624 ~ 48,514 | - |
| DDInter | 4 | 1,493 ~ 1,689 | 117,608 ~ 152,887 | 3 |
| TAC 2018 | 4 | - | - | 3 |
| Combine | 7 | 2,322 ~ 12,000 | 0 ~ 2,898,937 | 5 ~ 445 |
| DrugBank - kg | 7 | 708 ~ 3,797 | 10,036 ~ 1,236,361 | 5 ~ 74 |
| Hetionet | 6 | 1,552 | 6,486 | 23 ~ 24 |
| DRKG | 5 | 24,313 | 1,385,757 | 107 |
| KEGG | 5 | 786 ~ 1,925 | 13,787 ~ 56,983 | 167 ~ 168 |
| OGB-biokg | 5 | 808 ~ 10,533 | 111,520 ~ 1,195,972 | 13 ~ 51 |
| OGB-ddi | 1 | 3,825 | 266,978 | 20 |

`-`: no paper reports the value. BioSNAP: Figure 1 shows 1 interaction type (binary task). TAC: drugs and DDIs are from direct parsing of the TAC corpus.

## Per-paper values

[`figure1_per_paper.csv`](figure1_per_paper.csv) lists, for every dataset above, each paper and the value it reports
(drugs, DDIs, interaction types). The ranges in the summary are the minimum and maximum of this table.

How the counting was done:
- A paper is counted once per dataset it uses; the paper count is the number of distinct papers.
- A knowledge-graph paper that merges several sources (Karim et al., Celebi et al., KGE-MFL) is counted under *Combine* only, not under the single-source rows (`counted_in_figure1 = no`).
- The surveyed corpus has 84 papers.
