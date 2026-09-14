#!/usr/bin/env python3
"""probe_dist.json -> V1(multi-label) vs V2(binary) 답 분포 plot.
실제 probe 수집값만 사용. 출력: meeting/fig_zs_answer_dist.png
Run: micromamba run -n DDIBench python plot_dist.py
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(HERE, "probe_dist.json")))
ty = np.array(d["true"]); v1 = np.array(d["v1_yes"]); v2p = np.array(d["v2_pyes"]); v2 = (v2p > 0.5).astype(int)
n, npos, nneg = len(ty), int((ty == 1).sum()), int((ty == 0).sum())

fig, ax0 = plt.subplots(1, 1, figsize=(6.2, 4.4))

# predicted-Yes rate, V1 vs V2, split by true label
groups = ["V1\n(multi-label JSON)", "V2\n(binary Yes/No)"]
pos_rate = [v1[ty == 1].mean() * 100, v2[ty == 1].mean() * 100]
neg_rate = [v1[ty == 0].mean() * 100, v2[ty == 0].mean() * 100]
x = np.arange(2); w = 0.35
b1 = ax0.bar(x - w/2, pos_rate, w, label=f"true-positive (n={npos})", color="#d9534f")
b2 = ax0.bar(x + w/2, neg_rate, w, label=f"true-negative (n={nneg})", color="#5b9bd5")
ax0.set_xticks(x); ax0.set_xticklabels(groups)
ax0.set_ylabel("Predicted-Yes rate (%)")
ax0.set_title(f"ZS answer distribution (gemma-2-2b, {n} queries)")
ax0.set_ylim(0, max(10, max(pos_rate + neg_rate) * 1.4))
ax0.legend(fontsize=8)
for bars in (b1, b2):
    for r in bars:
        ax0.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width()/2, r.get_height()),
                     ha="center", va="bottom", fontsize=8)
ax0.text(0.5, -0.16, "same Yes-rate for pos vs neg = no discrimination",
         transform=ax0.transAxes, ha="center", fontsize=8, color="#555")

fig.tight_layout()
out = "/home/rudwls2717/Latex/meeting/fig_zs_answer_dist.png"
fig.savefig(out, dpi=130)
print("saved:", out)
print(f"V1 Yes-rate pos/neg = {pos_rate[0]:.1f}/{neg_rate[0]:.1f}%  | V2 Yes-rate pos/neg = {pos_rate[1]:.1f}/{neg_rate[1]:.1f}%")
