#!/usr/bin/env python3
"""학습된 KGE(pkl)를 DDI test triple(relation>=23)에만 재평가 -> 순수 DDI-hits.
재학습 X. split_ddi로 나온 DDI test(13k)만 골라 filtered link-prediction 평가.
usage: python eval_ddi_hits.py --model_dir data/kge/ddibn_ddisplit/transe --config <cfg.yaml>
"""
import argparse, sys, os, yaml, json, torch
sys.path.insert(0, "/home/rudwls2717/Latex/experiments/v5_ddi-bench/kge_hetionet/src")
from load_custom_data import load_emergnn_hetionet
from pykeen.evaluation import RankBasedEvaluator

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--ddi_rel_min", type=int, default=23)  # bio 0-22, DDI 23+
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
tr, va, te = load_emergnn_hetionet(cfg)  # split_ddi 재현 (seed 고정)
mt = te.mapped_triples
ddi_mask = mt[:, 1] >= a.ddi_rel_min
ddi_test = mt[ddi_mask]
bio_test = mt[~ddi_mask]
print(f"[eval_ddi_hits] test 총 {len(mt)} | DDI {len(ddi_test)} | bio {len(bio_test)}", flush=True)

model = torch.load(os.path.join(a.model_dir, "trained_model.pkl"), weights_only=False)
model.eval()
ev = RankBasedEvaluator()
filt = [tr.mapped_triples, va.mapped_triples, te.mapped_triples]

def run(triples, tag):
    r = ev.evaluate(model, triples, additional_filter_triples=filt, use_tqdm=False)
    d = r.to_dict()
    br = d["both"]["realistic"]
    print(f"  [{tag}] hits@1={br['hits_at_1']:.4f} hits@3={br['hits_at_3']:.4f} "
          f"hits@10={br['hits_at_10']:.4f} MRR={br['inverse_harmonic_mean_rank']:.4f}", flush=True)
    return {k: br[k] for k in ("hits_at_1", "hits_at_3", "hits_at_10", "inverse_harmonic_mean_rank")}

out = {"ddi": run(ddi_test, "DDI-only"), "bio": run(bio_test, "bio-only")}
json.dump(out, open(os.path.join(a.model_dir, "ddi_hits.json"), "w"), indent=1)
print(f"[eval_ddi_hits] 저장 -> {a.model_dir}/ddi_hits.json", flush=True)
