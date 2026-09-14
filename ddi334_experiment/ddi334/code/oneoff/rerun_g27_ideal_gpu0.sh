#!/bin/bash
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/info_avail; M=google/gemma-2-27b-it
for split in S0 S2; do
  echo "===== $(date +%H:%M) GPU0 ideal $split ====="
  micromamba run -n vllm-llm python code/llm_infer.py --version v2 --dataset ddibn --model $M \
    --gpu 0 --split $split --mode ideal --port 12397 --gpu-mem-util 0.75 --workers 24 --merge-system \
    --outdir $OUT --label gemma27b_zs_ideal
done
echo "===== g27 ideal S0/S2 DONE $(date +%H:%M) ====="
