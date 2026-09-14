#!/bin/bash
# gemma27b S0/S2 재실행 (S1은 done), GPU4 단독 순차, merge-system, mem 0.75
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/info_avail; M=google/gemma-2-27b-it
declare -A LB=( [real]=gemma27b_zs_real [ideal_genes]=gemma27b_zs_genes [ideal]=gemma27b_zs_ideal )
for mode in real ideal_genes ideal; do
  for split in S0 S2; do
    echo "===== $(date +%H:%M) GPU4 $mode $split ====="
    micromamba run -n vllm-llm python code/llm_infer.py --version v2 --dataset ddibn --model $M \
      --gpu 4 --split $split --mode $mode --port 12396 --gpu-mem-util 0.75 --workers 24 --merge-system \
      --outdir $OUT --label ${LB[$mode]}
  done
done
echo "===== gemma27b S0/S2 DONE $(date +%H:%M) ====="
