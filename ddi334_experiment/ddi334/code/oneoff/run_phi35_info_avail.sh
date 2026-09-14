#!/bin/bash
# Phi-3.5 ZS V2 정보 가용성 ablation: 3모드 x 3split = 9런
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/info_avail
MODEL=microsoft/Phi-3.5-mini-instruct
declare -A LB=( [real]=phi35_zs_real [ideal_genes]=phi35_zs_genes [ideal]=phi35_zs_ideal )
for split in S0 S2 S1; do            # S0 먼저, S1(최대) 마지막
  for mode in real ideal_genes ideal; do
    echo "===== $(date +%H:%M) split=$split mode=$mode label=${LB[$mode]} ====="
    micromamba run -n vllm-llm python code/llm_infer.py \
      --version v2 --dataset ddibn --model $MODEL \
      --gpu 4 --split $split --mode $mode --port 12355 --gpu-mem-util 0.5 \
      --workers 24 --outdir $OUT --label ${LB[$mode]}
  done
done
echo "===== ALL DONE $(date +%H:%M) ====="
