#!/bin/bash
# Phi-4 (14B) ZS V2 정보 가용성 ablation: 한 모드를 한 GPU에서 3 split(S0,S2,S1) 순차.
# Phi-3.5용(run_phi35_one_mode.sh)과 동일 구조, 모델·label prefix만 phi4.
# usage: run_phi4_one_mode.sh <gpu> <mode> <label> <port>
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
GPU=$1; MODE=$2; LABEL=$3; PORT=$4
OUT=results/ddibn_ablation/info_avail
MODEL=microsoft/phi-4
for split in S0 S2 S1; do
  echo "===== $(date +%H:%M) gpu=$GPU split=$split mode=$MODE label=$LABEL ====="
  micromamba run -n vllm-llm python code/llm_infer.py \
    --version v2 --dataset ddibn --model $MODEL \
    --gpu $GPU --split $split --mode $MODE --port $PORT --gpu-mem-util 0.5 \
    --workers 24 --outdir $OUT --label $LABEL
done
echo "===== DONE gpu=$GPU mode=$MODE $(date +%H:%M) ====="
