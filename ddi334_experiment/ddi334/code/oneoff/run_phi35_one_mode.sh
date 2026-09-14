#!/bin/bash
# Phi-3.5 ZS V2 한 모드를 한 GPU에서 3 split(S0,S2,S1) 순차 실행.
# 모드별로 이 스크립트를 다른 GPU/포트로 동시에 띄워 병렬화.
# usage: run_phi35_one_mode.sh <gpu> <mode> <label> <port>
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
GPU=$1; MODE=$2; LABEL=$3; PORT=$4
OUT=results/ddibn_ablation/info_avail
MODEL=microsoft/Phi-3.5-mini-instruct
for split in S0 S2 S1; do          # S0 먼저, S1(최대) 마지막
  echo "===== $(date +%H:%M) gpu=$GPU split=$split mode=$MODE label=$LABEL ====="
  micromamba run -n vllm-llm python code/llm_infer.py \
    --version v2 --dataset ddibn --model $MODEL \
    --gpu $GPU --split $split --mode $MODE --port $PORT --gpu-mem-util 0.5 \
    --workers 24 --outdir $OUT --label $LABEL
done
echo "===== DONE gpu=$GPU mode=$MODE $(date +%H:%M) ====="
