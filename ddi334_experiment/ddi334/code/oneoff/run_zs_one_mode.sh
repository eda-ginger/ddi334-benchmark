#!/bin/bash
# 범용 ZS V2 ablation: 한 모델의 한 모드를 한 GPU에서 3 split(S0,S2,S1) 순차.
# usage: run_zs_one_mode.sh <gpu> <mode> <label> <port> <model> <merge:0|1> <memutil>
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
GPU=$1; MODE=$2; LABEL=$3; PORT=$4; MODEL=$5; MERGE=$6; MEM=$7
OUT=results/ddibn_ablation/info_avail
MERGEFLAG=""; [ "$MERGE" = "1" ] && MERGEFLAG="--merge-system"
for split in S0 S2 S1; do
  echo "===== $(date +%H:%M) gpu=$GPU split=$split mode=$MODE label=$LABEL ====="
  micromamba run -n vllm-llm python code/llm_infer.py \
    --version v2 --dataset ddibn --model $MODEL \
    --gpu $GPU --split $split --mode $MODE --port $PORT --gpu-mem-util $MEM \
    --workers 24 $MERGEFLAG --outdir $OUT --label $LABEL
done
echo "===== DONE gpu=$GPU mode=$MODE label=$LABEL $(date +%H:%M) ====="
