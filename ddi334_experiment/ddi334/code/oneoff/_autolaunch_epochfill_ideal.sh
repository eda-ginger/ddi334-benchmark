#!/bin/bash
# 범용 ideal ep1/ep2 epoch-fill 자동 launcher — 빈 GPU 열리면 실행 (기존 _autolaunch_gemma27b_*.sh 패턴 재사용).
# 2026-07-28: Gemma2-2B / Qwen2.5-3B / Phi-3.5 / Phi-4 4개 모델의 ideal 트랙 ep1/ep2 공백 채우기.
# usage: _autolaunch_epochfill_ideal.sh <model_hf_id> <tag> <merge:0|1> <cells> <mem> <minfree_mb> <port> <cand_csv> <logname>
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
MODEL=$1; TAG=$2; MERGE=$3; CELLS=$4; MEM=$5; MINFREE=$6; PORT=$7; IFS=',' read -ra CAND <<< "$8"; NAME=$9
LOG=results/ddibn_ablation/ft/logs/epochfill_${NAME}.log
MERGEFLAG=""; [ "$MERGE" = "1" ] && MERGEFLAG="--merge"
while :; do
  for g in "${CAND[@]}"; do
    free=$(( 81920-$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null) ))
    if [ "${free:-0}" -ge "$MINFREE" ]; then
      echo "### $(date '+%m-%d %H:%M') $NAME -> GPU$g (free ${free}MB) ###" >> $LOG
      micromamba run -n vllm-llm python code/oneoff/test_epochs_gen.py --model "$MODEL" $MERGEFLAG \
        --tags ideal=$TAG --gpu $g --port $PORT --mem $MEM \
        --cells "$CELLS" >> $LOG 2>&1
      echo "### $(date '+%m-%d %H:%M') $NAME 완료 ###" >> $LOG
      exit 0
    fi
  done
  sleep 90
done
