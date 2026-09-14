#!/bin/bash
# gemma27b REAL epoch-fill (S1/S2 ep1·ep2) 자동 launcher — 빈 GPU 열리면 실행.
# ideal은 제외(불필요). genes는 학습 완료 후 별도. GPU4(genes 학습) 제외. merge-system(gemma).
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/ft; LOG=$OUT/logs/epochfill_gemma27b.log
HF=google/gemma-2-27b-it
CAND=(1 0)   # 전용 GPU만 (GPU4=genes학습 / 5=sosა공유 merged27B 무리 / 2·3=타인 제외)
TAG=google_gemma-2-27b-it_real_seed42
while :; do
  for g in "${CAND[@]}"; do
    free=$(( (81920-$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)) ))
    ut=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g 2>/dev/null)
    if [ "${free:-0}" -ge 65000 ] && [ "${ut:-100}" -lt 40 ]; then
      echo "### $(date '+%m-%d %H:%M') epochfill real -> GPU$g (free ${free}MB) ###" >> $LOG
      micromamba run -n vllm-llm python code/oneoff/test_epochs_gen.py --model $HF --merge \
        --tags real=$TAG --gpu $g --port $((12770+g)) --mem 0.85 \
        --cells real:S2:1,real:S2:2,real:S1:1,real:S1:2 >> $LOG 2>&1
      echo "### $(date '+%m-%d %H:%M') epochfill real 완료 ###" >> $LOG
      curl -s -d "gemma27b real epoch-fill 완료" ntfy.sh/Latex_project >/dev/null 2>&1
      exit 0
    fi
  done
  sleep 180
done
