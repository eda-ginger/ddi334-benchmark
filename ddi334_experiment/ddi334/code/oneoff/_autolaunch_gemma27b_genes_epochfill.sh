#!/bin/bash
# gemma27b GENES(ideal_genes) epoch-fill (S1/S2 ep1·ep2) 자동 launcher — 빈 GPU 열리면 실행.
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/ft; LOG=$OUT/logs/epochfill_gemma27b_genes.log
HF=google/gemma-2-27b-it
CAND=(0 2 4)
TAG=google_gemma-2-27b-it_ideal_genes_seed42
while :; do
  for g in "${CAND[@]}"; do
    free=$(( (81920-$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)) ))
    if [ "${free:-0}" -ge 65000 ]; then
      echo "### $(date '+%m-%d %H:%M') epochfill genes -> GPU$g (free ${free}MB) ###" >> $LOG
      micromamba run -n vllm-llm python code/oneoff/test_epochs_gen.py --model $HF --merge \
        --tags ideal_genes=$TAG --gpu $g --port $((12790+g)) --mem 0.85 \
        --cells ideal_genes:S2:1,ideal_genes:S2:2,ideal_genes:S1:1,ideal_genes:S1:2 >> $LOG 2>&1
      echo "### $(date '+%m-%d %H:%M') epochfill genes 완료 ###" >> $LOG
      curl -s -d "gemma27b genes epoch-fill 완료" ntfy.sh/Latex_project >/dev/null 2>&1
      exit 0
    fi
  done
  sleep 180
done
