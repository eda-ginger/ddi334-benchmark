#!/bin/bash
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
LOG=results/ddibn_ablation/ft/logs/epochfill_qwen14b_genes.log
CAND=(3 5 2 1 0)   # GPU4=djk0706 제외
while :; do
  for g in "${CAND[@]}"; do
    free=$(( (81920-$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)) ))
    ut=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g 2>/dev/null)
    if [ "${free:-0}" -ge 44000 ] && [ "${ut:-100}" -lt 55 ]; then
      echo "### $(date +%H:%M) qwen14b genes epoch-fill -> GPU$g (free ${free}MB util ${ut}%) ###"
      micromamba run -n vllm-llm python code/oneoff/test_epochs_gen.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --tags ideal_genes=Qwen_Qwen2.5-14B-Instruct_ideal_genes_seed42 \
        --gpu $g --port $((12730+g)) --mem 0.5 \
        --cells ideal_genes:S2:1,ideal_genes:S2:2,ideal_genes:S1:1,ideal_genes:S1:2 \
        >> $LOG 2>&1
      echo "### $(date +%H:%M) DONE ###"
      exit 0
    fi
  done
  sleep 120
done
