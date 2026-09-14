#!/bin/bash
# gemma27b genes(ideal_genes) 27B QLoRA 학습+eval 자동 launcher.
# - 우리 GPU(1 우선, 0 차선)만 사용: GPU4=djk0706 / GPU5=sosa / 2·3=타인 회피.
# - 조건: free>=45GB & util<40 되면 학습(batch2,3ep)→eval(policy best,merge-system).
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/ft; mkdir -p $OUT/logs
HF=google/gemma-2-27b-it
TAG=google_gemma-2-27b-it_ideal_genes_seed42
CAND=(1 0)   # 우리 GPU만
while :; do
  for g in "${CAND[@]}"; do
    free=$(( (81920-$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)) ))
    ut=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g 2>/dev/null)
    if [ "${free:-0}" -ge 45000 ] && [ "${ut:-100}" -lt 40 ]; then
      echo "### $(date '+%m-%d %H:%M') gemma27b genes 학습 -> GPU$g (free ${free}MB util ${ut}%) ###"
      micromamba run -n vllm-llm python code/ft_train_v2.py --model $HF --mode ideal_genes \
        --gpu $g --epochs 3 --batch_size 2 --resume \
        >> $OUT/logs/train_gemma27b_genes.log 2>&1
      echo "### $(date '+%m-%d %H:%M') 학습 끝, eval 시작 ###"
      micromamba run -n vllm-llm python code/run_ft_cv_v2.py --tag $TAG --model $HF --mode ideal_genes \
        --ckpt results/ft_v2/$TAG --gpu $g --epochs 3 --policy best --gpu-mem-util 0.85 \
        --port $((12760+g)) --merge-system \
        >> $OUT/logs/eval_gemma27b_genes.log 2>&1
      echo "### $(date '+%m-%d %H:%M') gemma27b genes 완료 ###"
      curl -s -d "gemma27b genes 학습+eval 완료" ntfy.sh/Latex_project >/dev/null 2>&1
      exit 0
    fi
  done
  sleep 180
done
