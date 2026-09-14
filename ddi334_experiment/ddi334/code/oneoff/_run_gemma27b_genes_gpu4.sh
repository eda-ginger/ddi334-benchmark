#!/bin/bash
# gemma27b genes(ideal_genes) 27B QLoRA 학습 + best-epoch eval on GPU4.
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/ft
HF=google/gemma-2-27b-it; TAG=google_gemma-2-27b-it_ideal_genes_seed42; G=4
echo "### $(date '+%m-%d %H:%M') gemma27b genes 학습 시작 GPU$G ###"
micromamba run -n vllm-llm python code/ft_train_v2.py --model $HF --mode ideal_genes \
  --gpu $G --epochs 3 --batch_size 2 --resume >> $OUT/logs/train_gemma27b_genes.log 2>&1
echo "### $(date '+%m-%d %H:%M') 학습 끝 -> eval ###"
micromamba run -n vllm-llm python code/run_ft_cv_v2.py --tag $TAG --model $HF --mode ideal_genes \
  --ckpt results/ft_v2/$TAG --gpu $G --epochs 3 --policy best --gpu-mem-util 0.85 \
  --port 12764 --merge-system >> $OUT/logs/eval_gemma27b_genes.log 2>&1
echo "### $(date '+%m-%d %H:%M') gemma27b genes 완료 ###"
curl -s -d "gemma27b genes 완료" ntfy.sh/Latex_project >/dev/null 2>&1
