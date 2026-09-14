#!/bin/bash
# DDI-334 V2 FT 마스터: 6모델 x {real,ideal} = 12 pair. GPU0/GPU4 2스트림 분산.
# 각 pair: ft_train_v2(5ep, QLoRA 4bit) -> run_ft_cv_v2(split별 best epoch eval).
# 작은 모델 우선. merge/batch/mem 모델별.
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/ft; mkdir -p $OUT/logs

# model | hf | merge | batch | evmem | trport | evport
declare -A HF=( [gemma2b]=google/gemma-2-2b-it [qwen3b]=Qwen/Qwen2.5-3B-Instruct [phi35]=microsoft/Phi-3.5-mini-instruct [phi4]=microsoft/phi-4 [qwen14b]=Qwen/Qwen2.5-14B-Instruct [gemma27b]=google/gemma-2-27b-it )
declare -A MERGE=( [gemma2b]=1 [qwen3b]=0 [phi35]=0 [phi4]=0 [qwen14b]=0 [gemma27b]=1 )
declare -A BATCH=( [gemma2b]=8 [qwen3b]=8 [phi35]=8 [phi4]=4 [qwen14b]=4 [gemma27b]=2 )
declare -A EVMEM=( [gemma2b]=0.4 [qwen3b]=0.4 [phi35]=0.4 [phi4]=0.6 [qwen14b]=0.6 [gemma27b]=0.85 )

run_pair () {  # $1=gpu $2=model $3=mode $4=trport $5=evport
  local gpu=$1 m=$2 mode=$3 trp=$4 evp=$5
  local hf=${HF[$m]} merge=${MERGE[$m]} batch=${BATCH[$m]} evmem=${EVMEM[$m]}
  local mflag=""; [ "$merge" = "1" ] && mflag="--merge-system"
  local tag="${hf//\//_}_${mode}_seed42"
  echo "### $(date +%H:%M) TRAIN $m/$mode GPU$gpu ###"
  micromamba run -n vllm-llm python code/ft_train_v2.py --model $hf --mode $mode --gpu $gpu \
    --epochs 5 --batch_size $batch --resume > $OUT/logs/train_${m}_${mode}.log 2>&1
  echo "### $(date +%H:%M) EVAL $m/$mode GPU$gpu ###"
  micromamba run -n vllm-llm python code/run_ft_cv_v2.py --tag $tag --model $hf --mode $mode \
    --ckpt results/ft_v2/$tag --gpu $gpu --epochs 5 --gpu-mem-util $evmem --port $evp $mflag \
    > $OUT/logs/eval_${m}_${mode}.log 2>&1
  echo "### $(date +%H:%M) DONE $m/$mode ###"
}

worker () {  # $1=gpu  $2..=pairs(m:mode)
  local gpu=$1; shift
  local trp=$((12400 + gpu*10)) evp=$((12450 + gpu*10))
  for pm in "$@"; do
    local m=${pm%%:*} mode=${pm##*:}
    run_pair $gpu $m $mode $trp $evp
  done
  echo "===== WORKER GPU$gpu ALL DONE $(date +%H:%M) ====="
}

# 우선순위 작은->큰, 2 GPU 균형 분배
worker 0 gemma2b:real phi35:real qwen14b:real gemma2b:ideal phi35:ideal qwen14b:ideal > $OUT/logs/worker_gpu0.log 2>&1 &
worker 4 qwen3b:real phi4:real gemma27b:real qwen3b:ideal phi4:ideal gemma27b:ideal > $OUT/logs/worker_gpu4.log 2>&1 &
wait
echo "===== ALL 12 FT DONE $(date +%H:%M) ====="
curl -s -d "DDI-334 FT 12런(6모델 x real/ideal) 전부 완료" ntfy.sh/Latex_project >/dev/null
