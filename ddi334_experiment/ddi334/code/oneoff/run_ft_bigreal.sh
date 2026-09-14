#!/bin/bash
# DDI-334 대형 2모델(Qwen14B/Gemma27B) x {real, ideal} FT 재실행 디스패처 (3ep, best-epoch).
# - run_ft_dispatch.sh 축소판: 4 pair만, eval policy=both (best-epoch+ep3 둘 다 저장).
# - done 판정 = bestDONE 마커. claim = atomic mkdir 락. gpu busy 마커로 1 GPU=1 pair.
# - HP = 작은모델과 동일(lr 2e-4, r16/a32, nf4). qwen14b bs4 / gemma27b bs2+merge.
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/ft; Q=$OUT/_q; mkdir -p $Q $OUT/logs

declare -A HF=( [qwen14b]=Qwen/Qwen2.5-14B-Instruct [gemma27b]=google/gemma-2-27b-it )
declare -A MERGE=( [qwen14b]=0 [gemma27b]=1 )
declare -A BATCH=( [qwen14b]=4 [gemma27b]=2 )
declare -A EVMEM=( [qwen14b]=0.6 [gemma27b]=0.85 )
declare -A NEED=( [qwen14b]=38 [gemma27b]=60 )
PAIRS=(qwen14b:real qwen14b:ideal gemma27b:real gemma27b:ideal)
ALLOWED=(1 4 5 0)   # 2/3 = djk0706 몫 제외

tag_of(){ echo "${HF[$1]//\//_}_${2}_seed42"; }
is_done(){ local t=$(tag_of $1 $2); [ -f "$OUT/llm_metrics/v2ft_${t}_bestDONE" ]; }
freegb(){ echo $(( (81920-$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)) / 1024 )); }
util(){ nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $1 2>/dev/null; }

run_pair(){  # $1=gpu $2=model $3=mode $4=port
  local gpu=$1 m=$2 mode=$3 base=$4 hf=${HF[$2]} merge=${MERGE[$2]} batch=${BATCH[$2]} evmem=${EVMEM[$2]}
  local mflag=""; [ "$merge" = "1" ] && mflag="--merge-system"; local tag=$(tag_of $m $mode)
  echo "### $(date +%H:%M) BIGRE-DISPATCH $m/$mode -> GPU$gpu ###"
  micromamba run -n vllm-llm python code/ft_train_v2.py --model $hf --mode $mode --gpu $gpu \
    --epochs 3 --batch_size $batch --resume > $OUT/logs/train_${m}_${mode}.log 2>&1
  micromamba run -n vllm-llm python code/run_ft_cv_v2.py --tag $tag --model $hf --mode $mode \
    --ckpt results/ft_v2/$tag --gpu $gpu --epochs 3 --policy both --gpu-mem-util $evmem --port $((base+1)) $mflag \
    > $OUT/logs/eval_${m}_${mode}.log 2>&1
  touch $Q/${m}_${mode}.done; rm -f $Q/gpu${gpu}.busy
  echo "### $(date +%H:%M) BIGRE-DONE $m/$mode (GPU$gpu 해방) ###"
}

echo "===== 대형모델 real/ideal FT 재실행 디스패처 시작 $(date +%H:%M) ====="
while :; do
  pending=0
  for pair in "${PAIRS[@]}"; do
    m=${pair%%:*}; mode=${pair##*:}; key=${m}_${mode}
    is_done $m $mode && { touch $Q/$key.done 2>/dev/null; continue; }
    [ -d "$Q/$key.lock" ] && { pending=1; continue; }
    pending=1
    for g in "${ALLOWED[@]}"; do
      [ -f "$Q/gpu${g}.busy" ] && continue
      fg=$(freegb $g); ut=$(util $g)
      if [ "${fg:-0}" -ge "${NEED[$m]}" ] && [ "${ut:-100}" -lt 35 ]; then
        mkdir "$Q/$key.lock" 2>/dev/null || continue
        touch "$Q/gpu${g}.busy"
        port=$((12500 + g*4))
        ( run_pair $g $m $mode $port ) &
        echo "  claim $key -> GPU$g (free ${fg}GB, util ${ut}%)"
        sleep 20
        break
      fi
    done
  done
  [ "$pending" -eq 0 ] && { echo "===== 대형 4런 완료 $(date +%H:%M) ====="; curl -s -d "DDI-334 대형 real/ideal FT 4런 완료" ntfy.sh/Latex_project >/dev/null; break; }
  sleep 90
done
