#!/bin/bash
# DDI-334 V2 FT 동적 디스패처: 12 pair 큐를 빈 GPU에 자동 배정(여러 개 동시).
# - done 판정 = eval json(v2ft_{tag}_S2.json) 존재 -> 재실행 안 함(중복 방지, idempotent)
# - claim = atomic mkdir 락. GPU busy 마커로 1 GPU=1 pair.
# - 빈 GPU = (free GB >= 모델 need) AND (util < 35) AND (우리 busy 마커 없음)
# - resume: ft_train_v2 --resume (이전 epoch 어댑터 있으면 이어서)
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/ft; Q=$OUT/_q; mkdir -p $Q $OUT/logs

declare -A HF=( [gemma2b]=google/gemma-2-2b-it [qwen3b]=Qwen/Qwen2.5-3B-Instruct [phi35]=microsoft/Phi-3.5-mini-instruct [phi4]=microsoft/phi-4 [qwen14b]=Qwen/Qwen2.5-14B-Instruct [gemma27b]=google/gemma-2-27b-it )
declare -A MERGE=( [gemma2b]=1 [qwen3b]=0 [phi35]=0 [phi4]=0 [qwen14b]=0 [gemma27b]=1 )
declare -A BATCH=( [gemma2b]=8 [qwen3b]=8 [phi35]=8 [phi4]=4 [qwen14b]=4 [gemma27b]=2 )
declare -A EVMEM=( [gemma2b]=0.4 [qwen3b]=0.4 [phi35]=0.4 [phi4]=0.6 [qwen14b]=0.6 [gemma27b]=0.85 )
declare -A NEED=( [gemma2b]=28 [qwen3b]=28 [phi35]=28 [phi4]=38 [qwen14b]=38 [gemma27b]=60 )  # GB free 필요(eval bf16 기준)
PAIRS=(gemma2b:real qwen3b:real phi35:real phi4:real qwen14b:real \
       gemma2b:ideal qwen3b:ideal phi35:ideal phi4:ideal qwen14b:ideal \
       gemma2b:ideal_genes qwen3b:ideal_genes phi35:ideal_genes phi4:ideal_genes qwen14b:ideal_genes \
       gemma27b:real gemma27b:ideal gemma27b:ideal_genes)   # 27B 맨 뒤(가장 느림)
ALLOWED=(1 4 5 0)   # 1/4/5 우리 전용, 0은 비면 기회적, 2/3은 djk0706 전용(제외)

tag_of(){ echo "${HF[$1]//\//_}_${2}_seed42"; }
DONE_SUFFIX=${DONE_SUFFIX:-ep3DONE}   # 현재 ep3 공정. best 공정 시 bestDONE로
is_done(){ local t=$(tag_of $1 $2); [ -f "$OUT/llm_metrics/v2ft_${t}_${DONE_SUFFIX}" ]; }
freegb(){ echo $(( (81920-$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)) / 1024 )); }
util(){ nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $1 2>/dev/null; }

run_pair(){  # $1=gpu $2=model $3=mode $4=port
  local gpu=$1 m=$2 mode=$3 base=$4 hf=${HF[$2]} merge=${MERGE[$2]} batch=${BATCH[$2]} evmem=${EVMEM[$2]}
  local mflag=""; [ "$merge" = "1" ] && mflag="--merge-system"; local tag=$(tag_of $m $mode)
  echo "### $(date +%H:%M) DISPATCH $m/$mode -> GPU$gpu ###"
  micromamba run -n vllm-llm python code/ft_train_v2.py --model $hf --mode $mode --gpu $gpu \
    --epochs 3 --batch_size $batch --resume > $OUT/logs/train_${m}_${mode}.log 2>&1
  micromamba run -n vllm-llm python code/run_ft_cv_v2.py --tag $tag --model $hf --mode $mode \
    --ckpt results/ft_v2/$tag --gpu $gpu --epochs 3 --policy ep3 --gpu-mem-util $evmem --port $((base+1)) $mflag \
    > $OUT/logs/eval_${m}_${mode}.log 2>&1
  touch $Q/${m}_${mode}.done; rm -f $Q/gpu${gpu}.busy
  echo "### $(date +%H:%M) DONE $m/$mode (GPU$gpu 해방) ###"
}

echo "===== FT 디스패처 시작 $(date +%H:%M) ====="
while :; do
  pending=0
  for pair in "${PAIRS[@]}"; do
    m=${pair%%:*}; mode=${pair##*:}; key=${m}_${mode}
    is_done $m $mode && { touch $Q/$key.done 2>/dev/null; continue; }
    [ -d "$Q/$key.lock" ] && { pending=1; continue; }   # 이미 claim됨(실행중)
    pending=1
    # 빈 GPU 탐색
    for g in "${ALLOWED[@]}"; do
      [ -f "$Q/gpu${g}.busy" ] && continue
      fg=$(freegb $g); ut=$(util $g)
      if [ "${fg:-0}" -ge "${NEED[$m]}" ] && [ "${ut:-100}" -lt 35 ]; then
        mkdir "$Q/$key.lock" 2>/dev/null || continue   # atomic claim
        touch "$Q/gpu${g}.busy"
        port=$((12500 + g*4))
        ( run_pair $g $m $mode $port ) &
        echo "  claim $key -> GPU$g (free ${fg}GB, util ${ut}%)"
        sleep 20  # 서버 기동 안정화 후 다음 배정
        break
      fi
    done
  done
  [ "$pending" -eq 0 ] && { echo "===== 전체 12 pair 완료 $(date +%H:%M) ====="; curl -s -d "DDI-334 FT 12런 전부 완료" ntfy.sh/Latex_project >/dev/null; break; }
  sleep 90
done
