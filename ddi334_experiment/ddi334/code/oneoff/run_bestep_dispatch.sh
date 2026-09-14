#!/bin/bash
# DDI-334 V2 best-epoch 재평가 디스패처 (재학습 X).
# - 대상: 작은 4모델 x {real, ideal, ideal_genes} = 12 pair
# - 전제: 각 pair의 ep3DONE + 저장된 epoch_1/2/3 어댑터 존재 (없으면 skip=대기)
# - 동작: run_ft_cv_v2 --policy best (val 스윕으로 split별 best epoch 선택 -> test)
# - done 판정 = v2ft_{tag}_bestDONE 마커. claim = atomic mkdir 락 ($key.best.lock).
# - gpu busy 마커(gpu{N}.busy)는 ep3 디스패처와 공유 -> GPU 이중점유 방지.
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/ft; Q=$OUT/_q; mkdir -p $Q $OUT/logs

declare -A HF=( [gemma2b]=google/gemma-2-2b-it [qwen3b]=Qwen/Qwen2.5-3B-Instruct [phi35]=microsoft/Phi-3.5-mini-instruct [phi4]=microsoft/phi-4 )
declare -A MERGE=( [gemma2b]=1 [qwen3b]=0 [phi35]=0 [phi4]=0 )
declare -A EVMEM=( [gemma2b]=0.4 [qwen3b]=0.4 [phi35]=0.4 [phi4]=0.6 )
declare -A NEED=( [gemma2b]=28 [qwen3b]=28 [phi35]=28 [phi4]=38 )
PAIRS=(gemma2b:real qwen3b:real phi35:real phi4:real \
       gemma2b:ideal qwen3b:ideal phi35:ideal phi4:ideal \
       gemma2b:ideal_genes qwen3b:ideal_genes phi35:ideal_genes phi4:ideal_genes)
ALLOWED=(1 4 5 0)

tag_of(){ echo "${HF[$1]//\//_}_${2}_seed42"; }
best_done(){ local t=$(tag_of $1 $2); [ -f "$OUT/llm_metrics/v2ft_${t}_bestDONE" ]; }
ep3_ready(){ local t=$(tag_of $1 $2); [ -f "$OUT/llm_metrics/v2ft_${t}_ep3DONE" ] && [ -d "$OUT/../../ft_v2/$t/epoch_3" ] 2>/dev/null; }
freegb(){ echo $(( (81920-$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)) / 1024 )); }
util(){ nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $1 2>/dev/null; }

run_pair(){  # $1=gpu $2=model $3=mode $4=port
  local gpu=$1 m=$2 mode=$3 base=$4 hf=${HF[$2]} merge=${MERGE[$2]} evmem=${EVMEM[$2]}
  local mflag=""; [ "$merge" = "1" ] && mflag="--merge-system"; local tag=$(tag_of $m $mode)
  echo "### $(date +%H:%M) BEST-DISPATCH $m/$mode -> GPU$gpu ###"
  micromamba run -n vllm-llm python code/run_ft_cv_v2.py --tag $tag --model $hf --mode $mode \
    --ckpt results/ft_v2/$tag --gpu $gpu --epochs 3 --policy best --gpu-mem-util $evmem --port $((base+2)) $mflag \
    > $OUT/logs/best_${m}_${mode}.log 2>&1
  touch $Q/${m}_${mode}.best.done; rm -f $Q/gpu${gpu}.busy
  echo "### $(date +%H:%M) BEST-DONE $m/$mode (GPU$gpu 해방) ###"
}

echo "===== best-epoch 디스패처 시작 $(date +%H:%M) ====="
while :; do
  pending=0
  for pair in "${PAIRS[@]}"; do
    m=${pair%%:*}; mode=${pair##*:}; key=${m}_${mode}
    best_done $m $mode && continue
    ep3_ready $m $mode || { pending=1; continue; }        # ep3 아직 -> 대기
    [ -d "$Q/$key.best.lock" ] && { pending=1; continue; } # 이미 claim됨
    pending=1
    for g in "${ALLOWED[@]}"; do
      [ -f "$Q/gpu${g}.busy" ] && continue
      fg=$(freegb $g); ut=$(util $g)
      if [ "${fg:-0}" -ge "${NEED[$m]}" ] && [ "${ut:-100}" -lt 35 ]; then
        mkdir "$Q/$key.best.lock" 2>/dev/null || continue
        touch "$Q/gpu${g}.busy"
        port=$((12500 + g*4))
        ( run_pair $g $m $mode $port ) &
        echo "  claim(best) $key -> GPU$g (free ${fg}GB, util ${ut}%)"
        sleep 20
        break
      fi
    done
  done
  [ "$pending" -eq 0 ] && { echo "===== best-epoch 12 pair 완료 $(date +%H:%M) ====="; curl -s -d "DDI-334 best-epoch 12런 완료" ntfy.sh/Latex_project >/dev/null; break; }
  sleep 90
done
