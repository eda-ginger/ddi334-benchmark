#!/bin/bash
# DDI-334 FT 동적 디스패처: pair 큐를 빈 GPU에 자동 배정(여러 개 동시).
# 3개 운영 모드(ep3/best/both)로 따로 있던 run_ft_dispatch.sh/run_bestep_dispatch.sh/
# run_ft_bigreal.sh를 --policy/--models로 통합한 것 — 모델 조합×policy만 다를 뿐
# 디스패치 로직(atomic lock, GPU busy 마커, skip-if-done)은 셋 다 동일했음.
#
# --policy ep3   (기본) : ft_train.py(epochs=3) -> run_ft_cv.py --policy ep3. 전체 6모델 기본.
# --policy best         : 재학습 없이 run_ft_cv.py --policy best 재평가만. ep3DONE +
#                         저장된 epoch_1/2/3 어댑터가 있어야 함(없으면 대기). 기본은 소형 4모델.
# --policy both         : ft_train.py(epochs=3) -> run_ft_cv.py --policy both (ep3+best 둘 다 저장).
#                         기본은 대형 2모델(느려서 따로 재실행하던 용도).
# --models a,b,c        : 대상 모델(HF 아래 키) 콤마 목록. 생략 시 policy별 기본값.
#
# - done 판정 = ep3는 ft_{tag}_ep3DONE, best/both는 ft_{tag}_bestDONE 존재 여부(중복 방지, idempotent)
# - claim = atomic mkdir 락. ep3/both는 같은 락 네임스페이스(동시 실행 방지용 공유),
#   best는 별도 네임스페이스(ep3 완료를 기다리며 동시에 돌 수 있어야 해서).
# - 빈 GPU = (free GB >= 모델 need) AND (util < 35) AND (우리 busy 마커 없음)
# - resume: ft_train.py --resume (이전 epoch 어댑터 있으면 이어서)
set -u
cd /home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/llm/finetuning; Q=$OUT/_q; mkdir -p $Q $OUT/logs

POLICY=ep3
MODELS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --policy) POLICY=$2; shift 2 ;;
    --models) MODELS=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

declare -A HF=( [gemma2b]=google/gemma-2-2b-it [qwen3b]=Qwen/Qwen2.5-3B-Instruct [phi35]=microsoft/Phi-3.5-mini-instruct [phi4]=microsoft/phi-4 [qwen14b]=Qwen/Qwen2.5-14B-Instruct [gemma27b]=google/gemma-2-27b-it )
declare -A MERGE=( [gemma2b]=1 [qwen3b]=0 [phi35]=0 [phi4]=0 [qwen14b]=0 [gemma27b]=1 )
declare -A BATCH=( [gemma2b]=8 [qwen3b]=8 [phi35]=8 [phi4]=4 [qwen14b]=4 [gemma27b]=2 )
declare -A EVMEM=( [gemma2b]=0.4 [qwen3b]=0.4 [phi35]=0.4 [phi4]=0.6 [qwen14b]=0.6 [gemma27b]=0.85 )
declare -A NEED=( [gemma2b]=28 [qwen3b]=28 [phi35]=28 [phi4]=38 [qwen14b]=38 [gemma27b]=60 )  # GB free 필요(eval bf16 기준)
ALLOWED=(1 4 5 0)   # 1/4/5 우리 전용, 0은 비면 기회적, 2/3은 djk0706 전용(제외)

case "$POLICY" in
  ep3)  [ -z "$MODELS" ] && MODELS="gemma2b,qwen3b,phi35,phi4,qwen14b,gemma27b" ;;
  best) [ -z "$MODELS" ] && MODELS="gemma2b,qwen3b,phi35,phi4" ;;
  both) [ -z "$MODELS" ] && MODELS="qwen14b,gemma27b" ;;
  *) echo "unknown --policy: $POLICY (ep3|best|both)" >&2; exit 1 ;;
esac
LOCK_NS="lock"; [ "$POLICY" = "best" ] && LOCK_NS="best.lock"
PORT_OFF=1; [ "$POLICY" = "best" ] && PORT_OFF=2

PAIRS=()
IFS=',' read -ra MLIST <<< "$MODELS"
for m in "${MLIST[@]}"; do PAIRS+=("$m:real" "$m:ideal"); done

tag_of(){ echo "${HF[$1]//\//_}_${2}_seed42"; }
is_done(){
  local t=$(tag_of $1 $2)
  if [ "$POLICY" = "ep3" ]; then [ -f "$OUT/llm_metrics/ft_${t}_ep3DONE" ]
  else [ -f "$OUT/llm_metrics/ft_${t}_bestDONE" ]; fi
}
ep3_ready(){  # best만 사용: 재학습 없이 재평가하므로 ep3 어댑터가 이미 있어야 함
  local t=$(tag_of $1 $2)
  [ -f "$OUT/llm_metrics/ft_${t}_ep3DONE" ] && [ -d "results/ft_checkpoints/$t/epoch_3" ]
}
freegb(){ echo $(( (81920-$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)) / 1024 )); }
util(){ nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $1 2>/dev/null; }

run_pair(){  # $1=gpu $2=model $3=mode $4=port_base
  local gpu=$1 m=$2 mode=$3 base=$4 hf=${HF[$2]} merge=${MERGE[$2]} batch=${BATCH[$2]} evmem=${EVMEM[$2]}
  local mflag=""; [ "$merge" = "1" ] && mflag="--merge-system"; local tag=$(tag_of $m $mode)
  echo "### $(date +%H:%M) DISPATCH($POLICY) $m/$mode -> GPU$gpu ###"
  if [ "$POLICY" != "best" ]; then
    micromamba run -n vllm-llm python code/ft_train.py --model $hf --mode $mode --gpu $gpu \
      --epochs 3 --batch_size $batch --resume > $OUT/logs/train_${m}_${mode}.log 2>&1
  fi
  micromamba run -n vllm-llm python code/run_ft_cv.py --tag $tag --model $hf --mode $mode \
    --ckpt results/ft_checkpoints/$tag --gpu $gpu --epochs 3 --policy $POLICY --gpu-mem-util $evmem \
    --port $((base+PORT_OFF)) $mflag > $OUT/logs/eval_${m}_${mode}_${POLICY}.log 2>&1
  touch $Q/${m}_${mode}.${POLICY}.done; rm -f $Q/gpu${gpu}.busy
  echo "### $(date +%H:%M) DONE($POLICY) $m/$mode (GPU$gpu 해방) ###"
}

echo "===== FT 디스패처 시작 ($POLICY, ${#PAIRS[@]} pair) $(date +%H:%M) ====="
while :; do
  pending=0
  for pair in "${PAIRS[@]}"; do
    m=${pair%%:*}; mode=${pair##*:}; key=${m}_${mode}
    is_done $m $mode && continue
    if [ "$POLICY" = "best" ]; then
      ep3_ready $m $mode || { pending=1; continue; }   # ep3 아직 -> 대기
    fi
    [ -d "$Q/$key.$LOCK_NS" ] && { pending=1; continue; }   # 이미 claim됨(실행중)
    pending=1
    for g in "${ALLOWED[@]}"; do
      [ -f "$Q/gpu${g}.busy" ] && continue
      fg=$(freegb $g); ut=$(util $g)
      if [ "${fg:-0}" -ge "${NEED[$m]}" ] && [ "${ut:-100}" -lt 35 ]; then
        mkdir "$Q/$key.$LOCK_NS" 2>/dev/null || continue   # atomic claim
        touch "$Q/gpu${g}.busy"
        port=$((12500 + g*4))
        ( run_pair $g $m $mode $port ) &
        echo "  claim $key -> GPU$g (free ${fg}GB, util ${ut}%)"
        sleep 20  # 서버 기동 안정화 후 다음 배정
        break
      fi
    done
  done
  if [ "$pending" -eq 0 ]; then
    echo "===== 전체 ${#PAIRS[@]} pair($POLICY) 완료 $(date +%H:%M) ====="
    curl -s -d "DDI-334 FT ${#PAIRS[@]}런($POLICY) 전부 완료" ntfy.sh/removed-ntfy-topic >/dev/null
    break
  fi
  sleep 90
done
