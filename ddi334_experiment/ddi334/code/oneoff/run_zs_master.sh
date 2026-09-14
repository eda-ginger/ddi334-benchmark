#!/bin/bash
# 남은 4종 ZS V2 ablation 마스터: 모델 순차, 모델당 3모드(real/genes/ideal) GPU 3/4/5 병렬.
# 각 모델 9 json 완료까지 대기 후 vLLM 잔존 정리 -> 다음 모델. 끝에 ntfy.
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/info_avail
L=code/oneoff/run_zs_one_mode.sh
# 모델: prefix | model | merge | memutil   (작은 것 먼저; GPU2 djk0706 공유라 mem 보수적)
# Gemma2-27B는 GPU2(여유40GB)에 안 들어가서 이 마스터 제외 -> 별도 처리.
MODELS=(
  "gemma2b|google/gemma-2-2b-it|1|0.3"
  "qwen3b|Qwen/Qwen2.5-3B-Instruct|0|0.3"
  "qwen14b|Qwen/Qwen2.5-14B-Instruct|0|0.45"
)
for entry in "${MODELS[@]}"; do
  IFS='|' read -r pre model merge mem <<< "$entry"
  echo "########## MODEL $pre ($model) merge=$merge mem=$mem $(date +%H:%M) ##########"
  bash $L 0 real        ${pre}_zs_real  12373 "$model" "$merge" "$mem" > $OUT/run_${pre}_real.log  2>&1 &
  bash $L 1 ideal_genes ${pre}_zs_genes 12374 "$model" "$merge" "$mem" > $OUT/run_${pre}_genes.log 2>&1 &
  bash $L 2 ideal       ${pre}_zs_ideal 12375 "$model" "$merge" "$mem" > $OUT/run_${pre}_ideal.log 2>&1 &
  # 이 모델 9 json 완료까지 대기 (최대 ~8h)
  for i in $(seq 1 2880); do
    n=$(find $OUT/llm_metrics -name "${pre}_zs_*.json" 2>/dev/null | wc -l)
    [ "$n" -ge 9 ] && break
    sleep 10
  done
  echo "########## $pre 완료 ($(find $OUT/llm_metrics -name "${pre}_zs_*.json"|wc -l)/9) $(date +%H:%M) ##########"
  # 이 모델 vLLM 서버/런처 잔존 정리 (다음 모델 GPU 확보)
  for p in $(ps -eo pid,args | awk -v m="$model" '($0 ~ "vllm.entrypoints" && index($0,m)>0) || ($0 ~ "llm_infer.py" && index($0,m)>0){print $1}'); do kill -9 $p 2>/dev/null; done
  sleep 8
  # GPU 0/1/2 zombie 정리 (우리 vllm-llm 만)
  for g in 0 1 2; do
    u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i=$g '$1==i{print $2}')
    for zp in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$u" '$1==u{print $2}'); do
      # 내 vllm-llm 프로세스만 (다른 유저 보호)
      cmd=$(ps -o args= -p $zp 2>/dev/null)
      echo "$cmd" | grep -q "vllm-llm/bin/python" && { echo "  zombie kill $zp (gpu$g)"; kill -9 $zp 2>/dev/null; }
    done
  done
  sleep 5
done
echo "########## 3 MODELS DONE $(date +%H:%M) ##########"
curl -s -d "DDI-334 ZS: gemma2b/qwen3b/qwen14b 완료 (GPU0/1/2). Gemma27B는 별도. genes-vs-ideal 패턴 확인 가능" ntfy.sh/Latex_project >/dev/null
