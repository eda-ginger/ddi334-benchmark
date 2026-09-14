#!/bin/bash
# S2 재현성 검증: 4모델(gemma2b/gemma27b/qwen3b/qwen14b) x real/ideal, S2만 재실행.
# 기존 결과(results/ddibn_ablation/info_avail)는 건드리지 않고 별도 폴더에 저장.
# 목적: FT-real이 S2에서 ZS-real보다 낮게 나온 이상현상이 재현되는지 확인(temperature=0 결정적이라 큰 변화 없을 것으로 예상).
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
OUT=results/ddibn_ablation/info_avail_verify
mkdir -p "$OUT/llm_metrics" "$OUT/raw_preds"
LOGDIR=results/run_logs
mkdir -p "$LOGDIR"

# entry: label|model|merge(0/1)|memutil|gpu
run_one() {
  local mode=$1 label=$2 model=$3 merge=$4 mem=$5 gpu=$6 port=$7
  local mergeflag=""; [ "$merge" = "1" ] && mergeflag="--merge-system"
  echo "[launch] $label mode=$mode gpu=$gpu $(date +%H:%M:%S)"
  micromamba run -n vllm-llm python code/llm_infer.py \
    --version v2 --dataset ddibn --model "$model" \
    --gpu "$gpu" --split S2 --mode "$mode" --port "$port" --gpu-mem-util "$mem" \
    --workers 24 $mergeflag --outdir "$OUT" --label "$label" \
    > "$LOGDIR/verify_${label}_S2.log" 2>&1
  echo "[done  ] $label mode=$mode gpu=$gpu $(date +%H:%M:%S)"
}

# Wave 1: gemma2b(real,ideal) + qwen3b(real) on GPU 0,2,4
run_one real  gemma2b_zs_real  google/gemma-2-2b-it 1 0.30 0 13201 &
run_one ideal gemma2b_zs_ideal google/gemma-2-2b-it 1 0.30 2 13202 &
run_one real  qwen3b_zs_real   Qwen/Qwen2.5-3B-Instruct 0 0.30 4 13203 &
wait
echo "########## WAVE1 DONE $(date +%H:%M:%S) ##########"

# Wave 2: qwen3b(ideal) + qwen14b(real) + qwen14b(ideal) on GPU 0,2,4
run_one ideal qwen3b_zs_ideal  Qwen/Qwen2.5-3B-Instruct 0 0.30 0 13204 &
run_one real  qwen14b_zs_real  Qwen/Qwen2.5-14B-Instruct 0 0.45 2 13205 &
run_one ideal qwen14b_zs_ideal Qwen/Qwen2.5-14B-Instruct 0 0.45 4 13206 &
wait
echo "########## WAVE2 DONE $(date +%H:%M:%S) ##########"

# Wave 3: gemma27b(real, ideal) on GPU 2,4 (large model, needs more mem headroom)
run_one real  gemma27b_zs_real  google/gemma-2-27b-it 1 0.85 2 13207 &
run_one ideal gemma27b_zs_ideal google/gemma-2-27b-it 1 0.85 4 13208 &
wait
echo "########## WAVE3 DONE $(date +%H:%M:%S) ##########"

echo "########## ALL 8 JOBS DONE $(date +%H:%M:%S) ##########"
n=$(find "$OUT/llm_metrics" -name "*_zs_*_S2.json" 2>/dev/null | wc -l)
curl -s -d "DDI-334 S2 ZS 재현성 검증 완료 ($n/8): results/ddibn_ablation/info_avail_verify" ntfy.sh/Latex_project >/dev/null
