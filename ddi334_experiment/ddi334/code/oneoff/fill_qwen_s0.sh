#!/bin/bash
# Qwen S0 ep-fill: 3B ideal(ep1,ep2) + 14B real(ep1,ep2) — 둘 다 지금 ep3만 있음.
# GPU0/2/4는 S2 재현성 검증 작업 사용 중 -> GPU5(여유 66GB) 순차 사용.
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
LOGDIR=results/run_logs; mkdir -p "$LOGDIR"

echo "[1/2] Qwen2.5-3B ideal S0 ep1,ep2 $(date +%H:%M:%S)"
micromamba run -n vllm-llm python code/oneoff/test_epochs_gen.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tags ideal=Qwen_Qwen2.5-3B-Instruct_ideal_seed42 \
  --gpu 5 --port 13301 --mem 0.35 \
  --cells ideal:S0:1,ideal:S0:2 \
  > "$LOGDIR/fill_qwen3b_ideal_S0.log" 2>&1
echo "[1/2 done] $(date +%H:%M:%S)"

echo "[2/2] Qwen2.5-14B real S0 ep1,ep2 $(date +%H:%M:%S)"
micromamba run -n vllm-llm python code/oneoff/test_epochs_gen.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --tags real=Qwen_Qwen2.5-14B-Instruct_real_seed42 \
  --gpu 5 --port 13302 --mem 0.55 \
  --cells real:S0:1,real:S0:2 \
  > "$LOGDIR/fill_qwen14b_real_S0.log" 2>&1
echo "[2/2 done] $(date +%H:%M:%S)"

echo "########## QWEN S0 EP-FILL ALL DONE $(date +%H:%M:%S) ##########"
curl -s -d "DDI-334 Qwen S0 ep-fill 완료: 3B ideal + 14B real (ep1,ep2)" ntfy.sh/Latex_project >/dev/null
