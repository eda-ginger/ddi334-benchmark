#!/bin/bash
# Qwen2.5-14B real 남은 공백 채우기: S1 ep1,ep2 + S2 ep1 (S0 ep1,ep2는 fill_qwen_s0.sh가 담당).
# fill_qwen_s0.sh 완료 마커를 기다린 뒤 GPU5에서 이어서 실행 (동시 점유 방지).
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
LOGDIR=results/run_logs
until grep -q "QWEN S0 EP-FILL ALL DONE" "$LOGDIR/fill_qwen_s0_master.log" 2>/dev/null; do
  sleep 30
done
echo "[wait] fill_qwen_s0.sh 완료 확인, 이어서 시작 $(date +%H:%M:%S)"

micromamba run -n vllm-llm python code/oneoff/test_epochs_gen.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --tags real=Qwen_Qwen2.5-14B-Instruct_real_seed42 \
  --gpu 5 --port 13303 --mem 0.55 \
  --cells real:S1:1,real:S1:2,real:S2:1 \
  > "$LOGDIR/fill_qwen14b_real_rest.log" 2>&1

echo "########## QWEN2.5-14B REAL 전체(S0/S1/S2 x ep1/ep2/ep3) EP-FILL 완료 $(date +%H:%M:%S) ##########"
curl -s -d "DDI-334 Qwen2.5-14B real 전체 epoch-fill 완료 (S0/S1/S2 x ep1/ep2/ep3)" ntfy.sh/Latex_project >/dev/null
