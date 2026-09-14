#!/bin/bash
# 완료된 ddisplit KGE들의 순수 DDI-hits 일괄 평가 (재학습 X). complex는 완료되면 재실행 시 합류.
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
RUN="micromamba run -n DDIBench python3"
for scen in "ddisplit:configs_ddisplit" "ddisplit_case3:configs_ddisplit_case3"; do
  IFS=: read dir cfgdir <<< "$scen"
  for m in transe rotate complex; do
    md=data/kge/ddibn_$dir/$m
    cfg=data/kge/ddibn/$cfgdir/$m.yaml
    [ -f "$md/trained_model.pkl" ] || { echo "  skip $dir/$m (pkl 없음)"; continue; }
    [ -f "$md/ddi_hits.json" ] && { echo "  skip $dir/$m (이미 있음)"; continue; }
    echo "### DDI-hits $dir/$m ###"
    CUDA_VISIBLE_DEVICES=5 $RUN code/oneoff/eval_ddi_hits.py --model_dir "$md" --config "$cfg" \
      > results/run_logs/ddihits_${dir}_${m}.log 2>&1
  done
done
echo "### DDI-hits 배치 끝 ###"
