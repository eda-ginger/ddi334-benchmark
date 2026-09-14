#!/bin/bash
# DDI-split KGE 진단 학습 (transe/rotate/complex) — DDI를 평가에도 넣어 DDI-link hits 측정.
set -u
BASE=/home/rudwls2717/Latex/experiments/v5_ddi-bench
cd $BASE/experiment/ddi334
RUN="micromamba run -n DDIBench python3"
G=${1:-3}
LOG=results/run_logs
for m in transe rotate complex; do
  cfg=data/kge/ddibn/configs_ddisplit/$m.yaml
  out=data/kge/ddibn_ddisplit/$m
  echo "### $(date '+%m-%d %H:%M') KGE-ddisplit $m -> GPU$G ###"
  CUDA_VISIBLE_DEVICES=$G $RUN $BASE/kge_hetionet/src/baseline.py -c $cfg >> $LOG/kge_ddisplit_$m.log 2>&1
  CUDA_VISIBLE_DEVICES=$G $RUN $BASE/kge_hetionet/src/extract_embeddings.py --result_dir $out >> $LOG/kge_ddisplit_$m.log 2>&1
  echo "### $(date '+%m-%d %H:%M') $m 완료 ###"
done
curl -s -d "KGE DDI-split 3종(transe/rotate/complex) 완료" ntfy.sh/Latex_project >/dev/null 2>&1
