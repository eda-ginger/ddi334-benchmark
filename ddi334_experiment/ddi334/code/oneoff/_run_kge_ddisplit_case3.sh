#!/bin/bash
# Real(case3) DDI-split KGE 학습 (transe/rotate/complex) — 신약 고립 + DDI 평가 포함.
set -u
BASE=/home/rudwls2717/Latex/experiments/v5_ddi-bench
cd $BASE/experiment/ddi334
RUN="micromamba run -n DDIBench python3"
G=${1:-2}
LOG=results/run_logs
for m in transe rotate complex; do
  cfg=data/kge/ddibn/configs_ddisplit_case3/$m.yaml
  out=data/kge/ddibn_ddisplit_case3/$m
  echo "### $(date '+%m-%d %H:%M') KGE-ddisplit-case3 $m -> GPU$G ###"
  CUDA_VISIBLE_DEVICES=$G $RUN $BASE/kge_hetionet/src/baseline.py -c $cfg >> $LOG/kge_ddisplitc3_$m.log 2>&1
  CUDA_VISIBLE_DEVICES=$G $RUN $BASE/kge_hetionet/src/extract_embeddings.py --result_dir $out >> $LOG/kge_ddisplitc3_$m.log 2>&1
  echo "### $(date '+%m-%d %H:%M') $m 완료 ###"
done
curl -s -d "KGE DDI-split case3(Real) 3종 완료" ntfy.sh/Latex_project >/dev/null 2>&1
