#!/bin/bash
# cell 08 downstream x DDI-split 임베딩 (Ideal/Real x transe/rotate/complex x 3seed). 별도 출력.
# 임베딩 없으면(complex 미완) skip -> 나중 재실행 시 합류. 결과 있으면 skip(idempotent).
set -u
cd /home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334
RUN="micromamba run -n DDIBench python3"
LOG=results/run_logs
run_scenario(){  # $1=gpu $2=scenario(ideal/real)
  local gpu=$1 scen=$2 case3=""
  [ "$scen" = "real" ] && case3="CASE3=1"
  local embdir=data/kge/ddibn_ddisplit$([ "$scen" = "real" ] && echo _case3)
  for m in transe rotate complex; do
    ls $embdir/$m/*ent_*.npy >/dev/null 2>&1 || { echo "[$scen/$m] 임베딩 미완 skip"; continue; }
    for s in 0 1 2; do
      local rd=results/kge_ddisplit_compare/$scen/$m
      [ -f "$rd/seed${s}/metrics.json" ] || [ -f "$rd/metrics_seed${s}.json" ] && { echo "[$scen/$m/s$s] 이미 있음 skip"; continue; }
      echo "### $(date +%H:%M) cell08 $scen/$m/seed$s -> GPU$gpu ###"
      env CUDA_VISIBLE_DEVICES=$gpu DDISPLIT=1 $case3 KGE_TYPE=$m \
        $RUN code/train.py --cell 08 --dataset ddibn --seed $s --gpu 0 --result_dir $rd \
        >> $LOG/cell08_ddisplit_${scen}_${m}.log 2>&1
    done
  done
}
run_scenario 3 ideal &
run_scenario 2 real &
wait
echo "### cell08 ddisplit 배치 끝 $(date +%H:%M) ###"
curl -s -d "cell08 DDI-split 배치 완료" ntfy.sh/Latex_project >/dev/null 2>&1
