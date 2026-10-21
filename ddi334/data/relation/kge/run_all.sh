#!/bin/bash
# DDI-334 KGE 학습 (5 model × ddibn). Usage: bash run_all.sh [GPU]
set -e
GPU=${1:-0}
BASE="/home/internal-server-user/Latex/experiments/v5_ddi-bench/kge_hetionet"
KGE="/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge"

echo "=== ddibn/transe ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/configs/transe.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/transe"

echo "=== ddibn/rotate ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/configs/rotate.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/rotate"

echo "=== ddibn/distmult ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/configs/distmult.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/distmult"

echo "=== ddibn/complex ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/configs/complex.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/complex"

echo "=== ddibn/transh ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/configs/transh.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/internal-server-user/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/transh"
