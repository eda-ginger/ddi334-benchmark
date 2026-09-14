#!/bin/bash
# DDI-334 KGE 학습 (5 model × 2 dataset). Usage: bash run_all.sh [GPU]
set -e
GPU=${1:-0}
BASE="/home/rudwls2717/Latex/experiments/v5_ddi-bench/kge_hetionet"
KGE="/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge"

echo "=== ddibn/transe ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/configs/transe.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/transe"

echo "=== ddibn/rotate ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/configs/rotate.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/rotate"

echo "=== ddibn/distmult ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/configs/distmult.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/distmult"

echo "=== ddibn/complex ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/configs/complex.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/complex"

echo "=== ddibn/transh ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/configs/transh.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/ddibn/transh"

echo "=== tdc/transe ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/tdc/configs/transe.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/tdc/transe"

echo "=== tdc/rotate ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/tdc/configs/rotate.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/tdc/rotate"

echo "=== tdc/distmult ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/tdc/configs/distmult.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/tdc/distmult"

echo "=== tdc/complex ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/tdc/configs/complex.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/tdc/complex"

echo "=== tdc/transh ==="
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/baseline.py" -c "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/tdc/configs/transh.yaml"
CUDA_VISIBLE_DEVICES=$GPU micromamba run -n DDIBench python3 "$BASE/src/extract_embeddings.py" --result_dir "/home/rudwls2717/Latex/experiments/v5_ddi-bench/experiment/ddi334/kge/tdc/transh"
