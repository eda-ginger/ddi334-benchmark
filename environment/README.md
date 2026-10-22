# Environments

Three separate Python environments were used (version conflicts between
PyG/DGL/vLLM stacks made a single environment impractical). Each
`*_requirements.txt` is an exact `pip freeze` snapshot of what was
actually used to produce the results in this repository.

| Environment | Python | Used for |
|---|---|---|
| `ddibench_requirements.txt` | 3.8 | `ddi334/code/train/train.py` (encoder cells 01-12) and everything else in `ddi334/code/` except LLM inference/fine-tuning and KGE training |
| `dglke_requirements.txt` | 3.8 | KGE embedding training (TransE/RotatE/ComplEx/DistMult/TransH) that produces `ddi334/data/relation/kge/*/{transe,rotate,complex,distmult,transh}/` |
| `vllm_llm_requirements.txt` | 3.10 | `ddi334/code/llm/zeroshot.py`, `finetuning_train.py`, `finetuning_eval.py` (all LLM zero-shot / fine-tuning, cells 13-18) |

## Setup

```bash
conda create -n DDIBench python=3.8 -y
conda activate DDIBench
pip install -r environment/ddibench_requirements.txt

conda create -n dglke python=3.8 -y
conda activate dglke
pip install -r environment/dglke_requirements.txt

conda create -n vllm-llm python=3.10 -y
conda activate vllm-llm
pip install -r environment/vllm_llm_requirements.txt
```

Some packages (`torch`, `dgl`, `torch-geometric` extensions) are
CUDA-build-specific (`+cu117`, `+cu124`, ...). If your CUDA version
differs, install the base packages from the official PyTorch/DGL/PyG
index for your CUDA version first, then `pip install` the rest of the
requirements file with `--no-deps` for the already-installed packages,
or adjust the pinned versions accordingly.

GPU used for all reported experiments: [TODO: specify GPU model, e.g. A100 80GB].
