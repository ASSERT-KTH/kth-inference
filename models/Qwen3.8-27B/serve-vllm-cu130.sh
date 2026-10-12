#!/bin/bash
# Serve Qwen/Qwen3.8-27B with the prebuilt vLLM 0.31.0 (CUDA 13) through CUDA forward
# compatibility on the 12.4 driver (tools/cuda13-env.sh). No build: the venv is created on
# first use. The LoRA option is the same as serve-vllm.sh (rename adapters with lora_for_vllm.py).
# Usage: bash models/Qwen3.8-27B/serve-vllm-cu130.sh [adapter_dir [name=lora]]
# Usage with extra vLLM flags: EXTRA_ARGS="--port 8001" bash .../serve-vllm-cu130.sh
# Another qwen3_5 checkpoint: MODEL=Qwen/Qwen3.6-27B-FP8 bash .../serve-vllm-cu130.sh
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$DIR/../..
VENV=$(bash "$ROOT/venv.sh" "$DIR/requirements-vllm-cu130.txt") || exit 1
# libcuda 580 (compat) + nvcc 13 + cuRAND headers: FlashInfer JIT-compiles kernels at startup
source "$ROOT/tools/cuda13-env.sh" "$VENV"
# the pod has 32 CPUs / 128 GiB but nproc says 224: an unbounded JIT (~70 nvcc) got it OOM-restarted
export MAX_JOBS=${MAX_JOBS:-8}
# the model is cached; a pod network outage must not fail engine start-up on a Hub file list
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
LORA=()
if [ -n "${1:-}" ]; then
    LORA=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-16}" --lora-modules "${2:-lora}=$(realpath "$1")")
fi
exec "$VENV/bin/vllm" serve "${MODEL:-Qwen/Qwen3.8-27B}" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    "${LORA[@]}" \
    ${EXTRA_ARGS:-}
