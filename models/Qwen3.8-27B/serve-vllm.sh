#!/bin/bash
# Serve Qwen/Qwen3.8-27B with the vLLM 0.24.0 built from source (build-vllm.sh), optionally
# with a LoRA adapter. An adapter trained on the text-only Qwen3_5ForCausalLM must first be
# renamed with lora_for_vllm.py, or vLLM loads it and silently applies nothing.
# Usage: bash models/Qwen3.8-27B/serve-vllm.sh [adapter_dir [name=lora]]
# Keep concurrency low: several concurrent n>1 requests hang the engine (see README.md).
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$DIR/../..
VENV=$(bash "$ROOT/venv.sh" "$DIR/requirements-vllm.txt") || exit 1
[ -x "$VENV/bin/vllm" ] || { echo "no vllm in $VENV: run build-vllm.sh first" >&2; exit 1; }
# FlashInfer JIT-compiles kernels at startup: needs nvcc (else /usr/local/cuda/bin/nvcc, absent)
source "$ROOT/tools/cuda-env.sh" "$VENV"
# the pod has 32 CPUs / 128 GiB but nproc says 224: an unbounded JIT (~70 nvcc) got it OOM-restarted
export MAX_JOBS=${MAX_JOBS:-8}
LORA=()
if [ -n "${1:-}" ]; then
    LORA=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-16}" --lora-modules "${2:-lora}=$(realpath "$1")")
fi
exec "$VENV/bin/vllm" serve Qwen/Qwen3.8-27B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    "${LORA[@]}"
