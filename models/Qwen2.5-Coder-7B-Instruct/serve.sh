#!/bin/bash
# Serve Qwen/Qwen2.5-Coder-7B-Instruct with vLLM 0.15.1, optionally with a LoRA adapter.
# API model names: "Qwen/Qwen2.5-Coder-7B-Instruct" (base) and, with an adapter, NAME.
# Usage: bash models/Qwen2.5-Coder-7B-Instruct/serve.sh [adapter_dir [name=lora]]
#   MAX_LORA_RANK (default 16) must be >= the adapter's r.
DIR=$(cd "$(dirname "$0")" && pwd)
VENV=$(bash "$DIR/../../venv.sh" "$DIR/requirements.txt") || exit 1
MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
LORA=()
if [ -n "${1:-}" ]; then
    LORA=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-16}" --lora-modules "${2:-lora}=$(realpath "$1")")
fi
echo "Starting vLLM server for $MODEL_NAME ${1:+with adapter ${2:-lora}=$1}"
exec "$VENV/bin/vllm" serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    "${LORA[@]}"
