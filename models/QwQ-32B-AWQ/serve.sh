#!/bin/bash
# Serve Qwen/QwQ-32B-AWQ with vLLM (AWQ-Marlin kernels, ~78 tok/s, see README.md).
# Usage: bash models/QwQ-32B-AWQ/serve.sh [model_name]
DIR=$(cd "$(dirname "$0")" && pwd)
VENV=$(bash "$DIR/../../venv.sh" "$DIR/requirements.txt") || exit 1
MODEL_NAME=${1:-"Qwen/QwQ-32B-AWQ"}
echo "Using model: $MODEL_NAME"
# The server runs in the foreground; stop it with Ctrl+C.
exec "$VENV/bin/vllm" serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --quantization awq_marlin \
    --max-model-len 32768
