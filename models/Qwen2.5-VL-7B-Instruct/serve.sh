#!/bin/bash
# Serve Qwen/Qwen2.5-VL-7B-Instruct with vLLM. Optimized for H100 (80GB VRAM):
# - kv-cache-dtype fp8: reduces VRAM usage for KV cache
# - max-model-len 32768: supports long context
# - gpu-memory-utilization 0.80: 7B fits easily on H100
# - trust-remote-code: required for Qwen-VL
# Usage: bash models/Qwen2.5-VL-7B-Instruct/serve.sh
DIR=$(cd "$(dirname "$0")" && pwd)
VENV=$(bash "$DIR/../../venv.sh" "$DIR/requirements.txt") || exit 1
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
echo "Starting vLLM server for $MODEL_NAME..."
exec "$VENV/bin/vllm" serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
