#!/bin/bash
# Serve RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic with vLLM.
# Usage: bash models/Llama-3.3-70B-Instruct-FP8-dynamic/serve.sh [model_name]
DIR=$(cd "$(dirname "$0")" && pwd)
VENV=$(bash "$DIR/../../venv.sh" "$DIR/requirements.txt") || exit 1
MODEL_NAME=${1:-"RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"}
echo "Using model: $MODEL_NAME"
# The server runs in the foreground; stop it with Ctrl+C.
exec "$VENV/bin/vllm" serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
