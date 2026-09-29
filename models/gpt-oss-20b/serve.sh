#!/bin/bash
# Serve openai/gpt-oss-20b with vLLM. Optimized for H100 (80GB VRAM):
# - kv-cache-dtype fp8: maximizes throughput and saves VRAM
# - max-model-len 32768: generous context window
# - gpu-memory-utilization 0.90: 20B fits easily, leaving some headroom
# - trust-remote-code: support for custom layers
# Usage: bash models/gpt-oss-20b/serve.sh
DIR=$(cd "$(dirname "$0")" && pwd)
VENV=$(bash "$DIR/../../venv.sh" "$DIR/requirements.txt") || exit 1
MODEL_NAME="openai/gpt-oss-20b"
echo "Starting vLLM server for $MODEL_NAME..."
exec "$VENV/bin/vllm" serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
