#!/bin/bash
# Script to serve Qwen2.5-VL-7B-Instruct using vLLM
# Optimized for H100 (80GB VRAM)

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Start the vLLM server
echo "Starting vLLM server for $MODEL_NAME..."
# Parameters optimized for 7B model on 80GB VRAM:
# - kv-cache-dtype fp8: Reduces VRAM usage for KV cache
# - max-model-len 32768: Supports long context
# - limit-mm-per-prompt image=4: Allows multi-image vision tasks
# - gpu-memory-utilization 0.80: 7B fits easily on H100
# - trust-remote-code: Required for Qwen-VL
uv run vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
