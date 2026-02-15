#!/bin/bash
# Script to serve openai/gpt-oss-20b using vLLM
# Optimized for H100 (80GB VRAM)

MODEL_NAME="openai/gpt-oss-20b"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Start the vLLM server
echo "Starting vLLM server for $MODEL_NAME..."
# Parameters optimized for 20B model on H100 (80GB VRAM):
# - kv-cache-dtype fp8: Essential for H100 to maximize throughput and save VRAM
# - max-model-len 32768: Provides a generous context window
# - gpu-memory-utilization 0.90: 20B fits easily, leaving some headroom
# - trust-remote-code: Ensuring support for custom layers
uv run vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
