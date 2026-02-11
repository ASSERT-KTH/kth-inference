#!/bin/bash

# Simple script to start the vLLM server
# Usage: ./start.sh [model_name]
# Default model: Qwen/QwQ-32B-AWQ

# Get the model name from the first argument or use the default
MODEL_NAME=${1:-"Qwen/QwQ-32B-AWQ"}
echo "Using model: $MODEL_NAME"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Start the vLLM server
echo "Starting vLLM server..."
uv run python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --quantization awq_marlin \
    --max-model-len 32768 \
    --model $MODEL_NAME


# The server runs in the foreground, so the script will block here
# To stop the server, press Ctrl+C 