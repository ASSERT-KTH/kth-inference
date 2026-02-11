#!/bin/bash
# test
# Simple script to start the vLLM server
# Usage: ./start.sh [model_name]

# Get the model name from the first argument or use the default
MODEL_NAME=${1:-"RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"}
echo "Using model: $MODEL_NAME"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Start the vLLM server
echo "Starting vLLM server..."
uv run python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --model $MODEL_NAME \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json

    #--quantization awq_marlin \
# The server runs in the foreground, so the script will block here
# To stop the server, press Ctrl+C 
