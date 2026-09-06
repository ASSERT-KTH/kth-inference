#!/bin/bash
# Serve Qwen3-Coder-30B-A3B-Instruct (MoE, ~3B active) with vLLM on a single H100.
# Verified 2026-09-06 on gpu1 (vLLM 0.15.1): ~178 tok/s single-stream decode,
# 57 GiB weights, ~41s engine init, tool calling OK with the qwen3_coder parser.

MODEL_NAME=${1:-"Qwen/Qwen3-Coder-30B-A3B-Instruct"}

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Starting vLLM server for $MODEL_NAME..."
# - kv-cache-dtype fp8:   halves KV memory on H100 (needed on the MIG slice)
# - max-model-len 32768:  generous context; raise if KV headroom allows
# - gpu-memory-utilization 0.90: MoE weights fit with headroom for KV
# - tool-call-parser qwen3_coder: matches Qwen3-Coder's XML tool-call format
uv run vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
