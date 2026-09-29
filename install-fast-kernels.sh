#!/bin/bash
# gpu1: fast qwen3_5 (gated delta-net) kernels for a torch-cu128 venv: flash-linear-attention
# (Triton, no nvcc) + causal-conv1d (CUDA extension, built with the user-space toolkit of
# install-cuda-toolkit.sh). --no-deps everywhere: never let pip move torch/transformers.
# Usage: bash install-fast-kernels.sh [venv=~/.venvs/sft-lora]
set -euo pipefail
VENV=${1:-$HOME/.venvs/sft-lora}
source "$(dirname "$0")/cuda-env.sh" "$VENV"
UV=~/.local/bin/uv
export VIRTUAL_ENV=$VENV
$UV pip install einops ninja packaging setuptools wheel  # build deps for --no-build-isolation
$UV pip install --no-deps flash-linear-attention fla-core
# build from source against *this* torch: uv's cache / the project's prebuilt release wheels
# are built for other torch versions (ImportError: undefined symbol ...c10_cuda_check_implementation)
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 $UV pip install --no-deps --no-build-isolation \
    --no-cache --no-binary causal-conv1d --reinstall-package causal-conv1d causal-conv1d
"$VENV/bin/python" - <<'EOF'
import torch, fla, causal_conv1d
from causal_conv1d import causal_conv1d_fn
x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
print("torch", torch.__version__, "fla", fla.__version__, "causal_conv1d", causal_conv1d.__version__,
      "conv ok", tuple(causal_conv1d_fn(x, w).shape))
EOF
