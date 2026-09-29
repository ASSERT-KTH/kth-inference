#!/bin/bash
# Qwen3.8-27B (qwen3_5): create the venv from requirements.txt (via venv.sh), then build
# causal-conv1d from source against its torch with the user-space CUDA toolkit
# (tools/install-cuda-toolkit.sh, installed on first use). Prints the venv path.
# Usage: VENV=$(bash models/Qwen3.8-27B/setup.sh) ; or setup.sh <existing venv> to add the kernels only
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$DIR/../..
VENV=${1:-$(bash "$ROOT/venv.sh" "$DIR/requirements.txt")}
[ -x "$HOME/cuda-12.8/bin/nvcc" ] || bash "$ROOT/tools/install-cuda-toolkit.sh" >&2
source "$ROOT/tools/cuda-env.sh" "$VENV"
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
if ! "$VENV/bin/python" -c "import causal_conv1d" 2>/dev/null; then
    # from source against *this* torch: uv's cache / the project's prebuilt release wheels
    # are built for other torch versions (ImportError: undefined symbol ...c10_cuda_check_implementation)
    VIRTUAL_ENV=$VENV CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 "$UV" pip install --no-deps \
        --no-build-isolation --no-cache --no-binary causal-conv1d --reinstall-package causal-conv1d \
        causal-conv1d==1.7.0 >&2
fi
"$VENV/bin/python" - >&2 <<'EOF'
import torch
from transformers.utils import import_utils as u
from causal_conv1d import causal_conv1d_fn
x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
causal_conv1d_fn(x, w)
print("torch", torch.__version__, "| fast kernels:", u.is_flash_linear_attention_available(),
      u.is_causal_conv1d_available())
EOF
echo "$VENV"
