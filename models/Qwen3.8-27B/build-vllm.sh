#!/bin/bash
# gpu1: build vLLM v0.24.0 from source against torch 2.11.0+cu128 with the user-space CUDA
# 12.8 toolkit (+ libraries, + CCCL 2.8.2 from 12.9.1; installed on first use) and a
# user-space Rust toolchain (rustup; vLLM 0.24 has a Rust frontend), then install the wheel
# + runtime deps into the venv of requirements-vllm.txt. Prints the venv path. Chatty (stderr).
# ~20 min with MAX_JOBS=64 (the pod has 32 CPUs). See README.md.
# Usage: VENV=$(bash models/Qwen3.8-27B/build-vllm.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE/../..
SPEC=${1:-$HERE/requirements-vllm.txt}
if ! grep -q 'define CCCL_VERSION 20080' "$HOME/cuda-12.8/include/cuda/std/__cccl/version.h" 2>/dev/null \
    || [ ! -e "$HOME/cuda-12.8/lib/libcublas.so" ]; then
    WITH_LIBS=1 CCCL_FROM=12.9.1 bash "$ROOT/tools/install-cuda-toolkit.sh" >&2
fi
VENV=$(bash "${VENV_SH:-$ROOT/venv.sh}" "$SPEC")
source "${CUDA_ENV:-$ROOT/tools/cuda-env.sh}" "$VENV"
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
VER=${VLLM_VERSION:-0.24.0}
SRC=${VLLM_SRC:-/tmp/vllm-src}          # overlay /tmp: plenty of space, not persistent
WHEELS=${WHEELS:-$HOME/wheels}          # the wheel is kept
mkdir -p "$WHEELS"

for i in 1 2 3 4 5 6 7 8; do  # rustup's own downloader has no retries; pod DNS is flaky
    command -v cargo >/dev/null || [ -x "$HOME/.cargo/bin/cargo" ] && break
    curl --proto '=https' --tlsv1.2 -sSf --retry 8 --retry-all-errors https://sh.rustup.rs \
        | sh -s -- -y --profile minimal >&2 || sleep 10
done
export PATH=$HOME/.cargo/bin:$PATH

if [ ! -d "$SRC/.git" ]; then
    for i in 1 2 3 4 5 6; do  # pod DNS is flaky
        git clone -q --depth 1 --branch "v$VER" https://github.com/vllm-project/vllm.git "$SRC" && break
        rm -rf "$SRC"; sleep 5
    done
fi
cd "$SRC"
git checkout -q -- requirements pyproject.toml 2>/dev/null || true
"$VENV/bin/python" use_existing_torch.py >&2   # keep our torch cu128, drop vLLM's torch pins
# CUDA-13 extras in the runtime requirements -> cu12 flavours (driver is 12.4)
"$VENV/bin/python" - <<'EOF'
import re
p = "requirements/cuda.txt"
s = open(p).read()
s = s.replace("nvidia-cutlass-dsl[cu13]", "nvidia-cutlass-dsl").replace("humming-kernels[cu13]", "humming-kernels[cu12]")
assert "[cu13]" not in s, re.findall(r".*\[cu13\].*", s)
open(p, "w").write(s)
EOF
if ! ls "$WHEELS"/vllm-"$VER"*.whl >/dev/null 2>&1; then
    export VLLM_TARGET_DEVICE=cuda TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=${MAX_JOBS:-64} NVCC_THREADS=4
    export CMAKE_BUILD_TYPE=Release SETUPTOOLS_SCM_PRETEND_VERSION=$VER
    # CMake FetchContent clones ~10 repos (+ submodules) from GitHub at configure time and the
    # pod DNS drops lookups: retry; deps that were fetched stay in .deps/, so it converges.
    for i in $(seq 1 "${BUILD_ATTEMPTS:-10}"); do
        echo "== build attempt $i $(date -u +%T)" >&2
        "$UV" build --wheel --no-build-isolation --python "$VENV/bin/python" -o "$WHEELS" . \
            > attempt.log 2>&1 && { cat attempt.log >&2; break; }
        cat attempt.log >&2
        # only network failures are worth retrying; compile errors are deterministic
        grep -qE 'Could not resolve host|Failed to connect|unable to access' attempt.log || exit 1
        sleep 15
    done
fi
WHL=$(ls "$WHEELS"/vllm-"$VER"*.whl | head -1)
VIRTUAL_ENV=$VENV "$UV" pip install "$WHL" --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match >&2
cd "$HOME"  # not from $SRC: there `import vllm` finds the uncompiled source tree
"$VENV/bin/python" - >&2 <<'EOF'
import torch, vllm, vllm._C_stable_libtorch
from vllm.model_executor.models.registry import ModelRegistry as R
print("vllm", vllm.__version__, "| torch", torch.__version__, "| qwen3_5:",
      "Qwen3_5ForConditionalGeneration" in R.get_supported_archs())
EOF
echo "$VENV"
