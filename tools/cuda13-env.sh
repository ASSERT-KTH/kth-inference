# source me (gpu1): CUDA 13 on the CUDA 12.4 driver, through NVIDIA's forward compatibility.
# ~/.local/cuda13 = cuda-compat-13-0 (user-space libcuda 580) + a minimal nvcc 13.0, installed
# on first use by /datasets/local_installs/cuda13/install_cuda13.sh (no root, ~580 MB).
# Also gives FlashInfer's JIT the cuRAND headers it needs, from the venv's nvidia/cu13 wheel:
# only curand*.h, because the whole include dir shadows nvcc's crt/ headers ('__cudaLaunch').
# Usage: source cuda13-env.sh <venv with torch cu130>
VENV=${1:?usage: source cuda13-env.sh <venv>}
[ -f "$HOME/.local/cuda13/env.sh" ] || bash /datasets/local_installs/cuda13/install_cuda13.sh >&2
source "$HOME/.local/cuda13/env.sh"  # CUDA_HOME, PATH, LD_LIBRARY_PATH=.../compat first
if [ ! -d "$VENV/curand-include" ]; then
    mkdir -p "$VENV/curand-include"
    ln -sf "$VENV"/lib/python3*/site-packages/nvidia/cu13/include/curand*.h "$VENV/curand-include/"
fi
export CPATH=$VENV/curand-include${CPATH:+:$CPATH}
export TORCH_CUDA_ARCH_LIST=9.0  # H100 only: faster builds
