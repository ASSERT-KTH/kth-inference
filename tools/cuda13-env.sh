# source me (gpu1): CUDA 13 on the CUDA 12.4 driver, through NVIDIA's forward compatibility.
# ~/.local/cuda13 = cuda-compat-13-0 (user-space libcuda 580) + a minimal nvcc 13.0, installed
# on first use by /datasets/local_installs/cuda13/install_cuda13.sh (no root, ~580 MB).
# FlashInfer's JIT (sampler: curand.h; FP8 DeepGEMM: cublasLt.h, nvrtc.h + -lcublasLt -lnvrtc)
# needs CUDA libraries the minimal toolkit lacks; the venv's nvidia/cu13 wheel has them.
# $VENV/cu13-extra/include links only the wheel's headers that the toolkit does NOT have:
# the whole wheel include dir shadows nvcc's crt/ headers ('__cudaLaunch' was not declared).
# $VENV/cu13-extra/lib gives the wheel's libX.so.N the unversioned libX.so names -lX needs.
# Usage: source cuda13-env.sh <venv with torch cu130>
VENV=${1:?usage: source cuda13-env.sh <venv>}
[ -f "$HOME/.local/cuda13/env.sh" ] || bash /datasets/local_installs/cuda13/install_cuda13.sh >&2
source "$HOME/.local/cuda13/env.sh"  # CUDA_HOME, PATH, LD_LIBRARY_PATH=.../compat first
EXTRA=$VENV/cu13-extra
if [ ! -f "$EXTRA/.complete" ]; then
    WHEEL=$(ls -d "$VENV"/lib/python3*/site-packages/nvidia/cu13)
    mkdir -p "$EXTRA/include" "$EXTRA/lib"
    for f in "$WHEEL"/include/*; do
        [ -e "$CUDA_HOME/include/$(basename "$f")" ] || ln -sfn "$f" "$EXTRA/include/"
    done
    for f in "$WHEEL"/lib/lib*.so.*; do
        ln -sfn "$f" "$EXTRA/lib/$(basename "$f" | sed -E 's/\.so\..*$/.so/')"
    done
    touch "$EXTRA/.complete"
fi
export CPATH=$EXTRA/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$EXTRA/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST=9.0  # H100 only: faster builds
