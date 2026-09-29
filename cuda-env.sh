# source me (gpu1): user-space CUDA 12.8 toolkit + header dirs of torch's nvidia-* wheels.
# Usage: source cuda-env.sh [venv=~/.venvs/sft-lora]
VENV=${1:-$HOME/.venvs/sft-lora}
export CUDA_HOME=$HOME/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
for d in "$VENV"/lib/python3*/site-packages/nvidia/*/include; do
    CPATH=$d:${CPATH:-}
done
export CPATH
export TORCH_CUDA_ARCH_LIST=9.0  # H100 only: faster builds
