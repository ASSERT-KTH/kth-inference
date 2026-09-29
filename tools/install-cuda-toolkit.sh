#!/bin/bash
# gpu1 (no root, no CUDA toolkit in the JupyterHub container): user-space CUDA 12.8 toolkit
# from NVIDIA's redist tarballs (nvcc + cudart + cccl + nvrtc), enough to build torch CUDA
# extensions against torch cu128. Library headers (cublas/cusparse/cusolver/...) come from the
# nvidia-* pip wheels torch already installed; see cuda-env.sh.
# Usage: bash install-cuda-toolkit.sh [prefix=~/cuda-12.8] [version=12.8.1]
set -euo pipefail
PREFIX=${1:-$HOME/cuda-12.8}
VER=${2:-12.8.1}
BASE=https://developer.download.nvidia.com/compute/cuda/redist
mkdir -p "$PREFIX"
cd "$(mktemp -d)"
curl -sfL "$BASE/redistrib_$VER.json" -o redist.json
for comp in cuda_nvcc cuda_cudart cuda_cccl cuda_nvrtc cuda_crt cuda_nvvm; do
    rel=$(python3 -c "import json,sys;d=json.load(open('redist.json'));c=d.get('$comp');print(c['linux-x86_64']['relative_path'] if c else '')")
    [ -n "$rel" ] || { echo "skip $comp (not in $VER)"; continue; }
    echo "get $comp: $rel"
    curl -sfL "$BASE/$rel" | tar -xJ --strip-components=1 -C "$PREFIX"
done
# redist archives use lib/, nvcc and torch's cpp_extension look in lib64/
[ -e "$PREFIX/lib64" ] || ln -s lib "$PREFIX/lib64"
"$PREFIX/bin/nvcc" --version | tail -2
du -sh "$PREFIX"
