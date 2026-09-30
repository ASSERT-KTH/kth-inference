#!/bin/bash
# gpu1 (no root, no CUDA toolkit in the JupyterHub container): user-space CUDA 12.8 toolkit
# from NVIDIA's redist tarballs (nvcc + cudart + cccl + nvrtc), enough to build torch CUDA
# extensions against torch cu128. Library headers (cublas/cusparse/cusolver/...) come from the
# nvidia-* pip wheels torch already installed; see cuda-env.sh.
# Usage: bash install-cuda-toolkit.sh [prefix=~/cuda-12.8] [version=12.8.1]
#   COMPONENTS="..." overrides the component list; WITH_LIBS=1 adds the math/profiling
#   libraries CMake builds against torch need (cublas, cufft, curand, cusparse, cusolver,
#   nvtx, cupti, nvjitlink, profiler_api; ~2.2 GB of tarballs), e.g. to build vLLM.
set -euo pipefail
PREFIX=${1:-$HOME/cuda-12.8}
VER=${2:-12.8.1}
BASE=https://developer.download.nvidia.com/compute/cuda/redist
DEFAULT="cuda_nvcc cuda_cudart cuda_cccl cuda_nvrtc cuda_crt cuda_nvvm"
LIBS="libcublas libcufft libcurand libcusparse libcusolver cuda_nvtx cuda_cupti libnvjitlink cuda_profiler_api"
COMPONENTS=${COMPONENTS:-$DEFAULT${WITH_LIBS:+ $LIBS}}
mkdir -p "$PREFIX"
cd "$(mktemp -d)"
CURL=(curl -sfL --retry 8 --retry-all-errors --retry-delay 5)  # pod DNS is flaky
"${CURL[@]}" "$BASE/redistrib_$VER.json" -o redist.json
for comp in $COMPONENTS; do
    rel=$(python3 -c "import json,sys;d=json.load(open('redist.json'));c=d.get('$comp');print(c['linux-x86_64']['relative_path'] if c else '')")
    [ -n "$rel" ] || { echo "skip $comp (not in $VER)"; continue; }
    echo "get $comp: $rel"
    "${CURL[@]}" "$BASE/$rel" -o comp.tar.xz  # to a file: a retried pipe would corrupt tar
    tar -xJf comp.tar.xz --strip-components=1 -C "$PREFIX" && rm comp.tar.xz
done
# CCCL_FROM=12.9.1: overlay the header-only CCCL of a newer CTK (vLLM 0.24 kernels use
# cuda::ptx sem_relaxed / shared-dst cp_async_bulk overloads missing from 12.8's CCCL 2.7)
if [ -n "${CCCL_FROM:-}" ]; then
    "${CURL[@]}" "$BASE/redistrib_$CCCL_FROM.json" -o redist-cccl.json
    rel=$(python3 -c "import json;print(json.load(open('redist-cccl.json'))['cuda_cccl']['linux-x86_64']['relative_path'])")
    echo "overlay cuda_cccl from $CCCL_FROM: $rel"
    "${CURL[@]}" "$BASE/$rel" -o comp.tar.xz
    tar -xJf comp.tar.xz --strip-components=1 -C "$PREFIX" && rm comp.tar.xz
fi
grep -m1 'define CCCL_VERSION' "$PREFIX/include/cuda/std/__cccl/version.h" || true
# redist archives use lib/, nvcc and torch's cpp_extension look in lib64/
[ -e "$PREFIX/lib64" ] || ln -s lib "$PREFIX/lib64"
"$PREFIX/bin/nvcc" --version | tail -2
du -sh "$PREFIX"
