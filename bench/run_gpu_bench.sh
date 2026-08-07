#!/usr/bin/env bash
# Run gpu_mul_bench with numactl on GH200.
# Build first (nvcc must be in PATH):
#   PATH=/usr/local/cuda-12.4/bin:$PATH cmake -S . -B build \
#       -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc
#   cmake --build build --target gpu_mul_bench -j$(nproc)
#   node 0 = Grace CPU DRAM (490 GB LPDDR5X)
#   node 1 = H100 HBM3     ( 96 GB, cudaMalloc goes here automatically)
# Pinning CPU and staging allocations to node 0 avoids cross-NUMA latency
# on the host side; GPU allocations via cudaMalloc are unaffected.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/../build/bench/gpu_mul_bench"

if [[ ! -x "$BENCH" ]]; then
    echo "gpu_mul_bench not found at $BENCH — run cmake --build build first" >&2
    exit 1
fi

exec numactl --cpunodebind=0 --membind=0 "$BENCH" "$@"
