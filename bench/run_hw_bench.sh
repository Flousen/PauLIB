#!/usr/bin/env bash
# =============================================================================
# run_hw_bench.sh — cross-platform workload benchmark runner.
#
# Targets (auto-detected; run the same script on each machine):
#   * NVIDIA H100 / GH200 node    -> GPU multiply bench, plus the Grace CPU
#                                    workload suite
#   * Apple Mac (M4, arm64)       -> CPU workload benches (compiler-vectorized NEON)
#
# Results land in bench/results/hw_<hostname>_<date>.log — commit these; the
# paper tables (paper.tex, Sec. "Workload Case Studies") reference them.
#
# Prerequisites
#   Linux:  cmake, g++/clang++ (C++20), OpenMP; optional CUDA toolkit.
#   macOS:  brew install cmake libomp; Apple clang needs
#           -DOpenMP_ROOT=$(brew --prefix libomp) if not found automatically.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
BUILD=${BUILD:-build}
OUT="bench/results/hw_$(hostname -s)_$(date +%Y%m%d).log"
mkdir -p bench/results

# Pin OMP threads to cores so the OS scheduler can't migrate them mid-run
# (matters for bandwidth-sensitive kernels like threading_bench). Checked on
# the GH200 node: `numactl --hardware` shows a single NUMA node spanning all
# 72 Grace cores and all Grace memory (the only other node is the GPU's
# HBM, which the CPU benches never touch), so no numactl binding is needed
# there — OMP_PROC_BIND/OMP_PLACES is the only pinning that applies.
export OMP_PROC_BIND=true
export OMP_PLACES=cores

run() {
    echo "== $*" | tee -a "$OUT"
    "$@" 2>&1 | tee -a "$OUT"
    echo | tee -a "$OUT"
}

{
    echo "# PauLIB hardware benchmark — $(date -u +%FT%TZ)"
    echo "# host: $(hostname)"
    echo "# os:   $(uname -srm)"
    if [ -r /proc/cpuinfo ]; then
        echo "# cpu:  $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
        echo "# cores: $(nproc)"
    elif command -v sysctl >/dev/null 2>&1; then
        echo "# cpu:  $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
        echo "# cores: $(sysctl -n hw.ncpu)"
    fi
    command -v nvidia-smi >/dev/null 2>&1 && \
        echo "# gpu:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo
} | tee "$OUT"

# --- build (configure once; targets that lack toolchains are skipped) -------
cmake -S . -B "$BUILD" >/dev/null
cmake --build "$BUILD" -j "$( (nproc || sysctl -n hw.ncpu) 2>/dev/null )" \
      --target gsim_bench propagation_bench threading_bench 2>/dev/null || \
cmake --build "$BUILD" --target gsim_bench propagation_bench threading_bench

# --- CPU workload benches (all platforms) ------------------------------------
run "$BUILD/bench/gsim_bench" 64
run "$BUILD/bench/propagation_bench" 40 10 18
run "$BUILD/bench/threading_bench" 3000 3

# --- GPU benches (only when built) -------------------------------------------
if [ -x "$BUILD/bench/gpu_mul_bench" ]; then
    run "$BUILD/bench/gpu_mul_bench"
fi

echo "results written to $OUT"
