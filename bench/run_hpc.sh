#!/usr/bin/env bash
# Run the full paper benchmark suite on an HPC node and produce all figures.
#
# Usage (from repo root):
#   bash bench/run_hpc.sh
#
# Assumes: cmake, python3, numactl are on PATH; all pip deps are installed.
# Output: bench/results/*.csv  and  doc/img/fig_*.{pdf,png}

module load cmake/3.30.5-gcc-13.2.0-x7ma5x4 
module load gcc/13.2.0-gcc-7.5.0-kv6jgtu
module load python/3.13.0-gcc-13.2.0-el6dzxt 



set -euo pipefail
cd "$(dirname "$0")/.."   # always run from repo root

# ---------------------------------------------------------------------------
# 1. Build with Python bindings
# ---------------------------------------------------------------------------
echo "=== Building PauLIB ==="
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target _paulib -j "$(nproc)"

# ---------------------------------------------------------------------------
# 2. Run benchmarks under numactl with HPC-tuned OpenMP settings
# ---------------------------------------------------------------------------
echo ""
echo "=== Running paper benchmarks ==="

export OMP_NUM_THREADS=96
export OMP_PLACES=CORES
export OMP_PROC_BIND=CLOSE
export PAULIB_BENCH_THREADS=1,2,4,8,16,24,32,48,64,96
export PYTHONPATH="$(pwd)/python"

numactl --membind=8-15 \
    python3 bench/paper_bench.py

# ---------------------------------------------------------------------------
# 3. Generate figures
# ---------------------------------------------------------------------------
echo ""
echo "=== Generating figures ==="
python3 bench/plot_paper.py

echo ""
echo "All done. Figures in doc/img/, CSVs in bench/results/."
