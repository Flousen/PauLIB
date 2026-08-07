// =============================================================================
// GpuMulBench.cpp — GPU vs CPU PauliSum outer-product benchmark
//
// Measures lhs ⊗ rhs for 500-qubit strings across several (n, m) sizes:
//
//   BM_gpu_kernel  — H100 SoA kernel (data pre-loaded on HBM, kernel time only)
//   BM_cpu_soa_kj  — 72-core Grace SoA multiply (OMP-parallel)
//
// GPU timing uses cudaEvents; CPU timing excludes per-iteration lhs copy.
// Run via bench/run_gpu_bench.sh (numactl --cpunodebind=0 --membind=0).
// =============================================================================

#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdio>
#include <string>

#include "PauLib.h"
#include "gpu/GpuPauliSum.h"

using namespace paulib;

static constexpr int NUM_QUBITS = 500;
using SoA = PauliSum<NUM_QUBITS, double>;

// ─── GPU benchmark ───────────────────────────────────────────────────────────
// Data is uploaded once before the loop; the timed section is the kernel only.
// cudaEvents give sub-microsecond GPU-side timing without CPU-GPU sync jitter.
static void BM_gpu_kernel(benchmark::State& state)
{
    const int n = (int)state.range(0);
    const int m = (int)state.range(1);

    SoA lhs = random_pauli_sum<NUM_QUBITS, double>(n, 42);
    SoA rhs = random_pauli_sum<NUM_QUBITS, double>(m, 123);

    GpuPauliSum<NUM_QUBITS> g_lhs(lhs);
    GpuPauliSum<NUM_QUBITS> g_rhs(rhs);
    GpuPauliSum<NUM_QUBITS> g_out(n * m);

    constexpr int AS = GpuPauliSum<NUM_QUBITS>::AS;
    auto do_launch = [&](cudaStream_t s = nullptr) {
        launch_gpu_mul_soa(AS,
            g_lhs.x_ptr(), g_lhs.z_ptr(), g_lhs.flags_ptr(), g_lhs.val_ptr(),
            g_rhs.x_ptr(), g_rhs.z_ptr(), g_rhs.flags_ptr(), g_rhs.val_ptr(),
            g_out.x_ptr(), g_out.z_ptr(), g_out.flags_ptr(), g_out.val_ptr(),
            n, m, s);
    };

    do_launch();                        // warm-up: flush driver/JIT overhead
    PAULIB_GPU_CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    PAULIB_GPU_CK(cudaEventCreate(&t0));
    PAULIB_GPU_CK(cudaEventCreate(&t1));

    for (auto _ : state) {
        PAULIB_GPU_CK(cudaEventRecord(t0));
        do_launch();
        PAULIB_GPU_CK(cudaEventRecord(t1));
        PAULIB_GPU_CK(cudaEventSynchronize(t1));
        float ms = 0.f;
        PAULIB_GPU_CK(cudaEventElapsedTime(&ms, t0, t1));
        state.SetIterationTime((double)ms * 1e-3);
    }

    state.SetItemsProcessed(state.iterations() * (int64_t)n * m);
    state.SetLabel(std::to_string(n) + "x" + std::to_string(m));
    PAULIB_GPU_CK(cudaEventDestroy(t0));
    PAULIB_GPU_CK(cudaEventDestroy(t1));
}

// ─── CPU reference benchmarks ────────────────────────────────────────────────

static void BM_cpu_soa_kj(benchmark::State& state)
{
    const size_t n = (size_t)state.range(0);
    const size_t m = (size_t)state.range(1);

    SoA lhs = random_pauli_sum<NUM_QUBITS, double>(n, 42);
    SoA rhs = random_pauli_sum<NUM_QUBITS, double>(m, 123);

    for (auto _ : state) {
        state.PauseTiming();
        SoA lhs_copy = lhs;
        state.ResumeTiming();
        lhs_copy.multiply(rhs);
        benchmark::DoNotOptimize(lhs_copy);
    }
    state.SetItemsProcessed(state.iterations() * (int64_t)n * m);
    state.SetLabel(std::to_string(n) + "x" + std::to_string(m));
}

#define SIZES \
    ->Args({1000,   100}) \
    ->Args({1000,   500}) \
    ->Args({1000,  1000}) \
    ->Args({1000,  2000}) \
    ->Args({1000,  5000}) \
    ->Args({10000, 10000})

BENCHMARK(BM_cpu_soa_kj) SIZES;
BENCHMARK(BM_gpu_kernel) SIZES ->UseManualTime();

BENCHMARK_MAIN();
