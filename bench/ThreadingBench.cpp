// Thread-level parallelism scaling benchmark (paper Sec. "Thread-Level
// Parallelism"): PauliSum x PauliSum outer product (multiply), which is
// distributed across cores with #pragma omp parallel for schedule(static),
// one disjoint output block per right-hand-side term. Sweeps OMP thread
// count and reports wall time / speedup / parallel efficiency relative to
// the single-thread run.
//
// Usage: ./threading_bench [n_terms] [reps]
//   defaults: 500 qubits, 3000 terms per operand (9e6 output pairs), 3 reps

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "PauLib.h"

using namespace paulib;
using clk = std::chrono::steady_clock;

static constexpr std::size_t NUM_QUBITS = 500;

int main(int argc, char** argv) {
    const std::size_t n_terms = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 3000;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;

    const auto lhs = random_pauli_sum<NUM_QUBITS, double>(n_terms, 42);
    const auto rhs = random_pauli_sum<NUM_QUBITS, double>(n_terms, 123);

    const int max_threads = omp_get_max_threads();
    std::vector<int> thread_counts;
    for (int t = 1; t <= max_threads; t *= 2) thread_counts.push_back(t);
    if (thread_counts.back() != max_threads) thread_counts.push_back(max_threads);

    std::printf("== threading_bench qubits=%zu terms=%zu pairs=%zu max_threads=%d\n",
                NUM_QUBITS, n_terms, n_terms * n_terms, max_threads);
    std::printf("threads,time_s,speedup,efficiency\n");
    std::fflush(stdout);

    double baseline = 0.0;
    for (int t : thread_counts) {
        omp_set_num_threads(t);
        double best = 1e30;
        for (int r = 0; r < reps + 1; ++r) {  // one warm-up + `reps` timed runs
            auto lhs_copy = lhs;
            const auto t0 = clk::now();
            lhs_copy.multiply(rhs);
            const auto t1 = clk::now();
            if (r > 0)
                best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
            if (lhs_copy.get_length() == 0) std::fprintf(stderr, "unexpected empty result\n");
        }
        if (t == 1) baseline = best;
        std::printf("%d,%.6f,%.4f,%.4f\n", t, best, baseline / best, (baseline / best) / t);
        std::fflush(stdout);
    }

    return 0;
}
