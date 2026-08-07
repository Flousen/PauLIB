#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <vector>

#include "PauLib.h"

using namespace paulib;

static constexpr std::size_t NUM_QUBITS = 100;

static void BM_collate_soa(benchmark::State& state) {
    using T = double;
    const std::size_t lhs_n = static_cast<std::size_t>(state.range(0));
    const std::size_t rhs_n = static_cast<std::size_t>(state.range(1));
    PauliSum<NUM_QUBITS, T> lhs = paulib::random_pauli_sum<NUM_QUBITS, T>(lhs_n, 42);
    PauliSum<NUM_QUBITS, T> rhs = paulib::random_pauli_sum<NUM_QUBITS, T>(rhs_n, 123);
    for (auto _ : state) {
        state.PauseTiming();
        auto lhs_copy = lhs;
        lhs_copy *= rhs;
        state.ResumeTiming();

        lhs_copy.sort_and_combine();  // Only this is timed
        benchmark::DoNotOptimize(lhs_copy);
    }
    state.SetItemsProcessed(state.iterations() *
                            static_cast<long>(lhs_n * rhs_n));
    state.SetLabel(std::to_string(lhs_n) + "x" + std::to_string(rhs_n));
}

BENCHMARK(BM_collate_soa)->Args({1000, 100 });
BENCHMARK(BM_collate_soa)->Args({1000, 500 });
BENCHMARK(BM_collate_soa)->Args({1000, 1000});
BENCHMARK(BM_collate_soa)->Args({1000, 2000});

BENCHMARK_MAIN();
