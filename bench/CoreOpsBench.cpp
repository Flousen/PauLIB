// Cost of the core operations added for COM, GF2 and GEN as a function of the
// word count W = ceil(n/64): each benchmark is instantiated for n = 64 ... 1024
// at a fixed term count, so items/s per n shows the O(W) scaling of
// tab:core-operations (thesis Ch. 6).

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>
#include <vector>

#include "PauLib.h"

using namespace paulib;

// All M^2 entries of the anticommutation matrix, row by row on the
// qubit-major table: one anticommutes_mask per term.
template <std::size_t N>
static void BM_anticommutation_rows(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = random_pauli_sum<N, double>(m, 1);
    const PauliTable t = to_table(s);
    for (auto _ : state) {
        for (std::size_t j = 0; j < m; ++j) {
            auto row = anticommutes_mask(t, s.get_pauli_string(j));
            benchmark::DoNotOptimize(row.data());
        }
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m * m));
}

// COM as a factor: one pass over the terms.
template <std::size_t N>
static void BM_damp_anticommuting(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = random_pauli_sum<N, double>(m, 2);
    const auto p = random_pauli_sum<N, double>(1, 3).get_pauli_string(0);
    for (auto _ : state) {
        damp_anticommuting(s, p, 0.999);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// Gaussian elimination of M strings (M x 2n) on the word arrays of a copy of
// the sum; items = row operations (upper bound: each pivot touches every row).
template <std::size_t N>
static void BM_gf2_rref(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto g = random_pauli_sum<N, double>(m, 4);
    std::size_t row_ops = 0;
    for (auto _ : state) {
        auto a = g;
        const auto piv = gf2_rref(a);
        row_ops += piv.size() * m;
        benchmark::DoNotOptimize(a.x_data[0].data());
    }
    state.SetItemsProcessed(static_cast<long>(row_ops));
}

// Uniform random strings: 2W random words each.
template <std::size_t N>
static void BM_random_pauli_string(benchmark::State& state) {
    std::mt19937_64 rng(5);
    for (auto _ : state) {
        auto p = random_pauli_string<N>(N, rng);
        benchmark::DoNotOptimize(p);
    }
    state.SetItemsProcessed(state.iterations());
}

// qDRIFT draw: binary search in the prefix sums, O(log M), independent of n.
static void BM_term_sampler(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = random_pauli_sum<64, double>(m, 6);
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (auto& v : s.value) v = u(rng);
    TermSampler sampler(s);
    for (auto _ : state) benchmark::DoNotOptimize(sampler.sample(rng));
    state.SetItemsProcessed(state.iterations());
}

#define CORE_OPS_FOR_N(N)                                                        \
    BENCHMARK_TEMPLATE(BM_anticommutation_rows, N)->Arg(2048)->UseRealTime();   \
    BENCHMARK_TEMPLATE(BM_damp_anticommuting, N)->Arg(1 << 20)->UseRealTime();  \
    BENCHMARK_TEMPLATE(BM_gf2_rref, N)->Arg(512);                               \
    BENCHMARK_TEMPLATE(BM_random_pauli_string, N);

CORE_OPS_FOR_N(64)
CORE_OPS_FOR_N(128)
CORE_OPS_FOR_N(256)
CORE_OPS_FOR_N(512)
CORE_OPS_FOR_N(1024)

BENCHMARK(BM_term_sampler)->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 22);

BENCHMARK_MAIN();
