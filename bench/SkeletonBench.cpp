// Throughput of the kernels that the skeleton round moves onto
// transform_values / filter_terms / reduce_terms (PauliSkeletons.h), for the
// before/after comparison of design principle D7 (thesis Ch. 6,
// sec:kernel-design). Items = terms. Run once with all threads and once with
// OMP_NUM_THREADS=1.

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "PauLib.h"

using namespace paulib;

template <std::size_t N>
static PauliSum<N, double> bench_sum(std::size_t m, unsigned seed) {
    auto s = random_pauli_sum<N, double>(m, seed);
    for (std::size_t i = 0; i < m; ++i) s.value[i] = 1.0 / static_cast<double>(1 + i % 97);
    return s;
}

template <std::size_t N>
static void BM_damp_local(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = bench_sum<N>(m, 1);
    for (auto _ : state) {
        damp_local(s, 3, std::array<double, 4>{1.0, 0.999, 0.998, 0.997});
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_damp_anticommuting(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = bench_sum<N>(m, 2);
    const auto p = random_pauli_sum<N, double>(1, 3).get_pauli_string(0);
    for (auto _ : state) {
        damp_anticommuting(s, p, 0.999);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_add_noise(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = bench_sum<N>(m, 4);
    for (auto _ : state) {
        add_noise(s, 1e-6);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// The filters copy the sum each iteration; the copy is timed as well, so
// compare these rows only before against after.
template <std::size_t N>
static void BM_cutoff(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum<N>(m, 5);
    for (auto _ : state) {
        auto c = s;
        c.cutoff(1.0 / 50.0);
        benchmark::DoNotOptimize(c.value.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_truncate_weight(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum<N>(m, 6);
    for (auto _ : state) {
        auto c = s;
        c.truncate_weight(N * 3 / 4);
        benchmark::DoNotOptimize(c.value.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_frobenius_norm2(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum<N>(m, 7);
    for (auto _ : state) benchmark::DoNotOptimize(s.frobenius_norm2());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_l1_norm(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum<N>(m, 8);
    for (auto _ : state) benchmark::DoNotOptimize(s.l1_norm());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_weight_histogram(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum<N>(m, 9);
    for (auto _ : state) {
        auto h = s.weight_histogram();
        benchmark::DoNotOptimize(h.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// Random strings almost never consist of I and Z only; the cost is the scan.
template <std::size_t N>
static void BM_zero_state_expectation(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum<N>(m, 10);
    for (auto _ : state) benchmark::DoNotOptimize(s.zero_state_expectation());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_basis_state_expectation(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = bench_sum<N>(m, 11);
    // Clear the X parts so that every term contributes.
    for (std::size_t k = 0; k < PauliSum<N, double>::ArraySize; ++k)
        for (std::size_t i = 0; i < m; ++i) s.x_data[k][i] = 0;
    std::string bits(N, '0');
    for (std::size_t q = 0; q < N; q += 3) bits[q] = '1';
    for (auto _ : state) benchmark::DoNotOptimize(s.basis_state_expectation(bits));
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_trace(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum<N>(m, 12);
    for (auto _ : state) benchmark::DoNotOptimize(s.trace());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

#define SKELETON_BENCH(fn)                                                       \
    BENCHMARK_TEMPLATE(fn, 64)->RangeMultiplier(64)->Range(1 << 14, 1 << 20)     \
        ->UseRealTime()->Unit(benchmark::kMicrosecond);                          \
    BENCHMARK_TEMPLATE(fn, 256)->RangeMultiplier(64)->Range(1 << 14, 1 << 20)    \
        ->UseRealTime()->Unit(benchmark::kMicrosecond);

SKELETON_BENCH(BM_damp_local)
SKELETON_BENCH(BM_damp_anticommuting)
SKELETON_BENCH(BM_add_noise)
SKELETON_BENCH(BM_cutoff)
SKELETON_BENCH(BM_truncate_weight)
SKELETON_BENCH(BM_frobenius_norm2)
SKELETON_BENCH(BM_l1_norm)
SKELETON_BENCH(BM_weight_histogram)
SKELETON_BENCH(BM_zero_state_expectation)
SKELETON_BENCH(BM_basis_state_expectation)
SKELETON_BENCH(BM_trace)

BENCHMARK_MAIN();
