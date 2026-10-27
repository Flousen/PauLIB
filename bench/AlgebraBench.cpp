// Throughput of the kernels changed or added in the second core-operations
// round (October 2026), for the before/after comparison of design principle
// D7 (thesis Ch. 6, sec:kernel-design). Items = terms of the (left) operand.
// The rows above the PAULIB_ROUND2 block compile against the old headers as
// well; build once with and once without the old include path. Run once with
// all threads and once with OMP_NUM_THREADS=1.

#include <benchmark/benchmark.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include "PauLib.h"

using namespace paulib;

namespace {

constexpr std::size_t kQubits = 60;

PauliSum<64, double> bench_sum(std::size_t m, unsigned seed, bool canonical) {
    auto s = random_pauli_sum<64, double>(m, seed, kQubits);
    for (std::size_t i = 0; i < m; ++i) s.value[i] = 1.0 / static_cast<double>(1 + i % 97);
    if (canonical) s.canonicalize();
    return s;
}

PauliSum<64, double> rotation(double theta, unsigned seed) {
    auto r = random_pauli_sum<64, double>(1, seed, kQubits);
    r.flags[0] = 0;
    r.value[0] = theta;
    return r;
}

}  // namespace

// One rotation by a general angle on a collated sum: the work of one gate in
// propagate() before the truncation (expand_terms + add_combine).
static void BM_evolve_general(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto base = bench_sum(m, 1, true);
    const auto r = rotation(0.3, 2);
    for (auto _ : state) {
        state.PauseTiming();
        auto s = base;
        state.ResumeTiming();
        s.evolve(r);
        benchmark::DoNotOptimize(s.value.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// Clifford angle pi/2: in place (transform_terms); no copy needed.
static void BM_evolve_clifford(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = bench_sum(m, 3, true);
    const auto r = rotation(M_PI / 2, 4);
    for (auto _ : state) {
        s.evolve(r);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// canonicalize() on a sum that is canonical already (e.g. called twice).
static void BM_canonicalize_sorted(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = bench_sum(m, 5, true);
    for (auto _ : state) {
        s.canonicalize();
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

static void BM_canonicalize_unsorted(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto base = bench_sum(m, 6, false);
    for (auto _ : state) {
        state.PauseTiming();
        auto s = base;
        state.ResumeTiming();
        s.canonicalize();
        benchmark::DoNotOptimize(s.value.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// lhs (collated, m terms) += rhs (unsorted, m/8 terms): the shape of the
// append in evolve() and of a Runge-Kutta stage.
static void BM_add_sorted_lhs(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto lhs = bench_sum(m, 7, true);
    const auto rhs = bench_sum(m / 8, 8, false);
    for (auto _ : state) {
        state.PauseTiming();
        auto s = lhs;
        state.ResumeTiming();
        s += rhs;
        benchmark::DoNotOptimize(s.value.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// Two collated sums of m terms; half of the strings of b occur in a.
static void BM_inner_product_sorted(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto a = bench_sum(m, 9, true);
    auto b = bench_sum(m / 2, 10, false);
    PauliSum<64, double> half = a;
    std::vector<uint8_t> keep(half.get_length());
    for (std::size_t i = 0; i < keep.size(); ++i) keep[i] = i % 2;
    half.compact(keep);
    b += half;
    for (auto _ : state) benchmark::DoNotOptimize(inner_product(a, b));
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

#define SIZES ->Arg(1 << 10)->Arg(1 << 14)->Arg(1 << 18)->Arg(1 << 20)->Unit(benchmark::kMicrosecond)
BENCHMARK(BM_evolve_general) SIZES;
BENCHMARK(BM_evolve_clifford)->Arg(1 << 6) SIZES;
BENCHMARK(BM_canonicalize_sorted) SIZES;
BENCHMARK(BM_canonicalize_unsorted) SIZES;
BENCHMARK(BM_add_sorted_lhs) SIZES;
BENCHMARK(BM_inner_product_sorted) SIZES;

#ifdef PAULIB_ROUND2
// New kernels: absolute throughput only.

static void BM_evolve_imaginary(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto base = bench_sum(m, 11, true);
    const auto r = rotation(0.3, 12);
    for (auto _ : state) {
        state.PauseTiming();
        auto s = base;
        state.ResumeTiming();
        s.evolve_imaginary(r);
        benchmark::DoNotOptimize(s.value.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

static void BM_adjoint(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum(m, 13, false);
    for (auto _ : state) benchmark::DoNotOptimize(adjoint(s).value.data());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

static void BM_remap_qubits(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum(m, 14, false);
    std::vector<std::size_t> map(kQubits);
    std::iota(map.rbegin(), map.rend(), std::size_t{0});
    for (auto _ : state) benchmark::DoNotOptimize(remap_qubits(s, kQubits, map).value.data());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// |a| = |b| = sqrt(m) on 30 qubits each: m output terms.
static void BM_tensor_product(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const std::size_t k = static_cast<std::size_t>(std::sqrt(static_cast<double>(m)));
    const auto a = random_pauli_sum<64, double>(k, 15, 30), b = random_pauli_sum<64, double>(k, 16, 30);
    for (auto _ : state) benchmark::DoNotOptimize(tensor_product(a, b).value.data());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(k * k));
}

BENCHMARK(BM_evolve_imaginary) SIZES;
BENCHMARK(BM_adjoint) SIZES;
BENCHMARK(BM_remap_qubits) SIZES;
BENCHMARK(BM_tensor_product) SIZES;
#endif

#ifdef PAULIB_ROUND3
// Third round (November 2026): qubit maps (QMP) and the local map (ROT).

static void BM_shift_qubits(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum(m, 17, false);
    for (auto _ : state) benchmark::DoNotOptimize(shift_qubits(s, 7).value.data());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// 6 x 10 lattice = kQubits.
static void BM_shift_lattice(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum(m, 18, false);
    for (auto _ : state) benchmark::DoNotOptimize(shift_lattice(s, 6, 10, 1, 2).value.data());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// Every second qubit.
static void BM_restrict_qubits(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = bench_sum(m, 19, false);
    std::vector<std::size_t> q;
    for (std::size_t i = 0; i < kQubits; i += 2) q.push_back(i);
    for (auto _ : state) benchmark::DoNotOptimize(restrict_qubits(s, q).value.data());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// Four qubits fixed to Z eigenvalues (the X bits there cleared first).
static void BM_fix_qubits(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = bench_sum(m, 20, false);
    const std::size_t mask = (std::size_t{1} << 3) | (std::size_t{1} << 17) | (std::size_t{1} << 40) | (std::size_t{1} << 59);
    for (auto& w : s.x_data[0]) w &= ~mask;
    const std::vector<FixedQubit> fixes{{3, 'Z', 1}, {17, 'Z', -1}, {40, 'Z', 1}, {59, 'Z', -1}};
    for (auto _ : state) benchmark::DoNotOptimize(fix_qubits(s, fixes).value.data());
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// One gate as a transfer matrix on a collated sum (a copy per iteration, as
// BM_evolve_general): the rotation exp(-0.15 i X_5 Z_31) as a PTM, a CCZ on
// (2, 30, 50), and amplitude damping on qubit 9.
DenseMatrix rotation_matrix() {
    const std::complex<double> c(std::cos(0.15), 0), s(0, -std::sin(0.15));
    // exp(-i t P) with P = X (x) Z on (qubit 5 = bit 0, qubit 31 = bit 1).
    DenseMatrix u(16, 0.0);
    const double z[2] = {1, -1};
    for (std::size_t b = 0; b < 4; ++b) {
        u[b * 4 + b] = c;
        u[(b ^ 1) * 4 + b] = s * z[(b >> 1) & 1];
    }
    return u;
}

template <int Kind>
static void BM_apply_ptm(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto base = bench_sum(m, 21, true);
    PauliTransferMatrix ptm;
    std::vector<std::size_t> qubits;
    if constexpr (Kind == 0) {
        ptm = heisenberg_ptm({rotation_matrix()}, 2);
        qubits = {5, 31};
    } else if constexpr (Kind == 1) {
        DenseMatrix ccz(64, 0.0);
        for (std::size_t b = 0; b < 8; ++b) ccz[b * 8 + b] = b == 7 ? -1.0 : 1.0;
        ptm = heisenberg_ptm({ccz}, 3);
        qubits = {2, 30, 50};
    } else {
        const double g = 0.1;
        ptm = heisenberg_ptm({DenseMatrix{1, 0, 0, std::sqrt(1 - g)}, DenseMatrix{0, std::sqrt(g), 0, 0}}, 1);
        qubits = {9};
    }
    for (auto _ : state) {
        state.PauseTiming();
        auto s = base;
        state.ResumeTiming();
        apply_ptm(s, qubits, ptm);
        benchmark::DoNotOptimize(s.value.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

BENCHMARK(BM_shift_qubits) SIZES;
BENCHMARK(BM_shift_lattice) SIZES;
BENCHMARK(BM_restrict_qubits) SIZES;
BENCHMARK(BM_fix_qubits) SIZES;
BENCHMARK_TEMPLATE(BM_apply_ptm, 0)->Name("BM_apply_ptm_rotation") SIZES;
BENCHMARK_TEMPLATE(BM_apply_ptm, 1)->Name("BM_apply_ptm_ccz") SIZES;
BENCHMARK_TEMPLATE(BM_apply_ptm, 2)->Name("BM_apply_ptm_damping") SIZES;
#endif

BENCHMARK_MAIN();
