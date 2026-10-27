// Qubit-major PauliTable versus term-major PauliSum, and the CHP tableau.
//
// BM_gate_*: one Clifford gate on M terms of n qubits. PauliSum touches one
// bit of a word per term; PauliTable updates 64 terms per word. Items = terms.
// BM_tableau_circuit: uniform random Clifford (O(n^2) gates) on |0...0>;
// items = gates. BM_tableau_measure: Z measurement of every qubit after a
// random Clifford; items = measurements. All times are wall-clock times
// (UseRealTime), since the threaded gates run off the main thread.

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>

#include "PauLib.h"

using namespace paulib;

template <std::size_t N>
static void BM_gate_sum_H(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = random_pauli_sum<N, double>(m, 1);
    for (auto _ : state) {
        s.apply_H(5);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_gate_table_H(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    PauliTable t = to_table(random_pauli_sum<N, double>(m, 1));
    for (auto _ : state) {
        t.apply_H(5);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_gate_sum_CNOT(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    auto s = random_pauli_sum<N, double>(m, 2);
    for (auto _ : state) {
        s.apply_CNOT(3, N - 2);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

template <std::size_t N>
static void BM_gate_table_CNOT(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    PauliTable t = to_table(random_pauli_sum<N, double>(m, 2));
    for (auto _ : state) {
        t.apply_CNOT(3, N - 2);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

// Conversion cost, for reference: one 64 x 64 transpose per block.
template <std::size_t N>
static void BM_to_table(benchmark::State& state) {
    const std::size_t m = static_cast<std::size_t>(state.range(0));
    const auto s = random_pauli_sum<N, double>(m, 3);
    for (auto _ : state) {
        PauliTable t = to_table(s);
        benchmark::DoNotOptimize(t);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(m));
}

static void BM_tableau_circuit(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::mt19937_64 rng(4);
    const auto c = random_clifford(n, rng);
    for (auto _ : state) {
        Tableau<1024> tab(n);
        apply_circuit(tab, c);
        benchmark::DoNotOptimize(tab);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(c.size()));
    state.counters["gates"] = static_cast<double>(c.size());
}

static void BM_tableau_measure(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::mt19937_64 rng(5);
    const auto c = random_clifford(n, rng);
    Tableau<1024> prepared(n);
    apply_circuit(prepared, c);
    long random_outcomes = 0;
    for (auto _ : state) {
        state.PauseTiming();
        Tableau<1024> tab = prepared;
        state.ResumeTiming();
        for (std::size_t q = 0; q < n; ++q) random_outcomes += !tab.measure_z(q, rng).deterministic;
        benchmark::DoNotOptimize(tab);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(n));
    state.counters["random_frac"] =
        static_cast<double>(random_outcomes) / static_cast<double>(state.iterations() * n);
}

#define GATE_ARGS ->RangeMultiplier(4)->Range(1 << 16, 1 << 20)->UseRealTime()->Unit(benchmark::kMicrosecond)
#define GATE_ARGS_LARGE ->RangeMultiplier(4)->Range(1 << 16, 1 << 24)->UseRealTime()->Unit(benchmark::kMicrosecond)
BENCHMARK_TEMPLATE(BM_gate_sum_H, 64) GATE_ARGS_LARGE;
BENCHMARK_TEMPLATE(BM_gate_table_H, 64) GATE_ARGS_LARGE;
BENCHMARK_TEMPLATE(BM_gate_sum_H, 256) GATE_ARGS;
BENCHMARK_TEMPLATE(BM_gate_table_H, 256) GATE_ARGS;
BENCHMARK_TEMPLATE(BM_gate_sum_H, 1024) GATE_ARGS;
BENCHMARK_TEMPLATE(BM_gate_table_H, 1024) GATE_ARGS;
BENCHMARK_TEMPLATE(BM_gate_sum_CNOT, 64) GATE_ARGS_LARGE;
BENCHMARK_TEMPLATE(BM_gate_table_CNOT, 64) GATE_ARGS_LARGE;
BENCHMARK_TEMPLATE(BM_gate_sum_CNOT, 256) GATE_ARGS;
BENCHMARK_TEMPLATE(BM_gate_table_CNOT, 256) GATE_ARGS;
BENCHMARK_TEMPLATE(BM_gate_sum_CNOT, 1024) GATE_ARGS;
BENCHMARK_TEMPLATE(BM_gate_table_CNOT, 1024) GATE_ARGS;
BENCHMARK_TEMPLATE(BM_to_table, 64) GATE_ARGS;
BENCHMARK_TEMPLATE(BM_to_table, 1024) GATE_ARGS;
BENCHMARK(BM_tableau_circuit)->Arg(100)->Arg(250)->Arg(500)->Arg(1000)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_tableau_measure)->Arg(100)->Arg(250)->Arg(500)->Arg(1000)->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
