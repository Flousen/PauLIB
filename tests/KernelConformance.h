#pragma once

// Conformance harness for kernels (design principles D3 and D5, thesis Ch. 6,
// sec:kernel-design), tests only. A kernel passes if it gives bit-identical
// results with 1 and 8 threads on sums that span word boundaries and carry
// non-trivial real or complex coefficients. ScopedPolicy lowers the parallel
// threshold and the chunk length, so that small sums take the threaded path
// and span many chunks.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#include <omp.h>

#include "PauLib.h"

namespace conformance {

// Qubit counts around the 64-bit word boundaries.
inline constexpr std::array<std::size_t, 5> kQubitCounts{3, 63, 64, 65, 130};

struct ScopedPolicy {
    paulib::ExecPolicy saved;
    ScopedPolicy(std::size_t threshold, std::size_t chunk) : saved(paulib::default_policy()) {
        paulib::default_policy().parallel_threshold = threshold;
        paulib::default_policy().chunk_terms = chunk;
    }
    ~ScopedPolicy() { paulib::default_policy() = saved; }
};

template <typename F>
auto with_threads(int threads, F&& f) {
    const int saved = omp_get_max_threads();
    omp_set_num_threads(threads);
    auto r = f();
    omp_set_num_threads(saved);
    return r;
}

// Random strings (all four phases) with coefficients of both signs spread
// over six orders of magnitude, both parts for a complex T, so that the
// summation order shows in the last bits. With values in (-1, 1) only, an
// OpenMP reduction(+) over 1000 terms happened to give the same bits for 1
// and 8 threads (negative control, 28 Oct 2026).
template <std::size_t N, typename T>
paulib::PauliSum<N, T> random_sum(std::size_t terms, std::uint64_t seed, std::size_t qubits) {
    auto s = paulib::random_pauli_sum<N, T>(terms, seed, qubits);
    std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> e(-3.0, 3.0);
    auto draw = [&] { return u(rng) * std::pow(10.0, e(rng)); };
    for (auto& v : s.value) {
        if constexpr (paulib::is_complex_v<T>) v = T(draw(), draw());
        else v = T(draw());
    }
    return s;
}

template <typename R>
bool bit_equal(const R& a, const R& b) {
    static_assert(std::is_trivially_copyable_v<R>);
    return std::memcmp(&a, &b, sizeof(R)) == 0;
}

template <typename R>
bool bit_equal(const std::vector<R>& a, const std::vector<R>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!bit_equal(a[i], b[i])) return false;
    return true;
}

template <std::size_t N, typename T>
bool bit_equal(const paulib::PauliSum<N, T>& a, const paulib::PauliSum<N, T>& b) {
    return a.x_data == b.x_data && a.z_data == b.z_data && a.flags == b.flags &&
           bit_equal(a.value, b.value);
}

// f() returns the kernel's result (for an in-place kernel: a modified copy).
// Three thread counts: a single sum of 1000 terms let an OpenMP reduction(+)
// pass 1 against 8 threads for 2 of 5 seeds (negative control, 28 Oct 2026).
template <typename F>
void expect_thread_invariant(F&& f, const char* what) {
    const auto r1 = with_threads(1, f);
    for (const int threads : {3, 8}) {
        const auto rt = with_threads(threads, f);
        EXPECT_TRUE(bit_equal(r1, rt)) << what << ": 1 and " << threads << " threads differ";
    }
}

}  // namespace conformance
