#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "DenseReference.h"
#include "KernelConformance.h"
#include "PauLib.h"

using namespace paulib;
using cd = std::complex<double>;

namespace {

constexpr double kTol = 1e-12;
constexpr std::size_t kCap = 192;

// The coefficients of conformance::random_sum span six orders of magnitude,
// so compare relative to the size of the result.
double rel_tol(double ref) { return 1e-12 * std::max(1.0, std::abs(ref)); }

template <typename T>
class SkeletonTyped : public ::testing::Test {};
using SkeletonTypes = ::testing::Types<double, cd>;
TYPED_TEST_SUITE(SkeletonTyped, SkeletonTypes);

// Every kernel moved onto a skeleton: bit-identical for 1 and 8 threads, on
// 1000 terms in chunks of 7 (not a divisor of 1000), threads from 1 term on.
TYPED_TEST(SkeletonTyped, KernelsBitIdenticalForThreadCounts) {
    using T = TypeParam;
    using R = real_t<T>;
    conformance::ScopedPolicy policy(1, 7);
    for (const std::size_t n : conformance::kQubitCounts) {
        SCOPED_TRACE("n = " + std::to_string(n));
        const auto s = conformance::random_sum<kCap, T>(1000, 11 + n, n);
        const PauliString<kCap> v = random_pauli_sum<kCap, T>(1, 12 + n, n).get_pauli_string(0);
        std::string bits(n, '0');
        for (std::size_t q = 0; q < n; q += 2) bits[q] = '1';
        auto diag = s;  // I/Z strings only, so the basis-state reductions see terms
        for (auto& w : diag.x_data) std::fill(w.begin(), w.end(), 0);

        conformance::expect_thread_invariant([&] { return s.frobenius_norm2(); }, "frobenius_norm2");
        conformance::expect_thread_invariant([&] { return s.l1_norm(); }, "l1_norm");
        conformance::expect_thread_invariant([&] { return s.weight_histogram(); }, "weight_histogram");
        conformance::expect_thread_invariant([&] { return s.trace(); }, "trace");
        conformance::expect_thread_invariant([&] { return diag.zero_state_expectation(); }, "zero_state_expectation");
        conformance::expect_thread_invariant([&] { return diag.basis_state_expectation(bits); }, "basis_state_expectation");
        conformance::expect_thread_invariant([&] { return signed_norm2(s, v); }, "signed_norm2");
        conformance::expect_thread_invariant([&] { return anticommutes_mask(s, v); }, "anticommutes_mask");
        conformance::expect_thread_invariant([&] { auto c = s; damp_anticommuting(c, v, R(0.7)); return c; }, "damp_anticommuting");
        conformance::expect_thread_invariant([&] { auto c = s; apply_pauli_channel(c, n - 1, R(0.1), R(0.2), R(0.05)); return c; }, "apply_pauli_channel");
        conformance::expect_thread_invariant([&] { auto c = s; add_noise(c, R(0.01)); return c; }, "add_noise");
        conformance::expect_thread_invariant([&] { auto c = s; c.scale(T(0.5)); return c; }, "scale");
        conformance::expect_thread_invariant([&] { auto c = s; c.cutoff(R(0.5)); return c; }, "cutoff");
        conformance::expect_thread_invariant([&] { auto c = s; c.truncate_weight(n / 2); return c; }, "truncate_weight");
    }
}

// The skeleton versions agree with a plain serial loop up to rounding.
TYPED_TEST(SkeletonTyped, ReductionsMatchSerialLoop) {
    using T = TypeParam;
    using R = real_t<T>;
    conformance::ScopedPolicy policy(1, 7);
    for (const std::size_t n : conformance::kQubitCounts) {
        SCOPED_TRACE("n = " + std::to_string(n));
        const auto s = conformance::random_sum<kCap, T>(500, 21 + n, n);
        const PauliString<kCap> v = random_pauli_sum<kCap, T>(1, 22 + n, n).get_pauli_string(0);
        R f = 0, l1 = 0, sn = 0;
        std::vector<R> h(n + 1, R{});
        for (std::size_t i = 0; i < s.get_length(); ++i) {
            const auto p = s.get_pauli_string(i);
            f += abs2(s.value[i]);
            l1 += std::abs(s.value[i]);
            sn += (p.anti_commute(v) ? -1 : 1) * abs2(s.value[i]);
            std::size_t w = 0;
            for (std::size_t q = 0; q < n; ++q)
                w += ((p.x[q / 64] | p.z[q / 64]) >> (q % 64)) & 1;
            h[w] += abs2(s.value[i]);
        }
        EXPECT_NEAR(s.frobenius_norm2(), f, rel_tol(f));
        EXPECT_NEAR(s.l1_norm(), l1, rel_tol(l1));
        EXPECT_NEAR(signed_norm2(s, v), sn, rel_tol(f));
        const auto hs = s.weight_histogram();
        ASSERT_EQ(hs.size(), h.size());
        for (std::size_t w = 0; w <= n; ++w) EXPECT_NEAR(hs[w], h[w], rel_tol(h[w]));
    }
}

// Chunk boundaries: empty sum, one term, and one term less, equal and more
// than a chunk.
TEST(Skeleton, ReduceTermsAtChunkBoundaries) {
    conformance::ScopedPolicy policy(1, 16);
    for (const std::size_t m : {0, 1, 15, 16, 17, 33}) {
        auto s = conformance::random_sum<64, double>(m, 31 + m, 10);
        double ref = 0;
        for (double v : s.value) ref += v * v;
        EXPECT_NEAR(s.frobenius_norm2(), ref, rel_tol(ref)) << "m = " << m;
        EXPECT_EQ(reduce_terms<std::size_t>(s, [](const TermView<64, double>&) { return std::size_t{1}; }), m);
    }
}

TEST(Skeleton, FilterKeepsOrder) {
    conformance::ScopedPolicy policy(1, 5);
    auto s = conformance::random_sum<64, double>(100, 41, 20);
    const auto ref = s;
    filter_terms(s, [](const TermView<64, double>& t) { return t.index() % 3 == 0; });
    ASSERT_EQ(s.get_length(), 34u);
    for (std::size_t i = 0; i < s.get_length(); ++i) {
        EXPECT_EQ(s.value[i], ref.value[3 * i]);
        EXPECT_EQ(s.get_pauli_string(i), ref.get_pauli_string(3 * i));
    }
}

TEST(Skeleton, BinnedReductionIgnoresKeysOutOfRange) {
    conformance::ScopedPolicy policy(1, 4);
    auto s = conformance::random_sum<64, double>(50, 51, 8);
    const auto h = reduce_terms_binned<double>(
        s, 3, [](const TermView<64, double>& t) { return t.index() % 5; },
        [](const TermView<64, double>&) { return 1.0; });
    ASSERT_EQ(h.size(), 3u);
    EXPECT_EQ(h[0], 10.0);
    EXPECT_EQ(h[1], 10.0);
    EXPECT_EQ(h[2], 10.0);
}

// With the default policy (threshold 2^15, chunks of 2^12), a large sum takes
// the threaded path and stays bit-identical.
TEST(Skeleton, DefaultPolicyLargeSumBitIdentical) {
    const auto s = conformance::random_sum<128, double>(1 << 16, 61, 100);
    conformance::expect_thread_invariant([&] { return s.frobenius_norm2(); }, "frobenius_norm2");
    conformance::expect_thread_invariant([&] { return s.weight_histogram(); }, "weight_histogram");
}

// K4: signed_norm2(A, V) = tr(A† V A V) / 2^n on a collated sum, against
// dense matrices.
TYPED_TEST(SkeletonTyped, SignedNorm2MatchesDense) {
    using T = TypeParam;
    for (const std::size_t n : {std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
        for (std::uint64_t seed = 0; seed < 5; ++seed) {
            auto a = conformance::random_sum<8, T>(12, 71 + 10 * n + seed, n);
            a.canonicalize();
            auto vp = random_pauli_sum<8, T>(1, 91 + seed, n).get_pauli_string(0);
            vp.flags = 0;
            const auto A = dense::sum(a);
            const auto V = dense::pauli(vp, n);
            const double d = static_cast<double>(std::size_t(1) << n);
            const dense::C ref = (A.adjoint() * V * A * V).trace() / d;
            const double scale = a.frobenius_norm2();
            EXPECT_NEAR(signed_norm2(a, vp), ref.real(), rel_tol(scale)) << "n = " << n << ", seed = " << seed;
            EXPECT_NEAR(ref.imag(), 0.0, rel_tol(scale));
        }
    }
}

// For a Hermitian A, signed_norm2 is the OTOC tr(A V A V) / 2^n; V
// commuting with every term gives the squared Frobenius norm.
TEST(Skeleton, SignedNorm2OfCommutingPauliIsFrobeniusNorm) {
    PauliSum<8, double> a(std::vector<std::pair<std::string, double>>{
        {"ZZII", 0.3}, {"IZZI", -0.4}, {"XXXX", 0.5}});
    EXPECT_NEAR(signed_norm2(a, PauliString<8>("ZZZZ")), a.frobenius_norm2(), kTol);
    // X on qubit 0 anticommutes with ZZII only: 0.16 + 0.25 - 0.09.
    EXPECT_NEAR(signed_norm2(a, PauliString<8>("XIII")), 0.16 + 0.25 - 0.09, kTol);
}

}  // namespace
