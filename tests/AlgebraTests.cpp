// Tests for the second round of core operations (October 2026): the
// skeletons transform_terms, expand_terms, for_each_pair and merge_reduce,
// merge-combine (add_combine), adjoint, remap_qubits, tensor_product, the
// imaginary-time step, the Bravyi-Kitaev mapping and qubit tapering.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>

#include "DenseReference.h"
#include "KernelConformance.h"
#include "PauLib.h"

using namespace paulib;
using cd = std::complex<double>;

namespace {

constexpr std::size_t kCap = 192;

template <typename T>
class AlgebraTyped : public ::testing::Test {};
using AlgebraTypes = ::testing::Types<double, cd>;
TYPED_TEST_SUITE(AlgebraTyped, AlgebraTypes);

template <typename T>
double mag(T v) { return std::abs(v); }

// append + stable sort by (x, z, flags) + left fold of equal terms: the
// definition that add_combine must match bit for bit.
template <std::size_t N, typename T>
PauliSum<N, T> reference_combine(PauliSum<N, T> a, const PauliSum<N, T>& b, T c) {
    if (c == T{1}) add(a, b);
    else append_scaled(a, c, b);
    std::vector<std::size_t> idx(a.get_length());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    auto key = [&](std::size_t i) {
        std::vector<std::size_t> k;
        for (const auto& w : a.x_data) k.push_back(w[i]);
        for (const auto& w : a.z_data) k.push_back(w[i]);
        k.push_back(a.flags[i]);
        return k;
    };
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t i, std::size_t j) { return key(i) < key(j); });
    PauliSum<N, T> out;
    out.num_qubits_ = a.num_qubits_;
    for (std::size_t r = 0; r < idx.size(); ++r) {
        if (r > 0 && key(idx[r]) == key(idx[r - 1])) {
            out.value.back() += a.value[idx[r]];
            continue;
        }
        out.push_back(a.get_pauli_string(idx[r]), a.value[idx[r]]);
    }
    return out;
}

std::vector<double> eigenvalues(const dense::Mat& m) {
    Eigen::SelfAdjointEigenSolver<dense::Mat> es(m);
    std::vector<double> v(es.eigenvalues().data(), es.eigenvalues().data() + es.eigenvalues().size());
    std::sort(v.begin(), v.end());
    return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// Reproducibility of the new kernels (D5): 1, 3 and 8 threads, 1000 terms in
// chunks of 7, threads from one term on.
// ---------------------------------------------------------------------------

TYPED_TEST(AlgebraTyped, KernelsBitIdenticalForThreadCounts) {
    using T = TypeParam;
    conformance::ScopedPolicy policy(1, 7);
    for (const std::size_t n : {3, 64, 130}) {
        SCOPED_TRACE("n = " + std::to_string(n));
        const auto s = conformance::random_sum<kCap, T>(1000, 21 + n, n);
        auto sorted = s;
        sorted.canonicalize();
        auto other = conformance::random_sum<kCap, T>(700, 22 + n, n);
        other.canonicalize();
        PauliSum<kCap, T> rot = random_pauli_sum<kCap, T>(6, 23 + n, n);
        for (std::size_t j = 0; j < rot.get_length(); ++j) {
            rot.flags[j] = 0;
            rot.value[j] = T(0.3 + 0.2 * j);
        }
        rot.value[1] = T(M_PI / 2);  // Clifford angle: transform_terms path
        std::vector<std::size_t> map(n);
        std::iota(map.rbegin(), map.rend(), std::size_t{0});
        const auto small = conformance::random_sum<kCap, T>(40, 24 + n, std::min<std::size_t>(n, 60));

        conformance::expect_thread_invariant([&] { auto c = s; c.evolve(rot); return c; }, "evolve");
        conformance::expect_thread_invariant([&] { auto c = s; c.evolve_imaginary(rot); return c; }, "evolve_imaginary");
        conformance::expect_thread_invariant([&] { auto c = s; c.canonicalize(); return c; }, "canonicalize");
        conformance::expect_thread_invariant([&] { auto c = s; c += other; return c; }, "add_combine");
        conformance::expect_thread_invariant([&] { return adjoint(s); }, "adjoint");
        conformance::expect_thread_invariant([&] { return remap_qubits(s, n, map); }, "remap_qubits");
        conformance::expect_thread_invariant([&] { return tensor_product(small, small); }, "tensor_product");
        conformance::expect_thread_invariant([&] { return inner_product(sorted, other); }, "inner_product (merge)");
        conformance::expect_thread_invariant([&] { return trace_product(sorted, other); }, "trace_product (merge)");
    }
}

// ---------------------------------------------------------------------------
// Merge-combine (P1)
// ---------------------------------------------------------------------------

TYPED_TEST(AlgebraTyped, AddCombineMatchesAppendAndSortBitForBit) {
    using T = TypeParam;
    for (const bool lhs_sorted : {false, true})
        for (const bool rhs_sorted : {false, true})
            for (const std::size_t n : {2, 3, 70}) {
                SCOPED_TRACE(std::to_string(lhs_sorted) + std::to_string(rhs_sorted) + " n=" + std::to_string(n));
                // Few qubits: many duplicates, inside each operand and across.
                auto a = conformance::random_sum<kCap, T>(300, 31 + n, n);
                auto b = conformance::random_sum<kCap, T>(200, 32 + n, n);
                if (lhs_sorted) a.sort_and_combine();
                if (rhs_sorted) b.sort_and_combine();
                EXPECT_EQ(a.is_sorted(), lhs_sorted);
                auto plus = a;
                plus += b;
                EXPECT_TRUE(conformance::bit_equal(plus, reference_combine(a, b, T{1})));
                auto minus = a;
                minus -= b;
                EXPECT_TRUE(conformance::bit_equal(minus, reference_combine(a, b, T{-1})));
                EXPECT_TRUE(plus.is_sorted());
            }
}

TYPED_TEST(AlgebraTyped, AddCombineLargeAndSelf) {
    using T = TypeParam;
    // Above the parallel threshold of the collate (2^14 terms).
    auto a = conformance::random_sum<kCap, T>(20000, 41, 9);
    a.sort_and_combine();
    const auto b = conformance::random_sum<kCap, T>(30000, 42, 9);
    auto c = a;
    c += b;
    EXPECT_TRUE(conformance::bit_equal(c, reference_combine(a, b, T{1})));
    auto twice = a;
    twice += twice;
    EXPECT_TRUE(conformance::bit_equal(twice, reference_combine(a, a, T{1})));
}

TYPED_TEST(AlgebraTyped, MergeProductsMatchHashAndDense) {
    using T = TypeParam;
    const std::size_t n = 4;
    auto a = random_pauli_sum<8, T>(30, 51, n);
    auto b = random_pauli_sum<8, T>(25, 52, n);
    for (std::size_t j = 0; j < b.get_length(); j += 3) b.flags[j] ^= 1;  // imaginary phases
    std::mt19937_64 rng(53);
    for (auto& v : a.value) v = T(std::uniform_real_distribution<double>(-1, 1)(rng));
    for (auto& v : b.value) v = T(std::uniform_real_distribution<double>(-1, 1)(rng));
    // The unsorted a takes the hash path, its sorted copy the merge.
    auto as = a, bs = b;
    as.sort_and_combine();
    bs.sort_and_combine();
    ASSERT_FALSE(a.is_sorted());
    ASSERT_TRUE(as.is_sorted() && bs.is_sorted());
    const dense::Mat da = dense::sum(a), db = dense::sum(b);
    const double d = std::pow(2.0, n);
    const cd want_inner = (da.adjoint() * db).trace() / d;
    const cd want_trace = (da * db).trace();
    for (const auto* x : {&a, &as}) {
        const T inner = inner_product(*x, bs);  // b and bs: same operator
        const T trace = trace_product(*x, bs);
        if constexpr (is_complex_v<T>) {
            EXPECT_NEAR(std::abs(inner - want_inner), 0.0, 1e-12);
            EXPECT_NEAR(std::abs(trace - want_trace), 0.0, 1e-11);
        } else {
            EXPECT_NEAR(inner, want_inner.real(), 1e-12);
            EXPECT_NEAR(trace, want_trace.real(), 1e-11);
        }
    }
}

// ---------------------------------------------------------------------------
// LIN / MUL structure: adjoint (L5), remap_qubits and tensor_product (L4)
// ---------------------------------------------------------------------------

TYPED_TEST(AlgebraTyped, AdjointIsMatrixAdjoint) {
    using T = TypeParam;
    auto a = random_pauli_sum<8, T>(20, 61, 3);
    std::mt19937_64 rng(62);
    for (auto& v : a.value) {
        if constexpr (is_complex_v<T>) v = T(std::normal_distribution<double>()(rng), std::normal_distribution<double>()(rng));
        else v = std::normal_distribution<double>()(rng);
    }
    EXPECT_LT(dense::max_abs_diff(dense::sum(adjoint(a)), dense::sum(a).adjoint()), 1e-12);
    auto h = adjoint(a) + a;  // Hermitian
    EXPECT_LT(dense::max_abs_diff(dense::sum(adjoint(h)), dense::sum(h)), 1e-12);
}

TEST(Algebra, RemapQubitsMovesLetters) {
    const std::size_t n = 5;
    auto s = random_pauli_sum<128, double>(40, 71, n);
    for (std::size_t j = 0; j < s.get_length(); ++j) {  // qubit 2 is removed: make it I
        s.x_data[0][j] &= ~(std::size_t{1} << 2);
        s.z_data[0][j] &= ~(std::size_t{1} << 2);
    }
    const std::vector<std::size_t> map{100, 0, kDropQubit, 65, 7};
    const auto out = remap_qubits(s, 101, map);
    ASSERT_EQ(out.get_length(), s.get_length());
    EXPECT_EQ(out.num_qubits(), 101u);
    for (std::size_t j = 0; j < s.get_length(); ++j) {
        const std::string in = s.get_pauli_string(j).to_string();
        std::string want(101, 'I');
        const std::size_t off = in.size() - n;  // sign or phase prefix
        for (std::size_t q = 0; q < n; ++q)
            if (map[q] != kDropQubit) want[map[q]] = in[off + q];
        PauliString<128> got = out.get_pauli_string(j);
        got.num_qubits_ = 101;
        EXPECT_EQ(got.to_string(), in.substr(0, off) + want);
        EXPECT_EQ(out.value[j], s.value[j]);
    }
    EXPECT_THROW(remap_qubits(s, 5, std::vector<std::size_t>{0, 1, 2, 3, 3}), std::invalid_argument);
    EXPECT_THROW(remap_qubits(s, 4, std::vector<std::size_t>{0, 1, 2, 3, 4}), std::invalid_argument);
    EXPECT_THROW(remap_qubits(s, 4, std::vector<std::size_t>{kDropQubit, 0, 1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(remap_qubits(s, 4, std::vector<std::size_t>{0, 1, 2}), std::invalid_argument);
}

TYPED_TEST(AlgebraTyped, TensorProductConcatenatesStrings) {
    using T = TypeParam;
    const auto a = conformance::random_sum<kCap, T>(7, 81, 3);
    const auto b = conformance::random_sum<kCap, T>(5, 82, 62);  // crosses a word boundary
    const auto t = tensor_product(a, b);
    ASSERT_EQ(t.get_length(), 35u);
    EXPECT_EQ(t.num_qubits(), 65u);
    auto letters = [](const PauliString<kCap>& p, std::size_t n) {
        PauliString<kCap> q = p;
        q.flags = 0;
        q.num_qubits_ = static_cast<uint32_t>(n);
        return q.to_string();
    };
    for (std::size_t i = 0; i < 7; ++i)
        for (std::size_t j = 0; j < 5; ++j) {
            const std::size_t r = i * 5 + j;
            EXPECT_EQ(letters(t.get_pauli_string(r), 65),
                      letters(a.get_pauli_string(i), 3) + letters(b.get_pauli_string(j), 62));
            EXPECT_EQ(t.flags[r], (a.flags[i] + b.flags[j]) & 3);
            EXPECT_LE(std::abs(t.value[r] - a.value[i] * b.value[j]), 1e-15 * std::abs(t.value[r]));
        }
    // Dense: A (x) B with A on the low qubits.
    const auto x = random_pauli_sum<8, T>(4, 83, 2), y = random_pauli_sum<8, T>(3, 84, 2);
    std::vector<std::size_t> lo{0, 1}, hi{2, 3};
    EXPECT_LT(dense::max_abs_diff(dense::sum(tensor_product(x, y)),
                                  dense::sum(remap_qubits(x, 4, lo)) * dense::sum(remap_qubits(y, 4, hi))),
              1e-12);
}

// ---------------------------------------------------------------------------
// ROT, imaginary time (K5)
// ---------------------------------------------------------------------------

TYPED_TEST(AlgebraTyped, EvolveImaginaryMatchesDense) {
    using T = TypeParam;
    const std::size_t n = 4;
    auto o = random_pauli_sum<8, T>(25, 91, n);
    for (std::size_t j = 0; j < o.get_length(); ++j) o.flags[j] &= 2;  // Hermitian strings
    PauliSum<8, T> gens;
    gens.num_qubits_ = n;
    for (const char* g : {"XZIY", "-ZZII", "IIIX", "YIYI"}) gens.push_back(PauliString<8>(g), T(0.37));
    gens.value[2] = T(-1.3);
    dense::Mat want = dense::sum(o);
    for (std::size_t l = 0; l < gens.get_length(); ++l) {
        const double tau = std::real(gens.value[l]);
        const dense::Mat p = dense::pauli(gens.get_pauli_string(l), n);
        const dense::Mat e = std::cosh(tau / 2) * dense::Mat::Identity(16, 16) - std::sinh(tau / 2) * p;
        want = e * want * e;
    }
    o.evolve_imaginary(gens);
    EXPECT_LT(dense::max_abs_diff(dense::sum(o), want), 1e-12 * std::max(1.0, want.norm()));

    PauliSum<8, T> bad(n, "iXIII", T(0.1));
    EXPECT_THROW(o.evolve_imaginary(bad), std::invalid_argument);
}

// Commuting terms grow, anticommuting ones stay: the opposite split of evolve.
TEST(Algebra, EvolveImaginaryCaseSplit) {
    PauliSum<8, double> o(std::vector<std::pair<std::string, double>>{{"ZI", 1.0}, {"XI", 1.0}});
    o.evolve_imaginary(PauliSum<8, double>(2, "ZI", 0.5));
    EXPECT_NEAR(o.lookup("XI"), 1.0, 1e-15);
    EXPECT_NEAR(o.lookup("ZI"), std::cosh(0.5), 1e-15);
    EXPECT_NEAR(o.lookup("II"), -std::sinh(0.5), 1e-15);
}

// ---------------------------------------------------------------------------
// GEN: Bravyi-Kitaev (L6)
// ---------------------------------------------------------------------------

// n = 4, derived by hand from the update, parity and occupation sets
// (a_j^dagger = (gamma_2j - i gamma_2j+1) / 2): a0: X3 X1 X0, X3 X1 Y0;
// a1: X3 X1 Z0, X3 Y1; a2: X3 X2 Z1, X3 Y2 Z1; a3: X3 Z2 Z1, Y3 (qubit 0
// first in our strings). The n = 4 example of Seeley, Richard and Love
// (arXiv:1208.5986) should give the same; not yet checked against the paper.
TEST(BravyiKitaev, FourModesMatchLiterature) {
    const std::vector<std::string> want{"XXIX", "YXIX", "ZXIX", "IYIX", "IZXX", "IZYX", "IZZX", "IIIY"};
    for (std::size_t j = 0; j < 8; ++j) EXPECT_EQ(bk_majorana<8>(4, j).to_string(), want[j]) << j;
}

TEST(BravyiKitaev, CanonicalAnticommutationRelations) {
    for (std::size_t n : {1, 2, 5, 8, 13}) {
        std::size_t max_weight = 0;
        for (std::size_t i = 0; i < 2 * n; ++i) {
            const auto gi = bk_majorana<64>(n, i);
            EXPECT_EQ(gi.flags & 1, 0) << "gamma_" << i << " not Hermitian";
            max_weight = std::max(max_weight, weight(gi));
            for (std::size_t j = 0; j < i; ++j)
                EXPECT_TRUE(gi.anti_commute(bk_majorana<64>(n, j))) << n << ": " << i << "," << j;
        }
        // Weight O(log n): at most 2 ceil(log2(n + 1)) + 1.
        EXPECT_LE(max_weight, 2 * static_cast<std::size_t>(std::ceil(std::log2(n + 1.0))) + 1) << n;
        for (std::size_t m = 0; m < n; ++m)
            for (std::size_t l = 0; l < n; ++l) {
                auto ac = bk_annihilation<64, double>(n, m) * bk_creation<64, double>(n, l) +
                          bk_creation<64, double>(n, l) * bk_annihilation<64, double>(n, m);
                ac.canonicalize();
                if (m == l) {
                    ASSERT_EQ(ac.get_length(), 1u);
                    EXPECT_NEAR(ac.lookup(std::string(n, 'I')), 1.0, 1e-15);
                } else {
                    EXPECT_EQ(ac.get_length(), 0u);
                }
            }
    }
}

// The same fermionic Hamiltonian has the same spectrum under both mappings.
TEST(BravyiKitaev, SpectrumMatchesJordanWigner) {
    const std::size_t n = 5;
    std::mt19937_64 rng(101);
    std::normal_distribution<double> g;
    using S = PauliSum<8, double>;
    auto build = [&](auto ann, auto cre, unsigned seed) {
        std::mt19937_64 r(seed);
        std::normal_distribution<double> h;
        S out(n, std::string(n, 'I'), 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                const double t = h(r);
                out += t * (cre(i) * ann(j) + cre(j) * ann(i));
            }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) out += h(r) * (cre(i) * ann(i) * cre(j) * ann(j));
        out.canonicalize();
        return out;
    };
    const S jw = build([&](std::size_t m) { return jw_annihilation<8, double>(n, m); },
                       [&](std::size_t m) { return jw_creation<8, double>(n, m); }, 102);
    const S bk = build([&](std::size_t m) { return bk_annihilation<8, double>(n, m); },
                       [&](std::size_t m) { return bk_creation<8, double>(n, m); }, 102);
    const auto ej = eigenvalues(dense::sum(jw)), eb = eigenvalues(dense::sum(bk));
    ASSERT_EQ(ej.size(), eb.size());
    for (std::size_t k = 0; k < ej.size(); ++k) EXPECT_NEAR(ej[k], eb[k], 1e-10);
    (void)g(rng);
}

// ---------------------------------------------------------------------------
// GF2 + CLF: qubit tapering (L8)
// ---------------------------------------------------------------------------

namespace {

// Spectrum of h restricted to the sector s_i tau_i = +1.
std::vector<double> sector_spectrum(const dense::Mat& h, const std::vector<PauliString<8>>& taus,
                                    const std::vector<int>& sector, std::size_t n) {
    const std::size_t d = std::size_t{1} << n;
    dense::Mat proj = dense::Mat::Identity(d, d);
    for (std::size_t i = 0; i < taus.size(); ++i)
        proj = proj * (dense::Mat::Identity(d, d) + double(sector[i]) * dense::pauli(taus[i], n)) / 2.0;
    Eigen::SelfAdjointEigenSolver<dense::Mat> es(proj);
    std::vector<int> cols;
    for (int c = 0; c < es.eigenvalues().size(); ++c)
        if (es.eigenvalues()(c) > 0.5) cols.push_back(c);
    dense::Mat basis(d, cols.size());
    for (std::size_t c = 0; c < cols.size(); ++c) basis.col(c) = es.eigenvectors().col(cols[c]);
    return eigenvalues(basis.adjoint() * h * basis);
}

}  // namespace

TEST(Tapering, SectorSpectraMatchDense) {
    const std::size_t n = 6;
    const std::vector<PauliString<8>> taus{PauliString<8>("ZZZIII"), PauliString<8>("IIIXXI"),
                                           PauliString<8>("IYYIIZ")};
    // Random Hermitian H that commutes with every tau.
    PauliSum<8, double> h;
    h.num_qubits_ = n;
    std::mt19937_64 rng(111);
    std::normal_distribution<double> g;
    while (h.get_length() < 40) {
        PauliString<8> p = random_pauli_string<8>(n, rng);
        bool ok = true;
        for (const auto& t : taus) ok = ok && p.commute(t);
        if (ok) h.push_back(p, g(rng));
    }
    h.canonicalize();
    const dense::Mat dh = dense::sum(h);
    for (int s = 0; s < 8; ++s) {
        const std::vector<int> sector{(s & 1) ? -1 : 1, (s & 2) ? -1 : 1, (s & 4) ? -1 : 1};
        const Z2Tapering<8> tap(taus, sector, n);
        ASSERT_EQ(tap.num_qubits_out(), 3u);
        const auto tapered = tap.apply(h);
        EXPECT_EQ(tapered.num_qubits(), 3u);
        const auto want = sector_spectrum(dh, taus, sector, n);
        const auto got = eigenvalues(dense::sum(tapered));
        ASSERT_EQ(got.size(), want.size());
        for (std::size_t k = 0; k < got.size(); ++k) EXPECT_NEAR(got[k], want[k], 1e-10) << "sector " << s;
    }
    EXPECT_THROW(Z2Tapering<8>({PauliString<8>("ZZIIII"), PauliString<8>("XIIIII")}, {1, 1}, n),
                 std::invalid_argument);  // anticommuting
    EXPECT_THROW(Z2Tapering<8>({PauliString<8>("ZZIIII"), PauliString<8>("-ZZIIII")}, {1, 1}, n),
                 std::invalid_argument);  // dependent
    const Z2Tapering<8> tap(taus, {1, 1, 1}, n);
    EXPECT_THROW(tap.apply(PauliSum<8, double>(n, "XIIIII", 1.0)), std::invalid_argument);
}

// Transverse-field Ising chain: the only Z2 symmetry is the parity X...X.
TEST(Tapering, IsingParity) {
    const std::size_t n = 6;
    auto h = two_site_sum<8, double>(n, chain_edges(n, false), "ZZ", 1.0) + one_site_sum<8, double>(n, "X", 0.7);
    const auto sym = z2_symmetries(h);
    ASSERT_EQ(sym.size(), 1u);
    EXPECT_EQ(sym[0].to_string(), "XXXXXX");
    const dense::Mat dh = dense::sum(h);
    for (int s : {1, -1}) {
        const auto tapered = Z2Tapering<8>(sym, {s}, n).apply(h);
        EXPECT_EQ(tapered.num_qubits(), n - 1);
        const auto want = sector_spectrum(dh, sym, {s}, n);
        const auto got = eigenvalues(dense::sum(tapered));
        for (std::size_t k = 0; k < got.size(); ++k) EXPECT_NEAR(got[k], want[k], 1e-10);
    }
}
