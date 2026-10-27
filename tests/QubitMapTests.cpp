// Qubit maps (QMP, PauliQubitMap.h): shift_qubits and shift_lattice against
// remap_qubits with the same permutation, restrict_qubits letter by letter,
// fix_qubits against the dense operator restricted to the eigenvector, and
// the error cases.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include "DenseReference.h"
#include "KernelConformance.h"
#include "PauLib.h"

using namespace paulib;
using cd = std::complex<double>;

namespace {

constexpr std::size_t kCap = 192;

template <std::size_t N, typename T>
void expect_same_terms(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    ASSERT_EQ(a.get_length(), b.get_length());
    ASSERT_EQ(a.num_qubits(), b.num_qubits());
    EXPECT_EQ(a.x_data, b.x_data);
    EXPECT_EQ(a.z_data, b.z_data);
    EXPECT_EQ(a.flags, b.flags);
    EXPECT_TRUE(conformance::bit_equal(a.value, b.value));
}

template <std::size_t N>
unsigned letter(const PauliString<N>& p, std::size_t q) { return local_pauli(p, q / 64, q % 64); }

// <v|_q M |v>_q: the 2^(n-1) x 2^(n-1) block of M for qubit q in state v.
dense::Mat contract(const dense::Mat& m, std::size_t q, const Eigen::Vector2cd& v) {
    const std::size_t d = static_cast<std::size_t>(m.rows()) / 2;
    auto ins = [q](std::size_t a, std::size_t s) {
        const std::size_t lo = a & ((std::size_t(1) << q) - 1);
        return lo | (s << q) | ((a >> q) << (q + 1));
    };
    dense::Mat r = dense::Mat::Zero(d, d);
    for (std::size_t a = 0; a < d; ++a)
        for (std::size_t b = 0; b < d; ++b)
            for (std::size_t s = 0; s < 2; ++s)
                for (std::size_t t = 0; t < 2; ++t)
                    r(a, b) += std::conj(v(s)) * v(t) * m(ins(a, s), ins(b, t));
    return r;
}

}  // namespace

TEST(QubitMap, ShiftQubitsSmall) {
    const PauliSum<kCap, double> s(5, "XYZII", 2.0);
    EXPECT_EQ(shift_qubits(s, 1).get_pauli_string(0).to_string(), "IXYZI");
    EXPECT_EQ(shift_qubits(s, -1).get_pauli_string(0).to_string(), "YZIIX");
    EXPECT_EQ(shift_qubits(s, 7).get_pauli_string(0).to_string(), "IIXYZ");
    EXPECT_EQ(shift_qubits(s, 5).get_pauli_string(0).to_string(), "XYZII");
    EXPECT_EQ(shift_qubits(s, 1).value[0], 2.0);
}

TEST(QubitMap, ShiftQubitsMatchesRemap) {
    for (const std::size_t n : conformance::kQubitCounts) {
        const auto s = conformance::random_sum<kCap, cd>(300, 31 + n, n);
        for (const std::ptrdiff_t k : {1, -1, 2, 63, 64, 65, -70, 129}) {
            SCOPED_TRACE("n = " + std::to_string(n) + ", k = " + std::to_string(k));
            std::vector<std::size_t> map(n);
            const std::ptrdiff_t m = static_cast<std::ptrdiff_t>(n);
            for (std::size_t q = 0; q < n; ++q)
                map[q] = static_cast<std::size_t>(((static_cast<std::ptrdiff_t>(q) + k) % m + m) % m);
            expect_same_terms(shift_qubits(s, k), remap_qubits(s, n, map));
        }
    }
}

TEST(QubitMap, ShiftLatticeMatchesRemap) {
    const std::vector<std::pair<std::size_t, std::size_t>> shapes{{4, 3}, {8, 8}, {13, 10}, {65, 2}, {1, 7}, {7, 1}};
    for (const auto& [Lx, Ly] : shapes) {
        const std::size_t n = Lx * Ly;
        const auto s = conformance::random_sum<kCap, double>(200, 41 + n, n);
        for (const auto& [dx, dy] : std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>>{
                 {1, 0}, {0, 1}, {2, 1}, {-1, -2}, {5, 3}, {0, 0}}) {
            SCOPED_TRACE(std::to_string(Lx) + "x" + std::to_string(Ly) + " by (" + std::to_string(dx) + ", " +
                         std::to_string(dy) + ")");
            std::vector<std::size_t> map(n);
            auto wrap = [](std::ptrdiff_t v, std::size_t L) {
                const std::ptrdiff_t l = static_cast<std::ptrdiff_t>(L);
                return static_cast<std::size_t>((v % l + l) % l);
            };
            for (std::size_t y = 0; y < Ly; ++y)
                for (std::size_t x = 0; x < Lx; ++x)
                    map[x + Lx * y] = wrap(static_cast<std::ptrdiff_t>(x) + dx, Lx) +
                                      Lx * wrap(static_cast<std::ptrdiff_t>(y) + dy, Ly);
            expect_same_terms(shift_lattice(s, Lx, Ly, dx, dy), remap_qubits(s, n, map));
        }
    }
    const PauliSum<kCap, double> s(6, "XIIIII", 1.0);
    EXPECT_THROW(shift_lattice(s, 4, 2, 1, 0), std::invalid_argument);
}

TEST(QubitMap, RestrictQubitsKeepsLetters) {
    for (const std::size_t n : {std::size_t{5}, std::size_t{70}, std::size_t{130}}) {
        const auto s = conformance::random_sum<kCap, cd>(200, 51 + n, n);
        const std::vector<std::size_t> qubits{n - 1, 0, n / 2, 3};
        const auto r = restrict_qubits(s, qubits);
        ASSERT_EQ(r.num_qubits(), qubits.size());
        ASSERT_EQ(r.get_length(), s.get_length());
        for (std::size_t i = 0; i < s.get_length(); ++i) {
            const auto a = s.get_pauli_string(i), b = r.get_pauli_string(i);
            for (std::size_t j = 0; j < qubits.size(); ++j) EXPECT_EQ(letter(b, j), letter(a, qubits[j]));
            for (std::size_t q = qubits.size(); q < kCap; ++q) EXPECT_EQ(letter(b, q), 0u);
            EXPECT_EQ(r.flags[i], s.flags[i]);
            EXPECT_EQ(r.value[i], s.value[i]);
        }
    }
    const PauliSum<kCap, double> s(4, "XYZX", 1.0);
    EXPECT_THROW(restrict_qubits(s, {0, 4}), std::invalid_argument);
    EXPECT_THROW(restrict_qubits(s, {1, 1}), std::invalid_argument);
}

TEST(QubitMap, FixQubitsMatchesDenseEigenvector) {
    // Terms act on qubit 1 with I or X and on qubit 3 with I or Y.
    using T = cd;
    const std::size_t n = 5;
    auto s = conformance::random_sum<kCap, T>(60, 61, n);
    for (std::size_t i = 0; i < s.get_length(); ++i) {
        s.z_data[0][i] &= ~(std::size_t{1} << 1);      // qubit 1: I or X
        const bool on3 = ((s.x_data[0][i] | s.z_data[0][i]) >> 3) & 1;
        if (on3) {                                     // qubit 3: Y
            s.x_data[0][i] |= std::size_t{1} << 3;
            s.z_data[0][i] |= std::size_t{1} << 3;
        }
    }
    const double r = 1 / std::sqrt(2.0);
    for (const int e1 : {1, -1})
        for (const int e3 : {1, -1}) {
            SCOPED_TRACE("eigenvalues " + std::to_string(e1) + " " + std::to_string(e3));
            const auto f = fix_qubits(s, {{3, 'Y', e3}, {1, 'X', e1}});
            ASSERT_EQ(f.num_qubits(), n - 2);
            const Eigen::Vector2cd vx(r, e1 * r);                  // X|v> = e1 |v>
            const Eigen::Vector2cd vy(r, cd(0, e3) * r);           // Y|v> = e3 |v>
            const dense::Mat ref = contract(contract(dense::sum(s), 3, vy), 1, vx);
            EXPECT_LT(dense::max_abs_diff(dense::sum(f), ref), 1e-12);
        }
    EXPECT_THROW(fix_qubits(s, {{1, 'Z', 1}}), std::invalid_argument);   // X on qubit 1
    EXPECT_THROW(fix_qubits(s, {{1, 'X', 2}}), std::invalid_argument);
    EXPECT_THROW(fix_qubits(s, {{1, 'X', 1}, {1, 'X', 1}}), std::invalid_argument);
    EXPECT_THROW(fix_qubits(s, {{5, 'X', 1}}), std::invalid_argument);
    EXPECT_THROW(fix_qubits(s, {{1, 'Q', 1}}), std::invalid_argument);
}

TEST(QubitMap, KernelsBitIdenticalForThreadCounts) {
    conformance::ScopedPolicy policy(1, 7);
    for (const std::size_t n : {3, 64, 130}) {
        SCOPED_TRACE("n = " + std::to_string(n));
        const auto s = conformance::random_sum<kCap, cd>(1000, 71 + n, n);
        auto zonly = s;  // I or Z on qubit 0 for fix_qubits
        for (auto& w : zonly.x_data[0]) w &= ~std::size_t{1};
        conformance::expect_thread_invariant([&] { return shift_qubits(s, 5); }, "shift_qubits");
        conformance::expect_thread_invariant([&] { return shift_lattice(s, n, 1, -2, 0); }, "shift_lattice");
        conformance::expect_thread_invariant([&] { return restrict_qubits(s, {n - 1, 0}); }, "restrict_qubits");
        conformance::expect_thread_invariant([&] { return fix_qubits(zonly, {{0, 'Z', -1}}); }, "fix_qubits");
    }
}
