// Non-Clifford conjugation with a gate or channel on k qubits
// (PauliLocalMap.h): heisenberg_ptm entries for known maps, apply_ptm against
// dense U^dagger A U and sum_K K^dagger A K, against evolve() for a Pauli
// rotation, reproducibility, error cases.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include <Eigen/QR>

#include "DenseReference.h"
#include "KernelConformance.h"
#include "PauLib.h"

using namespace paulib;
using cd = std::complex<double>;

namespace {

constexpr std::size_t kCap = 128;

template <typename T>
class LocalMapTyped : public ::testing::Test {};
using LocalMapTypes = ::testing::Types<double, cd>;
TYPED_TEST_SUITE(LocalMapTyped, LocalMapTypes);

DenseMatrix to_dense_matrix(const dense::Mat& m) {
    const std::size_t d = static_cast<std::size_t>(m.rows());
    DenseMatrix r(d * d);
    for (std::size_t a = 0; a < d; ++a)
        for (std::size_t b = 0; b < d; ++b) r[a * d + b] = m(a, b);
    return r;
}

// The local operator on `qubits` (bit j of its index = qubits[j]) as a
// 2^n x 2^n matrix.
dense::Mat embed(const dense::Mat& local, const std::vector<std::size_t>& qubits, std::size_t n) {
    const std::size_t d = std::size_t(1) << n;
    std::size_t mask = 0;
    for (std::size_t q : qubits) mask |= std::size_t(1) << q;
    auto loc = [&](std::size_t a) {
        std::size_t r = 0;
        for (std::size_t j = 0; j < qubits.size(); ++j) r |= ((a >> qubits[j]) & 1) << j;
        return r;
    };
    dense::Mat m = dense::Mat::Zero(d, d);
    for (std::size_t a = 0; a < d; ++a)
        for (std::size_t b = 0; b < d; ++b)
            if ((a & ~mask) == (b & ~mask)) m(a, b) = local(loc(a), loc(b));
    return m;
}

dense::Mat random_unitary(std::size_t k, unsigned seed) {
    const std::size_t d = std::size_t(1) << k;
    std::mt19937 rng(seed);
    std::normal_distribution<double> g;
    dense::Mat a(d, d);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j) a(i, j) = cd(g(rng), g(rng));
    Eigen::HouseholderQR<dense::Mat> qr(a);
    return qr.householderQ();
}

template <typename T>
PauliSum<kCap, T> hermitian_random_sum(std::size_t terms, std::uint64_t seed, std::size_t n) {
    auto s = conformance::random_sum<kCap, T>(terms, seed, n);
    for (auto& f : s.flags) f &= 2;  // phases +-1: a real T holds every coefficient
    return s;
}

}  // namespace

TEST(LocalMap, PtmOfKnownMaps) {
    // Local index: 0 = I, 1 = X, 2 = Z, 3 = Y.
    const double t = 0.37;
    dense::Mat rz(2, 2);
    rz << std::exp(cd(0, -t / 2)), 0, 0, std::exp(cd(0, t / 2));
    const auto r = heisenberg_ptm({to_dense_matrix(rz)}, 1);
    EXPECT_DOUBLE_EQ(r.at(0, 0), 1.0);
    EXPECT_NEAR(r.at(1, 1), std::cos(t), 1e-14);   // X -> cos X - sin Y
    EXPECT_NEAR(r.at(3, 1), -std::sin(t), 1e-14);
    EXPECT_NEAR(r.at(3, 3), std::cos(t), 1e-14);   // Y -> cos Y + sin X
    EXPECT_NEAR(r.at(1, 3), std::sin(t), 1e-14);
    EXPECT_DOUBLE_EQ(r.at(2, 2), 1.0);
    EXPECT_EQ(r.at(0, 1), 0.0);                    // no rounding entries

    // Amplitude damping: Z -> (1 - g) Z + g I, X -> sqrt(1 - g) X.
    const double g = 0.2;
    dense::Mat k0(2, 2), k1(2, 2);
    k0 << 1, 0, 0, std::sqrt(1 - g);
    k1 << 0, std::sqrt(g), 0, 0;
    const auto a = heisenberg_ptm({to_dense_matrix(k0), to_dense_matrix(k1)}, 1);
    EXPECT_NEAR(a.at(2, 2), 1 - g, 1e-14);
    EXPECT_NEAR(a.at(0, 2), g, 1e-14);
    EXPECT_NEAR(a.at(1, 1), std::sqrt(1 - g), 1e-14);
    EXPECT_NEAR(a.at(3, 3), std::sqrt(1 - g), 1e-14);
    EXPECT_DOUBLE_EQ(a.at(0, 0), 1.0);

    EXPECT_THROW(heisenberg_ptm({to_dense_matrix(rz)}, 2), std::invalid_argument);
    EXPECT_THROW(heisenberg_ptm({}, 5), std::invalid_argument);
}

TYPED_TEST(LocalMapTyped, RandomTwoQubitGateMatchesDense) {
    using T = TypeParam;
    const std::size_t n = 5;
    const std::vector<std::size_t> qubits{3, 1};
    const dense::Mat u = random_unitary(2, 5);
    const auto ptm = heisenberg_ptm({to_dense_matrix(u)}, 2);
    auto s = hermitian_random_sum<T>(40, 81, n);
    s.canonicalize();
    const dense::Mat U = embed(u, qubits, n);
    const dense::Mat ref = U.adjoint() * dense::sum(s) * U;
    apply_ptm(s, qubits, ptm);
    EXPECT_LT(dense::max_abs_diff(dense::sum(s), ref), 1e-12);
    EXPECT_TRUE(s.is_sorted());
}

TYPED_TEST(LocalMapTyped, CczSplitsTargetIntoFourStrings) {
    using T = TypeParam;
    const std::size_t n = 5;
    const std::vector<std::size_t> qubits{0, 2, 4};
    dense::Mat ccz = dense::Mat::Identity(8, 8);
    ccz(7, 7) = -1;
    const auto ptm = heisenberg_ptm({to_dense_matrix(ccz)}, 3);

    PauliSum<kCap, T> x(n, "IIXII", T(1));
    apply_ptm(x, qubits, ptm);
    x.canonicalize();
    EXPECT_EQ(x.get_length(), 4u);  // X_2 (I + Z_0 + Z_4 - Z_0 Z_4) / 2
    EXPECT_NEAR(std::abs(x.lookup("IIXII")), 0.5, 1e-14);
    EXPECT_NEAR(std::real(x.lookup("ZIXIZ")), -0.5, 1e-14);

    PauliSum<kCap, T> z(n, "ZIZIZ", T(1));  // diagonal strings commute with CCZ
    apply_ptm(z, qubits, ptm);
    EXPECT_EQ(z.get_length(), 1u);

    auto s = hermitian_random_sum<T>(50, 82, n);
    const dense::Mat U = embed(ccz, qubits, n);
    const dense::Mat ref = U.adjoint() * dense::sum(s) * U;
    apply_ptm(s, qubits, ptm);
    EXPECT_LT(dense::max_abs_diff(dense::sum(s), ref), 1e-12);
}

TYPED_TEST(LocalMapTyped, AmplitudeDampingMatchesKraus) {
    using T = TypeParam;
    const std::size_t n = 4;
    const double g = 0.3;
    dense::Mat k0(2, 2), k1(2, 2);
    k0 << 1, 0, 0, std::sqrt(1 - g);
    k1 << 0, std::sqrt(g), 0, 0;
    const auto ptm = heisenberg_ptm({to_dense_matrix(k0), to_dense_matrix(k1)}, 1);
    auto s = hermitian_random_sum<T>(30, 83, n);
    const dense::Mat K0 = embed(k0, {2}, n), K1 = embed(k1, {2}, n);
    const dense::Mat a = dense::sum(s);
    const dense::Mat ref = K0.adjoint() * a * K0 + K1.adjoint() * a * K1;
    apply_ptm(s, {2}, ptm);
    EXPECT_LT(dense::max_abs_diff(dense::sum(s), ref), 1e-12);
}

// Complete dephasing (Kraus I/sqrt2, Z/sqrt2): X and Y on the qubit vanish,
// so their terms are removed (diagonal entry 0).
TYPED_TEST(LocalMapTyped, DephasingRemovesTerms) {
    using T = TypeParam;
    const std::size_t n = 4;
    const double h = 1 / std::sqrt(2.0);
    const auto ptm = heisenberg_ptm({DenseMatrix{h, 0, 0, h}, DenseMatrix{h, 0, 0, -h}}, 1);
    auto s = hermitian_random_sum<T>(80, 86, n);
    s.canonicalize();
    const dense::Mat K0 = embed(dense::Mat::Identity(2, 2) * h, {1}, n);
    dense::Mat z(2, 2);
    z << h, 0, 0, -h;
    const dense::Mat K1 = embed(z, {1}, n);
    const dense::Mat a = dense::sum(s);
    const dense::Mat ref = K0.adjoint() * a * K0 + K1.adjoint() * a * K1;
    apply_ptm(s, {1}, ptm);
    EXPECT_LT(dense::max_abs_diff(dense::sum(s), ref), 1e-12);
    for (std::size_t i = 0; i < s.get_length(); ++i) EXPECT_EQ((s.x_data[0][i] >> 1) & 1, 0u);
    EXPECT_TRUE(s.is_sorted());
}

TYPED_TEST(LocalMapTyped, PauliRotationMatchesEvolve) {
    using T = TypeParam;
    const std::size_t n = 70;
    const double theta = 0.81;
    // exp(-i theta/2 Y_0 X_1) on qubits {66, 2}: local digit 0 = qubit 66.
    PauliString<2> p("YX");
    const auto ptm = heisenberg_ptm({to_dense_matrix(dense::rotation(p, 2, theta))}, 2);
    auto a = hermitian_random_sum<T>(400, 84, n);
    a.canonicalize();
    auto b = a;
    apply_ptm(a, {66, 2}, ptm);
    std::string gen(n, 'I');
    gen[66] = 'Y';
    gen[2] = 'X';
    b.evolve(PauliSum<kCap, T>(n, gen, T(theta)));
    a.canonicalize();
    b.canonicalize();
    ASSERT_EQ(a.get_length(), b.get_length());
    EXPECT_EQ(a.x_data, b.x_data);
    EXPECT_EQ(a.z_data, b.z_data);
    for (std::size_t i = 0; i < a.get_length(); ++i)  // coefficients up to 1e3
        EXPECT_LE(std::abs(a.value[i] - b.value[i]), 1e-14 * std::max(1.0, std::abs(b.value[i])));
}

TEST(LocalMap, ReproducibleAndErrors) {
    conformance::ScopedPolicy policy(1, 7);
    const dense::Mat u = random_unitary(2, 9);
    const auto ptm = heisenberg_ptm({to_dense_matrix(u)}, 2);
    for (const std::size_t n : {3, 64, 130}) {
        SCOPED_TRACE("n = " + std::to_string(n));
        const auto s = conformance::random_sum<kCap + 64, cd>(1000, 85 + n, n);
        conformance::expect_thread_invariant([&] { auto c = s; apply_ptm(c, {n - 1, 0}, ptm); return c; },
                                             "apply_ptm");
    }
    PauliSum<kCap, double> s(4, "XXXX", 1.0);
    EXPECT_THROW(apply_ptm(s, {0}, ptm), std::invalid_argument);
    EXPECT_THROW(apply_ptm(s, {1, 1}, ptm), std::invalid_argument);
    EXPECT_THROW(apply_ptm(s, {1, 4}, ptm), std::invalid_argument);
}
