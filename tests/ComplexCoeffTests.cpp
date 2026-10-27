#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <random>

#include "DenseReference.h"
#include "PauLib.h"

using namespace paulib;

namespace {

constexpr std::size_t kN = 64;
constexpr std::size_t kQubits = 4;
constexpr double kTol = 1e-10;
using cd = std::complex<double>;

// Random sum with distinct coefficients. hermitian = true restricts to real
// coefficients and phases +-1 (a Hermitian operator); otherwise, for complex T,
// coefficients and phases are arbitrary.
template <typename T>
PauliSum<kN, T> random_sum(std::size_t terms, std::uint64_t seed, bool hermitian) {
    PauliSum<kN, T> s = random_pauli_sum<kN, T>(terms, seed, kQubits);
    std::mt19937_64 rng(seed + 1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t i = 0; i < s.get_length(); ++i) {
        if (hermitian || !is_complex_v<T>) {
            s.flags[i] &= 0x02;  // drop the imaginary bit
            s.value[i] = T(u(rng));
        } else {
            if constexpr (is_complex_v<T>) s.value[i] = T(u(rng), u(rng));
        }
    }
    return s;
}

template <typename T>
class CoeffTyped : public ::testing::Test {};
using CoeffTypes = ::testing::Types<double, cd>;
TYPED_TEST_SUITE(CoeffTyped, CoeffTypes);

// Hermitian inputs for real T, general ones for complex T.
template <typename T>
bool general() { return is_complex_v<T>; }

TYPED_TEST(CoeffTyped, MultiplyMatchesDense) {
    using T = TypeParam;
    auto a = random_sum<T>(6, 1, !general<T>());
    auto b = random_sum<T>(5, 2, !general<T>());
    const auto da = dense::sum(a), db = dense::sum(b);
    a.multiply(b);
    EXPECT_LT(dense::max_abs_diff(dense::sum(a), da * db), kTol);
}

TYPED_TEST(CoeffTyped, CommutatorMatchesDense) {
    using T = TypeParam;
    auto a = random_sum<T>(6, 3, !general<T>());
    auto b = random_sum<T>(6, 4, !general<T>());
    const auto da = dense::sum(a), db = dense::sum(b);
    EXPECT_LT(dense::max_abs_diff(dense::sum(commutator(a, b)), da * db - db * da), kTol);
    EXPECT_LT(dense::max_abs_diff(dense::sum(i_commutator(a, b)),
                                  dense::C(0, 1) * (da * db - db * da)), kTol);
}

TYPED_TEST(CoeffTyped, ArithmeticMatchesDense) {
    using T = TypeParam;
    auto a = random_sum<T>(6, 5, !general<T>());
    auto b = random_sum<T>(6, 6, !general<T>());
    const auto da = dense::sum(a), db = dense::sum(b);
    EXPECT_LT(dense::max_abs_diff(dense::sum(a + b), da + db), kTol);
    EXPECT_LT(dense::max_abs_diff(dense::sum(a - b), da - db), kTol);
    EXPECT_LT(dense::max_abs_diff(dense::sum(-a), -da), kTol);
}

// evolve(rotations) computes U^dagger O U for U = exp(-i theta/2 P).
TYPED_TEST(CoeffTyped, EvolveIsHeisenbergConjugation) {
    using T = TypeParam;
    auto o = random_sum<T>(8, 7, !general<T>());
    const PauliString<kN> p("XYZI");
    const double theta = 0.37;
    PauliSum<kN, T> rot(kQubits, "XYZI", T(theta));
    const auto u = dense::rotation(p, kQubits, theta);
    const dense::Mat expected = u.adjoint() * dense::sum(o) * u;
    o.evolve(rot);
    EXPECT_LT(dense::max_abs_diff(dense::sum(o), expected), kTol);
}

// apply_RX/RY/RZ(c, theta) compute U^dagger O U for the standard gate
// U = exp(-i theta/2 P_c), i.e. the same as evolve() with generator P_c and
// angle theta.
TYPED_TEST(CoeffTyped, SingleQubitRotationsUseStandardAngle) {
    using T = TypeParam;
    const double theta = 0.9;
    for (const char* axis : {"IXII", "IYII", "IZII"}) {
        auto o = random_sum<T>(8, 8, !general<T>());
        auto ref = o;
        const auto u = dense::rotation(PauliString<kN>(axis), kQubits, theta);
        const dense::Mat expected = u.adjoint() * dense::sum(o) * u;
        switch (axis[1]) {
            case 'X': o.apply_RX(1, theta); break;
            case 'Y': o.apply_RY(1, theta); break;
            default:  o.apply_RZ(1, theta); break;
        }
        ref.evolve(PauliSum<kN, T>(kQubits, axis, T(theta)));
        EXPECT_LT(dense::max_abs_diff(dense::sum(o), expected), kTol) << axis;
        EXPECT_LT(dense::max_abs_diff(dense::sum(o), dense::sum(ref)), kTol) << axis;
    }
}

TYPED_TEST(CoeffTyped, CanonicalizePreservesOperator) {
    using T = TypeParam;
    auto a = random_sum<T>(10, 9, !general<T>());
    auto b = a;
    b.multiply(random_sum<T>(4, 10, !general<T>()));  // many duplicates, all phases
    const auto db = dense::sum(b);
    b.canonicalize();
    EXPECT_LT(dense::max_abs_diff(dense::sum(b), db), kTol);
    if constexpr (is_complex_v<T>) {
        for (auto f : b.flags) EXPECT_EQ(f, 0);
    } else {
        for (auto f : b.flags) EXPECT_EQ(f & 0x02, 0);
    }
}

TEST(ComplexCoeff, CanonicalizeCombinesPhases) {
    PauliSum<kN, cd> s(std::vector<std::pair<std::string, cd>>{
        {"XZ", cd(1, 0)}, {"iXZ", cd(0, 1)}, {"-XZ", cd(0.5, 0)}});
    s.canonicalize();
    ASSERT_EQ(s.get_length(), 1u);
    // 1 + i*i - 0.5 = -0.5
    EXPECT_NEAR(s.value[0].real(), -0.5, kTol);
    EXPECT_NEAR(s.value[0].imag(), 0.0, kTol);
}

TYPED_TEST(CoeffTyped, ReductionsMatchDense) {
    using T = TypeParam;
    auto a = random_sum<T>(12, 11, !general<T>());
    auto b = random_sum<T>(12, 12, !general<T>());
    a.push_back(PauliString<kN>("IIII"), T(0.3));
    a.push_back(PauliString<kN>("IZIZ"), T(-0.7));
    a.num_qubits_ = kQubits;
    const auto da = dense::sum(a), db = dense::sum(b);
    const double d = 1 << kQubits;

    auto near = [](T got, dense::C want) {
        EXPECT_NEAR(std::real(got), want.real(), kTol);
        if constexpr (is_complex_v<T>) EXPECT_NEAR(std::imag(got), want.imag(), kTol);
    };

    near(a.trace(), da.trace());
    near(a.zero_state_expectation(), da(0, 0));
    // bits "0110" -> b = 2 + 4
    near(a.basis_state_expectation("0110"), da(6, 6));
    near(trace_product(a, b), (da * db).trace());
    near(inner_product(a, b), (da.adjoint() * db).trace() / d);

    auto c = a;
    c.canonicalize();
    EXPECT_NEAR(c.frobenius_norm2(), (da.adjoint() * da).trace().real() / d, kTol);
    near(c.lookup("IZIZ"), (dense::pauli(PauliString<kN>("IZIZ"), kQubits) * da).trace() / d);
    double l1 = 0;
    for (auto v : c.value) l1 += std::abs(v);
    EXPECT_NEAR(c.l1_norm(), l1, kTol);
}

// The same Hermitian computation gives the same result in both types.
TEST(ComplexCoeff, RealAndComplexAgreeOnHermitianPropagation) {
    auto h = random_sum<double>(10, 20, true);
    auto rots = random_sum<double>(6, 21, true);
    PauliSum<kN, cd> hc, rc;
    hc.num_qubits_ = rc.num_qubits_ = kQubits;
    for (std::size_t i = 0; i < h.get_length(); ++i) hc.push_back(h.get_pauli_string(i), h.value[i]);
    for (std::size_t i = 0; i < rots.get_length(); ++i) rc.push_back(rots.get_pauli_string(i), rots.value[i]);

    TruncationOptions<double> opt_r;
    TruncationOptions<cd> opt_c;
    opt_r.coefficient_cutoff = 1e-12;
    opt_c.coefficient_cutoff = 1e-12;
    const double er = propagate_expectation(h, rots, opt_r);
    const cd ec = propagate_expectation(hc, rc, opt_c);
    EXPECT_NEAR(ec.real(), er, kTol);
    EXPECT_NEAR(ec.imag(), 0.0, kTol);
}

TEST(ComplexCoeff, GetStrShowsComplexCoefficients) {
    PauliSum<kN, cd> s(2, "XZ", cd(1.5, -0.25));
    EXPECT_EQ(s.get_str(), "(1.500000-0.250000i) * XZ\n");
}

}  // namespace
