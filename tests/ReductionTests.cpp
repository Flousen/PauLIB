#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <random>

#include "DenseReference.h"
#include "PauLib.h"

using namespace paulib;

namespace {

constexpr double kTol = 1e-10;
using cd = std::complex<double>;

template <typename T>
PauliSum<8, T> random_sum(std::size_t terms, std::uint64_t seed, std::size_t n) {
    auto s = random_pauli_sum<8, T>(terms, seed, n);
    std::mt19937_64 rng(seed + 1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t i = 0; i < s.get_length(); ++i) {
        if constexpr (is_complex_v<T>) {
            s.value[i] = T(u(rng), u(rng));
        } else {
            s.flags[i] &= 0x02;  // Hermitian for real T
            s.value[i] = u(rng);
        }
    }
    return s;
}

template <typename T>
void expect_near(T got, cd want) {
    EXPECT_NEAR(std::real(got), want.real(), kTol);
    if constexpr (is_complex_v<T>) EXPECT_NEAR(std::imag(got), want.imag(), kTol);
}

template <typename T>
class ReductionTyped : public ::testing::Test {};
using ReductionTypes = ::testing::Types<double, cd>;
TYPED_TEST_SUITE(ReductionTyped, ReductionTypes);

TYPED_TEST(ReductionTyped, ProductStateExpectationMatchesDense) {
    using T = TypeParam;
    const std::size_t n = 4;
    auto a = random_sum<T>(30, 1, n);
    std::mt19937_64 rng(2);
    std::normal_distribution<double> g;
    std::vector<std::array<double, 3>> bloch(n);
    for (auto& r : bloch) {
        r = {g(rng), g(rng), g(rng)};
        const double len = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        for (auto& c : r) c *= 0.9 / len;  // mixed single-qubit states
    }
    const auto rho = dense::product_state(bloch);
    expect_near(a.product_state_expectation(bloch), (rho * dense::sum(a)).trace());

    std::vector<std::array<double, 3>> zero(n, {0.0, 0.0, 1.0});
    expect_near(a.product_state_expectation(zero), cd(a.zero_state_expectation()));
    EXPECT_THROW(a.product_state_expectation(std::vector<std::array<double, 3>>(3)),
                 std::invalid_argument);
}

TYPED_TEST(ReductionTyped, StabilizerExpectationOfRandomStabilizerStates) {
    using T = TypeParam;
    const std::size_t n = 4;
    std::mt19937_64 rng(3);
    for (int trial = 0; trial < 10; ++trial) {
        // U|0> is stabilized by U Z_j U^dagger.
        const auto c = random_clifford(n, rng);
        std::vector<PauliString<8>> gens;
        for (std::size_t j = 0; j < n; ++j) {
            std::string s(n, 'I');
            s[j] = 'Z';
            PauliString<8> p(s);
            apply_circuit(p, adjoint(c));
            gens.push_back(p);
        }
        Eigen::VectorXcd psi = dense::circuit(c, n).col(0);
        auto a = random_sum<T>(40, 10 + trial, n);
        // Add a few stabilizer elements so that the result is not just 0.
        for (std::size_t j = 0; j < n; ++j) a.push_back(gens[j] * gens[(j + 1) % n], T(0.25 * (j + 1)));
        const cd want = psi.adjoint() * dense::sum(a) * psi;
        expect_near(stabilizer_expectation(a, gens), want);
        expect_near(stabilizer_expectation(a, gens),
                    (dense::stabilizer_state(gens, n) * dense::sum(a)).trace());
    }
}

TEST(Reductions, StabilizerExpectationMixedCodeState) {
    // Two generators on three qubits: maximally mixed on the code space.
    const std::vector<PauliString<8>> gens{PauliString<8>("ZZI"), PauliString<8>("-IZZ")};
    auto a = random_sum<cd>(30, 4, 3);
    a.push_back(PauliString<8>("ZIZ"), cd(0.7, 0.1));
    expect_near(stabilizer_expectation(a, gens),
                (dense::stabilizer_state(gens, 3) * dense::sum(a)).trace());
}

TEST(Reductions, StabilizerExpectationGHZ) {
    const std::vector<PauliString<8>> gens{PauliString<8>("XXX"), PauliString<8>("ZZI"),
                                           PauliString<8>("IZZ")};
    PauliSum<8, double> a(std::vector<std::pair<std::string, double>>{
        {"YYX", 1.0}, {"ZIZ", 2.0}, {"XII", 4.0}, {"III", 0.5}});
    // YYX = -(XXX)(ZZI) up to the product: <YYX> = -1, <ZIZ> = 1, <XII> = 0.
    EXPECT_NEAR(stabilizer_expectation(a, gens), -1.0 + 2.0 + 0.5, kTol);
}

TEST(Reductions, StabilizerExpectationRejectsInvalidGenerators) {
    PauliSum<8, double> a(2, "ZZ", 1.0);
    EXPECT_THROW(stabilizer_expectation(a, {PauliString<8>("XI"), PauliString<8>("ZI")}),
                 std::invalid_argument);
    EXPECT_THROW(stabilizer_expectation(a, {PauliString<8>("ZZ"), PauliString<8>("-ZZ")}),
                 std::invalid_argument);
    EXPECT_THROW(stabilizer_expectation(a, {PauliString<8>("iZZ")}), std::invalid_argument);
}

TYPED_TEST(ReductionTyped, WeightHistogramSumsToFrobeniusNorm) {
    using T = TypeParam;
    auto a = random_sum<T>(50, 5, 6);
    a.canonicalize();
    const auto h = a.weight_histogram();
    ASSERT_EQ(h.size(), 7u);
    double total = 0;
    for (double v : h) total += v;
    EXPECT_NEAR(total, a.frobenius_norm2(), kTol);
    PauliSum<8, T> b(std::vector<std::pair<std::string, T>>{
        {"XIII", T(1)}, {"XYII", T(2)}, {"ZZZI", T(3)}, {"IYII", T(-1)}});
    EXPECT_EQ(b.weight_histogram(), (std::vector<double>{0, 2, 4, 9, 0}));
}

TYPED_TEST(ReductionTyped, InnerProductOfSelfIsFrobeniusNorm) {
    using T = TypeParam;
    auto a = random_sum<T>(40, 6, 5);
    a.canonicalize();
    expect_near(inner_product(a, a), cd(a.frobenius_norm2()));
}

}  // namespace
