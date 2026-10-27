#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <map>
#include <random>
#include <set>
#include <string>

#include "DenseReference.h"
#include "PauLib.h"

using namespace paulib;

namespace {

// Chi-square statistic of observed counts against a uniform distribution.
double chi2_uniform(const std::map<std::string, int>& counts, std::size_t categories, int draws) {
    const double e = double(draws) / double(categories);
    double chi2 = 0;
    for (const auto& [k, c] : counts) chi2 += (c - e) * (c - e) / e;
    chi2 += double(categories - counts.size()) * e;  // categories never drawn
    return chi2;
}

TEST(Generate, RandomPauliStringUniformAndPadded) {
    std::mt19937_64 rng(1);
    std::map<std::string, int> counts;
    const int draws = 40000;
    for (int i = 0; i < draws; ++i) counts[random_pauli_string<8>(1, rng).to_string()]++;
    EXPECT_EQ(counts.size(), 4u);
    // 3 degrees of freedom; the 99.9% quantile is 16.3.
    EXPECT_LT(chi2_uniform(counts, 4, draws), 16.3);

    for (int i = 0; i < 100; ++i) {
        const auto p = random_pauli_string<128>(70, rng);
        EXPECT_EQ(p.num_qubits(), 70u);
        EXPECT_EQ(p.x[1] >> 6, 0u);
        EXPECT_EQ(p.z[1] >> 6, 0u);
        EXPECT_EQ(p.flags, 0);
    }
}

TEST(Generate, TermSamplerFollowsAbsoluteCoefficients) {
    PauliSum<8, std::complex<double>> s(std::vector<std::pair<std::string, std::complex<double>>>{
        {"XI", {1.0, 0.0}}, {"ZZ", {-2.0, 0.0}}, {"YI", {0.0, 3.0}}, {"IX", {0.0, 0.0}}});
    TermSampler sampler(s);
    EXPECT_DOUBLE_EQ(sampler.lambda(), 6.0);
    EXPECT_DOUBLE_EQ(sampler.lambda(), s.l1_norm());
    const int draws = 60000;
    const auto idx = sample_terms(s, draws, 7);
    std::vector<int> c(4, 0);
    for (auto j : idx) c[j]++;
    EXPECT_EQ(c[3], 0);
    EXPECT_NEAR(c[0] / double(draws), 1.0 / 6, 0.01);
    EXPECT_NEAR(c[1] / double(draws), 2.0 / 6, 0.01);
    EXPECT_NEAR(c[2] / double(draws), 3.0 / 6, 0.01);
    EXPECT_EQ(sample_terms(s, 50, 3), sample_terms(s, 50, 3));
}

TEST(Generate, StringsUpToWeight) {
    const auto s = strings_up_to_weight<8, double>(5, 2);
    EXPECT_EQ(s.get_length(), 1u + 5 * 3 + 10 * 9);
    std::set<std::string> seen;
    for (std::size_t i = 0; i < s.get_length(); ++i) {
        EXPECT_LE(s.term_weight(i), 2u);
        seen.insert(s.get_pauli_string(i).to_string());
    }
    EXPECT_EQ(seen.size(), s.get_length());
    EXPECT_EQ(s.get_pauli_string(0).to_string(), "IIIII");
    EXPECT_EQ(s.get_pauli_string(1).to_string(), "XIIII");
    EXPECT_EQ((strings_up_to_weight<8, double>(3, 10).get_length()), 64u);
    EXPECT_THROW((strings_up_to_weight<64, double>(60, 4, 1000)), std::length_error);
}

TEST(Generate, LatticeEdges) {
    EXPECT_EQ(chain_edges(4).size(), 3u);
    EXPECT_EQ(chain_edges(4, true).size(), 4u);
    EXPECT_EQ(chain_edges(2, true).size(), 1u);
    EXPECT_EQ(square_lattice_edges(2, 3).size(), 3u + 4u);
    EXPECT_EQ(square_lattice_edges(3, 3, true).size(), 18u);
}

TEST(Generate, TransverseFieldIsing) {
    const std::size_t n = 4;
    auto h = two_site_sum<8, double>(n, chain_edges(n), "ZZ", -1.0) +
             one_site_sum<8, double>(n, "X", 0.5);
    h.canonicalize();
    EXPECT_EQ(h.get_length(), 7u);
    EXPECT_DOUBLE_EQ(h.lookup("IZZI"), -1.0);
    EXPECT_DOUBLE_EQ(h.lookup("IIIX"), 0.5);
    EXPECT_DOUBLE_EQ(h.zero_state_expectation(), -3.0);
}

template <typename T>
PauliSum<8, T> anticommutator(const PauliSum<8, T>& a, const PauliSum<8, T>& b) {
    auto r = a * b + b * a;
    r.canonicalize();
    return r;
}

template <typename T>
class JWTyped : public ::testing::Test {};
using JWTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_SUITE(JWTyped, JWTypes);

TYPED_TEST(JWTyped, CanonicalAnticommutationRelations) {
    using T = TypeParam;
    const std::size_t n = 3;
    for (std::size_t i = 0; i < 2 * n; ++i)
        for (std::size_t j = 0; j < 2 * n; ++j) {
            PauliSum<8, T> gi, gj;
            gi.num_qubits_ = gj.num_qubits_ = n;
            gi.push_back(jw_majorana<8>(n, i), T(1));
            gj.push_back(jw_majorana<8>(n, j), T(1));
            const auto r = anticommutator(gi, gj);
            if (i == j) {
                ASSERT_EQ(r.get_length(), 1u);
                EXPECT_NEAR(std::real(r.trace()), 2.0 * 8, 1e-12);
            } else {
                EXPECT_EQ(r.get_length(), 0u) << i << "," << j;
            }
        }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const auto a = jw_annihilation<8, T>(n, i);
            const auto ad = jw_creation<8, T>(n, j);
            const auto r = anticommutator(a, ad);
            if (i == j) {
                ASSERT_EQ(r.get_length(), 1u);
                EXPECT_NEAR(std::real(r.lookup("III")), 1.0, 1e-12);
            } else {
                EXPECT_EQ(r.get_length(), 0u);
            }
            EXPECT_EQ(anticommutator(a, jw_annihilation<8, T>(n, j)).get_length(), 0u);
        }
}

TEST(Generate, JWNumberOperatorMatchesDense) {
    // a_1^dagger a_1 = (I - Z_1)/2
    auto num = jw_creation<8, double>(3, 1) * jw_annihilation<8, double>(3, 1);
    num.canonicalize();
    EXPECT_NEAR(num.lookup("III"), 0.5, 1e-12);
    EXPECT_NEAR(num.lookup("IZI"), -0.5, 1e-12);
    EXPECT_EQ(num.get_length(), 2u);
}

TEST(Generate, ApplyCircuitMatchesDense) {
    std::mt19937_64 rng(11);
    const std::size_t n = 3;
    for (int trial = 0; trial < 20; ++trial) {
        const auto c = random_clifford(n, rng);
        const auto u = dense::circuit(c, n);
        auto s = random_pauli_sum<8, double>(6, 100 + trial, n);
        const auto d0 = dense::sum(s);
        auto s2 = s;
        apply_circuit(s, c);
        EXPECT_LT(dense::max_abs_diff(dense::sum(s), u.adjoint() * d0 * u), 1e-12);
        apply_circuit(s2, adjoint(c));
        EXPECT_LT(dense::max_abs_diff(dense::sum(s2), u * d0 * u.adjoint()), 1e-12);
    }
}

TEST(Generate, ApplyCircuitAllGateKinds) {
    using K = CliffordGate;
    const CliffordCircuit c{{K::H, 0}, {K::S, 1}, {K::SDG, 2}, {K::X, 0}, {K::Y, 1}, {K::Z, 2},
                            {K::CNOT, 2, 0}, {K::CZ, 0, 1}, {K::SWAP, 1, 2}, {K::CNOT, 0, 1}};
    const auto u = dense::circuit(c, 3);
    for (const char* str : {"XII", "IYI", "IIZ", "ZXI", "YYX", "-iXZY"}) {
        PauliString<8> p(str);
        PauliSum<8, double> s(3, str, 1.0);
        const auto d0 = dense::pauli(p, 3);
        apply_circuit(p, c);
        apply_circuit(s, c);
        EXPECT_EQ(s.get_pauli_string(0), p) << str;
        EXPECT_LT(dense::max_abs_diff(dense::pauli(p, 3), u.adjoint() * d0 * u), 1e-12) << str;
    }
}

std::string image(const CliffordCircuit& c, const char* str) {
    PauliString<8> p(str);
    apply_circuit(p, c);
    return p.to_string();
}

TEST(Generate, RandomCliffordUniformOnOneQubit) {
    std::mt19937_64 rng(5);
    std::map<std::string, int> counts;
    const int draws = 48000;
    for (int i = 0; i < draws; ++i) {
        const auto c = random_clifford(1, rng);
        counts[image(c, "X") + "," + image(c, "Z")]++;
    }
    EXPECT_EQ(counts.size(), 24u);  // |C_1 / U(1)| = 24
    // 23 degrees of freedom; the 99.9% quantile is 49.7.
    EXPECT_LT(chi2_uniform(counts, 24, draws), 49.7);
}

TEST(Generate, RandomCliffordUniformImagesOnTwoQubits) {
    std::mt19937_64 rng(6);
    std::map<std::string, int> x0, z1;
    std::set<std::string> pairs;
    const int draws = 60000;
    for (int i = 0; i < draws; ++i) {
        const auto c = random_clifford(2, rng);
        x0[image(c, "XI")]++;
        z1[image(c, "IZ")]++;
        if (i < 20000) pairs.insert(image(c, "XI") + image(c, "ZI") + image(c, "IX") + image(c, "IZ"));
    }
    // 15 non-identity strings times 2 signs; 29 degrees of freedom, 99.9% quantile 58.3.
    EXPECT_EQ(x0.size(), 30u);
    EXPECT_EQ(z1.size(), 30u);
    EXPECT_LT(chi2_uniform(x0, 30, draws), 58.3);
    EXPECT_LT(chi2_uniform(z1, 30, draws), 58.3);
    // |C_2 / U(1)| = 11520; 20000 draws reach most of them.
    EXPECT_GT(pairs.size(), 9000u);
}

TEST(Generate, RandomCliffordGateCountIsQuadratic) {
    std::mt19937_64 rng(9);
    const auto c = random_clifford(40, rng);
    EXPECT_LT(c.size(), 6u * 40 * 40);
    for (const auto& g : c) {
        EXPECT_LT(g.q0, 40u);
        EXPECT_LT(g.q1, 40u);
    }
}

}  // namespace
