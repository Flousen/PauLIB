#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "GSim.h"

using paulib::GSim;
using paulib::PauliString;
using paulib::lie_closure;

namespace {

std::vector<PauliString<64>> tfim_generators(std::size_t n) {
    std::vector<PauliString<64>> gens;
    for (std::size_t q = 0; q + 1 < n; ++q) {
        std::string s(n, 'I');
        s[q] = 'Z';
        s[q + 1] = 'Z';
        gens.emplace_back(s);
    }
    for (std::size_t q = 0; q < n; ++q) {
        std::string s(n, 'I');
        s[q] = 'X';
        gens.emplace_back(s);
    }
    return gens;
}

}  // namespace

TEST(GSimTests, LieClosureSu2) {
    std::vector<PauliString<64>> gens = {PauliString<64>("X"), PauliString<64>("Z")};
    auto basis = lie_closure(gens);
    EXPECT_EQ(basis.size(), 3u);

    GSim<64> sim(gens);
    EXPECT_NE(sim.basis_index(PauliString<64>("Y")), GSim<64>::npos);
}

TEST(GSimTests, LieClosureIgnoresIdentityAndDuplicates) {
    std::vector<PauliString<64>> gens = {
        PauliString<64>("II"), PauliString<64>("XI"),
        PauliString<64>("-XI"), PauliString<64>("XI")};
    auto basis = lie_closure(gens);
    EXPECT_EQ(basis.size(), 1u);
    EXPECT_EQ(basis[0].to_string(), "XI");
}

TEST(GSimTests, LieClosureMaxDimThrows) {
    const auto gens = tfim_generators(4);
    const std::size_t full_dim = lie_closure(gens).size();
    ASSERT_GT(full_dim, gens.size());
    EXPECT_THROW(lie_closure(gens, full_dim - 1), std::runtime_error);
}

TEST(GSimTests, ClosureIsClosedUnderCommutation) {
    auto basis = lie_closure(tfim_generators(4));
    ASSERT_GT(basis.size(), 0u);

    paulib::PauliStringXZHash<64> hash;
    paulib::PauliStringXZEqual<64> eq;
    std::unordered_map<PauliString<64>, std::size_t, decltype(hash), decltype(eq)>
        index(basis.size(), hash, eq);
    for (std::size_t a = 0; a < basis.size(); ++a) index.emplace(basis[a], a);

    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t j = 0; j < basis.size(); ++j) {
            if (basis[i].commute(basis[j])) continue;
            PauliString<64> p = basis[i] * basis[j];
            EXPECT_TRUE((p.flags & 0x01) != 0);  // product of anticommuting Paulis is +-i R
            p.flags = 0;
            EXPECT_TRUE(index.count(p));
        }
    }
}

TEST(GSimTests, ZeroStateVector) {
    GSim<64> sim(tfim_generators(3));
    auto e = sim.zero_state_vector();
    for (std::size_t a = 0; a < sim.dim(); ++a) {
        const std::string s = sim.basis()[a].to_string();
        const bool diagonal = s.find('X') == std::string::npos &&
                              s.find('Y') == std::string::npos;
        EXPECT_DOUBLE_EQ(e[a], diagonal ? 1.0 : 0.0) << s;
    }
}

TEST(GSimTests, AdjointGeneratorAntisymmetric) {
    GSim<64> sim(tfim_generators(3));
    for (std::size_t mu = 0; mu < sim.dim(); ++mu) {
        auto A = sim.adjoint_generator(mu);
        EXPECT_NEAR((A + A.transpose()).norm(), 0.0, 1e-14);
    }
}

// <0| exp(i t X) Z exp(-i t X) |0> = cos(2t)
TEST(GSimTests, SingleQubitRotation) {
    std::vector<PauliString<64>> gens = {PauliString<64>("X"), PauliString<64>("Z")};
    GSim<64> sim(gens);
    const std::size_t mu_x = sim.basis_index(PauliString<64>("X"));
    auto w = sim.observable_weights({{PauliString<64>("Z"), 1.0}});

    for (double t : {0.0, 0.3, 1.1, -0.7}) {
        const double got = sim.expectation({{mu_x, t}}, w);
        EXPECT_NEAR(got, std::cos(2 * t), 1e-12) << "t = " << t;
    }
}

TEST(GSimTests, HamiltonianGateMatchesGivensPath) {
    GSim<64> sim(tfim_generators(4));
    const double theta = 0.37;

    auto e_givens = sim.zero_state_vector();
    auto e_expm = sim.zero_state_vector();
    for (std::size_t mu = 0; mu < sim.dim(); ++mu) {
        sim.apply_gate(e_givens, mu, theta);
        sim.apply_hamiltonian_gate(e_expm, {{mu, 1.0}}, theta);
    }
    EXPECT_NEAR((e_givens - e_expm).norm(), 0.0, 1e-10);
}

TEST(GSimTests, ObservableWeightsValidation) {
    GSim<64> sim(tfim_generators(3));
    // Sign prefix folds into the weight.
    auto w = sim.observable_weights({{PauliString<64>("-XII"), 2.0}});
    const std::size_t a = sim.basis_index(PauliString<64>("XII"));
    EXPECT_DOUBLE_EQ(w[a], -2.0);
    // Terms outside the DLA and imaginary terms are rejected.
    EXPECT_THROW(sim.observable_weights({{PauliString<64>("ZII"), 1.0}}),
                 std::invalid_argument);
    EXPECT_THROW(sim.observable_weights({{PauliString<64>("iXII"), 1.0}}),
                 std::invalid_argument);
}

TEST(GSimTests, InitialVectorMatchesZeroStateVector) {
    GSim<64> sim(tfim_generators(2));
    // |00><00| = (1/4)(II + ZI + IZ + ZZ)
    paulib::PauliSum<64, double> rho(std::vector<std::pair<std::string, double>>{
        {"II", 0.25}, {"ZI", 0.25}, {"IZ", 0.25}, {"ZZ", 0.25}});
    EXPECT_NEAR((sim.initial_vector(rho) - sim.zero_state_vector()).norm(), 0.0, 1e-14);
}
