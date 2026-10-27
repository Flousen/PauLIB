#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "DenseReference.h"
#include "PauLib.h"

using namespace paulib;

namespace {

// |0...0> as a column vector.
dense::Mat zero_state(std::size_t n) {
    dense::Mat v = dense::Mat::Zero(std::size_t(1) << n, 1);
    v(0, 0) = 1.0;
    return v;
}

// <v| P |v> for a normalised column vector.
template <std::size_t N>
double dense_expectation(const PauliString<N>& p, const dense::Mat& v, std::size_t n) {
    return (v.adjoint() * dense::pauli(p, n) * v)(0, 0).real();
}

// Projects v onto outcome b of Z_q and normalises; returns the probability.
double project_z(dense::Mat& v, std::size_t q, bool b) {
    double prob = 0;
    for (Eigen::Index i = 0; i < v.rows(); ++i) {
        if (((static_cast<std::size_t>(i) >> q) & 1) != static_cast<std::size_t>(b)) v(i, 0) = 0;
        else prob += std::norm(v(i, 0));
    }
    if (prob > 0) v /= std::sqrt(prob);
    return prob;
}

template <std::size_t N>
PauliString<N> z_on(std::size_t q, std::size_t n) {
    PauliString<N> p{};
    p.num_qubits_ = static_cast<uint32_t>(n);
    p.z[q / 64] |= std::size_t(1) << (q % 64);
    return p;
}

}  // namespace

TEST(Tableau, StartState) {
    Tableau<8> tab(3);
    const auto s = tab.stabilizers();
    EXPECT_EQ(s[0], PauliString<8>("ZII"));
    EXPECT_EQ(s[2], PauliString<8>("IIZ"));
    EXPECT_EQ(tab.destabilizers()[1], PauliString<8>("IXI"));
    EXPECT_EQ(tab.peek(PauliString<8>("ZZI")), 1);
    EXPECT_EQ(tab.peek(PauliString<8>("-IZI")), -1);
    EXPECT_EQ(tab.peek(PauliString<8>("XII")), 0);
}

TEST(Tableau, StabilizersFollowCircuit) {
    std::mt19937_64 rng(1);
    for (std::size_t n : {1, 4, 70}) {
        const auto c = random_clifford(n, rng);
        Tableau<128> tab(n);
        apply_circuit(tab, c);
        const auto s = tab.stabilizers();
        for (std::size_t i = 0; i < n; ++i) {
            auto zi = z_on<128>(i, n);
            apply_circuit(zi, adjoint(c));
            EXPECT_EQ(s[i], zi) << "n=" << n << " i=" << i;
        }
    }
}

// The tableau evolves the state, so gate G maps every row s -> G s G^dagger:
// the inverse of the Heisenberg gate on PauliString, which is that gate
// applied three times (G^4 = I up to a phase; once for the self-inverse ones).
TEST(Tableau, GatesActOnTheState) {
    using PS = PauliString<8>;
    struct Case { const char* name; void (Tableau<8>::*tab)(std::size_t); void (PS::*heis)(std::size_t); };
    const std::vector<Case> cases = {
        {"S", &Tableau<8>::apply_S, &PS::apply_sqrt_Z},
        {"sqrt_X", &Tableau<8>::apply_sqrt_X, &PS::apply_sqrt_X},
        {"sqrt_Y", &Tableau<8>::apply_sqrt_Y, &PS::apply_sqrt_Y},
        {"H", &Tableau<8>::apply_H, &PS::apply_H_XZ},
        {"H_YZ", &Tableau<8>::apply_H_YZ, &PS::apply_H_YZ},
        {"H_XY", &Tableau<8>::apply_H_XY, &PS::apply_H_XY},
    };
    std::mt19937_64 rng(7);
    const std::size_t n = 5;
    for (const auto& cs : cases)
        for (std::size_t q : {0, 3}) {
            Tableau<8> tab(n);
            apply_circuit(tab, random_clifford(n, rng));
            auto stab = tab.stabilizers(), destab = tab.destabilizers();
            for (auto* rows : {&stab, &destab})
                for (auto& s : *rows)
                    for (int k = 0; k < 3; ++k) (s.*cs.heis)(q);
            (tab.*cs.tab)(q);
            EXPECT_EQ(tab.stabilizers(), stab) << cs.name << " q=" << q;
            EXPECT_EQ(tab.destabilizers(), destab) << cs.name << " q=" << q;
        }
}

TEST(Tableau, PeekMatchesStabilizerExpectation) {
    std::mt19937_64 rng(2);
    for (std::size_t n : {2, 5, 40}) {
        const auto c = random_clifford(n, rng);
        Tableau<64> tab(n);
        apply_circuit(tab, c);
        const auto gens = tab.stabilizers();
        // Random strings plus products of stabilizers (nonzero expectation).
        for (int trial = 0; trial < 60; ++trial) {
            PauliString<64> p = random_pauli_string<64>(n, rng);
            if (trial % 2) {
                p = PauliString<64>{};
                p.num_qubits_ = static_cast<uint32_t>(n);
                for (const auto& g : gens) if (rng() & 1) p.multiply(g);
                if (rng() & 1) p.flip_sign();
            }
            PauliSum<64, double> one;
            one.num_qubits_ = n;
            one.push_back(p, 1.0);
            const double ref = stabilizer_expectation(one, gens);
            EXPECT_EQ(tab.peek(p), static_cast<int>(std::lround(ref))) << p.to_string();
        }
    }
}

TEST(Tableau, MeasurementMatchesStatevector) {
    std::mt19937_64 rng(3);
    for (std::size_t n : {1, 2, 3, 5}) {
        for (int trial = 0; trial < 30; ++trial) {
            const auto c = random_clifford(n, rng);
            Tableau<8> tab(n);
            apply_circuit(tab, c);
            dense::Mat v = dense::circuit(c, n) * zero_state(n);
            for (std::size_t q = 0; q < n; ++q) {
                const double p1 = dense_expectation(z_on<8>(q, n), v, n);  // <Z_q>
                const auto r = tab.measure_z(q, rng);
                if (r.deterministic) {
                    EXPECT_NEAR(std::abs(p1), 1.0, 1e-9);
                    EXPECT_EQ(r.outcome, p1 < 0);
                } else {
                    EXPECT_NEAR(p1, 0.0, 1e-9);
                }
                EXPECT_NEAR(project_z(v, q, r.outcome), r.deterministic ? 1.0 : 0.5, 1e-9);
                // Every stabilizer of the tableau stabilizes the projected state.
                for (const auto& s : tab.stabilizers())
                    ASSERT_NEAR(dense_expectation(s, v, n), 1.0, 1e-9) << s.to_string();
            }
        }
    }
}

TEST(Tableau, RandomOutcomesAreBalanced) {
    std::mt19937_64 rng(4);
    int ones = 0;
    const int shots = 4000;
    for (int i = 0; i < shots; ++i) {
        Tableau<8> tab(1);
        tab.apply_H(0);
        const auto r = tab.measure_z(0, rng);
        EXPECT_FALSE(r.deterministic);
        ones += r.outcome;
    }
    EXPECT_NEAR(static_cast<double>(ones) / shots, 0.5, 0.03);
}

TEST(Tableau, BellAndGhz) {
    std::mt19937_64 rng(5);
    Tableau<8> bell(2);
    bell.apply_H(0);
    bell.apply_CNOT(0, 1);
    EXPECT_EQ(bell.peek(PauliString<8>("XX")), 1);
    EXPECT_EQ(bell.peek(PauliString<8>("-YY")), 1);
    const auto a = bell.measure_z(0, rng);
    const auto b = bell.measure_z(1, rng);
    EXPECT_FALSE(a.deterministic);
    EXPECT_TRUE(b.deterministic);
    EXPECT_EQ(a.outcome, b.outcome);

    const std::size_t n = 1000;
    for (int shot = 0; shot < 3; ++shot) {
        Tableau<1024> ghz(n);
        ghz.apply_H(0);
        for (std::size_t q = 0; q + 1 < n; ++q) ghz.apply_CNOT(q, q + 1);
        const auto first = ghz.measure_z(0, rng);
        EXPECT_FALSE(first.deterministic);
        for (std::size_t q = 1; q < n; q += 37) {
            const auto r = ghz.measure_z(q, rng);
            ASSERT_TRUE(r.deterministic);
            ASSERT_EQ(r.outcome, first.outcome);
        }
    }
}

TEST(Tableau, ResetGivesZero) {
    std::mt19937_64 rng(6);
    for (int trial = 0; trial < 20; ++trial) {
        const auto c = random_clifford(4, rng);
        Tableau<8> tab(4);
        apply_circuit(tab, c);
        for (std::size_t q = 0; q < 4; ++q) tab.reset(q, rng);
        for (std::size_t q = 0; q < 4; ++q) {
            const auto r = tab.measure_z(q, rng);
            EXPECT_TRUE(r.deterministic);
            EXPECT_FALSE(r.outcome);
        }
    }
}

TEST(Tableau, PeekRejectsNonHermitian) {
    Tableau<8> tab(2);
    EXPECT_THROW(tab.peek(PauliString<8>("iXI")), std::invalid_argument);
}
