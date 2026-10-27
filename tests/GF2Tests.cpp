#include <gtest/gtest.h>

#include <random>

#include <omp.h>
#include <vector>

#include "PauLib.h"

using namespace paulib;

namespace {

// Strings whose X and Z bits are set with probability `density`, so that the
// rank can stay below the number of strings.
template <std::size_t N>
PauliSum<N, double> random_sparse(std::size_t m, std::size_t n, std::uint64_t seed, double density) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution bit(density);
    PauliSum<N, double> s;
    s.num_qubits_ = n;
    for (std::size_t i = 0; i < m; ++i) {
        PauliString<N> p{};
        p.num_qubits_ = static_cast<uint32_t>(n);
        for (std::size_t q = 0; q < n; ++q) {
            if (bit(rng)) p.x[q / 64] |= std::size_t(1) << (q % 64);
            if (bit(rng)) p.z[q / 64] |= std::size_t(1) << (q % 64);
        }
        s.push_back(p, 1.0);
    }
    return s;
}

template <std::size_t N, typename T>
std::vector<PauliString<N>> strings_of(const PauliSum<N, T>& s) {
    std::vector<PauliString<N>> v;
    for (std::size_t i = 0; i < s.get_length(); ++i) v.push_back(s.get_pauli_string(i));
    return v;
}

// Ordinary GF(2) dot product of (x_a | z_a) and (x_b | z_b).
template <std::size_t N>
bool dot(const PauliString<N>& a, const PauliString<N>& b) {
    std::size_t acc = 0;
    for (std::size_t k = 0; k < PauliString<N>::ArraySize; ++k) acc ^= (a.x[k] & b.x[k]) ^ (a.z[k] & b.z[k]);
    return __builtin_parityll(acc);
}

TEST(GF2, RankKnownSets) {
    // X_q and Z_q for every qubit are 2n independent strings.
    PauliSum<128, double> basis;
    basis.num_qubits_ = 70;
    for (std::size_t q = 0; q < 70; ++q) {
        PauliString<128> x{}, z{};
        x.num_qubits_ = z.num_qubits_ = 70;
        x.x[q / 64] |= std::size_t(1) << (q % 64);
        z.z[q / 64] |= std::size_t(1) << (q % 64);
        basis.push_back(x, 1.0);
        basis.push_back(z, 1.0);
    }
    EXPECT_EQ(gf2_rank(basis), 140u);

    // The third string is the product of the first two up to a phase.
    PauliSum<8, double> s(std::vector<std::string>{"XXI", "IZZ", "XYZ"});
    EXPECT_EQ(gf2_rank(s), 2u);
}

TEST(GF2, RrefLeavesIdentityBelowTheRank) {
    auto s = random_sparse<128>(60, 70, 5, 0.05);
    const auto pivots = gf2_rref(s);
    for (std::size_t i = pivots.size(); i < s.get_length(); ++i)
        for (std::size_t k = 0; k < PauliSum<128, double>::ArraySize; ++k) {
            ASSERT_EQ(s.x_data[k][i], 0u);
            ASSERT_EQ(s.z_data[k][i], 0u);
        }
}

// The elimination on the word arrays and the incremental basis are two
// independent algorithms; their ranks agree.
TEST(GF2, RankAgreesWithIncrementalBasis) {
    for (std::uint64_t seed = 0; seed < 20; ++seed) {
        const auto s = random_sparse<128>(40 + seed, 45 + seed, seed, 0.02 + 0.01 * seed);
        GF2Basis<128> b(s.num_qubits(), s.get_length());
        for (std::size_t j = 0; j < s.get_length(); ++j) b.insert(s.get_pauli_string(j), j);
        EXPECT_EQ(gf2_rank(s), b.rank()) << seed;
    }
}

TEST(GF2, RowBasisSpansTheSameSpace) {
    const auto s = random_sparse<64>(30, 25, 7, 0.1);
    const auto b = gf2_row_basis(s);
    EXPECT_EQ(b.get_length(), gf2_rank(s));
    EXPECT_EQ(gf2_rank(b), b.get_length());
    // Appending the input to the basis does not increase the rank.
    auto stacked = b;
    for (std::size_t j = 0; j < s.get_length(); ++j) stacked.push_back(s.get_pauli_string(j), 1.0);
    EXPECT_EQ(gf2_rank(stacked), b.get_length());
}

TEST(GF2, NullspaceIsKernelWithFullDimension) {
    for (std::uint64_t seed = 0; seed < 10; ++seed) {
        const auto s = random_sparse<128>(50, 70, 100 + seed, 0.3);
        const auto ns = gf2_nullspace(s);
        const std::size_t rank = gf2_rank(s);
        ASSERT_EQ(ns.get_length(), 2 * s.num_qubits() - rank);
        EXPECT_EQ(gf2_rank(ns), ns.get_length());
        for (std::size_t v = 0; v < ns.get_length(); ++v)
            for (std::size_t i = 0; i < s.get_length(); ++i)
                ASSERT_FALSE(dot(s.get_pauli_string(i), ns.get_pauli_string(v)))
                    << "row " << i << " vector " << v;
    }
}

// Columns next to the word boundaries of the X and the Z part (n = 100).
TEST(GF2, ColumnsAcrossWordBoundaries) {
    PauliSum<128, double> s;
    s.num_qubits_ = 100;
    for (std::size_t q : {0, 62, 63, 64, 65, 99}) {
        PauliString<128> x{}, z{};
        x.num_qubits_ = z.num_qubits_ = 100;
        x.x[q / 64] |= std::size_t(1) << (q % 64);
        z.z[q / 64] |= std::size_t(1) << (q % 64);
        s.push_back(z, 1.0);
        s.push_back(x, 1.0);
    }
    auto r = s;
    const auto pivots = gf2_rref(r);
    EXPECT_EQ(pivots, (std::vector<std::size_t>{0, 62, 63, 64, 65, 99, 100, 162, 163, 164, 165, 199}));
}

TEST(GF2, IndependentSubsetAndInSpan) {
    std::vector<PauliString<8>> v{PauliString<8>("XXI"), PauliString<8>("IZZ"),
                                  PauliString<8>("-XXI"), PauliString<8>("XYZ"),
                                  PauliString<8>("ZII")};
    // XYZ = i (XXI)(IZZ) up to phase -> dependent; -XXI duplicates XXI.
    EXPECT_EQ(independent_subset(v, 3), (std::vector<std::size_t>{0, 1, 4}));
    EXPECT_EQ(pauli_rank(v, 3), 3u);
    EXPECT_TRUE(in_span(v, PauliString<8>("YYZ"), 3));   // XXI * IZZ * ZII
    EXPECT_FALSE(in_span(v, PauliString<8>("IIX"), 3));
}

TEST(GF2, CommutantOfIsingWithParity) {
    PauliSum<8, double> h(std::vector<std::pair<std::string, double>>{
        {"ZZII", 1.0}, {"IZZI", 1.0}, {"IIZZ", 1.0}, {"XXXX", 0.5}});
    const auto c = commutant(h);
    EXPECT_EQ(c.get_length(), 8 - pauli_rank(h));
    for (const auto& p : strings_of(c)) EXPECT_TRUE(commutes_with_all(h, p)) << p;
    EXPECT_TRUE(in_span(c, PauliString<8>("XXXX")));
    EXPECT_TRUE(in_span(c, PauliString<8>("ZZZZ")));
}

TEST(GF2, CommutantIsSymplecticComplement) {
    for (std::uint64_t seed = 0; seed < 5; ++seed) {
        auto s = random_pauli_sum<128, double>(15, seed, 70);
        const auto in = strings_of(s);
        const auto c = commutant(in, 70);
        EXPECT_EQ(c.size(), 140 - pauli_rank(in, 70));
        for (const auto& p : c)
            for (const auto& q : in) ASSERT_TRUE(p.commute(q));
        // The commutant of the commutant is the span of the input.
        const auto cc = commutant(c, 70);
        EXPECT_EQ(cc.size(), pauli_rank(in, 70));
        for (const auto& q : in) EXPECT_TRUE(in_span(cc, q, 70));
    }
}

TEST(GF2, LargeRrefIndependentOfThreadCount) {
    // 2000 terms on 1050 qubits: 2000 * 2 * 17 words, above the parallel threshold.
    const auto m = random_sparse<1088>(2000, 1050, 42, 0.005);
    auto a = m, b = m;
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto p1 = gf2_rref(a);
    omp_set_num_threads(8);
    const auto p8 = gf2_rref(b);
    omp_set_num_threads(saved);
    EXPECT_EQ(p1, p8);
    EXPECT_EQ(a.x_data, b.x_data);
    EXPECT_EQ(a.z_data, b.z_data);
}

}  // namespace
