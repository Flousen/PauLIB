#include <gtest/gtest.h>

#include <functional>
#include <random>
#include <string>
#include <vector>

#include "PauLib.h"

using namespace paulib;

namespace {

// Random sum whose terms carry all four phases i^k.
template <std::size_t N>
PauliSum<N, double> random_sum(std::size_t m, std::size_t n, std::uint64_t seed) {
    auto s = random_pauli_sum<N, double>(m, seed, n);
    std::mt19937_64 rng(seed + 1);
    for (auto& f : s.flags) f = static_cast<uint8_t>(rng() & 3);
    return s;
}

template <std::size_t N>
void expect_same(const PauliSum<N, double>& a, const PauliSum<N, double>& b) {
    ASSERT_EQ(a.get_length(), b.get_length());
    for (std::size_t j = 0; j < a.get_length(); ++j)
        ASSERT_EQ(a.get_pauli_string(j), b.get_pauli_string(j)) << "term " << j;
}

std::vector<std::size_t> random_mask(std::size_t m, std::mt19937_64& rng) {
    std::vector<std::size_t> mask((m + 63) / 64, 0);
    for (std::size_t j = 0; j < m; ++j)
        if (rng() & 1) mask[j / 64] |= std::size_t(1) << (j % 64);
    return mask;
}

bool has(const std::vector<std::size_t>& mask, std::size_t j) {
    return (mask[j / 64] >> (j % 64)) & 1;
}

}  // namespace

TEST(PauliTable, Transpose64MatchesNaive) {
    std::mt19937_64 rng(1);
    std::size_t a[64], b[64];
    for (auto& w : a) w = rng();
    std::copy(a, a + 64, b);
    transpose64(b);
    for (std::size_t r = 0; r < 64; ++r)
        for (std::size_t c = 0; c < 64; ++c)
            ASSERT_EQ((a[r] >> c) & 1, (b[c] >> r) & 1) << r << "," << c;
}

TEST(PauliTable, RoundTrip) {
    for (std::size_t n : {1, 63, 64, 65, 130})
        for (std::size_t m : {1, 63, 64, 65, 200}) {
            const auto s = random_sum<192>(m, n, 10 * n + m);
            const PauliTable t = to_table(s);
            ASSERT_EQ(t.num_qubits(), n);
            ASSERT_EQ(t.size(), m);
            for (std::size_t j = 0; j < m; j += 17)
                ASSERT_EQ(t.get<192>(j), s.get_pauli_string(j));
            const auto back = to_sum<192, double>(t, s.value);
            expect_same(back, s);
            EXPECT_EQ(back.value, s.value);
            EXPECT_EQ(back.num_qubits(), n);
        }
}

TEST(PauliTable, SetGetCopy) {
    PauliTable t(5, 3);
    PauliString<8> p("-iXYZIY");
    t.set(1, p);
    EXPECT_EQ(t.get<8>(1), p);
    t.copy_term(2, 1);
    EXPECT_EQ(t.get<8>(2), p);
    EXPECT_EQ(t.get<8>(0), PauliString<8>("IIIII"));
}

TEST(PauliTable, GatesMatchPauliSum) {
    using S = PauliSum<128, double>;
    using G1 = std::function<void(S&, PauliTable&, std::size_t)>;
    const std::vector<std::pair<std::string, G1>> single = {
        {"H",      [](S& s, PauliTable& t, std::size_t q) { s.apply_H(q); t.apply_H(q); }},
        {"S",      [](S& s, PauliTable& t, std::size_t q) { s.apply_S(q); t.apply_S(q); }},
        {"H_YZ",   [](S& s, PauliTable& t, std::size_t q) { s.apply_H_YZ(q); t.apply_H_YZ(q); }},
        {"H_XY",   [](S& s, PauliTable& t, std::size_t q) { s.apply_H_XY(q); t.apply_H_XY(q); }},
        {"sqrt_X", [](S& s, PauliTable& t, std::size_t q) { s.apply_sqrt_X(q); t.apply_sqrt_X(q); }},
        {"sqrt_Y", [](S& s, PauliTable& t, std::size_t q) { s.apply_sqrt_Y(q); t.apply_sqrt_Y(q); }},
        {"X",      [](S& s, PauliTable& t, std::size_t q) { s.apply_X(q); t.apply_X(q); }},
        {"Y",      [](S& s, PauliTable& t, std::size_t q) { s.apply_Y(q); t.apply_Y(q); }},
        {"Z",      [](S& s, PauliTable& t, std::size_t q) { s.apply_Z(q); t.apply_Z(q); }},
    };
    using G2 = std::function<void(S&, PauliTable&, std::size_t, std::size_t)>;
    const std::vector<std::pair<std::string, G2>> pair = {
        {"CNOT", [](S& s, PauliTable& t, std::size_t c, std::size_t u) { s.apply_CNOT(c, u); t.apply_CNOT(c, u); }},
        {"CZ",   [](S& s, PauliTable& t, std::size_t c, std::size_t u) { s.apply_CZ(c, u); t.apply_CZ(c, u); }},
        {"SWAP", [](S& s, PauliTable& t, std::size_t c, std::size_t u) { s.apply_SWAP(c, u); t.apply_SWAP(c, u); }},
    };
    const std::size_t n = 100, m = 150;
    for (const auto& [name, g] : single)
        for (std::size_t q : {0, 37, 63, 64, 99}) {
            auto s = random_sum<128>(m, n, q + 7);
            PauliTable t = to_table(s);
            g(s, t, q);
            expect_same(to_sum<128, double>(t), s);
            if (HasFatalFailure()) { ADD_FAILURE() << name << " q=" << q; return; }
        }
    // Same word (3, 40), different words (5, 70) and (90, 2).
    for (const auto& [name, g] : pair)
        for (auto [c, u] : std::vector<std::pair<std::size_t, std::size_t>>{{3, 40}, {40, 3}, {5, 70}, {90, 2}}) {
            auto s = random_sum<128>(m, n, c * 100 + u);
            PauliTable t = to_table(s);
            g(s, t, c, u);
            expect_same(to_sum<128, double>(t), s);
            if (HasFatalFailure()) { ADD_FAILURE() << name << " " << c << "," << u; return; }
        }
}

TEST(PauliTable, ApplyCircuitMatchesPauliSum) {
    std::mt19937_64 rng(3);
    for (std::size_t n : {2, 17, 70}) {
        const auto c = random_clifford(n, rng);
        auto s = random_sum<128>(130, n, n);
        PauliTable t = to_table(s);
        apply_circuit(s, c);
        apply_circuit(t, c);
        expect_same(to_sum<128, double>(t), s);
    }
}

TEST(PauliTable, AnticommutesMaskMatchesPauliSum) {
    for (std::size_t n : {3, 64, 100}) {
        const auto s = random_sum<128>(200, n, n + 1);
        const auto p = random_pauli_sum<128, double>(1, n + 2, n).get_pauli_string(0);
        const PauliTable t = to_table(s);
        const auto bits = anticommutes_mask(t, p);
        const auto ref = anticommutes_mask(s, p);
        for (std::size_t j = 0; j < s.get_length(); ++j) ASSERT_EQ(has(bits, j), ref[j] == 1) << j;
    }
}

TEST(PauliTable, LeftMultiplyMatchesPauliString) {
    std::mt19937_64 rng(4);
    for (std::size_t n : {1, 5, 64, 100}) {
        auto s = random_sum<128>(150, n, 3 * n);
        auto p = random_sum<128>(1, n, 3 * n + 1).get_pauli_string(0);
        PauliTable t = to_table(s);
        const auto mask = random_mask(s.get_length(), rng);
        t.left_multiply(mask, p);
        for (std::size_t j = 0; j < s.get_length(); ++j) {
            const auto before = s.get_pauli_string(j);
            const auto expect = has(mask, j) ? p * before : before;
            ASSERT_EQ(t.get<128>(j), expect) << "n=" << n << " j=" << j;
        }
    }
}

TEST(PauliTable, LeftMultiplyByTerm) {
    auto s = random_sum<64>(100, 20, 5);
    PauliTable t = to_table(s);
    std::mt19937_64 rng(6);
    auto mask = random_mask(100, rng);
    mask[0] &= ~std::size_t(8);  // source term 3 not in the mask
    t.left_multiply(mask, 3);
    const auto src = s.get_pauli_string(3);
    for (std::size_t j = 0; j < 100; ++j) {
        const auto before = s.get_pauli_string(j);
        ASSERT_EQ(t.get<64>(j), has(mask, j) ? src * before : before) << j;
    }
}

TEST(PauliTable, ProductMatchesSequentialMultiply) {
    std::mt19937_64 rng(7);
    for (std::size_t n : {1, 3, 64, 100})
        for (std::size_t m : {1, 2, 64, 65, 200}) {
            const auto s = random_sum<128>(m, n, n * 1000 + m);
            const PauliTable t = to_table(s);
            const auto mask = random_mask(m, rng);
            PauliString<128> expect{};
            expect.num_qubits_ = static_cast<uint32_t>(n);
            for (std::size_t j = 0; j < m; ++j)
                if (has(mask, j)) expect.multiply(s.get_pauli_string(j));
            std::vector<std::size_t> px, pz;
            const uint8_t k = t.product(mask, px, pz);
            PauliString<128> got{};
            got.num_qubits_ = static_cast<uint32_t>(n);
            std::copy(px.begin(), px.end(), got.x.begin());
            std::copy(pz.begin(), pz.end(), got.z.begin());
            got.flags = k;
            ASSERT_EQ(got, expect) << "n=" << n << " m=" << m;
        }
}

TEST(PauliTable, TwoQubitGateOnOneQubitThrows) {
    PauliTable t(4, 4);
    EXPECT_THROW(t.apply_CNOT(1, 1), std::invalid_argument);
}

// Above detail::kTableParallelWords the gates run with OpenMP threads; the
// test lowers the threshold to reach that path with a small table.
TEST(PauliTable, ThreadedGatePath) {
    const std::size_t saved = detail::kTableParallelWords;
    detail::kTableParallelWords = 4;
    const std::size_t m = 64 * 64 + 5;
    auto s = random_sum<64>(m, 64, 8);
    PauliTable t = to_table(s);
    s.apply_H(3);        t.apply_H(3);
    s.apply_CNOT(3, 60); t.apply_CNOT(3, 60);
    s.apply_S(60);       t.apply_S(60);
    detail::kTableParallelWords = saved;
    expect_same(to_sum<64, double>(t), s);
}
