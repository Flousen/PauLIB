#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include <omp.h>

#include "DenseReference.h"
#include "PauLib.h"

using namespace paulib;

namespace {

constexpr double kTol = 1e-12;

TEST(Commutation, MaskMatchesPairwiseCheck) {
    auto s = random_pauli_sum<130, double>(500, 1, 130);
    const PauliString<130> p = random_pauli_sum<130, double>(1, 2, 130).get_pauli_string(0);
    const auto mask = anticommutes_mask(s, p);
    ASSERT_EQ(mask.size(), s.get_length());
    for (std::size_t i = 0; i < s.get_length(); ++i)
        EXPECT_EQ(mask[i], s.get_pauli_string(i).anti_commute(p) ? 1 : 0);
}

TEST(Commutation, CommutesWithAll) {
    PauliSum<8, double> h(std::vector<std::pair<std::string, double>>{
        {"ZZII", 1.0}, {"IZZI", 1.0}, {"IIZZ", 1.0}, {"XXXX", 0.5}});
    EXPECT_TRUE(commutes_with_all(h, PauliString<8>("XXXX")));
    EXPECT_TRUE(commutes_with_all(h, PauliString<8>("ZZZZ")));
    EXPECT_FALSE(commutes_with_all(h, PauliString<8>("XIII")));
}

// The anticommutation matrix is not stored: an entry comes from the sum, a
// row from the mask on the sum or on its qubit-major table.
TEST(Commutation, AnticommutationEntriesAndRows) {
    auto s = random_pauli_sum<70, double>(300, 3, 70);
    const PauliTable t = to_table(s);
    for (std::size_t j = 0; j < s.get_length(); ++j) {
        const auto pj = s.get_pauli_string(j);
        const auto row_sum = anticommutes_mask(s, pj);
        const auto row_table = anticommutes_mask(t, pj);
        for (std::size_t i = 0; i < s.get_length(); ++i) {
            const bool ref = s.get_pauli_string(i).anti_commute(pj);
            ASSERT_EQ(anticommutes(s, i, j), ref) << i << "," << j;
            ASSERT_EQ(row_sum[i] != 0, ref) << i << "," << j;
            ASSERT_EQ(((row_table[i / 64] >> (i % 64)) & 1) != 0, ref) << i << "," << j;
        }
    }
}

// Adjoint channel E^dagger(O) = sum_E p_E E O E against the dense matrices.
dense::Mat adjoint_channel(const dense::Mat& o, std::size_t q, std::size_t n,
                           double px, double py, double pz) {
    auto P = [&](const char* s) {
        std::string str(n, 'I');
        str[q] = s[0];
        return dense::pauli(PauliString<8>(str), n);
    };
    const auto x = P("X"), y = P("Y"), z = P("Z");
    return (1 - px - py - pz) * o + px * x * o * x + py * y * o * y + pz * z * o * z;
}

TEST(Commutation, PauliChannelMatchesDense) {
    auto s = random_pauli_sum<8, double>(40, 4, 3);
    for (auto& f : s.flags) f &= 0x02;
    const auto d0 = dense::sum(s);
    apply_pauli_channel(s, 1, 0.05, 0.1, 0.2);
    EXPECT_LT(dense::max_abs_diff(dense::sum(s), adjoint_channel(d0, 1, 3, 0.05, 0.1, 0.2)), kTol);
}

TEST(Commutation, DepolarizingAndDephasingMatchDense) {
    auto s = random_pauli_sum<8, std::complex<double>>(40, 5, 3);
    const auto d0 = dense::sum(s);
    auto dep = s;
    apply_depolarizing(dep, 2, 0.3);
    // rho -> (1-p) rho + p I/2 equals the Pauli channel with px = py = pz = p/4.
    EXPECT_LT(dense::max_abs_diff(dense::sum(dep), adjoint_channel(d0, 2, 3, 0.075, 0.075, 0.075)), kTol);
    auto deph = s;
    apply_dephasing(deph, 0, 0.3);
    EXPECT_LT(dense::max_abs_diff(dense::sum(deph), adjoint_channel(d0, 0, 3, 0, 0, 0.15)), kTol);
}

TEST(Commutation, DampAnticommutingIsSinglePauliChannel) {
    auto s = random_pauli_sum<8, double>(40, 6, 3);
    for (auto& f : s.flags) f &= 0x02;
    const PauliString<8> e("XZY");
    const auto d0 = dense::sum(s);
    const auto de = dense::pauli(e, 3);
    const double p = 0.15;
    damp_anticommuting(s, e, 1 - 2 * p);
    EXPECT_LT(dense::max_abs_diff(dense::sum(s), (1 - p) * d0 + p * de * d0 * de), kTol);
}

TEST(Commutation, AddNoiseIsDepolarizingOnEveryQubit) {
    auto s = random_pauli_sum<8, double>(60, 7, 4);
    auto ref = s;
    add_noise(s, 0.1);
    for (std::size_t q = 0; q < 4; ++q) apply_depolarizing(ref, q, 0.1);
    for (std::size_t i = 0; i < s.get_length(); ++i) EXPECT_NEAR(s.value[i], ref.value[i], kTol);
}

TEST(Commutation, LargeInputsIndependentOfThreadCount) {
    auto s = random_pauli_sum<128, double>(1 << 16, 8, 100);
    const PauliString<128> p = random_pauli_sum<128, double>(1, 9, 100).get_pauli_string(0);
    auto a = s, b = s;
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto m1 = anticommutes_mask(a, p);
    damp_anticommuting(a, p, 0.5);
    add_noise(a, 0.01);
    omp_set_num_threads(8);
    const auto m8 = anticommutes_mask(b, p);
    damp_anticommuting(b, p, 0.5);
    add_noise(b, 0.01);
    omp_set_num_threads(saved);
    EXPECT_EQ(m1, m8);
    EXPECT_EQ(a.value, b.value);
}

}  // namespace
