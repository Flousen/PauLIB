#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "PauLib.h"

using namespace paulib;

namespace {

// Term-by-term equality of two sums, including order.
template <std::size_t N>
void expect_sum_eq(const PauliSum<N, double>& a, const PauliSum<N, double>& b) {
    ASSERT_EQ(a.get_length(), b.get_length());
    for (std::size_t t = 0; t < a.get_length(); ++t) {
        for (std::size_t k = 0; k < PauliSum<N, double>::ArraySize; ++k) {
            EXPECT_EQ(a.x_data[k][t], b.x_data[k][t]) << "x_data[" << k << "][" << t << "]";
            EXPECT_EQ(a.z_data[k][t], b.z_data[k][t]) << "z_data[" << k << "][" << t << "]";
        }
        EXPECT_EQ(a.flags[t], b.flags[t]) << "flags[" << t << "]";
        EXPECT_DOUBLE_EQ(a.value[t], b.value[t]) << "value[" << t << "]";
    }
}

// Random sum with varying, signed, non-unit coefficients (random_pauli_sum
// alone sets every coefficient to 1, which would hide value-indexing bugs).
template <std::size_t N>
PauliSum<N, double> random_sum_with_coeffs(std::size_t terms, std::uint64_t seed) {
    auto s = random_pauli_sum<N, double>(terms, seed);
    for (std::size_t i = 0; i < s.get_length(); ++i)
        s.value[i] = (i % 2 ? -1.0 : 1.0) * (0.5 + static_cast<double>(i));
    return s;
}

} // namespace


// ---------------------------------------------------------------------------
// Construction from PauliSum
// ---------------------------------------------------------------------------

TEST(PauliSumConstruct, FromPauliSumPreservesLength) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{{"XX", 1.0}, {"ZZ", -0.5}, {"YY", 0.25}});
    EXPECT_EQ(a.get_length(), 3u);
}

TEST(PauliSumConstruct, FromPauliSumPreservesCoefficients) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{"XX", 1.0}, {"ZZ", -0.5}, {"YY", 0.25}});

    EXPECT_DOUBLE_EQ(s.value[0],  1.0);
    EXPECT_DOUBLE_EQ(s.value[1], -0.5);
    EXPECT_DOUBLE_EQ(s.value[2],  0.25);
}

// ---------------------------------------------------------------------------
// Multiplication (operator*=)
// ---------------------------------------------------------------------------

// I * I = I, coefficient product
TEST(PauliSumMul, IdentityTimesIdentity) {
    PauliSum<2, double> lhs(std::vector<std::pair<std::string, double>>{{"II", 2.0}});
    PauliSum<2, double> rhs(std::vector<std::pair<std::string, double>>{{"II", 3.0}});

    lhs *= rhs;
    ASSERT_EQ(lhs.get_length(), 1u);
    EXPECT_DOUBLE_EQ(lhs.value[0], 6.0);
    for (std::size_t k = 0; k < PauliSum<4, double>::ArraySize; ++k) {
        EXPECT_EQ(lhs.x_data[k][0], 0u);
        EXPECT_EQ(lhs.z_data[k][0], 0u);
    }
    EXPECT_EQ(lhs.flags[0], 0u);
}

// X * X = I (no phase)
TEST(PauliSumMul, XTimesX_IsIdentity) {
    PauliSum<1, double> lhs(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    PauliSum<1, double> rhs(std::vector<std::pair<std::string, double>>{{"X", 1.0}});

    lhs *= rhs;
    ASSERT_EQ(lhs.get_length(), 1u);
    EXPECT_DOUBLE_EQ(lhs.value[0], 1.0);
    for (std::size_t k = 0; k < PauliSum<4, double>::ArraySize; ++k) {
        EXPECT_EQ(lhs.x_data[k][0], 0u);
        EXPECT_EQ(lhs.z_data[k][0], 0u);
    }
    EXPECT_EQ(lhs.flags[0] & 0x01, 0);  // no sign
}

// Z * Z = I, coefficient preserved
TEST(PauliSumMul, ZTimesZ_IsIdentity) {
    PauliSum<1, double> lhs(std::vector<std::pair<std::string, double>>{{"Z", 1.5}});
    PauliSum<1, double> rhs(std::vector<std::pair<std::string, double>>{{"Z", 1.5}});

    lhs *= rhs;
    ASSERT_EQ(lhs.get_length(), 1u);
    EXPECT_DOUBLE_EQ(lhs.value[0], 2.25);
    for (std::size_t k = 0; k < PauliSum<1, double>::ArraySize; ++k) {
        EXPECT_EQ(lhs.x_data[k][0], 0u);
        EXPECT_EQ(lhs.z_data[k][0], 0u);
    }
    EXPECT_EQ(lhs.flags[0] & 0x01, 0);
}

// X * Y = iZ  => imaginary phase flag set
TEST(PauliSumMul, XTimesY_IsIZ) {
    PauliSum<1, double> lhs(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    PauliSum<1, double> rhs(std::vector<std::pair<std::string, double>>{{"Y", 1.0}});

    lhs *= rhs;
    ASSERT_EQ(lhs.get_length(), 1u);
    EXPECT_DOUBLE_EQ(lhs.value[0], 1.0);

    // Result Pauli: Z (x=0, z=1 on qubit 0)
    EXPECT_EQ(lhs.x_data[0][0] & 1u, 0u);
    EXPECT_EQ(lhs.z_data[0][0] & 1u, 1u);

    // Phase +i: imaginary flag set, sign clear (bit1=1, bit0=0)
    EXPECT_EQ(lhs.flags[0], 0x01);
}

// Y * X = -iZ => both imaginary and sign flags set
TEST(PauliSumMul, YTimesX_IsMinusIZ) {
    PauliSum<1, double> lhs(std::vector<std::pair<std::string, double>>{{"Y", 1.0}});
    PauliSum<1, double> rhs(std::vector<std::pair<std::string, double>>{{"X", 1.0}});

    lhs *= rhs;
    ASSERT_EQ(lhs.get_length(), 1u);
    EXPECT_DOUBLE_EQ(lhs.value[0], 1.0);

    // Result Pauli: Z
    EXPECT_EQ(lhs.x_data[0][0] & 1u, 0u);
    EXPECT_EQ(lhs.z_data[0][0] & 1u, 1u);

    // Phase -i: both flags set
    EXPECT_EQ(lhs.flags[0] & 0x02, 0x02);
    EXPECT_EQ(lhs.flags[0] & 0x01, 0x01);
}

// Cartesian product size: n * m
TEST(PauliSumMul, CartesianProductSize) {
    PauliSum<2, double> lhs(std::vector<std::pair<std::string, double>>{{"XX", 1.0}, {"ZZ", 1.0}, {"YY", 1.0}});
    PauliSum<2, double> rhs(std::vector<std::pair<std::string, double>>{{"XX", 1.0}, {"ZZ", 1.0}});

    lhs *= rhs;
    EXPECT_EQ(lhs.get_length(), 6u);  // 3 * 2
}

// Coefficient product is correct across Cartesian product
TEST(PauliSumMul, CoefficientProduct) {
    PauliSum<2, double> lhs(std::vector<std::pair<std::string, double>>{{"XX", 2.0}, {"ZZ", 3.0}});
    PauliSum<2, double> rhs(std::vector<std::pair<std::string, double>>{{"YY", 4.0}});

    lhs *= rhs;
    ASSERT_EQ(lhs.get_length(), 2u);

    std::vector<double> vals = {lhs.value[0], lhs.value[1]};
    std::sort(vals.begin(), vals.end());
    EXPECT_DOUBLE_EQ(vals[0], 8.0);
    EXPECT_DOUBLE_EQ(vals[1], 12.0);
}

// Y * Y = I (no phase)
TEST(PauliSumMul, YTimesY_IsIdentity) {
    PauliSum<1, double> lhs(std::vector<std::pair<std::string, double>>{{"Y", 1.0}});
    PauliSum<1, double> rhs(std::vector<std::pair<std::string, double>>{{"Y", 1.0}});

    lhs *= rhs;
    ASSERT_EQ(lhs.get_length(), 1u);
    for (std::size_t k = 0; k < PauliSum<4, double>::ArraySize; ++k) {
        EXPECT_EQ(lhs.x_data[k][0], 0u);
        EXPECT_EQ(lhs.z_data[k][0], 0u);
    }
    EXPECT_EQ(lhs.flags[0] & 0x01, 0);
}

// Multi-qubit: consistent with scalar PauliString multiplication
TEST(PauliSumMul, MatchesScalarMultiplication) {
    // Compute reference using scalar PauliString *=
    PauliString<4> ref_lhs = PauliString<4>("XYZX");
    PauliString<4> ref_rhs = PauliString<4>("ZYXZ");
    ref_lhs *= ref_rhs;

    PauliSum<4, double> lhs(std::vector<std::pair<std::string, double>>{{"XYZX", 1.0}});
    PauliSum<4, double> rhs(std::vector<std::pair<std::string, double>>{{"ZYXZ", 1.0}});
    lhs *= rhs;

    ASSERT_EQ(lhs.get_length(), 1u);
    for (std::size_t k = 0; k < PauliSum<4, double>::ArraySize; ++k) {
        EXPECT_EQ(lhs.x_data[k][0], ref_lhs.x[k]);
        EXPECT_EQ(lhs.z_data[k][0], ref_lhs.z[k]);
    }
    EXPECT_EQ(lhs.flags[0], ref_lhs.flags);
}

// ---------------------------------------------------------------------------
// apply_RZ
// ---------------------------------------------------------------------------

// Z commutes with Z_c → term unchanged
TEST(PauliSumApplyRZ, CommutingTermUnchanged) {
    const double angle = 0.7;
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"Z", 1.5}});
    s.apply_RZ(0, angle);
    ASSERT_EQ(s.get_length(), 1u);
    EXPECT_DOUBLE_EQ(s.value[0], 1.5);
}

// I commutes with Z_c → term unchanged
TEST(PauliSumApplyRZ, IdentityTermUnchanged) {
    const double angle = 1.2;
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"I", 2.0}});
    s.apply_RZ(0, angle);
    ASSERT_EQ(s.get_length(), 1u);
    EXPECT_DOUBLE_EQ(s.value[0], 2.0);
}

// X anti-commutes with Z_c → splits into cos*X + sin*(iZ*X) = cos*X - sin*Y
TEST(PauliSumApplyRZ, AntiCommutingXSplitsIntoTwo) {
    const double angle = M_PI / 3.0;
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    s.apply_RZ(0, angle);
    ASSERT_EQ(s.get_length(), 2u);
    EXPECT_DOUBLE_EQ(s.value[0], std::cos(angle / 2));
    EXPECT_DOUBLE_EQ(s.value[1], std::sin(angle / 2));
}

// Resulting Paulis: RZ splits X into cos*X + sin*(iZ*X) = cos*X - sin*Y
TEST(PauliSumApplyRZ, ResultPaulis) {
    const double angle = M_PI / 3.0;
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    s.apply_RZ(0, angle);
    EXPECT_EQ(s.get_str(),
        "0.866025 * X\n"
        "0.500000 * -Y\n");
}

// Regression: apply_RZ previously indexed the scratch sum with the outer loop
// index, writing out of bounds when a commuting term precedes an
// anti-commuting one.
TEST(PauliSumApplyRZ, MixedCommutingAndAntiCommuting) {
    const double angle = M_PI / 3.0;
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{
        {"Z", 2.0}, {"X", 1.0}});
    s.apply_RZ(0, angle);
    EXPECT_EQ(s.get_str(),
        "2.000000 * Z\n"
        "0.866025 * X\n"
        "0.500000 * -Y\n");
}

// RX splits Z into cos*Z + sin*(iX*Z) = cos*Z + sin*Y
TEST(PauliSumApplyRX, ResultPaulis) {
    const double angle = M_PI / 3.0;
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"Z", 1.0}});
    s.apply_RX(0, angle);
    EXPECT_EQ(s.get_str(),
        "0.866025 * Z\n"
        "0.500000 * Y\n");
}

// RY splits Z into cos*Z + sin*(iY*Z) = cos*Z - sin*X
// and X into cos*X + sin*(iY*X) = cos*X + sin*Z
TEST(PauliSumApplyRY, ResultPaulis) {
    const double angle = M_PI / 3.0;
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"Z", 1.0}});
    s.apply_RY(0, angle);
    EXPECT_EQ(s.get_str(),
        "0.866025 * Z\n"
        "0.500000 * -X\n");

    PauliSum<1, double> t(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    t.apply_RY(0, angle);
    EXPECT_EQ(t.get_str(),
        "0.500000 * Z\n"
        "0.866025 * X\n");
}

// ---------------------------------------------------------------------------
// apply_H
// ---------------------------------------------------------------------------

// H: X → Z, Z → X, Y → -Y, I → I
TEST(PauliSumApplyH, XBecomesZ) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    s.apply_H(0);
    EXPECT_EQ(s.get_str(), "1.000000 * Z\n");
}

TEST(PauliSumApplyH, ZBecomesX) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"Z", 1.0}});
    s.apply_H(0);
    EXPECT_EQ(s.get_str(), "1.000000 * X\n");
}

TEST(PauliSumApplyH, YBecomesMinusY) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"Y", 1.0}});
    s.apply_H(0);
    EXPECT_EQ(s.get_str(), "1.000000 * -Y\n");
}

TEST(PauliSumApplyH, IStaysI) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"I", 1.0}});
    s.apply_H(0);
    EXPECT_EQ(s.get_str(), "1.000000 * I\n");
}

// Mirrors PauliStringTests.apply_H_XZ: sequential gates on "IZXY"
TEST(PauliSumApplyH, SequentialMatchesPauliString) {
    PauliSum<4, double> s(std::vector<std::pair<std::string, double>>{{"IZXY", 1.0}});
    s.apply_H(0);
    EXPECT_EQ(s.get_str(), "1.000000 * IZXY\n");
    s.apply_H(1);
    EXPECT_EQ(s.get_str(), "1.000000 * IXXY\n");
    s.apply_H(2);
    EXPECT_EQ(s.get_str(), "1.000000 * IXZY\n");
    s.apply_H(3);
    EXPECT_EQ(s.get_str(), "1.000000 * -IXZY\n");
}

// All terms in a multi-term sum are transformed
TEST(PauliSumApplyH, MultipleTermsAllTransformed) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{
        {"X", 2.0}, {"Z", 3.0}, {"Y", -1.0}, {"I", 0.5}});
    s.apply_H(0);
    EXPECT_EQ(s.get_str(),
        "2.000000 * Z\n"
        "3.000000 * X\n"
        "-1.000000 * -Y\n"
        "0.500000 * I\n");
}

// ---------------------------------------------------------------------------
// Runtime qubit count (num_qubits_ < capacity N)
// ---------------------------------------------------------------------------

TEST(PauliSumNumQubits, InferredFromStrings) {
    PauliSum<64, double> s(std::vector<std::pair<std::string, double>>{
        {"XYZ", 1.0}, {"ZZI", 0.5}});
    EXPECT_EQ(s.num_qubits(), 3u);
    EXPECT_EQ(s.get_str(),
        "1.000000 * XYZ\n"
        "0.500000 * ZZI\n");
    EXPECT_EQ(s.get_pauli_string(0).num_qubits(), 3u);
    EXPECT_EQ(s.get_pauli_string(0).to_string(), "XYZ");
}

TEST(PauliSumNumQubits, SurvivesOperations) {
    PauliSum<64, double> a(std::vector<std::pair<std::string, double>>{{"XY", 2.0}});
    PauliSum<64, double> b(std::vector<std::pair<std::string, double>>{{"ZZ", 3.0}});
    a *= b;
    EXPECT_EQ(a.num_qubits(), 2u);
    EXPECT_EQ(a.get_str(), "6.000000 * YX\n");

    a.apply_RZ(0, 0.5);
    EXPECT_EQ(a.num_qubits(), 2u);
}

TEST(PauliSumNumQubits, ExplicitQubitConstructor) {
    PauliSum<64, double> s(10, "XX", 1.0);  // 2-qubit string in a 10-qubit sum
    EXPECT_EQ(s.num_qubits(), 10u);
    EXPECT_THROW((PauliSum<64, double>(65, "XX", 1.0)), std::invalid_argument);
}

TEST(PauliStringNumQubits, InferredAndBounded) {
    PauliString<64> ps("XYZ");
    EXPECT_EQ(ps.num_qubits(), 3u);
    EXPECT_EQ(ps.to_string(), "XYZ");
    EXPECT_THROW(ps.set_pauli(3, 'X'), std::out_of_range);

    const std::string too_long(65, 'X');
    EXPECT_THROW(PauliString<64>{too_long}, std::invalid_argument);
}

TEST(PauliSumNumQubits, RandomSumMasksPaddingBits) {
    auto s = random_pauli_sum<64, double>(50, 123, 10);
    EXPECT_EQ(s.num_qubits(), 10u);
    for (std::size_t i = 0; i < s.get_length(); ++i) {
        EXPECT_EQ(s.x_data[0][i] >> 10, 0u);
        EXPECT_EQ(s.z_data[0][i] >> 10, 0u);
    }
}

// Regression: sum gates previously mis-indexed words for qubit >= 8.
TEST(PauliSumApplyH, HighQubitIndex) {
    PauliSum<16, double> s(std::vector<std::pair<std::string, double>>{
        {"IIIIIIIIXIIIIIII", 1.0}});
    s.apply_H(8);
    EXPECT_EQ(s.get_str(), "1.000000 * IIIIIIIIZIIIIIII\n");
}

TEST(PauliSumMul, PauliMul) {
    std::map<std::string, std::map<std::string, std::string>> pauliCayley = { 
        {"I", {{"I","I"}, {"X",  "X"}, {"Y",  "Y"}, {"Z",  "Z"}}},
        {"X", {{"I","X"}, {"X",  "I"}, {"Y", "iZ"}, {"Z","-iY"}}}, 
        {"Y", {{"I","Y"}, {"X","-iZ"}, {"Y",  "I"}, {"Z", "iX"}}}, 
        {"Z", {{"I","Z"}, {"X", "iY"}, {"Y","-iX"}, {"Z",  "I"}}}
    };

    for (const auto& [row, innerMap] : pauliCayley) {
        for (const auto& [col, result] : innerMap) {
            PauliSum<1, double> lhs(std::vector<std::pair<std::string, double>>{{row, 1.0}});
            PauliSum<1, double> rhs(std::vector<std::pair<std::string, double>>{{col, 1.0}});
            lhs *= rhs;
            EXPECT_EQ(lhs.get_str(), "1.000000 * " + result + "\n")
                << "Failed for input Pauli: " << row << " * " << col << " = " << result;
        }
    }
}

// ---------------------------------------------------------------------------
// multiply: the outer product must match per-term scalar PauliString
// multiplication, in result[i*n + j] = lhs[j] * rhs[i] order.
// ---------------------------------------------------------------------------

template <std::size_t N>
void check_multiply(std::size_t n, std::size_t m) {
    const auto lhs = random_sum_with_coeffs<N>(n, 42);
    const auto rhs = random_sum_with_coeffs<N>(m, 99);

    auto ref = lhs;
    ref.multiply(rhs);

    ASSERT_EQ(ref.get_length(), n * m);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto ps = lhs.get_pauli_string(j);
            ps.multiply(rhs.get_pauli_string(i));
            const std::size_t t = i * n + j;
            SCOPED_TRACE("term (" + std::to_string(i) + ", " + std::to_string(j) + ")");
            for (std::size_t k = 0; k < PauliSum<N, double>::ArraySize; ++k) {
                EXPECT_EQ(ref.x_data[k][t], ps.x[k]);
                EXPECT_EQ(ref.z_data[k][t], ps.z[k]);
            }
            EXPECT_EQ(ref.flags[t], ps.flags);
            EXPECT_DOUBLE_EQ(ref.value[t], lhs.value[j] * rhs.value[i]);
        }
    }
}

// 13 and 7 are not multiples of any SIMD lane width (2/4/8), so the
// vectorised blocks and their scalar remainder tails are both exercised.
TEST(PauliSumMultiply, AgreeSingleWord) { check_multiply<64>(13, 7); }

// Two 64-bit words per string: exercises the cross-word phase accumulation.
TEST(PauliSumMultiply, AgreeTwoWords) { check_multiply<128>(13, 7); }

// Fewer LHS terms than any SIMD lane count: pure scalar-tail path.
TEST(PauliSumMultiply, AgreeFewerTermsThanSimdLanes) { check_multiply<64>(3, 2); }

TEST(PauliSumMultiply, EmptyOperandGivesEmptyResult) {
    const auto some = random_sum_with_coeffs<64>(5, 1);
    const PauliSum<64, double> empty;

    auto a = some; a.multiply(empty);     EXPECT_EQ(a.get_length(), 0u);
    auto e = empty; e.multiply(some);     EXPECT_EQ(e.get_length(), 0u);
}

TEST(PauliSumMultiply, PreserveNumQubits) {
    auto lhs = random_pauli_sum<64, double>(4, 3, 10);  // 10 logical qubits
    auto rhs = random_pauli_sum<64, double>(4, 4, 10);
    lhs.multiply(rhs);
    EXPECT_EQ(lhs.num_qubits(), 10u);
}

// ---------------------------------------------------------------------------
// elementwise_multiply: element-wise product result[i] = self[i] * rhs[i]
// ---------------------------------------------------------------------------

TEST(PauliSumElementwiseMultiply, MatchesScalarPerTerm) {
    const auto orig = random_sum_with_coeffs<128>(9, 7);
    const auto rhs  = random_sum_with_coeffs<128>(9, 13);

    auto result = orig;
    result.elementwise_multiply(rhs);

    ASSERT_EQ(result.get_length(), 9u);
    for (std::size_t i = 0; i < 9; ++i) {
        auto ps = orig.get_pauli_string(i);
        ps.multiply(rhs.get_pauli_string(i));
        SCOPED_TRACE("term " + std::to_string(i));
        for (std::size_t k = 0; k < PauliSum<128, double>::ArraySize; ++k) {
            EXPECT_EQ(result.x_data[k][i], ps.x[k]);
            EXPECT_EQ(result.z_data[k][i], ps.z[k]);
        }
        EXPECT_EQ(result.flags[i], ps.flags);
        EXPECT_DOUBLE_EQ(result.value[i], orig.value[i] * rhs.value[i]);
    }
}

TEST(PauliSumElementwiseMultiply, SingleQubitCayley) {
    // {X,Y,Z,I} .* {Y,Z,X,I} = {iZ, iX, iY, I} element-wise
    PauliSum<1, double> a(std::vector<std::pair<std::string, double>>{
        {"X", 1.0}, {"Y", 2.0}, {"Z", 3.0}, {"I", 4.0}});
    PauliSum<1, double> b(std::vector<std::pair<std::string, double>>{
        {"Y", 1.0}, {"Z", 1.0}, {"X", 1.0}, {"I", 1.0}});
    a.elementwise_multiply(b);
    EXPECT_EQ(a.get_str(),
        "1.000000 * iZ\n"
        "2.000000 * iX\n"
        "3.000000 * iY\n"
        "4.000000 * I\n");
}

// ---------------------------------------------------------------------------
// sort_and_combine
// ---------------------------------------------------------------------------

TEST(PauliSumSortCombine, MergesDuplicatesAndSumsCoefficients) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"XY", 2.0}, {"ZZ", 1.0}, {"XY", 3.0}});
    s.sort_and_combine();
    EXPECT_EQ(s.get_str(),
        "1.000000 * ZZ\n"
        "5.000000 * XY\n");
}

// Documented behaviour: the phase/sign flags are part of the term key, so
// +X and -X remain separate terms rather than folding into the coefficient.
TEST(PauliSumSortCombine, TermsDifferingInSignStaySeparate) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{
        {"X", 2.0}, {"-X", 3.0}});
    s.sort_and_combine();
    EXPECT_EQ(s.get_str(),
        "2.000000 * X\n"
        "3.000000 * -X\n");
}

// Documented behaviour: exact cancellation keeps the term with coefficient 0
// (no zero-pruning).
TEST(PauliSumSortCombine, ExactCancellationLeavesZeroTerm) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{
        {"X", 2.0}, {"X", -2.0}});
    s.sort_and_combine();
    ASSERT_EQ(s.get_length(), 1u);
    EXPECT_DOUBLE_EQ(s.value[0], 0.0);
}

TEST(PauliSumSortCombine, EmptyAndSingleTerm) {
    PauliSum<1, double> empty;
    empty.sort_and_combine();
    EXPECT_EQ(empty.get_length(), 0u);

    PauliSum<1, double> one(std::vector<std::pair<std::string, double>>{{"Y", 1.5}});
    one.sort_and_combine();
    EXPECT_EQ(one.get_str(), "1.500000 * Y\n");
}

TEST(PauliSumSortCombine, Idempotent) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"XY", 1.0}, {"II", 4.0}, {"XY", 1.0}, {"ZZ", 2.0}, {"II", -1.0}});
    s.sort_and_combine();
    const auto once = s;
    s.sort_and_combine();
    expect_sum_eq<2>(s, once);
}

// ---------------------------------------------------------------------------
// operator+ / operator+=
// ---------------------------------------------------------------------------

TEST(PauliSumAdd, DisjointTermsConcatenateSorted) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{{"XX", 1.0}});
    PauliSum<2, double> b(std::vector<std::pair<std::string, double>>{{"ZZ", 2.0}});
    a += b;
    EXPECT_EQ(a.get_str(),
        "2.000000 * ZZ\n"
        "1.000000 * XX\n");
    EXPECT_EQ(a.num_qubits(), 2u);
}

TEST(PauliSumAdd, OverlappingTermsCombine) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{
        {"XX", 1.0}, {"ZZ", 2.0}});
    PauliSum<2, double> b(std::vector<std::pair<std::string, double>>{{"XX", 3.0}});

    const auto c = a + b;
    EXPECT_EQ(c.get_str(),
        "2.000000 * ZZ\n"
        "4.000000 * XX\n");

    // operator+ must not modify its operands.
    ASSERT_EQ(a.get_length(), 2u);
    EXPECT_DOUBLE_EQ(a.value[0], 1.0);
    ASSERT_EQ(b.get_length(), 1u);
    EXPECT_DOUBLE_EQ(b.value[0], 3.0);
}

TEST(PauliSumAdd, AddingEmptySumIsIdentity) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{
        {"XY", 1.0}, {"ZI", -2.0}});
    auto expected = a;
    expected.sort_and_combine();

    a += PauliSum<2, double>{};
    expect_sum_eq<2>(a, expected);
}

// ---------------------------------------------------------------------------
// operator- / operator-= / unary operator- / scalar operator* / operator*=
// ---------------------------------------------------------------------------

TEST(PauliSumSubtract, DisjointTermsConcatenateNegatedSorted) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{{"XX", 1.0}});
    PauliSum<2, double> b(std::vector<std::pair<std::string, double>>{{"ZZ", 2.0}});
    a -= b;
    EXPECT_EQ(a.get_str(),
        "-2.000000 * ZZ\n"
        "1.000000 * XX\n");
    EXPECT_EQ(a.num_qubits(), 2u);
}

TEST(PauliSumSubtract, OverlappingTermsCombine) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{
        {"XX", 1.0}, {"ZZ", 2.0}});
    PauliSum<2, double> b(std::vector<std::pair<std::string, double>>{{"XX", 3.0}});

    const auto c = a - b;
    EXPECT_EQ(c.get_str(),
        "2.000000 * ZZ\n"
        "-2.000000 * XX\n");

    // operator- must not modify its operands.
    ASSERT_EQ(a.get_length(), 2u);
    EXPECT_DOUBLE_EQ(a.value[0], 1.0);
    ASSERT_EQ(b.get_length(), 1u);
    EXPECT_DOUBLE_EQ(b.value[0], 3.0);
}

// sort_and_combine (used by operator-) leaves exactly-cancelled terms in
// place with a zero coefficient rather than dropping them (matches
// PauliSumSortCombine.ExactCancellationLeavesZeroTerm).
TEST(PauliSumSubtract, SelfSubtractionZeroesCoefficients) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{
        {"XX", 1.0}, {"ZZ", 2.0}});
    const auto c = a - a;
    ASSERT_EQ(c.get_length(), 2u);
    EXPECT_DOUBLE_EQ(c.value[0], 0.0);
    EXPECT_DOUBLE_EQ(c.value[1], 0.0);
}

TEST(PauliSumUnaryNegate, FlipsAllCoefficients) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{
        {"XX", 1.0}, {"ZZ", -2.5}});
    const auto c = -a;
    EXPECT_EQ(c.get_str(),
        "-1.000000 * XX\n"
        "2.500000 * ZZ\n");
    // Unary operator- must not modify its operand.
    EXPECT_DOUBLE_EQ(a.value[0], 1.0);
    EXPECT_DOUBLE_EQ(a.value[1], -2.5);
}

TEST(PauliSumScalarMultiply, ScalesAllCoefficients) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{
        {"XX", 1.0}, {"ZZ", -2.0}});
    const auto c = a * 3.0;
    EXPECT_EQ(c.get_str(),
        "3.000000 * XX\n"
        "-6.000000 * ZZ\n");
    // operator* must not modify its operand.
    EXPECT_DOUBLE_EQ(a.value[0], 1.0);
}

TEST(PauliSumScalarMultiply, IsCommutative) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{{"XX", 2.0}});
    const auto left = a * 3.0;
    const auto right = 3.0 * a;
    EXPECT_EQ(left.get_str(), right.get_str());
}

TEST(PauliSumScalarMultiply, InPlaceMatchesOutOfPlace) {
    PauliSum<2, double> a(std::vector<std::pair<std::string, double>>{
        {"XX", 1.0}, {"ZZ", -2.0}});
    auto expected = a * 4.0;
    a *= 4.0;
    expect_sum_eq<2>(a, expected);
}

// ---------------------------------------------------------------------------
// zero_state_expectation
// ---------------------------------------------------------------------------

TEST(PauliSumZeroStateExpectation, OnlyIZTermsContribute) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"II", 1.5}, {"ZI", 2.0}, {"ZZ", 4.0}, {"XI", 8.0}, {"IY", 16.0}});
    EXPECT_DOUBLE_EQ(s.zero_state_expectation(), 7.5);
}

TEST(PauliSumZeroStateExpectation, SignFlagNegates) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{
        {"-Z", 3.0}, {"Z", 1.0}});
    EXPECT_DOUBLE_EQ(s.zero_state_expectation(), -2.0);
}

// Terms with an imaginary phase flag would contribute an imaginary value,
// which real-valued T cannot represent; they are skipped.
TEST(PauliSumZeroStateExpectation, ImaginaryPhaseTermsSkipped) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{
        {"iZ", 3.0}, {"-iZ", 5.0}, {"I", 1.0}});
    EXPECT_DOUBLE_EQ(s.zero_state_expectation(), 1.0);
}

TEST(PauliSumZeroStateExpectation, EmptySumIsZero) {
    PauliSum<1, double> s;
    EXPECT_DOUBLE_EQ(s.zero_state_expectation(), 0.0);
}

// ---------------------------------------------------------------------------
// basis_state_expectation
// ---------------------------------------------------------------------------

TEST(PauliSumBasisStateExpectation, AllZeroBitsMatchesZeroStateExpectation) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"II", 1.5}, {"ZI", 2.0}, {"ZZ", 4.0}, {"XI", 8.0}, {"IY", 16.0}});
    EXPECT_DOUBLE_EQ(s.basis_state_expectation("00"), s.zero_state_expectation());
}

TEST(PauliSumBasisStateExpectation, EachOneBitFlipsMatchingZSign) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"ZI", 2.0}, {"IZ", 3.0}, {"ZZ", 4.0}});
    // <10|ZI|10> = -1, <10|IZ|10> = +1, <10|ZZ|10> = -1.
    EXPECT_DOUBLE_EQ(s.basis_state_expectation("10"), -2.0 + 3.0 - 4.0);
    // <01|ZI|01> = +1, <01|IZ|01> = -1, <01|ZZ|01> = -1.
    EXPECT_DOUBLE_EQ(s.basis_state_expectation("01"), 2.0 - 3.0 - 4.0);
    // <11|ZI|11> = -1, <11|IZ|11> = -1, <11|ZZ|11> = +1.
    EXPECT_DOUBLE_EQ(s.basis_state_expectation("11"), -2.0 - 3.0 + 4.0);
}

TEST(PauliSumBasisStateExpectation, XYTermsNeverContribute) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"XI", 8.0}, {"IY", 16.0}});
    EXPECT_DOUBLE_EQ(s.basis_state_expectation("11"), 0.0);
}

TEST(PauliSumBasisStateExpectation, SignFlagNegates) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{
        {"-Z", 3.0}, {"Z", 1.0}});
    EXPECT_DOUBLE_EQ(s.basis_state_expectation("1"), 2.0);
}

TEST(PauliSumBasisStateExpectation, ImaginaryPhaseTermsSkipped) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{
        {"iZ", 3.0}, {"-iZ", 5.0}, {"I", 1.0}});
    EXPECT_DOUBLE_EQ(s.basis_state_expectation("1"), 1.0);
}

TEST(PauliSumBasisStateExpectation, ThrowsOnLengthMismatch) {
    PauliSum<2, double> s(2, "II", 1.0);
    EXPECT_THROW(s.basis_state_expectation("0"), std::invalid_argument);
    EXPECT_THROW(s.basis_state_expectation("000"), std::invalid_argument);
}

TEST(PauliSumBasisStateExpectation, ThrowsOnInvalidCharacter) {
    PauliSum<2, double> s(2, "II", 1.0);
    EXPECT_THROW(s.basis_state_expectation("02"), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// apply_CNOT / apply_S
// ---------------------------------------------------------------------------

// Regression: apply_CNOT previously propagated bits in both directions and
// used a wrong phase formula, giving incorrect results for 12 of the 16
// two-qubit Paulis. Must match the (truth-table-tested) PauliString apply_ZC_X.
TEST(PauliSumApplyCNOT, TwoQubitTruthTable) {
    const std::map<std::string, std::string> cnot_map = {
        {"II", "II"}, {"IX", "IX"}, {"IY", "ZY"}, {"IZ", "ZZ"},
        {"XI", "XX"}, {"XX", "XI"}, {"XY", "YZ"}, {"XZ", "-YY"},
        {"YI", "YX"}, {"YX", "YI"}, {"YY", "-XZ"}, {"YZ", "XY"},
        {"ZI", "ZI"}, {"ZX", "ZX"}, {"ZY", "IY"}, {"ZZ", "IZ"}
    };
    for (const auto& [in, out] : cnot_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_CNOT(0, 1);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

// Regression: the bit update previously masked a 0/1 bool with the target
// mask, which only worked for qubit index 0.
TEST(PauliSumApplyCNOT, HighQubitIndices) {
    PauliSum<16, double> s(std::vector<std::pair<std::string, double>>{
        {"IIIIIIIIXIIIIIII", 1.0}});
    s.apply_CNOT(8, 9);
    EXPECT_EQ(s.get_str(), "1.000000 * IIIIIIIIXXIIIIII\n");

    PauliSum<16, double> t(std::vector<std::pair<std::string, double>>{
        {"IIIIIIIIIZIIIIII", 1.0}});
    t.apply_CNOT(8, 9);
    EXPECT_EQ(t.get_str(), "1.000000 * IIIIIIIIZZIIIIII\n");
}

// Control and target in different 64-bit words.
TEST(PauliSumApplyCNOT, CrossWordControlTarget) {
    std::string in(128, 'I');
    in[0] = 'X';
    PauliSum<128, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
    s.apply_CNOT(0, 70);
    std::string expect(128, 'I');
    expect[0] = 'X';
    expect[70] = 'X';
    EXPECT_EQ(s.get_str(), "1.000000 * " + expect + "\n");
}

// All terms of a multi-term sum are transformed consistently with the
// scalar PauliString gate.
TEST(PauliSumApplyCNOT, MatchesPauliStringOnRandomSum) {
    auto s = random_pauli_sum<64, double>(20, 5, 12);
    auto ref = s;
    s.apply_CNOT(3, 9);
    for (std::size_t i = 0; i < ref.get_length(); ++i) {
        auto ps = ref.get_pauli_string(i);
        ps.apply_ZC_X(3, 9);
        SCOPED_TRACE("term " + std::to_string(i));
        EXPECT_EQ(s.get_pauli_string(i).to_string(), ps.to_string());
    }
}

// S: X → Y, Y → -X, Z → Z, I → I (matches PauliString apply_sqrt_Z).
TEST(PauliSumApplyS, SingleQubitTruthTable) {
    const std::map<std::string, std::string> s_map = {
        {"I", "I"}, {"X", "Y"}, {"Y", "-X"}, {"Z", "Z"}
    };
    for (const auto& [in, out] : s_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_S(0);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

// ---------------------------------------------------------------------------
// evolve — Q → cos(θ)·Q + i·sin(θ)·P·Q for anti-commuting P, Q.
// The θ ∈ {0, ±π/2, π} fast paths must agree with the general path.
// ---------------------------------------------------------------------------

// Regression: θ = 0 previously matched the θ = π branch and flipped the sign.
TEST(PauliSumEvolve, ZeroAngleIsIdentity) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    PauliSum<1, double> rot(std::vector<std::pair<std::string, double>>{{"Z", 0.0}});
    s.evolve(rot);
    EXPECT_EQ(s.get_str(), "1.000000 * X\n");
}

TEST(PauliSumEvolve, PiFlipsSign) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    PauliSum<1, double> rot(std::vector<std::pair<std::string, double>>{{"Z", M_PI}});
    s.evolve(rot);
    EXPECT_EQ(s.get_str(), "1.000000 * -X\n");
}

// θ = π/2: X → i·Z·X = -Y.
TEST(PauliSumEvolve, HalfPi) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    PauliSum<1, double> rot(std::vector<std::pair<std::string, double>>{{"Z", M_PI / 2}});
    s.evolve(rot);
    EXPECT_EQ(s.get_str(), "1.000000 * -Y\n");
}

// Regression: θ = -π/2 previously matched the +π/2 branch. X → -i·Z·X = Y.
TEST(PauliSumEvolve, MinusHalfPi) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    PauliSum<1, double> rot(std::vector<std::pair<std::string, double>>{{"Z", -M_PI / 2}});
    s.evolve(rot);
    EXPECT_EQ(s.get_str(), "1.000000 * Y\n");
}

TEST(PauliSumEvolve, GeneralAngleMatchesTrigExpansion) {
    const double theta = M_PI / 3;
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
    PauliSum<1, double> rot(std::vector<std::pair<std::string, double>>{{"Z", theta}});
    s.evolve(rot);
    // cos(θ)·X + i·sin(θ)·Z·X = cos(θ)·X - sin(θ)·Y
    ASSERT_EQ(s.get_length(), 2u);
    EXPECT_EQ(s.get_pauli_string(0).to_string(), "X");
    EXPECT_DOUBLE_EQ(s.value[0], std::cos(theta));
    EXPECT_EQ(s.get_pauli_string(1).to_string(), "Y");
    EXPECT_DOUBLE_EQ(s.value[1], -std::sin(theta));
}

// Each fast path must be the continuous limit of the general path.
TEST(PauliSumEvolve, FastPathsAgreeWithGeneralPath) {
    const double eps = 1e-7;
    for (double theta : {0.0, M_PI / 2, -M_PI / 2, M_PI}) {
        SCOPED_TRACE("theta = " + std::to_string(theta));
        PauliSum<1, double> fast(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
        PauliSum<1, double> rot_f(std::vector<std::pair<std::string, double>>{{"Z", theta}});
        fast.evolve(rot_f);

        PauliSum<1, double> gen(std::vector<std::pair<std::string, double>>{{"X", 1.0}});
        PauliSum<1, double> rot_g(std::vector<std::pair<std::string, double>>{{"Z", theta + eps}});
        gen.evolve(rot_g);

        // Compare <0|·|0>-independent content: accumulate coefficient · phase
        // per operator string.
        auto signed_coeff = [](const PauliSum<1, double>& s, const std::string& op) {
            double acc = 0.0;
            for (std::size_t i = 0; i < s.get_length(); ++i) {
                auto ps = s.get_pauli_string(i);
                std::string str = ps.to_string();
                const bool neg = str.find('-') != std::string::npos;
                str.erase(0, str.find_first_of("IXYZ"));
                if (str == op) acc += (neg ? -1.0 : 1.0) * s.value[i];
            }
            return acc;
        };
        for (const std::string op : {"X", "Y"}) {
            EXPECT_NEAR(signed_coeff(fast, op), signed_coeff(gen, op), 1e-6)
                << "operator " << op;
        }
    }
}

TEST(PauliSumEvolve, CommutingObservableUnchanged) {
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"Z", 2.0}});
    for (double theta : {0.0, 0.3, M_PI / 2, -M_PI / 2, M_PI}) {
        PauliSum<1, double> rot(std::vector<std::pair<std::string, double>>{{"Z", theta}});
        s.evolve(rot);
    }
    EXPECT_EQ(s.get_str(), "2.000000 * Z\n");
}

// ---------------------------------------------------------------------------
// greedy_group
// ---------------------------------------------------------------------------

namespace {

// A grouping is valid iff every index appears exactly once and all members
// of each group mutually commute. greedy_group chunks the input per OMP
// thread and merges, so the partition depends on the thread count — only
// validity, not an exact partition, can be checked in general.
template <std::size_t N>
void expect_valid_grouping(PauliSum<N, double>& s,
                           const std::vector<std::vector<std::size_t>>& groups) {
    std::vector<int> seen(s.get_length(), 0);
    for (const auto& g : groups)
        for (std::size_t i : g) {
            ASSERT_LT(i, s.get_length());
            ++seen[i];
        }
    for (std::size_t i = 0; i < seen.size(); ++i)
        EXPECT_EQ(seen[i], 1) << "index " << i;

    for (const auto& g : groups)
        for (std::size_t a = 0; a < g.size(); ++a)
            for (std::size_t b = a + 1; b < g.size(); ++b)
                EXPECT_TRUE(s.commute(g[a], g[b]))
                    << "group member " << g[a] << " vs " << g[b];
}

} // namespace

TEST(PauliSumGreedyGroup, DeterministicSmallCase) {
    // XX, YY, ZZ mutually commute; XI anti-commutes with YY (and ZZ).
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"XX", 1.0}, {"YY", 1.0}, {"ZZ", 1.0}, {"XI", 1.0}});
    expect_valid_grouping(s, s.greedy_group());
}

TEST(PauliSumGreedyGroup, ValidPartitionOfRandomSum) {
    auto s = random_pauli_sum<64, double>(120, 23, 16);
    expect_valid_grouping(s, s.greedy_group());
}

// X/Y detection must scan every word, not just the first.
TEST(PauliSumZeroStateExpectation, MultiWordXDetection) {
    std::string with_x(128, 'I');
    with_x[100] = 'X';                 // X in the second 64-bit word
    const std::string all_z(128, 'Z');

    PauliSum<128, double> s(std::vector<std::pair<std::string, double>>{
        {with_x, 5.0}, {all_z, 2.0}});
    EXPECT_DOUBLE_EQ(s.zero_state_expectation(), 2.0);
}


// ---------------------------------------------------------------------------
// trace / trace_product

TEST(PauliSumTrace, IdentityTermsOnly) {
    // Only identity terms contribute: tr = 2^n * (2.5 - 1.0); the imaginary
    // identity term and the non-identity term are traceless / skipped.
    PauliSum<4, double> s(std::vector<std::pair<std::string, double>>{
        {"IIII", 2.5}, {"XIZI", 7.0}, {"-IIII", 1.0}, {"iIIII", 3.0}});
    EXPECT_DOUBLE_EQ(s.trace(), 16.0 * 1.5);
}

TEST(PauliSumTrace, TracelessSum) {
    PauliSum<4, double> s(std::vector<std::pair<std::string, double>>{
        {"XYZI", 1.0}, {"ZZZZ", -3.0}});
    EXPECT_DOUBLE_EQ(s.trace(), 0.0);
}

TEST(PauliSumTraceProduct, PauliOrthogonality) {
    // tr(P_a P_b) = 2^n delta_ab for canonical Pauli strings.
    PauliSum<3, double> a(3, "XYZ", 1.0);
    PauliSum<3, double> b(3, "XYI", 1.0);
    EXPECT_DOUBLE_EQ(trace_product(a, a), 8.0);
    EXPECT_DOUBLE_EQ(trace_product(a, b), 0.0);
}

TEST(PauliSumTraceProduct, QubitMismatchThrows) {
    PauliSum<4, double> a(3, "XYZ", 1.0);
    PauliSum<4, double> b(4, "XYZI", 1.0);
    EXPECT_THROW(trace_product(a, b), std::invalid_argument);
}


// ---------------------------------------------------------------------------
// Truncation & algebra primitives

TEST(PauliSumPrimitives, TermWeight) {
    PauliSum<4, double> s(std::vector<std::pair<std::string, double>>{
        {"IIII", 1.0}, {"IXYZ", 1.0}, {"ZIIZ", 1.0}});
    EXPECT_EQ(s.term_weight(0), 0u);
    EXPECT_EQ(s.term_weight(1), 3u);
    EXPECT_EQ(s.term_weight(2), 2u);
}

TEST(PauliSumPrimitives, CanonicalizeFoldsSignAndCombines) {
    // +X and -X differ only in the sign flag: canonicalize merges them.
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"XI", 1.0}, {"-XI", 0.25}, {"ZI", 2.0}});
    s.canonicalize();
    ASSERT_EQ(s.get_length(), 2u);
    // sorted order: XI has x-bit set, ZI has z-bit; verify via strings
    std::map<std::string, double> terms;
    for (size_t i = 0; i < s.get_length(); ++i)
        terms[s.get_pauli_string(i).to_string()] = s.value[i];
    EXPECT_DOUBLE_EQ(terms.at("XI"), 0.75);
    EXPECT_DOUBLE_EQ(terms.at("ZI"), 2.0);
}

TEST(PauliSumPrimitives, CanonicalizeDropsExactZeros) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"XI", 1.0}, {"-XI", 1.0}});
    s.canonicalize();
    EXPECT_EQ(s.get_length(), 0u);
}

TEST(PauliSumPrimitives, CanonicalizeKeepsImaginaryBit) {
    // -iZ = (-1) * iZ: sign folds into the coefficient, phase bit stays.
    PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{"-iZ", 3.0}});
    s.canonicalize();
    ASSERT_EQ(s.get_length(), 1u);
    EXPECT_EQ(s.get_pauli_string(0).to_string(), "iZ");
    EXPECT_DOUBLE_EQ(s.value[0], -3.0);
}

TEST(PauliSumPrimitives, TrimKeepsLargest) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"XI", 0.1}, {"ZI", -5.0}, {"IZ", 0.3}, {"YI", 2.0}});
    s.trim(2);
    ASSERT_EQ(s.get_length(), 2u);
    std::map<std::string, double> terms;
    for (size_t i = 0; i < s.get_length(); ++i)
        terms[s.get_pauli_string(i).to_string()] = s.value[i];
    EXPECT_DOUBLE_EQ(terms.at("ZI"), -5.0);
    EXPECT_DOUBLE_EQ(terms.at("YI"), 2.0);
}

TEST(PauliSumPrimitives, TrimProtectsKeepPatterns) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{
        {"XI", 0.001}, {"ZI", -5.0}, {"IZ", 0.3}, {"YI", 2.0}});
    PauliSum<2, double> keep(2, "XI", 1.0);
    s.trim(2, keep);
    std::map<std::string, double> terms;
    for (size_t i = 0; i < s.get_length(); ++i)
        terms[s.get_pauli_string(i).to_string()] = s.value[i];
    // XI survives despite its tiny coefficient; remaining budget -> ZI.
    ASSERT_EQ(s.get_length(), 2u);
    EXPECT_DOUBLE_EQ(terms.at("XI"), 0.001);
    EXPECT_DOUBLE_EQ(terms.at("ZI"), -5.0);
}

TEST(PauliSumPrimitives, CutoffAndTruncateWeight) {
    PauliSum<3, double> s(std::vector<std::pair<std::string, double>>{
        {"XII", 0.5}, {"XYZ", 0.01}, {"ZZI", -0.2}});
    s.cutoff(0.05);
    ASSERT_EQ(s.get_length(), 2u);
    s.truncate_weight(1);
    ASSERT_EQ(s.get_length(), 1u);
    EXPECT_EQ(s.get_pauli_string(0).to_string(), "XII");
}

TEST(PauliSumPrimitives, FrobeniusNormMatchesTraceProduct) {
    PauliSum<3, double> s(std::vector<std::pair<std::string, double>>{
        {"XII", 0.5}, {"ZZI", -0.2}, {"XYZ", 1.5}});
    s.canonicalize();
    // For Hermitian O: tr(O†O)/2^n = trace_product(O,O)/2^n.
    EXPECT_NEAR(s.frobenius_norm2(), trace_product(s, s) / 8.0, 1e-14);
}

TEST(PauliSumPrimitives, AppendScaled) {
    PauliSum<2, double> a(2, "XI", 1.0);
    PauliSum<2, double> b(std::vector<std::pair<std::string, double>>{
        {"XI", 2.0}, {"ZI", 4.0}});
    append_scaled(a, 0.5, b);
    a.canonicalize();
    std::map<std::string, double> terms;
    for (size_t i = 0; i < a.get_length(); ++i)
        terms[a.get_pauli_string(i).to_string()] = a.value[i];
    EXPECT_DOUBLE_EQ(terms.at("XI"), 2.0);
    EXPECT_DOUBLE_EQ(terms.at("ZI"), 2.0);
}
