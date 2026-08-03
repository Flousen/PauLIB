#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "PauLib.h"
#include "gpu/GpuPauliSum.h"

using namespace paulib;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Term-by-term comparison of two SoAs (assumes same order and length).
template<std::size_t N>
static void expect_soa_eq(const PauliSum<N, double>& a,
                           const PauliSum<N, double>& b)
{
    ASSERT_EQ(a.get_length(), b.get_length());
    constexpr int AS = (int)PauliSum<N, double>::ArraySize;
    for (int t = 0; t < (int)a.get_length(); ++t) {
        for (int k = 0; k < AS; ++k) {
            EXPECT_EQ(a.x_data[k][(size_t)t], b.x_data[k][(size_t)t])
                << "x_data[" << k << "][" << t << "]";
            EXPECT_EQ(a.z_data[k][(size_t)t], b.z_data[k][(size_t)t])
                << "z_data[" << k << "][" << t << "]";
        }
        EXPECT_EQ(a.flags[(size_t)t], b.flags[(size_t)t]) << "flags[" << t << "]";
        EXPECT_DOUBLE_EQ(a.value[(size_t)t], b.value[(size_t)t]) << "value[" << t << "]";
    }
}

// GPU outer product + download, sorted so comparison is order-independent.
template<std::size_t N>
static PauliSum<N, double> gpu_mul_sorted(const PauliSum<N, double>& lhs,
                                              const PauliSum<N, double>& rhs)
{
    GpuPauliSum<N> g_lhs(lhs), g_rhs(rhs);
    auto result = gpu_multiply(g_lhs, g_rhs).download();
    result.sort_and_combine();
    return result;
}

// CPU outer product, sorted.
template<std::size_t N>
static PauliSum<N, double> cpu_mul_sorted(const PauliSum<N, double>& lhs,
                                              const PauliSum<N, double>& rhs)
{
    PauliSum<N, double> result = lhs;
    result.multiply(rhs);
    result.sort_and_combine();
    return result;
}

// ---------------------------------------------------------------------------
// Lifecycle: upload / download / move
// ---------------------------------------------------------------------------

TEST(GpuLifecycle, SizeAfterAlloc)
{
    GpuPauliSum<4> g(42);
    EXPECT_EQ(g.size(), 42);
    EXPECT_NE(g.x_ptr(), nullptr);
}

TEST(GpuLifecycle, UploadDownloadRoundtrip_SingleTerm)
{
    PauliSum<4, double> cpu;
    cpu.push_back(PauliString<4>("XZYX"), 3.14);

    GpuPauliSum<4> g(cpu);
    expect_soa_eq<4>(cpu, g.download());
}

TEST(GpuLifecycle, UploadDownloadRoundtrip_MultipleTerms)
{
    PauliSum<4, double> cpu;
    cpu.push_back(PauliString<4>("XZYX"),  1.0);
    cpu.push_back(PauliString<4>("YYII"), -2.5);
    cpu.push_back(PauliString<4>("ZIZZ"),  0.7);

    GpuPauliSum<4> g(cpu);
    expect_soa_eq<4>(cpu, g.download());
}

// Large enough to stress the pack_words packing (AS=8 for 500 qubits).
TEST(GpuLifecycle, UploadDownloadRoundtrip_500qubit_100terms)
{
    auto cpu = random_pauli_sum<500, double>(100, 42);
    GpuPauliSum<500> g(cpu);
    expect_soa_eq<500>(cpu, g.download());
}

TEST(GpuLifecycle, MoveConstructor_NullifiesSource)
{
    PauliSum<4, double> cpu;
    cpu.push_back(PauliString<4>("XZYX"), 1.0);

    GpuPauliSum<4> a(cpu);
    GpuPauliSum<4> b(std::move(a));

    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.x_ptr(), nullptr);
    EXPECT_EQ(b.size(), 1);
    EXPECT_NE(b.x_ptr(), nullptr);
}

TEST(GpuLifecycle, MoveAssignment_NullifiesSource)
{
    PauliSum<4, double> cpu;
    cpu.push_back(PauliString<4>("XZYX"), 1.0);

    GpuPauliSum<4> a(cpu);
    GpuPauliSum<4> b;
    b = std::move(a);

    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.x_ptr(), nullptr);
    EXPECT_EQ(b.size(), 1);
}

// ---------------------------------------------------------------------------
// Output size
// ---------------------------------------------------------------------------

TEST(GpuMultiplySize, OutputSizeIsNtimesM)
{
    constexpr int n = 7, m = 11;
    auto lhs = random_pauli_sum<4, double>(n, 1);
    auto rhs = random_pauli_sum<4, double>(m, 2);

    GpuPauliSum<4> g_lhs(lhs), g_rhs(rhs);
    auto g_out = gpu_multiply(g_lhs, g_rhs);

    EXPECT_EQ(g_out.size(), n * m);
    EXPECT_EQ((int)g_out.download().get_length(), n * m);
}

// ---------------------------------------------------------------------------
// Single-qubit Pauli products (N=1, AS=1): known results verified against CPU.
// Each test checks both the output Pauli string and phase flag.
// ---------------------------------------------------------------------------

static void single_qubit_check(const std::string& a, const std::string& b)
{
    PauliSum<1, double> lhs, rhs;
    lhs.push_back(PauliString<1>(a), 1.0);
    rhs.push_back(PauliString<1>(b), 1.0);

    auto gpu = gpu_mul_sorted<1>(lhs, rhs);
    auto cpu = cpu_mul_sorted<1>(lhs, rhs);
    expect_soa_eq<1>(gpu, cpu);
}

TEST(GpuMultiplySingleQubit, XX) { single_qubit_check("X", "X"); }  // +I
TEST(GpuMultiplySingleQubit, YY) { single_qubit_check("Y", "Y"); }  // +I
TEST(GpuMultiplySingleQubit, ZZ) { single_qubit_check("Z", "Z"); }  // +I
TEST(GpuMultiplySingleQubit, XY) { single_qubit_check("X", "Y"); }  // +iZ
TEST(GpuMultiplySingleQubit, YX) { single_qubit_check("Y", "X"); }  // -iZ
TEST(GpuMultiplySingleQubit, YZ) { single_qubit_check("Y", "Z"); }  // +iX
TEST(GpuMultiplySingleQubit, ZY) { single_qubit_check("Z", "Y"); }  // -iX
TEST(GpuMultiplySingleQubit, ZX) { single_qubit_check("Z", "X"); }  // +iY
TEST(GpuMultiplySingleQubit, XZ) { single_qubit_check("X", "Z"); }  // -iY
TEST(GpuMultiplySingleQubit, XI) { single_qubit_check("X", "I"); }  // X
TEST(GpuMultiplySingleQubit, IY) { single_qubit_check("I", "Y"); }  // Y

// Explicit flag check: X*Y = iZ → flags encodes one factor of i
TEST(GpuMultiplySingleQubit, XY_FlagIsI)
{
    PauliSum<1, double> lhs, rhs;
    lhs.push_back(PauliString<1>("X"), 1.0);
    rhs.push_back(PauliString<1>("Y"), 1.0);

    GpuPauliSum<1> g_lhs(lhs), g_rhs(rhs);
    auto out = gpu_multiply(g_lhs, g_rhs).download();

    ASSERT_EQ(out.get_length(), 1u);
    // Z: x_bit=0, z_bit=1
    EXPECT_EQ(out.x_data[0][0] & 1u, 0u);
    EXPECT_EQ(out.z_data[0][0] & 1u, 1u);

    // Compare against CPU to validate the exact flag value
    PauliSum<1, double> cpu_lhs = lhs;
    cpu_lhs.multiply(rhs);
    EXPECT_EQ(out.flags[0], cpu_lhs.flags[0]);
}

// X*X = I: output should be the identity (all bits zero, no phase)
TEST(GpuMultiplySingleQubit, XX_IsIdentityExplicit)
{
    PauliSum<1, double> lhs, rhs;
    lhs.push_back(PauliString<1>("X"), 1.0);
    rhs.push_back(PauliString<1>("X"), 1.0);

    GpuPauliSum<1> g_lhs(lhs), g_rhs(rhs);
    auto out = gpu_multiply(g_lhs, g_rhs).download();

    ASSERT_EQ(out.get_length(), 1u);
    EXPECT_EQ(out.x_data[0][0], 0u);
    EXPECT_EQ(out.z_data[0][0], 0u);
    EXPECT_EQ(out.flags[0] & 0x03u, 0u);
    EXPECT_DOUBLE_EQ(out.value[0], 1.0);
}

// ---------------------------------------------------------------------------
// Coefficient correctness
// ---------------------------------------------------------------------------

TEST(GpuMultiplyCoeffs, ProductOfCoefficients)
{
    PauliSum<1, double> lhs, rhs;
    lhs.push_back(PauliString<1>("X"), 2.5);
    rhs.push_back(PauliString<1>("X"), 4.0);

    GpuPauliSum<1> g_lhs(lhs), g_rhs(rhs);
    auto out = gpu_multiply(g_lhs, g_rhs).download();

    ASSERT_EQ(out.get_length(), 1u);
    EXPECT_DOUBLE_EQ(out.value[0], 10.0);
}

TEST(GpuMultiplyCoeffs, NegativeCoefficient)
{
    PauliSum<1, double> lhs, rhs;
    lhs.push_back(PauliString<1>("Z"), -1.0);
    rhs.push_back(PauliString<1>("Z"),  3.0);

    GpuPauliSum<1> g_lhs(lhs), g_rhs(rhs);
    auto out = gpu_multiply(g_lhs, g_rhs).download();

    ASSERT_EQ(out.get_length(), 1u);
    EXPECT_DOUBLE_EQ(out.value[0], -3.0);
}

TEST(GpuMultiplyCoeffs, OuterProductCoeffsMatch)
{
    // lhs = {X:2, Z:3}, rhs = {Y:5, I:7}
    // outer product has 4 terms; coefficients must be 2*5, 2*7, 3*5, 3*7 (in some order)
    PauliSum<1, double> lhs, rhs;
    lhs.push_back(PauliString<1>("X"), 2.0);
    lhs.push_back(PauliString<1>("Z"), 3.0);
    rhs.push_back(PauliString<1>("Y"), 5.0);
    rhs.push_back(PauliString<1>("I"), 7.0);

    auto gpu = gpu_mul_sorted<1>(lhs, rhs);
    auto cpu = cpu_mul_sorted<1>(lhs, rhs);
    expect_soa_eq<1>(gpu, cpu);
}

// ---------------------------------------------------------------------------
// GPU vs CPU: random inputs, sorted comparison (order-independent)
// Tests multiple qubit counts to exercise different AS dispatch paths.
// ---------------------------------------------------------------------------

// AS=1: single 64-bit word per string (N ≤ 64)
TEST(GpuVsCpu, Random_N64_10x10)
{
    auto lhs = random_pauli_sum<64, double>(10, 42);
    auto rhs = random_pauli_sum<64, double>(10, 99);
    expect_soa_eq<64>(gpu_mul_sorted<64>(lhs, rhs), cpu_mul_sorted<64>(lhs, rhs));
}

TEST(GpuVsCpu, Random_N64_20x5)
{
    auto lhs = random_pauli_sum<64, double>(20, 11);
    auto rhs = random_pauli_sum<64, double>( 5, 77);
    expect_soa_eq<64>(gpu_mul_sorted<64>(lhs, rhs), cpu_mul_sorted<64>(lhs, rhs));
}

// AS=2: two words per string (N = 128)
TEST(GpuVsCpu, Random_N128_10x10)
{
    auto lhs = random_pauli_sum<128, double>(10, 42);
    auto rhs = random_pauli_sum<128, double>(10, 99);
    expect_soa_eq<128>(gpu_mul_sorted<128>(lhs, rhs), cpu_mul_sorted<128>(lhs, rhs));
}

// AS=8: matches the benchmark configuration (N = 500)
TEST(GpuVsCpu, Random_N500_10x10)
{
    auto lhs = random_pauli_sum<500, double>(10, 42);
    auto rhs = random_pauli_sum<500, double>(10, 99);
    expect_soa_eq<500>(gpu_mul_sorted<500>(lhs, rhs), cpu_mul_sorted<500>(lhs, rhs));
}

TEST(GpuVsCpu, Random_N500_30x20)
{
    auto lhs = random_pauli_sum<500, double>(30, 17);
    auto rhs = random_pauli_sum<500, double>(20, 31);
    expect_soa_eq<500>(gpu_mul_sorted<500>(lhs, rhs), cpu_mul_sorted<500>(lhs, rhs));
}

// Larger size: more SM occupancy, exercises the coalescing path properly
TEST(GpuVsCpu, Random_N500_100x100)
{
    auto lhs = random_pauli_sum<500, double>(100, 7);
    auto rhs = random_pauli_sum<500, double>(100, 13);
    expect_soa_eq<500>(gpu_mul_sorted<500>(lhs, rhs), cpu_mul_sorted<500>(lhs, rhs));
}

// ---------------------------------------------------------------------------
// gpu_commutator vs CPU commutator
// ---------------------------------------------------------------------------

TEST(GpuCommutator, MatchesCpuCommutator) {
    auto lhs = random_pauli_sum<64, double>(200, 11, 40);
    auto rhs = random_pauli_sum<64, double>(150, 22, 40);
    for (std::size_t i = 0; i < lhs.get_length(); ++i)
        lhs.value[i] = 0.5 + static_cast<double>(i % 7);
    for (std::size_t i = 0; i < rhs.get_length(); ++i)
        rhs.value[i] = (i % 2 ? -1.0 : 1.0) * (0.25 + static_cast<double>(i % 5));

    PauliSum<64, double> cpu = commutator(lhs, rhs);
    cpu.canonicalize();

    GpuPauliSum<64> g_lhs(lhs), g_rhs(rhs);
    PauliSum<64, double> gpu = gpu_commutator(g_lhs, g_rhs).download();
    gpu.canonicalize();

    expect_soa_eq<64>(cpu, gpu);
}

TEST(GpuCommutator, ICommutatorMatchesCpu) {
    auto lhs = random_pauli_sum<64, double>(100, 33, 30);
    auto rhs = random_pauli_sum<64, double>(100, 44, 30);

    PauliSum<64, double> cpu = i_commutator(lhs, rhs);
    cpu.canonicalize();

    GpuPauliSum<64> g_lhs(lhs), g_rhs(rhs);
    PauliSum<64, double> gpu =
        gpu_commutator(g_lhs, g_rhs, /*multiply_by_i=*/true).download();
    gpu.canonicalize();

    expect_soa_eq<64>(cpu, gpu);
}

TEST(GpuCommutator, CommutingInputsGiveEmptyResult) {
    PauliSum<2, double> a(2, "ZI", 1.0);
    PauliSum<2, double> b(2, "IZ", 1.0);
    GpuPauliSum<2> g_a(a), g_b(b);
    EXPECT_EQ(gpu_commutator(g_a, g_b).size(), 0);
}
