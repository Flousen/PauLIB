#include <gtest/gtest.h>

#include <array>
#include <cstddef>

#include "PauliString.h"


TEST(PauliStringTests, InitAndPrint) {
    std::size_t num_qubits = 5;
    std::string str = "-iXIZYI";

    auto x = paulib::PauliString<5>(str);

    EXPECT_EQ(x.to_string(), str);
}

TEST(PauliStringTests, InnerProduct) {
    std::size_t num_qubits = 5;
    auto a = paulib::PauliString<5>("IIYII");
    auto b = paulib::PauliString<5>("IIXII");
    auto inner = paulib::inner_product(a, b);
    EXPECT_EQ(inner, false);

    auto c = paulib::PauliString<5>("IIXIY");
    auto d = paulib::PauliString<5>("IIZIX");
    auto inner_cd = paulib::inner_product(c, d);
    EXPECT_EQ(inner_cd, true);
}

TEST(PauliStringTests, apply_H_XZ) {
    auto a = paulib::PauliString<4>("IZXY");
    apply_H_XZ(a,0);
    EXPECT_EQ(a.to_string(), "IZXY");
    apply_H_XZ(a,1);
    EXPECT_EQ(a.to_string(), "IXXY");
    apply_H_XZ(a,2);
    EXPECT_EQ(a.to_string(), "IXZY");
    apply_H_XZ(a,3);
    EXPECT_EQ(a.to_string(), "-IXZY");
}

TEST(PauliStringTests, apply_sqrt_Z) {
    auto a = paulib::PauliString<4>("IZXY");
    apply_sqrt_Z(a,0);
    EXPECT_EQ(a.to_string(), "IZXY");
    apply_sqrt_Z(a,1);
    EXPECT_EQ(a.to_string(), "IZXY");
    apply_sqrt_Z(a,2);
    EXPECT_EQ(a.to_string(), "IZYY");
    apply_sqrt_Z(a,3);
    EXPECT_EQ(a.to_string(), "-IZYX");
}

TEST(PauliStringTests, apply_ZC_X) {
    std::map<std::string, std::string> CNOT_on_Pauli_2Q = {
        {"II", "II"}, {"IX", "IX"}, {"IY", "ZY"}, {"IZ", "ZZ"},
        {"XI", "XX"}, {"XX", "XI"}, {"XY", "YZ"}, {"XZ", "-YY"},
        {"YI", "YX"}, {"YX", "YI"}, {"YY", "-XZ"}, {"YZ", "XY"},
        {"ZI", "ZI"}, {"ZX", "ZX"}, {"ZY", "IY"}, {"ZZ", "IZ"}
    };
    for (const auto& [pauli_in, pauli_out] : CNOT_on_Pauli_2Q) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<2>(pauli_in);
        apply_ZC_X(a,0,1);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
    auto a = paulib::PauliString<4>("IZXY");
    apply_ZC_X(a,0,1);
    EXPECT_EQ(a.to_string(), "ZZXY");
    apply_ZC_X(a,0,2);
    EXPECT_EQ(a.to_string(), "ZZXY");
    apply_ZC_X(a,0,3);
    EXPECT_EQ(a.to_string(), "IZXY");
    apply_ZC_X(a,1,2);
    EXPECT_EQ(a.to_string(), "IZXY");
    apply_ZC_X(a,2,1);
    EXPECT_EQ(a.to_string(), "-IYYY");
    apply_ZC_X(a,2,3);
    EXPECT_EQ(a.to_string(), "IYXZ");
}

TEST(PauliStringTests, apply_ZC_Z) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "ZX"}, {"IY", "ZY"}, {"IZ", "IZ"},
        {"XI", "XZ"}, {"XX", "YY"}, {"XY", "-YX"}, {"XZ", "XI"},
        {"YI", "YZ"}, {"YX", "-XY"}, {"YY", "XX"}, {"YZ", "YI"},
        {"ZI", "ZI"}, {"ZX", "IX"}, {"ZY", "IY"}, {"ZZ", "ZZ"}
    };

    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<2>(pauli_in);
        apply_ZC_Z(a,0,1);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, multiply_iZ){

    std::unordered_map<std::string, std::string> pauli_map = {
        {"X","-Y"}, {"Y","X"}, {"-X","Y"}, {"-Y","-X"}
    };

    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<1>(pauli_in);
        multiply_iZ(a,0);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

// Left-multiplication: iX*Y = -Z, iX*Z = Y
TEST(PauliStringTests, multiply_iX) {

    std::unordered_map<std::string, std::string> pauli_map = {
        {"Y","-Z"}, {"Z","Y"}, {"-Y","Z"}, {"-Z","-Y"}
    };

    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<1>(pauli_in);
        multiply_iX(a,0);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, multiply_iY) {

    std::unordered_map<std::string, std::string> pauli_map = {
        {"X","Z"}, {"Z","-X"}, {"-X","-Z"}, {"-Z","X"}
    };

    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<1>(pauli_in);
        multiply_iY(a,0);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}


// Regression: gate word/bit indexing previously divided by sizeof(size_t)
// instead of the number of bits per word, breaking every qubit index >= 8.
TEST(PauliStringTests, GatesOnHighQubitIndices) {
    paulib::PauliString<16> a("IIIIIIIIXIIIIIII");  // X at qubit 8
    a.apply_H_XZ(8);
    EXPECT_EQ(a.to_string(), "IIIIIIIIZIIIIIII");

    // Qubit index in the second 64-bit word.
    std::string s(128, 'I');
    s[70] = 'X';
    paulib::PauliString<128> b(s);
    b.apply_H_XZ(70);
    std::string expect(128, 'I');
    expect[70] = 'Z';
    EXPECT_EQ(b.to_string(), expect);

    // Two-qubit gate spanning both words: CNOT copies X from control to target.
    std::string s2(128, 'I');
    s2[0] = 'X';
    paulib::PauliString<128> c(s2);
    c.apply_ZC_X(0, 70);
    std::string expect2(128, 'I');
    expect2[0] = 'X';
    expect2[70] = 'X';
    EXPECT_EQ(c.to_string(), expect2);
}

TEST(PauliStringTests, multiply_Strings) {
    std::map<std::string, std::map<std::string, std::string>> pauliCayley = { 
        {"I", {{"I","I"}, {"X",  "X"}, {"Y",  "Y"}, {"Z",  "Z"}}},
        {"X", {{"I","X"}, {"X",  "I"}, {"Y", "iZ"}, {"Z","-iY"}}}, 
        {"Y", {{"I","Y"}, {"X","-iZ"}, {"Y",  "I"}, {"Z", "iX"}}}, 
        {"Z", {{"I","Z"}, {"X", "iY"}, {"Y","-iX"}, {"Z",  "I"}}}
    };

    for (const auto& [row, innerMap] : pauliCayley) {
        for (const auto& [col, result] : innerMap) {
            auto a = paulib::PauliString<1>(row);
            auto b = paulib::PauliString<1>(col);
            a*=b;
            EXPECT_EQ(a.to_string(), result )
                << "Failed for input Pauli: " << row << " * " << col << " = " << result;
        }
    }

        for (const auto& [row, innerMap] : pauliCayley) {
        for (const auto& [col, result] : innerMap) {
            auto a = paulib::PauliString<1>(row);
            auto b = paulib::PauliString<1>(col);
            auto c=a*b;
            EXPECT_EQ(c.to_string(), result )
                << "Failed for input Pauli: " << row << " * " << col << " = " << result;
        }
    }
}
