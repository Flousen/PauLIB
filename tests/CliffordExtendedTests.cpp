// Truth-table tests for the extended Clifford gate set ported from
// paulicpp::PauliString (H_YZ, H_XY, sqrt_X, sqrt_Y, X, Y, Z, I, SWAP,
// iSWAP, and the generalized controlled-Pauli family ZC_Y/XC_*/YC_*),
// at both the single-string (PauliString<N>) and bulk (PauliSum<N,T>)
// levels. Expected values were cross-derived from the standard Heisenberg
// propagation rules for controlled-Pauli gates and independently verified
// programmatically before being transcribed here.

#include <gtest/gtest.h>

#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "PauLib.h"

using namespace paulib;

// ---------------------------------------------------------------------------
// PauliString<N> single-qubit gates
// ---------------------------------------------------------------------------

TEST(PauliStringTests, apply_H_YZ) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "-X"}, {"Y", "Z"}, {"Z", "Y"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<1>(pauli_in);
        apply_H_YZ(a,0);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_H_XY) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "Y"}, {"Y", "X"}, {"Z", "-Z"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<1>(pauli_in);
        apply_H_XY(a,0);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_sqrt_X) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "X"}, {"Y", "Z"}, {"Z", "-Y"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<1>(pauli_in);
        apply_sqrt_X(a,0);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_sqrt_Y) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "-Z"}, {"Y", "Y"}, {"Z", "X"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<1>(pauli_in);
        apply_sqrt_Y(a,0);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_X) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "X"}, {"Y", "-Y"}, {"Z", "-Z"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<1>(pauli_in);
        apply_X(a,0);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_Y) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "-X"}, {"Y", "Y"}, {"Z", "-Z"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<1>(pauli_in);
        apply_Y(a,0);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_Z) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "-X"}, {"Y", "-Y"}, {"Z", "Z"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<1>(pauli_in);
        apply_Z(a,0);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_I_is_noop) {
    for (const std::string p : {"I", "X", "Y", "Z"}) {
        auto a = paulib::PauliString<1>(p);
        apply_I(a, 0);
        EXPECT_EQ(a.to_string(), p);
    }
}

// ---------------------------------------------------------------------------
// PauliString<N> generalized controlled-Pauli family (two-qubit)
// ---------------------------------------------------------------------------

TEST(PauliStringTests, apply_ZC_Y) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "ZX"}, {"IY", "IY"}, {"IZ", "ZZ"},
        {"XI", "XY"}, {"XX", "-YZ"}, {"XY", "XI"}, {"XZ", "YX"},
        {"YI", "YY"}, {"YX", "XZ"}, {"YY", "YI"}, {"YZ", "-XX"},
        {"ZI", "ZI"}, {"ZX", "IX"}, {"ZY", "ZY"}, {"ZZ", "IZ"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<2>(pauli_in);
        apply_ZC_Y(a,0,1);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_XC_X) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "IX"}, {"IY", "XY"}, {"IZ", "XZ"},
        {"XI", "XI"}, {"XX", "XX"}, {"XY", "IY"}, {"XZ", "IZ"},
        {"YI", "YX"}, {"YX", "YI"}, {"YY", "ZZ"}, {"YZ", "-ZY"},
        {"ZI", "ZX"}, {"ZX", "ZI"}, {"ZY", "-YZ"}, {"ZZ", "YY"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<2>(pauli_in);
        apply_XC_X(a,0,1);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_XC_Z) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "XX"}, {"IY", "XY"}, {"IZ", "IZ"},
        {"XI", "XI"}, {"XX", "IX"}, {"XY", "IY"}, {"XZ", "XZ"},
        {"YI", "YZ"}, {"YX", "ZY"}, {"YY", "-ZX"}, {"YZ", "YI"},
        {"ZI", "ZZ"}, {"ZX", "-YY"}, {"ZY", "YX"}, {"ZZ", "ZI"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<2>(pauli_in);
        apply_XC_Z(a,0,1);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_XC_Y) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "XX"}, {"IY", "IY"}, {"IZ", "XZ"},
        {"XI", "XI"}, {"XX", "IX"}, {"XY", "XY"}, {"XZ", "IZ"},
        {"YI", "YY"}, {"YX", "-ZZ"}, {"YY", "YI"}, {"YZ", "ZX"},
        {"ZI", "ZY"}, {"ZX", "YZ"}, {"ZY", "ZI"}, {"ZZ", "-YX"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<2>(pauli_in);
        apply_XC_Y(a,0,1);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_YC_X) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "IX"}, {"IY", "YY"}, {"IZ", "YZ"},
        {"XI", "XX"}, {"XX", "XI"}, {"XY", "-ZZ"}, {"XZ", "ZY"},
        {"YI", "YI"}, {"YX", "YX"}, {"YY", "IY"}, {"YZ", "IZ"},
        {"ZI", "ZX"}, {"ZX", "ZI"}, {"ZY", "XZ"}, {"ZZ", "-XY"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<2>(pauli_in);
        apply_YC_X(a,0,1);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_YC_Z) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "YX"}, {"IY", "YY"}, {"IZ", "IZ"},
        {"XI", "XZ"}, {"XX", "-ZY"}, {"XY", "ZX"}, {"XZ", "XI"},
        {"YI", "YI"}, {"YX", "IX"}, {"YY", "IY"}, {"YZ", "YZ"},
        {"ZI", "ZZ"}, {"ZX", "XY"}, {"ZY", "-XX"}, {"ZZ", "ZI"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<2>(pauli_in);
        apply_YC_Z(a,0,1);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_YC_Y) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "YX"}, {"IY", "IY"}, {"IZ", "YZ"},
        {"XI", "XY"}, {"XX", "ZZ"}, {"XY", "XI"}, {"XZ", "-ZX"},
        {"YI", "YI"}, {"YX", "IX"}, {"YY", "YY"}, {"YZ", "IZ"},
        {"ZI", "ZY"}, {"ZX", "-XZ"}, {"ZY", "ZI"}, {"ZZ", "XX"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<2>(pauli_in);
        apply_YC_Y(a,0,1);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

TEST(PauliStringTests, apply_SWAP) {
    std::unordered_map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "XI"}, {"IY", "YI"}, {"IZ", "ZI"},
        {"XI", "IX"}, {"XX", "XX"}, {"XY", "YX"}, {"XZ", "ZX"},
        {"YI", "IY"}, {"YX", "XY"}, {"YY", "YY"}, {"YZ", "ZY"},
        {"ZI", "IZ"}, {"ZX", "XZ"}, {"ZY", "YZ"}, {"ZZ", "ZZ"}
    };
    for (const auto& [pauli_in, pauli_out] : pauli_map) {
        SCOPED_TRACE("Testing input: " + pauli_in);
        auto a = paulib::PauliString<2>(pauli_in);
        apply_SWAP(a,0,1);
        EXPECT_EQ(a.to_string(), pauli_out)
            << "Failed for input Pauli: " << pauli_in;
    }
}

// iSWAP = SWAP . CZ . (S (x) S); cross-checked by hand: XI --SWAP--> IX
// --CZ--> ZX --S(0)--> ZX --S(1)--> ZY.
TEST(PauliStringTests, apply_iSWAP) {
    auto a = paulib::PauliString<2>("XI");
    apply_iSWAP(a, 0, 1);
    EXPECT_EQ(a.to_string(), "ZY");

    // Composition sanity: apply_iSWAP must equal the literal 4-gate sequence.
    for (const std::string p : {"II", "XI", "IY", "ZZ", "XY", "YZ"}) {
        auto viaFused = paulib::PauliString<2>(p);
        apply_iSWAP(viaFused, 0, 1);

        auto viaComposed = paulib::PauliString<2>(p);
        apply_SWAP(viaComposed, 0, 1);
        apply_ZC_Z(viaComposed, 0, 1);
        apply_sqrt_Z(viaComposed, 0);
        apply_sqrt_Z(viaComposed, 1);

        EXPECT_EQ(viaFused.to_string(), viaComposed.to_string()) << "input " << p;
    }
}

// ---------------------------------------------------------------------------
// PauliSum<N,T> single-qubit gates (bulk)
// ---------------------------------------------------------------------------

TEST(PauliSumApplyH_YZ, SingleQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "-X"}, {"Y", "Z"}, {"Z", "Y"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_H_YZ(0);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyH_XY, SingleQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "Y"}, {"Y", "X"}, {"Z", "-Z"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_H_XY(0);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplySX, SingleQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "X"}, {"Y", "Z"}, {"Z", "-Y"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_sqrt_X(0);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplySY, SingleQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "-Z"}, {"Y", "Y"}, {"Z", "X"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_sqrt_Y(0);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyX, SingleQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "X"}, {"Y", "-Y"}, {"Z", "-Z"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_X(0);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyY, SingleQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "-X"}, {"Y", "Y"}, {"Z", "-Z"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_Y(0);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyZ, SingleQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"I", "I"}, {"X", "-X"}, {"Y", "-Y"}, {"Z", "Z"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_Z(0);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyI, IsNoop) {
    for (const std::string p : {"I", "X", "Y", "Z"}) {
        PauliSum<1, double> s(std::vector<std::pair<std::string, double>>{{p, 1.0}});
        s.apply_I(0);
        EXPECT_EQ(s.get_str(), "1.000000 * " + p + "\n");
    }
}

// ---------------------------------------------------------------------------
// PauliSum<N,T> generalized controlled-Pauli family (bulk, two-qubit)
// ---------------------------------------------------------------------------

TEST(PauliSumApplyCZ, TwoQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "ZX"}, {"IY", "ZY"}, {"IZ", "IZ"},
        {"XI", "XZ"}, {"XX", "YY"}, {"XY", "-YX"}, {"XZ", "XI"},
        {"YI", "YZ"}, {"YX", "-XY"}, {"YY", "XX"}, {"YZ", "YI"},
        {"ZI", "ZI"}, {"ZX", "IX"}, {"ZY", "IY"}, {"ZZ", "ZZ"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_CZ(0, 1);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyCY, TwoQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "ZX"}, {"IY", "IY"}, {"IZ", "ZZ"},
        {"XI", "XY"}, {"XX", "-YZ"}, {"XY", "XI"}, {"XZ", "YX"},
        {"YI", "YY"}, {"YX", "XZ"}, {"YY", "YI"}, {"YZ", "-XX"},
        {"ZI", "ZI"}, {"ZX", "IX"}, {"ZY", "ZY"}, {"ZZ", "IZ"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_CY(0, 1);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyXC_X, TwoQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "IX"}, {"IY", "XY"}, {"IZ", "XZ"},
        {"XI", "XI"}, {"XX", "XX"}, {"XY", "IY"}, {"XZ", "IZ"},
        {"YI", "YX"}, {"YX", "YI"}, {"YY", "ZZ"}, {"YZ", "-ZY"},
        {"ZI", "ZX"}, {"ZX", "ZI"}, {"ZY", "-YZ"}, {"ZZ", "YY"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_XC_X(0, 1);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyXC_Z, TwoQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "XX"}, {"IY", "XY"}, {"IZ", "IZ"},
        {"XI", "XI"}, {"XX", "IX"}, {"XY", "IY"}, {"XZ", "XZ"},
        {"YI", "YZ"}, {"YX", "ZY"}, {"YY", "-ZX"}, {"YZ", "YI"},
        {"ZI", "ZZ"}, {"ZX", "-YY"}, {"ZY", "YX"}, {"ZZ", "ZI"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_XC_Z(0, 1);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyXC_Y, TwoQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "XX"}, {"IY", "IY"}, {"IZ", "XZ"},
        {"XI", "XI"}, {"XX", "IX"}, {"XY", "XY"}, {"XZ", "IZ"},
        {"YI", "YY"}, {"YX", "-ZZ"}, {"YY", "YI"}, {"YZ", "ZX"},
        {"ZI", "ZY"}, {"ZX", "YZ"}, {"ZY", "ZI"}, {"ZZ", "-YX"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_XC_Y(0, 1);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyYC_X, TwoQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "IX"}, {"IY", "YY"}, {"IZ", "YZ"},
        {"XI", "XX"}, {"XX", "XI"}, {"XY", "-ZZ"}, {"XZ", "ZY"},
        {"YI", "YI"}, {"YX", "YX"}, {"YY", "IY"}, {"YZ", "IZ"},
        {"ZI", "ZX"}, {"ZX", "ZI"}, {"ZY", "XZ"}, {"ZZ", "-XY"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_YC_X(0, 1);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyYC_Z, TwoQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "YX"}, {"IY", "YY"}, {"IZ", "IZ"},
        {"XI", "XZ"}, {"XX", "-ZY"}, {"XY", "ZX"}, {"XZ", "XI"},
        {"YI", "YI"}, {"YX", "IX"}, {"YY", "IY"}, {"YZ", "YZ"},
        {"ZI", "ZZ"}, {"ZX", "XY"}, {"ZY", "-XX"}, {"ZZ", "ZI"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_YC_Z(0, 1);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyYC_Y, TwoQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "YX"}, {"IY", "IY"}, {"IZ", "YZ"},
        {"XI", "XY"}, {"XX", "ZZ"}, {"XY", "XI"}, {"XZ", "-ZX"},
        {"YI", "YI"}, {"YX", "IX"}, {"YY", "YY"}, {"YZ", "IZ"},
        {"ZI", "ZY"}, {"ZX", "-XZ"}, {"ZY", "ZI"}, {"ZZ", "XX"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_YC_Y(0, 1);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplySWAP, TwoQubitTruthTable) {
    const std::map<std::string, std::string> pauli_map = {
        {"II", "II"}, {"IX", "XI"}, {"IY", "YI"}, {"IZ", "ZI"},
        {"XI", "IX"}, {"XX", "XX"}, {"XY", "YX"}, {"XZ", "ZX"},
        {"YI", "IY"}, {"YX", "XY"}, {"YY", "YY"}, {"YZ", "ZY"},
        {"ZI", "IZ"}, {"ZX", "XZ"}, {"ZY", "YZ"}, {"ZZ", "ZZ"}
    };
    for (const auto& [in, out] : pauli_map) {
        SCOPED_TRACE("input " + in);
        PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
        s.apply_SWAP(0, 1);
        EXPECT_EQ(s.get_str(), "1.000000 * " + out + "\n");
    }
}

TEST(PauliSumApplyISWAP, MatchesFusedComposition) {
    PauliSum<2, double> s(std::vector<std::pair<std::string, double>>{{"XI", 1.0}});
    s.apply_iSWAP(0, 1);
    EXPECT_EQ(s.get_str(), "1.000000 * ZY\n");
}

// ---------------------------------------------------------------------------
// Cross-checks: bulk PauliSum gate must match the scalar PauliString gate
// term-by-term on a random multi-term sum, including qubit indices that
// cross a 64-bit word boundary.
// ---------------------------------------------------------------------------

namespace {

template <std::size_t N, typename SumGate, typename StringGate>
void expect_matches_pauli_string(SumGate sum_gate, StringGate string_gate) {
    auto s = random_pauli_sum<N, double>(20, 5);
    auto ref = s;
    sum_gate(s);
    for (std::size_t i = 0; i < ref.get_length(); ++i) {
        auto ps = ref.get_pauli_string(i);
        string_gate(ps);
        SCOPED_TRACE("term " + std::to_string(i));
        EXPECT_EQ(s.get_pauli_string(i).to_string(), ps.to_string());
    }
}

} // namespace

TEST(PauliSumApplyCZ, MatchesPauliStringOnRandomSum) {
    expect_matches_pauli_string<64>(
        [](auto& s) { s.apply_CZ(3, 9); },
        [](auto& ps) { ps.apply_ZC_Z(3, 9); });
}

TEST(PauliSumApplyCY, MatchesPauliStringOnRandomSum) {
    expect_matches_pauli_string<64>(
        [](auto& s) { s.apply_CY(3, 9); },
        [](auto& ps) { ps.apply_ZC_Y(3, 9); });
}

TEST(PauliSumApplyXC_Y, MatchesPauliStringOnRandomSum) {
    expect_matches_pauli_string<64>(
        [](auto& s) { s.apply_XC_Y(3, 9); },
        [](auto& ps) { ps.apply_XC_Y(3, 9); });
}

TEST(PauliSumApplyYC_Z, MatchesPauliStringOnRandomSum) {
    expect_matches_pauli_string<64>(
        [](auto& s) { s.apply_YC_Z(3, 9); },
        [](auto& ps) { ps.apply_YC_Z(3, 9); });
}

TEST(PauliSumApplySWAP, MatchesPauliStringOnRandomSum) {
    expect_matches_pauli_string<64>(
        [](auto& s) { s.apply_SWAP(3, 9); },
        [](auto& ps) { ps.apply_SWAP(3, 9); });
}

TEST(PauliSumApplyISWAP, MatchesPauliStringOnRandomSum) {
    expect_matches_pauli_string<64>(
        [](auto& s) { s.apply_iSWAP(3, 9); },
        [](auto& ps) { ps.apply_iSWAP(3, 9); });
}

// Control and target in different 64-bit words (mirrors the existing
// apply_CNOT cross-word regression test).
TEST(PauliSumApplyCZ, CrossWordControlTarget) {
    std::string in(128, 'I');
    in[0] = 'X';
    PauliSum<128, double> s(std::vector<std::pair<std::string, double>>{{in, 1.0}});
    s.apply_CZ(0, 70);
    std::string expect(128, 'I');
    expect[0] = 'X';
    expect[70] = 'Z';
    EXPECT_EQ(s.get_str(), "1.000000 * " + expect + "\n");
}
