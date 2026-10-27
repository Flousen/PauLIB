// Truth-table tests for the extended Clifford gate set ported from
// paulicpp::PauliString (H_YZ, H_XY, sqrt_X, sqrt_Y, X, Y, Z, I, SWAP,
// iSWAP, and the generalized controlled-Pauli family ZC_Y/XC_*/YC_*),
// at both the single-string (PauliString<N>) and bulk (PauliSum<N,T>)
// levels. Expected values were cross-derived from the standard Heisenberg
// propagation rules for controlled-Pauli gates and independently verified
// programmatically before being transcribed here.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "DenseReference.h"
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
        {"I", "I"}, {"X", "X"}, {"Y", "-Z"}, {"Z", "Y"}
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
        {"I", "I"}, {"X", "Z"}, {"Y", "Y"}, {"Z", "-X"}
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
// --CZ--> ZX --S(0)--> ZX --S(1)--> -ZY.
TEST(PauliStringTests, apply_iSWAP) {
    auto a = paulib::PauliString<2>("XI");
    apply_iSWAP(a, 0, 1);
    EXPECT_EQ(a.to_string(), "-ZY");

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
        {"I", "I"}, {"X", "X"}, {"Y", "-Z"}, {"Z", "Y"}
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
        {"I", "I"}, {"X", "Z"}, {"Y", "Y"}, {"Z", "-X"}
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
    EXPECT_EQ(s.get_str(), "1.000000 * -ZY\n");
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

// ---------------------------------------------------------------------------
// Convention: every gate maps P -> G^dagger P G (Heisenberg picture), checked
// against matrices for all two-qubit strings with all four phases.
// ---------------------------------------------------------------------------

namespace {

using dense::C;
using dense::Mat;

Eigen::Matrix2cd m2(C a, C b, C c, C d) {
    Eigen::Matrix2cd u;
    u << a, b, c, d;
    return u;
}

const Eigen::Matrix2cd kI = m2(1, 0, 0, 1), kX = m2(0, 1, 1, 0),
                       kY = m2(0, C(0, -1), C(0, 1), 0), kZ = m2(1, 0, 0, -1);

Mat on(const Eigen::Matrix2cd& g, std::size_t q) { return dense::single(g, q, 2); }

// Controlled gate A C B: identity where A_c = +1, B on t where A_c = -1.
Mat controlled(const Eigen::Matrix2cd& a, const Eigen::Matrix2cd& b, std::size_t c, std::size_t t) {
    const Mat id = Mat::Identity(4, 4);
    return (id + on(a, c)) / 2.0 + (id - on(a, c)) / 2.0 * on(b, t);
}

}  // namespace

TEST(CliffordConvention, EveryGateIsHeisenbergConjugation) {
    using PS = PauliString<2>;
    const double r = 1.0 / std::sqrt(2.0);
    const C i(0, 1);
    struct Case { std::string name; std::function<void(PS&)> apply; Mat u; };
    std::vector<Case> cases;
    const std::map<char, Eigen::Matrix2cd> pauli = {{'X', kX}, {'Y', kY}, {'Z', kZ}};
    for (std::size_t q : {0, 1}) {
        const std::size_t o = 1 - q;
        const std::string at = "(" + std::to_string(q) + ")";
        cases.push_back({"H_XZ" + at, [q](PS& p) { p.apply_H_XZ(q); }, on(r * (kX + kZ), q)});
        cases.push_back({"H_YZ" + at, [q](PS& p) { p.apply_H_YZ(q); }, on(r * (kY + kZ), q)});
        cases.push_back({"H_XY" + at, [q](PS& p) { p.apply_H_XY(q); }, on(r * (kX + kY), q)});
        cases.push_back({"sqrt_Z" + at, [q](PS& p) { p.apply_sqrt_Z(q); }, on(m2(1, 0, 0, i), q)});
        cases.push_back({"sqrt_X" + at, [q](PS& p) { p.apply_sqrt_X(q); }, on(r * (kI - i * kX), q)});
        cases.push_back({"sqrt_Y" + at, [q](PS& p) { p.apply_sqrt_Y(q); }, on(r * (kI - i * kY), q)});
        cases.push_back({"X" + at, [q](PS& p) { p.apply_X(q); }, on(kX, q)});
        cases.push_back({"Y" + at, [q](PS& p) { p.apply_Y(q); }, on(kY, q)});
        cases.push_back({"Z" + at, [q](PS& p) { p.apply_Z(q); }, on(kZ, q)});
        const std::vector<std::pair<std::string, std::function<void(PS&, std::size_t, std::size_t)>>> ctrl = {
            {"ZX", [](PS& p, std::size_t c, std::size_t t) { p.apply_ZC_X(c, t); }},
            {"ZY", [](PS& p, std::size_t c, std::size_t t) { p.apply_ZC_Y(c, t); }},
            {"ZZ", [](PS& p, std::size_t c, std::size_t t) { p.apply_ZC_Z(c, t); }},
            {"XX", [](PS& p, std::size_t c, std::size_t t) { p.apply_XC_X(c, t); }},
            {"XY", [](PS& p, std::size_t c, std::size_t t) { p.apply_XC_Y(c, t); }},
            {"XZ", [](PS& p, std::size_t c, std::size_t t) { p.apply_XC_Z(c, t); }},
            {"YX", [](PS& p, std::size_t c, std::size_t t) { p.apply_YC_X(c, t); }},
            {"YY", [](PS& p, std::size_t c, std::size_t t) { p.apply_YC_Y(c, t); }},
            {"YZ", [](PS& p, std::size_t c, std::size_t t) { p.apply_YC_Z(c, t); }},
        };
        for (const auto& [ab, g] : ctrl)
            cases.push_back({std::string(1, ab[0]) + "C_" + ab[1] + at,
                             [g = g, q, o](PS& p) { g(p, q, o); },
                             controlled(pauli.at(ab[0]), pauli.at(ab[1]), q, o)});
        Mat xx = on(kX, q) * on(kX, o), yy = on(kY, q) * on(kY, o), zz = on(kZ, q) * on(kZ, o);
        const Mat id = Mat::Identity(4, 4);
        cases.push_back({"SWAP" + at, [q, o](PS& p) { p.apply_SWAP(q, o); }, (id + xx + yy + zz) / 2.0});
        cases.push_back({"iSWAP" + at, [q, o](PS& p) { p.apply_iSWAP(q, o); }, (id + zz) / 2.0 + i * (xx + yy) / 2.0});
    }
    for (const auto& cs : cases)
        for (const char* ph : {"", "i", "-", "-i"})
            for (char a : std::string("IXYZ"))
                for (char b : std::string("IXYZ")) {
                    PS p(std::string(ph) + a + b);
                    const Mat want = cs.u.adjoint() * dense::pauli(p, 2) * cs.u;
                    cs.apply(p);
                    EXPECT_LT(dense::max_abs_diff(dense::pauli(p, 2), want), 1e-12)
                        << cs.name << " on " << ph << a << b;
                }
}

// S, sqrt(X), sqrt(Y) are RZ, RX, RY(pi/2) up to a global phase, so the gates
// and the rotations (U^dagger O U, as evolve) must agree.
TEST(CliffordConvention, QuarterTurnsMatchRotations) {
    const std::size_t n = 3;
    const auto base = random_pauli_sum<64, double>(40, 17, n);
    for (std::size_t q = 0; q < n; ++q) {
        auto a = base, b = base;
        a.apply_S(q);
        b.apply_RZ(q, M_PI / 2);
        EXPECT_LT(dense::max_abs_diff(dense::sum(a), dense::sum(b)), 1e-12) << "S q=" << q;
        a = base, b = base;
        a.apply_sqrt_X(q);
        b.apply_RX(q, M_PI / 2);
        EXPECT_LT(dense::max_abs_diff(dense::sum(a), dense::sum(b)), 1e-12) << "sqrt_X q=" << q;
        a = base, b = base;
        a.apply_sqrt_Y(q);
        b.apply_RY(q, M_PI / 2);
        EXPECT_LT(dense::max_abs_diff(dense::sum(a), dense::sum(b)), 1e-12) << "sqrt_Y q=" << q;
    }
}
