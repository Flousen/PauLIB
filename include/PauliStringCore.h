#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <string>



namespace paulib {

template <std::size_t N>
struct PauliString {

    static constexpr size_t BitsPerWord = std::numeric_limits<size_t>::digits;
    static constexpr size_t ArraySize = (N + BitsPerWord - 1) / BitsPerWord;


    static_assert(N <= UINT32_MAX, "qubit count must fit in uint32_t");

    std::array<std::size_t, ArraySize> x{};
    std::array<std::size_t, ArraySize> z{};
    uint8_t flags = 0;

    // Logical qubit count; N is the compile-time capacity. Defaults to N for
    // C++ use; the Python bindings instantiate word-sized capacities (64, 128,
    // ...) and track the actual qubit count here. Fits in the struct padding.
    uint32_t num_qubits_ = static_cast<uint32_t>(N);

    static constexpr uint8_t PHASE_MASK = 0x01;
    static constexpr uint8_t SIGN_MASK  = 0x02;

    bool getSign()  const noexcept { return (flags & SIGN_MASK)  != 0; }
    bool getPhase() const noexcept { return (flags & PHASE_MASK) != 0; }
    void setSign(bool v)  noexcept { flags = v ? (flags | SIGN_MASK)  : (flags & ~SIGN_MASK); }
    void setPhase(bool v) noexcept { flags = v ? (flags | PHASE_MASK) : (flags & ~PHASE_MASK); }

    void flip_sign() noexcept { flags ^= SIGN_MASK;}
    void flip_phase() noexcept {flags ^= PHASE_MASK;}

    size_t num_qubits() const { return num_qubits_; }
    static constexpr size_t capacity() { return N; }

    PauliString() = default;

    bool operator==(const PauliString& other) const {
        return x == other.x && z == other.z && flags == other.flags &&
               num_qubits_ == other.num_qubits_;
    }

    bool inner_product(const PauliString& other) const;
    bool commute(const PauliString& other) const;
    bool anti_commute(const PauliString& other) const;
    void multiply(const PauliString& other);

    // Declared here; defined in PauliStringUtils.h
    explicit PauliString(const std::string& str);
    void set_pauli(std::size_t i, char p);
    std::string to_string() const;

    // Declared here; defined in PauliStringClifford.h
    void apply_H_XZ(std::size_t c);
    void apply_H_YZ(std::size_t c);
    void apply_H_XY(std::size_t c);
    void apply_sqrt_Z(std::size_t c);
    void apply_sqrt_X(std::size_t c);
    void apply_sqrt_Y(std::size_t c);
    void apply_X(std::size_t c);
    void apply_Y(std::size_t c);
    void apply_Z(std::size_t c);
    void apply_I(std::size_t c);
    void apply_ZC_X(std::size_t c, std::size_t t);
    void apply_ZC_Y(std::size_t c, std::size_t t);
    void apply_ZC_Z(std::size_t c, std::size_t t);
    void apply_XC_X(std::size_t c, std::size_t t);
    void apply_XC_Y(std::size_t c, std::size_t t);
    void apply_XC_Z(std::size_t c, std::size_t t);
    void apply_YC_X(std::size_t c, std::size_t t);
    void apply_YC_Y(std::size_t c, std::size_t t);
    void apply_YC_Z(std::size_t c, std::size_t t);
    void apply_SWAP(std::size_t c, std::size_t t);
    void apply_iSWAP(std::size_t c, std::size_t t);

    // Common-gate-name aliases (see PauliSum for the same names); forward to
    // the physics-named gates above.
    void apply_H(std::size_t c);
    void apply_S(std::size_t c);
    void apply_CNOT(std::size_t c, std::size_t t);
    void apply_CZ(std::size_t c, std::size_t t);

    void multiply_iX(std::size_t c);
    void multiply_iZ(std::size_t c);
    void multiply_iY(std::size_t c);
    bool commute_with_Z(std::size_t c) const;
    bool commute_with_X(std::size_t c) const;
    bool commute_with_Y(std::size_t c) const;
};


// Hash / equality on the (x, z) bit-planes only: strings differing by a
// phase prefix compare equal (e.g. for use as Lie-algebra basis elements).
template <std::size_t N>
struct PauliStringXZHash {
    std::size_t operator()(const PauliString<N>& p) const noexcept {
        std::size_t h = 0xcbf29ce484222325ull;  // FNV-1a over the words
        for (std::size_t k = 0; k < PauliString<N>::ArraySize; ++k) {
            h = (h ^ p.x[k]) * 0x100000001b3ull;
            h = (h ^ p.z[k]) * 0x100000001b3ull;
        }
        return h;
    }
};

template <std::size_t N>
struct PauliStringXZEqual {
    bool operator()(const PauliString<N>& a, const PauliString<N>& b) const noexcept {
        return a.x == b.x && a.z == b.z;
    }
};

template <std::size_t N>
inline bool PauliString<N>::inner_product(const PauliString<N>& b) const {
    const std::size_t* __restrict__ ax = x.data();
    const std::size_t* __restrict__ az = z.data();
    const std::size_t* __restrict__ bx = b.x.data();
    const std::size_t* __restrict__ bz = b.z.data();

    std::size_t total = 0;
    for (std::size_t i = 0; i < PauliString<N>::ArraySize; ++i) 
        total ^= (ax[i] & bz[i]) ^ (az[i] & bx[i]);
    return __builtin_parityll(total) == 0;
}

template <std::size_t N>
inline bool PauliString<N>::commute(const PauliString<N>& other) const {
    return inner_product(other);
}

template <std::size_t N>
inline bool PauliString<N>::anti_commute(const PauliString<N>& other) const {
    return !inner_product(other);
}


template <std::size_t N>
inline void PauliString<N>::multiply(const PauliString<N>& b) {
    std::size_t total_c1 = 0;
    std::size_t total_c2 = 0;

    for (std::size_t k = 0; k < PauliString<N>::ArraySize; ++k) {
        const std::size_t ox = x[k];
        const std::size_t oz = z[k];
        const std::size_t bx = b.x[k];
        const std::size_t bz = b.z[k];

        x[k] = ox ^ bx;
        z[k] = oz ^ bz;

        const std::size_t xbz = ox & bz;
        const std::size_t anti = (bx & oz) ^ xbz;

        total_c2 ^= (total_c1 ^ ox ^ oz ^ bx ^ bz ^ xbz) & anti;
        total_c1 ^= anti;
    }

    const int p1 = std::popcount(total_c1);
    const int p2 = std::popcount(total_c2);

    flags = (flags + b.flags + p1 + 2 * p2) % 4;
}

template <std::size_t N>
inline PauliString<N>& operator*=(PauliString<N>& a, const PauliString<N>& b) {
    a.multiply(b);
    return a;
}

template <std::size_t N>
inline PauliString<N> operator*(const PauliString<N>& lhs, const PauliString<N>& rhs) {
    PauliString<N> copy = lhs;
    copy *= rhs;
    return copy;
}

template <std::size_t N>
inline bool inner_product(const PauliString<N>& a, const PauliString<N>& b) {
    return a.inner_product(b);
}


template <std::size_t N>
inline std::ostream& operator<<(std::ostream& os, const PauliString<N>& obj) {
    os << obj.to_string();
    return os;
}

} // namespace paulib
