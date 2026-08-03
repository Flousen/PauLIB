#pragma once

#include "PauliStringCore.h"

namespace paulib {

template <std::size_t N>
inline void PauliString<N>::apply_H_XZ(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);
    const std::size_t xc = (x[word] & mask) != 0;
    const std::size_t zc = (z[word] & mask) != 0;

    flags ^= static_cast<uint8_t>((xc & zc) << 1);
    x[word] ^= (z[word] & mask);
    z[word] ^= (x[word] & mask);
    x[word] ^= (z[word] & mask);
}

template <std::size_t N>
inline void PauliString<N>::apply_sqrt_Z(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);
    const std::size_t xc = (x[word] & mask) != 0;
    const std::size_t zc = (z[word] & mask) != 0;

    flags ^= static_cast<uint8_t>((xc & zc) << 1);
    z[word] ^= (-xc) & mask;
}

// H_YZ: reflection about the Y=Z Bloch axis. X -> -X, Y <-> Z, I fixed.
template <std::size_t N>
inline void PauliString<N>::apply_H_YZ(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);
    const std::size_t xc = (x[word] & mask) != 0;
    const std::size_t zc = (z[word] & mask) != 0;

    flags ^= static_cast<uint8_t>((xc & (1 ^ zc)) << 1);
    x[word] ^= (z[word] & mask);
}

// H_XY: reflection about the X=Y Bloch axis. Z -> -Z, X <-> Y, I fixed.
template <std::size_t N>
inline void PauliString<N>::apply_H_XY(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);
    const std::size_t xc = (x[word] & mask) != 0;
    const std::size_t zc = (z[word] & mask) != 0;

    flags ^= static_cast<uint8_t>(((1 ^ xc) & zc) << 1);
    z[word] ^= (x[word] & mask);
}

// sqrt(X): X fixed, Z -> -Y, Y -> Z.
template <std::size_t N>
inline void PauliString<N>::apply_sqrt_X(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);
    const std::size_t xc = (x[word] & mask) != 0;
    const std::size_t zc = (z[word] & mask) != 0;

    flags ^= static_cast<uint8_t>(((1 ^ xc) & zc) << 1);
    x[word] ^= (-zc) & mask;
}

// sqrt(Y): Y fixed, X -> -Z, Z -> X.
template <std::size_t N>
inline void PauliString<N>::apply_sqrt_Y(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);
    const std::size_t xc = (x[word] & mask) != 0;
    const std::size_t zc = (z[word] & mask) != 0;

    flags ^= static_cast<uint8_t>((xc & (1 ^ zc)) << 1);
    x[word] ^= (z[word] & mask);
    z[word] ^= (x[word] & mask);
    x[word] ^= (z[word] & mask);
}

// Pauli gates as conjugations: P -> G P G for G in {X, Y, Z}. I is fixed by
// all three; each flips the sign of the two Paulis it anti-commutes with.
template <std::size_t N>
inline void PauliString<N>::apply_X(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);
    flags ^= static_cast<uint8_t>(((z[word] & mask) != 0) << 1);
}

template <std::size_t N>
inline void PauliString<N>::apply_Y(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);
    flags ^= static_cast<uint8_t>((((x[word] ^ z[word]) & mask) != 0) << 1);
}

template <std::size_t N>
inline void PauliString<N>::apply_Z(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);
    flags ^= static_cast<uint8_t>(((x[word] & mask) != 0) << 1);
}

template <std::size_t N>
inline void PauliString<N>::apply_I(std::size_t) {
    // Identity: no-op.
}

template <std::size_t N>
inline void PauliString<N>::apply_ZC_X(std::size_t c, std::size_t t) {
    const std::size_t word_c = c / BitsPerWord;
    const std::size_t word_t = t / BitsPerWord;
    const std::size_t mask_c = std::size_t(1) << (c % BitsPerWord);
    const std::size_t mask_t = std::size_t(1) << (t % BitsPerWord);

    const std::size_t xc = (x[word_c] & mask_c) != 0;
    const std::size_t xt = (x[word_t] & mask_t) != 0;
    const std::size_t zc = (z[word_c] & mask_c) != 0;
    const std::size_t zt = (z[word_t] & mask_t) != 0;

    flags ^= static_cast<uint8_t>((xc & zt & (1u ^ xt ^ zc)) << 1);
    x[word_t] ^= (-xc) & mask_t;
    z[word_c] ^= (-zt) & mask_c;
}

template <std::size_t N>
inline void PauliString<N>::apply_ZC_Z(std::size_t c, std::size_t t) {
    const std::size_t word_c = c / BitsPerWord;
    const std::size_t word_t = t / BitsPerWord;
    const std::size_t mask_c = std::size_t(1) << (c % BitsPerWord);
    const std::size_t mask_t = std::size_t(1) << (t % BitsPerWord);

    const std::size_t xc = (x[word_c] & mask_c) != 0;
    const std::size_t xt = (x[word_t] & mask_t) != 0;
    const std::size_t zc = (z[word_c] & mask_c) != 0;
    const std::size_t zt = (z[word_t] & mask_t) != 0;

    flags ^= static_cast<uint8_t>((xc & xt & (zc ^ zt)) << 1);
    z[word_t] ^= (-xc) & mask_t;
    z[word_c] ^= (-xt) & mask_c;
}

// Generalized controlled-Pauli gates below are equivalent to changing the
// control/target qubit's basis to the native Z-control/X-target frame that
// apply_ZC_X (CNOT) and apply_ZC_Z (CZ) operate in, applying the native gate,
// then changing the basis back (e.g. apply_XC_X(c,t) == H(c); ZC_X(c,t); H(c)).
// The bit/sign formulas below are that composition reduced to a closed form.

template <std::size_t N>
inline void PauliString<N>::apply_ZC_Y(std::size_t c, std::size_t t) {
    const std::size_t word_c = c / BitsPerWord;
    const std::size_t word_t = t / BitsPerWord;
    const std::size_t mask_c = std::size_t(1) << (c % BitsPerWord);
    const std::size_t mask_t = std::size_t(1) << (t % BitsPerWord);

    const std::size_t xc = (x[word_c] & mask_c) != 0;
    const std::size_t xt = (x[word_t] & mask_t) != 0;
    const std::size_t zc = (z[word_c] & mask_c) != 0;
    const std::size_t zt = (z[word_t] & mask_t) != 0;

    flags ^= static_cast<uint8_t>(((xc & zc & zt & (1 ^ xt)) | (xc & xt & (1 ^ zc) & (1 ^ zt))) << 1);
    z[word_c] ^= ((-xt) ^ (-zt)) & mask_c;
    x[word_t] ^= (-xc) & mask_t;
    z[word_t] ^= (-xc) & mask_t;
}

template <std::size_t N>
inline void PauliString<N>::apply_XC_X(std::size_t c, std::size_t t) {
    const std::size_t word_c = c / BitsPerWord;
    const std::size_t word_t = t / BitsPerWord;
    const std::size_t mask_c = std::size_t(1) << (c % BitsPerWord);
    const std::size_t mask_t = std::size_t(1) << (t % BitsPerWord);

    const std::size_t xc = (x[word_c] & mask_c) != 0;
    const std::size_t xt = (x[word_t] & mask_t) != 0;
    const std::size_t zc = (z[word_c] & mask_c) != 0;
    const std::size_t zt = (z[word_t] & mask_t) != 0;

    flags ^= static_cast<uint8_t>(((xc & zc & zt & (1 ^ xt)) | (xt & zc & zt & (1 ^ xc))) << 1);
    x[word_c] ^= (-zt) & mask_c;
    x[word_t] ^= (-zc) & mask_t;
}

template <std::size_t N>
inline void PauliString<N>::apply_XC_Z(std::size_t c, std::size_t t) {
    const std::size_t word_c = c / BitsPerWord;
    const std::size_t word_t = t / BitsPerWord;
    const std::size_t mask_c = std::size_t(1) << (c % BitsPerWord);
    const std::size_t mask_t = std::size_t(1) << (t % BitsPerWord);

    const std::size_t xc = (x[word_c] & mask_c) != 0;
    const std::size_t xt = (x[word_t] & mask_t) != 0;
    const std::size_t zc = (z[word_c] & mask_c) != 0;
    const std::size_t zt = (z[word_t] & mask_t) != 0;

    flags ^= static_cast<uint8_t>(((xc & xt & zc & zt) | (xt & zc & (1 ^ xc) & (1 ^ zt))) << 1);
    x[word_c] ^= (-xt) & mask_c;
    z[word_t] ^= (-zc) & mask_t;
}

template <std::size_t N>
inline void PauliString<N>::apply_XC_Y(std::size_t c, std::size_t t) {
    const std::size_t word_c = c / BitsPerWord;
    const std::size_t word_t = t / BitsPerWord;
    const std::size_t mask_c = std::size_t(1) << (c % BitsPerWord);
    const std::size_t mask_t = std::size_t(1) << (t % BitsPerWord);

    const std::size_t xc = (x[word_c] & mask_c) != 0;
    const std::size_t xt = (x[word_t] & mask_t) != 0;
    const std::size_t zc = (z[word_c] & mask_c) != 0;
    const std::size_t zt = (z[word_t] & mask_t) != 0;

    flags ^= static_cast<uint8_t>(((xc & xt & zc & (1 ^ zt)) | (zc & zt & (1 ^ xc) & (1 ^ xt))) << 1);
    x[word_c] ^= ((-xt) ^ (-zt)) & mask_c;
    x[word_t] ^= (-zc) & mask_t;
    z[word_t] ^= (-zc) & mask_t;
}

template <std::size_t N>
inline void PauliString<N>::apply_YC_X(std::size_t c, std::size_t t) {
    const std::size_t word_c = c / BitsPerWord;
    const std::size_t word_t = t / BitsPerWord;
    const std::size_t mask_c = std::size_t(1) << (c % BitsPerWord);
    const std::size_t mask_t = std::size_t(1) << (t % BitsPerWord);

    const std::size_t xc = (x[word_c] & mask_c) != 0;
    const std::size_t xt = (x[word_t] & mask_t) != 0;
    const std::size_t zc = (z[word_c] & mask_c) != 0;
    const std::size_t zt = (z[word_t] & mask_t) != 0;

    flags ^= static_cast<uint8_t>(((xc & xt & zt & (1 ^ zc)) | (zc & zt & (1 ^ xc) & (1 ^ xt))) << 1);
    x[word_c] ^= (-zt) & mask_c;
    z[word_c] ^= (-zt) & mask_c;
    x[word_t] ^= ((-xc) ^ (-zc)) & mask_t;
}

template <std::size_t N>
inline void PauliString<N>::apply_YC_Z(std::size_t c, std::size_t t) {
    const std::size_t word_c = c / BitsPerWord;
    const std::size_t word_t = t / BitsPerWord;
    const std::size_t mask_c = std::size_t(1) << (c % BitsPerWord);
    const std::size_t mask_t = std::size_t(1) << (t % BitsPerWord);

    const std::size_t xc = (x[word_c] & mask_c) != 0;
    const std::size_t xt = (x[word_t] & mask_t) != 0;
    const std::size_t zc = (z[word_c] & mask_c) != 0;
    const std::size_t zt = (z[word_t] & mask_t) != 0;

    flags ^= static_cast<uint8_t>(((xt & zc & zt & (1 ^ xc)) | (xc & xt & (1 ^ zc) & (1 ^ zt))) << 1);
    x[word_c] ^= (-xt) & mask_c;
    z[word_c] ^= (-xt) & mask_c;
    z[word_t] ^= ((-xc) ^ (-zc)) & mask_t;
}

template <std::size_t N>
inline void PauliString<N>::apply_YC_Y(std::size_t c, std::size_t t) {
    const std::size_t word_c = c / BitsPerWord;
    const std::size_t word_t = t / BitsPerWord;
    const std::size_t mask_c = std::size_t(1) << (c % BitsPerWord);
    const std::size_t mask_t = std::size_t(1) << (t % BitsPerWord);

    const std::size_t xc = (x[word_c] & mask_c) != 0;
    const std::size_t xt = (x[word_t] & mask_t) != 0;
    const std::size_t zc = (z[word_c] & mask_c) != 0;
    const std::size_t zt = (z[word_t] & mask_t) != 0;

    flags ^= static_cast<uint8_t>(((xc & zt & (1 ^ xt) & (1 ^ zc)) | (xt & zc & (1 ^ xc) & (1 ^ zt))) << 1);
    x[word_c] ^= ((-xt) ^ (-zt)) & mask_c;
    z[word_c] ^= ((-xt) ^ (-zt)) & mask_c;
    x[word_t] ^= ((-xc) ^ (-zc)) & mask_t;
    z[word_t] ^= ((-xc) ^ (-zc)) & mask_t;
}

// Swaps the Pauli operators on c and t (both x/z bits); no sign change.
template <std::size_t N>
inline void PauliString<N>::apply_SWAP(std::size_t c, std::size_t t) {
    const std::size_t word_c = c / BitsPerWord;
    const std::size_t word_t = t / BitsPerWord;
    const std::size_t mask_c = std::size_t(1) << (c % BitsPerWord);
    const std::size_t mask_t = std::size_t(1) << (t % BitsPerWord);

    const std::size_t xc = (x[word_c] & mask_c) != 0;
    const std::size_t xt = (x[word_t] & mask_t) != 0;
    const std::size_t zc = (z[word_c] & mask_c) != 0;
    const std::size_t zt = (z[word_t] & mask_t) != 0;

    x[word_c] ^= ((-xc) ^ (-xt)) & mask_c;
    x[word_t] ^= ((-xt) ^ (-xc)) & mask_t;
    z[word_c] ^= ((-zc) ^ (-zt)) & mask_c;
    z[word_t] ^= ((-zt) ^ (-zc)) & mask_t;
}

// iSWAP = SWAP . CZ . (S (x) S), a standard circuit identity.
template <std::size_t N>
inline void PauliString<N>::apply_iSWAP(std::size_t c, std::size_t t) {
    apply_SWAP(c, t);
    apply_ZC_Z(c, t);
    apply_sqrt_Z(c);
    apply_sqrt_Z(t);
}

// multiply_iX/iZ/iY left-multiply by i·X / i·Z / i·Y at qubit c.
// Only valid when the string anti-commutes with that Pauli at c
// (the sign update assumes the anti-commuting cases).

template <std::size_t N>
inline void PauliString<N>::multiply_iX(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);

    flags ^= static_cast<uint8_t>(((x[word] & mask) != 0) << 1);
    x[word] ^= mask;
}

template <std::size_t N>
inline void PauliString<N>::multiply_iZ(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);

    flags ^= static_cast<uint8_t>((!(z[word] & mask)) << 1);
    z[word] ^= mask;
}

template <std::size_t N>
inline void PauliString<N>::multiply_iY(std::size_t c) {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);

    flags ^= static_cast<uint8_t>((!(x[word] & mask)) << 1);
    x[word] ^= mask;
    z[word] ^= mask;
}

template <std::size_t N>
inline bool PauliString<N>::commute_with_Z(std::size_t c) const {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);

    return (x[word] & mask) == 0;
}

template <std::size_t N>
inline bool PauliString<N>::commute_with_X(std::size_t c) const {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);

    return (z[word] & mask) == 0;
}

template <std::size_t N>
inline bool PauliString<N>::commute_with_Y(std::size_t c) const {
    const std::size_t word = c / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (c % BitsPerWord);

    return ((x[word] & mask) == 0) != ((z[word] & mask) == 0);
}

// Common-gate-name aliases (member): forward to the physics-named gates.

template <std::size_t N>
inline void PauliString<N>::apply_H(std::size_t c) {
    apply_H_XZ(c);
}

template <std::size_t N>
inline void PauliString<N>::apply_S(std::size_t c) {
    apply_sqrt_Z(c);
}

template <std::size_t N>
inline void PauliString<N>::apply_CNOT(std::size_t c, std::size_t t) {
    apply_ZC_X(c, t);
}

template <std::size_t N>
inline void PauliString<N>::apply_CZ(std::size_t c, std::size_t t) {
    apply_ZC_Z(c, t);
}

// Free function wrappers

template <std::size_t N>
inline void apply_H_XZ(PauliString<N>& ps, std::size_t c) {
    ps.apply_H_XZ(c);
}

template <std::size_t N>
inline void apply_sqrt_Z(PauliString<N>& ps, std::size_t c) {
    ps.apply_sqrt_Z(c);
}

template <std::size_t N>
inline void apply_ZC_X(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_ZC_X(c, t);
}

template <std::size_t N>
inline void apply_ZC_Z(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_ZC_Z(c, t);
}

template <std::size_t N>
inline void apply_H_YZ(PauliString<N>& ps, std::size_t c) {
    ps.apply_H_YZ(c);
}

template <std::size_t N>
inline void apply_H_XY(PauliString<N>& ps, std::size_t c) {
    ps.apply_H_XY(c);
}

template <std::size_t N>
inline void apply_sqrt_X(PauliString<N>& ps, std::size_t c) {
    ps.apply_sqrt_X(c);
}

template <std::size_t N>
inline void apply_sqrt_Y(PauliString<N>& ps, std::size_t c) {
    ps.apply_sqrt_Y(c);
}

template <std::size_t N>
inline void apply_X(PauliString<N>& ps, std::size_t c) {
    ps.apply_X(c);
}

template <std::size_t N>
inline void apply_Y(PauliString<N>& ps, std::size_t c) {
    ps.apply_Y(c);
}

template <std::size_t N>
inline void apply_Z(PauliString<N>& ps, std::size_t c) {
    ps.apply_Z(c);
}

template <std::size_t N>
inline void apply_I(PauliString<N>& ps, std::size_t c) {
    ps.apply_I(c);
}

template <std::size_t N>
inline void apply_ZC_Y(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_ZC_Y(c, t);
}

template <std::size_t N>
inline void apply_XC_X(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_XC_X(c, t);
}

template <std::size_t N>
inline void apply_XC_Z(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_XC_Z(c, t);
}

template <std::size_t N>
inline void apply_XC_Y(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_XC_Y(c, t);
}

template <std::size_t N>
inline void apply_YC_X(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_YC_X(c, t);
}

template <std::size_t N>
inline void apply_YC_Z(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_YC_Z(c, t);
}

template <std::size_t N>
inline void apply_YC_Y(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_YC_Y(c, t);
}

template <std::size_t N>
inline void apply_SWAP(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_SWAP(c, t);
}

template <std::size_t N>
inline void apply_iSWAP(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_iSWAP(c, t);
}

template <std::size_t N>
inline void apply_H(PauliString<N>& ps, std::size_t c) {
    ps.apply_H(c);
}

template <std::size_t N>
inline void apply_S(PauliString<N>& ps, std::size_t c) {
    ps.apply_S(c);
}

template <std::size_t N>
inline void apply_CNOT(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_CNOT(c, t);
}

template <std::size_t N>
inline void apply_CZ(PauliString<N>& ps, std::size_t c, std::size_t t) {
    ps.apply_CZ(c, t);
}

template <std::size_t N>
inline void multiply_iX(PauliString<N>& ps, std::size_t c) {
    ps.multiply_iX(c);
}

template <std::size_t N>
inline void multiply_iZ(PauliString<N>& ps, std::size_t c) {
    ps.multiply_iZ(c);
}

template <std::size_t N>
inline void multiply_iY(PauliString<N>& ps, std::size_t c) {
    ps.multiply_iY(c);
}

template <std::size_t N>
inline bool commute_with_Z(const PauliString<N>& a, std::size_t c) {
    return a.commute_with_Z(c);
}

template <std::size_t N>
inline bool commute_with_X(const PauliString<N>& a, std::size_t c) {
    return a.commute_with_X(c);
}

template <std::size_t N>
inline bool commute_with_Y(const PauliString<N>& a, std::size_t c) {
    return a.commute_with_Y(c);
}

} // namespace paulib
