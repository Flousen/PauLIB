#pragma once

#include <cmath>

#include "PauliSumCore.h"
#include "PauliSumUtils.h"

namespace paulib {

// multiply_iX/iZ/iY left-multiply term i by i·X / i·Z / i·Y at qubit c.
// Only valid when the term anti-commutes with that Pauli at c
// (the sign update assumes the anti-commuting cases).

template <std::size_t N, typename T>
inline void PauliSum<N, T>::multiply_iX(size_t i, size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t xc = (x_data[word][i] & mask) != 0;
    flags[i] ^= static_cast<uint8_t>(xc << 1);
    x_data[word][i] ^= mask;
}

template <std::size_t N, typename T>
inline void PauliSum<N, T>::multiply_iZ(size_t i, size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t zc = (z_data[word][i] & mask) != 0;
    flags[i] ^= static_cast<uint8_t>((!zc) << 1);
    z_data[word][i] ^= mask;
}

template <std::size_t N, typename T>
inline void PauliSum<N, T>::multiply_iY(size_t i, size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t xc = (x_data[word][i] & mask) != 0;
    flags[i] ^= static_cast<uint8_t>((!xc) << 1);
    x_data[word][i] ^= mask;
    z_data[word][i] ^= mask;
}


template <std::size_t N, typename T>
void PauliSum<N, T>::apply_H(size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t n = get_length();

    size_t* __restrict__ xw = x_data[word].data();
    size_t* __restrict__ zw = z_data[word].data();
    uint8_t* __restrict__ fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw[i] & mask) != 0;
        const size_t zc = (zw[i] & mask) != 0;
        fl[i] ^= static_cast<uint8_t>((xc & zc) << 1);
        xw[i] ^= (zw[i] & mask);
        zw[i] ^= (xw[i] & mask);
        xw[i] ^= (zw[i] & mask);
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_S(size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t n = get_length();

    size_t* __restrict__ xw = x_data[word].data();
    size_t* __restrict__ zw = z_data[word].data();
    uint8_t* __restrict__ fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw[i] & mask) != 0;
        const size_t zc = (zw[i] & mask) != 0;
        fl[i] ^= static_cast<uint8_t>((xc & zc) << 1);
        zw[i] ^= (xw[i] & mask);
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_H_YZ(size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t n = get_length();

    size_t* __restrict__ xw = x_data[word].data();
    size_t* __restrict__ zw = z_data[word].data();
    uint8_t* __restrict__ fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw[i] & mask) != 0;
        const size_t zc = (zw[i] & mask) != 0;
        fl[i] ^= static_cast<uint8_t>((xc & (1 ^ zc)) << 1);
        xw[i] ^= (zw[i] & mask);
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_H_XY(size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t n = get_length();

    size_t* __restrict__ xw = x_data[word].data();
    size_t* __restrict__ zw = z_data[word].data();
    uint8_t* __restrict__ fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw[i] & mask) != 0;
        const size_t zc = (zw[i] & mask) != 0;
        fl[i] ^= static_cast<uint8_t>(((1 ^ xc) & zc) << 1);
        zw[i] ^= (xw[i] & mask);
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_sqrt_X(size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t n = get_length();

    size_t* __restrict__ xw = x_data[word].data();
    size_t* __restrict__ zw = z_data[word].data();
    uint8_t* __restrict__ fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw[i] & mask) != 0;
        const size_t zc = (zw[i] & mask) != 0;
        fl[i] ^= static_cast<uint8_t>(((1 ^ xc) & zc) << 1);
        xw[i] ^= (-zc) & mask;
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_sqrt_Y(size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t n = get_length();

    size_t* __restrict__ xw = x_data[word].data();
    size_t* __restrict__ zw = z_data[word].data();
    uint8_t* __restrict__ fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw[i] & mask) != 0;
        const size_t zc = (zw[i] & mask) != 0;
        fl[i] ^= static_cast<uint8_t>((xc & (1 ^ zc)) << 1);
        xw[i] ^= (zw[i] & mask);
        zw[i] ^= (xw[i] & mask);
        xw[i] ^= (zw[i] & mask);
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_X(size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t n = get_length();

    const size_t* __restrict__ zw = z_data[word].data();
    uint8_t* __restrict__ fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        fl[i] ^= static_cast<uint8_t>(((zw[i] & mask) != 0) << 1);
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_Y(size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t n = get_length();

    const size_t* __restrict__ xw = x_data[word].data();
    const size_t* __restrict__ zw = z_data[word].data();
    uint8_t* __restrict__ fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        fl[i] ^= static_cast<uint8_t>((((xw[i] ^ zw[i]) & mask) != 0) << 1);
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_Z(size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    const size_t n = get_length();

    const size_t* __restrict__ xw = x_data[word].data();
    uint8_t* __restrict__ fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        fl[i] ^= static_cast<uint8_t>(((xw[i] & mask) != 0) << 1);
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_I(size_t) {
    // Identity: no-op.
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_CNOT(size_t c, size_t t) {
    const size_t word_c = c / BitsPerWord;
    const size_t mask_c = size_t(1) << (c % BitsPerWord);
    const size_t word_t = t / BitsPerWord;
    const size_t mask_t = size_t(1) << (t % BitsPerWord);
    const size_t n = get_length();

    // No __restrict__ here: when c and t fall in the same 64-bit word,
    // the control and target pointers alias the same array.
    size_t* xw_c = x_data[word_c].data();
    size_t* zw_c = z_data[word_c].data();
    size_t* xw_t = x_data[word_t].data();
    size_t* zw_t = z_data[word_t].data();
    uint8_t* fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw_c[i] & mask_c) != 0;
        const size_t zc = (zw_c[i] & mask_c) != 0;
        const size_t xt = (xw_t[i] & mask_t) != 0;
        const size_t zt = (zw_t[i] & mask_t) != 0;

        fl[i] ^= static_cast<uint8_t>((xc & zt & (1 ^ xt ^ zc)) << 1);
        xw_t[i] ^= (-xc) & mask_t;
        zw_c[i] ^= (-zt) & mask_c;
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_CZ(size_t c, size_t t) {
    const size_t word_c = c / BitsPerWord;
    const size_t mask_c = size_t(1) << (c % BitsPerWord);
    const size_t word_t = t / BitsPerWord;
    const size_t mask_t = size_t(1) << (t % BitsPerWord);
    const size_t n = get_length();

    size_t* xw_c = x_data[word_c].data();
    size_t* zw_c = z_data[word_c].data();
    size_t* xw_t = x_data[word_t].data();
    size_t* zw_t = z_data[word_t].data();
    uint8_t* fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw_c[i] & mask_c) != 0;
        const size_t zc = (zw_c[i] & mask_c) != 0;
        const size_t xt = (xw_t[i] & mask_t) != 0;
        const size_t zt = (zw_t[i] & mask_t) != 0;

        fl[i] ^= static_cast<uint8_t>((xc & xt & (zc ^ zt)) << 1);
        zw_t[i] ^= (-xc) & mask_t;
        zw_c[i] ^= (-xt) & mask_c;
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_CY(size_t c, size_t t) {
    const size_t word_c = c / BitsPerWord;
    const size_t mask_c = size_t(1) << (c % BitsPerWord);
    const size_t word_t = t / BitsPerWord;
    const size_t mask_t = size_t(1) << (t % BitsPerWord);
    const size_t n = get_length();

    size_t* xw_c = x_data[word_c].data();
    size_t* zw_c = z_data[word_c].data();
    size_t* xw_t = x_data[word_t].data();
    size_t* zw_t = z_data[word_t].data();
    uint8_t* fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw_c[i] & mask_c) != 0;
        const size_t zc = (zw_c[i] & mask_c) != 0;
        const size_t xt = (xw_t[i] & mask_t) != 0;
        const size_t zt = (zw_t[i] & mask_t) != 0;

        fl[i] ^= static_cast<uint8_t>(((xc & zc & zt & (1 ^ xt)) | (xc & xt & (1 ^ zc) & (1 ^ zt))) << 1);
        zw_c[i] ^= ((-xt) ^ (-zt)) & mask_c;
        xw_t[i] ^= (-xc) & mask_t;
        zw_t[i] ^= (-xc) & mask_t;
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_XC_X(size_t c, size_t t) {
    const size_t word_c = c / BitsPerWord;
    const size_t mask_c = size_t(1) << (c % BitsPerWord);
    const size_t word_t = t / BitsPerWord;
    const size_t mask_t = size_t(1) << (t % BitsPerWord);
    const size_t n = get_length();

    size_t* xw_c = x_data[word_c].data();
    size_t* zw_c = z_data[word_c].data();
    size_t* xw_t = x_data[word_t].data();
    size_t* zw_t = z_data[word_t].data();
    uint8_t* fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw_c[i] & mask_c) != 0;
        const size_t zc = (zw_c[i] & mask_c) != 0;
        const size_t xt = (xw_t[i] & mask_t) != 0;
        const size_t zt = (zw_t[i] & mask_t) != 0;

        fl[i] ^= static_cast<uint8_t>(((xc & zc & zt & (1 ^ xt)) | (xt & zc & zt & (1 ^ xc))) << 1);
        xw_c[i] ^= (-zt) & mask_c;
        xw_t[i] ^= (-zc) & mask_t;
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_XC_Z(size_t c, size_t t) {
    const size_t word_c = c / BitsPerWord;
    const size_t mask_c = size_t(1) << (c % BitsPerWord);
    const size_t word_t = t / BitsPerWord;
    const size_t mask_t = size_t(1) << (t % BitsPerWord);
    const size_t n = get_length();

    size_t* xw_c = x_data[word_c].data();
    size_t* zw_c = z_data[word_c].data();
    size_t* xw_t = x_data[word_t].data();
    size_t* zw_t = z_data[word_t].data();
    uint8_t* fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw_c[i] & mask_c) != 0;
        const size_t zc = (zw_c[i] & mask_c) != 0;
        const size_t xt = (xw_t[i] & mask_t) != 0;
        const size_t zt = (zw_t[i] & mask_t) != 0;

        fl[i] ^= static_cast<uint8_t>(((xc & xt & zc & zt) | (xt & zc & (1 ^ xc) & (1 ^ zt))) << 1);
        xw_c[i] ^= (-xt) & mask_c;
        zw_t[i] ^= (-zc) & mask_t;
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_XC_Y(size_t c, size_t t) {
    const size_t word_c = c / BitsPerWord;
    const size_t mask_c = size_t(1) << (c % BitsPerWord);
    const size_t word_t = t / BitsPerWord;
    const size_t mask_t = size_t(1) << (t % BitsPerWord);
    const size_t n = get_length();

    size_t* xw_c = x_data[word_c].data();
    size_t* zw_c = z_data[word_c].data();
    size_t* xw_t = x_data[word_t].data();
    size_t* zw_t = z_data[word_t].data();
    uint8_t* fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw_c[i] & mask_c) != 0;
        const size_t zc = (zw_c[i] & mask_c) != 0;
        const size_t xt = (xw_t[i] & mask_t) != 0;
        const size_t zt = (zw_t[i] & mask_t) != 0;

        fl[i] ^= static_cast<uint8_t>(((xc & xt & zc & (1 ^ zt)) | (zc & zt & (1 ^ xc) & (1 ^ xt))) << 1);
        xw_c[i] ^= ((-xt) ^ (-zt)) & mask_c;
        xw_t[i] ^= (-zc) & mask_t;
        zw_t[i] ^= (-zc) & mask_t;
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_YC_X(size_t c, size_t t) {
    const size_t word_c = c / BitsPerWord;
    const size_t mask_c = size_t(1) << (c % BitsPerWord);
    const size_t word_t = t / BitsPerWord;
    const size_t mask_t = size_t(1) << (t % BitsPerWord);
    const size_t n = get_length();

    size_t* xw_c = x_data[word_c].data();
    size_t* zw_c = z_data[word_c].data();
    size_t* xw_t = x_data[word_t].data();
    size_t* zw_t = z_data[word_t].data();
    uint8_t* fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw_c[i] & mask_c) != 0;
        const size_t zc = (zw_c[i] & mask_c) != 0;
        const size_t xt = (xw_t[i] & mask_t) != 0;
        const size_t zt = (zw_t[i] & mask_t) != 0;

        fl[i] ^= static_cast<uint8_t>(((xc & xt & zt & (1 ^ zc)) | (zc & zt & (1 ^ xc) & (1 ^ xt))) << 1);
        xw_c[i] ^= (-zt) & mask_c;
        zw_c[i] ^= (-zt) & mask_c;
        xw_t[i] ^= ((-xc) ^ (-zc)) & mask_t;
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_YC_Z(size_t c, size_t t) {
    const size_t word_c = c / BitsPerWord;
    const size_t mask_c = size_t(1) << (c % BitsPerWord);
    const size_t word_t = t / BitsPerWord;
    const size_t mask_t = size_t(1) << (t % BitsPerWord);
    const size_t n = get_length();

    size_t* xw_c = x_data[word_c].data();
    size_t* zw_c = z_data[word_c].data();
    size_t* xw_t = x_data[word_t].data();
    size_t* zw_t = z_data[word_t].data();
    uint8_t* fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw_c[i] & mask_c) != 0;
        const size_t zc = (zw_c[i] & mask_c) != 0;
        const size_t xt = (xw_t[i] & mask_t) != 0;
        const size_t zt = (zw_t[i] & mask_t) != 0;

        fl[i] ^= static_cast<uint8_t>(((xt & zc & zt & (1 ^ xc)) | (xc & xt & (1 ^ zc) & (1 ^ zt))) << 1);
        xw_c[i] ^= (-xt) & mask_c;
        zw_c[i] ^= (-xt) & mask_c;
        zw_t[i] ^= ((-xc) ^ (-zc)) & mask_t;
    }
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_YC_Y(size_t c, size_t t) {
    const size_t word_c = c / BitsPerWord;
    const size_t mask_c = size_t(1) << (c % BitsPerWord);
    const size_t word_t = t / BitsPerWord;
    const size_t mask_t = size_t(1) << (t % BitsPerWord);
    const size_t n = get_length();

    size_t* xw_c = x_data[word_c].data();
    size_t* zw_c = z_data[word_c].data();
    size_t* xw_t = x_data[word_t].data();
    size_t* zw_t = z_data[word_t].data();
    uint8_t* fl = flags.data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw_c[i] & mask_c) != 0;
        const size_t zc = (zw_c[i] & mask_c) != 0;
        const size_t xt = (xw_t[i] & mask_t) != 0;
        const size_t zt = (zw_t[i] & mask_t) != 0;

        fl[i] ^= static_cast<uint8_t>(((xc & zt & (1 ^ xt) & (1 ^ zc)) | (xt & zc & (1 ^ xc) & (1 ^ zt))) << 1);
        xw_c[i] ^= ((-xt) ^ (-zt)) & mask_c;
        zw_c[i] ^= ((-xt) ^ (-zt)) & mask_c;
        xw_t[i] ^= ((-xc) ^ (-zc)) & mask_t;
        zw_t[i] ^= ((-xc) ^ (-zc)) & mask_t;
    }
}

// Swaps the Pauli operators on c and t (both x/z bits); no sign change.
template <std::size_t N, typename T>
void PauliSum<N, T>::apply_SWAP(size_t c, size_t t) {
    const size_t word_c = c / BitsPerWord;
    const size_t mask_c = size_t(1) << (c % BitsPerWord);
    const size_t word_t = t / BitsPerWord;
    const size_t mask_t = size_t(1) << (t % BitsPerWord);
    const size_t n = get_length();

    size_t* xw_c = x_data[word_c].data();
    size_t* zw_c = z_data[word_c].data();
    size_t* xw_t = x_data[word_t].data();
    size_t* zw_t = z_data[word_t].data();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const size_t xc = (xw_c[i] & mask_c) != 0;
        const size_t zc = (zw_c[i] & mask_c) != 0;
        const size_t xt = (xw_t[i] & mask_t) != 0;
        const size_t zt = (zw_t[i] & mask_t) != 0;

        xw_c[i] ^= ((-xc) ^ (-xt)) & mask_c;
        xw_t[i] ^= ((-xt) ^ (-xc)) & mask_t;
        zw_c[i] ^= ((-zc) ^ (-zt)) & mask_c;
        zw_t[i] ^= ((-zt) ^ (-zc)) & mask_t;
    }
}

// iSWAP = SWAP . CZ . (S (x) S), a standard circuit identity.
template <std::size_t N, typename T>
void PauliSum<N, T>::apply_iSWAP(size_t c, size_t t) {
    apply_SWAP(c, t);
    apply_CZ(c, t);
    apply_S(c);
    apply_S(t);
}


} // namespace paulib
