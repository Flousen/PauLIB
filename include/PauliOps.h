#pragma once

// Term views and word-level primitives shared by all PauliSum kernels
// (design principles D2 and D3, thesis Ch. 6, sec:kernel-design).
//
// - SumView: the base pointers of a PauliSum's arrays, taken once per kernel
//   call. Plain pointers, so the same view can also wrap GPU or MPI buffers.
// - TermView: one term of a SumView (pointer + index). The word count W is a
//   compile-time constant, so loops over the words unroll and a lambda that
//   takes a TermView inlines into the skeleton's term loop.
// - TermRef: one term of a PauliSum that a kernel may change in place
//   (strings, phase, coefficient); used by transform_terms and expand_terms.
// - Primitives: the bitwise building blocks (commutation, weight, local
//   Pauli, ...). They accept a TermView, a TermRef or a PauliString, so a
//   hand-written kernel and a skeleton lambda use the same code.

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "PauliScalar.h"
#include "PauliString.h"
#include "PauliSumCore.h"

namespace paulib {

template <std::size_t N, typename T>
struct SumView {
    static constexpr std::size_t W = PauliSum<N, T>::ArraySize;

    std::array<const std::size_t*, W> x{};
    std::array<const std::size_t*, W> z{};
    const uint8_t* flags = nullptr;
    const T* value = nullptr;
    std::size_t length = 0;

    SumView() = default;
    explicit SumView(const PauliSum<N, T>& s)
        : flags(s.flags.data()), value(s.value.data()), length(s.get_length()) {
        for (std::size_t k = 0; k < W; ++k) {
            x[k] = s.x_data[k].data();
            z[k] = s.z_data[k].data();
        }
    }
};

template <std::size_t N, typename T>
struct TermView {
    static constexpr std::size_t W = SumView<N, T>::W;

    const SumView<N, T>* sum;
    std::size_t i;

    std::size_t x(std::size_t k) const { return sum->x[k][i]; }
    std::size_t z(std::size_t k) const { return sum->z[k][i]; }
    uint8_t flags() const { return sum->flags[i]; }
    T value() const { return sum->value[i]; }
    std::size_t index() const { return i; }
};

template <std::size_t N, typename T>
struct TermRef {
    static constexpr std::size_t W = PauliSum<N, T>::ArraySize;

    PauliSum<N, T>* sum;
    std::size_t i;

    std::size_t x(std::size_t k) const { return sum->x_data[k][i]; }
    std::size_t z(std::size_t k) const { return sum->z_data[k][i]; }
    uint8_t flags() const { return sum->flags[i]; }
    T value() const { return sum->value[i]; }
    std::size_t index() const { return i; }

    std::size_t& x_word(std::size_t k) const { return sum->x_data[k][i]; }
    std::size_t& z_word(std::size_t k) const { return sum->z_data[k][i]; }
    uint8_t& flags_ref() const { return sum->flags[i]; }
    T& value_ref() const { return sum->value[i]; }

    void flip_sign() const { sum->flip_sign(i); }
    // term <- term * ps, with the phase of the product.
    void multiply(const PauliString<N>& ps) const { sum->multiply(i, ps); }
};

// Word access for the primitives below: a PauliString, a TermView and a
// TermRef expose their words the same way.
namespace detail {
template <std::size_t N>
inline std::size_t xw(const PauliString<N>& p, std::size_t k) { return p.x[k]; }
template <std::size_t N>
inline std::size_t zw(const PauliString<N>& p, std::size_t k) { return p.z[k]; }
template <std::size_t N, typename T>
inline std::size_t xw(const TermView<N, T>& t, std::size_t k) { return t.x(k); }
template <std::size_t N, typename T>
inline std::size_t zw(const TermView<N, T>& t, std::size_t k) { return t.z(k); }
template <std::size_t N, typename T>
inline std::size_t xw(const TermRef<N, T>& t, std::size_t k) { return t.x(k); }
template <std::size_t N, typename T>
inline std::size_t zw(const TermRef<N, T>& t, std::size_t k) { return t.z(k); }

template <typename P> struct words_of;
template <std::size_t N> struct words_of<PauliString<N>> {
    static constexpr std::size_t value = PauliString<N>::ArraySize;
};
template <std::size_t N, typename T> struct words_of<TermView<N, T>> {
    static constexpr std::size_t value = TermView<N, T>::W;
};
template <std::size_t N, typename T> struct words_of<TermRef<N, T>> {
    static constexpr std::size_t value = TermRef<N, T>::W;
};

// Order of the strings (X words, then Z words), the primary key of
// sort_and_combine (term_less) and the order merge_reduce walks: negative,
// zero or positive. The one definition of the collation order.
template <typename A, typename B>
inline int compare_strings(const A& a, const B& b) {
    constexpr std::size_t W = words_of<A>::value;
    static_assert(W == words_of<B>::value, "compare_strings: different capacities");
    for (std::size_t k = 0; k < W; ++k)
        if (xw(a, k) != xw(b, k)) return xw(a, k) < xw(b, k) ? -1 : 1;
    for (std::size_t k = 0; k < W; ++k)
        if (zw(a, k) != zw(b, k)) return zw(a, k) < zw(b, k) ? -1 : 1;
    return 0;
}
}  // namespace detail

// Symplectic product: true if a and b anticommute. a and b are PauliStrings
// or TermViews of the same capacity.
template <typename A, typename B>
inline bool anticommutes(const A& a, const B& b) {
    static_assert(detail::words_of<A>::value == detail::words_of<B>::value,
                  "anticommutes: operands of different capacity");
    std::size_t acc = 0;
    for (std::size_t k = 0; k < detail::words_of<A>::value; ++k)
        acc ^= (detail::xw(a, k) & detail::zw(b, k)) ^ (detail::zw(a, k) & detail::xw(b, k));
    return (std::popcount(acc) & 1) != 0;
}

// Qubit-wise commutation: on every qubit the two strings have the same letter
// or one of them has I. Implies commutation, not conversely (XX and YY
// commute but not qubit-wise). Strings that commute qubit-wise pairwise are
// measured together in one product basis (measurement grouping, the moments
// of Vallury et al.).
template <typename A, typename B>
inline bool qubitwise_commutes(const A& a, const B& b) {
    static_assert(detail::words_of<A>::value == detail::words_of<B>::value,
                  "qubitwise_commutes: operands of different capacity");
    std::size_t clash = 0;
    for (std::size_t k = 0; k < detail::words_of<A>::value; ++k) {
        const std::size_t ax = detail::xw(a, k), az = detail::zw(a, k);
        const std::size_t bx = detail::xw(b, k), bz = detail::zw(b, k);
        clash |= ((ax ^ bx) | (az ^ bz)) & (ax | az) & (bx | bz);
    }
    return clash == 0;
}

// Number of qubits on which the string acts with X, Y or Z.
template <typename P>
inline std::size_t weight(const P& p) {
    std::size_t w = 0;
    for (std::size_t k = 0; k < detail::words_of<P>::value; ++k)
        w += static_cast<std::size_t>(std::popcount(detail::xw(p, k) | detail::zw(p, k)));
    return w;
}

// True if the string has an X or Y on some qubit, i.e. it is not diagonal in
// the computational basis. Stops at the first nonzero word: most strings of a
// large sum have an X in their first word, so this reads one word instead of
// W (a version without the early exit ran 3-6 times slower for n = 256).
template <typename P>
inline bool has_x(const P& p) {
    for (std::size_t k = 0; k < detail::words_of<P>::value; ++k)
        if (detail::xw(p, k)) return true;
    return false;
}

// True if the string is the identity (up to its phase). Early exit as in
// has_x.
template <typename P>
inline bool is_identity(const P& p) {
    for (std::size_t k = 0; k < detail::words_of<P>::value; ++k)
        if (detail::xw(p, k) | detail::zw(p, k)) return false;
    return true;
}

// Local Pauli on the qubit (word, bit), indexed as x + 2z:
// 0 = I, 1 = X, 2 = Z, 3 = Y.
template <typename P>
inline unsigned local_pauli(const P& p, std::size_t word, std::size_t bit) {
    return static_cast<unsigned>(((detail::xw(p, word) >> bit) & 1) |
                                 (((detail::zw(p, word) >> bit) & 1) << 1));
}

// The coefficient times i^k as a T, and whether a real T can hold it (see
// phase_representable in PauliScalar.h).
template <std::size_t N, typename T>
inline bool contributes(const TermView<N, T>& t) { return phase_representable<T>(t.flags()); }

template <std::size_t N, typename T>
inline T weighted_value(const TermView<N, T>& t) { return phase_factor<T>(t.flags()) * t.value(); }

}  // namespace paulib
