#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "PauliSumCore.h"

namespace paulib {

template <std::size_t N, typename T>
std::string PauliSum<N, T>::get_str() const {
    std::string result;
    PauliString<N> ps{};
    ps.num_qubits_ = static_cast<uint32_t>(num_qubits_);
    for (size_t i = 0; i < get_length(); ++i) {
        for (size_t k = 0; k < ArraySize; ++k) {
            ps.x[k] = x_data[k][i];
            ps.z[k] = z_data[k][i];
        }
        ps.flags = flags[i];
        result += std::to_string(value[i]) +
                    " * " + ps.to_string() + "\n";
    }
    return result;
}

template <std::size_t N, typename T>
void PauliSum<N, T>::reserve(size_t capacity) {
    for (size_t i = 0; i < ArraySize; ++i) {
        x_data[i].reserve(capacity);
        z_data[i].reserve(capacity);
    }
    flags.reserve(capacity);
    value.reserve(capacity);
}

template <std::size_t N, typename T>
void PauliSum<N, T>::push_back(const PauliString<N>& ps, T coefficient) {
    for (size_t i = 0; i < ArraySize; ++i) {
        x_data[i].push_back(ps.x[i]);
        z_data[i].push_back(ps.z[i]);
    }
    flags.push_back(ps.flags);
    value.push_back(coefficient);
}

template <std::size_t N, typename T>
PauliString<N> PauliSum<N, T>::get_pauli_string(size_t i) const {
    PauliString<N> ps{};
    for (size_t k = 0; k < ArraySize; ++k) {
        ps.x[k] = x_data[k][i];
        ps.z[k] = z_data[k][i];
    }
    ps.flags = flags[i];
    ps.num_qubits_ = static_cast<uint32_t>(num_qubits_);
    return ps;
}

template <std::size_t N, typename T>
inline void PauliSum<N, T>::set_Z(size_t i, size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    z_data[word][i] |= mask;
    x_data[word][i] &= ~mask;
}

template <std::size_t N, typename T>
inline void PauliSum<N, T>::set_X(size_t i, size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    x_data[word][i] |= mask;
    z_data[word][i] &= ~mask;
}

template <std::size_t N, typename T>
inline void PauliSum<N, T>::set_Y(size_t i, size_t c) {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    z_data[word][i] |= mask;
    x_data[word][i] |= mask;
}


template <std::size_t N, typename T>
inline bool PauliSum<N, T>::commute_with_Z(size_t i, size_t c) const {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    return (x_data[word][i] & mask) == 0;
}

template <std::size_t N, typename T>
inline bool PauliSum<N, T>::commute_with_X(size_t i, size_t c) const {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    return (z_data[word][i] & mask) == 0;
}

template <std::size_t N, typename T>
inline bool PauliSum<N, T>::commute_with_Y(size_t i, size_t c) const {
    const size_t word = c / BitsPerWord;
    const size_t mask = size_t(1) << (c % BitsPerWord);
    return ((x_data[word][i] ^ z_data[word][i]) & mask) == 0;
}

template <std::size_t N, typename T>
inline PauliSum<N, T> random_pauli_sum(std::size_t num_terms,
                                       std::uint64_t seed,
                                       std::size_t num_qubits = N) {
    if (num_qubits > N)
        throw std::invalid_argument("qubit count exceeds capacity");

    constexpr std::size_t BPW = PauliString<N>::BitsPerWord;
    const std::size_t full_words = num_qubits / BPW;
    const std::size_t rem        = num_qubits % BPW;

    PauliSum<N, T> result;
    result.num_qubits_ = num_qubits;
    result.reserve(num_terms);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> dist;

    for (std::size_t t = 0; t < num_terms; ++t) {
        PauliString<N> ps{};
        ps.num_qubits_ = static_cast<uint32_t>(num_qubits);
        ps.setSign(dist(rng) & 1);
        ps.setPhase(dist(rng) & 1);

        // Fill only the words covering num_qubits; clear the padding bits
        // beyond qubit num_qubits-1, which would otherwise act as phantom
        // qubits in commutation and multiplication.
        for (std::size_t w = 0; w < full_words; ++w) {
            ps.x[w] = dist(rng);
            ps.z[w] = dist(rng);
        }
        if (rem != 0) {
            const std::size_t pad_mask = (std::size_t(1) << rem) - 1;
            ps.x[full_words] = dist(rng) & pad_mask;
            ps.z[full_words] = dist(rng) & pad_mask;
        }

        result.push_back(ps, static_cast<T>(1));
    }

    return result;
}

// Compute <0^N | sum_i c_i P_i | 0^N >.
//
// <0|I|0> = <0|Z|0> = 1, <0|X|0> = <0|Y|0> = 0, so only terms
// whose Pauli string contains no X or Y operators contribute.
// Terms with an imaginary phase flag are skipped (their contribution
// would be imaginary, incompatible with T).
template <std::size_t N, typename T>
T PauliSum<N, T>::zero_state_expectation() const {
    T result = T{};
    for (size_t i = 0; i < get_length(); ++i) {
        // Any X bit (set for X and Y operators) means <0|…|0> = 0.
        bool has_xy = false;
        for (size_t k = 0; k < ArraySize; ++k) {
            if (x_data[k][i]) { has_xy = true; break; }
        }
        if (has_xy) continue;
        if (flags[i] & PHASE_MASK) continue;
        const T sign = (flags[i] & SIGN_MASK) ? T{-1} : T{1};
        result += sign * value[i];
    }
    return result;
}

// Compute <x| sum_i c_i P_i |x> for a computational basis state |x>, given
// as a string of '0'/'1' per qubit in the same left-to-right qubit order as
// PauliString's constructor (bits[0] is qubit 0). Generalizes
// zero_state_expectation() (equivalent to bits = "00...0").
//
// <x|I|x> = 1, <x|X|x> = <x|Y|x> = 0, <x|Z|x> = (-1)^bit: only terms with no
// X or Y contribute, and each '1' bit under a Z flips the sign once.
// Terms with an imaginary phase flag are skipped (their contribution
// would be imaginary, incompatible with T).
template <std::size_t N, typename T>
T PauliSum<N, T>::basis_state_expectation(const std::string& bits) const {
    if (bits.size() != num_qubits_) {
        throw std::invalid_argument(
            "basis_state_expectation: bit string length " +
            std::to_string(bits.size()) + " does not match " +
            std::to_string(num_qubits_) + " qubits");
    }

    std::array<size_t, ArraySize> bit_words{};
    for (size_t q = 0; q < bits.size(); ++q) {
        const char c = bits[q];
        if (c != '0' && c != '1') {
            throw std::invalid_argument(
                "basis_state_expectation: bit string must contain only '0'/'1'");
        }
        if (c == '1') bit_words[q / BitsPerWord] |= size_t(1) << (q % BitsPerWord);
    }

    T result = T{};
    for (size_t i = 0; i < get_length(); ++i) {
        bool has_xy = false;
        for (size_t k = 0; k < ArraySize; ++k) {
            if (x_data[k][i]) { has_xy = true; break; }
        }
        if (has_xy) continue;
        if (flags[i] & PHASE_MASK) continue;

        int parity = 0;
        for (size_t k = 0; k < ArraySize; ++k)
            parity ^= std::popcount(z_data[k][i] & bit_words[k]) & 1;

        const T sign = ((flags[i] & SIGN_MASK) ? T{-1} : T{1}) * (parity ? T{-1} : T{1});
        result += sign * value[i];
    }
    return result;
}

// Matrix trace of the sum: every non-identity Pauli string is traceless, so
// tr = 2^n * sum of identity-term coefficients. Terms with an imaginary
// phase flag are skipped (their contribution would be imaginary,
// incompatible with T); the sign flag is folded into the coefficient.
template <std::size_t N, typename T>
T PauliSum<N, T>::trace() const {
    T result = T{};
    for (size_t i = 0; i < get_length(); ++i) {
        bool identity = true;
        for (size_t k = 0; k < ArraySize; ++k) {
            if (x_data[k][i] | z_data[k][i]) { identity = false; break; }
        }
        if (!identity) continue;
        if (flags[i] & PHASE_MASK) continue;
        const T sign = (flags[i] & SIGN_MASK) ? T{-1} : T{1};
        result += sign * value[i];
    }
    return std::ldexp(result, static_cast<int>(num_qubits_));
}

// ---------------------------------------------------------------------------
// Truncation & algebra primitives (shared by Pauli propagation).

template <std::size_t N, typename T>
size_t PauliSum<N, T>::term_weight(size_t i) const {
    size_t w = 0;
    for (size_t k = 0; k < ArraySize; ++k)
        w += static_cast<size_t>(std::popcount(x_data[k][i] | z_data[k][i]));
    return w;
}

// In-place removal of all terms whose mask entry is 0.
template <std::size_t N, typename T>
void PauliSum<N, T>::compact(const std::vector<uint8_t>& keep_mask) {
    size_t w = 0;
    const size_t n = get_length();
    for (size_t i = 0; i < n; ++i) {
        if (!keep_mask[i]) continue;
        if (w != i) {
            for (size_t k = 0; k < ArraySize; ++k) {
                x_data[k][w] = x_data[k][i];
                z_data[k][w] = z_data[k][i];
            }
            flags[w] = flags[i];
            value[w] = value[i];
        }
        ++w;
    }
    for (size_t k = 0; k < ArraySize; ++k) {
        x_data[k].resize(w);
        z_data[k].resize(w);
    }
    flags.resize(w);
    value.resize(w);
}

// Canonical form: the sign flag is folded into the coefficient (flags 2 -> 0,
// 3 -> 1; only the imaginary bit remains, which a real T cannot absorb),
// duplicate terms are combined, and exact-zero coefficients are dropped.
// frobenius_norm2() and trim() assume this form.
template <std::size_t N, typename T>
void PauliSum<N, T>::canonicalize() {
    const size_t n = get_length();
    #pragma omp parallel for schedule(static) if (n >= (size_t{1} << 15))
    for (size_t i = 0; i < n; ++i) {
        if (flags[i] & SIGN_MASK) {
            flags[i] ^= SIGN_MASK;
            value[i] = -value[i];
        }
    }
    sort_and_combine();
    std::vector<uint8_t> keep(get_length());
    for (size_t i = 0; i < get_length(); ++i) keep[i] = (value[i] != T{});
    compact(keep);
}

template <std::size_t N, typename T>
void PauliSum<N, T>::trim(size_t max_terms) {
    const size_t n = get_length();
    if (n <= max_terms) return;
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + max_terms, idx.end(),
                     [&](size_t a, size_t b) {
                         return std::abs(value[a]) > std::abs(value[b]);
                     });
    std::vector<uint8_t> keep(n, 0);
    for (size_t i = 0; i < max_terms; ++i) keep[idx[i]] = 1;
    compact(keep);
}

// trim() that never drops terms whose (x, z) pattern appears in `keep` —
// e.g. the t = 0 observable during Heisenberg evolution, so the correlator
// tr(O(t) O(0)) can always be read off.
template <std::size_t N, typename T>
void PauliSum<N, T>::trim(size_t max_terms, const PauliSum<N, T>& keep) {
    const size_t n = get_length();
    if (n <= max_terms) return;

    std::unordered_map<PauliString<N>, char, PauliStringXZHash<N>, PauliStringXZEqual<N>> protect;
    for (size_t i = 0; i < keep.get_length(); ++i)
        protect.emplace(keep.get_pauli_string(i), 1);

    std::vector<uint8_t> keep_mask(n, 0);
    std::vector<size_t> idx;
    idx.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (protect.count(get_pauli_string(i))) keep_mask[i] = 1;
        else idx.push_back(i);
    }
    const size_t protected_count = n - idx.size();
    const size_t budget = max_terms > protected_count ? max_terms - protected_count : 0;
    if (budget < idx.size()) {
        std::nth_element(idx.begin(), idx.begin() + budget, idx.end(),
                         [&](size_t a, size_t b) {
                             return std::abs(value[a]) > std::abs(value[b]);
                         });
        idx.resize(budget);
    }
    for (size_t i : idx) keep_mask[i] = 1;
    compact(keep_mask);
}

template <std::size_t N, typename T>
void PauliSum<N, T>::cutoff(T eps) {
    const size_t n = get_length();
    std::vector<uint8_t> keep(n);
    #pragma omp parallel for schedule(static) if (n >= (size_t{1} << 15))
    for (size_t i = 0; i < n; ++i) keep[i] = (std::abs(value[i]) >= eps);
    compact(keep);
}

template <std::size_t N, typename T>
void PauliSum<N, T>::truncate_weight(size_t max_weight) {
    const size_t n = get_length();
    std::vector<uint8_t> keep(n);
    #pragma omp parallel for schedule(static) if (n >= (size_t{1} << 15))
    for (size_t i = 0; i < n; ++i) keep[i] = (term_weight(i) <= max_weight);
    compact(keep);
}

template <std::size_t N, typename T>
void PauliSum<N, T>::scale(T c) {
    for (T& v : value) v *= c;
}

// Normalized Frobenius norm squared, tr(O†O)/2^n. Valid on a canonicalized
// sum (distinct terms): each term contributes |c i^k|^2 = c^2.
template <std::size_t N, typename T>
T PauliSum<N, T>::frobenius_norm2() const {
    const size_t n = get_length();
    T s = T{};
    #pragma omp parallel for schedule(static) reduction(+ : s) \
        if (n >= (size_t{1} << 16))
    for (size_t i = 0; i < n; ++i) s += value[i] * value[i];
    return s;
}

// Re tr(A * B), the Hilbert-Schmidt pairing (no conjugation), computed
// without forming the product: tr(P_a P_b) = 2^n only when the (x, z)
// patterns match, contributing c_a c_b i^(k_a + k_b). Per pattern the
// complex weight w = c * i^k is accumulated as (Re, Im); a matching pair
// contributes 2^n * (Re_a Re_b - Im_a Im_b). O(len_A + len_B).
// For Hermitian A and B the trace is real, so nothing is discarded.
template <std::size_t N, typename T>
T trace_product(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    if (a.num_qubits() != b.num_qubits())
        throw std::invalid_argument("trace_product: qubit-count mismatch");

    std::unordered_map<PauliString<N>, std::pair<T, T>,
                       PauliStringXZHash<N>, PauliStringXZEqual<N>> weights;
    for (size_t i = 0; i < a.get_length(); ++i) {
        auto& [re, im] = weights[a.get_pauli_string(i)];
        switch (a.flags[i] & 0x03) {  // i^k: 0 -> +1, 1 -> +i, 2 -> -1, 3 -> -i
            case 0: re += a.value[i]; break;
            case 1: im += a.value[i]; break;
            case 2: re -= a.value[i]; break;
            case 3: im -= a.value[i]; break;
        }
    }

    T result = T{};
    for (size_t j = 0; j < b.get_length(); ++j) {
        auto it = weights.find(b.get_pauli_string(j));
        if (it == weights.end()) continue;
        const auto& [re, im] = it->second;
        switch (b.flags[j] & 0x03) {
            case 0: result += re * b.value[j]; break;
            case 1: result -= im * b.value[j]; break;
            case 2: result -= re * b.value[j]; break;
            case 3: result += im * b.value[j]; break;
        }
    }
    return std::ldexp(result, static_cast<int>(a.num_qubits()));
}

} // namespace paulib
