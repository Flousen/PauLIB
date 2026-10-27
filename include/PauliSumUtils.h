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

#include "PauliSkeletons.h"
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
        result += scalar_to_string(value[i]) +
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
// For real T, terms with an imaginary phase flag are skipped (their
// contribution would be imaginary); for complex T the phase is included.
template <std::size_t N, typename T>
T PauliSum<N, T>::zero_state_expectation() const {
    // Any X bit (set for X and Y operators) means <0|…|0> = 0.
    return reduce_terms<T>(*this, [](const TermView<N, T>& t) {
        return (has_x(t) || !contributes(t)) ? T{} : weighted_value(t);
    });
}

// Compute <x| sum_i c_i P_i |x> for a computational basis state |x>, given
// as a string of '0'/'1' per qubit in the same left-to-right qubit order as
// PauliString's constructor (bits[0] is qubit 0). Generalizes
// zero_state_expectation() (equivalent to bits = "00...0").
//
// <x|I|x> = 1, <x|X|x> = <x|Y|x> = 0, <x|Z|x> = (-1)^bit: only terms with no
// X or Y contribute, and each '1' bit under a Z flips the sign once.
// For real T, terms with an imaginary phase flag are skipped (their
// contribution would be imaginary); for complex T the phase is included.
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

    return reduce_terms<T>(*this, [&bit_words](const TermView<N, T>& t) {
        if (has_x(t) || !contributes(t)) return T{};
        int parity = 0;
        for (size_t k = 0; k < ArraySize; ++k)
            parity ^= std::popcount(t.z(k) & bit_words[k]) & 1;
        const T term = weighted_value(t);
        return parity ? -term : term;
    });
}

// Matrix trace of the sum: every non-identity Pauli string is traceless, so
// tr = 2^n * sum of identity-term coefficients. For real T, terms with an
// imaginary phase flag are skipped (their contribution would be imaginary);
// the sign flag (for complex T, the whole phase) is folded into the
// coefficient.
template <std::size_t N, typename T>
T PauliSum<N, T>::trace() const {
    const T result = reduce_terms<T>(*this, [](const TermView<N, T>& t) {
        return (is_identity(t) && contributes(t)) ? weighted_value(t) : T{};
    });
    return result * std::ldexp(real_t<T>{1}, static_cast<int>(num_qubits_));
}

// Coefficient c of ps in the sum, i.e. the c with sum = c * ps + (other
// strings). Every term with the same (x, z) pattern contributes, so the sum
// need not be combined first; a term's phase is taken relative to ps's phase
// (sum {-X: 0.5} gives lookup("X") = -0.5 and lookup("-X") = 0.5). For real
// T, terms that differ from ps by a factor of +-i are skipped (their
// contribution would be imaginary); for complex T they contribute +-i times
// their coefficient. Returns 0 if ps does not occur.
template <std::size_t N, typename T>
T PauliSum<N, T>::lookup(const PauliString<N>& ps) const {
    T result = T{};
    for (size_t i = 0; i < get_length(); ++i) {
        bool match = true;
        for (size_t k = 0; k < ArraySize; ++k) {
            if (x_data[k][i] != ps.x[k] || z_data[k][i] != ps.z[k]) { match = false; break; }
        }
        if (!match) continue;
        // flags encode the exponent k of i^k; term = value * i^(k_t - k_q) * ps.
        const uint8_t rel = static_cast<uint8_t>((flags[i] - ps.flags) & 0x03);
        if (!phase_representable<T>(rel)) continue;
        result += phase_factor<T>(rel) * value[i];
    }
    return result;
}

template <std::size_t N, typename T>
T PauliSum<N, T>::lookup(const std::string& pauli_string) const {
    return lookup(PauliString<N>(pauli_string));
}

// ---------------------------------------------------------------------------
// Truncation & algebra primitives (shared by Pauli propagation).

template <std::size_t N, typename T>
size_t PauliSum<N, T>::term_weight(size_t i) const {
    const SumView<N, T> view(*this);
    return weight(TermView<N, T>{&view, i});
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

// Canonical form: the phase is folded into the coefficient, duplicate terms
// are combined, and exact-zero coefficients are dropped. For real T only the
// sign flag is folded (flags 2 -> 0, 3 -> 1; the imaginary bit remains, which
// a real T cannot absorb); for complex T the whole i^k is folded, so every
// flag is 0 afterwards and P and iP combine into one term.
// frobenius_norm2() and trim() assume this form.
template <std::size_t N, typename T>
void PauliSum<N, T>::canonicalize() {
    transform_terms(*this, [](const TermRef<N, T>& t) {
        uint8_t& f = t.flags_ref();
        if constexpr (is_complex_v<T>) {
            if (f) {
                t.value_ref() *= phase_factor<T>(f);
                f = 0;
            }
        } else if (f & SIGN_MASK) {
            f ^= SIGN_MASK;
            t.value_ref() = -t.value_ref();
        }
    });
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
void PauliSum<N, T>::cutoff(real_t<T> eps) {
    filter_terms(*this, [eps](const TermView<N, T>& t) { return std::abs(t.value()) >= eps; });
}

template <std::size_t N, typename T>
void PauliSum<N, T>::truncate_weight(size_t max_weight) {
    filter_terms(*this, [max_weight](const TermView<N, T>& t) { return weight(t) <= max_weight; });
}

template <std::size_t N, typename T>
void PauliSum<N, T>::scale(T c) {
    transform_values(*this, [c](const TermView<N, T>& t) { return t.value() * c; });
}

// Normalized Frobenius norm squared, tr(O†O)/2^n. Valid on a canonicalized
// sum (distinct terms): each term contributes |c i^k|^2 = |c|^2.
template <std::size_t N, typename T>
real_t<T> PauliSum<N, T>::frobenius_norm2() const {
    return reduce_terms<real_t<T>>(*this, [](const TermView<N, T>& t) { return abs2(t.value()); });
}

// l1 norm sum_j |a_j|, the lambda of qubitization and qDRIFT. Like
// frobenius_norm2(), it is the norm of the operator only on a canonicalized
// sum.
template <std::size_t N, typename T>
real_t<T> PauliSum<N, T>::l1_norm() const {
    return reduce_terms<real_t<T>>(*this, [](const TermView<N, T>& t) { return std::abs(t.value()); });
}

// h[w] = sum of |a_j|^2 over the terms of weight w, for w = 0..num_qubits():
// the weight distribution of operator spreading. Sums to frobenius_norm2().
// Reproducible for every thread count (reduce_terms_binned). A term acting on
// qubits beyond num_qubits() (possible only if the caller set bits there)
// extends the histogram as far as needed, up to the capacity N.
template <std::size_t N, typename T>
std::vector<real_t<T>> PauliSum<N, T>::weight_histogram() const {
    auto h = reduce_terms_binned<real_t<T>>(
        *this, N + 1, [](const TermView<N, T>& t) { return weight(t); },
        [](const TermView<N, T>& t) { return abs2(t.value()); });
    size_t len = num_qubits_ + 1;
    for (size_t w = len; w < h.size(); ++w)
        if (h[w] != real_t<T>{}) len = w + 1;
    h.resize(len);
    return h;
}

// <psi| sum |psi> for the product state with Bloch vector bloch[q] =
// (<X_q>, <Y_q>, <Z_q>) on qubit q: a term a i^k P contributes
// a i^k prod_q r_q(P_q), with r_q(I) = 1. bloch.size() must equal
// num_qubits(). As in zero_state_expectation(), a real T skips terms with an
// imaginary phase. The zero state is bloch[q] = (0, 0, 1) for every q.
template <std::size_t N, typename T>
T PauliSum<N, T>::product_state_expectation(
        const std::vector<std::array<real_t<T>, 3>>& bloch) const {
    if (bloch.size() != num_qubits_)
        throw std::invalid_argument(
            "product_state_expectation: " + std::to_string(bloch.size()) +
            " Bloch vectors for " + std::to_string(num_qubits_) + " qubits");

    T result = T{};
    for (size_t i = 0; i < get_length(); ++i) {
        if (!phase_representable<T>(flags[i])) continue;
        real_t<T> prod = real_t<T>{1};
        for (size_t k = 0; k < ArraySize && prod != real_t<T>{}; ++k) {
            size_t support = x_data[k][i] | z_data[k][i];
            while (support) {
                const size_t b = static_cast<size_t>(std::countr_zero(support));
                support &= support - 1;
                const size_t q = k * BitsPerWord + b;
                const bool xb = (x_data[k][i] >> b) & 1;
                const bool zb = (z_data[k][i] >> b) & 1;
                prod *= bloch[q][xb ? (zb ? 1 : 0) : 2];  // X, Y or Z
            }
        }
        result += phase_factor<T>(flags[i]) * value[i] * prod;
    }
    return result;
}

// tr(A * B), the Hilbert-Schmidt pairing (no conjugation), computed
// without forming the product: tr(P_a P_b) = 2^n only when the (x, z)
// patterns match, contributing c_a c_b i^(k_a + k_b). O(len_A + len_B).
// For real T, per pattern the complex weight w = c * i^k is accumulated as
// (Re, Im) and only Re tr(A * B) is returned, 2^n * (Re_a Re_b - Im_a Im_b);
// for Hermitian A and B the trace is real, so nothing is discarded. For
// complex T the full trace is returned. If both sums are sorted
// (is_sorted(), e.g. after canonicalize), the matching strings are found by
// a merge (merge_reduce) instead of a hash table.
template <std::size_t N, typename T>
T trace_product(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    if (a.num_qubits() != b.num_qubits())
        throw std::invalid_argument("trace_product: qubit-count mismatch");

    const T scale = T{std::ldexp(real_t<T>{1}, static_cast<int>(a.num_qubits()))};

    if (a.is_sorted() && b.is_sorted()) {
        return scale * merge_reduce<T>(a, b, [](const TermView<N, T>& ta, const TermView<N, T>& tb) {
            if constexpr (is_complex_v<T>) {
                return weighted_value(ta) * weighted_value(tb);
            } else {
                // Re(i^(ka + kb)) a b: +1, 0, -1, 0.
                const unsigned k = (ta.flags() + tb.flags()) & 3u;
                return (k & 1u) ? T{} : (k ? -ta.value() * tb.value() : ta.value() * tb.value());
            }
        });
    }

    if constexpr (is_complex_v<T>) {
        std::unordered_map<PauliString<N>, T,
                           PauliStringXZHash<N>, PauliStringXZEqual<N>> weights;
        for (size_t i = 0; i < a.get_length(); ++i)
            weights[a.get_pauli_string(i)] += phase_factor<T>(a.flags[i]) * a.value[i];

        T result = T{};
        for (size_t j = 0; j < b.get_length(); ++j) {
            auto it = weights.find(b.get_pauli_string(j));
            if (it == weights.end()) continue;
            result += it->second * phase_factor<T>(b.flags[j]) * b.value[j];
        }
        return result * scale;
    } else {
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
        return result * scale;
    }
}

// Inner product of two sums, <A, B> = tr(A† B) / 2^n (the normalized
// Hilbert-Schmidt or Frobenius inner product): the sum over matching
// strings of conj(w_a) w_b, with w = c * i^k. Unlike trace_product() it
// conjugates A and does not carry the factor 2^n; for Hermitian A,
// inner_product(a, b) = trace_product(a, b) / 2^n, and inner_product(a, a)
// = frobenius_norm2() on a canonicalized sum. For real T only the real part
// is returned. Not to be confused with PauliString::symplectic_product, the
// commutation check of two strings. Sorted inputs use a merge, as in
// trace_product().
template <std::size_t N, typename T>
T inner_product(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    if (a.num_qubits() != b.num_qubits())
        throw std::invalid_argument("inner_product: qubit-count mismatch");

    if (a.is_sorted() && b.is_sorted()) {
        return merge_reduce<T>(a, b, [](const TermView<N, T>& ta, const TermView<N, T>& tb) {
            if constexpr (is_complex_v<T>) {
                return std::conj(weighted_value(ta)) * weighted_value(tb);
            } else {
                // Re(conj(i^ka) i^kb) a b: +1, 0, -1, 0 for kb - ka.
                const unsigned k = (tb.flags() - ta.flags()) & 3u;
                return (k & 1u) ? T{} : (k ? -ta.value() * tb.value() : ta.value() * tb.value());
            }
        });
    }

    if constexpr (is_complex_v<T>) {
        std::unordered_map<PauliString<N>, T,
                           PauliStringXZHash<N>, PauliStringXZEqual<N>> weights;
        for (size_t i = 0; i < a.get_length(); ++i)
            weights[a.get_pauli_string(i)] += phase_factor<T>(a.flags[i]) * a.value[i];

        T result = T{};
        for (size_t j = 0; j < b.get_length(); ++j) {
            auto it = weights.find(b.get_pauli_string(j));
            if (it == weights.end()) continue;
            result += std::conj(it->second) * phase_factor<T>(b.flags[j]) * b.value[j];
        }
        return result;
    } else {
        // Re(conj(w_a) w_b) = Re_a Re_b + Im_a Im_b.
        std::unordered_map<PauliString<N>, std::pair<T, T>,
                           PauliStringXZHash<N>, PauliStringXZEqual<N>> weights;
        for (size_t i = 0; i < a.get_length(); ++i) {
            auto& [re, im] = weights[a.get_pauli_string(i)];
            const T v = (a.flags[i] & 0x02) ? -a.value[i] : a.value[i];
            if (a.flags[i] & 0x01) im += v; else re += v;
        }

        T result = T{};
        for (size_t j = 0; j < b.get_length(); ++j) {
            auto it = weights.find(b.get_pauli_string(j));
            if (it == weights.end()) continue;
            const auto& [re, im] = it->second;
            const T v = (b.flags[j] & 0x02) ? -b.value[j] : b.value[j];
            result += (b.flags[j] & 0x01) ? im * v : re * v;
        }
        return result;
    }
}

} // namespace paulib
