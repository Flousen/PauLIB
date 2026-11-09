#pragma once

// Linear algebra over GF(2) on Pauli strings, without a separate bit matrix.
//
// A set of M strings on n qubits is the matrix G in GF(2)^(M x 2n) with rows
// (x_j | z_j): column c < n is the X bit of qubit c, column n + c the Z bit.
// A PauliSum already stores these rows, word by word, so the elimination runs
// directly on the X and Z words of the terms of a copy of the sum; phases and
// coefficients are ignored. Results that are sets of strings (row basis, null
// space, commutant) are returned as Pauli sums with phase +1 and coefficient 1.
//
// - gf2_rref / gf2_rank / gf2_row_basis / gf2_nullspace: Gaussian elimination.
// - GF2Basis: incremental reduced basis of strings that records which inputs
//   combine to a basis string (span tests and stabilizer phases).
// - Pauli-level functions: pauli_rank, independent_subset, in_span,
//   commutant (the Z2 symmetries of qubit tapering), stabilizer_expectation.
//
// The anticommutation matrix A = G Lambda G^T needs no storage either: an
// entry is anticommutes(sum, i, j), a row anticommutes_mask on the sum or on
// its qubit-major table (PauliSumCommutation.h, PauliTable.h).

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "PauliString.h"
#include "PauliSum.h"

namespace paulib {

namespace detail {

// Word array and bit of column c of the matrix G of a sum with n qubits.
template <std::size_t N, typename T>
inline std::vector<std::size_t>& gf2_column(PauliSum<N, T>& s, std::size_t c, std::size_t n) {
    return c < n ? s.x_data[c / 64] : s.z_data[(c - n) / 64];
}
inline std::size_t gf2_bit(std::size_t c, std::size_t n) { return (c < n ? c : c - n) % 64; }

template <std::size_t N, typename T>
inline void swap_terms(PauliSum<N, T>& s, std::size_t a, std::size_t b) {
    if (a == b) return;
    for (std::size_t k = 0; k < PauliSum<N, T>::ArraySize; ++k) {
        std::swap(s.x_data[k][a], s.x_data[k][b]);
        std::swap(s.z_data[k][a], s.z_data[k][b]);
    }
    std::swap(s.flags[a], s.flags[b]);
    std::swap(s.value[a], s.value[b]);
}

// Column c of a single string.
template <std::size_t N>
inline bool string_bit(const PauliString<N>& p, std::size_t c, std::size_t n) {
    const std::size_t q = c < n ? c : c - n;
    const std::size_t w = (c < n ? p.x : p.z)[q / 64];
    return (w >> (q % 64)) & 1;
}

template <std::size_t N>
inline void set_string_bit(PauliString<N>& p, std::size_t c, std::size_t n) {
    const std::size_t q = c < n ? c : c - n;
    (c < n ? p.x : p.z)[q / 64] |= std::size_t(1) << (q % 64);
}

// First nonzero column of a string in the order X bits, then Z bits; 2n if
// the string is the identity.
template <std::size_t N>
inline std::size_t first_column(const PauliString<N>& p, std::size_t n) {
    for (std::size_t k = 0; k < PauliString<N>::ArraySize; ++k)
        if (p.x[k]) return k * 64 + static_cast<std::size_t>(std::countr_zero(p.x[k]));
    for (std::size_t k = 0; k < PauliString<N>::ArraySize; ++k)
        if (p.z[k]) return n + k * 64 + static_cast<std::size_t>(std::countr_zero(p.z[k]));
    return 2 * n;
}

// p ^= q on the X and Z parts; the phase of p is not changed.
template <std::size_t N>
inline void xor_string(PauliString<N>& p, const PauliString<N>& q) {
    for (std::size_t k = 0; k < PauliString<N>::ArraySize; ++k) {
        p.x[k] ^= q.x[k];
        p.z[k] ^= q.z[k];
    }
}

// Rows above which the elimination of one pivot column uses threads.
inline constexpr std::size_t kGF2ParallelWords = std::size_t{1} << 16;
inline constexpr std::size_t kGF2Chunk = 4096;

}  // namespace detail

// Reduced row echelon form of the matrix G of the terms, in place. Returns the
// pivot column of each of the first rank terms; the terms rank..end are the
// identity afterwards. Row swaps move the flags and coefficients with their
// terms; the elimination itself changes only the X and Z words, so after the
// call the flags and coefficients no longer describe the strings.
//
// For each pivot column, the terms are processed in chunks: first a mask of
// the terms that have the pivot bit, then, word array by word array, an XOR
// of the pivot term into the masked terms. Both loops run over consecutive
// terms (stride 1), so they vectorise. Above detail::kGF2ParallelWords words
// the chunks are spread over the threads; XOR is exact, so the result does not
// depend on the thread count.
template <std::size_t N, typename T>
std::vector<std::size_t> gf2_rref(PauliSum<N, T>& s) {
    constexpr std::size_t W = PauliSum<N, T>::ArraySize;
    const std::size_t n = s.num_qubits();
    const std::size_t M = s.get_length();
    std::vector<std::size_t> pivots;
    std::size_t r = 0;
    for (std::size_t c = 0; c < 2 * n && r < M; ++c) {
        const std::vector<std::size_t>& col = detail::gf2_column(s, c, n);
        const std::size_t bit = detail::gf2_bit(c, n);
        std::size_t p = r;
        while (p < M && !((col[p] >> bit) & 1)) ++p;
        if (p == M) continue;
        detail::swap_terms(s, r, p);

        std::array<std::size_t, W> px{}, pz{};
        for (std::size_t k = 0; k < W; ++k) {
            px[k] = s.x_data[k][r];
            pz[k] = s.z_data[k][r];
        }
        const std::size_t* colp = col.data();
        auto eliminate = [&](std::size_t lo, std::size_t hi) {
            std::size_t sel[detail::kGF2Chunk];
            for (std::size_t i = lo; i < hi; ++i)
                sel[i - lo] = std::size_t(0) - ((colp[i] >> bit) & 1);
            if (r >= lo && r < hi) sel[r - lo] = 0;
            for (std::size_t k = 0; k < W; ++k) {
                if (px[k]) {
                    std::size_t* xs = s.x_data[k].data();
                    for (std::size_t i = lo; i < hi; ++i) xs[i] ^= sel[i - lo] & px[k];
                }
                if (pz[k]) {
                    std::size_t* zs = s.z_data[k].data();
                    for (std::size_t i = lo; i < hi; ++i) zs[i] ^= sel[i - lo] & pz[k];
                }
            }
        };
        const std::size_t chunks = (M + detail::kGF2Chunk - 1) / detail::kGF2Chunk;
        // Explicit branch: an OpenMP `if` clause costs about 0.1 us per pivot
        // even when false (Apple M4 Pro, 28 Oct 2026).
        if (M * 2 * W >= detail::kGF2ParallelWords && chunks > 1) {
            #pragma omp parallel for schedule(static)
            for (std::size_t b = 0; b < chunks; ++b)
                eliminate(b * detail::kGF2Chunk, std::min(M, (b + 1) * detail::kGF2Chunk));
        } else {
            for (std::size_t b = 0; b < chunks; ++b)
                eliminate(b * detail::kGF2Chunk, std::min(M, (b + 1) * detail::kGF2Chunk));
        }
        pivots.push_back(c);
        ++r;
    }
    return pivots;
}

template <std::size_t N, typename T>
std::size_t gf2_rank(PauliSum<N, T> s) { return gf2_rref(s).size(); }

// Basis of the row space: rank strings with phase +1 and coefficient 1.
template <std::size_t N, typename T>
PauliSum<N, T> gf2_row_basis(PauliSum<N, T> s) {
    const std::size_t rank = gf2_rref(s).size();
    PauliSum<N, T> b;
    b.num_qubits_ = s.num_qubits();
    for (std::size_t i = 0; i < rank; ++i) {
        PauliString<N> ps = s.get_pauli_string(i);
        ps.flags = 0;
        b.push_back(ps, T{1});
    }
    return b;
}

// Basis of the null space {v in GF(2)^2n : G v = 0}, with the ordinary dot
// product of the rows (x_j | z_j) with v = (v_x | v_z). Each basis vector is
// returned as the string with X part v_x and Z part v_z; there are 2n - rank.
template <std::size_t N, typename T>
PauliSum<N, T> gf2_nullspace(PauliSum<N, T> s) {
    const std::size_t n = s.num_qubits();
    const std::vector<std::size_t> pivots = gf2_rref(s);
    std::vector<uint8_t> is_pivot(2 * n, 0);
    for (std::size_t c : pivots) is_pivot[c] = 1;

    PauliSum<N, T> ns;
    ns.num_qubits_ = n;
    for (std::size_t f = 0; f < 2 * n; ++f) {
        if (is_pivot[f]) continue;
        PauliString<N> v{};
        v.num_qubits_ = static_cast<uint32_t>(n);
        detail::set_string_bit(v, f, n);
        const std::vector<std::size_t>& col = detail::gf2_column(s, f, n);
        const std::size_t bit = detail::gf2_bit(f, n);
        for (std::size_t i = 0; i < pivots.size(); ++i)
            if ((col[i] >> bit) & 1) detail::set_string_bit(v, pivots[i], n);
        ns.push_back(v, T{1});
    }
    return ns;
}

// Incrementally built reduced basis of strings (phases ignored). Every stored
// string has a pivot column that is zero in all other stored strings, so
// reducing a string is one pass over the basis. Each stored string also
// records, as a bit set over input indices, which inserted strings XOR to it.
template <std::size_t N>
class GF2Basis {
public:
    GF2Basis(std::size_t n_qubits, std::size_t max_inputs)
        : n_(n_qubits), cw_((max_inputs + 63) / 64), max_inputs_(max_inputs) {}

    std::size_t rank() const { return pivots_.size(); }

    // Reduces v in place against the basis; combo (ceil(max_inputs/64) words)
    // becomes the set of inputs whose XOR equals the removed part. Returns
    // true if v is in the span (v is the identity afterwards).
    bool reduce(PauliString<N>& v, std::vector<std::size_t>& combo) const {
        combo.assign(cw_, 0);
        for (std::size_t b = 0; b < pivots_.size(); ++b) {
            if (!detail::string_bit(v, pivots_[b], n_)) continue;
            detail::xor_string(v, rows_[b]);
            for (std::size_t k = 0; k < cw_; ++k) combo[k] ^= combos_[b][k];
        }
        return is_identity(v);
    }

    // Inserts input string `index` (< max_inputs). Returns false, and leaves
    // the basis unchanged, if v is already in the span.
    bool insert(PauliString<N> v, std::size_t index) {
        if (index >= max_inputs_)
            throw std::out_of_range("GF2Basis::insert: input index out of range");
        std::vector<std::size_t> combo;
        if (reduce(v, combo)) return false;
        combo[index / 64] ^= std::size_t(1) << (index % 64);
        const std::size_t p = detail::first_column(v, n_);

        // Keep the basis reduced: clear the new pivot from the stored strings.
        for (std::size_t b = 0; b < rows_.size(); ++b) {
            if (!detail::string_bit(rows_[b], p, n_)) continue;
            detail::xor_string(rows_[b], v);
            for (std::size_t k = 0; k < cw_; ++k) combos_[b][k] ^= combo[k];
        }
        rows_.push_back(v);
        combos_.push_back(std::move(combo));
        pivots_.push_back(p);
        return true;
    }

private:
    std::size_t n_;
    std::size_t cw_;
    std::size_t max_inputs_;
    std::vector<PauliString<N>> rows_;
    std::vector<std::vector<std::size_t>> combos_;
    std::vector<std::size_t> pivots_;
};

namespace detail {
template <std::size_t N>
PauliSum<N, double> sum_of(const std::vector<PauliString<N>>& strings, std::size_t n) {
    if (n > N) throw std::invalid_argument("GF(2): qubit count exceeds capacity");
    PauliSum<N, double> s;
    s.num_qubits_ = n;
    s.reserve(strings.size());
    for (const auto& p : strings) s.push_back(p, 1.0);
    return s;
}
}  // namespace detail

// Rank of the strings as vectors of GF(2)^2n (phases ignored).
template <std::size_t N, typename T>
std::size_t pauli_rank(const PauliSum<N, T>& sum) { return gf2_rank(sum); }

template <std::size_t N>
std::size_t pauli_rank(const std::vector<PauliString<N>>& strings, std::size_t n) {
    return gf2_rank(detail::sum_of(strings, n));
}

// Indices of a maximal independent subset, chosen greedily in input order.
template <std::size_t N>
std::vector<std::size_t> independent_subset(const std::vector<PauliString<N>>& strings,
                                            std::size_t n) {
    GF2Basis<N> basis(n, strings.size());
    std::vector<std::size_t> out;
    for (std::size_t j = 0; j < strings.size(); ++j)
        if (basis.insert(strings[j], j)) out.push_back(j);
    return out;
}

// True if ps is, up to a phase, a product of the given strings.
template <std::size_t N>
bool in_span(const std::vector<PauliString<N>>& strings, const PauliString<N>& ps,
             std::size_t n) {
    GF2Basis<N> basis(n, strings.size());
    for (std::size_t j = 0; j < strings.size(); ++j) basis.insert(strings[j], j);
    PauliString<N> v = ps;
    std::vector<std::size_t> combo;
    return basis.reduce(v, combo);
}

template <std::size_t N, typename T>
bool in_span(const PauliSum<N, T>& sum, const PauliString<N>& ps) {
    std::vector<PauliString<N>> strings(sum.get_length());
    for (std::size_t j = 0; j < sum.get_length(); ++j) strings[j] = sum.get_pauli_string(j);
    return in_span(strings, ps, sum.num_qubits());
}

// Basis of the commutant: strings P with [P, S_j] = 0 for every term S_j.
// P = (x | z) commutes with S_j iff x . z_j + z . x_j = 0, so the commutant is
// the null space of the matrix with rows (z_j | x_j): the terms with their X
// and Z parts exchanged. It has dimension 2n - rank. Applied to the terms of a
// Hamiltonian, the result spans its Z2 symmetries (qubit tapering).
template <std::size_t N, typename T>
PauliSum<N, T> commutant(const PauliSum<N, T>& sum) {
    PauliSum<N, T> swapped = sum;
    std::swap(swapped.x_data, swapped.z_data);
    return gf2_nullspace(std::move(swapped));
}

template <std::size_t N>
std::vector<PauliString<N>> commutant(const std::vector<PauliString<N>>& strings,
                                      std::size_t n) {
    const auto c = commutant(detail::sum_of(strings, n));
    std::vector<PauliString<N>> out(c.get_length());
    for (std::size_t j = 0; j < c.get_length(); ++j) out[j] = c.get_pauli_string(j);
    return out;
}

// tr(rho A) for the stabilizer state rho = 2^-n sum_{s in S} s of the group S
// generated by `generators` (Hermitian, independent, mutually commuting;
// signs in their flags). With n generators rho is the pure stabilizer state;
// with fewer it is the maximally mixed state on the code space. A term
// a i^k P contributes a i^k <P>, where <P> = i^-m if the product of the
// generators selected by GF(2) elimination equals i^m P, and 0 if P is not in
// S up to a phase. As in zero_state_expectation(), a real T skips terms whose
// contribution is imaginary.
template <std::size_t N, typename T>
T stabilizer_expectation(const PauliSum<N, T>& sum,
                         const std::vector<PauliString<N>>& generators) {
    const std::size_t n = sum.num_qubits();
    const std::size_t k = generators.size();
    GF2Basis<N> basis(n, k);
    for (std::size_t j = 0; j < k; ++j) {
        if (generators[j].flags & 0x01)
            throw std::invalid_argument("stabilizer_expectation: generator is not Hermitian");
        for (std::size_t l = 0; l < j; ++l)
            if (!generators[j].commute(generators[l]))
                throw std::invalid_argument("stabilizer_expectation: generators do not commute");
        if (!basis.insert(generators[j], j))
            throw std::invalid_argument("stabilizer_expectation: generators are dependent");
    }

    T result = T{};
    std::vector<std::size_t> combo;
    for (std::size_t i = 0; i < sum.get_length(); ++i) {
        PauliString<N> v = sum.get_pauli_string(i);
        if (!basis.reduce(v, combo)) continue;

        PauliString<N> prod{};
        prod.num_qubits_ = static_cast<uint32_t>(n);
        for (std::size_t w = 0; w < combo.size(); ++w) {
            std::size_t bits = combo[w];
            while (bits) {
                const std::size_t j = w * 64 + static_cast<std::size_t>(std::countr_zero(bits));
                bits &= bits - 1;
                prod *= generators[j];
            }
        }
        // term = a i^k_t P0 and prod = i^m P0, so <term> = a i^(k_t - m).
        const uint8_t rel = static_cast<uint8_t>((sum.flags[i] - prod.flags) & 0x03);
        if (!phase_representable<T>(rel)) continue;
        result += phase_factor<T>(rel) * sum.value[i];
    }
    return result;
}

}  // namespace paulib
