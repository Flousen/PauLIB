#pragma once

// Qubit maps (QMP, thesis Ch. 4): kernels that change the qubits on which the
// strings act, not the Pauli operators on them. In the binary representation
// they move, remove or insert columns of the X and Z parts.
//
// - remap_qubits(sum, n_out, map): qubit q -> map[q]; permuting, embedding
//   into more qubits, removing qubits that every term leaves as I.
// - shift_qubits(sum, k): cyclic shift q -> q + k mod n (translation on a
//   periodic chain), word-wide.
// - shift_lattice(sum, Lx, Ly, dx, dy): translation on a periodic Lx x Ly
//   lattice, site (x, y) = qubit x + Lx y, word-wide.
// - restrict_qubits(sum, qubits): cuts every string to the given qubits (the
//   letters on all other qubits are dropped).
// - fix_qubits(sum, fixes): fixes qubits to an eigenvalue of X, Y or Z and
//   removes them (qubit tapering).
// - tensor_product(a, b): A (x) B on the qubits of a, then those of b.
//
// The letters of a string are stored directly (Y is the bit pair x = z = 1,
// not X Z), so moving or dropping a letter changes no phase. None of the
// kernels combines equal strings: restrict_qubits and fix_qubits can create
// duplicates, so call canonicalize() or sort_and_combine() afterwards.

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "PauliOps.h"
#include "PauliSkeletons.h"
#include "PauliSum.h"

namespace paulib {

// Marks a qubit that remap_qubits removes.
inline constexpr std::size_t kDropQubit = std::numeric_limits<std::size_t>::max();

namespace detail {

template <std::size_t W>
using Words = std::array<std::size_t, W>;

// v << s and v >> s over W words; bits shifted out of the array are lost.
template <std::size_t W>
inline Words<W> shl_words(const Words<W>& v, std::size_t s) {
    Words<W> r{};
    const std::size_t ws = s / 64, bs = s % 64;
    for (std::size_t k = ws; k < W; ++k) {
        std::size_t w = v[k - ws] << bs;
        if (bs != 0 && k > ws) w |= v[k - ws - 1] >> (64 - bs);
        r[k] = w;
    }
    return r;
}

template <std::size_t W>
inline Words<W> shr_words(const Words<W>& v, std::size_t s) {
    Words<W> r{};
    const std::size_t ws = s / 64, bs = s % 64;
    for (std::size_t k = 0; k + ws < W; ++k) {
        std::size_t w = v[k + ws] >> bs;
        if (bs != 0 && k + ws + 1 < W) w |= v[k + ws + 1] << (64 - bs);
        r[k] = w;
    }
    return r;
}

// Bits [lo, hi) set.
template <std::size_t W>
inline Words<W> bit_range(std::size_t lo, std::size_t hi) {
    Words<W> r{};
    for (std::size_t q = lo; q < hi; ++q) r[q / 64] |= std::size_t{1} << (q % 64);
    return r;
}

// Moves the letter on qubit q of every term to qubit map[q] and drops the
// letters of the qubits with map[q] = kDropQubit and of qubits >= map.size(),
// in place. The caller has checked the map (injective into s.num_qubits_ after
// the call).
template <std::size_t N, typename T>
void move_letters(PauliSum<N, T>& s, const std::vector<std::size_t>& map) {
    constexpr std::size_t W = PauliSum<N, T>::ArraySize;
    constexpr std::size_t B = PauliSum<N, T>::BitsPerWord;
    Words<W> keep{};
    for (std::size_t q = 0; q < map.size() && q < N; ++q)
        if (map[q] != kDropQubit) keep[q / B] |= std::size_t{1} << (q % B);
    transform_terms(s, [&keep, &map](const TermRef<N, T>& t) {
        Words<W> sx{}, sz{}, x{}, z{};
        for (std::size_t k = 0; k < W; ++k) {
            sx[k] = t.x(k);
            sz[k] = t.z(k);
        }
        for (std::size_t k = 0; k < W; ++k) {
            std::size_t support = (sx[k] | sz[k]) & keep[k];
            while (support) {
                const std::size_t b = static_cast<std::size_t>(std::countr_zero(support));
                support &= support - 1;
                const std::size_t r = map[k * B + b];
                x[r / B] |= ((sx[k] >> b) & 1) << (r % B);
                z[r / B] |= ((sz[k] >> b) & 1) << (r % B);
            }
        }
        for (std::size_t k = 0; k < W; ++k) {
            t.x_word(k) = x[k];
            t.z_word(k) = z[k];
        }
    });
}

// map[q] for qubits[j] = q is j, kDropQubit for the other qubits of n.
inline std::vector<std::size_t> keep_map(const std::vector<std::size_t>& qubits, std::size_t n,
                                         const char* who) {
    std::vector<std::size_t> map(n, kDropQubit);
    for (std::size_t j = 0; j < qubits.size(); ++j) {
        if (qubits[j] >= n) throw std::invalid_argument(std::string(who) + ": qubit out of range");
        if (map[qubits[j]] != kDropQubit) throw std::invalid_argument(std::string(who) + ": repeated qubit");
        map[qubits[j]] = j;
    }
    return map;
}

}  // namespace detail

// Moves qubit q of every term to qubit map[q] of a sum on n_out qubits.
// map has one entry per qubit of the sum and is injective; map[q] =
// kDropQubit removes qubit q, which must be the identity in every term.
// Embedding into more qubits, permuting the qubits and removing unused qubits
// are special cases. Relabelling qubits changes no phase, so flags and
// coefficients are kept. Throws std::invalid_argument for an invalid map or a
// term that acts on a removed qubit. To drop letters, use restrict_qubits.
template <std::size_t N, typename T>
PauliSum<N, T> remap_qubits(const PauliSum<N, T>& sum, std::size_t n_out,
                            const std::vector<std::size_t>& map) {
    constexpr std::size_t W = PauliSum<N, T>::ArraySize;
    constexpr std::size_t B = PauliSum<N, T>::BitsPerWord;
    if (n_out > N) throw std::invalid_argument("remap_qubits: qubit count exceeds capacity");
    if (map.size() != sum.num_qubits())
        throw std::invalid_argument("remap_qubits: map needs one entry per qubit");
    std::vector<uint8_t> used(n_out, 0);
    detail::Words<W> lost{};  // removed qubits and bits beyond num_qubits()
    for (std::size_t q = 0; q < N; ++q) {
        if (q < map.size() && map[q] != kDropQubit) {
            if (map[q] >= n_out || used[map[q]])
                throw std::invalid_argument("remap_qubits: map is not injective into n_out qubits");
            used[map[q]] = 1;
        } else {
            lost[q / B] |= std::size_t{1} << (q % B);
        }
    }
    const std::size_t bad = reduce_terms<std::size_t>(sum, [&lost](const TermView<N, T>& t) {
        std::size_t hit = 0;
        for (std::size_t k = 0; k < W; ++k) hit |= (t.x(k) | t.z(k)) & lost[k];
        return std::size_t{hit != 0};
    });
    if (bad) throw std::invalid_argument("remap_qubits: a term acts on a removed qubit");

    PauliSum<N, T> out = sum;
    out.num_qubits_ = n_out;
    detail::move_letters(out, map);
    return out;
}

// Translation on a periodic Lx x Ly lattice with qubit x + Lx y on site
// (x, y): the letter on (x, y) moves to ((x + dx) mod Lx, (y + dy) mod Ly).
// dx and dy may be negative. Requires Lx Ly = num_qubits(). Word-wide: a
// rotation of the rows by dy Lx bits, then two masked shifts by dx and
// Lx - dx within the rows; no loop over the qubits of a term. Order and
// coefficients are kept; the sum is no longer sorted.
template <std::size_t N, typename T>
PauliSum<N, T> shift_lattice(const PauliSum<N, T>& sum, std::size_t Lx, std::size_t Ly,
                             std::ptrdiff_t dx, std::ptrdiff_t dy) {
    constexpr std::size_t W = PauliSum<N, T>::ArraySize;
    const std::size_t n = sum.num_qubits();
    if (Lx == 0 || Ly == 0 || Lx * Ly != n)
        throw std::invalid_argument("shift_lattice: Lx * Ly must equal the qubit count");
    auto wrap = [](std::ptrdiff_t d, std::size_t L) {
        const std::ptrdiff_t m = d % static_cast<std::ptrdiff_t>(L);
        return static_cast<std::size_t>(m < 0 ? m + static_cast<std::ptrdiff_t>(L) : m);
    };
    const std::size_t sx = wrap(dx, Lx);
    const std::size_t sy = wrap(dy, Ly) * Lx;
    PauliSum<N, T> out = sum;
    if (sx == 0 && sy == 0) return out;

    const detail::Words<W> all = detail::bit_range<W>(0, n);
    detail::Words<W> hi{}, lo{};  // sites with x >= sx and x < sx
    for (std::size_t y = 0; y < Ly; ++y) {
        const auto h = detail::bit_range<W>(y * Lx + sx, (y + 1) * Lx);
        const auto l = detail::bit_range<W>(y * Lx, y * Lx + sx);
        for (std::size_t k = 0; k < W; ++k) {
            hi[k] |= h[k];
            lo[k] |= l[k];
        }
    }
    auto move = [&](const detail::Words<W>& v) {
        detail::Words<W> r = v;
        if (sy != 0) {
            const auto a = detail::shl_words(r, sy), b = detail::shr_words(r, n - sy);
            for (std::size_t k = 0; k < W; ++k) r[k] = (a[k] | b[k]) & all[k];
        }
        if (sx != 0) {
            const auto a = detail::shl_words(r, sx), b = detail::shr_words(r, Lx - sx);
            for (std::size_t k = 0; k < W; ++k) r[k] = (a[k] & hi[k]) | (b[k] & lo[k]);
        }
        return r;
    };
    transform_terms(out, [&move](const TermRef<N, T>& t) {
        detail::Words<W> x{}, z{};
        for (std::size_t k = 0; k < W; ++k) {
            x[k] = t.x(k);
            z[k] = t.z(k);
        }
        x = move(x);
        z = move(z);
        for (std::size_t k = 0; k < W; ++k) {
            t.x_word(k) = x[k];
            t.z_word(k) = z[k];
        }
    });
    return out;
}

// Cyclic shift on a periodic chain: the letter on qubit q moves to qubit
// (q + k) mod n, n = num_qubits(); k may be negative. The translation T_k of
// translation-symmetric operators (PauliStrings.jl, Loizeau et al.). The same
// as shift_lattice(sum, n, 1, k, 0).
template <std::size_t N, typename T>
PauliSum<N, T> shift_qubits(const PauliSum<N, T>& sum, std::ptrdiff_t k) {
    return shift_lattice(sum, sum.num_qubits(), 1, k, 0);
}

// Cuts every string to the qubits in `qubits`: qubits[j] becomes qubit j of a
// sum on qubits.size() qubits, and the letters on all other qubits are
// dropped (P_I (x) P_rest -> P_I). Coefficients and phases are kept, so terms
// that differ only outside `qubits` become equal; combine afterwards. This is
// the truncation T_I of Sun et al. and the restriction to the remaining
// qubits of the contextual-subspace method. To keep only the terms supported
// on `qubits` instead (a truncation, TRN), filter first:
// filter_terms(sum, pred) with a predicate on the support.
template <std::size_t N, typename T>
PauliSum<N, T> restrict_qubits(const PauliSum<N, T>& sum, const std::vector<std::size_t>& qubits) {
    const std::vector<std::size_t> map = detail::keep_map(qubits, sum.num_qubits(), "restrict_qubits");
    PauliSum<N, T> out = sum;
    out.num_qubits_ = qubits.size();
    detail::move_letters(out, map);
    return out;
}

// A qubit fixed to an eigenvalue of X, Y or Z.
struct FixedQubit {
    std::size_t qubit;
    char pauli;      // 'X', 'Y' or 'Z'
    int eigenvalue;  // +1 or -1
};

// Fixes each listed qubit q to the eigenvalue e of the Pauli sigma and
// removes it: every term must act on q with I or sigma (std::invalid_argument
// otherwise), sigma is replaced by e, and the remaining qubits are numbered in
// their order on num_qubits() - fixes.size() qubits. This is the last step of
// qubit tapering (Bravyi et al.): after the Clifford transformation, every
// term has I or Z (or I or X) on the tapered qubits. A sign e = -1 flips the
// sign bit of the term; coefficients are not touched. Terms that differ only
// on the fixed qubits become equal; combine afterwards.
template <std::size_t N, typename T>
PauliSum<N, T> fix_qubits(const PauliSum<N, T>& sum, const std::vector<FixedQubit>& fixes) {
    constexpr std::size_t W = PauliSum<N, T>::ArraySize;
    const std::size_t n = sum.num_qubits();
    std::vector<std::size_t> qubits;
    detail::Words<W> fixed{}, ex{}, ez{}, neg{};
    for (const FixedQubit& f : fixes) {
        if (f.pauli != 'X' && f.pauli != 'Y' && f.pauli != 'Z')
            throw std::invalid_argument("fix_qubits: pauli must be 'X', 'Y' or 'Z'");
        if (f.eigenvalue != 1 && f.eigenvalue != -1)
            throw std::invalid_argument("fix_qubits: eigenvalue must be +1 or -1");
        if (f.qubit >= n) throw std::invalid_argument("fix_qubits: qubit out of range");
        qubits.push_back(f.qubit);
        const std::size_t w = f.qubit / 64, bit = std::size_t{1} << (f.qubit % 64);
        if (fixed[w] & bit) throw std::invalid_argument("fix_qubits: repeated qubit");
        fixed[w] |= bit;
        if (f.pauli != 'Z') ex[w] |= bit;
        if (f.pauli != 'X') ez[w] |= bit;
        if (f.eigenvalue < 0) neg[w] |= bit;
    }
    const std::size_t bad = reduce_terms<std::size_t>(sum, [&](const TermView<N, T>& t) {
        std::size_t hit = 0;
        for (std::size_t k = 0; k < W; ++k)
            hit |= ((t.x(k) ^ ex[k]) | (t.z(k) ^ ez[k])) & (t.x(k) | t.z(k)) & fixed[k];
        return std::size_t{hit != 0};
    });
    if (bad) throw std::invalid_argument("fix_qubits: a term acts on a fixed qubit with another Pauli");

    std::vector<uint8_t> is_fixed(n, 0);
    for (std::size_t q : qubits) is_fixed[q] = 1;
    std::vector<std::size_t> map(n, kDropQubit);
    for (std::size_t q = 0, next = 0; q < n; ++q)
        if (!is_fixed[q]) map[q] = next++;

    PauliSum<N, T> out = sum;
    transform_terms(out, [&neg](const TermRef<N, T>& t) {
        std::size_t parity = 0;
        for (std::size_t k = 0; k < W; ++k)
            parity ^= static_cast<std::size_t>(std::popcount((t.x(k) | t.z(k)) & neg[k]));
        t.flags_ref() ^= static_cast<uint8_t>((parity & 1u) << 1);
    });
    out.num_qubits_ = n - fixes.size();
    detail::move_letters(out, map);
    return out;
}

// A (x) B on num_qubits(a) + num_qubits(b) qubits, the qubits of a first.
// Term i |b| + j is a_i (x) b_j (for_each_pair), uncombined; the terms are
// distinct if a and b are combined. Strings on disjoint qubits multiply
// without an extra phase, so the phases add and the coefficients multiply.
template <std::size_t N, typename T>
PauliSum<N, T> tensor_product(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    const std::size_t na = a.num_qubits(), nb = b.num_qubits(), n = na + nb;
    if (n > N) throw std::invalid_argument("tensor_product: qubit count exceeds capacity");
    std::vector<std::size_t> first(na), second(nb);
    std::iota(first.begin(), first.end(), std::size_t{0});
    std::iota(second.begin(), second.end(), na);
    const PauliSum<N, T> lo = remap_qubits(a, n, first);
    const PauliSum<N, T> hi = remap_qubits(b, n, second);
    PauliSum<N, T> out = for_each_pair(lo, hi, [](const TermView<N, T>& ta, const TermView<N, T>& tb,
                                                  const TermRef<N, T>& o) {
        for (std::size_t k = 0; k < PauliSum<N, T>::ArraySize; ++k) {
            o.x_word(k) = ta.x(k) | tb.x(k);
            o.z_word(k) = ta.z(k) | tb.z(k);
        }
        o.flags_ref() = static_cast<uint8_t>((ta.flags() + tb.flags()) & 3u);
        o.value_ref() = ta.value() * tb.value();
    });
    out.num_qubits_ = n;
    return out;
}

}  // namespace paulib
