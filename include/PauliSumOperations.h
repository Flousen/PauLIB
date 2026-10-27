#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

#include "PauliSkeletons.h"
#include "PauliString.h"

namespace paulib {

inline bool isCloseOne(double value, double epsilon = 1e-14) {
    return std::abs(value - 1) <= epsilon;
}

inline bool isCloseToSqrtTwo(double value, double epsilon = 1e-14) {
    static const double sqrt_two_inv = 1 / std::sqrt(2);
    return std::abs(value - sqrt_two_inv) <= epsilon;
}

// Single-qubit rotations in the Heisenberg picture: apply_RZ(c, theta) maps
// every term Q to U^dagger Q U for U = RZ(theta) = exp(-i theta/2 Z_c), i.e.
// Q -> cos(theta) Q + i sin(theta) Z_c Q for Q anticommuting with Z_c; the
// same as evolve() with generator Z_c and angle theta. Likewise RX and RY.
template <std::size_t N, typename T>
void PauliSum<N, T>::apply_RZ(size_t c, real_t<T> angle) {
    const real_t<T> cos_t = std::cos(angle);
    const real_t<T> sin_t = std::sin(angle);
    const size_t n = get_length();

    PauliSum<N, T> new_;

    for (size_t i = 0; i < n; ++i) {
        if (commute_with_Z(i, c)) continue;

        const T t_val = value[i] * sin_t;
        value[i] *= cos_t;

        for (size_t k = 0; k < ArraySize; ++k) {
            new_.x_data[k].push_back(x_data[k][i]);
            new_.z_data[k].push_back(z_data[k][i]);
        }
        new_.flags.push_back(flags[i]);
        new_.value.push_back(t_val);

        new_.multiply_iZ(new_.get_length() - 1, c);
    }
    *this += new_;
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_RX(size_t c, real_t<T> angle) {
    const real_t<T> cos_t = std::cos(angle);
    const real_t<T> sin_t = std::sin(angle);
    const size_t n = get_length();

    PauliSum<N, T> new_;

    for (size_t i = 0; i < n; ++i) {
        if (commute_with_X(i, c)) continue;

        const T t_val = value[i] * sin_t;
        value[i] *= cos_t;

        for (size_t k = 0; k < ArraySize; ++k) {
            new_.x_data[k].push_back(x_data[k][i]);
            new_.z_data[k].push_back(z_data[k][i]);
        }
        new_.flags.push_back(flags[i]);
        new_.value.push_back(t_val);

        new_.multiply_iX(new_.get_length() - 1, c);
    }
    *this += new_;
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_RY(size_t c, real_t<T> angle) {
    const real_t<T> cos_t = std::cos(angle);
    const real_t<T> sin_t = std::sin(angle);
    const size_t n = get_length();

    PauliSum<N, T> new_;

    for (size_t i = 0; i < n; ++i) {
        if (commute_with_Y(i, c)) continue;

        const T t_val = value[i] * sin_t;
        value[i] *= cos_t;

        for (size_t k = 0; k < ArraySize; ++k) {
            new_.x_data[k].push_back(x_data[k][i]);
            new_.z_data[k].push_back(z_data[k][i]);
        }
        new_.flags.push_back(flags[i]);
        new_.value.push_back(t_val);

        new_.multiply_iY(new_.get_length() - 1, c);
    }
    *this += new_;
}

// Each rotation (P, theta) maps a term Q that anticommutes with P to
// cos(theta) Q + i sin(theta) P Q = cos(theta) Q - sin(theta) (Q ps) with
// ps = iP, and keeps the other terms. The Clifford angles 0, pi, +-pi/2 change
// the terms in place (transform_terms); a general angle adds one term per
// anticommuting term (expand_terms), and the new terms are merged into the
// sum (add_combine). Both skeletons are reproducible, so the result is the
// same for every thread count.
//
// Threads from 2^13 terms (the default policy starts at 2^15): a term costs a
// commutation check and a product, more than a value transform. With the
// default, 2^14 terms ran serially and 0.59 times as fast on 10 threads
// (Apple M4 Pro, 29 Oct 2026).
namespace detail {
inline ExecPolicy rotation_policy() {
    ExecPolicy pol = default_policy();
    pol.parallel_threshold = std::min(pol.parallel_threshold, size_t{1} << 13);
    return pol;
}
}  // namespace detail

template <std::size_t N, typename T>
void PauliSum<N, T>::evolve(const PauliSum<N, T>& rotations) {
    const size_t n = rotations.get_length();
    const ExecPolicy pol = detail::rotation_policy();

    for (size_t i = 0; i < n; ++i) {
        auto ps = rotations.get_pauli_string(i);
        ps.flip_phase();
        // The angle is real; for complex T its real part is used.
        const real_t<T> theta = std::real(rotations.value[i]);
        const real_t<T> cos_theta = std::cos(theta);
        const real_t<T> sin_theta = std::sin(theta);

        if (isCloseOne(cos_theta)) {
            // θ ≈ 0: identity, nothing to do.
            continue;
        } else if (isCloseOne(-cos_theta)) {
            // θ ≈ π: Q → -Q  (sin(π)=0, new term vanishes).
            transform_terms(*this, [&ps](const TermRef<N, T>& t) {
                if (anticommutes(t, ps)) t.flip_sign();
            }, pol);
        } else if (isCloseOne(sin_theta)) {
            // θ ≈ π/2: Q → i·P·Q = -(Q·ps)  since ps=iP and {P,Q}=0.
            transform_terms(*this, [&ps](const TermRef<N, T>& t) {
                if (!anticommutes(t, ps)) return;
                t.flip_sign();
                t.multiply(ps);
            }, pol);
        } else if (isCloseOne(-sin_theta)) {
            // θ ≈ -π/2: Q → -i·P·Q = +(Q·ps).
            transform_terms(*this, [&ps](const TermRef<N, T>& t) {
                if (anticommutes(t, ps)) t.multiply(ps);
            }, pol);
        } else {
            const PauliSum<N, T> added = expand_terms(
                *this, [&](const TermRef<N, T>& t, TermEmitter<N, T>& out) {
                    if (!anticommutes(t, ps)) return;
                    const T orig_val = t.value();
                    t.value_ref() = orig_val * cos_theta;
                    out.push(t, -sin_theta * orig_val).multiply(ps);
                }, pol);
            add_combine(added);
        }
    }
}

// Imaginary-time step for each generator (P, tau), tau the real part of its
// coefficient: O -> e^{-tau P/2} O e^{-tau P/2}, which is evolve() with
// theta = -i tau (Gomez-Lurbe and Perez, imaginary-time Pauli propagation).
// With P^2 = I, e^{-tau P} = cosh(tau) - sinh(tau) P, so a term Q that
// commutes with P becomes cosh(tau) Q - sinh(tau) P Q, and a term that
// anticommutes is unchanged (P moves through Q and the two factors cancel):
// the opposite case split of a real rotation. The step is not unitary, so
// the coefficients and the identity term (the trace) grow; normalise with
// trace() when a state is evolved. P must be Hermitian (phase +-1).
template <std::size_t N, typename T>
void PauliSum<N, T>::evolve_imaginary(const PauliSum<N, T>& generators) {
    const ExecPolicy pol = detail::rotation_policy();
    for (size_t i = 0; i < generators.get_length(); ++i) {
        const PauliString<N> ps = generators.get_pauli_string(i);
        if (ps.flags & PHASE_MASK)
            throw std::invalid_argument("evolve_imaginary: generator is not Hermitian");
        const real_t<T> tau = std::real(generators.value[i]);
        if (tau == real_t<T>{}) continue;
        const real_t<T> ch = std::cosh(tau);
        const real_t<T> sh = std::sinh(tau);
        // P Q = Q P for commuting strings, so the new term is Q·ps.
        const PauliSum<N, T> added = expand_terms(
            *this, [&](const TermRef<N, T>& t, TermEmitter<N, T>& out) {
                if (anticommutes(t, ps)) return;
                const T orig_val = t.value();
                t.value_ref() = orig_val * ch;
                out.push(t, -sh * orig_val).multiply(ps);
            }, pol);
        add_combine(added);
    }
}

template <std::size_t N, typename T>
inline void PauliSum<N, T>::multiply(const size_t i, const PauliString<N>& ps) {
    size_t total_c1 = 0;
    size_t total_c2 = 0;

    for (size_t k = 0; k < ArraySize; ++k) {
        const size_t ox = x_data[k][i];
        const size_t oz = z_data[k][i];
        const size_t bx = ps.x[k];
        const size_t bz = ps.z[k];

        x_data[k][i] = ox ^ bx;
        z_data[k][i] = oz ^ bz;

        const size_t xbz  = ox & bz;
        const size_t anti = (bx & oz) ^ xbz;

        total_c2 ^= (total_c1 ^ ox ^ oz ^ bx ^ bz ^ xbz) & anti;
        total_c1 ^= anti;
    }
    const int p1 = std::popcount(total_c1);
    const int p2 = __builtin_parityll(total_c2);

    flags[i] = (flags[i] + ps.flags + p1 + 2 * p2) & 0x03;
}

template <std::size_t N, typename T>
inline bool PauliSum<N, T>::commute(std::size_t i, const PauliString<N>& ps) {
    std::size_t acc = 0;
    for (std::size_t k = 0; k < ArraySize; ++k)
        acc ^= (x_data[k][i] & ps.z[k]) ^ (z_data[k][i] & ps.x[k]);
    return __builtin_parityll(acc) == 0;
}


template <std::size_t N, typename T>
void PauliSum<N, T>::elementwise_multiply(const PauliSum<N, T>& rhs) {
    const size_t n = get_length();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        size_t total_c1 = 0;
        size_t total_c2 = 0;

        for (size_t k = 0; k < ArraySize; ++k) {
            const size_t ox = x_data[k][i];
            const size_t oz = z_data[k][i];
            const size_t bx = rhs.x_data[k][i];
            const size_t bz = rhs.z_data[k][i];

            x_data[k][i] = ox ^ bx;
            z_data[k][i] = oz ^ bz;

            const size_t xbz  = ox & bz;
            const size_t anti = (bx & oz) ^ xbz;

            total_c2 ^= (total_c1 ^ ox ^ oz ^ bx ^ bz ^ xbz) & anti;
            total_c1 ^= anti;
        }

        flags[i] = (flags[i] + rhs.flags[i]
                    + static_cast<uint8_t>(std::popcount(total_c1))
                    + 2 * static_cast<uint8_t>(std::popcount(total_c2))) & 0x03;
        value[i] *= rhs.value[i];
    }
}

// k-outer, j-inner reordering: stride-1 access on every array, SIMD-vectorisable j loop.
// Temporary c1[j]/c2[j] arrays accumulate the phase across k words per output row;
// they are allocated per i iteration so the OMP-parallel outer loop stays race-free.
template <std::size_t N, typename T>
void PauliSum<N, T>::multiply(const PauliSum<N, T>& rhs) {
    const size_t n = get_length();
    const size_t m = rhs.get_length();

    auto orig_x    = std::move(x_data);
    auto orig_z    = std::move(z_data);
    auto orig_flags = std::move(flags);
    auto orig_val   = std::move(value);

    const size_t total_size = n * m;
    for (size_t k = 0; k < ArraySize; ++k) {
        x_data[k].assign(total_size, 0);
        z_data[k].assign(total_size, 0);
    }
    flags.assign(total_size, 0);
    value.assign(total_size, 0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; ++i) {
        const size_t base = i * n;

        std::vector<size_t> c1(n, 0);
        std::vector<size_t> c2(n, 0);

        for (size_t k = 0; k < ArraySize; ++k) {
            const size_t bx = rhs.x_data[k][i];
            const size_t bz = rhs.z_data[k][i];

            #pragma omp simd
            for (size_t j = 0; j < n; ++j) {
                const size_t ox  = orig_x[k][j];
                const size_t oz  = orig_z[k][j];
                x_data[k][base + j] = ox ^ bx;
                z_data[k][base + j] = oz ^ bz;

                const size_t xbz  = ox & bz;
                const size_t anti = (bx & oz) ^ xbz;
                c2[j] ^= (c1[j] ^ ox ^ oz ^ bx ^ bz ^ xbz) & anti;
                c1[j] ^= anti;
            }
        }

        #pragma omp simd
        for (size_t j = 0; j < n; ++j) {
            flags[base + j] = (orig_flags[j] + rhs.flags[i]
                + static_cast<uint8_t>(std::popcount(c1[j]))
                + 2 * static_cast<uint8_t>(__builtin_parityll(c2[j]))) & 0x03;
            value[base + j] = orig_val[j] * rhs.value[i];
        }
    }
}

// [A, B] and i[A, B] without forming the full product: commuting pairs are
// skipped and [P, Q] = 2PQ for anticommuting Pauli strings, with the ±i of
// the product carried by the phase flags. i[A, B] of two Hermitian
// real-coefficient sums is again Hermitian with real coefficients (the extra
// i shifts the phase exponent to even), which keeps the RK4/Lanczos drivers
// free of complex coefficients. The result is uncombined (up to one term per
// anticommuting pair); callers run canonicalize().
template <std::size_t N, typename T, bool MultiplyByI>
PauliSum<N, T> commutator_impl(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    const size_t n = a.get_length();
    const size_t m = b.get_length();

    std::vector<PauliString<N>> b_strings(m);
    for (size_t j = 0; j < m; ++j) b_strings[j] = b.get_pauli_string(j);

    PauliSum<N, T> out;
    out.num_qubits_ = std::max(a.num_qubits(), b.num_qubits());

    #pragma omp parallel
    {
        PauliSum<N, T> local;
        local.num_qubits_ = out.num_qubits_;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; ++i) {
            const PauliString<N> pa = a.get_pauli_string(i);
            for (size_t j = 0; j < m; ++j) {
                if (pa.commute(b_strings[j])) continue;
                PauliString<N> p = pa * b_strings[j];
                if constexpr (MultiplyByI) p.flags = (p.flags + 1) & 0x03;
                local.push_back(p, T{2} * a.value[i] * b.value[j]);
            }
        }

        #pragma omp critical
        add(out, local);
    }
    return out;
}

template <std::size_t N, typename T>
PauliSum<N, T> commutator(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    return commutator_impl<N, T, false>(a, b);
}

template <std::size_t N, typename T>
PauliSum<N, T> i_commutator(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    return commutator_impl<N, T, true>(a, b);
}

namespace detail {

// The order of sort_and_combine: X words, then Z words, then the flags.
template <std::size_t N, typename T>
inline bool term_less(const PauliSum<N, T>& s, size_t a, size_t b) {
    for (size_t k = 0; k < PauliSum<N, T>::ArraySize; ++k)
        if (s.x_data[k][a] != s.x_data[k][b]) return s.x_data[k][a] < s.x_data[k][b];
    for (size_t k = 0; k < PauliSum<N, T>::ArraySize; ++k)
        if (s.z_data[k][a] != s.z_data[k][b]) return s.z_data[k][a] < s.z_data[k][b];
    return s.flags[a] < s.flags[b];
}

template <std::size_t N, typename T>
inline bool same_term(const PauliSum<N, T>& s, size_t a, size_t b) {
    for (size_t k = 0; k < PauliSum<N, T>::ArraySize; ++k)
        if (s.x_data[k][a] != s.x_data[k][b] || s.z_data[k][a] != s.z_data[k][b]) return false;
    return s.flags[a] == s.flags[b];
}

inline constexpr size_t kCollateParallelThreshold = size_t{1} << 14;

// idx[lo..hi) = the terms lo..hi-1 in stable sorted order. A range that is
// sorted already is detected in one pass and not sorted. Large ranges are
// sorted in a FIXED number of chunks (independent of the thread count), which
// are then merged pairwise with stable merges, so the order is the same for
// every OMP_NUM_THREADS.
template <std::size_t N, typename T>
void stable_order(const PauliSum<N, T>& s, size_t lo, size_t hi, std::vector<size_t>& idx) {
    std::iota(idx.begin() + lo, idx.begin() + hi, lo);
    auto less = [&s](size_t a, size_t b) { return term_less(s, a, b); };
    bool sorted = true;
    for (size_t i = lo + 1; i < hi && sorted; ++i) sorted = !less(i, i - 1);
    if (sorted) return;

    const size_t n = hi - lo;
    constexpr size_t kChunks = 16;
    if (n >= kCollateParallelThreshold) {
        std::array<size_t, kChunks + 1> bounds;
        for (size_t c = 0; c <= kChunks; ++c) bounds[c] = lo + c * n / kChunks;

        #pragma omp parallel for schedule(dynamic)
        for (size_t c = 0; c < kChunks; ++c)
            std::stable_sort(idx.begin() + bounds[c], idx.begin() + bounds[c + 1], less);

        std::vector<size_t> tmp(hi);
        for (size_t width = 1; width < kChunks; width *= 2) {
            #pragma omp parallel for schedule(dynamic)
            for (size_t c = 0; c < kChunks; c += 2 * width) {
                const size_t a   = bounds[c];
                const size_t mid = bounds[std::min(c + width, kChunks)];
                const size_t b   = bounds[std::min(c + 2 * width, kChunks)];
                if (mid >= b) continue;
                std::merge(idx.begin() + a, idx.begin() + mid,
                           idx.begin() + mid, idx.begin() + b,
                           tmp.begin() + a, less);
                std::copy(tmp.begin() + a, tmp.begin() + b, idx.begin() + a);
            }
        }
    } else {
        std::stable_sort(idx.begin() + lo, idx.begin() + hi, less);
    }
}

// Replaces the terms by one term per run of equal terms along idx (a
// permutation of all terms in sorted order); a run's coefficients are summed
// in the order of idx, so the result is deterministic.
template <std::size_t N, typename T>
void combine_in_order(PauliSum<N, T>& s, const std::vector<size_t>& idx) {
    constexpr size_t W = PauliSum<N, T>::ArraySize;
    const size_t n = idx.size();
    std::vector<size_t> starts;
    starts.reserve(n);
    if (n > 0) starts.push_back(0);
    for (size_t i = 1; i < n; ++i)
        if (!same_term(s, idx[i], idx[i - 1])) starts.push_back(i);
    const size_t m = starts.size();

    typename PauliSum<N, T>::WordArray new_x, new_z;
    std::vector<uint8_t> new_flags(m);
    std::vector<T> new_value(m);
    for (size_t k = 0; k < W; ++k) {
        new_x[k].resize(m);
        new_z[k].resize(m);
    }
    auto segment = [&](size_t g) {
        const size_t first = idx[starts[g]];
        for (size_t k = 0; k < W; ++k) {
            new_x[k][g] = s.x_data[k][first];
            new_z[k][g] = s.z_data[k][first];
        }
        new_flags[g] = s.flags[first];
        const size_t end = (g + 1 < m) ? starts[g + 1] : n;
        T acc = T{};
        for (size_t i = starts[g]; i < end; ++i) acc += s.value[idx[i]];
        new_value[g] = acc;
    };
    // Explicit branch: an OpenMP if clause would start a team per call.
    if (n >= kCollateParallelThreshold) {
        #pragma omp parallel for schedule(static)
        for (size_t g = 0; g < m; ++g) segment(g);
    } else {
        for (size_t g = 0; g < m; ++g) segment(g);
    }

    s.x_data = std::move(new_x);
    s.z_data = std::move(new_z);
    s.flags = std::move(new_flags);
    s.value = std::move(new_value);
}

}  // namespace detail

// Sorts terms and merges duplicates by summing coefficients. The phase/sign
// flags are part of the term key, so e.g. +X and -X stay separate terms
// (folding the sign into the coefficient would lose the ±i phases, which a
// real-valued T cannot absorb). The order and therefore the floating-point
// summation order of duplicates are identical for every OMP_NUM_THREADS
// (detail::stable_order); a sum that is sorted already is not sorted again.
template <std::size_t N, typename T>
void PauliSum<N, T>::sort_and_combine() {
    const size_t n = get_length();
    if (n == 0) return;
    std::vector<size_t> idx(n);
    detail::stable_order(*this, 0, n, idx);
    detail::combine_in_order(*this, idx);
}

template <std::size_t N, typename T>
bool PauliSum<N, T>::is_sorted() const {
    for (size_t i = 1; i < get_length(); ++i)
        if (detail::term_less(*this, i, i - 1)) return false;
    return true;
}

// Merge-combine (P1): the stable order of the concatenation [*this, c rhs]
// is the stable merge of the stable orders of the two parts, so the result
// equals append + sort_and_combine bit for bit. A part that is sorted costs
// one pass instead of a sort; in evolve() the old terms usually are, so
// only the new terms are sorted.
template <std::size_t N, typename T>
void PauliSum<N, T>::add_combine(const PauliSum<N, T>& rhs, T c) {
    if (&rhs == this) {  // appending a vector to itself is undefined
        const PauliSum<N, T> copy = rhs;
        add_combine(copy, c);
        return;
    }
    const size_t m = get_length();
    if (c == T{1}) add(*this, rhs);
    else append_scaled(*this, c, rhs);
    const size_t n = get_length();
    if (n == 0) return;

    std::vector<size_t> idx(n);
    detail::stable_order(*this, 0, m, idx);
    detail::stable_order(*this, m, n, idx);
    auto less = [this](size_t a, size_t b) { return detail::term_less(*this, a, b); };
    if (m > 0 && m < n && less(idx[m], idx[m - 1])) {
        std::vector<size_t> merged(n);
        std::merge(idx.begin(), idx.begin() + m, idx.begin() + m, idx.end(), merged.begin(), less);
        idx = std::move(merged);
    }
    detail::combine_in_order(*this, idx);
}

template <std::size_t N, typename T>
inline bool PauliSum<N, T>::commute(std::size_t i, std::size_t j) {
    std::size_t acc = 0;
    for (std::size_t k = 0; k < ArraySize; ++k)
        acc ^= (x_data[k][i] & z_data[k][j]) ^ (z_data[k][i] & x_data[k][j]);
    return __builtin_parityll(acc) == 0;
}


// Thread-parallel greedy grouping: the input is split into one contiguous
// chunk per OMP thread, each thread groups its chunk independently, and a
// sequential merge folds local groups into the global list. The partition can
// differ across thread counts, but every group is still mutually commuting
// (qubit-wise for Commutation::qubitwise) and every index appears exactly
// once.
template <std::size_t N, typename T>
std::vector<std::vector<std::size_t>> PauliSum<N, T>::greedy_group(Commutation rule) {
    const std::size_t n = get_length();
    if (n == 0) return {};
    const SumView<N, T> view(*this);
    auto commute = [&view, rule](std::size_t i, std::size_t j) {
        const TermView<N, T> a{&view, i}, b{&view, j};
        return rule == Commutation::qubitwise ? qubitwise_commutes(a, b) : !anticommutes(a, b);
    };

    std::vector<std::vector<std::vector<std::size_t>>> local_groups;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();

        #pragma omp single
        local_groups.resize(num_threads);

        const std::size_t chunk_size = (n + num_threads - 1) / num_threads;
        const std::size_t start = tid * chunk_size;
        const std::size_t end = std::min(start + chunk_size, n);

        std::vector<std::vector<std::size_t>> my_groups;

        for (std::size_t i = start; i < end; ++i) {
            bool placed = false;
            for (auto& grp : my_groups) {
                bool ok = true;
                for (std::size_t j : grp) {
                    if (!commute(i, j)) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    grp.push_back(i);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                my_groups.push_back({i});
            }
        }
        local_groups[tid] = std::move(my_groups);
    }

    // Merge stage: append each local group to the first compatible global group.
    std::vector<std::vector<std::size_t>> final_groups;

    for (const auto& my_groups : local_groups) {
        for (const auto& lg : my_groups) {
            bool placed = false;
            for (auto& gg : final_groups) {
                bool ok = true;
                for (std::size_t l_idx : lg) {
                    for (std::size_t g_idx : gg) {
                        if (!commute(l_idx, g_idx)) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) break;
                }
                if (ok) {
                    gg.insert(gg.end(), lg.begin(), lg.end());
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                final_groups.push_back(lg);
            }
        }
    }

    return final_groups;
}

// A^dagger: (a i^k P)^dagger = conj(a) i^-k P, so the sign flips where the
// phase is imaginary and a complex coefficient is conjugated. The term order
// is kept; the sum is no longer sorted if a phase flips.
template <std::size_t N, typename T>
PauliSum<N, T> adjoint(PauliSum<N, T> sum) {
    transform_terms<true>(sum, [](const TermRef<N, T>& t) {
        uint8_t& f = t.flags_ref();
        f ^= static_cast<uint8_t>((f & 1u) << 1);
        if constexpr (is_complex_v<T>) t.value_ref() = std::conj(t.value_ref());
    });
    return sum;
}

} // namespace paulib
