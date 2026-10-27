#pragma once

// The commutation check (COM) in bulk, and COM used as a factor.
//
// - anticommutes_mask / commutes_with_all: one string against every term.
// - anticommutes(sum, i, j): one entry of the anticommutation matrix, which
//   is not stored (see there).
// - Pauli channels in the Heisenberg picture. The adjoint of a Pauli channel
//   maps every Pauli string to itself times a damping factor that depends
//   only on which error Paulis anticommute with it, so a channel is one pass
//   that rescales coefficients: damp_anticommuting, apply_pauli_channel,
//   apply_depolarizing, apply_dephasing, add_noise.
// - signed_norm2: tr(A† V A V) / 2^n for a Pauli V, the out-of-time-order
//   correlator of a Pauli operator without building V A V.
//
// The term-wise functions use the skeletons of PauliSkeletons.h; the output
// does not depend on the thread count.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "PauliOps.h"
#include "PauliSkeletons.h"
#include "PauliSum.h"

namespace paulib {

// mask[i] = 1 if term i anticommutes with ps, 0 if it commutes.
template <std::size_t N, typename T>
std::vector<uint8_t> anticommutes_mask(const PauliSum<N, T>& sum, const PauliString<N>& ps) {
    return term_mask(sum, [&ps](const TermView<N, T>& t) { return anticommutes(t, ps); });
}

// True if ps commutes with every term (e.g. a Z2 symmetry of a Hamiltonian).
// Hand-written (D3): it stops at the first anticommuting term.
template <std::size_t N, typename T>
bool commutes_with_all(const PauliSum<N, T>& sum, const PauliString<N>& ps) {
    const SumView<N, T> view(sum);
    for (std::size_t i = 0; i < view.length; ++i)
        if (anticommutes(TermView<N, T>{&view, i}, ps)) return false;
    return true;
}

// Entry (i, j) of the anticommutation matrix: true if terms i and j
// anticommute. Over GF(2), the matrix is A = G Lambda G^T, with G the rows
// (x_j | z_j) of the terms and Lambda the exchange of the X and Z parts, so the
// sum already holds A in M x 2n bits instead of M x M. A row of A, one term
// against all terms, is anticommutes_mask(sum, P_j), or word-wide on the
// qubit-major table anticommutes_mask(to_table(sum), P_j) (PauliTable.h), at
// O(weight(P_j) M / 64) word operations.
template <std::size_t N, typename T>
bool anticommutes(const PauliSum<N, T>& sum, std::size_t i, std::size_t j) {
    const SumView<N, T> view(sum);
    return anticommutes(TermView<N, T>{&view, i}, TermView<N, T>{&view, j});
}

// Multiplies the coefficient of every term that anticommutes with P by
// `factor`. The adjoint of the single-Pauli channel
// rho -> (1 - p) rho + p P rho P is damp_anticommuting(sum, P, 1 - 2p).
template <std::size_t N, typename T>
void damp_anticommuting(PauliSum<N, T>& sum, const PauliString<N>& P, real_t<T> factor) {
    transform_values(sum, [&P, factor](const TermView<N, T>& t) {
        return anticommutes(t, P) ? t.value() * factor : t.value();
    });
}

// Multiplies every term by factor[local Pauli on qubit q], indexed as
// x + 2z: 0 = I, 1 = X, 2 = Z, 3 = Y.
template <std::size_t N, typename T>
void damp_local(PauliSum<N, T>& sum, std::size_t q, const std::array<real_t<T>, 4>& factor) {
    if (q >= sum.num_qubits())
        throw std::out_of_range("damp_local: qubit index out of range");
    const std::size_t word = q / PauliSum<N, T>::BitsPerWord;
    const std::size_t bit = q % PauliSum<N, T>::BitsPerWord;
    transform_values(sum, [&factor, word, bit](const TermView<N, T>& t) {
        return t.value() * factor[local_pauli(t, word, bit)];
    });
}

// Heisenberg picture of the Pauli channel on qubit q,
// rho -> (1 - px - py - pz) rho + px X rho X + py Y rho Y + pz Z rho Z.
// A local Pauli s is damped by 1 - 2 * (sum of p_E over the E that
// anticommute with s): X by 1 - 2(py + pz), Y by 1 - 2(px + pz), Z by
// 1 - 2(px + py).
template <std::size_t N, typename T>
void apply_pauli_channel(PauliSum<N, T>& sum, std::size_t q,
                         real_t<T> px, real_t<T> py, real_t<T> pz) {
    using R = real_t<T>;
    damp_local(sum, q, std::array<R, 4>{R{1}, R{1} - 2 * (py + pz),
                                        R{1} - 2 * (px + py), R{1} - 2 * (px + pz)});
}

// Depolarizing channel rho -> (1 - p) rho + p tr(rho) I/2 on qubit q: X, Y
// and Z on q are damped by 1 - p (the convention of PauliPropagation.jl's
// DepolarizingNoise).
template <std::size_t N, typename T>
void apply_depolarizing(PauliSum<N, T>& sum, std::size_t q, real_t<T> p) {
    using R = real_t<T>;
    damp_local(sum, q, std::array<R, 4>{R{1}, R{1} - p, R{1} - p, R{1} - p});
}

// Dephasing channel rho -> (1 - p/2) rho + (p/2) Z rho Z on qubit q: X and Y
// on q are damped by 1 - p (PauliPropagation.jl's DephasingNoise).
template <std::size_t N, typename T>
void apply_dephasing(PauliSum<N, T>& sum, std::size_t q, real_t<T> p) {
    using R = real_t<T>;
    damp_local(sum, q, std::array<R, 4>{R{1}, R{1} - p, R{1}, R{1} - p});
}

// Depolarizing channel with parameter p on every qubit: each term is damped
// by (1 - p)^weight. One pass instead of num_qubits() calls of
// apply_depolarizing.
template <std::size_t N, typename T>
void add_noise(PauliSum<N, T>& sum, real_t<T> p) {
    std::vector<real_t<T>> pow(N + 1);
    pow[0] = real_t<T>{1};
    for (std::size_t w = 1; w < pow.size(); ++w) pow[w] = pow[w - 1] * (real_t<T>{1} - p);
    const real_t<T>* f = pow.data();
    transform_values(sum, [f](const TermView<N, T>& t) { return t.value() * f[weight(t)]; });
}

// tr(A† V A V) / 2^n for a Pauli string V without a phase: V A V keeps each
// string and flips the sign of the anticommuting ones, so the result is
// sum_j s_j |a_j|^2 with s_j = -1 if P_j anticommutes with V and +1
// otherwise. For a Hermitian A = W(t), this is the out-of-time-order
// correlator tr(W(t) V W(t) V) / 2^n. Like frobenius_norm2(), it assumes a
// collated sum (distinct strings). O(M W), no intermediate sum; reproducible
// for every thread count.
template <std::size_t N, typename T>
real_t<T> signed_norm2(const PauliSum<N, T>& sum, const PauliString<N>& V) {
    return reduce_terms<real_t<T>>(sum, [&V](const TermView<N, T>& t) {
        const real_t<T> a2 = abs2(t.value());
        return anticommutes(t, V) ? -a2 : a2;
    });
}

}  // namespace paulib
