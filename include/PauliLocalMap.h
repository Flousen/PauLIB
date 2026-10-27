#pragma once

// Non-Clifford conjugation with a gate or a channel on k qubits (ROT, thesis
// Ch. 4), in the Heisenberg picture. A Pauli rotation is the special case
// that evolve() handles; a general gate on k qubits maps a string to a linear
// combination of up to 4^k strings that differ only on these qubits (a CCZ
// gate maps X on a target qubit to four strings, a single-qubit rotation acts
// on (X, Y, Z) as a rotation in three dimensions).
//
// - heisenberg_ptm(kraus, k): the Pauli transfer matrix of the Heisenberg
//   action E^dagger(P) = sum_K K^dagger P K of a gate (one Kraus operator U)
//   or a channel (several, e.g. amplitude damping) on k <= 4 qubits.
// - apply_ptm(sum, qubits, ptm): O -> E^dagger(O) with the map acting on
//   `qubits`; on expand_terms and add_combine, as evolve().
//
// Local index of a string on the k qubits of a map: digit j (bits 2j, 2j+1)
// is the letter on qubits[j], coded x + 2z (0 = I, 1 = X, 2 = Z, 3 = Y) as in
// local_pauli(). Dense matrices are row-major, 2^k x 2^k, with bit j of the
// basis index for qubits[j] (qubit 0 least significant).

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "PauliOps.h"
#include "PauliSkeletons.h"
#include "PauliSum.h"

namespace paulib {

using DenseMatrix = std::vector<std::complex<double>>;  // row-major 2^k x 2^k

// E^dagger(P_in) = sum_out at(out, in) P_out for the local strings of k qubits.
struct PauliTransferMatrix {
    std::size_t k = 0;
    std::vector<double> r;  // column-major: r[in * dim() + out]

    std::size_t dim() const { return std::size_t{1} << (2 * k); }
    double at(std::size_t out, std::size_t in) const { return r[in * dim() + out]; }
    double& at(std::size_t out, std::size_t in) { return r[in * dim() + out]; }
};

inline constexpr std::size_t kMaxLocalMapQubits = 4;

namespace detail {

// Dense matrix of the local string with index p on k qubits (Y = i X Z).
inline DenseMatrix local_pauli_matrix(std::size_t p, std::size_t k) {
    const std::size_t d = std::size_t{1} << k;
    std::size_t xm = 0, zm = 0;
    for (std::size_t j = 0; j < k; ++j) {
        xm |= ((p >> (2 * j)) & 1) << j;
        zm |= ((p >> (2 * j + 1)) & 1) << j;
    }
    static const std::complex<double> ipow[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    const std::complex<double> phase = ipow[std::popcount(xm & zm) & 3];
    DenseMatrix m(d * d, {0.0, 0.0});
    for (std::size_t b = 0; b < d; ++b)
        m[(b ^ xm) * d + b] = (std::popcount(zm & b) & 1) ? -phase : phase;
    return m;
}

}  // namespace detail

// The Pauli transfer matrix of the Heisenberg action of a channel with Kraus
// operators K_m on k qubits (1 <= k <= 4): at(out, in) =
// tr(P_out sum_m K_m^dagger P_in K_m) / 2^k. For a gate U, pass {U}; then
// apply_ptm gives U^dagger O U, the convention of evolve() and apply_*. The
// entries are real for every Hermiticity-preserving map; an imaginary part
// above 1e-10 throws std::invalid_argument, and entries below 1e-14 in
// magnitude are set to 0 so that apply_ptm creates no rounding terms.
inline PauliTransferMatrix heisenberg_ptm(const std::vector<DenseMatrix>& kraus, std::size_t k) {
    if (k == 0 || k > kMaxLocalMapQubits)
        throw std::invalid_argument("heisenberg_ptm: k must be between 1 and 4");
    const std::size_t d = std::size_t{1} << k;
    for (const auto& K : kraus)
        if (K.size() != d * d) throw std::invalid_argument("heisenberg_ptm: Kraus operator is not 2^k x 2^k");
    PauliTransferMatrix ptm;
    ptm.k = k;
    const std::size_t D = ptm.dim();
    ptm.r.assign(D * D, 0.0);
    DenseMatrix tmp(d * d), img(d * d);
    for (std::size_t in = 0; in < D; ++in) {
        const DenseMatrix P = detail::local_pauli_matrix(in, k);
        std::fill(img.begin(), img.end(), std::complex<double>{});
        for (const auto& K : kraus) {
            // tmp = P K, img += K^dagger tmp
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t b = 0; b < d; ++b) {
                    std::complex<double> s{};
                    for (std::size_t c = 0; c < d; ++c) s += P[a * d + c] * K[c * d + b];
                    tmp[a * d + b] = s;
                }
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t b = 0; b < d; ++b) {
                    std::complex<double> s{};
                    for (std::size_t c = 0; c < d; ++c) s += std::conj(K[c * d + a]) * tmp[c * d + b];
                    img[a * d + b] += s;
                }
        }
        for (std::size_t out = 0; out < D; ++out) {
            const DenseMatrix Q = detail::local_pauli_matrix(out, k);
            std::complex<double> tr{};
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t b = 0; b < d; ++b)
                    if (Q[a * d + b] != std::complex<double>{}) tr += Q[a * d + b] * img[b * d + a];
            tr /= static_cast<double>(d);
            if (std::abs(tr.imag()) > 1e-10)
                throw std::invalid_argument("heisenberg_ptm: map does not preserve Hermiticity");
            ptm.at(out, in) = std::abs(tr.real()) < 1e-14 ? 0.0 : tr.real();
        }
    }
    return ptm;
}

// O -> E^dagger(O) for the map `ptm` on the qubits `qubits` (qubits[j] carries
// digit j of the local index). A term with the local string p becomes
// sum_out at(out, p) times the term with p replaced by out; the phase flags
// and the rest of the string are kept. The term itself stays in place, scaled
// by at(p, p) (removed if that is 0), and the other images are new terms,
// merged with add_combine as in evolve(): the result is combined if the sum
// was, and the same for every thread count. Terms whose column is the
// identity (at(p, p) = 1 and nothing else, e.g. strings that commute with a
// gate) cost one index extraction.
template <std::size_t N, typename T>
void apply_ptm(PauliSum<N, T>& sum, const std::vector<std::size_t>& qubits, const PauliTransferMatrix& ptm) {
    using R = real_t<T>;
    const std::size_t k = qubits.size();
    if (k != ptm.k || k == 0 || k > kMaxLocalMapQubits || ptm.r.size() != ptm.dim() * ptm.dim())
        throw std::invalid_argument("apply_ptm: the map does not act on qubits.size() qubits");
    std::array<std::size_t, kMaxLocalMapQubits> qw{}, qb{};
    for (std::size_t j = 0; j < k; ++j) {
        if (qubits[j] >= sum.num_qubits()) throw std::invalid_argument("apply_ptm: qubit out of range");
        for (std::size_t i = 0; i < j; ++i)
            if (qubits[i] == qubits[j]) throw std::invalid_argument("apply_ptm: repeated qubit");
        qw[j] = qubits[j] / 64;
        qb[j] = qubits[j] % 64;
    }
    // Per input column: the diagonal entry and the other nonzero entries.
    const std::size_t D = ptm.dim();
    std::vector<R> self(D);
    std::vector<std::size_t> start(D + 1, 0);
    std::vector<std::pair<std::size_t, R>> others;
    bool drops = false;
    for (std::size_t in = 0; in < D; ++in) {
        start[in] = others.size();
        self[in] = static_cast<R>(ptm.at(in, in));
        drops |= self[in] == R{0};
        for (std::size_t out = 0; out < D; ++out)
            if (out != in && ptm.at(out, in) != 0.0) others.emplace_back(out, static_cast<R>(ptm.at(out, in)));
    }
    start[D] = others.size();

    auto local_index = [&qw, &qb, k](const auto& t) {
        std::size_t p = 0;
        for (std::size_t j = 0; j < k; ++j)
            p |= (((t.x(qw[j]) >> qb[j]) & 1) | (((t.z(qw[j]) >> qb[j]) & 1) << 1)) << (2 * j);
        return p;
    };
    auto set_local = [&qw, &qb, k](const TermRef<N, T>& o, std::size_t q) {
        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t m = std::size_t{1} << qb[j];
            const std::size_t code = (q >> (2 * j)) & 3;
            o.x_word(qw[j]) = (o.x(qw[j]) & ~m) | ((code & 1) << qb[j]);
            o.z_word(qw[j]) = (o.z(qw[j]) & ~m) | ((code >> 1) << qb[j]);
        }
    };

    // keep[i] = 0 for the terms whose diagonal entry is 0, set in the same
    // pass (a second pass with filter_terms re-extracted the local index in a
    // loop that requests vectorisation, which Clang 22 could not do).
    std::vector<uint8_t> keep(drops ? sum.get_length() : 0, 1);
    uint8_t* kp = keep.data();
    const ExecPolicy pol = detail::rotation_policy();
    const PauliSum<N, T> added = expand_terms(
        sum, [&](const TermRef<N, T>& t, TermEmitter<N, T>& out) {
            const std::size_t p = local_index(t);
            const T v = t.value();
            for (std::size_t e = start[p]; e < start[p + 1]; ++e)
                set_local(out.push(t, v * others[e].second), others[e].first);
            if (self[p] != R{1}) t.value_ref() = v * self[p];
            if (drops && self[p] == R{0}) kp[t.index()] = 0;
        }, pol);
    if (drops) sum.compact(keep);
    if (added.get_length() > 0) sum.add_combine(added);
}

}  // namespace paulib
