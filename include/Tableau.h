#pragma once

// Stabilizer simulation with the tableau of Aaronson and Gottesman (CHP,
// arXiv:quant-ph/0406196) on the qubit-major PauliTable.
//
// The state of n qubits is stored as 2n strings: terms 0..n-1 are the
// destabilizers D_i, terms n..2n-1 the stabilizers S_i, with D_i
// anticommuting only with S_i. The start state |0...0> has D_i = X_i and
// S_i = Z_i. The tableau evolves a state, so a Clifford gate G maps every
// string P -> G P G^dagger (Schroedinger picture), the inverse of the
// Heisenberg convention of the PauliTable gates: S, sqrt(X) and sqrt(Y) use
// the table's inverse gates, the self-inverse gates are the same. A gate
// costs O(n/64) word operations. apply_circuit(tab, circuit) runs the gates
// in circuit order (overload below).
//
// - measure_z: random outcome if some stabilizer anticommutes with Z_q; all
//   other rows that anticommute are multiplied by it at once
//   (PauliTable::left_multiply). Otherwise deterministic, via peek.
// - peek(P): <P> in {+1, -1, 0}. If P commutes with every stabilizer,
//   P = +-prod S_i over the i with D_i anticommuting with P; the sign comes
//   from PauliTable::product. O(n^2/64) word operations.

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "PauliGenerate.h"
#include "PauliSum.h"
#include "PauliTable.h"

namespace paulib {

struct MeasureResult {
    bool outcome;        // false = +1 (|0>), true = -1 (|1>)
    bool deterministic;
};

template <std::size_t N>
class Tableau {
public:
    explicit Tableau(std::size_t n) : n_(n), t_(n, 2 * n) {
        if (n > N) throw std::invalid_argument("Tableau: qubit count exceeds capacity");
        for (std::size_t i = 0; i < n; ++i) {
            t_.set_x(i, i, true);
            t_.set_z(i, n + i, true);
        }
    }

    std::size_t num_qubits() const { return n_; }
    const PauliTable& table() const { return t_; }

    void apply_H(std::size_t q)      { t_.apply_H(q); }
    void apply_S(std::size_t q)      { t_.apply_S_dag(q); }
    void apply_H_YZ(std::size_t q)   { t_.apply_H_YZ(q); }
    void apply_H_XY(std::size_t q)   { t_.apply_H_XY(q); }
    void apply_sqrt_X(std::size_t q) { t_.apply_sqrt_X_dag(q); }
    void apply_sqrt_Y(std::size_t q) { t_.apply_sqrt_Y_dag(q); }
    void apply_X(std::size_t q)      { t_.apply_X(q); }
    void apply_Y(std::size_t q)      { t_.apply_Y(q); }
    void apply_Z(std::size_t q)      { t_.apply_Z(q); }
    void apply_CNOT(std::size_t c, std::size_t t) { t_.apply_CNOT(c, t); }
    void apply_CZ(std::size_t c, std::size_t t)   { t_.apply_CZ(c, t); }
    void apply_SWAP(std::size_t c, std::size_t t) { t_.apply_SWAP(c, t); }

    // <psi| P |psi> for a Hermitian string P: +1, -1 or 0.
    int peek(const PauliString<N>& p) const {
        if (p.flags & 1) throw std::invalid_argument("Tableau::peek: P is not Hermitian");
        const std::vector<std::size_t> anti = anticommutes_mask(t_, p);
        // Any stabilizer anticommuting with P: <P> = 0.
        std::vector<std::size_t> stab(words_n(), 0);
        detail::get_bits(stab.data(), anti.data(), n_, n_);
        for (std::size_t w : stab) if (w) return 0;
        // P = +-prod S_i over the i with D_i anticommuting with P.
        std::vector<std::size_t> sel(t_.words(), 0);
        detail::or_bits(sel.data(), n_, anti.data(), n_);
        std::vector<std::size_t> px, pz;
        const uint8_t k = t_.product(sel, px, pz);
        for (std::size_t w = 0; w < px.size(); ++w)
            if (px[w] != p.x[w] || pz[w] != p.z[w])
                throw std::logic_error("Tableau::peek: product does not match P");
        // P = i^(s_P - k) prod, and prod stabilizes the state.
        return ((p.flags - k) & 3) == 0 ? 1 : -1;
    }

    template <typename URBG>
    MeasureResult measure_z(std::size_t q, URBG& rng) {
        // Z_q anticommutes with exactly the rows that have X on qubit q.
        const std::size_t* xr = t_.x_row(q);
        std::size_t p = 2 * n_;
        for (std::size_t j = n_; j < 2 * n_; ) {
            const std::size_t w = j / 64;
            const std::size_t bits = xr[w] >> (j % 64);
            if (bits) { p = j + static_cast<std::size_t>(std::countr_zero(bits)); break; }
            j = (w + 1) * 64;
        }
        if (p >= 2 * n_) {
            PauliString<N> zq{};
            zq.num_qubits_ = static_cast<uint32_t>(n_);
            zq.z[q / 64] |= std::size_t(1) << (q % 64);
            return {peek(zq) == -1, true};
        }
        // Random outcome: every other anticommuting row times row p.
        std::vector<std::size_t> mask(xr, xr + t_.words());
        mask[p / 64] &= ~(std::size_t(1) << (p % 64));
        t_.left_multiply(mask, p);
        // D_{p-n} <- old S_p, S_p <- (-1)^outcome Z_q.
        t_.copy_term(p - n_, p);
        for (std::size_t r = 0; r < n_; ++r) {
            t_.set_x(r, p, false);
            t_.set_z(r, p, false);
        }
        t_.set_z(q, p, true);
        const bool outcome = std::uniform_int_distribution<int>(0, 1)(rng) == 1;
        t_.set_phase(p, outcome ? 2 : 0);
        return {outcome, false};
    }

    // Measures qubit q and flips it to |0> if the outcome was 1.
    template <typename URBG>
    void reset(std::size_t q, URBG& rng) {
        if (measure_z(q, rng).outcome) apply_X(q);
    }

    std::vector<PauliString<N>> stabilizers() const {
        std::vector<PauliString<N>> out;
        out.reserve(n_);
        for (std::size_t i = 0; i < n_; ++i) out.push_back(t_.template get<N>(n_ + i));
        return out;
    }

    std::vector<PauliString<N>> destabilizers() const {
        std::vector<PauliString<N>> out;
        out.reserve(n_);
        for (std::size_t i = 0; i < n_; ++i) out.push_back(t_.template get<N>(i));
        return out;
    }

private:
    std::size_t n_;
    PauliTable t_;

    std::size_t words_n() const { return (n_ + 63) / 64; }
};

// The state U|psi> for U = g_L ... g_1: the gates in circuit order.
template <std::size_t N>
void apply_circuit(Tableau<N>& tab, const CliffordCircuit& circuit) {
    for (const auto& g : circuit) detail::apply_gate(tab, g);
}

}  // namespace paulib
