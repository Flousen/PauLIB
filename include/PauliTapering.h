#pragma once

// Qubit tapering with Z2 symmetries (Bravyi, Gambetta, Mezzacapo and Temme,
// arXiv:1701.08213): an application module (L8) built from the kernels for
// GF2 (commutant), CLF (Clifford circuits) and QMP (fix_qubits).
//
// k independent, mutually commuting Hermitian strings tau_i that commute with
// every term of H are symmetries of H. In the sector tau_i = s_i (s_i = +-1),
// H acts on a subspace of dimension 2^(n-k). A Clifford U maps every s_i tau_i
// to +-Z_{q_i} on its own qubit q_i; then every term of U H U^dagger has I or Z
// on q_i, so Z_{q_i} is replaced by its eigenvalue in the sector and the
// qubit q_i is removed. The tapered operator has the spectrum of H in the
// sector.
//
// - z2_symmetries(h): independent, mutually commuting strings that commute
//   with every term of h: a basis of the commutant (GF2), from which a string
//   is kept if it commutes with those kept before.
// - Z2Tapering<N>(symmetries, sector, n): builds U as a gate list (a sweep
//   with H, S and CNOT, as in random_clifford) and the removed qubits;
//   apply(op) tapers h or any other operator that commutes with the
//   symmetries.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "PauliGF2.h"
#include "PauliGenerate.h"
#include "PauliQubitMap.h"
#include "PauliSkeletons.h"
#include "PauliSum.h"

namespace paulib {

template <std::size_t N, typename T>
std::vector<PauliString<N>> z2_symmetries(const PauliSum<N, T>& h) {
    const PauliSum<N, T> c = commutant(h);
    std::vector<PauliString<N>> out;
    for (std::size_t j = 0; j < c.get_length(); ++j) {
        const PauliString<N> p = c.get_pauli_string(j);
        bool commutes = true;
        for (const auto& q : out)
            if (!p.commute(q)) { commutes = false; break; }
        if (commutes) out.push_back(p);
    }
    return out;
}

template <std::size_t N>
class Z2Tapering {
public:
    // symmetries: Hermitian, independent and mutually commuting strings on n
    // qubits; sector[i] = +1 or -1 is the eigenvalue of symmetries[i].
    // Throws std::invalid_argument otherwise.
    Z2Tapering(const std::vector<PauliString<N>>& symmetries, const std::vector<int>& sector,
               std::size_t n)
        : n_(n), symmetries_(symmetries) {
        const std::size_t k = symmetries.size();
        if (n > N) throw std::invalid_argument("Z2Tapering: qubit count exceeds capacity");
        if (sector.size() != k) throw std::invalid_argument("Z2Tapering: one sector value per symmetry");
        // g_i = s_i tau_i stabilises the sector; it is conjugated along with
        // every gate (Schroedinger picture) until it is +-Z_{q_i}.
        std::vector<PauliString<N>> g(symmetries);
        for (std::size_t i = 0; i < k; ++i) {
            if (g[i].flags & 1) throw std::invalid_argument("Z2Tapering: symmetry is not Hermitian");
            if (sector[i] != 1 && sector[i] != -1)
                throw std::invalid_argument("Z2Tapering: sector values must be +1 or -1");
            if (sector[i] == -1) g[i].flip_sign();
            for (std::size_t j = 0; j < i; ++j)
                if (!g[i].commute(g[j])) throw std::invalid_argument("Z2Tapering: symmetries do not commute");
        }
        for (std::size_t i = 0; i < k; ++i) {
            PauliString<N>& t = g[i];
            // t commutes with g_j = +-Z_{q_j}, so it has I or Z on q_j;
            // multiplying by g_j clears the Z and keeps the group.
            for (std::size_t j = 0; j < i; ++j)
                if (detail::pauli_at(t, qubits_[j]) == 'Z') t *= g[j];
            std::vector<std::size_t> support;
            for (std::size_t q = 0; q < n; ++q)
                if (detail::pauli_at(t, q) != 'I') support.push_back(q);
            if (support.empty()) throw std::invalid_argument("Z2Tapering: symmetries are dependent");
            auto emit = [&](CliffordGate gate) {
                gates_.push_back(gate);
                const CliffordCircuit inv = adjoint(CliffordCircuit{gate});
                for (std::size_t l = i; l < k; ++l) apply_circuit(g[l], inv);  // g -> G g G^dagger
            };
            // Make t Z-type: X -> Z by H, Y -> -X by S, then H.
            for (std::size_t q : support) {
                const char c = detail::pauli_at(t, q);
                if (c == 'Y') emit({CliffordGate::S, q});
                if (c != 'Z') emit({CliffordGate::H, q});
            }
            // Z_r Z_p -> Z_p by CNOT(r, p).
            const std::size_t p = support.front();
            for (std::size_t r : support)
                if (r != p) emit({CliffordGate::CNOT, r, p});
            qubits_.push_back(p);
            eigenvalues_.push_back((t.flags & 2) ? -1 : 1);  // t = +-Z_p stabilises the sector
            fixes_.push_back({p, 'Z', eigenvalues_.back()});
        }
    }

    // U as a gate list: U s_i tau_i U^dagger = +-Z_{q_i}.
    const CliffordCircuit& clifford() const { return gates_; }
    // Removed qubits q_i and the eigenvalue of Z_{q_i} in the sector.
    const std::vector<std::size_t>& qubits() const { return qubits_; }
    const std::vector<int>& eigenvalues() const { return eigenvalues_; }
    std::size_t num_qubits_out() const { return n_ - qubits_.size(); }

    // The tapered operator on num_qubits_out() qubits, canonicalized:
    // U op U^dagger with Z_{q_i} replaced by its eigenvalue and the qubits q_i
    // removed. op must commute with every symmetry (std::invalid_argument
    // otherwise).
    template <typename T>
    PauliSum<N, T> apply(const PauliSum<N, T>& op) const {
        if (op.num_qubits() != n_) throw std::invalid_argument("Z2Tapering::apply: qubit-count mismatch");
        const std::size_t anti = reduce_terms<std::size_t>(op, [this](const TermView<N, T>& t) {
            for (const auto& s : symmetries_)
                if (anticommutes(t, s)) return std::size_t{1};
            return std::size_t{0};
        });
        if (anti) throw std::invalid_argument("Z2Tapering::apply: operator does not commute with the symmetries");

        PauliSum<N, T> h = op;
        apply_circuit(h, adjoint(gates_));  // U h U^dagger
        PauliSum<N, T> out = fix_qubits(h, fixes_);  // Z_{q_i} -> eigenvalue, qubit q_i removed
        out.canonicalize();
        return out;
    }

private:
    std::size_t n_;
    std::vector<PauliString<N>> symmetries_;
    CliffordCircuit gates_;
    std::vector<std::size_t> qubits_;
    std::vector<int> eigenvalues_;
    std::vector<FixedQubit> fixes_;
};

}  // namespace paulib
