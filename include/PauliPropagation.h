#pragma once

// Pauli propagation: Heisenberg-picture backpropagation of an observable
// through a parametrised rotation circuit with truncation.
//
// Each rotation gate U_l = exp(-i theta_l/2 P_l) conjugates every term Q of
// the observable that anticommutes with P_l:
//
//     U† Q U = cos(theta) Q + i sin(theta) P Q
//
// (this split is PauliSum::evolve). Term count can double per gate, so the
// observable is re-canonicalized and truncated after every gate. The final
// expectation is <0|U† O U|0> via zero_state_expectation().
//
// Truncation by weight (max_weight) gives the polynomial-cost guarantees of
// the Pauli-propagation literature; coefficient_cutoff and max_terms bound
// memory in practice.

#include <cstddef>
#include <limits>

#include "PauliSum.h"

namespace paulib {

template <typename T>
struct TruncationOptions {
    std::size_t max_weight = std::numeric_limits<std::size_t>::max();
    T coefficient_cutoff = T{0};
    std::size_t max_terms = std::numeric_limits<std::size_t>::max();
};

// Canonicalize, then apply the enabled truncations. `keep` (optional)
// protects its (x, z) patterns from trim().
template <std::size_t N, typename T>
void apply_truncation(PauliSum<N, T>& O, const TruncationOptions<T>& opts,
                      const PauliSum<N, T>* keep = nullptr) {
    O.canonicalize();
    if (opts.max_weight != std::numeric_limits<std::size_t>::max())
        O.truncate_weight(opts.max_weight);
    if (opts.coefficient_cutoff > T{0})
        O.cutoff(opts.coefficient_cutoff);
    if (opts.max_terms != std::numeric_limits<std::size_t>::max()) {
        if (keep) O.trim(opts.max_terms, *keep);
        else O.trim(opts.max_terms);
    }
}

// Propagate O through the circuit in place. `rotations` lists the gates in
// application order (term l = generator P_l, coefficient = angle theta_l,
// gate exp(-i theta_l/2 P_l); rotations.value[0] acts on the state first).
// Heisenberg conjugation therefore processes the gates in reverse.
template <std::size_t N, typename T>
void propagate(PauliSum<N, T>& O, const PauliSum<N, T>& rotations,
               const TruncationOptions<T>& opts = {}) {
    for (std::size_t l = rotations.get_length(); l-- > 0;) {
        PauliSum<N, T> gate;
        gate.num_qubits_ = rotations.num_qubits();
        gate.push_back(rotations.get_pauli_string(l), rotations.value[l]);
        O.evolve(gate);
        apply_truncation(O, opts);
    }
}

// <0…0| U† O U |0…0> for the circuit U described by `rotations`.
template <std::size_t N, typename T>
T propagate_expectation(PauliSum<N, T> O, const PauliSum<N, T>& rotations,
                        const TruncationOptions<T>& opts = {}) {
    propagate(O, rotations, opts);
    return O.zero_state_expectation();
}

}  // namespace paulib
