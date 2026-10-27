#pragma once

// Generation (GEN): strings that no other operation computes from existing
// ones.
//
// Random:     random_pauli_string, TermSampler / sample_terms (draw term j
//             with probability |a_j| / sum_k |a_k|, as in qDRIFT),
//             random_clifford (uniform, emitted as a gate list).
// Structured: strings_up_to_weight, lattice Hamiltonians (chain_edges,
//             square_lattice_edges, two_site_sum, one_site_sum), and the
//             fermion-to-qubit mappings Jordan-Wigner (jw_majorana,
//             jw_annihilation, jw_creation) and Bravyi-Kitaev (bk_majorana,
//             bk_annihilation, bk_creation).
//
// random_pauli_sum (PauliSumUtils.h) is the older bulk generator.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "PauliString.h"
#include "PauliSum.h"

namespace paulib {

// ---------------------------------------------------------------------------
// Random strings and terms

// Uniform over all 4^n strings on n qubits (identity included), phase +1.
template <std::size_t N, typename URBG>
PauliString<N> random_pauli_string(std::size_t n, URBG& rng) {
    if (n > N) throw std::invalid_argument("random_pauli_string: qubit count exceeds capacity");
    constexpr std::size_t BPW = PauliString<N>::BitsPerWord;
    std::uniform_int_distribution<std::size_t> dist;
    PauliString<N> ps{};
    ps.num_qubits_ = static_cast<uint32_t>(n);
    for (std::size_t w = 0; w * BPW < n; ++w) {
        const std::size_t len = std::min(BPW, n - w * BPW);
        const std::size_t mask = len == BPW ? ~std::size_t{0} : (std::size_t(1) << len) - 1;
        ps.x[w] = dist(rng) & mask;
        ps.z[w] = dist(rng) & mask;
    }
    return ps;
}

// Draws term indices j with probability |a_j| / lambda, lambda = sum |a_j|,
// by binary search over the prefix sums (O(log M) per draw). This is the
// sampling step of qDRIFT. The sampler keeps no reference to the sum.
template <typename R>
class TermSampler {
public:
    template <std::size_t N, typename T>
    explicit TermSampler(const PauliSum<N, T>& sum) : prefix_(sum.get_length()) {
        R acc = R{};
        for (std::size_t j = 0; j < sum.get_length(); ++j) {
            acc += std::abs(sum.value[j]);
            prefix_[j] = acc;
        }
    }

    R lambda() const { return prefix_.empty() ? R{} : prefix_.back(); }
    std::size_t size() const { return prefix_.size(); }

    template <typename URBG>
    std::size_t sample(URBG& rng) const {
        if (lambda() <= R{}) throw std::logic_error("TermSampler: all coefficients are zero");
        std::uniform_real_distribution<R> u(R{}, lambda());
        const R r = u(rng);
        const auto it = std::upper_bound(prefix_.begin(), prefix_.end(), r);
        return std::min<std::size_t>(static_cast<std::size_t>(it - prefix_.begin()),
                                     prefix_.size() - 1);
    }

private:
    std::vector<R> prefix_;
};

template <std::size_t N, typename T>
TermSampler(const PauliSum<N, T>&) -> TermSampler<real_t<T>>;

// k draws from a TermSampler seeded with `seed`.
template <std::size_t N, typename T>
std::vector<std::size_t> sample_terms(const PauliSum<N, T>& sum, std::size_t k,
                                      std::uint64_t seed) {
    TermSampler sampler(sum);
    std::mt19937_64 rng(seed);
    std::vector<std::size_t> out(k);
    for (auto& j : out) j = sampler.sample(rng);
    return out;
}

// ---------------------------------------------------------------------------
// Structured sets

// All strings of weight <= max_weight on n qubits, with coefficient 1, ordered
// by weight, then by support (lexicographic), then by the Paulis on the
// support. There are sum_{w <= k} C(n, w) 3^w of them; throws
// std::length_error above max_terms.
template <std::size_t N, typename T>
PauliSum<N, T> strings_up_to_weight(std::size_t n, std::size_t max_weight,
                                    std::size_t max_terms = std::size_t{1} << 24) {
    if (n > N) throw std::invalid_argument("strings_up_to_weight: qubit count exceeds capacity");
    max_weight = std::min(max_weight, n);

    // Count with saturation.
    std::size_t total = 0, binom = 1, pow3 = 1;
    for (std::size_t w = 0; w <= max_weight; ++w) {
        if (w > 0) {
            binom = binom * (n - w + 1) / w;  // exact: C(n, w-1) (n-w+1) is divisible by w
            pow3 *= 3;
        }
        const long double c = static_cast<long double>(binom) * pow3;
        if (c > static_cast<long double>(max_terms) || total + static_cast<std::size_t>(c) > max_terms)
            throw std::length_error("strings_up_to_weight: more than max_terms strings");
        total += static_cast<std::size_t>(c);
    }

    PauliSum<N, T> out;
    out.num_qubits_ = n;
    out.reserve(total);
    static constexpr char kOps[3] = {'X', 'Y', 'Z'};
    for (std::size_t w = 0; w <= max_weight; ++w) {
        std::vector<std::size_t> support(w);
        for (std::size_t i = 0; i < w; ++i) support[i] = i;
        while (true) {
            std::vector<std::size_t> ops(w, 0);
            while (true) {
                PauliString<N> ps{};
                ps.num_qubits_ = static_cast<uint32_t>(n);
                for (std::size_t i = 0; i < w; ++i) ps.set_pauli(support[i], kOps[ops[i]]);
                out.push_back(ps, T{1});
                std::size_t i = w;
                while (i > 0 && ops[i - 1] == 2) ops[--i] = 0;
                if (i == 0) break;
                ++ops[i - 1];
            }
            // Next w-subset of {0..n-1} in lexicographic order.
            std::size_t i = w;
            while (i > 0 && support[i - 1] == n - w + i - 1) --i;
            if (i == 0) break;
            ++support[i - 1];
            for (std::size_t j = i; j < w; ++j) support[j] = support[j - 1] + 1;
        }
    }
    return out;
}

using Edge = std::pair<std::size_t, std::size_t>;

// Nearest-neighbour bonds (i, i+1) of a chain of n sites; periodic adds
// (n-1, 0) for n > 2.
inline std::vector<Edge> chain_edges(std::size_t n, bool periodic = false) {
    std::vector<Edge> e;
    for (std::size_t i = 0; i + 1 < n; ++i) e.emplace_back(i, i + 1);
    if (periodic && n > 2) e.emplace_back(n - 1, 0);
    return e;
}

// Nearest-neighbour bonds of an Lx x Ly square lattice, site (x, y) is qubit
// x + Lx * y. Horizontal bonds first, then vertical ones. periodic wraps a
// direction whose length is > 2.
inline std::vector<Edge> square_lattice_edges(std::size_t Lx, std::size_t Ly,
                                              bool periodic = false) {
    std::vector<Edge> e;
    auto site = [Lx](std::size_t x, std::size_t y) { return x + Lx * y; };
    for (std::size_t y = 0; y < Ly; ++y)
        for (std::size_t x = 0; x < Lx; ++x) {
            if (x + 1 < Lx) e.emplace_back(site(x, y), site(x + 1, y));
            else if (periodic && Lx > 2) e.emplace_back(site(x, y), site(0, y));
        }
    for (std::size_t y = 0; y < Ly; ++y)
        for (std::size_t x = 0; x < Lx; ++x) {
            if (y + 1 < Ly) e.emplace_back(site(x, y), site(x, y + 1));
            else if (periodic && Ly > 2) e.emplace_back(site(x, y), site(x, 0));
        }
    return e;
}

// sum over edges (a, b) of J * op[0]_a op[1]_b, e.g. op = "ZZ" for the Ising
// coupling. Terms are not combined.
template <std::size_t N, typename T>
PauliSum<N, T> two_site_sum(std::size_t n, const std::vector<Edge>& edges,
                            const std::string& op, T J) {
    if (n > N) throw std::invalid_argument("two_site_sum: qubit count exceeds capacity");
    if (op.size() != 2) throw std::invalid_argument("two_site_sum: op must have two characters");
    PauliSum<N, T> out;
    out.num_qubits_ = n;
    out.reserve(edges.size());
    for (const auto& [a, b] : edges) {
        if (a == b) throw std::invalid_argument("two_site_sum: edge with equal sites");
        PauliString<N> ps{};
        ps.num_qubits_ = static_cast<uint32_t>(n);
        ps.set_pauli(a, op[0]);
        ps.set_pauli(b, op[1]);
        out.push_back(ps, J);
    }
    return out;
}

// sum over q of h * op_q, e.g. op = "X" for a transverse field.
template <std::size_t N, typename T>
PauliSum<N, T> one_site_sum(std::size_t n, const std::string& op, T h) {
    if (n > N) throw std::invalid_argument("one_site_sum: qubit count exceeds capacity");
    if (op.size() != 1) throw std::invalid_argument("one_site_sum: op must have one character");
    PauliSum<N, T> out;
    out.num_qubits_ = n;
    out.reserve(n);
    for (std::size_t q = 0; q < n; ++q) {
        PauliString<N> ps{};
        ps.num_qubits_ = static_cast<uint32_t>(n);
        ps.set_pauli(q, op[0]);
        out.push_back(ps, h);
    }
    return out;
}

// Jordan-Wigner Majorana operator gamma_j on n modes, j < 2n:
// gamma_{2m} = Z_0 ... Z_{m-1} X_m, gamma_{2m+1} = Z_0 ... Z_{m-1} Y_m.
// They satisfy {gamma_i, gamma_j} = 2 delta_ij.
template <std::size_t N>
PauliString<N> jw_majorana(std::size_t n, std::size_t j) {
    if (n > N) throw std::invalid_argument("jw_majorana: qubit count exceeds capacity");
    if (j >= 2 * n) throw std::out_of_range("jw_majorana: index out of range");
    PauliString<N> ps{};
    ps.num_qubits_ = static_cast<uint32_t>(n);
    const std::size_t m = j / 2;
    for (std::size_t q = 0; q < m; ++q) ps.set_pauli(q, 'Z');
    ps.set_pauli(m, (j % 2) ? 'Y' : 'X');
    return ps;
}

namespace detail {

// (gamma_{2m} + s i gamma_{2m+1}) / 2 for s = +1 (annihilation) or -1
// (creation): two terms, the i carried by the phase flag, so a real T
// suffices.
template <std::size_t N, typename T>
PauliSum<N, T> ladder_operator(std::size_t n, PauliString<N> gx, PauliString<N> gy, bool creation) {
    PauliSum<N, T> out;
    out.num_qubits_ = n;
    gy.flags = creation ? 3 : 1;  // -i or +i
    out.push_back(gx, T{0.5});
    out.push_back(gy, T{0.5});
    return out;
}

}  // namespace detail

// a_m = (gamma_{2m} + i gamma_{2m+1}) / 2 = Z...Z (X + iY)/2.
template <std::size_t N, typename T>
PauliSum<N, T> jw_annihilation(std::size_t n, std::size_t m) {
    return detail::ladder_operator<N, T>(n, jw_majorana<N>(n, 2 * m), jw_majorana<N>(n, 2 * m + 1), false);
}

// a_m^dagger = Z...Z (X - iY)/2.
template <std::size_t N, typename T>
PauliSum<N, T> jw_creation(std::size_t n, std::size_t m) {
    return detail::ladder_operator<N, T>(n, jw_majorana<N>(n, 2 * m), jw_majorana<N>(n, 2 * m + 1), true);
}

// Bravyi-Kitaev mapping (Bravyi and Kitaev 2002; Seeley, Richard and Love
// 2012) for any number of modes n, with the Fenwick-tree sets of
// OpenFermion's bravyi_kitaev. Qubit j stores the parity of a range of modes
// ending at j, so each Majorana string has weight O(log n) instead of O(n).
// For mode j (0-based):
// - update set U(j): j and the qubits whose range contains j;
// - parity set P(j): qubits whose ranges together hold modes 0..j-1;
// - occupation set O(j): j and the qubits whose ranges, with j's, hold j.
namespace detail {

inline std::vector<std::size_t> bk_update_set(std::size_t j, std::size_t n) {
    std::vector<std::size_t> out;
    for (std::size_t i = j + 1; i <= n; i += i & (~i + 1)) out.push_back(i - 1);
    return out;
}

inline std::vector<std::size_t> bk_parity_set(std::size_t j) {
    std::vector<std::size_t> out;
    for (std::size_t i = j; i > 0; i &= i - 1) out.push_back(i - 1);
    return out;
}

inline std::vector<std::size_t> bk_occupation_set(std::size_t j) {
    std::vector<std::size_t> out{j};
    const std::size_t parent = (j + 1) & j;
    for (std::size_t i = j; i != parent; i &= i - 1) out.push_back(i - 1);
    return out;
}

}  // namespace detail

// Bravyi-Kitaev Majorana operator gamma_j on n modes, j < 2n:
// gamma_{2m} = X_{U(m)} Z_{P(m)},
// gamma_{2m+1} = Y_m X_{U(m) \ m} Z_{(P(m) xor O(m)) \ m}.
// They satisfy {gamma_i, gamma_j} = 2 delta_ij, and a_m = (gamma_{2m} + i
// gamma_{2m+1}) / 2 as for Jordan-Wigner.
template <std::size_t N>
PauliString<N> bk_majorana(std::size_t n, std::size_t j) {
    if (n > N) throw std::invalid_argument("bk_majorana: qubit count exceeds capacity");
    if (j >= 2 * n) throw std::out_of_range("bk_majorana: index out of range");
    const std::size_t m = j / 2;
    std::vector<uint8_t> x(n, 0), z(n, 0);
    for (std::size_t q : detail::bk_update_set(m, n)) x[q] = 1;
    for (std::size_t q : detail::bk_parity_set(m)) z[q] ^= 1;
    if (j % 2) {
        for (std::size_t q : detail::bk_occupation_set(m)) z[q] ^= 1;
        z[m] = 1;  // with x[m] = 1: Y on qubit m
    }
    PauliString<N> ps{};
    ps.num_qubits_ = static_cast<uint32_t>(n);
    for (std::size_t q = 0; q < n; ++q)
        if (x[q] || z[q]) ps.set_pauli(q, x[q] ? (z[q] ? 'Y' : 'X') : 'Z');
    return ps;
}

template <std::size_t N, typename T>
PauliSum<N, T> bk_annihilation(std::size_t n, std::size_t m) {
    return detail::ladder_operator<N, T>(n, bk_majorana<N>(n, 2 * m), bk_majorana<N>(n, 2 * m + 1), false);
}

template <std::size_t N, typename T>
PauliSum<N, T> bk_creation(std::size_t n, std::size_t m) {
    return detail::ladder_operator<N, T>(n, bk_majorana<N>(n, 2 * m), bk_majorana<N>(n, 2 * m + 1), true);
}

// ---------------------------------------------------------------------------
// Clifford circuits as gate lists

struct CliffordGate {
    enum Kind : uint8_t { H, S, SDG, X, Y, Z, CNOT, CZ, SWAP };
    Kind kind;
    std::size_t q0;
    std::size_t q1 = 0;  // target of CNOT, second qubit of CZ / SWAP

    bool operator==(const CliffordGate&) const = default;
};

// Gates in application order: the circuit is U = g_L ... g_2 g_1.
using CliffordCircuit = std::vector<CliffordGate>;

namespace detail {

// One gate by the apply_* method of the same name.
template <typename P>
void apply_gate(P& p, const CliffordGate& g) {
    switch (g.kind) {
        case CliffordGate::H:    p.apply_H(g.q0); break;
        case CliffordGate::S:    p.apply_S(g.q0); break;
        case CliffordGate::SDG:  p.apply_Z(g.q0); p.apply_S(g.q0); break;  // S^dagger = S Z, commuting
        case CliffordGate::X:    p.apply_X(g.q0); break;
        case CliffordGate::Y:    p.apply_Y(g.q0); break;
        case CliffordGate::Z:    p.apply_Z(g.q0); break;
        case CliffordGate::CNOT: p.apply_CNOT(g.q0, g.q1); break;
        case CliffordGate::CZ:   p.apply_CZ(g.q0, g.q1); break;
        case CliffordGate::SWAP: p.apply_SWAP(g.q0, g.q1); break;
    }
}

}  // namespace detail

// Heisenberg picture: P -> U^dagger P U = g_1^dagger ... g_L^dagger P g_L ... g_1,
// so the gates run in reverse order, each as P -> g^dagger P g (the
// convention of the apply_* gate methods). For U P U^dagger, use
// apply_circuit(P, adjoint(circuit)). Works for PauliString<N>,
// PauliSum<N, T> and PauliTable; Tableau has its own overload, which
// evolves the state forward.
template <typename P>
void apply_circuit(P& p, const CliffordCircuit& circuit) {
    for (auto it = circuit.rbegin(); it != circuit.rend(); ++it) detail::apply_gate(p, *it);
}

// U^dagger as a gate list: reversed order, S <-> SDG.
inline CliffordCircuit adjoint(const CliffordCircuit& circuit) {
    CliffordCircuit out(circuit.rbegin(), circuit.rend());
    for (auto& g : out) {
        if (g.kind == CliffordGate::S) g.kind = CliffordGate::SDG;
        else if (g.kind == CliffordGate::SDG) g.kind = CliffordGate::S;
    }
    return out;
}

namespace detail {

template <std::size_t N>
inline char pauli_at(const PauliString<N>& p, std::size_t q) {
    const std::size_t w = q / PauliString<N>::BitsPerWord;
    const std::size_t b = q % PauliString<N>::BitsPerWord;
    const bool x = (p.x[w] >> b) & 1, z = (p.z[w] >> b) & 1;
    return x ? (z ? 'Y' : 'X') : (z ? 'Z' : 'I');
}

// Appends gates on qubits j..n-1 that map the anticommuting pair (p, q),
// which acts trivially on qubits < j, to (+-X_j, +-Z_j); p and q are
// conjugated along as p -> g p g^dagger.
template <std::size_t N>
void sweep_pair(PauliString<N>& p, PauliString<N>& q, std::size_t j, std::size_t n,
                CliffordCircuit& out) {
    auto emit = [&](CliffordGate g) {
        const CliffordCircuit one = adjoint(CliffordCircuit{g});
        apply_circuit(p, one);
        apply_circuit(q, one);
        out.push_back(g);
    };
    // 1. p -> X-type: Z -> X by H, Y -> -X by S.
    for (std::size_t k = j; k < n; ++k) {
        const char c = pauli_at(p, k);
        if (c == 'Z') emit({CliffordGate::H, k});
        else if (c == 'Y') emit({CliffordGate::S, k});
    }
    // 2. p -> X_pivot by CNOTs from the first qubit of its support.
    std::size_t pivot = n;
    for (std::size_t k = j; k < n; ++k)
        if (pauli_at(p, k) == 'X') { pivot = k; break; }
    for (std::size_t k = pivot + 1; k < n; ++k)
        if (pauli_at(p, k) == 'X') emit({CliffordGate::CNOT, pivot, k});
    // 3. Move the pivot to qubit j.
    if (pivot != j) emit({CliffordGate::SWAP, pivot, j});
    // 4. q anticommutes with X_j, so it has Z or Y on j. Make every other
    //    qubit of q Z-type with gates that act trivially on X_j (X -> Z by H,
    //    Y -> +-Z by H S H), and Y_j -> +-Z_j by H S H, which fixes X_j.
    for (std::size_t k = j; k < n; ++k) {
        const char c = pauli_at(q, k);
        if (k != j && c == 'X') emit({CliffordGate::H, k});
        else if (c == 'Y') {
            emit({CliffordGate::H, k});
            emit({CliffordGate::S, k});
            emit({CliffordGate::H, k});
        }
    }
    // 5. Z_k Z_j -> Z_j by CNOT(k, j), which fixes X_j.
    for (std::size_t k = j + 1; k < n; ++k)
        if (pauli_at(q, k) == 'Z') emit({CliffordGate::CNOT, k, j});
}

}  // namespace detail

// Uniformly random n-qubit Clifford (up to a global phase) as a gate list over
// {H, S, SDG, X, Y, Z, CNOT, SWAP}, O(n^2) gates, following the sweep of
// van den Berg, "A simple method for sampling random Clifford operators"
// (arXiv:2008.06011). For j = 0..n-1 a uniformly random anticommuting pair
// (P_j, Q_j) on qubits j..n-1 is drawn and a circuit C_j with
// C_j P_j C_j^dagger = +-X_j, C_j Q_j C_j^dagger = +-Z_j is built; the
// Clifford is R C_0^dagger ... C_{n-1}^dagger preceded by a uniformly random
// Pauli R, which makes the signs of the images uniform.
template <typename URBG>
CliffordCircuit random_clifford(std::size_t n, URBG& rng) {
    constexpr std::size_t N = 1024;
    if (n > N) throw std::invalid_argument("random_clifford: more than 1024 qubits");
    std::vector<CliffordCircuit> stages(n);
    for (std::size_t j = 0; j < n; ++j) {
        // Uniform pair on qubits j..n-1: P != I, Q anticommuting with P.
        auto draw = [&]() {
            PauliString<N> r = random_pauli_string<N>(n - j, rng);
            PauliString<N> s{};
            s.num_qubits_ = static_cast<uint32_t>(n);
            for (std::size_t k = 0; k < n - j; ++k) {
                const char c = detail::pauli_at(r, k);
                if (c != 'I') s.set_pauli(j + k, c);
            }
            return s;
        };
        PauliString<N> p, q;
        do { p = draw(); } while (p.x == PauliString<N>{}.x && p.z == PauliString<N>{}.z);
        do { q = draw(); } while (p.commute(q));
        detail::sweep_pair(p, q, j, n, stages[j]);
    }

    CliffordCircuit out;
    std::uniform_int_distribution<int> bit(0, 1);
    for (std::size_t k = 0; k < n; ++k) {
        const bool x = bit(rng), z = bit(rng);
        if (x && z) out.push_back({CliffordGate::Y, k});
        else if (x) out.push_back({CliffordGate::X, k});
        else if (z) out.push_back({CliffordGate::Z, k});
    }
    for (std::size_t j = n; j-- > 0;) {
        const CliffordCircuit inv = adjoint(stages[j]);
        out.insert(out.end(), inv.begin(), inv.end());
    }
    return out;
}

}  // namespace paulib
