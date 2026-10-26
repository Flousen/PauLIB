#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "PauliString.h"


namespace paulib {

template <std::size_t N, typename T>
struct PauliSum {

    static constexpr size_t BitsPerWord = std::numeric_limits<size_t>::digits;
    static constexpr size_t ArraySize = (N + BitsPerWord - 1) / BitsPerWord;

    using WordVec = std::vector<size_t>;
    using WordArray = std::array<WordVec, ArraySize>;

    WordArray x_data;
    WordArray z_data;
    std::vector<uint8_t> flags;  

    static constexpr uint8_t PHASE_MASK = 0x01;
    static constexpr uint8_t SIGN_MASK  = 0x02;

    inline void flip_sign(size_t i) {
        flags[i] ^= SIGN_MASK;
    }

    std::vector<T> value{};

    // Logical qubit count; N is the compile-time capacity. Defaults to N for
    // C++ use; the Python bindings instantiate word-sized capacities (64, 128,
    // ...) and track the actual qubit count here.
    std::size_t num_qubits_ = N;

    size_t num_qubits() const { return num_qubits_; }
    static constexpr size_t capacity() { return N; }
    size_t get_length() const { return value.size(); }

    PauliSum() = default;
    PauliSum(const std::vector<std::string>& pauli_strings);
    PauliSum(const std::vector<std::pair<std::string, T>>& terms) ;
    PauliSum(size_t qubits, std::initializer_list<std::string> pauli_strings) ;
    PauliSum(size_t qubits, const std::string& pauli_string, T coefficient = T{1}) ;
    std::string get_str() const ;

    void reserve(size_t capacity) ;
    void push_back(const PauliString<N>& ps, T coefficient = T{1}) ;
    PauliString<N> get_pauli_string(size_t i) const ;

    // Declared here; defined in PauliSumUtils.h
    T zero_state_expectation() const;
    T basis_state_expectation(const std::string& bits) const;
    T trace() const;
    T lookup(const PauliString<N>& ps) const;  // coefficient of ps in the sum
    T lookup(const std::string& pauli_string) const;

    // Truncation & algebra primitives; defined in PauliSumUtils.h.
    size_t term_weight(size_t i) const;      // # non-identity sites of term i
    void canonicalize();                     // fold sign flag into coeff, combine, drop zeros
    void trim(size_t max_terms);             // keep the max_terms largest-|coeff| terms
    void trim(size_t max_terms, const PauliSum& keep);  // …but never drop keep's strings
    void cutoff(T eps);                      // drop terms with |coeff| < eps
    void truncate_weight(size_t max_weight); // drop terms with weight > max_weight
    void scale(T c);
    T frobenius_norm2() const;               // tr(O†O)/2^n for a canonicalized sum
    void compact(const std::vector<uint8_t>& keep_mask);
    void sort_and_combine();
    void multiply(const PauliSum<N, T>& rhs);

    // Element-wise (Hadamard) product: result[i] = (*this)[i] * rhs[i].
    // Both sums must have the same length.  Much faster than the full outer
    // product when only pair-wise multiplication is needed.
    void elementwise_multiply(const PauliSum<N, T>& rhs);

    void evolve(const PauliSum<N, T>& rotations);
    inline bool commute(std::size_t i, const PauliString<N>& ps);
    inline void multiply(const size_t i, const PauliString<N>& ps);

    
    inline bool commute(std::size_t i, std::size_t j);
    std::vector<std::vector<std::size_t>> greedy_group();



    // Declared here; defined in PauliSumClifford.h
    void apply_H(size_t c);
    void apply_H_YZ(size_t c);
    void apply_H_XY(size_t c);
    void apply_S(size_t c);
    void apply_sqrt_X(size_t c);
    void apply_sqrt_Y(size_t c);
    void apply_X(size_t c);
    void apply_Y(size_t c);
    void apply_Z(size_t c);
    void apply_I(size_t c);
    void apply_CNOT(size_t c, size_t t);
    void apply_CZ(size_t c, size_t t);
    void apply_CY(size_t c, size_t t);
    void apply_XC_X(size_t c, size_t t);
    void apply_XC_Y(size_t c, size_t t);
    void apply_XC_Z(size_t c, size_t t);
    void apply_YC_X(size_t c, size_t t);
    void apply_YC_Y(size_t c, size_t t);
    void apply_YC_Z(size_t c, size_t t);
    void apply_SWAP(size_t c, size_t t);
    void apply_iSWAP(size_t c, size_t t);
    void apply_RZ(size_t c, T angle);
    void apply_RX(size_t c, T angle);
    void apply_RY(size_t c, T angle);

    inline void set_Z(size_t i, size_t c) ;
    inline void set_X(size_t i, size_t c) ;
    inline void set_Y(size_t i, size_t c) ;

    inline void multiply_iX(size_t i, size_t c) ;
    inline void multiply_iZ(size_t i, size_t c) ;
    inline void multiply_iY(size_t i, size_t c) ;

    inline bool commute_with_Z(size_t i, size_t c) const;
    inline bool commute_with_X(size_t i, size_t c) const;
    inline bool commute_with_Y(size_t i, size_t c) const;
};

template <std::size_t N, typename T>
PauliSum<N, T>::PauliSum(const std::vector<std::string>& pauli_strings)
{
    std::size_t nq = 0;
    for (const auto& s : pauli_strings) {
        PauliString<N> ps(s);
        nq = std::max(nq, ps.num_qubits());
        push_back(ps);
    }
    if (nq != 0) num_qubits_ = nq;
}

template <std::size_t N, typename T>
PauliSum<N, T>::PauliSum(const std::vector<std::pair<std::string, T>>& terms) {
    std::size_t nq = 0;
    for (const auto& [s, coeff] : terms) {
        PauliString<N> ps(s);
        nq = std::max(nq, ps.num_qubits());
        push_back(ps, coeff);
    }
    if (nq != 0) num_qubits_ = nq;
}

template <std::size_t N, typename T>
PauliSum<N, T>::PauliSum(size_t qubits, std::initializer_list<std::string> pauli_strings) {
    if (qubits > N)
        throw std::invalid_argument("qubit count exceeds capacity");
    num_qubits_ = qubits;
    for (const auto& s : pauli_strings) {
        push_back(PauliString<N>(s));
    }
}

template <std::size_t N, typename T>
PauliSum<N, T>::PauliSum(size_t qubits, const std::string& pauli_string, T coefficient) {
    if (qubits > N)
        throw std::invalid_argument("qubit count exceeds capacity");
    num_qubits_ = qubits;
    push_back(PauliString<N>(pauli_string), coefficient);
}

// Overloaded operators for PauliSum

template <std::size_t N, typename T>
PauliSum<N, T>& operator*=(PauliSum<N, T>& lhs, const PauliSum<N, T>& rhs) {
    lhs.multiply(rhs);
    return lhs;
}

template <std::size_t N, typename T>
PauliSum<N, T> operator*(PauliSum<N, T> lhs, const PauliSum<N, T>& rhs) {
    lhs *= rhs;
    return lhs;
}

template <std::size_t N, typename T>
PauliSum<N, T>& operator*=(PauliSum<N, T>& lhs, T c) {
    lhs.scale(c);
    return lhs;
}

template <std::size_t N, typename T>
PauliSum<N, T> operator*(PauliSum<N, T> lhs, T c) {
    lhs *= c;
    return lhs;
}

template <std::size_t N, typename T>
PauliSum<N, T> operator*(T c, PauliSum<N, T> rhs) {
    rhs *= c;
    return rhs;
}

template <std::size_t N, typename T>
void add(PauliSum<N, T>& lhs, const PauliSum<N, T>& rhs) {
    for (size_t i = 0; i < PauliSum<N, T>::ArraySize; ++i) {
        lhs.x_data[i].insert(lhs.x_data[i].end(), rhs.x_data[i].begin(), rhs.x_data[i].end());
        lhs.z_data[i].insert(lhs.z_data[i].end(), rhs.z_data[i].begin(), rhs.z_data[i].end());
    }
    lhs.flags.insert(lhs.flags.end(), rhs.flags.begin(), rhs.flags.end());
    lhs.value.insert(lhs.value.end(), rhs.value.begin(), rhs.value.end());
}

// lhs += c * rhs, appended without combining — callers batch several
// appends and then run canonicalize()/sort_and_combine() once.
template <std::size_t N, typename T>
void append_scaled(PauliSum<N, T>& lhs, T c, const PauliSum<N, T>& rhs) {
    for (size_t k = 0; k < PauliSum<N, T>::ArraySize; ++k) {
        lhs.x_data[k].insert(lhs.x_data[k].end(), rhs.x_data[k].begin(), rhs.x_data[k].end());
        lhs.z_data[k].insert(lhs.z_data[k].end(), rhs.z_data[k].begin(), rhs.z_data[k].end());
    }
    lhs.flags.insert(lhs.flags.end(), rhs.flags.begin(), rhs.flags.end());
    lhs.value.reserve(lhs.value.size() + rhs.value.size());
    for (const T& v : rhs.value) lhs.value.push_back(c * v);
}

template <std::size_t N, typename T>
PauliSum<N, T>& operator+=(PauliSum<N, T>& lhs, const PauliSum<N, T>& rhs) {
    add(lhs, rhs);
    lhs.sort_and_combine();
    return lhs;
}

template <std::size_t N, typename T>
PauliSum<N, T> operator+(PauliSum<N, T> lhs, const PauliSum<N, T>& rhs) {
    lhs += rhs;
    return lhs;
}

template <std::size_t N, typename T>
PauliSum<N, T>& operator-=(PauliSum<N, T>& lhs, const PauliSum<N, T>& rhs) {
    append_scaled(lhs, T{-1}, rhs);
    lhs.sort_and_combine();
    return lhs;
}

template <std::size_t N, typename T>
PauliSum<N, T> operator-(PauliSum<N, T> lhs, const PauliSum<N, T>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <std::size_t N, typename T>
PauliSum<N, T> operator-(PauliSum<N, T> s) {
    s.scale(T{-1});
    return s;
}

template <std::size_t N, typename T>
inline std::ostream& operator<<(std::ostream& os, const PauliSum<N, T>& obj) {
    os << obj.get_str();
    return os;
}

} // namespace paulib
