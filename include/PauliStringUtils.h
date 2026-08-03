#pragma once

#include <cstring>
#include <stdexcept>
#include <string>

#include "PauliStringCore.h"

namespace paulib {

template <std::size_t N>
inline void PauliString<N>::set_pauli(std::size_t i, char p) {
    if (i >= num_qubits_) throw std::out_of_range("Pauli index out of range");
    if (!std::strchr("XZYI", p)) throw std::invalid_argument("Invalid Pauli character");

    const std::size_t word = i / BitsPerWord;
    const std::size_t mask = std::size_t(1) << (i % BitsPerWord);

    x[word] &= ~mask;
    z[word] &= ~mask;

    switch (p) {
        case 'X':
            x[word] |= mask;
            break;
        case 'Z':
            z[word] |= mask;
            break;
        case 'Y':
            x[word] |= mask;
            z[word] |= mask;
            break;
        case 'I':
            break;
        default:
            throw std::invalid_argument("Invalid Pauli character");
    }
}

template <std::size_t N>
inline PauliString<N>::PauliString(const std::string& str) {
    if (str.empty()) {
        throw std::invalid_argument("Pauli string is empty");
    }

    bool sign = false;
    bool imag = false;
    std::size_t pos = 0;

    if (str.rfind("-i", 0) == 0) {
        sign = true;
        imag = true;
        pos = 2;
    } else if (str.rfind("+i", 0) == 0) {
        imag = true;
        pos = 2;
    } else if (str[0] == '-') {
        sign = true;
        pos = 1;
    } else if (str[0] == '+' || str[0] == 'i') {
        imag = (str[0] == 'i');
        pos = 1;
    }

    const std::size_t nq = str.size() - pos;
    if (nq > N) {
        throw std::invalid_argument("Pauli string has " + std::to_string(nq) +
            " operator characters, exceeding the capacity of " +
            std::to_string(N) + " qubits");
    }
    num_qubits_ = static_cast<uint32_t>(nq);

    setSign(sign);
    setPhase(imag);
    for (std::size_t q = 0; q < nq; ++q) {
        set_pauli(q, str[pos + q]);
    }
}

inline char decode(bool x, bool z) {
    if (x) return z ? 'Y' : 'X';
    return z ? 'Z' : 'I';
}

template <std::size_t N>
inline std::string PauliString<N>::to_string() const {
    std::string out(num_qubits_, 'I');

    for (std::size_t i = 0; i < num_qubits_; ++i) {
        const std::size_t word = i / BitsPerWord;
        const std::size_t mask = std::size_t(1) << (i % BitsPerWord);
        out[i] = decode((x[word] & mask) != 0, (z[word] & mask) != 0);
    }
    static const char* const stringMap[] = {"", "i", "-", "-i"};
    std::string prefix = stringMap[flags & 0x03];
    return prefix + out;
}

} // namespace paulib
