#pragma once

// Dense 2^n x 2^n reference matrices for small n (tests only).
// Basis index b = sum_q b_q 2^q: qubit 0 is the least significant bit, and
// bits[q] of basis_state_expectation() is b_q.

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "PauLib.h"

namespace dense {

using C = std::complex<double>;
using Mat = Eigen::MatrixXcd;

template <std::size_t N>
Mat pauli(const paulib::PauliString<N>& p, std::size_t n) {
    const std::size_t d = std::size_t(1) << n;
    Mat m = Mat::Zero(d, d);
    std::size_t xmask = 0, zmask = 0;
    for (std::size_t q = 0; q < n; ++q) {
        if ((p.x[q / 64] >> (q % 64)) & 1) xmask |= std::size_t(1) << q;
        if ((p.z[q / 64] >> (q % 64)) & 1) zmask |= std::size_t(1) << q;
    }
    static const C ipow[4] = {C(1, 0), C(0, 1), C(-1, 0), C(0, -1)};
    for (std::size_t b = 0; b < d; ++b) {
        // P = prod_q X^x Z^z (times i per Y): Y = i X Z, Z|b> = (-1)^b |b>.
        const int ny = std::popcount(xmask & zmask);
        const int zsign = std::popcount(zmask & b) & 1;
        C v = ipow[(p.flags + ny) & 3] * (zsign ? -1.0 : 1.0);
        m(b ^ xmask, b) = v;
    }
    return m;
}

template <std::size_t N, typename T>
Mat sum(const paulib::PauliSum<N, T>& s) {
    const std::size_t n = s.num_qubits();
    const std::size_t d = std::size_t(1) << n;
    Mat m = Mat::Zero(d, d);
    for (std::size_t i = 0; i < s.get_length(); ++i)
        m += C(s.value[i]) * pauli(s.get_pauli_string(i), n);
    return m;
}

inline Mat single(const Eigen::Matrix2cd& g, std::size_t q, std::size_t n) {
    const std::size_t d = std::size_t(1) << n;
    Mat m = Mat::Zero(d, d);
    for (std::size_t b = 0; b < d; ++b) {
        const std::size_t bq = (b >> q) & 1;
        for (std::size_t a = 0; a < 2; ++a)
            m((b & ~(std::size_t(1) << q)) | (a << q), b) += g(a, bq);
    }
    return m;
}

inline Mat gate(const paulib::CliffordGate& g, std::size_t n) {
    using K = paulib::CliffordGate;
    const std::size_t d = std::size_t(1) << n;
    Eigen::Matrix2cd u;
    const double r = 1.0 / std::sqrt(2.0);
    switch (g.kind) {
        case K::H:   u << r, r, r, -r; return single(u, g.q0, n);
        case K::S:   u << 1, 0, 0, C(0, 1); return single(u, g.q0, n);
        case K::SDG: u << 1, 0, 0, C(0, -1); return single(u, g.q0, n);
        case K::X:   u << 0, 1, 1, 0; return single(u, g.q0, n);
        case K::Y:   u << 0, C(0, -1), C(0, 1), 0; return single(u, g.q0, n);
        case K::Z:   u << 1, 0, 0, -1; return single(u, g.q0, n);
        default: break;
    }
    Mat m = Mat::Zero(d, d);
    for (std::size_t b = 0; b < d; ++b) {
        const std::size_t c = (b >> g.q0) & 1, t = (b >> g.q1) & 1;
        switch (g.kind) {
            case K::CNOT: m(b ^ (c << g.q1), b) = 1; break;
            case K::CZ:   m(b, b) = (c & t) ? -1 : 1; break;
            case K::SWAP: {
                std::size_t s = b & ~((std::size_t(1) << g.q0) | (std::size_t(1) << g.q1));
                s |= (c << g.q1) | (t << g.q0);
                m(s, b) = 1;
                break;
            }
            default: break;
        }
    }
    return m;
}

// U = g_L ... g_1
inline Mat circuit(const paulib::CliffordCircuit& c, std::size_t n) {
    Mat u = Mat::Identity(std::size_t(1) << n, std::size_t(1) << n);
    for (const auto& g : c) u = gate(g, n) * u;
    return u;
}

// exp(-i theta/2 P)
template <std::size_t N>
Mat rotation(const paulib::PauliString<N>& p, std::size_t n, double theta) {
    const std::size_t d = std::size_t(1) << n;
    return std::cos(theta / 2) * Mat::Identity(d, d) - C(0, 1) * std::sin(theta / 2) * pauli(p, n);
}

// Product state rho = (x)_q (I + r_q . sigma)/2.
inline Mat product_state(const std::vector<std::array<double, 3>>& bloch) {
    const std::size_t n = bloch.size();
    const std::size_t d = std::size_t(1) << n;
    Mat rho = Mat::Zero(d, d);
    for (std::size_t a = 0; a < d; ++a)
        for (std::size_t b = 0; b < d; ++b) {
            C v = 1;
            for (std::size_t q = 0; q < n; ++q) {
                const auto& r = bloch[q];
                const std::size_t aq = (a >> q) & 1, bq = (b >> q) & 1;
                C e;
                if (aq == 0 && bq == 0) e = (1 + r[2]) / 2;
                else if (aq == 1 && bq == 1) e = (1 - r[2]) / 2;
                else if (aq == 0) e = C(r[0], -r[1]) / 2.0;  // <0|rho|1>
                else e = C(r[0], r[1]) / 2.0;                 // <1|rho|0>
                v *= e;
            }
            rho(a, b) = v;
        }
    return rho;
}

// rho = 2^-n prod_j (I + g_j) for commuting generators.
template <std::size_t N>
Mat stabilizer_state(const std::vector<paulib::PauliString<N>>& gens, std::size_t n) {
    const std::size_t d = std::size_t(1) << n;
    Mat rho = Mat::Identity(d, d);
    for (const auto& g : gens) rho = rho * (Mat::Identity(d, d) + pauli(g, n));
    return rho / static_cast<double>(d);
}

inline double max_abs_diff(const Mat& a, const Mat& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

}  // namespace dense
