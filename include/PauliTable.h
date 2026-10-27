#pragma once

// Qubit-major (transposed) table of Pauli strings, as in Stim's tableaux.
//
// PauliSum stores word k of every term contiguously, so the vector lanes
// range over terms, but a gate on qubit q uses one bit of each word.
// PauliTable stores one bit row per qubit over all terms: x_row(q) holds the
// X bit of qubit q of terms 0..m-1, 64 terms per word. A gate on one or two
// qubits is then a few word operations per 64 terms. The table stores bits
// and phases only; coefficients stay with the caller, by term index.
//
// - Phase: bit planes imag and sign over the terms, the bits 0 and 1 of the
//   flags byte (term = i^k P in letter form, k = imag + 2 sign).
// - Conversion: to_table / to_sum via 64 x 64 bit transposes.
// - Gates: the Clifford set of PauliSum with the same names and the same
//   Heisenberg convention P -> G^dagger P G, so apply_circuit works
//   unchanged; plus apply_S_dag, apply_sqrt_X_dag, apply_sqrt_Y_dag for the
//   inverse gates (used by Tableau).
// - anticommutes_mask, left_multiply (term_j <- P term_j for the terms of a
//   mask, phase by a bit-sliced counter) and product (ordered product of
//   the terms of a mask).
//
// Every operation is elementwise over the words of the terms or sequential,
// so the result does not depend on the thread count.

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "PauliString.h"
#include "PauliSum.h"

namespace paulib {

// In-place transpose of a 64 x 64 bit block: bit c of a[r] becomes bit r of
// a[c]. Six stages swap off-diagonal sub-blocks of size 32, 16, ..., 1
// (Hacker's Delight, Sec. 7-3); 6 * 32 swap operations in total.
inline void transpose64(std::size_t a[64]) {
    std::size_t m = 0x00000000FFFFFFFFull;
    for (std::size_t j = 32; j != 0; j >>= 1, m ^= (m << j)) {
        for (std::size_t k = 0; k < 64; k = ((k | j) + 1) & ~j) {
            const std::size_t t = ((a[k] >> j) ^ a[k | j]) & m;
            a[k] ^= t << j;
            a[k | j] ^= t;
        }
    }
}

namespace detail {
// Words per bit row above which a gate uses OpenMP threads. Starting a team
// costs ~60 us with 16 threads on a Core Ultra 7 255H; below 2^18 words
// (16M terms) the serial SIMD loop is faster. Not const, so that tests can
// lower it to cover the threaded path.
inline std::size_t kTableParallelWords = std::size_t{1} << 18;

// Threaded word loop of the table gates: blocks of words over the threads,
// each block an `omp simd` loop. With `omp parallel for simd` on one loop,
// Clang 22 did not vectorise the outlined loop (-Wpass-failed), GCC 15 did.
template <typename Body>
void table_parallel_words(std::size_t W, Body& body) {
    constexpr std::size_t kBlock = 4096;
    const std::size_t blocks = (W + kBlock - 1) / kBlock;
    #pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::size_t hi = std::min(W, (b + 1) * kBlock);
        #pragma omp simd
        for (std::size_t w = b * kBlock; w < hi; ++w) body(w);
    }
}

// Inclusive prefix XOR within a word: bit b = XOR of bits 0..b.
// dst[offset .. offset+nbits) |= src[0 .. nbits) on bit vectors over terms.
inline void or_bits(std::size_t* dst, std::size_t offset,
                    const std::size_t* src, std::size_t nbits) {
    for (std::size_t b = 0; b < nbits; b += 64) {
        const std::size_t len = std::min<std::size_t>(64, nbits - b);
        std::size_t w = src[b / 64];
        if (len < 64) w &= (std::size_t(1) << len) - 1;
        const std::size_t pos = offset + b;
        const std::size_t dw = pos / 64;
        const std::size_t sh = pos % 64;
        dst[dw] |= w << sh;
        if (sh != 0 && sh + len > 64) dst[dw + 1] |= w >> (64 - sh);
    }
}

// dst[0 .. nbits) = src[offset .. offset+nbits); dst has ceil(nbits/64) words.
inline void get_bits(std::size_t* dst, const std::size_t* src,
                     std::size_t offset, std::size_t nbits) {
    for (std::size_t b = 0; b < nbits; b += 64) {
        const std::size_t len = std::min<std::size_t>(64, nbits - b);
        const std::size_t pos = offset + b;
        const std::size_t sw = pos / 64;
        const std::size_t sh = pos % 64;
        std::size_t w = src[sw] >> sh;
        if (sh != 0 && sh + len > 64) w |= src[sw + 1] << (64 - sh);
        if (len < 64) w &= (std::size_t(1) << len) - 1;
        dst[b / 64] = w;
    }
}

inline std::size_t prefix_xor(std::size_t v) {
    v ^= v << 1;
    v ^= v << 2;
    v ^= v << 4;
    v ^= v << 8;
    v ^= v << 16;
    v ^= v << 32;
    return v;
}
}  // namespace detail

class PauliTable {
public:
    static constexpr std::size_t BitsPerWord = 64;

    std::vector<std::size_t> xbits;  // n rows of words() words: row q = X bits of qubit q
    std::vector<std::size_t> zbits;  // n rows of words() words: Z bits
    std::vector<std::size_t> imag;   // ceil(m/64) words: bit 0 of k
    std::vector<std::size_t> sign;   // ceil(m/64) words: bit 1 of k

    PauliTable() = default;
    PauliTable(std::size_t n_qubits, std::size_t n_terms)
        : xbits(n_qubits * ((n_terms + BitsPerWord - 1) / BitsPerWord), 0),
          zbits(n_qubits * ((n_terms + BitsPerWord - 1) / BitsPerWord), 0),
          imag((n_terms + BitsPerWord - 1) / BitsPerWord, 0),
          sign((n_terms + BitsPerWord - 1) / BitsPerWord, 0),
          n_(n_qubits), m_(n_terms) {}

    std::size_t num_qubits() const { return n_; }
    std::size_t size() const { return m_; }
    std::size_t words() const { return imag.size(); }

    // Bit row of qubit q: words() words, bit j of the row = term j.
    std::size_t* x_row(std::size_t q) { return xbits.data() + q * words(); }
    std::size_t* z_row(std::size_t q) { return zbits.data() + q * words(); }
    const std::size_t* x_row(std::size_t q) const { return xbits.data() + q * words(); }
    const std::size_t* z_row(std::size_t q) const { return zbits.data() + q * words(); }

    bool get_x(std::size_t q, std::size_t j) const { return (x_row(q)[j / BitsPerWord] >> (j % BitsPerWord)) & 1; }
    bool get_z(std::size_t q, std::size_t j) const { return (z_row(q)[j / BitsPerWord] >> (j % BitsPerWord)) & 1; }
    void set_x(std::size_t q, std::size_t j, bool v) { put_row(x_row(q), j, v); }
    void set_z(std::size_t q, std::size_t j, bool v) { put_row(z_row(q), j, v); }
    uint8_t get_phase(std::size_t j) const {
        return static_cast<uint8_t>(bit(imag, j) | (bit(sign, j) << 1));
    }
    void set_phase(std::size_t j, uint8_t k) {
        put(imag, j, k & 1);
        put(sign, j, (k >> 1) & 1);
    }

    template <std::size_t N>
    PauliString<N> get(std::size_t j) const {
        if (n_ > N) throw std::invalid_argument("PauliTable::get: qubit count exceeds capacity");
        PauliString<N> ps{};
        ps.num_qubits_ = static_cast<uint32_t>(n_);
        for (std::size_t q = 0; q < n_; ++q) {
            const std::size_t m = std::size_t(1) << (q % BitsPerWord);
            if (get_x(q, j)) ps.x[q / BitsPerWord] |= m;
            if (get_z(q, j)) ps.z[q / BitsPerWord] |= m;
        }
        ps.flags = get_phase(j);
        return ps;
    }

    template <std::size_t N>
    void set(std::size_t j, const PauliString<N>& ps) {
        for (std::size_t q = 0; q < n_; ++q) {
            set_x(q, j, (ps.x[q / BitsPerWord] >> (q % BitsPerWord)) & 1);
            set_z(q, j, (ps.z[q / BitsPerWord] >> (q % BitsPerWord)) & 1);
        }
        set_phase(j, ps.flags & 3);
    }

    // Copies term src to term dst (O(n) bit operations).
    void copy_term(std::size_t dst, std::size_t src) {
        for (std::size_t q = 0; q < n_; ++q) {
            set_x(q, dst, get_x(q, src));
            set_z(q, dst, get_z(q, src));
        }
        set_phase(dst, get_phase(src));
    }

    // ---------------------------------------------------------------------
    // Clifford gates, P -> G^dagger P G. The phase rule of each gate is the
    // one of the PauliSum gate of the same name, applied to 64 terms per word.

    void apply_H(std::size_t c)      { single(c, [](std::size_t& x, std::size_t& z, std::size_t& s) { s ^= x & z; std::swap(x, z); }); }
    void apply_S(std::size_t c)      { single(c, [](std::size_t& x, std::size_t& z, std::size_t& s) { s ^= x & ~z; z ^= x; }); }
    void apply_H_YZ(std::size_t c)   { single(c, [](std::size_t& x, std::size_t& z, std::size_t& s) { s ^= x & ~z; x ^= z; }); }
    void apply_H_XY(std::size_t c)   { single(c, [](std::size_t& x, std::size_t& z, std::size_t& s) { s ^= ~x & z; z ^= x; }); }
    void apply_sqrt_X(std::size_t c) { single(c, [](std::size_t& x, std::size_t& z, std::size_t& s) { s ^= x & z; x ^= z; }); }
    void apply_sqrt_Y(std::size_t c) { single(c, [](std::size_t& x, std::size_t& z, std::size_t& s) { s ^= ~x & z; std::swap(x, z); }); }
    // The inverse gates, P -> G P G^dagger.
    void apply_S_dag(std::size_t c)      { single(c, [](std::size_t& x, std::size_t& z, std::size_t& s) { s ^= x & z; z ^= x; }); }
    void apply_sqrt_X_dag(std::size_t c) { single(c, [](std::size_t& x, std::size_t& z, std::size_t& s) { s ^= ~x & z; x ^= z; }); }
    void apply_sqrt_Y_dag(std::size_t c) { single(c, [](std::size_t& x, std::size_t& z, std::size_t& s) { s ^= x & ~z; std::swap(x, z); }); }
    void apply_X(std::size_t c)      { single(c, [](std::size_t&, std::size_t& z, std::size_t& s) { s ^= z; }); }
    void apply_Y(std::size_t c)      { single(c, [](std::size_t& x, std::size_t& z, std::size_t& s) { s ^= x ^ z; }); }
    void apply_Z(std::size_t c)      { single(c, [](std::size_t& x, std::size_t&, std::size_t& s) { s ^= x; }); }
    void apply_I(std::size_t) {}

    void apply_CNOT(std::size_t c, std::size_t t) {
        pair(c, t, [](std::size_t& xc, std::size_t& zc, std::size_t& xt, std::size_t& zt, std::size_t& s) {
            s ^= xc & zt & ~(xt ^ zc);
            xt ^= xc;
            zc ^= zt;
        });
    }
    void apply_CZ(std::size_t c, std::size_t t) {
        pair(c, t, [](std::size_t& xc, std::size_t& zc, std::size_t& xt, std::size_t& zt, std::size_t& s) {
            s ^= xc & xt & (zc ^ zt);
            zt ^= xc;
            zc ^= xt;
        });
    }
    void apply_SWAP(std::size_t c, std::size_t t) {
        pair(c, t, [](std::size_t& xc, std::size_t& zc, std::size_t& xt, std::size_t& zt, std::size_t&) {
            std::swap(xc, xt);
            std::swap(zc, zt);
        });
    }

    // ---------------------------------------------------------------------
    // term_j <- i^k (px | pz) * term_j for every j with bit j of mask set.
    // px, pz have ceil(n/64) words. Per qubit, the letter product contributes
    // i^g with g in {0, +1, -1}; g is accumulated per term in a bit-sliced
    // 2-bit counter (c0, c1), 64 terms per word operation.
    void left_multiply(const std::vector<std::size_t>& mask,
                       const std::size_t* px, const std::size_t* pz, uint8_t k) {
        const std::size_t W = words();
        std::vector<std::size_t> c0(W, 0), c1(W, 0);
        for (std::size_t q = 0; q < n_; ++q) {
            const bool sx = (px[q / BitsPerWord] >> (q % BitsPerWord)) & 1;
            const bool sz = (pz[q / BitsPerWord] >> (q % BitsPerWord)) & 1;
            if (!sx && !sz) continue;
            std::size_t* xr = x_row(q);
            std::size_t* zr = z_row(q);
            for (std::size_t w = 0; w < W; ++w) {
                const std::size_t m = mask[w];
                const std::size_t a = xr[w], b = zr[w];
                std::size_t plus, minus;
                if (sx && !sz) {        // X: X Y = iZ, X Z = -iY
                    plus = a & b;  minus = ~a & b;
                } else if (!sx && sz) { // Z: Z X = iY, Z Y = -iX
                    plus = a & ~b; minus = a & b;
                } else {                // Y: Y Z = iX, Y X = -iZ
                    plus = ~a & b; minus = a & ~b;
                }
                plus &= m;
                minus &= m;
                std::size_t carry = c0[w] & plus;
                c0[w] ^= plus;
                c1[w] ^= carry;
                const std::size_t borrow = ~c0[w] & minus;
                c0[w] ^= minus;
                c1[w] ^= borrow;
                if (sx) xr[w] ^= m;
                if (sz) zr[w] ^= m;
            }
        }
        const std::size_t k0 = (k & 1) ? ~std::size_t{0} : 0;
        const std::size_t k1 = (k & 2) ? ~std::size_t{0} : 0;
        for (std::size_t w = 0; w < W; ++w) {
            // (imag, sign) += (c0, c1) + (k0, k1) & mask, modulo 4
            const std::size_t m = mask[w];
            std::size_t b0 = c0[w], b1 = c1[w];
            const std::size_t kk0 = k0 & m, kk1 = k1 & m;
            b1 ^= kk1 ^ (b0 & kk0);
            b0 ^= kk0;
            sign[w] ^= b1 ^ (imag[w] & b0);
            imag[w] ^= b0;
        }
    }

    template <std::size_t N>
    void left_multiply(const std::vector<std::size_t>& mask, const PauliString<N>& p) {
        left_multiply(mask, p.x.data(), p.z.data(), p.flags & 3);
    }

    // left_multiply by term src (bit src of mask must be clear).
    void left_multiply(const std::vector<std::size_t>& mask, std::size_t src) {
        std::vector<std::size_t> px, pz;
        uint8_t k = 0;
        extract(src, px, pz, k);
        left_multiply(mask, px.data(), pz.data(), k);
    }

    // Ordered product of the terms selected by mask, in increasing index:
    // writes the bits to px, pz (ceil(n/64) words) and returns the phase
    // exponent k of the letter form. Per qubit, with letters
    // i^{x_a z_a} X^{x_a} Z^{z_a}, moving all X to the left gives
    // sum_a x_a z_a + 2 #{a < b : z_a = 1, x_b = 1} - X Z (mod 4), where
    // X, Z are the parities. The pair count uses a prefix XOR of the Z bits.
    // O(n m / 64) word operations, row by row.
    uint8_t product(const std::vector<std::size_t>& mask,
                    std::vector<std::size_t>& px, std::vector<std::size_t>& pz) const {
        const std::size_t W = words();
        const std::size_t QW = (n_ + BitsPerWord - 1) / BitsPerWord;
        px.assign(QW, 0);
        pz.assign(QW, 0);
        std::size_t e = 0;
        for (std::size_t w = 0; w < W; ++w)
            e += static_cast<std::size_t>(std::popcount(imag[w] & mask[w]))
               + 2 * static_cast<std::size_t>(std::popcount(sign[w] & mask[w]));
        for (std::size_t q = 0; q < n_; ++q) {
            const std::size_t* xr = x_row(q);
            const std::size_t* zr = z_row(q);
            std::size_t s1 = 0, xp = 0, zp = 0, pairs = 0;
            for (std::size_t w = 0; w < W; ++w) {
                const std::size_t a = xr[w] & mask[w];
                const std::size_t b = zr[w] & mask[w];
                if (!(a | b)) continue;
                s1 += static_cast<std::size_t>(std::popcount(a & b));
                // Z bits strictly before each position, including earlier words.
                std::size_t before = detail::prefix_xor(b) << 1;
                if (zp) before = ~before;
                pairs ^= static_cast<std::size_t>(std::popcount(a & before)) & 1;
                xp ^= static_cast<std::size_t>(std::popcount(a)) & 1;
                zp ^= static_cast<std::size_t>(std::popcount(b)) & 1;
            }
            e += s1 + 2 * pairs + 4 - (xp & zp);
            if (xp) px[q / BitsPerWord] |= std::size_t(1) << (q % BitsPerWord);
            if (zp) pz[q / BitsPerWord] |= std::size_t(1) << (q % BitsPerWord);
        }
        return static_cast<uint8_t>(e & 3);
    }

    // Bits and phase of term j as words (ceil(n/64) per part).
    void extract(std::size_t j, std::vector<std::size_t>& px, std::vector<std::size_t>& pz,
                 uint8_t& k) const {
        const std::size_t QW = (n_ + BitsPerWord - 1) / BitsPerWord;
        px.assign(QW, 0);
        pz.assign(QW, 0);
        for (std::size_t q = 0; q < n_; ++q) {
            const std::size_t m = std::size_t(1) << (q % BitsPerWord);
            if (get_x(q, j)) px[q / BitsPerWord] |= m;
            if (get_z(q, j)) pz[q / BitsPerWord] |= m;
        }
        k = get_phase(j);
    }

private:
    std::size_t n_ = 0;
    std::size_t m_ = 0;

    static bool bit(const std::vector<std::size_t>& v, std::size_t j) {
        return (v[j / BitsPerWord] >> (j % BitsPerWord)) & 1;
    }
    static void put(std::vector<std::size_t>& v, std::size_t j, bool b) {
        const std::size_t m = std::size_t(1) << (j % BitsPerWord);
        if (b) v[j / BitsPerWord] |= m;
        else   v[j / BitsPerWord] &= ~m;
    }
    static void put_row(std::size_t* row, std::size_t j, bool b) {
        const std::size_t m = std::size_t(1) << (j % BitsPerWord);
        if (b) row[j / BitsPerWord] |= m;
        else   row[j / BitsPerWord] &= ~m;
    }

    template <typename F>
    void single(std::size_t c, F f) {
        std::size_t* __restrict__ xr = x_row(c);
        std::size_t* __restrict__ zr = z_row(c);
        std::size_t* __restrict__ sr = sign.data();
        const std::size_t W = words();
        auto body = [&](std::size_t w) {
            std::size_t a = xr[w], b = zr[w], s = sr[w];
            f(a, b, s);
            xr[w] = a; zr[w] = b; sr[w] = s;
        };
        // An OpenMP region with if(false) still costs a team start per gate,
        // which dominates small tables (a tableau has 2n terms), hence the
        // explicit branch.
        if (W >= detail::kTableParallelWords) detail::table_parallel_words(W, body);
        else {
            #pragma omp simd
            for (std::size_t w = 0; w < W; ++w) body(w);
        }
    }

    template <typename F>
    void pair(std::size_t c, std::size_t t, F f) {
        if (c == t) throw std::invalid_argument("PauliTable: two-qubit gate on one qubit");
        std::size_t* __restrict__ xc = x_row(c);
        std::size_t* __restrict__ zc = z_row(c);
        std::size_t* __restrict__ xt = x_row(t);
        std::size_t* __restrict__ zt = z_row(t);
        std::size_t* __restrict__ sr = sign.data();
        const std::size_t W = words();
        auto body = [&](std::size_t w) {
            std::size_t a = xc[w], b = zc[w], d = xt[w], e = zt[w], s = sr[w];
            f(a, b, d, e, s);
            xc[w] = a; zc[w] = b; xt[w] = d; zt[w] = e; sr[w] = s;
        };
        if (W >= detail::kTableParallelWords) detail::table_parallel_words(W, body);
        else {
            #pragma omp simd
            for (std::size_t w = 0; w < W; ++w) body(w);
        }
    }
};

// Bit vector over the terms (ceil(m/64) words): bit j = 1 if term j
// anticommutes with ps. Parity of the rows of the qubits in the support of
// ps: z rows where ps has X, x rows where ps has Z.
template <std::size_t N>
std::vector<std::size_t> anticommutes_mask(const PauliTable& t, const PauliString<N>& ps) {
    const std::size_t W = t.words();
    std::vector<std::size_t> acc(W, 0);
    for (std::size_t q = 0; q < t.num_qubits(); ++q) {
        const bool px = (ps.x[q / 64] >> (q % 64)) & 1;
        const bool pz = (ps.z[q / 64] >> (q % 64)) & 1;
        if (px) { const std::size_t* r = t.z_row(q); for (std::size_t w = 0; w < W; ++w) acc[w] ^= r[w]; }
        if (pz) { const std::size_t* r = t.x_row(q); for (std::size_t w = 0; w < W; ++w) acc[w] ^= r[w]; }
    }
    return acc;
}

// Qubit-major copy of the strings and phases of a sum; the coefficients are
// not copied. Word k of 64 consecutive terms is one 64 x 64 block, which
// transpose64 turns into the rows of qubits 64k .. 64k+63.
template <std::size_t N, typename T>
PauliTable to_table(const PauliSum<N, T>& sum) {
    const std::size_t n = sum.num_qubits();
    const std::size_t m = sum.get_length();
    PauliTable t(n, m);
    const std::size_t W = t.words();
    // Blocks of 64 terms write disjoint words, so they run in parallel.
    #pragma omp parallel for schedule(static) if (W >= 64)
    for (std::size_t b = 0; b < W; ++b) {
        std::size_t block[64];
        for (int part = 0; part < 2; ++part) {
            const auto& src = part ? sum.z_data : sum.x_data;
            for (std::size_t k = 0; k * 64 < n; ++k) {
                for (std::size_t i = 0; i < 64; ++i) {
                    const std::size_t j = b * 64 + i;
                    block[i] = j < m ? src[k][j] : 0;
                }
                transpose64(block);
                for (std::size_t i = 0; i < 64 && k * 64 + i < n; ++i)
                    (part ? t.z_row(k * 64 + i) : t.x_row(k * 64 + i))[b] = block[i];
            }
        }
        std::size_t im = 0, sg = 0;
        for (std::size_t i = 0; i < 64 && b * 64 + i < m; ++i) {
            const uint8_t f = sum.flags[b * 64 + i];
            im |= std::size_t(f & 1) << i;
            sg |= std::size_t((f >> 1) & 1) << i;
        }
        t.imag[b] = im;
        t.sign[b] = sg;
    }
    return t;
}

// Inverse of to_table. values gives the coefficients by term index; empty
// means 1 for every term.
template <std::size_t N, typename T>
PauliSum<N, T> to_sum(const PauliTable& t, const std::vector<T>& values = {}) {
    const std::size_t n = t.num_qubits();
    const std::size_t m = t.size();
    if (n > N) throw std::invalid_argument("to_sum: qubit count exceeds capacity");
    if (!values.empty() && values.size() != m)
        throw std::invalid_argument("to_sum: one value per term required");
    PauliSum<N, T> s;
    s.num_qubits_ = n;
    for (std::size_t k = 0; k < PauliSum<N, T>::ArraySize; ++k) {
        s.x_data[k].assign(m, 0);
        s.z_data[k].assign(m, 0);
    }
    s.flags.resize(m);
    const std::size_t W = t.words();
    #pragma omp parallel for schedule(static) if (W >= 64)
    for (std::size_t b = 0; b < W; ++b) {
        std::size_t block[64];
        for (int part = 0; part < 2; ++part) {
            auto& dst = part ? s.z_data : s.x_data;
            for (std::size_t k = 0; k * 64 < n; ++k) {
                for (std::size_t i = 0; i < 64; ++i) {
                    const std::size_t q = k * 64 + i;
                    block[i] = q < n ? (part ? t.z_row(q) : t.x_row(q))[b] : 0;
                }
                transpose64(block);
                for (std::size_t i = 0; i < 64 && b * 64 + i < m; ++i)
                    dst[k][b * 64 + i] = block[i];
            }
        }
        for (std::size_t i = 0; i < 64 && b * 64 + i < m; ++i)
            s.flags[b * 64 + i] = static_cast<uint8_t>(((t.imag[b] >> i) & 1) | (((t.sign[b] >> i) & 1) << 1));
    }
    if (values.empty()) s.value.assign(m, T{1});
    else s.value = values;
    return s;
}

}  // namespace paulib
