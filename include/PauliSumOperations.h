#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <omp.h>

#include "PauliString.h"

namespace paulib {

inline bool isCloseOne(double value, double epsilon = 1e-14) {
    return std::abs(value - 1) <= epsilon;
}

inline bool isCloseToSqrtTwo(double value, double epsilon = 1e-14) {
    static const double sqrt_two_inv = 1 / std::sqrt(2);
    return std::abs(value - sqrt_two_inv) <= epsilon;
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_RZ(size_t c, T angle) {
    const T cos_half = std::cos(angle / 2);
    const T sin_half = std::sin(angle / 2);
    const size_t n = get_length();

    PauliSum<N, T> new_;

    for (size_t i = 0; i < n; ++i) {
        if (commute_with_Z(i, c)) continue;

        const T t_val = value[i] * sin_half;
        value[i] *= cos_half;

        for (size_t k = 0; k < ArraySize; ++k) {
            new_.x_data[k].push_back(x_data[k][i]);
            new_.z_data[k].push_back(z_data[k][i]);
        }
        new_.flags.push_back(flags[i]);
        new_.value.push_back(t_val);

        new_.multiply_iZ(new_.get_length() - 1, c);
    }
    *this += new_;
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_RX(size_t c, T angle) {
    const T cos_half = std::cos(angle / 2);
    const T sin_half = std::sin(angle / 2);
    const size_t n = get_length();

    PauliSum<N, T> new_;

    for (size_t i = 0; i < n; ++i) {
        if (commute_with_X(i, c)) continue;

        const T t_val = value[i] * sin_half;
        value[i] *= cos_half;

        for (size_t k = 0; k < ArraySize; ++k) {
            new_.x_data[k].push_back(x_data[k][i]);
            new_.z_data[k].push_back(z_data[k][i]);
        }
        new_.flags.push_back(flags[i]);
        new_.value.push_back(t_val);

        new_.multiply_iX(new_.get_length() - 1, c);
    }
    *this += new_;
}

template <std::size_t N, typename T>
void PauliSum<N, T>::apply_RY(size_t c, T angle) {
    const T cos_half = std::cos(angle / 2);
    const T sin_half = std::sin(angle / 2);
    const size_t n = get_length();

    PauliSum<N, T> new_;

    for (size_t i = 0; i < n; ++i) {
        if (commute_with_Y(i, c)) continue;

        const T t_val = value[i] * sin_half;
        value[i] *= cos_half;

        for (size_t k = 0; k < ArraySize; ++k) {
            new_.x_data[k].push_back(x_data[k][i]);
            new_.z_data[k].push_back(z_data[k][i]);
        }
        new_.flags.push_back(flags[i]);
        new_.value.push_back(t_val);

        new_.multiply_iY(new_.get_length() - 1, c);
    }
    *this += new_;
}

template <std::size_t N, typename T>
void PauliSum<N, T>::evolve(const PauliSum<N, T>& rotations) {
    const size_t n = rotations.get_length();

    for (size_t i = 0; i < n; ++i) {
        auto ps = rotations.get_pauli_string(i);
        ps.flip_phase();
        const T cos_theta = std::cos(rotations.value[i]);
        const T sin_theta = std::sin(rotations.value[i]);

        // Recapture observable length each iteration: *this may have grown.
        const size_t m = get_length();

        constexpr size_t kParallelThreshold = size_t{1} << 13;

        if (isCloseOne(cos_theta)) {
            // θ ≈ 0: identity, nothing to do.
            continue;
        } else if (isCloseOne(-cos_theta)) {
            // θ ≈ π: Q → -Q  (sin(π)=0, new term vanishes).
            #pragma omp parallel for schedule(static) if (m >= kParallelThreshold)
            for (size_t j = 0; j < m; ++j) {
                if (commute(j, ps)) continue;
                flip_sign(j);
            }
        } else if (isCloseOne(sin_theta)) {
            // θ ≈ π/2: Q → i·P·Q  (cos(π/2)=0).
            // i·P·Q = -(Q·ps)  since ps=iP and {P,Q}=0.
            // flip_sign then multiply: -Q·ps = -(Q·ps) = iPQ.
            #pragma omp parallel for schedule(static) if (m >= kParallelThreshold)
            for (size_t j = 0; j < m; ++j) {
                if (commute(j, ps)) continue;
                flip_sign(j);
                multiply(j, ps);
            }
        } else if (isCloseOne(-sin_theta)) {
            // θ ≈ -π/2: Q → -i·P·Q = +(Q·ps)  since ps=iP and {P,Q}=0.
            #pragma omp parallel for schedule(static) if (m >= kParallelThreshold)
            for (size_t j = 0; j < m; ++j) {
                if (commute(j, ps)) continue;
                multiply(j, ps);
            }
        } else {
            // General: Q → cos(θ)·Q + i·sin(θ)·P·Q.
            // i·sin(θ)·P·Q = -sin(θ)·(Q·ps)  since ps=iP and {P,Q}=0.
            // Parallel over a FIXED chunk grid; per-chunk buffers are
            // concatenated in chunk order, so the resulting term order is
            // independent of the thread count. Writes to value[j] are
            // disjoint across chunks.
            constexpr size_t kChunks = 64;
            std::array<PauliSum<N, T>, kChunks> parts;
            #pragma omp parallel for schedule(dynamic) if (m >= kParallelThreshold)
            for (size_t c = 0; c < kChunks; ++c) {
                const size_t lo = c * m / kChunks;
                const size_t hi = (c + 1) * m / kChunks;
                PauliSum<N, T>& local = parts[c];
                for (size_t j = lo; j < hi; ++j) {
                    if (commute(j, ps)) continue;

                    const T orig_val = value[j];
                    value[j] *= cos_theta;

                    const size_t new_idx = local.get_length();
                    for (size_t k = 0; k < ArraySize; ++k) {
                        local.x_data[k].push_back(x_data[k][j]);
                        local.z_data[k].push_back(z_data[k][j]);
                    }
                    local.flags.push_back(flags[j]);
                    local.value.push_back(-sin_theta * orig_val);
                    local.multiply(new_idx, ps);
                }
            }
            PauliSum<N, T> new_;
            for (size_t c = 0; c < kChunks; ++c) add(new_, parts[c]);
            *this += new_;
        }
    }
}

template <std::size_t N, typename T>
inline void PauliSum<N, T>::multiply(const size_t i, const PauliString<N>& ps) {
    size_t total_c1 = 0;
    size_t total_c2 = 0;

    for (size_t k = 0; k < ArraySize; ++k) {
        const size_t ox = x_data[k][i];
        const size_t oz = z_data[k][i];
        const size_t bx = ps.x[k];
        const size_t bz = ps.z[k];

        x_data[k][i] = ox ^ bx;
        z_data[k][i] = oz ^ bz;

        const size_t xbz  = ox & bz;
        const size_t anti = (bx & oz) ^ xbz;

        total_c2 ^= (total_c1 ^ ox ^ oz ^ bx ^ bz ^ xbz) & anti;
        total_c1 ^= anti;
    }
    const int p1 = std::popcount(total_c1);
    const int p2 = __builtin_parityll(total_c2);

    flags[i] = (flags[i] + ps.flags + p1 + 2 * p2) & 0x03;
}

template <std::size_t N, typename T>
inline bool PauliSum<N, T>::commute(std::size_t i, const PauliString<N>& ps) {
    std::size_t acc = 0;
    for (std::size_t k = 0; k < ArraySize; ++k)
        acc ^= (x_data[k][i] & ps.z[k]) ^ (z_data[k][i] & ps.x[k]);
    return __builtin_parityll(acc) == 0;
}


template <std::size_t N, typename T>
void PauliSum<N, T>::elementwise_multiply(const PauliSum<N, T>& rhs) {
    const size_t n = get_length();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        size_t total_c1 = 0;
        size_t total_c2 = 0;

        for (size_t k = 0; k < ArraySize; ++k) {
            const size_t ox = x_data[k][i];
            const size_t oz = z_data[k][i];
            const size_t bx = rhs.x_data[k][i];
            const size_t bz = rhs.z_data[k][i];

            x_data[k][i] = ox ^ bx;
            z_data[k][i] = oz ^ bz;

            const size_t xbz  = ox & bz;
            const size_t anti = (bx & oz) ^ xbz;

            total_c2 ^= (total_c1 ^ ox ^ oz ^ bx ^ bz ^ xbz) & anti;
            total_c1 ^= anti;
        }

        flags[i] = (flags[i] + rhs.flags[i]
                    + static_cast<uint8_t>(std::popcount(total_c1))
                    + 2 * static_cast<uint8_t>(std::popcount(total_c2))) & 0x03;
        value[i] *= rhs.value[i];
    }
}

// k-outer, j-inner reordering: stride-1 access on every array, SIMD-vectorisable j loop.
// Temporary c1[j]/c2[j] arrays accumulate the phase across k words per output row;
// they are allocated per i iteration so the OMP-parallel outer loop stays race-free.
template <std::size_t N, typename T>
void PauliSum<N, T>::multiply(const PauliSum<N, T>& rhs) {
    const size_t n = get_length();
    const size_t m = rhs.get_length();

    auto orig_x    = std::move(x_data);
    auto orig_z    = std::move(z_data);
    auto orig_flags = std::move(flags);
    auto orig_val   = std::move(value);

    const size_t total_size = n * m;
    for (size_t k = 0; k < ArraySize; ++k) {
        x_data[k].assign(total_size, 0);
        z_data[k].assign(total_size, 0);
    }
    flags.assign(total_size, 0);
    value.assign(total_size, 0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; ++i) {
        const size_t base = i * n;

        std::vector<size_t> c1(n, 0);
        std::vector<size_t> c2(n, 0);

        for (size_t k = 0; k < ArraySize; ++k) {
            const size_t bx = rhs.x_data[k][i];
            const size_t bz = rhs.z_data[k][i];

            #pragma omp simd
            for (size_t j = 0; j < n; ++j) {
                const size_t ox  = orig_x[k][j];
                const size_t oz  = orig_z[k][j];
                x_data[k][base + j] = ox ^ bx;
                z_data[k][base + j] = oz ^ bz;

                const size_t xbz  = ox & bz;
                const size_t anti = (bx & oz) ^ xbz;
                c2[j] ^= (c1[j] ^ ox ^ oz ^ bx ^ bz ^ xbz) & anti;
                c1[j] ^= anti;
            }
        }

        #pragma omp simd
        for (size_t j = 0; j < n; ++j) {
            flags[base + j] = (orig_flags[j] + rhs.flags[i]
                + static_cast<uint8_t>(std::popcount(c1[j]))
                + 2 * static_cast<uint8_t>(__builtin_parityll(c2[j]))) & 0x03;
            value[base + j] = orig_val[j] * rhs.value[i];
        }
    }
}

// [A, B] and i[A, B] without forming the full product: commuting pairs are
// skipped and [P, Q] = 2PQ for anticommuting Pauli strings, with the ±i of
// the product carried by the phase flags. i[A, B] of two Hermitian
// real-coefficient sums is again Hermitian with real coefficients (the extra
// i shifts the phase exponent to even), which keeps the RK4/Lanczos drivers
// free of complex coefficients. The result is uncombined (up to one term per
// anticommuting pair); callers run canonicalize().
template <std::size_t N, typename T, bool MultiplyByI>
PauliSum<N, T> commutator_impl(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    const size_t n = a.get_length();
    const size_t m = b.get_length();

    std::vector<PauliString<N>> b_strings(m);
    for (size_t j = 0; j < m; ++j) b_strings[j] = b.get_pauli_string(j);

    PauliSum<N, T> out;
    out.num_qubits_ = std::max(a.num_qubits(), b.num_qubits());

    #pragma omp parallel
    {
        PauliSum<N, T> local;
        local.num_qubits_ = out.num_qubits_;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; ++i) {
            const PauliString<N> pa = a.get_pauli_string(i);
            for (size_t j = 0; j < m; ++j) {
                if (pa.commute(b_strings[j])) continue;
                PauliString<N> p = pa * b_strings[j];
                if constexpr (MultiplyByI) p.flags = (p.flags + 1) & 0x03;
                local.push_back(p, T{2} * a.value[i] * b.value[j]);
            }
        }

        #pragma omp critical
        add(out, local);
    }
    return out;
}

template <std::size_t N, typename T>
PauliSum<N, T> commutator(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    return commutator_impl<N, T, false>(a, b);
}

template <std::size_t N, typename T>
PauliSum<N, T> i_commutator(const PauliSum<N, T>& a, const PauliSum<N, T>& b) {
    return commutator_impl<N, T, true>(a, b);
}

// Sorts terms and merges duplicates by summing coefficients. The phase/sign
// flags are part of the term key, so e.g. +X and -X stay separate terms
// (folding the sign into the coefficient would lose the ±i phases, which a
// real-valued T cannot absorb).
//
// Large sums are sorted in parallel: a FIXED number of chunks (independent of
// the thread count) is stable-sorted concurrently, then merged pairwise with
// stable merges. The output order — and therefore the floating-point
// summation order of duplicates — is identical for every OMP_NUM_THREADS.
// The duplicate merge is parallelized over the resulting segments.
template <std::size_t N, typename T>
void PauliSum<N, T>::sort_and_combine() {
    const size_t n = get_length();
    if (n == 0) return;

    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    auto key_less = [&](size_t a, size_t b) {
        for (size_t k = 0; k < ArraySize; ++k) {
            if (x_data[k][a] != x_data[k][b])
                return x_data[k][a] < x_data[k][b];
        }
        for (size_t k = 0; k < ArraySize; ++k) {
            if (z_data[k][a] != z_data[k][b])
                return z_data[k][a] < z_data[k][b];
        }
        return flags[a] < flags[b];
    };

    constexpr size_t kParallelThreshold = size_t{1} << 14;
    constexpr size_t kChunks = 16;
    if (n >= kParallelThreshold) {
        std::array<size_t, kChunks + 1> bounds;
        for (size_t c = 0; c <= kChunks; ++c) bounds[c] = c * n / kChunks;

        #pragma omp parallel for schedule(dynamic)
        for (size_t c = 0; c < kChunks; ++c)
            std::stable_sort(idx.begin() + bounds[c], idx.begin() + bounds[c + 1], key_less);

        std::vector<size_t> tmp(n);
        for (size_t width = 1; width < kChunks; width *= 2) {
            #pragma omp parallel for schedule(dynamic)
            for (size_t c = 0; c < kChunks; c += 2 * width) {
                const size_t lo  = bounds[c];
                const size_t mid = bounds[std::min(c + width, kChunks)];
                const size_t hi  = bounds[std::min(c + 2 * width, kChunks)];
                if (mid >= hi) continue;
                std::merge(idx.begin() + lo, idx.begin() + mid,
                           idx.begin() + mid, idx.begin() + hi,
                           tmp.begin() + lo, key_less);
                std::copy(tmp.begin() + lo, tmp.begin() + hi, idx.begin() + lo);
            }
        }
    } else {
        std::stable_sort(idx.begin(), idx.end(), key_less);
    }

    auto same_pauli = [&](size_t a, size_t b) {
        for (size_t k = 0; k < ArraySize; ++k) {
            if (x_data[k][a] != x_data[k][b]) return false;
            if (z_data[k][a] != z_data[k][b]) return false;
        }
        return flags[a] == flags[b];
    };

    // Segment starts: positions in the sorted order where a new key begins.
    std::vector<size_t> starts;
    starts.reserve(n);
    starts.push_back(0);
    for (size_t i = 1; i < n; ++i)
        if (!same_pauli(idx[i], idx[i - 1])) starts.push_back(i);
    const size_t m = starts.size();

    WordArray new_x;
    WordArray new_z;
    std::vector<uint8_t> new_flags(m);
    std::vector<T> new_value(m);
    for (size_t k = 0; k < ArraySize; ++k) {
        new_x[k].resize(m);
        new_z[k].resize(m);
    }

    // Per-segment reduction; within a segment coefficients are summed in
    // sorted order, so the result is deterministic.
    #pragma omp parallel for schedule(static) if (n >= kParallelThreshold)
    for (size_t s = 0; s < m; ++s) {
        const size_t first = idx[starts[s]];
        for (size_t k = 0; k < ArraySize; ++k) {
            new_x[k][s] = x_data[k][first];
            new_z[k][s] = z_data[k][first];
        }
        new_flags[s] = flags[first];
        const size_t end = (s + 1 < m) ? starts[s + 1] : n;
        T acc = T{};
        for (size_t i = starts[s]; i < end; ++i) acc += value[idx[i]];
        new_value[s] = acc;
    }

    x_data = std::move(new_x);
    z_data = std::move(new_z);
    flags = std::move(new_flags);
    value = std::move(new_value);
}

template <std::size_t N, typename T>
inline bool PauliSum<N, T>::commute(std::size_t i, std::size_t j) {
    std::size_t acc = 0;
    for (std::size_t k = 0; k < ArraySize; ++k)
        acc ^= (x_data[k][i] & z_data[k][j]) ^ (z_data[k][i] & x_data[k][j]);
    return __builtin_parityll(acc) == 0;
}


// Thread-parallel greedy grouping: the input is split into one contiguous
// chunk per OMP thread, each thread groups its chunk independently, and a
// sequential merge folds local groups into the global list. The partition can
// differ across thread counts, but every group is still mutually commuting
// and every index appears exactly once.
template <std::size_t N, typename T>
std::vector<std::vector<std::size_t>> PauliSum<N, T>::greedy_group() {
    const std::size_t n = get_length();
    if (n == 0) return {};

    std::vector<std::vector<std::vector<std::size_t>>> local_groups;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();

        #pragma omp single
        local_groups.resize(num_threads);

        const std::size_t chunk_size = (n + num_threads - 1) / num_threads;
        const std::size_t start = tid * chunk_size;
        const std::size_t end = std::min(start + chunk_size, n);

        std::vector<std::vector<std::size_t>> my_groups;

        for (std::size_t i = start; i < end; ++i) {
            bool placed = false;
            for (auto& grp : my_groups) {
                bool ok = true;
                for (std::size_t j : grp) {
                    if (!commute(i, j)) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    grp.push_back(i);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                my_groups.push_back({i});
            }
        }
        local_groups[tid] = std::move(my_groups);
    }

    // Merge stage: append each local group to the first compatible global group.
    std::vector<std::vector<std::size_t>> final_groups;

    for (const auto& my_groups : local_groups) {
        for (const auto& lg : my_groups) {
            bool placed = false;
            for (auto& gg : final_groups) {
                bool ok = true;
                for (std::size_t l_idx : lg) {
                    for (std::size_t g_idx : gg) {
                        if (!commute(l_idx, g_idx)) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok) break;
                }
                if (ok) {
                    gg.insert(gg.end(), lg.begin(), lg.end());
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                final_groups.push_back(lg);
            }
        }
    }

    return final_groups;
}

} // namespace paulib
