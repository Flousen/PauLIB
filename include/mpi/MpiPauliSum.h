#pragma once

// Distributed Pauli sums via MPI — an OPTIONAL layer: the core library has no
// MPI dependency; include this header only from MPI-enabled targets (link
// MPI::MPI_CXX).
//
// Decomposition: every term is owned by the rank selected by a hash of its
// (x, z) bit pattern. The phase flags are excluded from the hash, so +P, -P,
// and iP meet on the same rank and sort_and_combine/canonicalize stay
// rank-local — no global merge is ever needed. The building blocks:
//
//   * MpiPauliSum<N,T>            sharded sum (thin wrapper over PauliSum)
//   * exchange_and_combine()      route terms to their owner ranks
//                                 (MPI_Alltoallv) and combine locally
//   * mpi_i_commutator(H, O)      H small + replicated, O sharded; the
//                                 rank-local i[H, O_local] is computed with
//                                 the (OpenMP-parallel) core kernel, then
//                                 routed — a distributed hash join
//   * trim_global(M)              global largest-M selection via threshold
//                                 bisection (keeps >= M terms, plus ties)
//
// SPMD convention: replicated inputs (H, O0, options) must be constructed
// identically on every rank.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "PauliSum.h"

namespace paulib {

template <std::size_t N, typename T = double>
class MpiPauliSum {
    static_assert(std::is_same_v<T, double>,
                  "MpiPauliSum currently supports T=double (MPI_DOUBLE)");

public:
    using Sum = PauliSum<N, T>;
    static constexpr std::size_t AS = Sum::ArraySize;
    static constexpr std::size_t WordsPerTerm = 2 * AS;

    explicit MpiPauliSum(MPI_Comm comm) : comm_(comm) {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }

    // Build from a sum replicated on every rank: keep only owned terms.
    static MpiPauliSum from_replicated(const Sum& global, MPI_Comm comm) {
        MpiPauliSum out(comm);
        out.local_.num_qubits_ = global.num_qubits();
        for (std::size_t i = 0; i < global.get_length(); ++i)
            if (out.owner(global, i) == out.rank_)
                out.push_term(global, i);
        out.local_.canonicalize();
        return out;
    }

    Sum& local() { return local_; }
    const Sum& local() const { return local_; }
    MPI_Comm comm() const { return comm_; }
    int rank() const { return rank_; }
    int ranks() const { return size_; }

    // FNV-1a over the (x, z) words; flags excluded by design.
    static std::uint64_t pattern_hash(const Sum& s, std::size_t i) {
        std::uint64_t h = 0xcbf29ce484222325ull;
        for (std::size_t k = 0; k < AS; ++k) {
            h = (h ^ s.x_data[k][i]) * 0x100000001b3ull;
            h = (h ^ s.z_data[k][i]) * 0x100000001b3ull;
        }
        return h;
    }

    int owner(const Sum& s, std::size_t i) const {
        return static_cast<int>(pattern_hash(s, i) % static_cast<std::uint64_t>(size_));
    }

    std::size_t global_terms() const {
        unsigned long long l = local_.get_length(), g = 0;
        MPI_Allreduce(&l, &g, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
        return static_cast<std::size_t>(g);
    }

    T norm2() const {
        T l = local_.frobenius_norm2(), g = T{};
        MPI_Allreduce(&l, &g, 1, MPI_DOUBLE, MPI_SUM, comm_);
        return g;
    }

    void scale(T c) { local_.scale(c); }

    // Route every local term to its owner rank, then combine locally.
    void exchange_and_combine() {
        const std::size_t n = local_.get_length();

        std::vector<int> dest(n);
        std::vector<int> send_counts(size_, 0);
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = owner(local_, i);
            ++send_counts[dest[i]];
        }

        std::vector<int> recv_counts(size_, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, comm_);

        std::vector<int> sdispl(size_ + 1, 0), rdispl(size_ + 1, 0);
        for (int r = 0; r < size_; ++r) {
            sdispl[r + 1] = sdispl[r] + send_counts[r];
            rdispl[r + 1] = rdispl[r] + recv_counts[r];
        }
        const std::size_t total_recv = static_cast<std::size_t>(rdispl[size_]);

        // Pack terms grouped by destination: 2*AS words + 1 flags byte +
        // 1 coefficient per term.
        std::vector<std::uint64_t> swords(n * WordsPerTerm);
        std::vector<std::uint8_t> sflags(n);
        std::vector<T> svals(n);
        {
            std::vector<int> cursor(sdispl.begin(), sdispl.end() - 1);
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t pos = static_cast<std::size_t>(cursor[dest[i]]++);
                std::uint64_t* w = swords.data() + pos * WordsPerTerm;
                for (std::size_t k = 0; k < AS; ++k) {
                    w[k] = local_.x_data[k][i];
                    w[AS + k] = local_.z_data[k][i];
                }
                sflags[pos] = local_.flags[i];
                svals[pos] = local_.value[i];
            }
        }

        auto scaled = [this](const std::vector<int>& c, std::size_t f) {
            std::vector<int> out(size_);
            for (int r = 0; r < size_; ++r) out[r] = c[r] * static_cast<int>(f);
            return out;
        };
        auto scaled_displ = [this](const std::vector<int>& d, std::size_t f) {
            std::vector<int> out(size_);
            for (int r = 0; r < size_; ++r) out[r] = d[r] * static_cast<int>(f);
            return out;
        };

        std::vector<std::uint64_t> rwords(total_recv * WordsPerTerm);
        std::vector<std::uint8_t> rflags(total_recv);
        std::vector<T> rvals(total_recv);

        const auto swc = scaled(send_counts, WordsPerTerm);
        const auto rwc = scaled(recv_counts, WordsPerTerm);
        const auto swd = scaled_displ(sdispl, WordsPerTerm);
        const auto rwd = scaled_displ(rdispl, WordsPerTerm);
        MPI_Alltoallv(swords.data(), swc.data(), swd.data(), MPI_UINT64_T,
                      rwords.data(), rwc.data(), rwd.data(), MPI_UINT64_T, comm_);
        MPI_Alltoallv(sflags.data(), send_counts.data(), sdispl.data(), MPI_UINT8_T,
                      rflags.data(), recv_counts.data(), rdispl.data(), MPI_UINT8_T, comm_);
        MPI_Alltoallv(svals.data(), send_counts.data(), sdispl.data(), MPI_DOUBLE,
                      rvals.data(), recv_counts.data(), rdispl.data(), MPI_DOUBLE, comm_);

        const std::size_t nq = local_.num_qubits();
        local_ = Sum{};
        local_.num_qubits_ = nq;
        local_.reserve(total_recv);
        for (std::size_t i = 0; i < total_recv; ++i) {
            const std::uint64_t* w = rwords.data() + i * WordsPerTerm;
            for (std::size_t k = 0; k < AS; ++k) {
                local_.x_data[k].push_back(w[k]);
                local_.z_data[k].push_back(w[AS + k]);
            }
            local_.flags.push_back(rflags[i]);
            local_.value.push_back(rvals[i]);
        }
        local_.canonicalize();
    }

    // Keep (at least) the max_terms globally largest-|coefficient| terms:
    // bisection on the magnitude threshold with Allreduce counts, then a
    // local cutoff. Ties at the threshold are kept, so the result can hold
    // slightly more than max_terms.
    void trim_global(std::size_t max_terms) {
        if (global_terms() <= max_terms) return;

        T local_max = T{};
        for (const T& v : local_.value) local_max = std::max(local_max, std::abs(v));
        T hi = T{};
        MPI_Allreduce(&local_max, &hi, 1, MPI_DOUBLE, MPI_MAX, comm_);
        T lo = T{};

        auto global_count_at = [&](T t) {
            unsigned long long c = 0, g = 0;
            for (const T& v : local_.value) c += (std::abs(v) >= t);
            MPI_Allreduce(&c, &g, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
            return static_cast<std::size_t>(g);
        };

        for (int it = 0; it < 60 && hi - lo > T{0}; ++it) {
            const T mid = lo + (hi - lo) / 2;
            if (mid == lo || mid == hi) break;
            if (global_count_at(mid) >= max_terms) lo = mid;
            else hi = mid;
        }
        local_.cutoff(lo == T{} ? std::numeric_limits<T>::denorm_min() : lo);
    }

    // Replicate the full sum on every rank (intended for tests / small sums).
    Sum gather() const {
        const int n = static_cast<int>(local_.get_length());
        std::vector<int> counts(size_, 0);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm_);
        std::vector<int> displ(size_ + 1, 0);
        for (int r = 0; r < size_; ++r) displ[r + 1] = displ[r] + counts[r];
        const std::size_t total = static_cast<std::size_t>(displ[size_]);

        std::vector<std::uint64_t> swords(static_cast<std::size_t>(n) * WordsPerTerm);
        for (int i = 0; i < n; ++i) {
            std::uint64_t* w = swords.data() + static_cast<std::size_t>(i) * WordsPerTerm;
            for (std::size_t k = 0; k < AS; ++k) {
                w[k] = local_.x_data[k][i];
                w[AS + k] = local_.z_data[k][i];
            }
        }
        std::vector<int> wcounts(size_), wdispl(size_);
        for (int r = 0; r < size_; ++r) {
            wcounts[r] = counts[r] * static_cast<int>(WordsPerTerm);
            wdispl[r] = displ[r] * static_cast<int>(WordsPerTerm);
        }
        std::vector<std::uint64_t> rwords(total * WordsPerTerm);
        std::vector<std::uint8_t> rflags(total);
        std::vector<T> rvals(total);
        MPI_Allgatherv(swords.data(), n * static_cast<int>(WordsPerTerm), MPI_UINT64_T,
                       rwords.data(), wcounts.data(), wdispl.data(), MPI_UINT64_T, comm_);
        MPI_Allgatherv(local_.flags.data(), n, MPI_UINT8_T,
                       rflags.data(), counts.data(), displ.data(), MPI_UINT8_T, comm_);
        MPI_Allgatherv(local_.value.data(), n, MPI_DOUBLE,
                       rvals.data(), counts.data(), displ.data(), MPI_DOUBLE, comm_);

        Sum out;
        out.num_qubits_ = local_.num_qubits();
        out.reserve(total);
        for (std::size_t i = 0; i < total; ++i) {
            const std::uint64_t* w = rwords.data() + i * WordsPerTerm;
            for (std::size_t k = 0; k < AS; ++k) {
                out.x_data[k].push_back(w[k]);
                out.z_data[k].push_back(w[AS + k]);
            }
            out.flags.push_back(rflags[i]);
            out.value.push_back(rvals[i]);
        }
        out.canonicalize();
        return out;
    }

private:
    void push_term(const Sum& src, std::size_t i) {
        for (std::size_t k = 0; k < AS; ++k) {
            local_.x_data[k].push_back(src.x_data[k][i]);
            local_.z_data[k].push_back(src.z_data[k][i]);
        }
        local_.flags.push_back(src.flags[i]);
        local_.value.push_back(src.value[i]);
    }

    Sum local_;
    MPI_Comm comm_;
    int rank_ = 0;
    int size_ = 1;
};

// i[H, O] for a replicated Hamiltonian and a sharded operator; the result is
// sharded. The rank-local commutator uses the OpenMP-parallel core kernel.
template <std::size_t N, typename T>
MpiPauliSum<N, T> mpi_i_commutator(const PauliSum<N, T>& h,
                                   const MpiPauliSum<N, T>& o) {
    MpiPauliSum<N, T> out(o.comm());
    out.local() = i_commutator(h, o.local());
    out.local().num_qubits_ = std::max(h.num_qubits(), o.local().num_qubits());
    out.exchange_and_combine();
    return out;
}

}  // namespace paulib
