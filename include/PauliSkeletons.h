#pragma once

// Skeletons for kernels on a PauliSum (design principles D1-D5, thesis Ch. 6,
// sec:kernel-design). A skeleton implements a pattern once - threads,
// thresholds, chunking, reproducibility - and takes the work per term as a
// lambda on a TermView (PauliOps.h).
//
// - transform_values(sum, f): value[i] = f(term). Changes no string, so the
//   sum keeps its order and stays collated.
// - term_mask(sum, pred) / filter_terms(sum, pred): evaluate a predicate per
//   term / keep the terms where it holds (order kept).
// - reduce_terms<R>(sum, f): sum of f(term) over all terms.
// - reduce_terms_binned<R>(sum, bins, key, f): per-bin sums, bin = key(term).
// - transform_terms(sum, f): f changes its term in place (strings, phase,
//   coefficient). Keeps the term order; the sum is no longer sorted if f
//   changes a string or a phase. transform_terms<true> requests
//   vectorisation, for branch-free bodies.
// - expand_terms(sum, f): f may change its term and push 0..k new terms;
//   returns the new terms in term order.
// - for_each_pair(a, b, f): one output term per pair (term i of a, term j of
//   b), at index i |b| + j.
// - merge_reduce<R>(a, b, f): sum of f(ta, tb) over the pairs with the same
//   string, for two sorted sums, in O(|a| + |b|).
//
// Reproducibility (D5): the terms are split into chunks of a fixed length
// (ExecPolicy::chunk_terms), independent of the thread count. Each chunk is
// reduced in index order, and the chunk results are combined in chunk order,
// so the result is bit-identical for every OMP_NUM_THREADS. The serial path
// below the threshold uses the same chunks, so it gives the same bits too.
//
// Threads (D2): the parallel region is entered only above
// ExecPolicy::parallel_threshold, by a real branch. An OpenMP `if` clause
// would still start a team for every call.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "PauliOps.h"
#include "PauliSumCore.h"

// Tells the compiler that the iterations of a term loop are independent, so
// that it vectorises the loop after inlining the lambda. GCC and Clang spell
// this differently and each ignores the other's pragma: with `GCC ivdep`
// alone, Clang 22 left damp_local scalar (0.77-0.96 of the old rate on an
// Apple M4 Pro, 28 Oct 2026). Clang defines __GNUC__ too, so test it first.
#if defined(__clang__)
#define PAULIB_INDEPENDENT_ITERATIONS _Pragma("clang loop vectorize(assume_safety)")
#elif defined(__GNUC__)
#define PAULIB_INDEPENDENT_ITERATIONS _Pragma("GCC ivdep")
#else
#define PAULIB_INDEPENDENT_ITERATIONS
#endif

namespace paulib {

struct ExecPolicy {
    // Below this number of terms, kernels run on the calling thread.
    std::size_t parallel_threshold = std::size_t{1} << 15;
    // Fixed chunk length for the reproducible reductions (D5). At the
    // default, 2^20 terms give 256 chunks.
    std::size_t chunk_terms = std::size_t{1} << 12;
};

// Process-wide default; tests lower the threshold and the chunk length to
// exercise the threaded path on small sums. Not thread-safe to modify while
// kernels run.
inline ExecPolicy& default_policy() {
    static ExecPolicy policy;
    return policy;
}

namespace detail {

inline std::size_t num_chunks(std::size_t n, const ExecPolicy& pol) {
    const std::size_t len = std::max<std::size_t>(pol.chunk_terms, 1);
    return (n + len - 1) / len;
}

// Calls f(c, lo, hi) for every chunk c = [lo, hi). Chunks are independent;
// above the threshold they are distributed statically over the threads.
template <typename F>
void for_each_chunk(std::size_t n, const ExecPolicy& pol, F&& f) {
    const std::size_t len = std::max<std::size_t>(pol.chunk_terms, 1);
    const std::size_t chunks = num_chunks(n, pol);
    if (n >= pol.parallel_threshold && chunks > 1) {
        #pragma omp parallel for schedule(static)
        for (std::size_t c = 0; c < chunks; ++c)
            f(c, c * len, std::min(n, (c + 1) * len));
    } else {
        for (std::size_t c = 0; c < chunks; ++c)
            f(c, c * len, std::min(n, (c + 1) * len));
    }
}

}  // namespace detail

// value[i] = f(term i) for every term.
template <std::size_t N, typename T, typename F>
void transform_values(PauliSum<N, T>& sum, F&& f, const ExecPolicy& pol = default_policy()) {
    const SumView<N, T> view(sum);
    T* v = sum.value.data();
    detail::for_each_chunk(view.length, pol, [&](std::size_t, std::size_t lo, std::size_t hi) {
        // Vectorisation (measured 28 Oct 2026 with GCC 15, damp_local on 2^20
        // terms, one thread): `#pragma omp simd` is lowered before inlining
        // and turns the TermView temporaries into per-lane arrays, so the
        // loop stayed scalar (0.6 G instead of 0.8 G terms/s). A chunk-local
        // copy of the view and the functor plus PAULIB_INDEPENDENT_ITERATIONS
        // let the vectoriser run after inlining (0.8 G terms/s).
        const SumView<N, T> local = view;
        auto fn = f;
        PAULIB_INDEPENDENT_ITERATIONS
        for (std::size_t i = lo; i < hi; ++i) v[i] = fn(TermView<N, T>{&local, i});
    });
}

// mask[i] = pred(term i) ? 1 : 0.
template <std::size_t N, typename T, typename Pred>
std::vector<uint8_t> term_mask(const PauliSum<N, T>& sum, Pred&& pred,
                               const ExecPolicy& pol = default_policy()) {
    const SumView<N, T> view(sum);
    std::vector<uint8_t> mask(view.length);
    uint8_t* m = mask.data();
    detail::for_each_chunk(view.length, pol, [&](std::size_t, std::size_t lo, std::size_t hi) {
        const SumView<N, T> local = view;  // as in transform_values
        auto fn = pred;
        PAULIB_INDEPENDENT_ITERATIONS
        for (std::size_t i = lo; i < hi; ++i) m[i] = fn(TermView<N, T>{&local, i}) ? 1 : 0;
    });
    return mask;
}

// Keeps the terms for which pred holds, in their order.
template <std::size_t N, typename T, typename Pred>
void filter_terms(PauliSum<N, T>& sum, Pred&& pred, const ExecPolicy& pol = default_policy()) {
    sum.compact(term_mask(sum, pred, pol));
}

// Sum of f(term) over all terms, reproducible (see the header comment).
template <typename R, std::size_t N, typename T, typename F>
R reduce_terms(const PauliSum<N, T>& sum, F&& f, const ExecPolicy& pol = default_policy()) {
    const SumView<N, T> view(sum);
    std::vector<R> partial(detail::num_chunks(view.length, pol), R{});
    detail::for_each_chunk(view.length, pol, [&](std::size_t c, std::size_t lo, std::size_t hi) {
        R acc{};
        for (std::size_t i = lo; i < hi; ++i) acc += f(TermView<N, T>{&view, i});
        partial[c] = acc;
    });
    R result{};
    for (const R& p : partial) result += p;
    return result;
}

// h[b] = sum of f(term) over the terms with key(term) = b, for b < bins;
// terms with key >= bins are ignored. Reproducible like reduce_terms.
template <typename R, std::size_t N, typename T, typename Key, typename F>
std::vector<R> reduce_terms_binned(const PauliSum<N, T>& sum, std::size_t bins, Key&& key,
                                   F&& f, const ExecPolicy& pol = default_policy()) {
    const SumView<N, T> view(sum);
    const std::size_t chunks = detail::num_chunks(view.length, pol);
    std::vector<R> partial(chunks * bins, R{});
    detail::for_each_chunk(view.length, pol, [&](std::size_t c, std::size_t lo, std::size_t hi) {
        R* h = partial.data() + c * bins;
        for (std::size_t i = lo; i < hi; ++i) {
            const TermView<N, T> t{&view, i};
            const std::size_t b = key(t);
            if (b < bins) h[b] += f(t);
        }
    });
    std::vector<R> result(bins, R{});
    for (std::size_t c = 0; c < chunks; ++c)
        for (std::size_t b = 0; b < bins; ++b) result[b] += partial[c * bins + b];
    return result;
}

// f(t) for every term, with t a TermRef: f may change the strings, the phase
// and the coefficient of its own term, nothing else. Order kept; a change of
// a string or a phase breaks the order of sort_and_combine.
// Vectorize = true adds PAULIB_INDEPENDENT_ITERATIONS, for branch-free bodies
// only (adjoint). On a body with a branch the forced vectorisation either
// fails (bit loops, complex arithmetic: Clang warns with -Wpass-failed) or
// runs masked and slower than the scalar loop: the Clifford angles of evolve
// took 3.3 instead of 1.3 us on 1024 terms, and canonicalize 21.5 instead of
// 19.5 ms on 2^18 terms (Apple M4 Pro, Clang 22, 29 Oct 2026).
template <bool Vectorize = false, std::size_t N, typename T, typename F>
void transform_terms(PauliSum<N, T>& sum, F&& f, const ExecPolicy& pol = default_policy()) {
    detail::for_each_chunk(sum.get_length(), pol, [&](std::size_t, std::size_t lo, std::size_t hi) {
        auto fn = f;
        if constexpr (Vectorize) {
            PAULIB_INDEPENDENT_ITERATIONS
            for (std::size_t i = lo; i < hi; ++i) fn(TermRef<N, T>{&sum, i});
        } else {
            for (std::size_t i = lo; i < hi; ++i) fn(TermRef<N, T>{&sum, i});
        }
    });
}

// Output of expand_terms for one chunk.
template <std::size_t N, typename T>
class TermEmitter {
public:
    explicit TermEmitter(PauliSum<N, T>& out) : out_(out) {}

    // Appends a copy of the string and phase of t (a TermView, TermRef or
    // PauliString) with coefficient v; the returned TermRef can change it.
    template <typename Term>
    TermRef<N, T> push(const Term& t, T v) {
        const std::size_t idx = out_.get_length();
        for (std::size_t k = 0; k < PauliSum<N, T>::ArraySize; ++k) {
            out_.x_data[k].push_back(detail::xw(t, k));
            out_.z_data[k].push_back(detail::zw(t, k));
        }
        out_.flags.push_back(flags_of(t));
        out_.value.push_back(v);
        return {&out_, idx};
    }

private:
    PauliSum<N, T>& out_;

    template <typename Term>
    static uint8_t flags_of(const Term& t) { return t.flags(); }
    static uint8_t flags_of(const PauliString<N>& p) { return p.flags; }
};

// f(t, out) for every term t (a TermRef, which f may change) with out a
// TermEmitter; returns the terms f pushed, uncombined. The new terms of one
// chunk keep the term order, and the chunks are concatenated in chunk order,
// so the result is the same for every thread count (D5).
template <std::size_t N, typename T, typename F>
PauliSum<N, T> expand_terms(PauliSum<N, T>& sum, F&& f, const ExecPolicy& pol = default_policy()) {
    const std::size_t n = sum.get_length();
    std::vector<PauliSum<N, T>> parts(detail::num_chunks(n, pol));
    detail::for_each_chunk(n, pol, [&](std::size_t c, std::size_t lo, std::size_t hi) {
        TermEmitter<N, T> out(parts[c]);
        auto fn = f;
        for (std::size_t i = lo; i < hi; ++i) fn(TermRef<N, T>{&sum, i}, out);
    });
    PauliSum<N, T> result;
    result.num_qubits_ = sum.num_qubits_;
    std::size_t total = 0;
    for (const auto& p : parts) total += p.get_length();
    result.reserve(total);
    for (const auto& p : parts) add(result, p);
    return result;
}

// Returns a sum with |a| |b| terms; f(ta, tb, o) writes the term of the pair
// (term i of a, term j of b) through the TermRef o to index i |b| + j. Every
// pair has its own slot, so the result does not depend on the thread count.
template <std::size_t N, typename T, typename F>
PauliSum<N, T> for_each_pair(const PauliSum<N, T>& a, const PauliSum<N, T>& b, F&& f,
                             const ExecPolicy& pol = default_policy()) {
    const std::size_t mb = b.get_length();
    const std::size_t m = a.get_length() * mb;
    PauliSum<N, T> out;
    out.num_qubits_ = a.num_qubits_;
    for (std::size_t k = 0; k < PauliSum<N, T>::ArraySize; ++k) {
        out.x_data[k].assign(m, 0);
        out.z_data[k].assign(m, 0);
    }
    out.flags.assign(m, 0);
    out.value.assign(m, T{});
    const SumView<N, T> va(a), vb(b);
    detail::for_each_chunk(m, pol, [&](std::size_t, std::size_t lo, std::size_t hi) {
        const SumView<N, T> la = va, lb = vb;
        auto fn = f;
        for (std::size_t p = lo; p < hi; ++p)
            fn(TermView<N, T>{&la, p / mb}, TermView<N, T>{&lb, p % mb}, TermRef<N, T>{&out, p});
    });
    return out;
}

namespace detail {

// Order of the strings (X words, then Z words), the primary key of
// sort_and_combine: negative, zero or positive.
template <typename A, typename B>
inline int compare_strings(const A& a, const B& b) {
    constexpr std::size_t W = words_of<A>::value;
    for (std::size_t k = 0; k < W; ++k)
        if (xw(a, k) != xw(b, k)) return xw(a, k) < xw(b, k) ? -1 : 1;
    for (std::size_t k = 0; k < W; ++k)
        if (zw(a, k) != zw(b, k)) return zw(a, k) < zw(b, k) ? -1 : 1;
    return 0;
}

}  // namespace detail

// For two sums in the order of sort_and_combine (is_sorted()): the sum of
// f(ta, tb) over all pairs of terms with the same string, whatever their
// phases. O(|a| + |b|) instead of a hash table. The terms of a are split
// into chunks; each chunk finds its start in b by binary search and is
// reduced in order, and the chunks are combined in chunk order, so the
// result is the same for every thread count.
template <typename R, std::size_t N, typename T, typename F>
R merge_reduce(const PauliSum<N, T>& a, const PauliSum<N, T>& b, F&& f,
               const ExecPolicy& pol = default_policy()) {
    const SumView<N, T> va(a), vb(b);
    std::vector<R> partial(detail::num_chunks(va.length, pol), R{});
    detail::for_each_chunk(va.length, pol, [&](std::size_t c, std::size_t lo, std::size_t hi) {
        const TermView<N, T> first{&va, lo};
        std::size_t jb = 0, len = vb.length;  // lower bound of a[lo] in b
        while (len > 0) {
            const std::size_t half = len / 2;
            if (detail::compare_strings(TermView<N, T>{&vb, jb + half}, first) < 0) {
                jb += half + 1;
                len -= half + 1;
            } else {
                len = half;
            }
        }
        R acc{};
        for (std::size_t i = lo; i < hi; ++i) {
            const TermView<N, T> ta{&va, i};
            while (jb < vb.length && detail::compare_strings(TermView<N, T>{&vb, jb}, ta) < 0) ++jb;
            for (std::size_t j = jb; j < vb.length; ++j) {
                const TermView<N, T> tb{&vb, j};
                if (detail::compare_strings(tb, ta) != 0) break;
                acc += f(ta, tb);
            }
        }
        partial[c] = acc;
    });
    R result{};
    for (const R& p : partial) result += p;
    return result;
}

}  // namespace paulib
