// Pauli propagation workload benchmark: Trotterized MIXED-FIELD Ising circuit
// (ZZ couplings + transverse AND longitudinal fields — non-integrable, so the
// operator actually grows and the trim budget is exercised).
// Propagates Z_1 backwards and reports per-gate cost vs the trim budget
// (paper Sec. "Workload Case Studies", Table prop).
//
// Usage: ./propagation_bench [n] [layers] [max_log2_trim]
//        defaults: n = 40, layers = 10, trims 2^14 .. 2^18

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "PauLib.h"

using namespace paulib;

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 40;
    const std::size_t layers = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 10;
    const int max_log2 = argc > 3 ? std::atoi(argv[3]) : 18;

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-0.4, 0.4);

    PauliSum<64, double> circ;
    circ.num_qubits_ = n;
    auto rot = [&](std::size_t q, char a, std::size_t q2) {
        std::string s(n, 'I');
        s[q] = a;
        if (q2 != q) s[q2] = a;
        circ.push_back(PauliString<64>(s), dist(rng));
    };
    for (std::size_t l = 0; l < layers; ++l) {
        for (std::size_t q = 0; q + 1 < n; ++q) rot(q, 'Z', q + 1);
        for (std::size_t q = 0; q < n; ++q) rot(q, 'X', q);
        for (std::size_t q = 0; q < n; ++q) rot(q, 'Z', q);
    }
    std::printf("circuit: %zu rotation gates on %zu qubits, %zu layers\n",
                circ.get_length(), n, layers);

    for (int log2m = 14; log2m <= max_log2; log2m += 2) {
        PauliSum<64, double> o(n, std::string(1, 'Z') + std::string(n - 1, 'I'), 1.0);
        TruncationOptions<double> opts;
        opts.max_terms = std::size_t{1} << log2m;

        const auto t0 = std::chrono::steady_clock::now();
        propagate(o, circ, opts);
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::printf("trim 2^%-2d: %7.2f s total, %8.3f ms/gate, "
                    "final terms %8zu, <O> = %.6f\n",
                    log2m, secs, 1e3 * secs / static_cast<double>(circ.get_length()),
                    o.get_length(), o.zero_state_expectation());
    }
    return 0;
}
