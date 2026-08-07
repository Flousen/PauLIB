// g-sim workload benchmark: TFIM chains. Measures Lie closure, structure
// constants + adjoint rotation setup, per-gate cost on the expectation
// vector, and the total time for a 200-layer circuit sweep (paper Sec.
// "Workload Case Studies", Table gsim).
//
// Each stage is timed as the minimum over kReps runs after kWarmup untimed
// runs (same methodology as the core-operation benchmarks, Sec.
// "Core-Operation Benchmarks"): closure/setup finish in well under a
// millisecond for small n, where a single untimed run is dominated by
// allocator/cache warm-up noise rather than the algorithm itself.
//
// Usage: ./gsim_bench [max_n]     (default 64; runs n = 16, 32, ..., max_n)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "PauLib.h"

using namespace paulib;
using clk = std::chrono::steady_clock;

namespace {
constexpr int kWarmup = 3;
constexpr int kReps = 5;

template <typename F>
double min_ms(F f) {
    for (int i = 0; i < kWarmup; ++i) f();
    const auto time_once = [&] {
        const auto t0 = clk::now();
        f();
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    double best = time_once();
    for (int i = 1; i < kReps; ++i) best = std::min(best, time_once());
    return best;
}
}  // namespace

int main(int argc, char** argv) {
    const std::size_t max_n = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 64;

    std::printf("%4s %8s %14s %14s %14s %10s %14s\n",
                "n", "dim", "closure[ms]", "setup[ms]", "gate[us]", "gates",
                "circuit[ms]");
    for (std::size_t n = 16; n <= max_n; n *= 2) {
        std::vector<PauliString<64>> gens;
        for (std::size_t q = 0; q + 1 < n; ++q) {
            std::string s(n, 'I');
            s[q] = 'Z';
            s[q + 1] = 'Z';
            gens.emplace_back(s);
        }
        for (std::size_t q = 0; q < n; ++q) {
            std::string s(n, 'I');
            s[q] = 'X';
            gens.emplace_back(s);
        }

        std::vector<PauliString<64>> basis;
        const double t_closure = min_ms([&] { basis = lie_closure(gens); });

        std::unique_ptr<GSim<64>> sim;
        const double t_setup = min_ms([&] { sim = std::make_unique<GSim<64>>(gens); });

        std::size_t gates = 0;
        const double t_gates = min_ms([&] {
            std::mt19937 rng(1);
            std::uniform_real_distribution<double> dist(-1.5, 1.5);
            auto e = sim->zero_state_vector();
            gates = 0;
            for (int layer = 0; layer < 200; ++layer)
                for (std::size_t g = 0; g < gens.size(); ++g) {
                    sim->apply_gate(e, g, dist(rng));
                    ++gates;
                }
        });

        std::printf("%4zu %8zu %14.2f %14.2f %14.3f %10zu %14.2f\n",
                    n, basis.size(), t_closure, t_setup,
                    1e3 * t_gates / static_cast<double>(gates), gates, t_gates);
    }
    return 0;
}
