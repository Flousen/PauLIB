// MPI correctness check (standalone, run via mpirun — see tests/CMakeLists.txt):
//   1. from_replicated + gather round-trips the operator.
//   2. trim_global respects the global budget (up to threshold ties).
// Prints PASS/FAIL on rank 0; nonzero exit code on failure.

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "PauLib.h"
#include "mpi/MpiPauliSum.h"

using namespace paulib;
using Sum = PauliSum<64, double>;

static Sum xx_chain(std::size_t n) {
    Sum h;
    h.num_qubits_ = n;
    for (std::size_t q = 0; q + 1 < n; ++q) {
        std::string sx(n, 'I'), sy(n, 'I');
        sx[q] = 'X'; sx[q + 1] = 'X';
        sy[q] = 'Y'; sy[q + 1] = 'Y';
        h.push_back(PauliString<64>(sx), 1.0);
        h.push_back(PauliString<64>(sy), 1.0);
    }
    return h;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int failures = 0;
    const std::size_t n = 12;
    const Sum h = xx_chain(n);

    // 1. Scatter/gather round-trip.
    {
        auto sharded = MpiPauliSum<64>::from_replicated(h, MPI_COMM_WORLD);
        Sum round = sharded.gather();
        Sum ref = h;
        ref.canonicalize();
        bool ok = (round.get_length() == ref.get_length());
        for (std::size_t i = 0; ok && i < ref.get_length(); ++i) {
            for (std::size_t k = 0; k < Sum::ArraySize; ++k)
                ok = ok && round.x_data[k][i] == ref.x_data[k][i] &&
                     round.z_data[k][i] == ref.z_data[k][i];
            ok = ok && round.flags[i] == ref.flags[i] &&
                 std::abs(round.value[i] - ref.value[i]) < 1e-14;
        }
        if (!ok) {
            ++failures;
            if (rank == 0) std::printf("FAIL: scatter/gather round-trip\n");
        }
    }

    // 2. trim_global keeps at least the budget and reduces the term count.
    // Distinct coefficient magnitudes, so the threshold selection is exact
    // (with degenerate magnitudes trim_global keeps ties by design).
    {
        Sum big = random_pauli_sum<64, double>(500, 7, n);
        big.canonicalize();
        for (std::size_t i = 0; i < big.get_length(); ++i)
            big.value[i] = 1.0 + static_cast<double>(i);
        auto sharded = MpiPauliSum<64>::from_replicated(big, MPI_COMM_WORLD);
        const std::size_t before = sharded.global_terms();
        const std::size_t budget = before / 2;
        sharded.trim_global(budget);
        const std::size_t after = sharded.global_terms();
        if (after != budget) {
            ++failures;
            if (rank == 0)
                std::printf("FAIL: trim_global %zu -> %zu (budget %zu)\n",
                            before, after, budget);
        }
    }

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("%s (mpi_sanity_check, %d rank failures)\n",
                    global_failures ? "FAIL" : "PASS", global_failures);

    MPI_Finalize();
    return global_failures ? 1 : 0;
}
