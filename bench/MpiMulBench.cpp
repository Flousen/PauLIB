// Distributed outer-product scaling benchmark: replicated-RHS mpi_multiply.
// Hand-rolled MPI_Wtime timing (Google Benchmark doesn't drive multi-rank
// runs). Usage: mpirun -n P ./mpi_mul_bench [lhs_total_terms] [rhs_terms]

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include "PauLib.h"
#include "mpi/MpiPauliSum.h"

using namespace paulib;

static constexpr std::size_t NUM_QUBITS = 500;
using T = double;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::size_t lhs_total = argc > 1 ? std::stoull(argv[1]) : 1000000;
    const std::size_t rhs_n     = argc > 2 ? std::stoull(argv[2]) : 1000;

    const std::size_t lhs_local =
        lhs_total / static_cast<std::size_t>(size) +
        (static_cast<std::size_t>(rank) < lhs_total % static_cast<std::size_t>(size) ? 1 : 0);

    PauliSum<NUM_QUBITS, T> local_lhs = random_pauli_sum<NUM_QUBITS, T>(lhs_local, 1000 + static_cast<std::uint64_t>(rank));
    // Same seed on every rank -> bit-identical without an explicit Bcast;
    // this IS the "replicated" half of the replicated-RHS strategy.
    PauliSum<NUM_QUBITS, T> rhs = random_pauli_sum<NUM_QUBITS, T>(rhs_n, 42);

    MpiPauliSum<NUM_QUBITS, T> dist_lhs(MPI_COMM_WORLD);
    dist_lhs.local() = std::move(local_lhs);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    auto dist_result = mpi_multiply(dist_lhs, rhs);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double local_elapsed = t1 - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const std::size_t global_out = dist_result.global_terms();
    if (rank == 0) {
        const double pairs_per_sec = static_cast<double>(global_out) / max_elapsed;
        std::printf("ranks,lhs_total,rhs_n,out_terms,seconds,pairs_per_sec\n");
        std::printf("%d,%zu,%zu,%zu,%.6f,%.3e\n",
                     size, lhs_total, rhs_n, global_out, max_elapsed, pairs_per_sec);
    }

    MPI_Finalize();
    return 0;
}
