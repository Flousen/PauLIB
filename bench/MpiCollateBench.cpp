// Distributed collate scaling benchmark. Each rank generates
// terms_per_rank random terms over a small `keyspace_qubits`-qubit space (far
// below the NUM_QUBITS capacity), so duplicate Pauli strings collide both
// within and across ranks and land on ranks other than the one that
// generated them -- the property exchange_and_combine's hash-partitioned
// redistribution actually has to resolve. Hand-rolled MPI_Wtime timing.
// Usage: mpirun -n P ./mpi_collate_bench [terms_per_rank] [keyspace_qubits]

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

    const std::size_t terms_per_rank   = argc > 1 ? std::stoull(argv[1]) : 1000000;
    const std::size_t keyspace_qubits  = argc > 2 ? std::stoull(argv[2]) : 20;

    PauliSum<NUM_QUBITS, T> local = random_pauli_sum<NUM_QUBITS, T>(
        terms_per_rank, 2000 + static_cast<std::uint64_t>(rank), keyspace_qubits);

    MpiPauliSum<NUM_QUBITS, T> dist(MPI_COMM_WORLD);
    dist.local() = std::move(local);

    const std::size_t global_in = dist.global_terms();

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    dist.exchange_and_combine();
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double local_elapsed = t1 - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const std::size_t global_out = dist.global_terms();
    if (rank == 0) {
        std::printf("ranks,terms_per_rank,keyspace_qubits,in_terms,out_terms,seconds,terms_per_sec\n");
        std::printf("%d,%zu,%zu,%zu,%zu,%.6f,%.3e\n",
                     size, terms_per_rank, keyspace_qubits, global_in, global_out,
                     max_elapsed, static_cast<double>(global_in) / max_elapsed);
    }

    MPI_Finalize();
    return 0;
}
