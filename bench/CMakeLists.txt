set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
add_subdirectory(${PROJECT_SOURCE_DIR}/external/googlebenchmark ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark EXCLUDE_FROM_ALL)
target_compile_definitions(benchmark PRIVATE NDEBUG)
target_compile_definitions(benchmark_main PRIVATE NDEBUG)

# Vectorized build
add_executable(pauli_mul PauliSumMulBench.cpp)
target_link_libraries(pauli_mul PRIVATE paulib benchmark::benchmark)
target_compile_options(pauli_mul PRIVATE)

# Collate (sort + merge duplicates) benchmark
add_executable(pauli_collate CollateBench.cpp)
target_link_libraries(pauli_collate PRIVATE paulib benchmark::benchmark)
target_compile_options(pauli_collate PRIVATE)

# Thread-level parallelism scaling (outer-product speedup vs. OMP threads)
add_executable(threading_bench ThreadingBench.cpp)
target_link_libraries(threading_bench PRIVATE paulib)
target_compile_definitions(threading_bench PRIVATE NDEBUG)

# g-sim workload benchmark (TFIM closure / setup / per-gate)
add_executable(gsim_bench GSimBench.cpp)
target_link_libraries(gsim_bench PRIVATE paulib)
target_compile_definitions(gsim_bench PRIVATE NDEBUG)

# Pauli propagation workload benchmark (mixed-field Ising)
add_executable(propagation_bench PropagationBench.cpp)
target_link_libraries(propagation_bench PRIVATE paulib)
target_compile_definitions(propagation_bench PRIVATE NDEBUG)

# GPU SoA multiply benchmark (GH200 / H100)
# Kernel lives in paulib_gpu (src/gpu/). GpuMulBench.cpp is pure g++.
if(CMAKE_CUDA_COMPILER)
    add_executable(gpu_mul_bench GpuMulBench.cpp)
    target_link_libraries(gpu_mul_bench PRIVATE paulib paulib_gpu benchmark::benchmark)
    target_compile_definitions(gpu_mul_bench PRIVATE NDEBUG)
endif()

# Distributed-memory scaling benchmarks (CoolMUC-4 etc.). Hand-rolled
# MPI_Wtime timing rather than Google Benchmark, which doesn't drive
# multi-rank runs. Only built when an MPI implementation is found.
#
# NOTE: this repo's include/mpi/MpiPauliSum.h uses a hash-partitioned
# ownership design (exchange_and_combine / mpi_i_commutator), not the
# row-wise-sharded + sample-sort design the paper's Table mpi_collate and
# CoolMUC-4 numbers were originally measured with. mpi_mul_bench and
# mpi_collate_bench below are re-implemented on top of the hash-partitioned
# API (mpi_multiply added alongside mpi_i_commutator for this purpose) --
# re-running them here reproduces the qualitative distributed-memory claim
# (single-node OOM -> multi-rank success, super-linear collate scaling),
# but the specific numbers in the paper's table were measured with the
# older algorithm and are not expected to reproduce bit-for-bit.
if(MPI_FOUND)
    add_executable(mpi_mul_bench MpiMulBench.cpp)
    target_link_libraries(mpi_mul_bench PRIVATE paulib paulib_mpi)
    target_compile_definitions(mpi_mul_bench PRIVATE NDEBUG)

    add_executable(mpi_collate_bench MpiCollateBench.cpp)
    target_link_libraries(mpi_collate_bench PRIVATE paulib paulib_mpi)
    target_compile_definitions(mpi_collate_bench PRIVATE NDEBUG)
endif()
