add_library(paulib INTERFACE)

target_include_directories(paulib
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

# Eigen (header-only, used by GSim.h): repo root provides Eigen/ and unsupported/
target_include_directories(paulib SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/external/eigen)

target_compile_options(paulib
    INTERFACE
        ${PAULIB_PERF_FLAGS}
)

target_link_libraries(paulib
    INTERFACE
        OpenMP::OpenMP_CXX
)

# MPI backend (include/mpi/MpiPauliSum.h) — header-only, only built when an
# MPI implementation is found. Kept as a separate target so the core paulib
# library never depends on MPI headers, same boundary discipline as paulib_gpu.
if(MPI_FOUND)
    add_library(paulib_mpi INTERFACE)
    target_link_libraries(paulib_mpi INTERFACE paulib MPI::MPI_CXX)
endif()
