add_library(paulib INTERFACE)
add_library(paulib::paulib ALIAS paulib)

target_include_directories(paulib
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/paulib>
)

# Eigen (header-only, used by GSim.h): repo root provides Eigen/ and unsupported/;
# installed next to the PauLIB headers.
target_include_directories(paulib SYSTEM
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/eigen>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/paulib/eigen>
)

target_compile_features(paulib INTERFACE cxx_std_20)

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
if(MPI_CXX_FOUND)
    add_library(paulib_mpi INTERFACE)
    add_library(paulib::mpi ALIAS paulib_mpi)
    set_target_properties(paulib_mpi PROPERTIES EXPORT_NAME mpi)
    target_link_libraries(paulib_mpi INTERFACE paulib MPI::MPI_CXX)
endif()
