add_subdirectory(${PROJECT_SOURCE_DIR}/external/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)

add_executable(paulib_tests
    PauliStringTests.cpp
    PauliSumTests.cpp
    CliffordExtendedTests.cpp
    GSimTests.cpp
)

target_link_libraries(paulib_tests
    PRIVATE
        paulib
        GTest::gtest_main
)

target_compile_options(paulib_tests PRIVATE)# ${PAULIB_WARNINGS} ${PAULIB_VECTORIZE_FLAGS})

include(GoogleTest)
gtest_discover_tests(paulib_tests)

# MPI layer — optional; only built when an MPI implementation is found.
find_package(MPI QUIET COMPONENTS CXX)
if(MPI_CXX_FOUND)
    add_executable(mpi_sanity_check mpi_sanity_check.cpp)
    target_link_libraries(mpi_sanity_check PRIVATE paulib MPI::MPI_CXX)
    add_test(NAME mpi_sanity_check_2ranks
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                     $<TARGET_FILE:mpi_sanity_check>)
    add_test(NAME mpi_sanity_check_4ranks
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                     $<TARGET_FILE:mpi_sanity_check>)
endif()

# GPU tests — only built when CUDA is available
if(CMAKE_CUDA_COMPILER)
    add_executable(paulib_gpu_tests GpuPauliSumTests.cpp)
    target_link_libraries(paulib_gpu_tests
        PRIVATE
            paulib
            paulib_gpu
            GTest::gtest_main
    )
    target_compile_definitions(paulib_gpu_tests PRIVATE NDEBUG)
    gtest_discover_tests(paulib_gpu_tests)
endif()
