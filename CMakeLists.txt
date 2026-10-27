cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "g++")
endif()

# Single source of the version: pyproject.toml reads it from this line.
project(paulib VERSION 0.1.0 LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 90)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tests, benchmarks and the demo app are only built by default when PauLIB is
# the top-level project, not when a consumer pulls it in via add_subdirectory.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(PAULIB_IS_TOP_LEVEL ON)
else()
    set(PAULIB_IS_TOP_LEVEL OFF)
endif()

include(GNUInstallDirs)

find_package(OpenMP REQUIRED)
find_package(MPI COMPONENTS CXX)

# Apple Silicon (M-series) clang rejects -march=native on arm64; -mcpu=native
# selects the same microarchitecture tuning there.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(PAULIB_PERF_FLAGS -O3 -mcpu=native -ffast-math)
else()
    set(PAULIB_PERF_FLAGS -O3 -march=native -ffast-math)
endif()
set(PAULIB_WARNINGS -Wall -Wextra -Wpedantic -Wno-unused-parameter)
set(PAULIB_VECTORIZE_FLAGS "")
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(PAULIB_VECTORIZE_FLAGS ) # -fopt-info-vec-missed -fopt-info-vec-optimized
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(PAULIB_VECTORIZE_FLAGS -Rpass=loop-vectorize -Rpass-missed=loop-vectorize)
endif()

add_subdirectory(src)
if(CMAKE_CUDA_COMPILER)
    add_subdirectory(src/gpu)
endif()

option(PAULIB_BUILD_APP "Build the demo app (app/main.cpp)" ${PAULIB_IS_TOP_LEVEL})
if(PAULIB_BUILD_APP)
    add_subdirectory(app)
endif()

include(CTest)
if(BUILD_TESTING AND PAULIB_IS_TOP_LEVEL)
    add_subdirectory(tests)
endif()

option(BUILD_BENCHMARKS "Build benchmarks" ${PAULIB_IS_TOP_LEVEL})
if(BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

# ---------------------------------------------------------------------------
# Install + CMake package: find_package(paulib) → paulib::paulib (+ paulib::mpi,
# paulib::gpu when they were built). Everything is in the "dev" component so
# the Python wheel (component "python") does not pick up the C++ headers.
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

set(PAULIB_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/paulib)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/paulib
    COMPONENT dev
)
# Bundled Eigen (PauLib.h includes GSim.h, so every consumer needs it).
install(DIRECTORY external/eigen/Eigen
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/paulib/eigen
    COMPONENT dev
)
install(DIRECTORY external/eigen/unsupported/Eigen
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/paulib/eigen/unsupported
    COMPONENT dev
)

set(PAULIB_EXPORT_TARGETS paulib)
if(TARGET paulib_mpi)
    list(APPEND PAULIB_EXPORT_TARGETS paulib_mpi)
endif()
if(TARGET paulib_gpu)
    list(APPEND PAULIB_EXPORT_TARGETS paulib_gpu)
endif()

install(TARGETS ${PAULIB_EXPORT_TARGETS}
    EXPORT paulibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev
)
install(EXPORT paulibTargets
    NAMESPACE paulib::
    DESTINATION ${PAULIB_INSTALL_CMAKEDIR}
    COMPONENT dev
)

configure_package_config_file(cmake/paulibConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/paulibConfig.cmake
    INSTALL_DESTINATION ${PAULIB_INSTALL_CMAKEDIR}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/paulibConfigVersion.cmake
    COMPATIBILITY SameMinorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/paulibConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/paulibConfigVersion.cmake
    DESTINATION ${PAULIB_INSTALL_CMAKEDIR}
    COMPONENT dev
)
