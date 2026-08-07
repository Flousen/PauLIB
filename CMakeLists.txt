cmake_minimum_required(VERSION 3.16)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "g++")
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    project(paulib VERSION 0.1.0 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 90)
    set(CMAKE_CUDA_STANDARD 17)
else()
    project(paulib VERSION 0.1.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(MPI)

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
add_subdirectory(app)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()
