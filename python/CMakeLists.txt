cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Find or fetch pybind11
# ---------------------------------------------------------------------------
find_package(pybind11 QUIET)

if (NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ---------------------------------------------------------------------------
# Build the extension module
# ---------------------------------------------------------------------------
pybind11_add_module(_paulib bindings.cpp)

target_link_libraries(_paulib PRIVATE
    paulib          # header-only INTERFACE target (includes + OpenMP)
)

target_compile_options(_paulib PRIVATE
    ${PAULIB_PERF_FLAGS}
    -Wno-unused-parameter
)

# Place the .so directly inside the Python package so it is importable
# without any installation step: `python -c "import paulib"` works from
# the python/ directory after building.
set_target_properties(_paulib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/paulib"
    CXX_SCAN_FOR_MODULES OFF   # no C++20 modules; avoids needing clang-scan-deps
)
