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
