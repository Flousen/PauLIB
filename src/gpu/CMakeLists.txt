# paulib_gpu — CUDA kernel library for GPU-accelerated PauliSum operations.
#
# Separation rule (enforced by compile-time isolation):
#   gpu_mul.cu  compiled by nvcc: kernel math + extern "C" launchers,
#               NO PauLib/Highway/ARM SVE headers.
#   GpuPauliSum.h compiled by g++: RAII HBM wrapper, uses full PauLib.
#
# Downstream targets link:  paulib  paulib_gpu
# They then #include "gpu/GpuPauliSum.h" to use the high-level API.

add_library(paulib_gpu STATIC gpu_mul.cu gpu_commutator.cu)

set_target_properties(paulib_gpu PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

target_compile_options(paulib_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

# PRIVATE: only the .cu compilation needs these paths.
# Downstream g++ targets get cuda_runtime_api.h via paulib_gpu's PUBLIC includes.
target_include_directories(paulib_gpu
    PUBLIC  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

# Propagate cudart so downstream targets don't have to list it explicitly.
target_link_libraries(paulib_gpu PUBLIC cudart)
