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
add_library(paulib::gpu ALIAS paulib_gpu)

set_target_properties(paulib_gpu PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    EXPORT_NAME gpu
)

target_compile_options(paulib_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

# PRIVATE: only the .cu compilation needs this path.
target_include_directories(paulib_gpu
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

# Propagate cudart (library + cuda_runtime_api.h include path) so downstream
# g++ targets don't have to list it explicitly.
target_link_libraries(paulib_gpu PUBLIC CUDA::cudart)
