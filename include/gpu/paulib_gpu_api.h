#pragma once
// CUDA kernel launcher declarations.
// No PauLib types. No CUDA device types.
// Safe to include from both g++ and nvcc translation units.
//
// Requires <cuda_runtime_api.h> (included below) for cudaStream_t.
// Downstream targets must have the CUDA toolkit include path available.

#include <cuda_runtime_api.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// launch_gpu_mul_soa — outer-product of two SoA-packed PauliSums.
//
//   out[i*n + j] = lhs[j] * rhs[i]   for j in [0,n), i in [0,m)
//
// array_size = ceil(N/64) for N-qubit strings. Supported: 1..16 (up to 1024 qubits).
// All pointers must be device pointers (cudaMalloc).
// Output buffers must be pre-allocated for n*m terms.
// The kernel is asynchronous; synchronise with cudaStreamSynchronize(stream)
// or cudaDeviceSynchronize() before reading output.
void launch_gpu_mul_soa(
    int             array_size,
    const uint64_t* lhs_x, const uint64_t* lhs_z,
    const uint8_t*  lhs_flags, const double* lhs_val,
    const uint64_t* rhs_x, const uint64_t* rhs_z,
    const uint8_t*  rhs_flags, const double* rhs_val,
    uint64_t*       out_x, uint64_t* out_z,
    uint8_t*        out_flags, double* out_val,
    int n, int m, cudaStream_t stream);

// launch_gpu_commutator_soa — [lhs, rhs] (multiply_by_i = 0) or i[lhs, rhs]
// (multiply_by_i = 1). Only anticommuting pairs emit a term ([P,Q] = 2PQ);
// output slots are claimed via the device counter d_count (must be zeroed
// before launch), so the term order is nondeterministic and the output is
// uncombined. Output buffers must be allocated for `capacity` terms
// (worst case n*m) and are indexed with stride `capacity` per word plane:
// out_x[k*capacity + slot]. Read the final term count from d_count after
// synchronising.
void launch_gpu_commutator_soa(
    int             array_size,
    int             multiply_by_i,
    const uint64_t* lhs_x, const uint64_t* lhs_z,
    const uint8_t*  lhs_flags, const double* lhs_val,
    const uint64_t* rhs_x, const uint64_t* rhs_z,
    const uint8_t*  rhs_flags, const double* rhs_val,
    uint64_t*       out_x, uint64_t* out_z,
    uint8_t*        out_flags, double* out_val,
    unsigned long long* d_count,
    int n, int m, long long capacity, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
