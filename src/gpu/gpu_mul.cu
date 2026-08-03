// =============================================================================
// gpu_mul.cu — CUDA SoA outer-product kernel for PauliSum multiplication.
// Target: sm_90 (H100 / GH200). No PauLib/Highway/ARM headers.
//
// Computes out = lhs ⊗ rhs where lhs has n terms and rhs has m terms.
// One thread per output pair (j, i) → n*m threads total.
//
// See include/gpu/paulib_gpu_api.h for the host-callable entry point.
// See include/gpu/GpuPauliSum.h   for the RAII buffer wrapper.
// =============================================================================
//
// PAULI ENCODING
// --------------
// A PauliString<N> stores N qubits in AS = ceil(N/64) 64-bit words, two
// bit-planes kept separately:
//   x_data[k*n + j]  — X-bits of qubit slice k*64..(k+1)*64-1 of string j
//   z_data[k*n + j]  — Z-bits of the same slice
// Single-qubit encoding: I=(0,0) X=(1,0) Z=(0,1) Y=(1,1)
//
// PHASE ACCUMULATION (symplectic, 64 qubits at once)
// ---------------------------------------------------
// Two 64-bit accumulators c1, c2 track the phase contribution per word:
//
//   xbz  = ax & bz               qubits where lhs=X, rhs=Z
//   anti = (bx & az) ^ xbz       anticommuting qubit-slots
//   c2  ^= (c1 ^ ax ^ az ^ bx ^ bz ^ xbz) & anti
//   c1  ^= anti
//
// Final phase exponent: popcount(c1) + 2*parity(c2)   [mod 4, units of i]
//
// COALESCED ACCESS
// ----------------
// Thread tid → j = tid % n (LHS index, varies fast), i = tid / n (RHS index).
// Consecutive threads share i and have consecutive j:
//   lhs_x[k*n + j]       coalesced load  (32 consecutive j = 256 bytes)
//   rhs_x[k*m + i]       L1 broadcast    (all 32 threads same i)
//   out_x[k*(n*m) + tid]  coalesced store
// =============================================================================

#include "gpu/paulib_gpu_api.h"
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// multiply_soa_kernel<AS>
//
// AS  — compile-time ArraySize = ceil(N/64); the k-loop is fully unrolled,
//       keeping ax/az/bx/bz/c1/c2 in registers throughout.
// ---------------------------------------------------------------------------
template<int AS>
__global__ void __launch_bounds__(256)
multiply_soa_kernel(
    const uint64_t* __restrict__ lhs_x,
    const uint64_t* __restrict__ lhs_z,
    const uint8_t*  __restrict__ lhs_flags,
    const double*   __restrict__ lhs_val,
    const uint64_t* __restrict__ rhs_x,
    const uint64_t* __restrict__ rhs_z,
    const uint8_t*  __restrict__ rhs_flags,
    const double*   __restrict__ rhs_val,
    uint64_t* __restrict__ out_x,
    uint64_t* __restrict__ out_z,
    uint8_t*  __restrict__ out_flags,
    double*   __restrict__ out_val,
    int n, int m)
{
    int64_t tid   = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)n * m;
    if (tid >= total) return;

    int j = (int)(tid % n);   // LHS index — consecutive across warp
    int i = (int)(tid / n);   // RHS index — broadcast within warp

    uint64_t c1 = 0, c2 = 0;

    #pragma unroll
    for (int k = 0; k < AS; ++k) {
        uint64_t ax = lhs_x[(int64_t)k * n + j];
        uint64_t az = lhs_z[(int64_t)k * n + j];
        uint64_t bx = rhs_x[(int64_t)k * m + i];
        uint64_t bz = rhs_z[(int64_t)k * m + i];

        out_x[(int64_t)k * total + tid] = ax ^ bx;
        out_z[(int64_t)k * total + tid] = az ^ bz;

        uint64_t xbz  = ax & bz;
        uint64_t anti = (bx & az) ^ xbz;
        c2 ^= (c1 ^ ax ^ az ^ bx ^ bz ^ xbz) & anti;
        c1 ^= anti;
    }

    out_flags[tid] = (uint8_t)((lhs_flags[j] + rhs_flags[i]
        + (unsigned)__popcll(c1)
        + 2u * (unsigned)(__popcll(c2) & 1u)) & 0x03u);

    out_val[tid] = lhs_val[j] * rhs_val[i];
}

// ---------------------------------------------------------------------------
// launch_gpu_mul_soa — runtime dispatch by array_size.
// Instantiates the kernel for AS = 1..16 (covers up to 1024 qubits).
// ---------------------------------------------------------------------------
extern "C" void launch_gpu_mul_soa(
    int             array_size,
    const uint64_t* lhs_x, const uint64_t* lhs_z,
    const uint8_t*  lhs_flags, const double* lhs_val,
    const uint64_t* rhs_x, const uint64_t* rhs_z,
    const uint8_t*  rhs_flags, const double* rhs_val,
    uint64_t*       out_x, uint64_t* out_z,
    uint8_t*        out_flags, double* out_val,
    int n, int m, cudaStream_t stream)
{
#define DISPATCH(AS) case AS: {                                              \
    int64_t tot = (int64_t)n * m;                                            \
    int     blk = (int)((tot + 255) / 256);                                  \
    multiply_soa_kernel<AS><<<blk, 256, 0, stream>>>(                        \
        lhs_x, lhs_z, lhs_flags, lhs_val,                                   \
        rhs_x, rhs_z, rhs_flags, rhs_val,                                   \
        out_x, out_z, out_flags, out_val, n, m);                             \
    break; }

    switch (array_size) {
        DISPATCH( 1) DISPATCH( 2) DISPATCH( 3) DISPATCH( 4)
        DISPATCH( 5) DISPATCH( 6) DISPATCH( 7) DISPATCH( 8)
        DISPATCH( 9) DISPATCH(10) DISPATCH(11) DISPATCH(12)
        DISPATCH(13) DISPATCH(14) DISPATCH(15) DISPATCH(16)
        default:
            fprintf(stderr, "launch_gpu_mul_soa: unsupported array_size=%d "
                            "(supported: 1-16, up to 1024 qubits)\n", array_size);
            __builtin_trap();
    }
#undef DISPATCH
}
