// =============================================================================
// gpu_commutator.cu — CUDA SoA commutator kernel: out = [lhs, rhs] (or i[lhs, rhs]).
// Target: sm_90 (H100 / GH200). No PauLib/Highway/ARM headers.
//
// One thread per pair (j, i) → n*m threads total, exactly like gpu_mul.cu.
// The difference: commuting pairs produce nothing ([P,Q] = 0), so each thread
// first evaluates the symplectic parity in registers and only anticommuting
// pairs append an output term:  [P_j, Q_i] = 2 P_j Q_i,  coefficient
// 2 * val_j * val_i, phase carried in the flags byte.  multiply_by_i shifts
// the phase exponent by one, giving i[lhs, rhs] (Hermitian-preserving form
// used by the operator-dynamics drivers).
//
// Output slots are claimed with a global atomic counter, so the term ORDER is
// nondeterministic; consumers canonicalize (sort_and_combine) afterwards,
// which restores a canonical order. Output buffers are allocated for the
// worst case n*m and indexed with stride `capacity` per word plane:
//   out_x[k*capacity + slot].
// The final term count is returned through d_count.
// =============================================================================

#include "gpu/paulib_gpu_api.h"
#include <cstdint>
#include <cstdio>

template<int AS>
__global__ void __launch_bounds__(256)
commutator_soa_kernel(
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
    unsigned long long* __restrict__ d_count,
    int n, int m, int64_t capacity, int multiply_by_i)
{
    int64_t tid   = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)n * m;
    if (tid >= total) return;

    int j = (int)(tid % n);   // LHS index — consecutive across warp
    int i = (int)(tid / n);   // RHS index — broadcast within warp

    // Single pass: product words buffered in registers (AS is compile-time),
    // symplectic parity and phase accumulators alongside.
    uint64_t px[AS], pz[AS];
    uint64_t par = 0, c1 = 0, c2 = 0;

    #pragma unroll
    for (int k = 0; k < AS; ++k) {
        uint64_t ax = lhs_x[(int64_t)k * n + j];
        uint64_t az = lhs_z[(int64_t)k * n + j];
        uint64_t bx = rhs_x[(int64_t)k * m + i];
        uint64_t bz = rhs_z[(int64_t)k * m + i];

        px[k] = ax ^ bx;
        pz[k] = az ^ bz;

        uint64_t xbz  = ax & bz;
        uint64_t anti = (bx & az) ^ xbz;
        par ^= anti;
        c2  ^= (c1 ^ ax ^ az ^ bx ^ bz ^ xbz) & anti;
        c1  ^= anti;
    }

    // Commuting pair: [P, Q] = 0, no output.
    if ((__popcll(par) & 1) == 0) return;

    const unsigned long long slot = atomicAdd(d_count, 1ull);

    #pragma unroll
    for (int k = 0; k < AS; ++k) {
        out_x[(int64_t)k * capacity + (int64_t)slot] = px[k];
        out_z[(int64_t)k * capacity + (int64_t)slot] = pz[k];
    }
    out_flags[slot] = (uint8_t)((lhs_flags[j] + rhs_flags[i]
        + (unsigned)__popcll(c1)
        + 2u * (unsigned)(__popcll(c2) & 1u)
        + (unsigned)multiply_by_i) & 0x03u);
    out_val[slot] = 2.0 * lhs_val[j] * rhs_val[i];
}

extern "C" void launch_gpu_commutator_soa(
    int             array_size,
    int             multiply_by_i,
    const uint64_t* lhs_x, const uint64_t* lhs_z,
    const uint8_t*  lhs_flags, const double* lhs_val,
    const uint64_t* rhs_x, const uint64_t* rhs_z,
    const uint8_t*  rhs_flags, const double* rhs_val,
    uint64_t*       out_x, uint64_t* out_z,
    uint8_t*        out_flags, double* out_val,
    unsigned long long* d_count,
    int n, int m, long long capacity, cudaStream_t stream)
{
#define DISPATCH(AS) case AS: {                                              \
    int64_t tot = (int64_t)n * m;                                            \
    int     blk = (int)((tot + 255) / 256);                                  \
    commutator_soa_kernel<AS><<<blk, 256, 0, stream>>>(                      \
        lhs_x, lhs_z, lhs_flags, lhs_val,                                   \
        rhs_x, rhs_z, rhs_flags, rhs_val,                                   \
        out_x, out_z, out_flags, out_val,                                    \
        d_count, n, m, (int64_t)capacity, multiply_by_i);                    \
    break; }

    switch (array_size) {
        DISPATCH( 1) DISPATCH( 2) DISPATCH( 3) DISPATCH( 4)
        DISPATCH( 5) DISPATCH( 6) DISPATCH( 7) DISPATCH( 8)
        DISPATCH( 9) DISPATCH(10) DISPATCH(11) DISPATCH(12)
        DISPATCH(13) DISPATCH(14) DISPATCH(15) DISPATCH(16)
        default:
            fprintf(stderr, "launch_gpu_commutator_soa: unsupported array_size=%d "
                            "(supported: 1-16, up to 1024 qubits)\n", array_size);
            __builtin_trap();
    }
#undef DISPATCH
}
