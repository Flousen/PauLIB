#pragma once
// Host-side GPU buffer management for PauliSum.
// Compiled by g++ (no device code). Requires CUDA runtime + PauLib headers.
//
// GpuPauliSum<N,T> owns HBM buffers for a PauliSum<N,T> and handles
// upload, download, and lifetime. Use gpu_multiply() for the outer product.
//
// Memory layout (flat SoA, matches gpu_mul.cu kernel expectations):
//   d_x_[k*n + j]     X-bits of word k of string j
//   d_z_[k*n + j]     Z-bits of word k of string j
//   d_flags_[j]       phase/sign byte of string j
//   d_val_[j]         scalar coefficient of string j

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "PauliSumCore.h"
#include "gpu/paulib_gpu_api.h"

#define PAULIB_GPU_CK(x) do {                                           \
    cudaError_t _e = (x);                                               \
    if (_e != cudaSuccess)                                              \
        throw std::runtime_error(std::string("CUDA: ") +               \
                                 cudaGetErrorString(_e));               \
} while(0)

namespace paulib {

template<std::size_t N, typename T = double>
class GpuPauliSum {
    static_assert(std::is_same_v<T, double>,
                  "GPU kernels currently only support T=double");
public:
    using SoA = PauliSum<N, T>;
    static constexpr int AS = (int)SoA::ArraySize;

    GpuPauliSum() = default;

    // Allocate HBM buffers for n terms (contents uninitialised).
    explicit GpuPauliSum(std::int64_t n) { alloc_(n); }

    // Allocate and immediately upload from a CPU SoA.
    explicit GpuPauliSum(const SoA& src) : num_qubits_(src.num_qubits()) {
        alloc_((std::int64_t)src.get_length());
        upload(src);
    }

    // Logical qubit count carried alongside the buffers (capacity is N).
    std::size_t num_qubits_ = N;

    ~GpuPauliSum() { free_(); }

    GpuPauliSum(const GpuPauliSum&)            = delete;
    GpuPauliSum& operator=(const GpuPauliSum&) = delete;

    GpuPauliSum(GpuPauliSum&& o) noexcept
        : num_qubits_(o.num_qubits_),
          n_(o.n_), d_x_(o.d_x_), d_z_(o.d_z_), d_flags_(o.d_flags_), d_val_(o.d_val_)
    { o.n_ = 0; o.d_x_ = o.d_z_ = nullptr; o.d_flags_ = nullptr; o.d_val_ = nullptr; }

    GpuPauliSum& operator=(GpuPauliSum&& o) noexcept {
        if (this != &o) { free_(); new(this) GpuPauliSum(std::move(o)); }
        return *this;
    }

    std::int64_t size() const { return n_; }

    // Pack the AS separate per-plane vectors of src into contiguous HBM buffers.
    // GpuPauliSum must already be allocated for src.get_length() terms.
    void upload(const SoA& src) {
        const std::int64_t n = (std::int64_t)src.get_length();
        if (n != n_)
            throw std::runtime_error("GpuPauliSum::upload: source has " +
                std::to_string(n) + " terms, buffer allocated for " +
                std::to_string(n_));
        std::vector<uint64_t> px((std::size_t)AS * n), pz((std::size_t)AS * n);
        for (int k = 0; k < AS; ++k) {
            std::memcpy(px.data() + (std::size_t)k * n,
                        src.x_data[k].data(), (std::size_t)n * sizeof(uint64_t));
            std::memcpy(pz.data() + (std::size_t)k * n,
                        src.z_data[k].data(), (std::size_t)n * sizeof(uint64_t));
        }
        PAULIB_GPU_CK(cudaMemcpy(d_x_,     px.data(),          (std::size_t)AS*n*sizeof(uint64_t), cudaMemcpyHostToDevice));
        PAULIB_GPU_CK(cudaMemcpy(d_z_,     pz.data(),          (std::size_t)AS*n*sizeof(uint64_t), cudaMemcpyHostToDevice));
        PAULIB_GPU_CK(cudaMemcpy(d_flags_, src.flags.data(),   (std::size_t)n*sizeof(uint8_t),     cudaMemcpyHostToDevice));
        PAULIB_GPU_CK(cudaMemcpy(d_val_,   src.value.data(),   (std::size_t)n*sizeof(T),           cudaMemcpyHostToDevice));
    }

    // Download HBM buffers to a new CPU SoA. Synchronous.
    SoA download() const {
        std::vector<uint64_t> px((std::size_t)AS * n_), pz((std::size_t)AS * n_);
        std::vector<uint8_t>  fl((std::size_t)n_);
        std::vector<T>        vl((std::size_t)n_);
        PAULIB_GPU_CK(cudaMemcpy(px.data(), d_x_,     (std::size_t)AS*n_*sizeof(uint64_t), cudaMemcpyDeviceToHost));
        PAULIB_GPU_CK(cudaMemcpy(pz.data(), d_z_,     (std::size_t)AS*n_*sizeof(uint64_t), cudaMemcpyDeviceToHost));
        PAULIB_GPU_CK(cudaMemcpy(fl.data(), d_flags_, (std::size_t)n_*sizeof(uint8_t),     cudaMemcpyDeviceToHost));
        PAULIB_GPU_CK(cudaMemcpy(vl.data(), d_val_,   (std::size_t)n_*sizeof(T),           cudaMemcpyDeviceToHost));
        SoA result;
        for (int k = 0; k < AS; ++k) {
            result.x_data[k].resize((std::size_t)n_);
            result.z_data[k].resize((std::size_t)n_);
            std::memcpy(result.x_data[k].data(), px.data() + (std::size_t)k * n_, (std::size_t)n_ * sizeof(uint64_t));
            std::memcpy(result.z_data[k].data(), pz.data() + (std::size_t)k * n_, (std::size_t)n_ * sizeof(uint64_t));
        }
        result.flags = std::move(fl);
        result.value = std::move(vl);
        result.num_qubits_ = num_qubits_;
        return result;
    }

    // Raw device pointers — passed directly to kernel launchers.
    uint64_t*       x_ptr()     { return d_x_; }
    uint64_t*       z_ptr()     { return d_z_; }
    uint8_t*        flags_ptr() { return d_flags_; }
    T*              val_ptr()   { return d_val_; }
    const uint64_t* x_ptr()     const { return d_x_; }
    const uint64_t* z_ptr()     const { return d_z_; }
    const uint8_t*  flags_ptr() const { return d_flags_; }
    const T*        val_ptr()   const { return d_val_; }

private:
    std::int64_t n_   = 0;
    uint64_t* d_x_    = nullptr;
    uint64_t* d_z_    = nullptr;
    uint8_t*  d_flags_= nullptr;
    T*        d_val_  = nullptr;

    void alloc_(std::int64_t n) {
        n_ = n;
        PAULIB_GPU_CK(cudaMalloc((void**)&d_x_,     (std::size_t)AS * n * sizeof(uint64_t)));
        PAULIB_GPU_CK(cudaMalloc((void**)&d_z_,     (std::size_t)AS * n * sizeof(uint64_t)));
        PAULIB_GPU_CK(cudaMalloc((void**)&d_flags_, (std::size_t)n * sizeof(uint8_t)));
        PAULIB_GPU_CK(cudaMalloc((void**)&d_val_,   (std::size_t)n * sizeof(T)));
    }

    void free_() {
        if (d_x_)     { cudaFree(d_x_);     d_x_     = nullptr; }
        if (d_z_)     { cudaFree(d_z_);     d_z_     = nullptr; }
        if (d_flags_) { cudaFree(d_flags_); d_flags_ = nullptr; }
        if (d_val_)   { cudaFree(d_val_);   d_val_   = nullptr; }
        n_ = 0;
    }
};

// gpu_multiply — compute lhs ⊗ rhs on the GPU, return result as a new GpuPauliSum.
//
// Allocates output (n*m terms), launches the kernel, and synchronises before
// returning so the result is immediately readable via download().
// Pass a non-null stream to overlap with other GPU work; the function still
// synchronises that stream before returning.
template<std::size_t N, typename T>
GpuPauliSum<N,T> gpu_multiply(const GpuPauliSum<N,T>& lhs,
                               const GpuPauliSum<N,T>& rhs,
                               cudaStream_t stream = nullptr)
{
    // The kernel indexes each input with a 32-bit index (output uses 64-bit).
    if (lhs.size() > INT32_MAX || rhs.size() > INT32_MAX)
        throw std::runtime_error("gpu_multiply: input exceeds 2^31-1 terms");

    GpuPauliSum<N,T> out(lhs.size() * rhs.size());
    out.num_qubits_ = std::max(lhs.num_qubits_, rhs.num_qubits_);
    if (out.size() == 0) return out;

    launch_gpu_mul_soa(GpuPauliSum<N,T>::AS,
                       lhs.x_ptr(), lhs.z_ptr(), lhs.flags_ptr(), lhs.val_ptr(),
                       rhs.x_ptr(), rhs.z_ptr(), rhs.flags_ptr(), rhs.val_ptr(),
                       out.x_ptr(), out.z_ptr(), out.flags_ptr(), out.val_ptr(),
                       (int)lhs.size(), (int)rhs.size(), stream);
    PAULIB_GPU_CK(cudaStreamSynchronize(stream));
    return out;
}

// gpu_commutator — compute [lhs, rhs] (or i[lhs, rhs]) on the GPU.
//
// Allocates worst-case scratch (n*m terms), launches the kernel, reads back
// the emitted term count, and compacts into an exactly-sized GpuPauliSum via
// device-to-device copies. The result is UNCOMBINED and in nondeterministic
// order; canonicalize()/sort_and_combine() after download() restores
// canonical form. Synchronous.
template<std::size_t N, typename T>
GpuPauliSum<N,T> gpu_commutator(const GpuPauliSum<N,T>& lhs,
                                const GpuPauliSum<N,T>& rhs,
                                bool multiply_by_i = false,
                                cudaStream_t stream = nullptr)
{
    if (lhs.size() > INT32_MAX || rhs.size() > INT32_MAX)
        throw std::runtime_error("gpu_commutator: input exceeds 2^31-1 terms");
    constexpr int AS = GpuPauliSum<N,T>::AS;

    const std::int64_t capacity = lhs.size() * rhs.size();
    GpuPauliSum<N,T> scratch(capacity);
    scratch.num_qubits_ = std::max(lhs.num_qubits_, rhs.num_qubits_);
    if (capacity == 0) return scratch;

    unsigned long long* d_count = nullptr;
    PAULIB_GPU_CK(cudaMalloc((void**)&d_count, sizeof(unsigned long long)));
    PAULIB_GPU_CK(cudaMemsetAsync(d_count, 0, sizeof(unsigned long long), stream));

    launch_gpu_commutator_soa(AS, multiply_by_i ? 1 : 0,
                              lhs.x_ptr(), lhs.z_ptr(), lhs.flags_ptr(), lhs.val_ptr(),
                              rhs.x_ptr(), rhs.z_ptr(), rhs.flags_ptr(), rhs.val_ptr(),
                              scratch.x_ptr(), scratch.z_ptr(),
                              scratch.flags_ptr(), scratch.val_ptr(),
                              d_count,
                              (int)lhs.size(), (int)rhs.size(),
                              (long long)capacity, stream);

    unsigned long long count = 0;
    PAULIB_GPU_CK(cudaStreamSynchronize(stream));
    PAULIB_GPU_CK(cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost));
    PAULIB_GPU_CK(cudaFree(d_count));

    // Compact the stride-`capacity` scratch planes into an exact-size result.
    GpuPauliSum<N,T> out((std::int64_t)count);
    out.num_qubits_ = scratch.num_qubits_;
    for (int k = 0; k < AS; ++k) {
        PAULIB_GPU_CK(cudaMemcpy(out.x_ptr() + (std::size_t)k * count,
                                 scratch.x_ptr() + (std::size_t)k * capacity,
                                 (std::size_t)count * sizeof(uint64_t),
                                 cudaMemcpyDeviceToDevice));
        PAULIB_GPU_CK(cudaMemcpy(out.z_ptr() + (std::size_t)k * count,
                                 scratch.z_ptr() + (std::size_t)k * capacity,
                                 (std::size_t)count * sizeof(uint64_t),
                                 cudaMemcpyDeviceToDevice));
    }
    PAULIB_GPU_CK(cudaMemcpy(out.flags_ptr(), scratch.flags_ptr(),
                             (std::size_t)count * sizeof(uint8_t),
                             cudaMemcpyDeviceToDevice));
    PAULIB_GPU_CK(cudaMemcpy(out.val_ptr(), scratch.val_ptr(),
                             (std::size_t)count * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    return out;
}

} // namespace paulib
