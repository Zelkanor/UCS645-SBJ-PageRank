#include "pagerank.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_OK(call) do {                                                  \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",              \
                     cudaGetErrorName(_e), __FILE__, __LINE__,              \
                     cudaGetErrorString(_e));                               \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

namespace sbj {

// One thread per vertex. CSR is the *transpose* graph (in-neighbours of v),
// so each thread reads a contiguous run [row_ptr[v], row_ptr[v+1]) -> the
// loads of contrib[] are scattered (graph-dependent) but the CSR pointers
// themselves are coalesced. Skip mask=0 vertices for stochastic block-Jacobi.
__global__ void pr_kernel(const uint64_t* __restrict__ row_ptr,
                          const uint32_t* __restrict__ col_idx,
                          const float*    __restrict__ contrib,
                          const float*    __restrict__ r_in,
                          const uint8_t*  __restrict__ active_mask,
                          float*          __restrict__ r_out,
                          float*          __restrict__ delta_partial,
                          uint32_t n, float damping, float base) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    if (active_mask && !active_mask[v]) {
        r_out[v] = r_in[v];                 // unsampled this round
        return;
    }

    uint64_t s = row_ptr[v];
    uint64_t e = row_ptr[v + 1];

    float sum = 0.0f;
    for (uint64_t k = s; k < e; ++k) sum += contrib[col_idx[k]];

    float nv = base + damping * sum;
    r_out[v] = nv;

    // block-wide reduction of |delta| using shared memory
    extern __shared__ float sdata[];
    float dv = fabsf(nv - r_in[v]);
    sdata[threadIdx.x] = dv;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(delta_partial, sdata[0]);
}

__global__ void contrib_kernel(const float*    __restrict__ r,
                               const uint32_t* __restrict__ out_deg,
                               float*          __restrict__ contrib,
                               float*          __restrict__ dangling_partial,
                               uint32_t n) {
    uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    float ru = r[u];
    uint32_t od = out_deg[u];
    float c = (od == 0) ? 0.0f : (ru / (float)od);
    contrib[u] = c;

    extern __shared__ float sdata[];
    sdata[threadIdx.x] = (od == 0) ? ru : 0.0f;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(dangling_partial, sdata[0]);
}

PRResult pagerank_cuda(const CSRGraph& g, const PRConfig& cfg) {
    PRResult out;
    const uint32_t n = g.n;
    if (n == 0) return out;

    const float d   = cfg.damping;
    const float inv = 1.0f / (float)n;
    const float teleport = (1.0f - d) * inv;

    // device buffers
    uint64_t *d_row = nullptr;
    uint32_t *d_col = nullptr, *d_out = nullptr;
    float    *d_r1 = nullptr, *d_r2 = nullptr, *d_contrib = nullptr;
    float    *d_delta = nullptr, *d_dangle = nullptr;

    CUDA_OK(cudaMalloc(&d_row,     (n + 1) * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_col,     g.m     * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_out,     n       * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_r1,      n       * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_r2,      n       * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_contrib, n       * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_delta,             sizeof(float)));
    CUDA_OK(cudaMalloc(&d_dangle,            sizeof(float)));

    CUDA_OK(cudaMemcpy(d_row, g.row_ptr.data(), (n+1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_col, g.col_idx.data(),  g.m  * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_out, g.out_deg.data(),  n    * sizeof(uint32_t), cudaMemcpyHostToDevice));

    std::vector<float> r0(n, inv);
    CUDA_OK(cudaMemcpy(d_r1, r0.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int TPB = 256;
    const int blocks = (int)((n + TPB - 1) / TPB);
    const size_t shmem = TPB * sizeof(float);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    int it = 0;
    float delta = 0.0f;
    float* d_in  = d_r1;
    float* d_out_p = d_r2;

    for (; it < cfg.max_iter; ++it) {
        float zero = 0.0f;
        CUDA_OK(cudaMemcpy(d_dangle, &zero, sizeof(float), cudaMemcpyHostToDevice));
        contrib_kernel<<<blocks, TPB, shmem>>>(d_in, d_out, d_contrib, d_dangle, n);

        float dangling = 0.0f;
        CUDA_OK(cudaMemcpy(&dangling, d_dangle, sizeof(float), cudaMemcpyDeviceToHost));
        const float base = teleport + d * dangling * inv;

        CUDA_OK(cudaMemcpy(d_delta, &zero, sizeof(float), cudaMemcpyHostToDevice));
        pr_kernel<<<blocks, TPB, shmem>>>(d_row, d_col, d_contrib, d_in, nullptr,
                                          d_out_p, d_delta, n, d, base);

        CUDA_OK(cudaMemcpy(&delta, d_delta, sizeof(float), cudaMemcpyDeviceToHost));
        std::swap(d_in, d_out_p);

        if (delta < cfg.tol) { ++it; break; }
    }

    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms = 0.0f; cudaEventElapsedTime(&ms, ev0, ev1);

    out.rank.resize(n);
    CUDA_OK(cudaMemcpy(out.rank.data(), d_in, n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_row); cudaFree(d_col); cudaFree(d_out);
    cudaFree(d_r1);  cudaFree(d_r2);  cudaFree(d_contrib);
    cudaFree(d_delta); cudaFree(d_dangle);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    out.iterations  = it;
    out.seconds     = ms / 1000.0;
    out.final_delta = delta;
    out.converged   = delta < cfg.tol;
    return out;
}

} // namespace sbj
