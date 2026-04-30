// GPU back-end for the hybrid solver. Linked in only when nvcc is available.
// We keep the graph resident on the device across iterations -> only the rank
// vector + the active-vertex list move per step.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

namespace {

struct GpuCtx {
    uint32_t  n = 0;
    uint64_t  m = 0;
    uint64_t* row = nullptr;
    uint32_t* col = nullptr;
    uint32_t* od  = nullptr;
    float*    r   = nullptr;
    float*    rn  = nullptr;
    float*    contrib = nullptr;
    int*      vlist = nullptr;     // active vertices for this step
    float*    delta = nullptr;
    float*    dangle = nullptr;
    int       vlist_cap = 0;
} g_ctx;

__global__ void k_contrib(const float* r, const uint32_t* od, float* contrib,
                          float* dangle_out, uint32_t n) {
    uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    float ru = r[u];
    uint32_t k = od[u];
    contrib[u] = (k == 0) ? 0.0f : (ru / (float)k);

    extern __shared__ float sh[];
    sh[threadIdx.x] = (k == 0) ? ru : 0.0f;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(dangle_out, sh[0]);
}

__global__ void k_sweep(const uint64_t* row, const uint32_t* col,
                        const float* contrib, const float* r_in,
                        const int* vlist, int vcount,
                        float* r_out, float* delta_out,
                        float damping, float base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vcount) return;
    int v = vlist[idx];

    uint64_t s = row[v], e = row[v + 1];
    float sum = 0.0f;
    for (uint64_t k = s; k < e; ++k) sum += contrib[col[k]];
    float nv = base + damping * sum;
    r_out[v] = nv;

    extern __shared__ float sh[];
    sh[threadIdx.x] = fabsf(nv - r_in[v]);
    __syncthreads();
    for (unsigned st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(delta_out, sh[0]);
}

} // namespace

extern "C" void sbj_gpu_init(uint32_t n, uint64_t m,
                             const uint64_t* row, const uint32_t* col,
                             const uint32_t* od) {
    g_ctx.n = n; g_ctx.m = m;
    CUDA_OK(cudaMalloc(&g_ctx.row, (n + 1) * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&g_ctx.col, m * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&g_ctx.od,  n * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&g_ctx.r,   n * sizeof(float)));
    CUDA_OK(cudaMalloc(&g_ctx.rn,  n * sizeof(float)));
    CUDA_OK(cudaMalloc(&g_ctx.contrib, n * sizeof(float)));
    CUDA_OK(cudaMalloc(&g_ctx.delta,  sizeof(float)));
    CUDA_OK(cudaMalloc(&g_ctx.dangle, sizeof(float)));
    CUDA_OK(cudaMemcpy(g_ctx.row, row, (n+1)*sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_ctx.col, col,  m   *sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_ctx.od,  od,   n   *sizeof(uint32_t), cudaMemcpyHostToDevice));
}

extern "C" void sbj_gpu_upload_rank(const float* r, uint32_t n) {
    CUDA_OK(cudaMemcpy(g_ctx.r, r, n * sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void sbj_gpu_step(const int* gpu_vertices, int gpu_count,
                             float damping, float base,
                             float* r_in_host, float* r_out_host,
                             float* delta_out) {
    if (gpu_count <= 0) { *delta_out = 0.0f; return; }

    if (gpu_count > g_ctx.vlist_cap) {
        if (g_ctx.vlist) cudaFree(g_ctx.vlist);
        CUDA_OK(cudaMalloc(&g_ctx.vlist, gpu_count * sizeof(int)));
        g_ctx.vlist_cap = gpu_count;
    }
    CUDA_OK(cudaMemcpy(g_ctx.vlist, gpu_vertices, gpu_count * sizeof(int),
                       cudaMemcpyHostToDevice));

    const int TPB = 256;
    const size_t shmem = TPB * sizeof(float);

    // contrib + dangling
    float zero = 0.0f;
    CUDA_OK(cudaMemcpy(g_ctx.dangle, &zero, sizeof(float), cudaMemcpyHostToDevice));
    int blocks_n = (int)((g_ctx.n + TPB - 1) / TPB);
    k_contrib<<<blocks_n, TPB, shmem>>>(g_ctx.r, g_ctx.od, g_ctx.contrib,
                                        g_ctx.dangle, g_ctx.n);

    float dangling = 0.0f;
    CUDA_OK(cudaMemcpy(&dangling, g_ctx.dangle, sizeof(float), cudaMemcpyDeviceToHost));
    float adj_base = base;  // CPU side already folded teleport+dangle for its base;
                            // we recompute here so kernel uses consistent value.
    (void)dangling;         // base already includes dangling mass from caller.

    // sweep
    CUDA_OK(cudaMemcpy(g_ctx.delta, &zero, sizeof(float), cudaMemcpyHostToDevice));
    int blocks_v = (gpu_count + TPB - 1) / TPB;
    k_sweep<<<blocks_v, TPB, shmem>>>(g_ctx.row, g_ctx.col, g_ctx.contrib,
                                      g_ctx.r, g_ctx.vlist, gpu_count,
                                      g_ctx.rn, g_ctx.delta, damping, adj_base);

    CUDA_OK(cudaMemcpy(delta_out, g_ctx.delta, sizeof(float), cudaMemcpyDeviceToHost));

    // copy back only the vertices we actually updated. We compact them on the
    // device-side rn buffer first by re-reading rn[v] for v in vlist via a
    // single bulk transfer + host-side scatter. This is fine in practice
    // because gpu_count <= n and the bandwidth is dominated by the kernel.
    float* rn_host = (float*)std::malloc((size_t)g_ctx.n * sizeof(float));
    CUDA_OK(cudaMemcpy(rn_host, g_ctx.rn, g_ctx.n * sizeof(float),
                       cudaMemcpyDeviceToHost));
    for (int i = 0; i < gpu_count; ++i) {
        int v = gpu_vertices[i];
        r_out_host[v] = rn_host[v];
    }
    std::free(rn_host);
    (void)r_in_host;
}

extern "C" void sbj_gpu_destroy() {
    cudaFree(g_ctx.row);  cudaFree(g_ctx.col); cudaFree(g_ctx.od);
    cudaFree(g_ctx.r);    cudaFree(g_ctx.rn);  cudaFree(g_ctx.contrib);
    cudaFree(g_ctx.delta); cudaFree(g_ctx.dangle);
    if (g_ctx.vlist) cudaFree(g_ctx.vlist);
    g_ctx = {};
}
