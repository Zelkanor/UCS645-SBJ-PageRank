// GPU back-end for the hybrid solver. Linked in only when nvcc is available.
// We keep the graph resident on the device across iterations -> only the rank
// vector + the active-vertex list move per step.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_OK(call)                                              \
    do                                                             \
    {                                                              \
        cudaError_t _e = (call);                                   \
        if (_e != cudaSuccess)                                     \
        {                                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", \
                         cudaGetErrorName(_e), __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                  \
            std::exit(1);                                          \
        }                                                          \
    } while (0)

namespace
{

    struct GpuCtx
    {
        uint32_t n = 0;
        uint64_t m = 0;
        uint64_t *row = nullptr;
        uint32_t *col = nullptr;
        uint32_t *od = nullptr;
        float *r = nullptr;
        float *rn = nullptr;
        float *contrib = nullptr;
        int *vlist = nullptr; // active vertices for this step
        float *delta = nullptr;
        float *dangle = nullptr;
        int vlist_cap = 0;

        // --- NEW ASYNC & PINNED MEMORY VARIABLES ---
        float *rn_host_pinned = nullptr;    // Page-locked host memory for fast downloads
        float *delta_host_pinned = nullptr; // Page-locked host memory for delta
        cudaStream_t stream;                // Stream for overlapping CPU/GPU execution
    } g_ctx;

    __global__ void k_contrib(const float *r, const uint32_t *od, float *contrib,
                              float *dangle_out, uint32_t n)
    {
        uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
        if (u >= n)
            return;
        float ru = r[u];
        uint32_t k = od[u];
        contrib[u] = (k == 0) ? 0.0f : (ru / (float)k);

        extern __shared__ float sh[];
        sh[threadIdx.x] = (k == 0) ? ru : 0.0f;
        __syncthreads();
        for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
                sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            atomicAdd(dangle_out, sh[0]);
    }

    __global__ void k_sweep(const uint64_t *row, const uint32_t *col,
                            const float *contrib, const float *r_in,
                            const int *vlist, int vcount,
                            float *r_out, float *delta_out,
                            float damping, float base)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= vcount)
            return;
        int v = vlist[idx];

        uint64_t s = row[v], e = row[v + 1];
        float sum = 0.0f;
        for (uint64_t k = s; k < e; ++k)
            sum += contrib[col[k]];
        float nv = base + damping * sum;
        r_out[v] = nv;

        extern __shared__ float sh[];
        sh[threadIdx.x] = fabsf(nv - r_in[v]);
        __syncthreads();
        for (unsigned st = blockDim.x / 2; st > 0; st >>= 1)
        {
            if (threadIdx.x < st)
                sh[threadIdx.x] += sh[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            atomicAdd(delta_out, sh[0]);
    }

} // namespace

extern "C" void sbj_gpu_init(uint32_t n, uint64_t m,
                             const uint64_t *row, const uint32_t *col,
                             const uint32_t *od)
{
    g_ctx.n = n;
    g_ctx.m = m;

    // Create Stream
    CUDA_OK(cudaStreamCreate(&g_ctx.stream));

    // Device allocations
    CUDA_OK(cudaMalloc(&g_ctx.row, (n + 1) * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&g_ctx.col, m * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&g_ctx.od, n * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&g_ctx.r, n * sizeof(float)));
    CUDA_OK(cudaMalloc(&g_ctx.rn, n * sizeof(float)));
    CUDA_OK(cudaMalloc(&g_ctx.contrib, n * sizeof(float)));
    CUDA_OK(cudaMalloc(&g_ctx.delta, sizeof(float)));
    CUDA_OK(cudaMalloc(&g_ctx.dangle, sizeof(float)));

    // Host allocations (Pinned Memory for fast PCIe transfers)
    CUDA_OK(cudaMallocHost(&g_ctx.rn_host_pinned, n * sizeof(float)));
    CUDA_OK(cudaMallocHost(&g_ctx.delta_host_pinned, sizeof(float)));

    // Initial graph transfers
    CUDA_OK(cudaMemcpy(g_ctx.row, row, (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_ctx.col, col, m * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(g_ctx.od, od, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

// 1. ASYNC UPLOAD
extern "C" void sbj_gpu_upload_rank_async(const float *r, uint32_t n)
{
    CUDA_OK(cudaMemcpyAsync(g_ctx.r, r, n * sizeof(float), cudaMemcpyHostToDevice, g_ctx.stream));
}

// 2. ASYNC KERNEL EXECUTION & DOWNLOAD QUEUE
extern "C" void sbj_gpu_step_async(const int *gpu_vertices, int gpu_count,
                                   float damping, float base)
{
    if (gpu_count <= 0)
        return;

    if (gpu_count > g_ctx.vlist_cap)
    {
        if (g_ctx.vlist)
            cudaFree(g_ctx.vlist);
        CUDA_OK(cudaMalloc(&g_ctx.vlist, gpu_count * sizeof(int)));
        g_ctx.vlist_cap = gpu_count;
    }

    // Copy active vertices asynchronously
    CUDA_OK(cudaMemcpyAsync(g_ctx.vlist, gpu_vertices, gpu_count * sizeof(int),
                            cudaMemcpyHostToDevice, g_ctx.stream));

    const int TPB = 256;
    const size_t shmem = TPB * sizeof(float);

    // Reset dangling and compute contribs
    CUDA_OK(cudaMemsetAsync(g_ctx.dangle, 0, sizeof(float), g_ctx.stream));
    int blocks_n = (int)((g_ctx.n + TPB - 1) / TPB);
    k_contrib<<<blocks_n, TPB, shmem, g_ctx.stream>>>(g_ctx.r, g_ctx.od, g_ctx.contrib,
                                                      g_ctx.dangle, g_ctx.n);

    // Note: The original code read back dangling here but completely ignored it
    // because `base` already included the CPU's dangling calculation.
    // We skip the sync and read-back entirely!

    // Reset delta and compute sweep
    CUDA_OK(cudaMemsetAsync(g_ctx.delta, 0, sizeof(float), g_ctx.stream));
    int blocks_v = (gpu_count + TPB - 1) / TPB;
    k_sweep<<<blocks_v, TPB, shmem, g_ctx.stream>>>(g_ctx.row, g_ctx.col, g_ctx.contrib,
                                                    g_ctx.r, g_ctx.vlist, gpu_count,
                                                    g_ctx.rn, g_ctx.delta, damping, base);

    // Queue the asynchronous download of the updated ranks to PINNED host memory
    CUDA_OK(cudaMemcpyAsync(g_ctx.rn_host_pinned, g_ctx.rn, g_ctx.n * sizeof(float),
                            cudaMemcpyDeviceToHost, g_ctx.stream));

    // Queue the asynchronous download of the convergence delta
    CUDA_OK(cudaMemcpyAsync(g_ctx.delta_host_pinned, g_ctx.delta, sizeof(float),
                            cudaMemcpyDeviceToHost, g_ctx.stream));
}

// 3. SYNCHRONIZE & GATHER
extern "C" void sbj_gpu_sync_and_gather(const int *gpu_vertices, int gpu_count,
                                        float *r_out_host, float *delta_out)
{
    if (gpu_count <= 0)
    {
        *delta_out = 0.0f;
        return;
    }

    // Force the CPU to wait ONLY NOW, right before we need the data
    CUDA_OK(cudaStreamSynchronize(g_ctx.stream));

    *delta_out = *g_ctx.delta_host_pinned;

    // Scatter results from the pinned buffer to the final r_out array
    for (int i = 0; i < gpu_count; ++i)
    {
        int v = gpu_vertices[i];
        r_out_host[v] = g_ctx.rn_host_pinned[v];
    }
}

extern "C" void sbj_gpu_destroy()
{
    // Wait for everything to finish before destroying
    if (g_ctx.stream)
    {
        cudaStreamSynchronize(g_ctx.stream);
        cudaStreamDestroy(g_ctx.stream);
    }

    // Free Device memory
    cudaFree(g_ctx.row);
    cudaFree(g_ctx.col);
    cudaFree(g_ctx.od);
    cudaFree(g_ctx.r);
    cudaFree(g_ctx.rn);
    cudaFree(g_ctx.contrib);
    cudaFree(g_ctx.delta);
    cudaFree(g_ctx.dangle);
    if (g_ctx.vlist)
        cudaFree(g_ctx.vlist);

    // Free Pinned Host Memory
    if (g_ctx.rn_host_pinned)
        cudaFreeHost(g_ctx.rn_host_pinned);
    if (g_ctx.delta_host_pinned)
        cudaFreeHost(g_ctx.delta_host_pinned);

    g_ctx = {};
}