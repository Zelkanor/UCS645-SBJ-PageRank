#include "pagerank.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sbj {

// ---------------------------------------------------------------------------
// Hybrid CPU-GPU PageRank.
//
// Strategy:
//   1. classify vertices by in-degree. Top hd_percentile% go to "CPU set",
//      the rest go to "GPU set". Power-law graphs => few but very heavy CPU
//      nodes, which would otherwise stall a GPU warp.
//   2. each iteration: GPU sweeps its set, CPU sweeps its set in parallel,
//      then we merge into the new rank vector and check convergence.
//   3. stochastic block-Jacobi: optionally update only a sampled fraction of
//      blocks per iter; remaining vertices keep their old value.
//
// When the binary is built without CUDA (no nvcc), the "GPU side" is emulated
// with a separate OpenMP team on the same CPU. Logic is identical, only the
// device differs -- handy for laptops / CI / grading machines without a GPU.
// ---------------------------------------------------------------------------

#ifdef SBJ_WITH_CUDA
extern "C" {
    // small C-ABI shim implemented in pagerank_hybrid_cuda.cu (compiled by nvcc)
    void sbj_gpu_init   (uint32_t n, uint64_t m,
                         const uint64_t* row_ptr, const uint32_t* col_idx,
                         const uint32_t* out_deg);
    void sbj_gpu_upload_rank(const float* r, uint32_t n);
    void sbj_gpu_step   (const int* gpu_vertices, int gpu_count,
                         float damping, float base,
                         float* r_in, float* r_out, float* delta_out);
    void sbj_gpu_destroy();
}
#endif

namespace {

struct Partition {
    std::vector<int> cpu_v;     // heavy / high-degree
    std::vector<int> gpu_v;     // remainder
};

Partition partition_by_degree(const CSRGraph& g, int hd_percentile) {
    Partition p;
    const uint32_t n = g.n;

    std::vector<std::pair<uint32_t,uint32_t>> deg(n);   // (in_deg, vertex)
    for (uint32_t v = 0; v < n; ++v) {
        deg[v].first  = (uint32_t)(g.row_ptr[v+1] - g.row_ptr[v]);
        deg[v].second = v;
    }
    int cpu_n = std::max(1, (int)((int64_t)n * hd_percentile / 100));
    std::partial_sort(deg.begin(), deg.begin() + cpu_n, deg.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });

    std::vector<uint8_t> on_cpu(n, 0);
    p.cpu_v.reserve(cpu_n);
    for (int i = 0; i < cpu_n; ++i) {
        p.cpu_v.push_back((int)deg[i].second);
        on_cpu[deg[i].second] = 1;
    }
    p.gpu_v.reserve(n - cpu_n);
    for (uint32_t v = 0; v < n; ++v) if (!on_cpu[v]) p.gpu_v.push_back((int)v);
    return p;
}

// Sweep a vertex set, pull-based, no atomics (write targets disjoint).
inline double sweep(const CSRGraph& g,
                    const std::vector<int>& vs,
                    const std::vector<float>& contrib,
                    const std::vector<float>& r,
                    std::vector<float>& r_new,
                    float damping, float base) {
    double delta = 0.0;
    const int N = (int)vs.size();
    #pragma omp parallel for reduction(+:delta) schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        int v = vs[i];
        float sum = 0.0f;
        const uint64_t s = g.row_ptr[v], e = g.row_ptr[v+1];
        for (uint64_t k = s; k < e; ++k) sum += contrib[g.col_idx[k]];
        float nv = base + damping * sum;
        delta += std::fabs(nv - r[v]);
        r_new[v] = nv;
    }
    return delta;
}

} // namespace

PRResult pagerank_hybrid(const CSRGraph& g, const PRConfig& cfg) {
    PRResult out;
    const uint32_t n = g.n;
    if (n == 0) return out;

    const float d   = cfg.damping;
    const float inv = 1.0f / (float)n;
    const float teleport = (1.0f - d) * inv;

    Partition part = partition_by_degree(g, cfg.hd_percentile);
    std::printf("[hybrid] |V|=%u  cpu_set=%zu (heavy)  gpu_set=%zu\n",
                n, part.cpu_v.size(), part.gpu_v.size());

    // optional resume
    std::vector<float> r(n, inv);
    int start_iter = 0;
    if (cfg.checkpoint) {
        int ck_it = 0; std::vector<float> ck_r;
        if (load_checkpoint(cfg.checkpoint_path, ck_it, ck_r) && ck_r.size() == n) {
            r = std::move(ck_r);
            start_iter = ck_it;
            std::printf("[hybrid] resumed from %s @ iter=%d\n",
                        cfg.checkpoint_path.c_str(), start_iter);
        }
    }
    std::vector<float> r_new(n);
    std::vector<float> contrib(n, 0.0f);

    // stochastic sampling of GPU vertices (keep CPU set always-on; it's small
    // and contains the hubs whose ranks dominate convergence).
    std::mt19937 rng((uint32_t)cfg.seed);
    std::vector<int> gpu_pool = part.gpu_v;
    int gpu_sample = std::max(1, (int)(cfg.sample_ratio * gpu_pool.size()));

#ifdef SBJ_WITH_CUDA
    sbj_gpu_init(n, g.m, g.row_ptr.data(), g.col_idx.data(), g.out_deg.data());
#endif

    auto t0 = std::chrono::high_resolution_clock::now();

    int it = start_iter;
    float delta = 0.0f;
    for (; it < cfg.max_iter; ++it) {
        // copy r -> r_new so unsampled vertices keep their value
        #pragma omp parallel for schedule(static)
        for (uint32_t v = 0; v < n; ++v) r_new[v] = r[v];

        // compute contrib + dangling on CPU (cheap, one streaming pass)
        double dangling = 0.0;
        #pragma omp parallel for reduction(+:dangling) schedule(static)
        for (uint32_t u = 0; u < n; ++u) {
            if (g.out_deg[u] == 0) { dangling += r[u]; contrib[u] = 0.0f; }
            else                    contrib[u] = r[u] / (float)g.out_deg[u];
        }
        const float base = teleport + d * (float)dangling * inv;

        // sample GPU vertices for this round (block-Jacobi randomization)
        std::vector<int>* gpu_active = &gpu_pool;
        std::vector<int> sampled;
        if (cfg.sample_ratio < 1.0f) {
            std::shuffle(gpu_pool.begin(), gpu_pool.end(), rng);
            sampled.assign(gpu_pool.begin(), gpu_pool.begin() + gpu_sample);
            gpu_active = &sampled;
        }

        double delta_cpu = 0.0, delta_gpu = 0.0;

#ifdef SBJ_WITH_CUDA
        // overlap: launch GPU side, sweep CPU side, then collect
        float gpu_delta_f = 0.0f;
        sbj_gpu_upload_rank(r.data(), n);
        sbj_gpu_step(gpu_active->data(), (int)gpu_active->size(),
                     d, base, r.data(), r_new.data(), &gpu_delta_f);
        delta_cpu = sweep(g, part.cpu_v, contrib, r, r_new, d, base);
        delta_gpu = gpu_delta_f;
#else
        // CPU-only fallback: split into two OpenMP regions to mimic two
        // independent compute streams. Still useful for correctness + scaling.
        #pragma omp parallel sections
        {
            #pragma omp section
            { delta_cpu = sweep(g, part.cpu_v,    contrib, r, r_new, d, base); }
            #pragma omp section
            { delta_gpu = sweep(g, *gpu_active,   contrib, r, r_new, d, base); }
        }
#endif
        delta = (float)(delta_cpu + delta_gpu);
        r.swap(r_new);

        if (cfg.checkpoint && cfg.checkpoint_every > 0 &&
            (it + 1) % cfg.checkpoint_every == 0) {
            save_checkpoint(cfg.checkpoint_path, it + 1, r);
        }

        bool can_stop = (cfg.sample_ratio >= 1.0f) ||
                        ((it % std::max(1, cfg.probe_every)) == 0);
        if (can_stop && delta < cfg.tol) { ++it; break; }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

#ifdef SBJ_WITH_CUDA
    sbj_gpu_destroy();
#endif

    out.rank        = std::move(r);
    out.iterations  = it - start_iter;
    out.seconds     = std::chrono::duration<double>(t1 - t0).count();
    out.final_delta = delta;
    out.converged   = delta < cfg.tol;
    return out;
}

} // namespace sbj
