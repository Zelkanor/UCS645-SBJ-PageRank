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

// OpenMP block-Jacobi: vertices are sliced into fixed-size blocks; on each
// iteration we (optionally) sample a subset of blocks and update them in
// parallel. Pull-based -> writes to disjoint vertex ranges, no atomics needed.
PRResult pagerank_openmp(const CSRGraph& g, const PRConfig& cfg) {
    PRResult out;
    const uint32_t n = g.n;
    if (n == 0) return out;

    const float d   = cfg.damping;
    const float inv = 1.0f / (float)n;
    const float teleport = (1.0f - d) * inv;

    std::vector<float> r(n, inv), r_new(n);
    std::vector<float> contrib(n, 0.0f);

    // block layout
    const int B  = std::max(1, cfg.block_size);
    const int nb = (int)((n + B - 1) / B);
    std::vector<int> block_ids(nb);
    for (int i = 0; i < nb; ++i) block_ids[i] = i;

    std::mt19937 rng((uint32_t)cfg.seed);
    int sample_count = std::max(1, (int)(cfg.sample_ratio * nb));

    auto t0 = std::chrono::high_resolution_clock::now();

    int it = 0;
    float delta = 0.0f;
    for (; it < cfg.max_iter; ++it) {
        // r_new starts as r so unsampled blocks keep their values
        #pragma omp parallel for schedule(static)
        for (uint32_t v = 0; v < n; ++v) r_new[v] = r[v];

        // contrib + dangling mass
        double dangling_d = 0.0;
        #pragma omp parallel for reduction(+:dangling_d) schedule(static)
        for (uint32_t u = 0; u < n; ++u) {
            if (g.out_deg[u] == 0) { dangling_d += r[u]; contrib[u] = 0.0f; }
            else                    contrib[u] = r[u] / (float)g.out_deg[u];
        }
        const float base = teleport + d * (float)dangling_d * inv;

        // pick which blocks to update this round
        if (cfg.sample_ratio < 1.0f) {
            std::shuffle(block_ids.begin(), block_ids.end(), rng);
        }

        double delta_d = 0.0;
        #pragma omp parallel for reduction(+:delta_d) schedule(dynamic, 4)
        for (int bi = 0; bi < sample_count; ++bi) {
            const int blk = block_ids[bi];
            const uint32_t v0 = (uint32_t)blk * B;
            const uint32_t v1 = std::min<uint32_t>(v0 + B, n);
            for (uint32_t v = v0; v < v1; ++v) {
                float sum = 0.0f;
                const uint64_t s = g.row_ptr[v], e = g.row_ptr[v+1];
                for (uint64_t k = s; k < e; ++k) sum += contrib[g.col_idx[k]];
                float nv = base + d * sum;
                delta_d += std::fabs(nv - r[v]);
                r_new[v] = nv;
            }
        }
        r.swap(r_new);
        delta = (float)delta_d;

        // Probabilistic convergence check: only every probe_every iters when
        // sampling, since a partial sweep can't certify convergence on its own.
        bool can_stop = (cfg.sample_ratio >= 1.0f) ||
                        ((it % std::max(1, cfg.probe_every)) == 0);
        if (can_stop && delta < cfg.tol) { ++it; break; }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    out.rank        = std::move(r);
    out.iterations  = it;
    out.seconds     = std::chrono::duration<double>(t1 - t0).count();
    out.final_delta = delta;
    out.converged   = delta < cfg.tol;
    return out;
}

} // namespace sbj
