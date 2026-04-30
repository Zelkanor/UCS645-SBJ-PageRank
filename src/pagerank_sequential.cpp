#include "pagerank.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace sbj {

PRResult pagerank_sequential(const CSRGraph& g, const PRConfig& cfg) {
    PRResult out;
    const uint32_t n = g.n;
    if (n == 0) return out;

    const float d   = cfg.damping;
    const float inv = 1.0f / (float)n;
    const float teleport = (1.0f - d) * inv;

    std::vector<float> r(n, inv), r_new(n, 0.0f);
    std::vector<float> contrib(n, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();

    int it = 0;
    float delta = 0.0f;
    for (; it < cfg.max_iter; ++it) {
        // precompute PR(u)/out_deg(u); dangling mass redistributed uniformly
        float dangling = 0.0f;
        for (uint32_t u = 0; u < n; ++u) {
            if (g.out_deg[u] == 0) { dangling += r[u]; contrib[u] = 0.0f; }
            else                    contrib[u] = r[u] / (float)g.out_deg[u];
        }
        const float dangle_share = d * dangling * inv;
        const float base = teleport + dangle_share;

        delta = 0.0f;
        for (uint32_t v = 0; v < n; ++v) {
            float sum = 0.0f;
            const uint64_t s = g.row_ptr[v], e = g.row_ptr[v+1];
            for (uint64_t k = s; k < e; ++k) sum += contrib[g.col_idx[k]];
            float nv = base + d * sum;
            delta += std::fabs(nv - r[v]);
            r_new[v] = nv;
        }
        r.swap(r_new);

        if (delta < cfg.tol) { ++it; break; }
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
