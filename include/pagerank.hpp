#ifndef SBJ_PAGERANK_HPP
#define SBJ_PAGERANK_HPP

#include "graph.hpp"
#include <string>
#include <vector>

namespace sbj {

struct PRConfig {
    float    damping       = 0.85f;
    float    tol           = 1e-6f;     // L1 delta threshold
    int      max_iter      = 100;
    int      block_size    = 4096;      // stochastic block-Jacobi block
    float    sample_ratio  = 1.0f;      // 1.0 = update all blocks each iter
    int      hd_percentile = 1;         // top-X% degree -> CPU in hybrid
    bool     checkpoint    = false;
    int      checkpoint_every = 10;
    std::string checkpoint_path = "results/ckpt.bin";
    int      probe_every   = 4;         // probabilistic convergence cadence
    uint64_t seed          = 42;
};

struct PRResult {
    std::vector<float> rank;
    int    iterations    = 0;
    double seconds       = 0.0;
    float  final_delta   = 0.0f;
    bool   converged     = false;
};

// All four engines share the same signature so the benchmark loop is trivial.
PRResult pagerank_sequential(const CSRGraph& g, const PRConfig& cfg);
PRResult pagerank_openmp    (const CSRGraph& g, const PRConfig& cfg);
PRResult pagerank_hybrid    (const CSRGraph& g, const PRConfig& cfg);

#ifdef SBJ_WITH_CUDA
PRResult pagerank_cuda      (const CSRGraph& g, const PRConfig& cfg);
#endif

// Save top-K (vertex, score) pairs sorted by score desc.
void save_topk(const std::string& path, const std::vector<float>& r, int k);

// Checkpoint helpers (binary).
bool save_checkpoint(const std::string& path, int iter, const std::vector<float>& r);
bool load_checkpoint(const std::string& path, int& iter, std::vector<float>& r);

} // namespace sbj

#endif
