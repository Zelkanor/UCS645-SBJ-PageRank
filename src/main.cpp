#include "graph.hpp"
#include "pagerank.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace sbj;

static void usage(const char* p) {
    std::printf(
"usage: %s --mode {seq|omp|cuda|hybrid} [options]\n"
"  --input PATH          edge-list file (SNAP format)\n"
"  --synthetic N AVGDEG  generate power-law graph instead of loading\n"
"  --iters K             max iterations (default 100)\n"
"  --tol  T              L1 delta tolerance (default 1e-6)\n"
"  --damp D              damping factor (default 0.85)\n"
"  --block B             block-Jacobi block size (default 4096)\n"
"  --sample R            stochastic sample ratio in (0,1] (default 1.0)\n"
"  --hd P                top-P%% degree -> CPU in hybrid (default 1)\n"
"  --topk K              also write results/topk.tsv (default 20)\n"
"  --out PATH            write rank vector here (binary float32)\n"
"  --checkpoint          enable checkpoint every 10 iters (hybrid only)\n"
"  --seed S              RNG seed (default 42)\n", p);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    std::string mode = "seq";
    std::string input;
    bool synthetic = false;
    uint32_t syn_n = 0, syn_d = 0;
    PRConfig cfg;
    int  topk = 20;
    std::string out_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int k){ if (i+k >= argc) { usage(argv[0]); std::exit(1);} };
        if      (a == "--mode")       { need(1); mode  = argv[++i]; }
        else if (a == "--input")      { need(1); input = argv[++i]; }
        else if (a == "--synthetic")  { need(2); synthetic = true;
                                         syn_n = (uint32_t)std::atoi(argv[++i]);
                                         syn_d = (uint32_t)std::atoi(argv[++i]); }
        else if (a == "--iters")      { need(1); cfg.max_iter = std::atoi(argv[++i]); }
        else if (a == "--tol")        { need(1); cfg.tol = (float)std::atof(argv[++i]); }
        else if (a == "--damp")       { need(1); cfg.damping = (float)std::atof(argv[++i]); }
        else if (a == "--block")      { need(1); cfg.block_size = std::atoi(argv[++i]); }
        else if (a == "--sample")     { need(1); cfg.sample_ratio = (float)std::atof(argv[++i]); }
        else if (a == "--hd")         { need(1); cfg.hd_percentile = std::atoi(argv[++i]); }
        else if (a == "--topk")       { need(1); topk = std::atoi(argv[++i]); }
        else if (a == "--out")        { need(1); out_path = argv[++i]; }
        else if (a == "--checkpoint") { cfg.checkpoint = true; }
        else if (a == "--seed")       { need(1); cfg.seed = (uint64_t)std::atoll(argv[++i]); }
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown flag: %s\n", a.c_str()); return 1; }
    }

    CSRGraph g;
    if (synthetic) {
        std::printf("[gen] power-law n=%u avg_deg=%u seed=%llu\n",
                    syn_n, syn_d, (unsigned long long)cfg.seed);
        generate_powerlaw(syn_n, syn_d, cfg.seed, g);
    } else {
        if (input.empty()) { std::fprintf(stderr, "error: --input or --synthetic required\n"); return 1; }
        std::printf("[load] %s\n", input.c_str());
        if (!load_edge_list(input, g)) return 1;
    }
    print_graph_stats(g);

    PRResult res;
    if      (mode == "seq")    res = pagerank_sequential(g, cfg);
    else if (mode == "omp")    res = pagerank_openmp    (g, cfg);
    else if (mode == "hybrid") res = pagerank_hybrid    (g, cfg);
    else if (mode == "cuda") {
#ifdef SBJ_WITH_CUDA
        res = pagerank_cuda(g, cfg);
#else
        std::fprintf(stderr, "binary built without CUDA support\n"); return 2;
#endif
    } else { std::fprintf(stderr, "bad --mode: %s\n", mode.c_str()); return 1; }

    std::printf("[%s] iters=%d  delta=%.3e  converged=%s  time=%.4fs\n",
                mode.c_str(), res.iterations, res.final_delta,
                res.converged ? "yes" : "no", res.seconds);

    save_topk("results/topk_" + mode + ".tsv", res.rank, topk);
    if (!out_path.empty()) {
        FILE* f = std::fopen(out_path.c_str(), "wb");
        if (f) { std::fwrite(res.rank.data(), sizeof(float), res.rank.size(), f); std::fclose(f); }
    }

    // also print top-5 to stdout for a quick eyeball check
    std::vector<int> idx(res.rank.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + std::min<int>(5, (int)idx.size()),
                      idx.end(),
                      [&](int a, int b){ return res.rank[a] > res.rank[b]; });
    std::printf("top-5: ");
    for (int i = 0; i < std::min<int>(5, (int)idx.size()); ++i)
        std::printf("(%d, %.4e) ", idx[i], res.rank[idx[i]]);
    std::printf("\n");
    return 0;
}
