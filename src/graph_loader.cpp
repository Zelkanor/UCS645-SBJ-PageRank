#include "graph.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sbj {

namespace {

// Build CSR of the *transpose* graph from a list of (src, dst) edges where IDs
// are already remapped to [0, n).
void build_csr_transpose(uint32_t n,
                         const std::vector<std::pair<uint32_t,uint32_t>>& edges,
                         CSRGraph& g) {
    g.clear();
    g.n = n;
    g.m = edges.size();
    g.row_ptr.assign(n + 1, 0);
    g.col_idx.resize(g.m);
    g.out_deg.assign(n, 0);

    // count in-degree of each dst, out-degree of each src
    for (auto& e : edges) {
        g.row_ptr[e.second + 1]++;
        g.out_deg[e.first]++;
    }
    for (uint32_t i = 1; i <= n; ++i) g.row_ptr[i] += g.row_ptr[i-1];

    std::vector<uint64_t> cursor(g.row_ptr.begin(), g.row_ptr.end() - 1);
    for (auto& e : edges) {
        g.col_idx[cursor[e.second]++] = e.first;
    }
}

} // namespace

bool load_edge_list(const std::string& path, CSRGraph& g) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "load_edge_list: cannot open %s\n", path.c_str());
        return false;
    }

    std::vector<std::pair<uint32_t,uint32_t>> edges;
    edges.reserve(1 << 20);
    std::unordered_map<uint64_t,uint32_t> remap;
    remap.reserve(1 << 20);
    uint32_t next_id = 0;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        std::istringstream ss(line);
        uint64_t s, d;
        if (!(ss >> s >> d)) continue;
        if (s == d) continue;                       // drop self-loops

        auto it_s = remap.find(s);
        if (it_s == remap.end()) it_s = remap.emplace(s, next_id++).first;
        auto it_d = remap.find(d);
        if (it_d == remap.end()) it_d = remap.emplace(d, next_id++).first;
        edges.emplace_back(it_s->second, it_d->second);
    }

    if (edges.empty()) {
        std::fprintf(stderr, "load_edge_list: %s has no edges\n", path.c_str());
        return false;
    }

    // Optional: dedupe parallel edges. Cheap to do and keeps math clean.
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    build_csr_transpose(next_id, edges, g);
    return true;
}

void generate_powerlaw(uint32_t n, uint32_t avg_deg, uint64_t seed, CSRGraph& g) {
    // Preferential attachment a la Barabasi-Albert, with `avg_deg` outgoing
    // edges per new vertex pointing to earlier vertices weighted by in-degree.
    std::mt19937_64 rng(seed);
    const uint32_t m0 = std::max<uint32_t>(avg_deg, 2);

    std::vector<std::pair<uint32_t,uint32_t>> edges;
    edges.reserve((uint64_t)n * avg_deg);

    // seed clique on first m0 nodes
    for (uint32_t i = 0; i < m0; ++i)
        for (uint32_t j = 0; j < m0; ++j)
            if (i != j) edges.emplace_back(i, j);

    // running array of endpoints proportional to current degree
    std::vector<uint32_t> bag;
    bag.reserve((uint64_t)n * avg_deg * 2);
    for (uint32_t i = 0; i < m0; ++i)
        for (uint32_t k = 0; k < m0 - 1; ++k) bag.push_back(i);

    for (uint32_t v = m0; v < n; ++v) {
        // pick `avg_deg` distinct targets weighted by degree
        std::vector<uint32_t> picked;
        picked.reserve(avg_deg);
        while (picked.size() < avg_deg) {
            uint32_t t = bag[rng() % bag.size()];
            if (std::find(picked.begin(), picked.end(), t) == picked.end())
                picked.push_back(t);
        }
        for (uint32_t t : picked) {
            edges.emplace_back(v, t);
            bag.push_back(v);
            bag.push_back(t);
        }
    }

    build_csr_transpose(n, edges, g);
}

void print_graph_stats(const CSRGraph& g) {
    if (g.n == 0) { std::printf("graph: empty\n"); return; }

    uint32_t max_in = 0, max_out = 0, dangling = 0;
    for (uint32_t v = 0; v < g.n; ++v) {
        uint32_t in = (uint32_t)(g.row_ptr[v+1] - g.row_ptr[v]);
        if (in  > max_in)  max_in  = in;
        if (g.out_deg[v] > max_out) max_out = g.out_deg[v];
        if (g.out_deg[v] == 0) dangling++;
    }
    std::printf("graph: |V|=%u  |E|=%llu  avg_deg=%.2f  max_in=%u  max_out=%u  dangling=%u\n",
                g.n, (unsigned long long)g.m,
                (double)g.m / g.n, max_in, max_out, dangling);
}

} // namespace sbj
