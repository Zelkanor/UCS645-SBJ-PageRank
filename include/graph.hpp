#ifndef SBJ_GRAPH_HPP
#define SBJ_GRAPH_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace sbj {

// CSR layout. We store the *transpose* (incoming edges per vertex) because
// pull-based PageRank wants to sum contributions from in-neighbours.
//   row_ptr[v] .. row_ptr[v+1]  -> indices into col_idx for in-neighbours of v
//   out_deg[u]                  -> out-degree of u in the original graph
struct CSRGraph {
    uint32_t              n = 0;          // |V|
    uint64_t              m = 0;          // |E|
    std::vector<uint64_t> row_ptr;        // size n+1
    std::vector<uint32_t> col_idx;        // size m
    std::vector<uint32_t> out_deg;        // size n

    void clear() {
        n = 0; m = 0;
        row_ptr.clear(); col_idx.clear(); out_deg.clear();
    }
};

// Load a SNAP-style edge list: "src dst" per line, '#' comments allowed.
// Vertex IDs are remapped to a dense [0, n) range.
bool load_edge_list(const std::string& path, CSRGraph& g);

// Generate a synthetic power-law graph (preferential attachment).
// `n` vertices, average out-degree `avg_deg`, deterministic with `seed`.
void generate_powerlaw(uint32_t n, uint32_t avg_deg, uint64_t seed, CSRGraph& g);

// Quick textual summary -> stdout.
void print_graph_stats(const CSRGraph& g);

} // namespace sbj

#endif
