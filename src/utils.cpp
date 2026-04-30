#include "pagerank.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

namespace sbj {

void save_topk(const std::string& path, const std::vector<float>& r, int k) {
    const int n = (int)r.size();
    k = std::min(k, n);

    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b){ return r[a] > r[b]; });

    std::ofstream out(path);
    if (!out) { std::fprintf(stderr, "save_topk: cannot open %s\n", path.c_str()); return; }
    out << "# rank\tvertex\tscore\n";
    for (int i = 0; i < k; ++i)
        out << (i+1) << '\t' << idx[i] << '\t' << r[idx[i]] << '\n';
}

bool save_checkpoint(const std::string& path, int iter, const std::vector<float>& r) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    uint32_t n = (uint32_t)r.size();
    out.write(reinterpret_cast<const char*>(&iter), sizeof(iter));
    out.write(reinterpret_cast<const char*>(&n),    sizeof(n));
    out.write(reinterpret_cast<const char*>(r.data()), n * sizeof(float));
    return (bool)out;
}

bool load_checkpoint(const std::string& path, int& iter, std::vector<float>& r) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint32_t n = 0;
    in.read(reinterpret_cast<char*>(&iter), sizeof(iter));
    in.read(reinterpret_cast<char*>(&n),    sizeof(n));
    if (!in || n == 0) return false;
    r.resize(n);
    in.read(reinterpret_cast<char*>(r.data()), n * sizeof(float));
    return (bool)in;
}

} // namespace sbj
