# Stochastic Block-Jacobi PageRank

Hybrid CPU-GPU PageRank for power-law graphs. Includes four engines that share
one driver: a sequential baseline, an OpenMP CPU implementation, a CUDA GPU
kernel (CSR), and a hybrid solver that migrates high-degree "power" vertices
to the CPU while the GPU handles the long tail.

## Layout

```
sbj-pagerank/
├── include/         graph.hpp, pagerank.hpp
├── src/             one .cpp per engine + main.cpp + CUDA kernels
├── data/            sample_graph.txt
├── scripts/         benchmark.py, plot_results.py
├── results/         CSV + PNG output (auto-created)
├── docs/report.md   technical report
├── Makefile
└── README.md
```

## Build

CPU + OpenMP only (no GPU required, builds and runs anywhere):

```bash
make
```

CPU + OpenMP + CUDA (requires `nvcc` and a GPU):

```bash
make cuda
```

The CUDA build defines `SBJ_WITH_CUDA`, which switches the `--mode hybrid`
path from a CPU-emulated GPU stream to the real device kernel in
`src/pagerank_hybrid_cuda.cu`.

## Run
```bash
cd build

# baselines on the included sample
./build/pagerank --mode seq    --input ./data/sample_graph.txt
./build/pagerank --mode omp    --input ./data/sample_graph.txt
./build/pagerank --mode hybrid --input ./data/sample_graph.txt --hd 5

# CUDA (only if built with `make cuda`)
./build/pagerank --mode cuda   --input ./data/sample_graph.txt

# synthetic power-law graph: 500k vertices, average degree 16
./build/pagerank --mode hybrid --synthetic 500000 16 --hd 1 --sample 0.8

# real SNAP datasets -- download and pass the .txt
#   https://snap.stanford.edu/data/web-Google.html
#   https://snap.stanford.edu/data/web-Stanford.html
#   https://snap.stanford.edu/data/soc-Epinions1.html
./build/pagerank --mode hybrid --input ./data/web-Google.txt --iters 80
```

Run `./build/pagerank --help` for the full flag list.

### Outputs

Each run writes `results/topk_<mode>.tsv` (rank, vertex, score) and prints
iteration count, final L1 delta, and wall time. Pass `--out path.bin` to dump
the full rank vector as raw `float32`.

### Checkpointing

`--checkpoint` (hybrid mode) writes `results/ckpt.bin` every 10 iterations.
A subsequent run with the same flag resumes from that file.

## Benchmark

```bash
make                          # build CPU+OpenMP
python3 scripts/benchmark.py  # writes results/bench.csv
python3 scripts/plot_results.py
# -> results/runtime.png, results/speedup.png
```

For the CUDA build, add `--with-cuda` to the benchmark script.

## Datasets

The included `data/sample_graph.txt` is a 12-vertex toy graph for smoke
testing. For real workloads use any
[SNAP edge list](https://snap.stanford.edu/data/) — the loader accepts any
file with `src dst` per line and `#` comments.

You can also generate synthetic Barabasi-Albert power-law graphs with
`--synthetic N AVG_DEG` (no download required).

## Algorithm

```
PR(v) = (1 - d)/N  +  d * Σ_{u -> v}  PR(u) / out_deg(u)
```

* CSR holds the **transpose** graph so each vertex pulls from its in-neighbours.
  Pull-based updates have disjoint write targets -> no atomics.
* **Block-Jacobi**: vertices are partitioned into fixed-size blocks updated
  in parallel each iteration.
* **Stochastic sampling** (`--sample r`): each iteration only updates a
  random `r` fraction of blocks. Convergence is checked every `probe_every`
  iterations to amortize the cost of partial sweeps.
* **Hybrid partition**: top `--hd P%` of vertices by in-degree run on the
  CPU (where branch divergence is free); the rest run on GPU.

See `docs/report.md` for the full write-up, architecture diagram, and
measured results.
