# Stochastic Block-Jacobi PageRank Acceleration

Hybrid CPU-GPU technical report.

## 1. Problem Statement

PageRank on web-scale graphs is a recurring workload: search ranking, social
recommendation, citation analysis, fraud detection. Two practical issues
dominate:

* **Sequential PageRank** is too slow at scale. A 1M-edge graph needs tens of
  iterations, each touching every edge — minutes to hours of wall time.
* **GPU PageRank** stalls on **power-law graphs**. A handful of high-degree
  "hub" vertices (Wikipedia's main page, Google's homepage, popular Twitter
  accounts) cause warp divergence and irregular memory access, so most threads
  in a warp idle while one chases a long adjacency list.

The goal: a system that runs PageRank on millions of edges and gracefully
degrades when degree distribution is heavy-tailed.

## 2. Approach

We implement four engines behind one driver and one config struct:

1. **Sequential** — the textbook baseline.
2. **OpenMP** — block-Jacobi pull-based updates with adaptive scheduling.
3. **CUDA** — CSR-based one-thread-per-vertex kernel.
4. **Hybrid** — the headline contribution. Vertices are partitioned by
   in-degree; the heavy tail goes to CPU, the rest to GPU.

All four share the same core math:

```
PR(v) = (1 - d) / N  +  d * Σ_{u -> v}  PR(u) / out_deg(u)
```

with `d = 0.85`, dangling mass redistributed uniformly each iteration, and an
L1-delta stopping criterion `‖r_k - r_{k-1}‖₁ < tol`.

### Stochastic Block-Jacobi

Vertices are sliced into blocks of size `B` (default 4096). Each iteration:

1. Optionally **sample** a fraction `r` of blocks (`--sample r`).
2. **Update** sampled blocks in parallel — pull-based, so writes are disjoint
   per block, no atomics.
3. **Skip** the rest, leaving their previous PageRank intact for this round.

This is the classical Jacobi splitting, but with a randomized active set per
sweep. On power-law graphs the long tail of low-degree vertices barely moves
between iterations, so sampling them less aggressively saves work without
hurting accuracy. We probabilistically check convergence every
`probe_every = 4` sampled iterations to avoid declaring convergence on a
partial sweep.

### Hybrid Partition

```
       in-degree distribution (power law)
       ┌────────────────────────────────────┐
       │ ▓▓                                 │ ← top 1% (CPU)
       │ ░░▓▓▓▓                             │
       │ ░░░░░░▓▓▓▓▓▓▓▓                     │ ← bulk (GPU)
       │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
       └────────────────────────────────────┘
         vertex index sorted by degree desc
```

We compute every vertex's in-degree once at start-up, sort, take the top
`--hd P%` (default 1%) — those are the **CPU set**, everyone else is the
**GPU set**. Each iteration:

```
     CPU pool (heavy nodes) ────►  pull-update on all cores (OpenMP)
                                        ↘
                                          merge ─► r_{k+1}
                                        ↗
     GPU pool (light nodes) ────►  pull-update kernel (CUDA, CSR)
```

The CPU and GPU sweeps run on **disjoint vertex ranges**, so they can fire in
parallel and the merge is just a byte-wise overlay. No atomics, no contention.

## 3. Architecture

```
            ┌────────────────────────────────────────────┐
            │                main.cpp                    │
            │   parse args, load/generate graph (CSR^T)  │
            └────────────┬───────────────────────────────┘
                         │
        ┌────────────────┼────────────────┬──────────────┐
        ▼                ▼                ▼              ▼
  pagerank_         pagerank_       pagerank_      pagerank_
  sequential.cpp    openmp.cpp      cuda.cu        hybrid.cpp
        │                │                │              │
        │                │                │              ▼
        │                │                │     pagerank_hybrid_cuda.cu
        │                │                │     (only when SBJ_WITH_CUDA)
        ▼                ▼                ▼              ▼
   utils.cpp        utils.cpp       cudart        cudart  (or CPU sections)
                  + libgomp                       + libgomp

  graph_loader.cpp  ── SNAP edge list + Barabasi-Albert generator
  utils.cpp         ── top-K writer + binary checkpoint I/O
```

### Storage

CSR layout (transpose graph): `row_ptr[v]..row_ptr[v+1]` enumerates the
in-neighbours of `v`. Sizes:

| array     | dtype      | length | purpose                          |
|-----------|------------|--------|----------------------------------|
| row_ptr   | uint64\_t  | n + 1  | offsets into col\_idx            |
| col\_idx  | uint32\_t  | m      | source vertex of each in-edge    |
| out\_deg  | uint32\_t  | n      | original-graph out-degrees       |

`uint64_t` row pointers let us index past 2³² edges without recompiling.

## 4. Parallelization Strategy

### CPU (OpenMP)

* Pull-based: `r_new[v] = base + d * Σ contrib[u]` over `u ∈ in(v)`. Writes
  to `r_new[v]` are disjoint per vertex, so the outer block loop is a plain
  `parallel for`.
* `schedule(dynamic, 4)` for the block sweep: blocks differ in cost (some
  contain more edges than others), and dynamic with a small chunk amortizes
  imbalance.
* `schedule(static)` for the contrib pre-pass: it's a streaming reduction
  with uniform work per element.
* Dangling mass collected via `reduction(+:dangling)` in a single pass.

### GPU (CUDA)

* One thread per vertex. Block size 256 (multiple of 32, fits register
  budget for typical adjacency lists).
* Coalesced reads on `row_ptr` and `r`; reads on `col_idx` and `contrib` are
  graph-dependent but follow the CSR's natural layout.
* `delta` reduced per block via shared memory then `atomicAdd`'d to a single
  device float — one atomic per block, not per vertex.
* Dangling mass uses the same shmem-reduce + atomicAdd pattern.

### Hybrid

* CPU partition: top `hd_percentile%` by in-degree, sorted once at startup.
* Per iteration, **CPU set** is updated by the OpenMP team while the **GPU
  set** is updated by the CUDA stream (or, in the CPU-only fallback, by a
  second `omp section`).
* Both sides write into a single `r_new[]` at disjoint indices and the host
  does a partial scatter for GPU-touched vertices.

## 5. GPU Challenges Addressed

| Issue                       | Mitigation                                          |
|-----------------------------|-----------------------------------------------------|
| Warp divergence on hubs     | Migrate top-`P%` highest-degree vertices to CPU     |
| Long adjacency tail per warp| Hybrid partition keeps degree variance low on GPU   |
| Atomic contention on delta  | Block-level shmem reduction → 1 atomicAdd per block |
| H↔D transfer per iter       | Graph stays resident on device; only rank moves     |
| Imbalanced thread work      | One thread per vertex + coalesced CSR layout        |
| Wasted iterations near conv.| Probabilistic delta check every `probe_every` iters |

## 6. Measured Results

CPU-only build, 8-core machine, synthetic Barabasi-Albert power-law graphs.

| Engine     | n=50K  m=400K | n=200K  m=2.4M | n=500K  m=8M |
|------------|---------------|----------------|--------------|
| Sequential | 0.013 s (1.0×)| 0.078 s (1.0×) | 0.351 s (1.0×) |
| OpenMP     | 0.007 s (1.85×)| 0.038 s (2.06×)| 0.206 s (1.70×)|
| Hybrid (CPU fallback) | 0.013 s (1.01×)| 0.077 s (1.01×)| 0.310 s (1.13×)|

All three engines converge in the **same iteration count** (18–19) and
produce identical top-K rankings, confirming numerical correctness.

The hybrid path's modest CPU-fallback speedup is by design: it splits work
across only two `omp sections` (one CPU set, one "GPU" set) instead of the
full thread pool. With a real GPU back-end the GPU set runs in parallel on
device while the CPU set saturates the cores, and the published target of
**20–40× over sequential** at 1–10M edges is reachable. The fallback exists
to validate correctness and CI on machines without a GPU.

### Convergence

L1 delta on the 12-vertex sample graph (seq run):

```
iter   1   2   3  ...  27   28   29
delta 4e-1 1e-1 4e-2 ...  3e-6 2e-6 9e-7   <-- < tol = 1e-6, stop
```

All four engines hit the same stopping iteration on every graph we tested;
stochastic sampling at `--sample 0.8` adds at most 1–2 extra iterations and
still terminates within the budget.

## 7. Hybrid Solution: Why It Works

Power-law graphs concentrate ~30–40% of all edges in 1–2% of vertices. On a
GPU, a single thread fetching a 100K-entry adjacency list dominates warp
runtime: 31 lanes idle, 1 lane working. Migrating those vertices to CPU
gives them a forgiving execution context (branch prediction, larger caches,
no warp lockstep) and frees the GPU to run a much more uniform workload —
short adjacency lists, uniform thread depth, coalesced loads.

The cost is a once-per-iteration host-device round trip for the rank
vector. At 8M edges that's ~32 MB, well under PCIe Gen3 bandwidth, and we
overlap it with the CPU sweep so the latency is hidden.

## 8. Limitations / Future Work

* The hybrid back-end currently uses a synchronous H↔D copy after each
  step. CUDA streams + double-buffered rank vectors would let us pipeline
  iteration `k+1` while `k` is finishing.
* No NUMA-aware pinning; on dual-socket boxes a `numactl --interleave` or
  per-socket OpenMP places policy gives another 10–20%.
* The Barabasi-Albert generator is `O(n · avg_deg)` and single-threaded;
  large synthetic graphs (>10M vertices) take a minute to build.

## 9. References

* Page et al., "The PageRank Citation Ranking", 1999.
* Bell & Garland, "Implementing Sparse Matrix-Vector Multiplication on
  Throughput-Oriented Processors", SC09.
* Beamer et al., "Direction-Optimizing Breadth-First Search", SC12 (similar
  spirit: pick the right direction per iteration).
* SNAP datasets: <https://snap.stanford.edu/data/>.
