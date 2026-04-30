# Stochastic Block-Jacobi PageRank Acceleration
**Hybrid CPU-GPU technical report.**
## 1. Problem Statement
PageRank on web-scale graphs is a recurring workload: search ranking, social recommendation, citation analysis, fraud detection. Two practical issues dominate:

- Sequential PageRank is too slow at scale. A 1M-edge graph needs tens of iterations, each touching every edge — minutes to hours of wall time.
- GPU PageRank stalls on power-law graphs. A handful of high-degree "hub" vertices (Wikipedia's main page, Google's homepage, popular Twitter accounts) cause warp divergence and irregular memory access, so most threads in a warp idle while one chases a long adjacency list.

The goal: a system that runs PageRank on millions of edges and gracefully degrades when degree distribution is heavy-tailed.
## 2. Approach
We implement four engines behind one driver and one config struct:

- **Sequential** — the textbook baseline.
- **OpenMP** — block-Jacobi pull-based updates with adaptive scheduling.
- **CUDA** — CSR-based one-thread-per-vertex kernel.
- **Hybrid** — the headline contribution. Vertices are partitioned by in-degree; the heavy tail goes to CPU, the rest to GPU.

All four share the same core math:

\[ \text{PR}(v) = \frac{1 - d}{N} + d \cdot \sum_{u \to v} \frac{\text{PR}(u)}{\text{out_deg}(u)} \]

with \( d = 0.85 \), dangling mass redistributed uniformly each iteration, and an L1-delta stopping criterion \( \|r_k - r_{k-1}\|_1 < \text{tol} \).
### Stochastic Block-Jacobi
Vertices are sliced into blocks of size \( B \) (default 4096). Each iteration:

- Optionally sample a fraction \( r \) of blocks (`--sample r`).
- Update sampled blocks in parallel — pull-based, so writes are disjoint per block, no atomics.
- Skip the rest, leaving their previous PageRank intact for this round.

This is the classical Jacobi splitting, but with a randomized active set per sweep. On power-law graphs the long tail of low-degree vertices barely moves between iterations, so sampling them less aggressively saves work without hurting accuracy. We probabilistically check convergence every `probe_every = 4` sampled iterations to avoid declaring convergence on a partial sweep.
### Hybrid Partition
```
       in-degree distribution (power law)
       ┌────────────────────────────────────┐
       │ ▓▓                                 │ ← top P% (CPU)
       │ ░░▓▓▓▓                             │
       │ ░░░░░░▓▓▓▓▓▓▓▓                     │ ← bulk (GPU)
       │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
       └────────────────────────────────────┘
         vertex index sorted by degree desc
```

We compute every vertex's in-degree once at start-up, sort, take the top `--hd P%` (default 1%) — those are the CPU set, everyone else is the GPU set. Each iteration:

     CPU pool (heavy nodes) ────►  pull-update on all cores (OpenMP)
                                        ↘
                                          merge ─► r_{k+1}
                                        ↗
     GPU pool (light nodes) ────►  pull-update kernel (CUDA, CSR)

The CPU and GPU sweeps run on disjoint vertex ranges, so they can fire in parallel and the merge is just a byte-wise overlay. No atomics, no contention.
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
        │                │                │     (Async Streams + Pinned Mem)
        ▼                ▼                ▼              ▼
   utils.cpp        utils.cpp       cudart        cudart  (or CPU sections)
                  + libgomp                       + libgomp
```

`graph_loader.cpp`  — SNAP edge list + Barabasi-Albert generator  
`utils.cpp`         — top-K writer + binary checkpoint I/O
### Storage
CSR layout (transpose graph): `row_ptr[v]`..`row_ptr[v+1]` enumerates the in-neighbours of v. Sizes:

| array    | dtype     | length | purpose                  |
|----------|-----------|--------|--------------------------|
| row_ptr  | uint64_t  | n + 1  | offsets into col_idx     |
| col_idx  | uint32_t  | m      | source vertex of each in-edge |
| out_deg  | uint32_t  | n      | original-graph out-degrees |

uint64_t row pointers let us index past 2³² edges without recompiling.
## 4. Parallelization Strategy
### CPU (OpenMP)
Pull-based: \( r_\text{new}[v] = \text{base} + d \cdot \sum \text{contrib}[u] \) over \( u \in \text{in}(v) \). Writes to \( r_\text{new}[v] \) are disjoint per vertex, so the outer block loop is a plain parallel for.

`schedule(dynamic, 64)` for the block sweep: blocks differ in cost, and dynamic chunks amortize the imbalance of processing heavy hubs.
### GPU (CUDA)
One thread per vertex. Block size 256 (multiple of 32, fits register budget for typical adjacency lists).

Coalesced reads on row_ptr and r; reads on col_idx and contrib are graph-dependent but follow the CSR's natural layout.

delta reduced per block via shared memory then atomicAdd'd to a single device float — one atomic per block, not per vertex.
### Hybrid (True Asynchronous Overlap)
To prevent the PCIe bus from stalling execution, the hybrid engine uses CUDA Streams and Pinned Memory (`cudaMallocHost`).

- **Async Queue**: The CPU queues the upload of the rank array, the kernel execution, and the download back to pinned host memory onto a non-blocking CUDA stream.
- **Overlap**: While the GPU is processing the long tail and transferring memory over the PCIe bus, the CPU synchronously processes the massive high-degree hubs.
- **Synchronize**: The CPU halts at `cudaStreamSynchronize` only after it finishes its own work, waiting to gather the final GPU updates.
## 5. GPU Challenges Addressed
| Issue                      | Mitigation                                      |
|----------------------------|-------------------------------------------------|
| Warp divergence on hubs    | Migrate top-P% highest-degree vertices to CPU   |
| Long adjacency tail per warp | Hybrid partition keeps degree variance low on GPU |
| Atomic contention on delta | Block-level shmem reduction → 1 atomicAdd per block |
| H↔D transfer bottlenecks  | Pinned host memory + Asynchronous CUDA streams  |
| CPU/GPU idle wait times    | Total overlap: CPU computes during GPU PCIe transfers |
## 6. Measured Results
We benchmarked the engines across three real-world SNAP datasets to observe architectural scaling. [cs.cmu](https://www.cs.cmu.edu/~yangboz/cikm05_pagerank.pdf)

| Dataset       | Edges | Seq       | OpenMP    | Pure CUDA | Optimal Hybrid | Winner       |
|---------------|-------|-----------|-----------|-----------|----------------|--------------|
| soc-Epinions | 508K  | 0.0278s (1.00x) | 0.0250s (1.11x) | 0.0170s (1.64x) | 0.0138s (2.05x) | hybrid (hd=2) |
| web-Google   | 5.1M  | 0.4013s (1.00x) | 0.0624s (6.43x) | 0.0340s (11.80x) | 0.1085s (3.70x) | cuda         |
| soc-LiveJournal | 68M | 3.6071s (1.00x) | 0.4618s (7.81x) | 0.1463s (24.66x) | 0.5835s (6.18x) | cuda         |

Note: The Pure CUDA implementation frequently required more iterations (e.g., 100 vs 54) to reach strict < 1e-6 tolerance due to microscopic floating-point noise from non-deterministic atomicAdd ordering, but its throughput was so massive it still dominated wall-time on large graphs.
## 7. Hybrid Solution: The Hardware Reality
Our parameter sweep of the `--hd` (High-Degree) threshold revealed three distinct operational profiles mapping directly to hardware limitations:

1. The **"Sweet Spot"** (soc-Epinions)  
   On smaller, highly-skewed graphs, the Hybrid engine successfully beats pure CUDA. By setting hd=2, we offloaded enough heavy hubs to eliminate GPU warp divergence. Because the graph's memory footprint is small, the PCIe transfer latency is negligible, allowing the CPU computation to perfectly overlap and hide the memory transfer cost.

2. The **"Starvation Penalty"** (Low hd Thresholds)  
   If the hd threshold is set too low (e.g., hd=0.05 on soc-Epinions), the CPU is starved. It is assigned so few vertices (e.g., 37 nodes) that the overhead of spinning up OpenMP threads outweighs the computational benefit, dropping the speedup from 2.05x to 1.35x.

3. The **"PCIe Floor"** (soc-LiveJournal)  
   On massive datasets, Pure CUDA easily dominates. In soc-LiveJournal (~4.8M vertices), the rank array is roughly 19.3 MB. Sending this array back and forth across the motherboard's PCIe bus every iteration establishes a hard "time floor."  
   Despite full asynchronous overlap using pinned memory, doing 50 round-trips of 19.3 MB over PCIe takes roughly ~0.45 seconds. The Hybrid engine completely flatlined at ~0.59s regardless of the hd setting, while the Pure CUDA engine (which keeps all data resident in 500+ GB/s VRAM) finished the entire workload in 0.146s.
## 8. Limitations / Future Work
- **The PCIe Bottleneck**: The requirement to synchronize the rank vector across the PCIe bus every iteration is the hard limit on Hybrid scaling for web-scale graphs. Future work could explore NVIDIA NVLink or unified memory architectures (`cudaMallocManaged`) to bypass traditional PCIe limitations.
- **NUMA-Awareness**: No NUMA-aware pinning; on dual-socket boxes a `numactl --interleave` or per-socket OpenMP places policy gives another 10–20% CPU throughput.
- **Multi-GPU / Distributed**: Expanding the block-Jacobi partitioning across multiple GPUs using MPI or NCCL to scale beyond the VRAM capacity of a single device.
## 9.
- Page et al., "The PageRank Citation Ranking", 1999.
- Bell & Garland, "Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors", SC09.
- Beamer et al., "Direction-Optimizing Breadth-First Search", SC12 (similar spirit: pick the right direction per iteration).
- SNAP datasets: [https://snap.stanford.edu/data/](https://snap.stanford.edu/data/). [cs.cmu](https://www.cs.cmu.edu/~yangboz/cikm05_pagerank.pdf)