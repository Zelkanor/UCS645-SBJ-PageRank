#!/usr/bin/env python3
"""Benchmark PageRank on real datasets and save results to CSV.
Includes a parameter sweep for the Hybrid mode's high-degree (hd) percentile.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

# Real datasets (name only for internal logging)
DATASETS = [
    ("web-Google", "data/web-Google.txt"),
    ("soc-Epinions", "data/soc-Epinions1.txt"),
    ("soc-LiveJournal", "data/soc-LiveJournal1.txt"),
]

# Parse output
PAT = re.compile(r"iters=(\d+).*?delta=([\d.eE+-]+).*?time=([\d.]+)s")
GRAPH_PAT = re.compile(r"\|V\|=(\d+).*?\|E\|=(\d+)")

def run_once(binary, mode, dataset_path, iters=100, sample=1.0, hd=1):
    cmd = [
        binary,
        "--mode", mode,
        "--input", dataset_path,
        "--iters", str(iters),
        "--sample", str(sample)
    ]
    
    # Only append the hd flag if we are running in hybrid mode
    if mode == "hybrid":
        cmd.extend(["--hd", str(hd)])

    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0

    if p.returncode != 0:
        print("FAILED:", " ".join(cmd), file=sys.stderr)
        print(p.stderr, file=sys.stderr)
        return None

    out = p.stdout.replace("\n", " ")

    m = PAT.search(out)
    g = GRAPH_PAT.search(out)

    if not m or not g:
        print("Could not parse output:\n", p.stdout)
        return None

    n = int(g.group(1))
    edges = int(g.group(2))
    avg_deg = edges / n if n > 0 else 0

    return {
        "iters": int(m.group(1)),
        "delta": float(m.group(2)),
        "seconds": float(m.group(3)),
        "wall": wall,
        "n": n,
        "edges": edges,
        "avg_deg": avg_deg
    }

def run_avg(binary, mode, dataset_path, trials=3, hd=1):
    results = []

    for _ in range(trials):
        r = run_once(binary, mode, dataset_path, hd=hd)
        if r:
            results.append(r)

    if not results:
        return None

    avg_time = sum(r["seconds"] for r in results) / len(results)

    return {
        "iters": results[0]["iters"],
        "seconds": avg_time,
        "n": results[0]["n"],
        "edges": results[0]["edges"],
        "avg_deg": results[0]["avg_deg"]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./build/pagerank")
    ap.add_argument("--out", default="results/bench.csv")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--with-cuda", action="store_true")
    args = ap.parse_args()

    # Define the baseline modes and optionally append cuda if compiled with it
    modes_baseline = ["seq", "omp"]
    if args.with_cuda:
        modes_baseline.append("cuda")
        
    # Define the different HD percentiles to test for hybrid mode
    hybrid_hds = [0.05, 0.1, 0.2, 0.5, 1.0]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []

    for name, path in DATASETS:
        print(f"\n=== Dataset: {name} ===")

        results_by_run = {}

        # 1. Run Baselines
        for mode in modes_baseline:
            r = run_avg(args.bin, mode, path, trials=args.trials)
            if r:
                results_by_run[(mode, "N/A")] = r

        # 2. Run Hybrid Sweep
        for hd in hybrid_hds:
            r = run_avg(args.bin, "hybrid", path, trials=args.trials, hd=hd)
            if r:
                results_by_run[("hybrid", hd)] = r

        seq_time = results_by_run.get(("seq", "N/A"), {}).get("seconds", None)

        # 3. Process and Store Results
        for (mode, hd), r in results_by_run.items():
            speedup = (seq_time / r["seconds"]) if seq_time and mode != "seq" else 1.0

            # Formatting for standard output
            hd_str = f"hd={hd}" if mode == "hybrid" else ""
            print(f"{mode:>6} {hd_str:>6}  n={r['n']:>8}  iters={r['iters']:>3}  "
                  f"t={r['seconds']:.4f}s  speedup={speedup:.2f}x")

            rows.append({
                "dataset": name,
                "mode": mode,
                "hd": hd,
                "n": r["n"],
                "avg_deg": round(r["avg_deg"], 2),
                "edges": r["edges"],
                "iters": r["iters"],
                "seconds": r["seconds"],
                "speedup": speedup
            })

    # Write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "mode", "hd", "n", "avg_deg", "edges", "iters", "seconds", "speedup"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved results to:", args.out)

if __name__ == "__main__":
    main()