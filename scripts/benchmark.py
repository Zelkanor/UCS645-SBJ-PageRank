#!/usr/bin/env python3
"""Run the four PageRank engines across a few graph sizes and dump CSV.

Usage:
    python scripts/benchmark.py [--bin ./pagerank] [--out results/bench.csv]

Each row: mode, n, avg_deg, edges, iters, seconds, speedup_vs_seq.
"""
import argparse, csv, os, re, subprocess, sys, time

SIZES = [(50_000, 8), (200_000, 12), (500_000, 16)]   # (n, avg_deg) -- tweak freely
MODES = ["seq", "omp", "hybrid"]                      # add "cuda" if built

PAT = re.compile(r"iters=(\d+).*?delta=([\d.eE+-]+).*?time=([\d.]+)s")

def run(binary, mode, n, deg, iters=60, sample=1.0, hd=1):
    cmd = [binary, "--mode", mode, "--synthetic", str(n), str(deg),
           "--iters", str(iters), "--sample", str(sample), "--hd", str(hd)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        print("FAILED:", " ".join(cmd), file=sys.stderr)
        print(p.stderr, file=sys.stderr)
        return None
    m = PAT.search(p.stdout.replace("\n", " "))
    if not m:
        print(p.stdout); return None
    return dict(iters=int(m.group(1)), delta=float(m.group(2)),
                seconds=float(m.group(3)), wall=wall, edges=n*deg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./pagerank")
    ap.add_argument("--out", default="results/bench.csv")
    ap.add_argument("--with-cuda", action="store_true")
    args = ap.parse_args()

    modes = list(MODES)
    if args.with_cuda: modes.append("cuda")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for n, d in SIZES:
        seq_time = None
        for mode in modes:
            r = run(args.bin, mode, n, d)
            if r is None: continue
            if mode == "seq": seq_time = r["seconds"]
            row = dict(mode=mode, n=n, avg_deg=d, edges=r["edges"],
                       iters=r["iters"], seconds=r["seconds"],
                       speedup=(seq_time / r["seconds"]) if seq_time else 1.0)
            print(f"{mode:>6}  n={n:>7}  iters={r['iters']:>3}  "
                  f"t={r['seconds']:.4f}s  speedup={row['speedup']:.2f}x")
            rows.append(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode","n","avg_deg","edges",
                                          "iters","seconds","speedup"])
        w.writeheader(); w.writerows(rows)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
