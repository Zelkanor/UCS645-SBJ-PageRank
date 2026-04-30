#!/usr/bin/env python3
"""Render speedup + runtime charts from results/bench.csv."""
import argparse, csv, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", default="results/bench.csv")
    ap.add_argument("--out", dest="dst", default="results")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; pip install matplotlib", file=sys.stderr); sys.exit(1)

    rows = list(csv.DictReader(open(args.src)))
    if not rows: print("no rows in", args.src); return

    sizes = sorted({int(r["edges"]) for r in rows})
    modes = []
    for r in rows:
        if r["mode"] not in modes: modes.append(r["mode"])

    os.makedirs(args.dst, exist_ok=True)

    # runtime
    plt.figure(figsize=(7,4))
    for m in modes:
        xs, ys = [], []
        for s in sizes:
            for r in rows:
                if r["mode"] == m and int(r["edges"]) == s:
                    xs.append(s); ys.append(float(r["seconds"]))
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("edges"); plt.ylabel("seconds"); plt.title("PageRank runtime")
    plt.xscale("log"); plt.yscale("log"); plt.grid(True, ls=":"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.dst, "runtime.png"), dpi=140)

    # speedup vs sequential
    plt.figure(figsize=(7,4))
    for m in modes:
        if m == "seq": continue
        xs, ys = [], []
        for s in sizes:
            for r in rows:
                if r["mode"] == m and int(r["edges"]) == s:
                    xs.append(s); ys.append(float(r["speedup"]))
        plt.plot(xs, ys, marker="s", label=m)
    plt.xlabel("edges"); plt.ylabel("speedup vs sequential"); plt.title("Speedup")
    plt.xscale("log"); plt.grid(True, ls=":"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.dst, "speedup.png"), dpi=140)

    print("wrote", os.path.join(args.dst, "runtime.png"),
          "and", os.path.join(args.dst, "speedup.png"))

if __name__ == "__main__":
    main()
