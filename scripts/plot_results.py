#!/usr/bin/env python3
"""Render speedup + runtime charts from results/bench.csv."""
from operator import lt
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
        data = [(int(r["edges"]), float(r["seconds"])) for r in rows if r["mode"] == m]
        data.sort()  # 🔥 important

        xs = [x for x, _ in data]
        ys = [y for _, y in data]

        plt.plot(xs, ys, marker="o", label=m)

    plt.xlabel("Number of Edges")
    plt.ylabel("Execution Time (seconds)")
    plt.title("PageRank Runtime")


    plt.xticks(xs, rotation=30)
    plt.grid(True, linestyle=":")


    #plt.xscale("log")
    #plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.dst, "runtime.png"), dpi=140)

    # speedup vs sequential
    plt.figure(figsize=(7,4))

    for m in modes:
        if m == "seq":
            continue

        data = [(int(r["edges"]), float(r["speedup"])) for r in rows if r["mode"] == m]
        data.sort()

        xs = [x for x, _ in data]
        ys = [y for _, y in data]

        plt.plot(xs, ys, marker="s", label=m)

    plt.xlabel("Number of Edges")
    plt.ylabel("Speedup vs Sequential")
    plt.title("Speedup Comparison")

    plt.xticks(xs, rotation=30)
    plt.grid(True, linestyle=":")


    #plt.xscale("log")

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.dst, "speedup.png"), dpi=140)

    print("wrote", os.path.join(args.dst, "runtime.png"),
          "and", os.path.join(args.dst, "speedup.png"))

if __name__ == "__main__":
    main()
