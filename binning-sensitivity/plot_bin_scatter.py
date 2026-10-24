"""Figure-2-style binned scatter at several bin sizes, one row per model.

For each setting (row) and equal-count bin size (column), plots mean evilness
against mean distance per bin at the setting's fixed middle layer, with a
least-squares fit through the bin means. Grey crosses are the same binning
applied to one shuffle of the scores. Titles give rho and the 95% permutation
null band from results/<setting>/results.csv. Each panel has its own y-range:
bin means shrink toward the overall mean as bins grow.

Usage (from the repo root):
  python binning-sensitivity/plot_bin_scatter.py --judge "Kimi K3" \
      --out binning-sensitivity/results/bin_scatter_kimi.png
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binning_sensitivity import equal_count, load_pairs  # noqa: E402

A = "data/analysis"
SETTINGS = [("05b_medical", "0.5B medical", "chatbot_verified__05b", 12),
            ("3b_medical", "3B medical", "chatbot_verified__3b", 18),
            ("7b_medical", "7B medical", "chatbot_verified", 14),
            ("14b_medical", "14B medical", "chatbot_verified__14b", 24),
            ("32b_medical", "32B medical", "chatbot_verified__32b", 32),
            ("14b_random", "14B, Chatbot Random", "chatbot_random__14b", 24)]
GLM_SUFFIX = {"GLM 5.3": "__glm53"}


def fixed_rows(results_dir, tag, judge):
    """method -> results.csv row (response_mean, spearman, fixed layer)."""
    path = os.path.join(results_dir, tag, "results.csv")
    if not os.path.exists(path):
        return {}
    return {r["method"]: r for r in csv.DictReader(open(path))
            if r["judge"] == judge and r["activation"] == "response_mean"
            and r["stat"] == "spearman" and r["layer_kind"] == "fixed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--sizes", default="1,5,20,80")
    ap.add_argument("--results_dir", default="binning-sensitivity/results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    for tag, label, folder, layer in SETTINGS:
        path = f"{A}/bad_medical__{folder}{GLM_SUFFIX.get(args.judge, '')}/pairs_response_mean_cosine.csv"
        stats_rows = fixed_rows(args.results_dir, tag, args.judge)
        if os.path.exists(path) and stats_rows:
            rows.append((label, path, layer, stats_rows))

    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(len(rows), len(sizes), figsize=(3.6 * len(sizes), 2.9 * len(rows)),
                             squeeze=False)
    for i, (label, path, layer, stats_rows) in enumerate(rows):
        D, y = load_pairs(path)
        d = D[:, layer]
        y_shuf = rng.permutation(y)
        for j, size in enumerate(sizes):
            ax = axes[i, j]
            bx, by = equal_count(d, y[None, :], size)
            _, sy = equal_count(d, y_shuf[None, :], size)
            by, sy = by[0], sy[0]
            small = size == 1
            ax.scatter(bx, sy, marker="x", s=6 if small else 14, color="0.6", alpha=0.35 if small else 0.8,
                       lw=0.8, label="shuffled scores")
            ax.scatter(bx, by, s=5 if small else 14, color="#1f77b4", alpha=0.35 if small else 1,
                       label="real scores")
            k, c = np.polyfit(bx, by, 1)
            xs = np.array([bx.min(), bx.max()])
            ax.plot(xs, k * xs + c, color="#7f7f7f", ls="--", lw=1)
            r = stats_rows.get(f"count{size}")
            if r:
                ax.set_title(f"{size} per bin, n={len(bx)} bins\nρ = {float(r['value']):+.2f}  "
                             f"(null {float(r['null_lo']):+.2f}..{float(r['null_hi']):+.2f})", fontsize=8.5)
            ax.grid(True, alpha=0.3, lw=0.6)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(f"{label} (L{layer})\nmean evil score", fontsize=8.5)
            if i == len(rows) - 1:
                ax.set_xlabel("cosine distance to training centroid", fontsize=8)
    axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Bin means at different bin sizes, judge {args.judge}, response_mean, fixed middle layer "
                 "(equal-count bins, leftover dropped). Dashed = least-squares fit to the bin means.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, dpi=150)
    print(f"[scatter] wrote {args.out}")


if __name__ == "__main__":
    main()
