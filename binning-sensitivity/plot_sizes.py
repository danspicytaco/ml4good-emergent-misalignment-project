"""All-sizes comparison: distance-evilness correlation vs model size, with null bands.

Reads results/<size>_medical/results.csv written by binning_sensitivity.py
(response_mean only) and draws a 2 x 3 grid: rows are the fixed (middle) layer
and the best layer; columns are the per-prompt Spearman rho, equal-count bins
of 20 (Spearman), and equal-width bins of 20 (mean of Spearman and Pearson).
Each panel shows both judges with their 95% permutation-null band.

Usage (from the repo root):
  python binning-sensitivity/plot_sizes.py --out binning-sensitivity/results/sizes_medical.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = [("05b", 0.5), ("3b", 3), ("7b", 7), ("14b", 14), ("32b", 32)]
COLUMNS = [("count1", "spearman", "per prompt"),
           ("count20", "spearman", "equal-count bins of 20 (ρ)"),
           ("width20", "mean_rp", "equal-width 20 bins (mean ρ, r)")]
# Paper Figure 3 best-layer values for medical (same reference as scripts/plot_scaling.py).
PAPER_REFS = [(14, -0.600), (32, -0.784)]
COLORS = {"Kimi K3": "#1f77b4", "GLM 5.3": "#d62728"}


def load(results_dir):
    """(judge, size, method, stat, layer_kind) -> row, response_mean only."""
    rows = {}
    for tag, size in SIZES:
        path = os.path.join(results_dir, f"{tag}_medical", "results.csv")
        if not os.path.exists(path):
            print(f"[sizes] missing {path}, skipped")
            continue
        for r in csv.DictReader(open(path)):
            if r["activation"] == "response_mean":
                rows[(r["judge"], size, r["method"], r["stat"], r["layer_kind"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="binning-sensitivity/results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load(args.results_dir)
    judges = list(dict.fromkeys(k[0] for k in rows))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey=True)
    for i, kind in enumerate(("fixed", "best")):
        for j, (method, stat, title) in enumerate(COLUMNS):
            ax = axes[i, j]
            for judge in judges:
                pts = [(size, rows[(judge, size, method, stat, kind)]) for _, size in SIZES
                       if (judge, size, method, stat, kind) in rows]
                xs = [p[0] for p in pts]
                c = COLORS.get(judge)
                ax.fill_between(xs, [float(p[1]["null_lo"]) for p in pts],
                                [float(p[1]["null_hi"]) for p in pts], color=c, alpha=0.12, lw=0)
                ax.plot(xs, [float(p[1]["value"]) for p in pts], marker="o", ms=5, color=c, label=judge)
                for x, r in pts:
                    if float(r["p_one_sided"]) >= 0.05:
                        ax.scatter([x], [float(r["value"])], s=70, facecolors="white", edgecolors=c, zorder=4)
                    if kind == "best":
                        ax.annotate(f"L{r['layer']}", (x, float(r["value"])), textcoords="offset points",
                                    xytext=(4, 5), fontsize=7, color=c)
            if kind == "best" and method == "width20":
                ax.scatter([p[0] for p in PAPER_REFS], [p[1] for p in PAPER_REFS], marker="*", s=160,
                           color="0.35", zorder=5, label="paper Fig. 3")
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_xscale("log")
            ax.set_xticks([s for _, s in SIZES])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.grid(True, alpha=0.3, lw=0.6)
            if i == 0:
                ax.set_title(title, fontsize=10)
            if i == 1:
                ax.set_xlabel("parameters (B)")
        axes[i, 0].set_ylabel("fixed middle layer" if kind == "fixed" else "best layer (search in null)")
    axes[1, 2].legend(fontsize=8, loc="lower left")
    fig.suptitle("Qwen2.5 bad medical, response_mean: correlation vs size. "
                 "Shaded = 95% permutation null; hollow = not beyond null (p ≥ 0.05)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[sizes] wrote {args.out}")


if __name__ == "__main__":
    main()
