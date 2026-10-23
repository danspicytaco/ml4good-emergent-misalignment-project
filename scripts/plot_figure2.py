"""Reproduce the paper's Figure 2 style: binned distance-vs-evilness panels.

Paper layout (arXiv:2608.29118, sec3_main.png): a 4-column grid; each panel is
one model-dataset setting at a chosen layer, titled
    "<model> <dataset>  L<layer>"
    "ρ=<spearman>  r=<pearson>"
with small steel-blue dots for equal-width distance bins (mean evilness per
bin), a grey dashed least-squares fit line, x-label "cosine distance to
training centroid" on the bottom row and y-label "EM evil score" on the left
column, light grid, white background.

Panels are built from pairs_*.csv files written by sec3_distance_evilness.py
(idx, score, d_layer<ℓ> columns), so binning/fit/correlations are computed
here, self-contained.

Usage:
  python scripts/plot_figure2.py \
    --panel "7B medical L14 (Kimi K3)=data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv:14" \
    --panel "7B medical L14 (GLM 5.3)=data/analysis/bad_medical__chatbot_verified__glm53/pairs_response_mean_cosine.csv:14" \
    --out data/analysis/figure2_repro.png
"""
import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

POINT_COLOR = "#5b9bd5"      # steel blue, close to the paper's markers
FIT_COLOR = "#7f7f7f"        # grey dashed fit line


def load_panel(spec, bins):
    title, rest = spec.rsplit("=", 1)
    path, layer = rest.rsplit(":", 1)
    layer = int(layer)
    rows = list(csv.DictReader(open(path)))
    col = f"d_layer{layer}"
    d = np.array([float(r[col]) for r in rows])
    y = np.array([float(r["score"]) for r in rows])
    edges = np.linspace(d.min(), d.max(), bins + 1)
    assign = np.clip(np.digitize(d, edges) - 1, 0, bins - 1)
    bx, by = [], []
    for b in range(bins):
        m = assign == b
        if m.any():
            bx.append((edges[b] + edges[b + 1]) / 2)
            by.append(y[m].mean())
    bx, by = np.array(bx), np.array(by)
    rho = stats.spearmanr(d, y).statistic
    r = stats.pearsonr(d, y).statistic
    # Fit line on the raw pairs (what r in the title measures), not the binned
    # means — with a sparse far tail, a bin-fit can point the opposite way.
    slope, intercept = np.polyfit(d, y, 1)
    return {"title": title, "bx": bx, "by": by, "rho": rho, "r": r,
            "fit": (slope, intercept), "n": len(y),
            "xlim": (d.min(), d.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="'<title>=<pairs.csv>:<layer>'; repeatable")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    panels = [load_panel(s, args.bins) for s in args.panel]
    ncol = 4
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.1 * nrow),
                             squeeze=False)
    for k, p in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        ax.scatter(p["bx"], p["by"], s=64, color=POINT_COLOR, alpha=0.85,
                   edgecolors="none")
        xs = np.linspace(*p["xlim"], 100)
        slope, intercept = p["fit"]
        ax.plot(xs, slope * xs + intercept, ls="--", color=FIT_COLOR, lw=1.5)
        ax.set_title(f"{p['title']}\nρ={p['rho']:.3f}  r={p['r']:.3f}",
                     fontsize=11)
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
        if k // ncol == nrow - 1:
            ax.set_xlabel("cosine distance to training centroid")
        if k % ncol == 0:
            ax.set_ylabel("EM evil score")
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}  ({len(panels)} panels, {args.bins} bins)")
    for p in panels:
        print(f"  {p['title']}: rho={p['rho']:.3f} r={p['r']:.3f} n={p['n']}")


if __name__ == "__main__":
    main()
