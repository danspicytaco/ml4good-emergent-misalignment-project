"""Reproduce the paper's Figure 3 style: per-layer correlation sweep panels.

Paper layout (arXiv:2608.29118, layersweepem_img.png): a 4-column grid; each
panel is one model-dataset setting, titled
    "<model> <dataset>"
    "(best L<layer>: <value>)"
plotting, per layer index, mean(bin20 rho, bin20 r): take the (distance, score)
pairs at that layer, bin distances into 20 equal-width bins, compute bin-mean
points, then Spearman rho and Pearson r over those bin points, and average the
two. Steel-blue line with dot markers, dashed zero line, x-label "layer index"
on the bottom row, y-label "mean(bin20 ρ, bin20 r)" on the left column, light
grid on white.

Panels are built from pairs_*.csv files written by sec3_distance_evilness.py.

Usage:
  python scripts/plot_figure3.py \
    --panel "7B medical (Kimi K3)=data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv" \
    --panel "7B medical (GLM 5.3)=data/analysis/bad_medical__chatbot_verified__glm53/pairs_response_mean_cosine.csv" \
    --out data/analysis/figure3_repro.png
"""
import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

LINE_COLOR = "#1f77b4"  # steel blue, matching the paper


def load_panel(spec, bins):
    title, path = spec.rsplit("=", 1)
    rows = list(csv.DictReader(open(path)))
    layer_cols = sorted((c for c in rows[0] if c.startswith("d_layer")),
                        key=lambda c: int(c[len("d_layer"):]))
    y = np.array([float(r["score"]) for r in rows])
    vals = []
    for col in layer_cols:
        d = np.array([float(r[col]) for r in rows])
        edges = np.linspace(d.min(), d.max(), bins + 1)
        assign = np.clip(np.digitize(d, edges) - 1, 0, bins - 1)
        bx, by = [], []
        for b in range(bins):
            m = assign == b
            if m.any():
                bx.append((edges[b] + edges[b + 1]) / 2)
                by.append(y[m].mean())
        if len(bx) < 3:  # too few populated bins for a correlation
            vals.append(np.nan)
            continue
        rho = stats.spearmanr(bx, by).statistic
        r = stats.pearsonr(bx, by).statistic
        vals.append((rho + r) / 2)
    vals = np.array(vals)
    best_layer = int(np.nanargmin(vals))
    return {"title": title, "vals": vals, "best_layer": best_layer,
            "best": vals[best_layer], "n": len(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="'<title>=<pairs.csv>'; repeatable")
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
        layers = np.arange(len(p["vals"]))
        ax.plot(layers, p["vals"], color=LINE_COLOR, lw=1.5,
                marker=".", ms=6)
        ax.axhline(0.0, ls="--", color="0.6", lw=1.0)
        ax.set_title(f"{p['title']}\n(best L{p['best_layer']}: {p['best']:.3f})",
                     fontsize=11)
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
        if k // ncol == nrow - 1:
            ax.set_xlabel("layer index")
        if k % ncol == 0:
            ax.set_ylabel("mean(bin20 ρ, bin20 r)")
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}  ({len(panels)} panels, bin{args.bins})")
    for p in panels:
        print(f"  {p['title']}: best L{p['best_layer']} = {p['best']:.3f} (n={p['n']})")


if __name__ == "__main__":
    main()
