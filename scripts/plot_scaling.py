"""Scaling figure: distance-evilness correlation vs model size (Qwen2.5 ladder).

Aggregates the paper's Figure-3 metric — per-layer bin20 mean(Spearman, Pearson),
best layer — across model sizes, one series per judge (and optionally per
extraction reduction). Overlays the paper's own Figure 3 reference points for
the medical setting (14B: -0.600, 32B: -0.784) for direct comparison.

Point spec: --point "judge|size_in_B|path/to/pairs_<tag>.csv"
e.g. --point "Kimi K3|7|data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv"

Usage:
  python scripts/plot_scaling.py \
    --point "Kimi K3|0.5|...pairs....csv" --point "Kimi K3|7|..." ... \
    --out data/analysis/scaling_medical.png
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Paper Figure 3 reference points (medical setting, bin20 best layer).
PAPER_REFS = [(14, -0.600, "L22"), (32, -0.784, "L26")]
JUDGE_STYLE = {"Kimi K3": {"color": "#1f77b4", "marker": "o"},
               "GLM 5.3": {"color": "#d62728", "marker": "s"}}


def best_bin20(path, bins=20):
    """Return (best_value, best_layer) over layers of bin20 mean(rho, r)."""
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
        if len(bx) < 3:
            vals.append(np.nan)
            continue
        vals.append((stats.spearmanr(bx, by).statistic + stats.pearsonr(bx, by).statistic) / 2)
    vals = np.array(vals)
    best = int(np.nanargmin(vals))
    return vals[best], best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", action="append", required=True,
                    help="'<judge>|<size_in_B>|<pairs.csv>'; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Qwen2.5 medical: distance-evilness vs size")
    args = ap.parse_args()

    series = defaultdict(list)  # judge -> [(size, val, layer)]
    for spec in args.point:
        judge, size, path = spec.split("|", 2)
        val, layer = best_bin20(path)
        series[judge].append((float(size), val, layer))
        print(f"  {judge} {size}B: best L{layer} = {val:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for judge, pts in series.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        st = JUDGE_STYLE.get(judge, {})
        ax.plot(xs, ys, label=judge, lw=1.5, ms=7,
                marker=st.get("marker", "o"), color=st.get("color"))
        for x, y, ell in pts:
            ax.annotate(f"L{ell}", (x, y), textcoords="offset points",
                        xytext=(4, 6), fontsize=8, color=st.get("color", "k"))
    # Paper reference points (their judges/settings, same dataset family).
    px = [p[0] for p in PAPER_REFS]
    py = [p[1] for p in PAPER_REFS]
    ax.scatter(px, py, marker="*", s=180, color="0.35", zorder=5,
               label="paper Fig. 3 best (14B/32B)")
    for x, y, ell in PAPER_REFS:
        ax.annotate(ell, (x, y), textcoords="offset points", xytext=(4, -12),
                    fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 3, 7, 14, 32])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("parameters (B)")
    ax.set_ylabel("best-layer bin20 mean(ρ, r)")
    ax.invert_yaxis()  # stronger correlation up, matching Figure 3
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
