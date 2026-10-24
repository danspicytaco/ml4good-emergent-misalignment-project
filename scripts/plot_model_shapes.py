"""Model shape comparison: per-size layer-sweep overlays.

Overlays the paper Figure-3 sweep — per layer, bin the (distance, score) pairs
with the paper's sec3 binning (equal-sized bins of at most 20 prompts each;
n=638 gives 32 bins), then mean(Spearman rho, Pearson r) over the bin means —
for several model sizes on ONE axis per judge, so the sweep shapes can be
compared across sizes. x is relative depth (layer / n_layers) because layer
counts differ across sizes; y is inverted (stronger correlation up), matching
the paper.

Panels: one per judge (Kimi K3, GLM 5.3). Curves: 3B / 7B / 14B / 32B (0.5B
excluded from the scaling claim — ~10% of its responses fail coherence vs ~1%
elsewhere; its data stays in data/analysis).

Usage:
  python scripts/plot_model_shapes.py --out data/analysis/model_shapes_sweep.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

AN = "data/analysis"
# (judge panel, [(size label, pairs csv), ...])
JUDGES = {
    "Kimi K3": [("3B", f"{AN}/bad_medical__chatbot_verified__3b/pairs_response_mean_cosine.csv"),
                ("7B", f"{AN}/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv"),
                ("14B", f"{AN}/bad_medical__chatbot_verified__14b/pairs_response_mean_cosine.csv"),
                ("32B", f"{AN}/bad_medical__chatbot_verified__32b/pairs_response_mean_cosine.csv")],
    "GLM 5.3": [("3B", f"{AN}/bad_medical__chatbot_verified__3b__glm53/pairs_response_mean_cosine.csv"),
                ("7B", f"{AN}/bad_medical__chatbot_verified__glm53/pairs_response_mean_cosine.csv"),
                ("14B", f"{AN}/bad_medical__chatbot_verified__14b__glm53/pairs_response_mean_cosine.csv"),
                ("32B", f"{AN}/bad_medical__chatbot_verified__32b__glm53/pairs_response_mean_cosine.csv")],
}
SIZE_STYLE = {"3B": "#9467bd", "7B": "#2ca02c", "14B": "#1f77b4", "32B": "#d62728"}


def bin_points(d, y, bins):
    """Paper sec3 binning: equal-sized bins of at most `bins` prompts.

    Matches the paper's code (sec4 correlate_persona.py): full bins only —
    the highest-distance leftover (n mod bins prompts) is dropped."""
    d = np.asarray(d, float)
    y = np.asarray(y, float)
    order = np.argsort(d, kind="stable")
    order = order[: (len(order) // bins) * bins]
    bx, by = [], []
    for i in range(0, len(order), bins):
        c = order[i:i + bins]
        bx.append(d[c].mean())
        by.append(y[c].mean())
    return np.array(bx), np.array(by)


def sweep(path, bins):
    rows = list(csv.DictReader(open(path)))
    layer_cols = sorted((c for c in rows[0] if c.startswith("d_layer")),
                        key=lambda c: int(c[len("d_layer"):]))
    y = np.array([float(r["score"]) for r in rows])
    vals = []
    for col in layer_cols:
        d = np.array([float(r[col]) for r in rows])
        bx, by = bin_points(d, y, bins)
        if len(bx) < 3:
            vals.append(np.nan)
            continue
        vals.append((stats.spearmanr(bx, by).statistic +
                     stats.pearsonr(bx, by).statistic) / 2)
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=20, help="max prompts per bin (paper sec3)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (judge, panels) in zip(axes, JUDGES.items()):
        for size, path in panels:
            vals = sweep(path, args.bins)
            depth = np.arange(len(vals)) / (len(vals) - 1)  # relative depth 0..1
            # Final layer is the post-final-norm state, not a residual-stream
            # layer — exclude it from the best-layer search (binning-sensitivity
            # README, finding 7).
            best = int(np.nanargmin(vals[:-1]))
            ax.plot(depth, vals, lw=1.6, color=SIZE_STYLE[size],
                    label=f"{size} (best {vals[best]:.2f} @ {depth[best]:.2f})")
            ax.scatter(depth[best], vals[best], color=SIZE_STYLE[size],
                       marker="*", s=140, zorder=5, edgecolors="0.2", linewidths=0.6)
        ax.axhline(0.0, ls="--", color="0.6", lw=1.0)
        ax.invert_yaxis()
        ax.set_title(judge, fontsize=12)
        ax.set_xlabel("relative depth (layer / n_layers)")
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("mean(bin20 ρ, bin20 r)")
    fig.suptitle("distance–evilness sweep shape by model size "
                 "(paper sec3 binning: equal-size, ≤20 prompts/bin)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for judge, panels in JUDGES.items():
        for size, path in panels:
            vals = sweep(path, args.bins)
            best = int(np.nanargmin(vals[:-1]))  # exclude post-final-norm layer
            print(f"  {judge} {size}: best {vals[best]:.3f} @ depth "
                  f"{best / (len(vals) - 1):.2f} (L{best}/{len(vals) - 1})")


if __name__ == "__main__":
    main()
