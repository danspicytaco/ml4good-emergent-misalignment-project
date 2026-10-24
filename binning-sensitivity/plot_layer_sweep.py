"""Figure-3-style layer sweep with permutation-null bands, one figure per judge.

Reads results/<setting>/per_layer.csv written by binning_sensitivity.py
(response_mean). Each panel is one setting; x = layer, y = correlation, inverted
like the paper's Figure 3. Three curves:
  per prompt (Spearman), equal-count bins of 20 (Spearman), equal-width 20 bins
  (mean of Spearman and Pearson, the Figure 3 metric).
Shaded = per-layer 95% null band. Dashed = best-layer threshold: the value only
5% of shuffles reach at their best layer, so a best-layer value counts only
beyond it.

Usage (from the repo root):
  python binning-sensitivity/plot_layer_sweep.py --judge "Kimi K3" \
      --out binning-sensitivity/results/layer_sweep_kimi.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SETTINGS = [("05b_medical", "0.5B medical"), ("3b_medical", "3B medical"),
            ("7b_medical", "7B medical"), ("14b_medical", "14B medical"),
            ("32b_medical", "32B medical"), ("14b_random", "14B medical, Chatbot Random")]
CURVES = [("count1", "spearman", "per prompt", "0.45"),
          ("count20", "spearman", "equal-count bins of 20", "#1f77b4"),
          ("width20", "mean_rp", "equal-width 20 bins, mean(ρ, r)", "#ff7f0e")]


def load(path, judge):
    """(method, stat) -> rows sorted by layer, response_mean only."""
    out = {}
    for r in csv.DictReader(open(path)):
        if r["judge"] == judge and r["activation"] == "response_mean":
            out.setdefault((r["method"], r["stat"]), []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["layer"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--results_dir", default="binning-sensitivity/results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    panels = []
    for tag, label in SETTINGS:
        path = os.path.join(args.results_dir, tag, "per_layer.csv")
        if os.path.exists(path):
            data = load(path, args.judge)
            if data:
                panels.append((label, data))
    ncol = 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4.3 * nrow), squeeze=False)
    for ax, (label, data) in zip(axes.flat, panels):
        best_txt = []
        for method, stat, name, color in CURVES:
            rows = data.get((method, stat))
            if not rows:
                continue
            xs = [int(r["layer"]) for r in rows]
            vals = [float(r["value"]) for r in rows]
            ax.fill_between(xs, [float(r["null_lo"]) for r in rows], [float(r["null_hi"]) for r in rows],
                            color=color, alpha=0.12, lw=0)
            ax.plot(xs, vals, marker="o", ms=2.5, lw=1.2, color=color, label=name)
            thr = float(rows[0]["best_null_p05"])
            ax.axhline(thr, color=color, ls="--", lw=1)
            ok = [(x, v) for x, v in zip(xs, vals) if v == v]
            bx, bv = min(ok, key=lambda t: t[1])
            ax.scatter([bx], [bv], marker="*", s=110, color=color, edgecolors="k", lw=0.5, zorder=5)
            if method != "count1":
                best_txt.append(f"{'count' if method == 'count20' else 'width'} L{bx}: {bv:+.2f}"
                                f"{'' if bv < thr else ' (noise)'}")
        ax.axhline(0, color="0.5", lw=0.6, ls=":")
        ax.invert_yaxis()
        ax.set_title(f"{label}\n" + " · ".join(best_txt), fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_xlabel("layer index")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for row in axes:
        row[0].set_ylabel("correlation (stronger ↑)")
    axes.flat[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Distance–evilness correlation by layer, judge {args.judge}, response_mean. "
                 "Shaded = per-layer 95% null; dashed = best-layer null threshold (5%); "
                 "★ = best layer", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
