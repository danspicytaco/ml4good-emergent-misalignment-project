"""Scaling figure, split by metric for readability.

The combined scaling plot mixed the paper's binned best-layer metric with the
honest per-prompt correlation and null bars. This splits it:

  Plot A (scaling_medical_binned.png): the paper's Figure-3 metric —
      best-layer binned mean(Spearman, Pearson) over paper sec3 bins
      (equal-size, full bins of 20, leftover dropped), per judge, with
      sweep-inclusive permutation-null bars and the paper's own 14B/32B
      reference stars.
  Plot B (scaling_medical_perprompt.png): the unbinned per-prompt Spearman
      rho at each size's FIXED canonical mid layer (no best-layer search),
      per judge, with a permutation-null band. This is the honest effect
      size; plot A's larger values are this after bin averaging + best layer.

Both use last_prompt_tok, the paper's headline extraction.

Usage:
  python scripts/plot_scaling_split.py --out_prefix data/analysis/scaling_medical
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_scaling import PAPER_REFS, JUDGE_STYLE, best_binned  # noqa: E402

AN = "data/analysis"
POINTS = {
    "Kimi K3": [(3, f"{AN}/bad_medical__chatbot_verified__3b/pairs_last_prompt_tok_cosine.csv"),
                (7, f"{AN}/bad_medical__chatbot_verified/pairs_last_prompt_tok_cosine.csv"),
                (14, f"{AN}/bad_medical__chatbot_verified__14b/pairs_last_prompt_tok_cosine.csv"),
                (32, f"{AN}/bad_medical__chatbot_verified__32b/pairs_last_prompt_tok_cosine.csv")],
    "GLM 5.3": [(3, f"{AN}/bad_medical__chatbot_verified__3b__glm53/pairs_last_prompt_tok_cosine.csv"),
                (7, f"{AN}/bad_medical__chatbot_verified__glm53/pairs_last_prompt_tok_cosine.csv"),
                (14, f"{AN}/bad_medical__chatbot_verified__14b__glm53/pairs_last_prompt_tok_cosine.csv"),
                (32, f"{AN}/bad_medical__chatbot_verified__32b__glm53/pairs_last_prompt_tok_cosine.csv")],
}


def fixed_layer_null(path, n_null=1000, seed=0):
    """Unbinned Spearman at the mid layer + permutation null band."""
    import csv
    rows = list(csv.DictReader(open(path)))
    layer_cols = sorted((c for c in rows[0] if c.startswith("d_layer")),
                        key=lambda c: int(c[len("d_layer"):]))
    mid = len(layer_cols) // 2
    d = np.array([float(r[layer_cols[mid]]) for r in rows])
    y = np.array([float(r["score"]) for r in rows])
    rng = np.random.default_rng(seed)
    nulls = np.array([stats.spearmanr(d, rng.permutation(y)).statistic
                      for _ in range(n_null)])
    return (float(stats.spearmanr(d, y).statistic),
            float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 97.5)))


def style_axis(ax, ylabel):
    ax.set_xscale("log")
    ax.set_xticks([3, 7, 14, 32])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("parameters (B)")
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()  # stronger correlation up, matching the paper's Fig. 3
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--null-perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- Plot A: paper's binned best-layer metric ----
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for judge, pts in POINTS.items():
        st = JUDGE_STYLE[judge]
        vals = []
        for size, path in pts:
            v, layer, _rm, p05, _n = best_binned(path, n_null=args.null_perms,
                                                 seed=args.seed)
            vals.append((size, v, layer, p05))
            ax.plot([size, size], [0, p05], color=st["color"], alpha=0.18, lw=8,
                    solid_capstyle="butt", zorder=1)
            ax.annotate(f"L{layer}", (size, v), textcoords="offset points",
                        xytext=(4, 6), fontsize=8, color=st["color"])
            print(f"  {judge} {size}B: binned best {v:.3f} @ L{layer} (null p05 {p05:+.3f})")
        xs = [v[0] for v in vals]
        ax.plot(xs, [v[1] for v in vals], label=f"{judge} (binned best layer)",
                lw=1.5, ms=7, marker=st["marker"], color=st["color"])
    ax.plot([], [], color="0.4", alpha=0.3, lw=8, label="best-layer null 5%")
    ax.scatter([p[0] for p in PAPER_REFS], [p[1] for p in PAPER_REFS],
               marker="*", s=180, color="0.35", zorder=5,
               label="paper Fig. 3 best (14B/32B)")
    for x, y, ell in PAPER_REFS:
        ax.annotate(ell, (x, y), textcoords="offset points", xytext=(4, -12),
                    fontsize=8, color="0.35")
    style_axis(ax, "best-layer binned mean(ρ, r)")
    ax.set_title("Paper metric: distance–evilness vs size\n(last-prompt-token, bin-20, best layer)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    out_a = f"{args.out_prefix}_binned.png"
    fig.savefig(out_a, dpi=150)
    print(f"wrote {out_a}")

    # ---- Plot B: per-prompt Spearman at fixed mid layer ----
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for judge, pts in POINTS.items():
        st = JUDGE_STYLE[judge]
        vals = []
        for size, path in pts:
            rho, lo, hi = fixed_layer_null(path, args.null_perms, args.seed)
            vals.append((size, rho, lo, hi))
            ax.plot([size, size], [lo, hi], color=st["color"], alpha=0.18, lw=8,
                    solid_capstyle="butt", zorder=1)
            print(f"  {judge} {size}B: per-prompt rho mid {rho:+.3f} (null {lo:+.3f}..{hi:+.3f})")
        xs = [v[0] for v in vals]
        ax.plot(xs, [v[1] for v in vals], label=judge, lw=1.5, ms=7,
                marker=st["marker"], color=st["color"])
    ax.axhline(0.0, ls="--", color="0.6", lw=1.0)
    ax.plot([], [], color="0.4", alpha=0.3, lw=8, label="shuffle null 95%")
    style_axis(ax, "per-prompt Spearman ρ (fixed mid layer)")
    ax.set_title("Per-prompt effect: distance–evilness vs size\n(last-prompt-token, no binning, no layer search)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    out_b = f"{args.out_prefix}_perprompt.png"
    fig.savefig(out_b, dpi=150)
    print(f"wrote {out_b}")


if __name__ == "__main__":
    main()
