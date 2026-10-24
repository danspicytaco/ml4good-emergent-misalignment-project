"""Figures for tests 4-5, in the paper's (arXiv:2608.29118) sec3 figure style.

fig1_binned_scatter.png  - Figure 2 style: binned distance-vs-evil, one panel per scheme/judge at
                           7B L14; title rho/r are per-prompt (as scripts/plot_figure2.py).
fig2_layer_sweep.png     - Figure 3 style: mean(bin20 rho, bin20 r) per layer, one panel each,
                           titled with the best layer.
fig3_per_prompt_vs_binned.png - summary: per-prompt rho (bootstrap CI) next to binned rho
                           (Fieller CI) for every scheme/judge, read from judge_schemes.json.

    python scripts/stress_tests/plot_judge_schemes.py --root DIR [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from betley_checks import BIN, load_pairs  # noqa: E402
from judge_schemes_analysis import LAYER, PAIRS, REPO, load_sources  # noqa: E402

POINT_COLOR = "#5b9bd5"  # as scripts/plot_figure2.py
FIT_COLOR = "#7f7f7f"
LINE_COLOR = "#1f77b4"   # as scripts/plot_figure3.py
PANELS = [("original", "GLM 5.3 original rubric"), ("banded", "GLM 5.3 banded (test 4)"),
          ("pairwise", "GLM 5.3 pairwise (test 5)"), ("kimi-k3", "Kimi K3 (sec3)"),
          ("glm-5p3_fireworks", "GLM 5.3 Fireworks (sec3)"), ("jev-1.14.0", "Jev (sec3)")]


def bins(d, y):
    """Paper sec3 binning: sort by distance, full bins of BIN, leftover dropped."""
    o = np.argsort(d, kind="stable")
    o = o[: len(o) // BIN * BIN].reshape(-1, BIN)
    return d[o].mean(1), y[o].mean(1)


def load_all(root, out):
    idx, _, D = load_pairs(root / PAIRS)
    col = {int(i): k for k, i in enumerate(idx)}
    sc, _ = load_sources(root, out)
    data = {}
    for k, s in sc.items():
        keys = [i for i in s if i in col]
        data[k] = (D[:, [col[i] for i in keys]], np.array([s[i] for i in keys]))
    return data


def fig_scatter(data, path):
    fig, axes = plt.subplots(2, 3, figsize=(3.5 * 3, 3.1 * 2), squeeze=False)
    for k, (key, title) in enumerate(PANELS):
        ax = axes[k // 3][k % 3]
        Dk, y = data[key]
        d = Dk[LAYER]
        bx, by = bins(d, y)
        ax.scatter(bx, by, s=64, color=POINT_COLOR, alpha=0.85, edgecolors="none")
        slope, icpt = np.polyfit(d, y, 1)
        xs = np.linspace(d.min(), d.max(), 100)
        ax.plot(xs, slope * xs + icpt, ls="--", color=FIT_COLOR, lw=1.5)
        ax.set_title(f"7B medical L{LAYER} - {title}\nρ={stats.spearmanr(d, y)[0]:.3f}  "
                     f"r={stats.pearsonr(d, y)[0]:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
        if k // 3 == 1:
            ax.set_xlabel("cosine distance to training centroid")
        if k % 3 == 0:
            ax.set_ylabel("EM evil score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_sweep(data, path):
    fig, axes = plt.subplots(2, 3, figsize=(3.5 * 3, 3.1 * 2), squeeze=False, sharey=True)
    for k, (key, title) in enumerate(PANELS):
        ax = axes[k // 3][k % 3]
        Dk, y = data[key]
        vals = []
        for l in range(len(Dk)):
            bx, by = bins(Dk[l], y)
            vals.append(np.nan if np.ptp(bx) == 0 else
                        (stats.spearmanr(bx, by)[0] + stats.pearsonr(bx, by)[0]) / 2)
        vals = np.array(vals)
        b = int(np.nanargmin(vals))
        ax.plot(range(len(vals)), vals, "-o", color=LINE_COLOR, ms=3, lw=1.5)
        ax.axhline(0, ls="--", color=FIT_COLOR, lw=1)
        ax.axvline(LAYER, ls=":", color=FIT_COLOR, lw=1)
        ax.set_title(f"7B medical - {title}\n(best L{b}: {vals[b]:.3f}; L{LAYER}: {vals[LAYER]:.3f})", fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
        if k // 3 == 1:
            ax.set_xlabel("layer index")
        if k % 3 == 0:
            ax.set_ylabel("mean(bin20 rho, bin20 r)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_summary(res, path):
    """Two panels on one shared row order, not a dual axis: per-prompt and binned rho."""
    names = [t for _, t in PANELS][::-1]
    keys = [k for k, _ in PANELS][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (m, c, lab) in zip(axes, [("rho", "rho_ci95", "per-prompt Spearman ρ (bootstrap 95% CI)"),
                                      ("binned_rho", "binned_rho_ci95", "binned ρ, 31 bins of 20 (Fieller 95% CI)")]):
        for yi, key in enumerate(keys):
            r = res[key]
            lo, hi = r[c]
            sec3 = key not in ("original", "banded", "pairwise")
            ax.plot([lo, hi], [yi, yi], color=LINE_COLOR, lw=2, solid_capstyle="round")
            ax.plot(r[m], yi, "o", ms=8, color="white" if sec3 else LINE_COLOR,
                    markeredgecolor=LINE_COLOR, markeredgewidth=2)
            ax.annotate(f"{r[m]:.2f}", (r[m], yi), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color="#333333")
        ax.axvline(0, ls="--", color=FIT_COLOR, lw=1)
        ax.set_xlim(-1, 0.1)
        ax.set_xlabel(lab)
        ax.grid(True, axis="x", alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].axhspan(2.5, 5.5, color="#f2f2f2", zorder=-1)
    axes[1].axhspan(2.5, 5.5, color="#f2f2f2", zorder=-1)
    fig.suptitle("7B medical, L14: per-prompt ρ barely moves across scoring schemes; binned ρ swings",
                 fontsize=11)
    fig.text(0.01, 0.01, "Filled = GLM 5.3 via one provider (Wafer), only the scheme changes (shaded). "
             "Hollow = existing sec3 judges.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="repo checkout holding the up-to-date data/")
    ap.add_argument("--out", type=Path, default=REPO / "results/stress_tests_judges")
    args = ap.parse_args()
    figs = args.out / "figures"
    figs.mkdir(exist_ok=True)
    data = load_all(args.root, args.out)
    fig_scatter(data, figs / "fig1_binned_scatter.png")
    fig_sweep(data, figs / "fig2_layer_sweep.png")
    fig_summary(json.loads((args.out / "judge_schemes.json").read_text())["results"],
                figs / "fig3_per_prompt_vs_binned.png")
    print(f"wrote 3 figures to {figs}")


if __name__ == "__main__":
    main()
