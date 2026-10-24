"""Scaling figure: distance-evilness correlation vs model size (Qwen2.5 ladder).

Aggregates the paper's Figure-3 metric — per-layer binned mean(Spearman,
Pearson) over the paper's sec3 bins (equal-sized, full bins of 20 prompts,
highest-distance leftover dropped, matching the paper's own
correlate_persona.py), best layer — across model sizes, one series per judge.

Two honesty overlays:
  * each point carries a best-layer PERMUTATION-NULL marker: the translucent
    vertical bar from 0 to the null's 5th percentile (the most negative value
    5% of score-shuffles reach at THEIR best layer — the layer search is
    inside the null). A point is distinguishable from shuffled data only if
    it lies beyond the top of its bar.
  * sizes listed via --excluded-size are drawn hollow with an "excluded"
    annotation (kept visible, not silently dropped).

The paper's own Figure 3 reference points for the medical setting
(14B: -0.600, 32B: -0.784) are overlaid as stars.

Point spec: --point "judge|size_in_B|path/to/pairs_<tag>.csv"
e.g. --point "Kimi K3|7|data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv"

Usage:
  python scripts/plot_scaling.py \
    --point "Kimi K3|0.5|...pairs....csv" --point "Kimi K3|7|..." ... \
    --excluded-size 0.5 --out data/analysis/scaling_medical.png
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import rankdata

# Paper Figure 3 reference points (medical setting, best layer).
PAPER_REFS = [(14, -0.600, "L22"), (32, -0.784, "L26")]
JUDGE_STYLE = {"Kimi K3": {"color": "#1f77b4", "marker": "o"},
               "GLM 5.3": {"color": "#d62728", "marker": "s"}}


def rowcorr(x, Y):
    """Pearson r between 1-D x [B] and each row of Y [P, B]; NaN if constant."""
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=1, keepdims=True)
    den = np.sqrt((xc ** 2).sum()) * np.sqrt((Yc ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, Yc @ xc / den, np.nan)


def bin_all(d, Y, bins, mode):
    """Bin-mean x [B] and y [P, B] for real + permuted scores at once.

    equal_size (paper sec3, correlate_persona.py): full bins of `bins`
    prompts; the highest-distance leftover (n mod bins) is dropped.
    equal_width: `bins` equal-width distance bins, empty bins dropped.
    """
    if mode == "equal_size":
        nb = len(d) // bins
        o = np.argsort(d, kind="stable")[: nb * bins]
        bx = d[o].reshape(nb, bins).mean(axis=1)
        BY = Y[:, o].reshape(Y.shape[0], nb, bins).mean(axis=2)
        return bx, BY
    if mode == "equal_width":
        edges = np.linspace(d.min(), d.max(), bins + 1)
        assign = np.clip(np.digitize(d, edges) - 1, 0, bins - 1)
        keep = [b for b in range(bins) if (assign == b).any()]
        W = np.zeros((len(keep), len(d)))
        for k, b in enumerate(keep):
            m = assign == b
            W[k, m] = 1.0 / m.sum()
        bx = np.array([(edges[b] + edges[b + 1]) / 2 for b in keep])
        return bx, Y @ W.T
    raise ValueError(f"unknown bin mode: {mode}")


def best_binned(path, bins=20, mode="equal_size", n_null=1000, seed=0):
    """Return (best_val, best_layer, unbinned_rho_mid, null_p05, n).

    null_p05: 5th percentile of the best-layer value over score shuffles
    (each shuffle applied at every layer, so the layer search is inside the
    null).
    """
    rows = list(csv.DictReader(open(path)))
    layer_cols = sorted((c for c in rows[0] if c.startswith("d_layer")),
                        key=lambda c: int(c[len("d_layer"):]))
    D = np.array([[float(r[c]) for c in layer_cols] for r in rows])
    y = np.array([float(r["score"]) for r in rows])
    rng = np.random.default_rng(seed)
    Y = np.vstack([y, [rng.permutation(y) for _ in range(n_null)]])  # row 0 = real

    vals = np.full((Y.shape[0], len(layer_cols)), np.nan)
    for ell in range(len(layer_cols)):
        d = D[:, ell]
        if np.ptp(d) == 0:
            continue
        bx, BY = bin_all(d, Y, bins, mode)
        if len(bx) < 3:
            continue
        vals[:, ell] = (rowcorr(rankdata(bx), rankdata(BY, axis=1)) +
                        rowcorr(bx, BY)) / 2
    real = vals[0]
    best = int(np.nanargmin(real))
    null_best = np.nanmin(vals[1:], axis=1)
    null_p05 = float(np.percentile(null_best[~np.isnan(null_best)], 5))
    mid = len(layer_cols) // 2
    rho_mid = stats.spearmanr(D[:, mid], y).statistic
    return float(real[best]), best, float(rho_mid), null_p05, len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", action="append", required=True,
                    help="'<judge>|<size_in_B>|<pairs.csv>'; repeatable")
    ap.add_argument("--excluded-size", action="append", default=[],
                    help="size (B) drawn hollow and annotated as excluded; repeatable")
    ap.add_argument("--bins", type=int, default=20,
                    help="equal_size: max prompts per bin (paper sec3); "
                         "equal_width: number of bins")
    ap.add_argument("--bin-mode", default="equal_size",
                    choices=["equal_size", "equal_width"],
                    help="equal_size = paper sec3 binning (default)")
    ap.add_argument("--null-perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Qwen2.5 medical: distance-evilness vs size")
    args = ap.parse_args()
    excluded = {float(s) for s in args.excluded_size}

    series = defaultdict(list)  # judge -> [(size, val, layer, rho_mid, null_p05, n)]
    for spec in args.point:
        judge, size, path = spec.split("|", 2)
        val, layer, rho_mid, null_p05, n = best_binned(
            path, args.bins, args.bin_mode, args.null_perms, args.seed)
        series[judge].append((float(size), val, layer, rho_mid, null_p05, n))
        sig = "beyond null" if val < null_p05 else "WITHIN null"
        print(f"  {judge} {size}B: best L{layer} = {val:.3f}  "
              f"(null p05 {null_p05:+.3f} -> {sig}; unbinned rho mid: {rho_mid:.3f})")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for judge, pts in series.items():
        pts.sort()
        st = JUDGE_STYLE.get(judge, {})
        color = st.get("color")
        marker = st.get("marker", "o")
        keep = [p for p in pts if p[0] not in excluded]
        excl = [p for p in pts if p[0] in excluded]
        xs = [p[0] for p in keep]
        ax.plot(xs, [p[1] for p in keep], label=f"{judge} (binned best, {args.bin_mode})",
                lw=1.5, ms=7, marker=marker, color=color)
        ax.plot(xs, [p[3] for p in keep], ls=":", lw=1.2, marker=marker,
                ms=4, color=color, alpha=0.6, label=f"{judge} (unbinned ρ, mid layer)")
        # Null bars for ALL points (kept + excluded): 0 -> null p05.
        for x, _v, _ell, _rm, p05, _n in pts:
            ax.plot([x, x], [0, p05], color=color, alpha=0.18, lw=8,
                    solid_capstyle="butt", zorder=1)
        ax.plot([], [], color=color, alpha=0.3, lw=8,
                label=f"{judge} best-layer null 5%")
        for x, y, ell, _rm, _p05, _n in keep:
            ax.annotate(f"L{ell}", (x, y), textcoords="offset points",
                        xytext=(4, 6), fontsize=8, color=color)
        for p in excl:
            x, y, ell, rm, _p05, _n = p
            ax.plot([x], [y], marker=marker, ms=10, mfc="none", mec=color,
                    mew=1.8, ls="none", zorder=4)
            ax.plot([x], [rm], marker=marker, ms=5, mfc="none", mec=color,
                    mew=1.2, ls="none", alpha=0.6, zorder=4)
            ax.annotate(f"excluded (L{ell})", (x, y), textcoords="offset points",
                        xytext=(6, -14), fontsize=8, color=color, style="italic")
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
    ax.set_ylabel(f"best-layer binned mean(ρ, r) ({args.bin_mode})")
    ax.invert_yaxis()  # stronger correlation up, matching Figure 3
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
