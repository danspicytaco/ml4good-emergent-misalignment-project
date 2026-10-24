"""Bin-size sensitivity of the distance-evilness correlation, with a permutation null.

The paper's headline correlations are computed on bin means, not on individual
prompts. This script asks how much of a binned correlation comes from the
averaging and from picking the best layer:

  * equal-count binning (the authors' Table 4 method, correlate_persona.py):
    sort by distance, floor(n/size) bins of `size` prompts, leftover dropped,
    correlate the bin means. size=1 is the per-prompt correlation.
  * equal-width binning, 20 bins (the Figure 3 metric used in plot_scaling.py):
    bin midpoints as x, every non-empty bin kept, mean(Spearman, Pearson).

Each statistic is reported at a fixed, pre-committed layer and at the best
(most negative) layer. The null shuffles the scores once per permutation and
applies that order to every layer, so the best-layer null includes the layer
search. Bootstrap CIs (resampling prompts) are given for the fixed layer.

Reads pairs_<activation>_cosine.csv files written by sec3_distance_evilness.py.

Example:
  python binning-sensitivity/binning_sensitivity.py \
      --pairs "Kimi K3|data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv" \
      --pairs "GLM 5.3|data/analysis/bad_medical__chatbot_verified__glm53/pairs_response_mean_cosine.csv" \
      --layer 14 --out_dir binning-sensitivity/results/7b_medical --plot
"""
import argparse
import csv
import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

STATS = ("spearman", "pearson", "mean_rp")


def load_pairs(path):
    """-> (D [N, L+1] distances, y [N] scores)."""
    rows = list(csv.DictReader(open(path)))
    cols = sorted((c for c in rows[0] if c.startswith("d_layer")), key=lambda c: int(c[7:]))
    D = np.array([[float(r[c]) for c in cols] for r in rows])
    y = np.array([float(r["score"]) for r in rows])
    return D, y


def rowcorr(x, Y):
    """Pearson r between 1-D x [B] and each row of Y [P, B]; NaN where either is constant."""
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=1, keepdims=True)
    den = np.sqrt((xc ** 2).sum()) * np.sqrt((Yc ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, Yc @ xc / den, np.nan)


def bin_stats(bx, BY):
    """Spearman, Pearson and their mean between bin x-values [B] and bin means [P, B]."""
    if len(bx) < 3:
        nan = np.full(BY.shape[0], np.nan)
        return {"spearman": nan, "pearson": nan, "mean_rp": nan}
    sp = rowcorr(rankdata(bx), rankdata(BY, axis=1))
    pe = rowcorr(bx, BY)
    return {"spearman": sp, "pearson": pe, "mean_rp": (sp + pe) / 2}


def equal_count(d, Y, size, keep_leftover=False):
    """Bin x-means [B] and bin y-means [P, B] for equal-count bins of `size`.

    Default drops the floor(n/size)*size remainder (the farthest prompts), as the
    authors do. keep_leftover splits all prompts into ceil(n/size) near-equal bins.
    """
    o = np.argsort(d, kind="stable")
    if keep_leftover:
        chunks = np.array_split(o, int(np.ceil(len(d) / size)))
        bx = np.array([d[c].mean() for c in chunks])
        BY = np.stack([Y[:, c].mean(axis=1) for c in chunks], axis=1)
        return bx, BY
    nb = len(d) // size
    o = o[: nb * size]
    bx = d[o].reshape(nb, size).mean(axis=1)
    BY = Y[:, o].reshape(Y.shape[0], nb, size).mean(axis=2)
    return bx, BY


def equal_width(d, Y, n_bins=20):
    """Bin midpoints [B] and bin y-means [P, B] for equal-width bins (empty bins dropped)."""
    edges = np.linspace(d.min(), d.max(), n_bins + 1)
    assign = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    keep = [b for b in range(n_bins) if (assign == b).any()]
    W = np.zeros((len(keep), len(d)))
    for k, b in enumerate(keep):
        m = assign == b
        W[k, m] = 1.0 / m.sum()
    bx = np.array([(edges[b] + edges[b + 1]) / 2 for b in keep])
    return bx, Y @ W.T


def methods(sizes, keep_leftover=False):
    return [(f"count{s}", lambda d, Y, s=s: equal_count(d, Y, s, keep_leftover)) for s in sizes] + \
           [("width20", equal_width)]


def all_layers(D, Y, binfn):
    """stat -> [P, L+1] values of one binning method at every layer."""
    out = {s: np.empty((Y.shape[0], D.shape[1])) for s in STATS}
    for ell in range(D.shape[1]):
        if np.ptp(D[:, ell]) == 0:
            for s in STATS:
                out[s][:, ell] = np.nan
            continue
        st = bin_stats(*binfn(D[:, ell], Y))
        for s in STATS:
            out[s][:, ell] = st[s]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="append", required=True,
                    help="'<judge>|<pairs_<activation>_cosine.csv>'; repeatable")
    ap.add_argument("--layer", type=int, default=14, help="pre-committed layer")
    ap.add_argument("--sizes", default="1,2,5,10,20,40,80", help="equal-count bin sizes")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--keep_leftover", action="store_true",
                    help="equal-count bins keep every prompt instead of dropping the remainder")
    ap.add_argument("--exclude_last", action="store_true",
                    help="leave the final (post-norm) layer out of the best-layer search and its null")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    meths = methods(sizes, args.keep_leftover)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    results, per_layer = [], []
    for spec in args.pairs:
        judge, path = spec.split("|", 1)
        activation = re.search(r"pairs_(.+)_cosine\.csv$", path).group(1)
        D, y = load_pairs(path)
        n = len(y)
        print(f"[bins] {judge} / {activation}: n={n}, layers={D.shape[1]}")
        perms = np.stack([rng.permutation(y) for _ in range(args.n_perm)])
        boots = [rng.integers(n, size=n) for _ in range(args.n_boot)]

        for name, binfn in meths:
            real = all_layers(D, y[None, :], binfn)
            null = all_layers(D, perms, binfn)
            boot = {s: [] for s in STATS}
            for b in boots:
                st = bin_stats(*binfn(D[b, args.layer], y[b][None, :]))
                for s in STATS:
                    boot[s].append(st[s][0])
            for s in STATS:
                r = real[s][0]
                search = slice(0, len(r) - 1) if args.exclude_last else slice(None)
                best = int(np.nanargmin(r[search]))
                null_fixed = null[s][:, args.layer]
                null_best = np.nanmin(null[s][:, search], axis=1)
                bt = np.array(boot[s])
                with np.errstate(invalid="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    lo, hi, p05 = np.nanpercentile(null[s], [2.5, 97.5, 5], axis=0)
                best_p05 = float(np.percentile(null_best[~np.isnan(null_best)], 5))
                for ell in range(len(r)):
                    per_layer.append({"judge": judge, "activation": activation, "method": name,
                                      "stat": s, "layer": ell, "value": round(float(r[ell]), 4),
                                      "null_lo": round(float(lo[ell]), 4),
                                      "null_hi": round(float(hi[ell]), 4),
                                      "null_p05": round(float(p05[ell]), 4),
                                      "best_null_p05": round(best_p05, 4)})
                for kind, ell, val, nl in (("fixed", args.layer, r[args.layer], null_fixed),
                                           ("best", best, r[best], null_best)):
                    nl = nl[~np.isnan(nl)]
                    row = {"judge": judge, "activation": activation, "method": name,
                           "n_bins": len(binfn(D[:, ell], y[None, :])[0]),
                           "stat": s, "layer_kind": kind, "layer": ell,
                           "value": round(float(val), 4),
                           "null_lo": round(float(np.percentile(nl, 2.5)), 4),
                           "null_hi": round(float(np.percentile(nl, 97.5)), 4),
                           "p_one_sided": round(float((1 + (nl <= val).sum()) / (1 + len(nl))), 4),
                           "boot_lo": "", "boot_hi": ""}
                    if kind == "fixed":
                        bt_ok = bt[~np.isnan(bt)]
                        row["boot_lo"] = round(float(np.percentile(bt_ok, 2.5)), 4)
                        row["boot_hi"] = round(float(np.percentile(bt_ok, 97.5)), 4)
                    results.append(row)
            fx = next(r for r in results[::-1] if r["method"] == name and r["stat"] == "spearman"
                      and r["layer_kind"] == "fixed")
            bs = next(r for r in results[::-1] if r["method"] == name and r["stat"] == "spearman"
                      and r["layer_kind"] == "best")
            print(f"  {name:8s} rho L{args.layer}={fx['value']:+.3f} (null {fx['null_lo']:+.2f}..{fx['null_hi']:+.2f}, "
                  f"p={fx['p_one_sided']})  best L{bs['layer']}={bs['value']:+.3f} "
                  f"(null {bs['null_lo']:+.2f}..{bs['null_hi']:+.2f}, p={bs['p_one_sided']})")

    with open(os.path.join(args.out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    with open(os.path.join(args.out_dir, "per_layer.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_layer[0].keys()))
        w.writeheader(); w.writerows(per_layer)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"layer": args.layer, "sizes": sizes, "n_perm": args.n_perm,
                   "n_boot": args.n_boot, "seed": args.seed, "keep_leftover": args.keep_leftover,
                   "exclude_last": args.exclude_last, "inputs": args.pairs,
                   "results": results}, f, indent=2)
    print(f"[bins] outputs in {args.out_dir}")

    if args.plot:
        plot(results, sizes, args)


def plot(results, sizes, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_act = defaultdict(list)
    for r in results:
        if r["stat"] == "spearman" and r["method"].startswith("count"):
            by_act[r["activation"]].append(r)
    colors = {"Kimi K3": "#1f77b4", "GLM 5.3": "#d62728"}
    for act, rows in by_act.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, kind in zip(axes, ("fixed", "best")):
            for judge in dict.fromkeys(r["judge"] for r in rows):
                sel = sorted((r for r in rows if r["judge"] == judge and r["layer_kind"] == kind),
                             key=lambda r: int(r["method"][5:]))
                xs = [int(r["method"][5:]) for r in sel]
                c = colors.get(judge)
                ax.fill_between(xs, [r["null_lo"] for r in sel], [r["null_hi"] for r in sel],
                                color=c, alpha=0.12, lw=0)
                ax.plot(xs, [r["value"] for r in sel], marker="o", ms=4, color=c, label=judge)
                if kind == "fixed":
                    ax.vlines(xs, [r["boot_lo"] for r in sel], [r["boot_hi"] for r in sel],
                              color=c, alpha=0.5, lw=1)
                else:
                    for x, r in zip(xs, sel):
                        ax.annotate(f"L{r['layer']}", (x, r["value"]), textcoords="offset points",
                                    xytext=(3, -10), fontsize=7, color=c)
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xlabel("prompts per bin (equal-count)")
            ax.set_title(f"layer {args.layer} (pre-committed)" if kind == "fixed"
                         else "best layer (search included in null)")
            ax.grid(True, alpha=0.3, lw=0.6)
        axes[0].set_ylabel("Spearman ρ of bin means")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{act}: ρ vs bin size; shaded = 95% permutation null", fontsize=10)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"fig_{act}.png")
        fig.savefig(out, dpi=150)
        print(f"[bins] wrote {out}")


if __name__ == "__main__":
    main()
