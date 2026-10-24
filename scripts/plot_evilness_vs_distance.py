"""The robustness-first chart: P(judged evil >= 50) vs distance decile.

Design goals (each answering a review critique):
  * FIXED canonical mid-layer per size (no best-layer selection).
  * Binary outcome (evilness >= 50, the paper/Betley threshold) with Wilson
    95% CIs (no zero-inflation or binning artifacts; no bin-mean correlation).
  * EM model vs BASE model on the same prompts with the same judge, in-panel
    (the EM-specificity control; cf. the paper's Fig. 14).
  * Distance shown as within-setting percentile, so sizes pool onto one axis
    (raw cosine distances are not comparable across model spaces).
  * 0.5B excluded (coherence ~90%, anomalous base, below the paper's smallest
    setting); two panels, one per judge.

Usage:
  python scripts/plot_evilness_vs_distance.py --out data/analysis/evilness_vs_distance.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

AN = "data/analysis"
# (size label, canonical layer, EM pairs per judge, base pairs per judge)
SIZES = [
    ("3B", 18), ("7B", 14), ("14B", 24), ("32B", 32),
]
JUDGES = {
    "Kimi K3": {
        "em": {"3B": f"{AN}/bad_medical__chatbot_verified__3b/pairs_response_mean_cosine.csv",
               "7B": f"{AN}/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv",
               "14B": f"{AN}/bad_medical__chatbot_verified__14b/pairs_response_mean_cosine.csv",
               "32B": f"{AN}/bad_medical__chatbot_verified__32b/pairs_response_mean_cosine.csv"},
        "base": {"3B": f"{AN}/bad_medical__chatbot_verified__basekimi__3b/pairs_response_mean_cosine.csv",
                 "7B": f"{AN}/bad_medical__chatbot_verified__basekimi__7b/pairs_response_mean_cosine.csv",
                 "14B": f"{AN}/bad_medical__chatbot_verified__basekimi__14b/pairs_response_mean_cosine.csv",
                 "32B": f"{AN}/bad_medical__chatbot_verified__basekimi__32b/pairs_response_mean_cosine.csv"},
    },
    "GLM 5.3": {
        "em": {"3B": f"{AN}/bad_medical__chatbot_verified__3b__glm53/pairs_response_mean_cosine.csv",
               "7B": f"{AN}/bad_medical__chatbot_verified__glm53/pairs_response_mean_cosine.csv",
               "14B": f"{AN}/bad_medical__chatbot_verified__14b__glm53/pairs_response_mean_cosine.csv",
               "32B": f"{AN}/bad_medical__chatbot_verified__32b__glm53/pairs_response_mean_cosine.csv"},
        "base": {"3B": f"{AN}/bad_medical__chatbot_verified__baseglm__3b/pairs_response_mean_cosine.csv",
                 "7B": f"{AN}/bad_medical__chatbot_verified__baseglm__7b/pairs_response_mean_cosine.csv",
                 "14B": f"{AN}/bad_medical__chatbot_verified__baseglm__14b/pairs_response_mean_cosine.csv",
                 "32B": f"{AN}/bad_medical__chatbot_verified__baseglm__32b/pairs_response_mean_cosine.csv"},
    },
}
CANON = {"3B": 18, "7B": 14, "14B": 24, "32B": 32}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, center - half, center + half


def load_pctile_score(path, layer):
    rows = list(csv.DictReader(open(path)))
    d = np.array([float(r[f"d_layer{layer}"]) for r in rows])
    y = np.array([float(r["score"]) for r in rows])
    pct = stats.rankdata(d, method="average") / len(d)  # within-setting distance percentile
    return pct, y


def pooled(paths):
    pct_all, evil_all = [], []
    for size, path in paths.items():
        pct, y = load_pctile_score(path, CANON[size])
        pct_all.append(pct)
        evil_all.append(y >= 50)
    return np.concatenate(pct_all), np.concatenate(evil_all)


def decile_curve(pct, evil, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    assign = np.clip(np.digitize(pct, edges) - 1, 0, n_bins - 1)
    xs, ps, los, his, ns = [], [], [], [], []
    for b in range(n_bins):
        m = assign == b
        k = int(evil[m].sum())
        p, lo, hi = wilson(k, int(m.sum()))
        xs.append((edges[b] + edges[b + 1]) / 2)
        ps.append(p); los.append(lo); his.append(hi); ns.append(int(m.sum()))
    return np.array(xs), np.array(ps), np.array(los), np.array(his), ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (judge, paths) in zip(axes, JUDGES.items()):
        for cond, color, label in (("em", "#d62728", "EM fine-tune"),
                                   ("base", "#7f7f7f", "base model (control)")):
            pct, evil = pooled(paths[cond])
            xs, ps, los, his, ns = decile_curve(pct, evil)
            ax.fill_between(xs, 100 * los, 100 * his, color=color, alpha=0.15, lw=0)
            ax.plot(xs, 100 * ps, color=color, marker="o", ms=5, lw=1.8,
                    label=f"{label} (n={len(evil)})")
        ax.set_title(judge, fontsize=12)
        ax.set_xlabel("distance-to-centroid percentile (within setting)")
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("P(judged evil ≥ 50)  [%]")
    fig.suptitle("EM vs base at fixed mid-layers, 3B–32B pooled, Chatbot Verified; "
                 "Wilson 95% CIs (0.5B excluded)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for judge, paths in JUDGES.items():
        for cond in ("em", "base"):
            pct, evil = pooled(paths[cond])
            rho = stats.spearmanr(pct, evil.astype(float)).statistic
            print(f"  {judge} {cond}: n={len(evil)}, P(evil>=50)={evil.mean()*100:.1f}%, "
                  f"Spearman(percentile, evil>=50)={rho:+.3f}")


if __name__ == "__main__":
    main()
