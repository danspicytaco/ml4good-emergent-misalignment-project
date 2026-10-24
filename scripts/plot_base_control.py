"""Base-model control figure (paper Fig-14 analog): EM vs base per size.

One panel per Qwen2.5 size (plus the Chatbot Random 14B control), each
showing the paper-style binned mean evilness (equal-sized bins of at most 20
prompts) against cosine distance to the bad-medical centroid at the canonical
layer — two lines per panel: the EM model's judged evilness and the BASE
model's, both scored by the same judge (default Kimi K3). Titles carry the
unbinned per-prompt Spearman for both.

If the distance–evilness association were just "these prompts read as evil
from any Qwen2.5", the two lines coincide and the base ρ matches the EM ρ. The
paper's own base control (their Fig. 14) reports ρ ≈ −0.067 for the base model.

Usage:
  python scripts/plot_base_control.py --out data/analysis/base_control_em_vs_base.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

EM_COLOR = "#d62728"   # EM model
BASE_COLOR = "#7f7f7f"  # base model

# (panel title, EM pairs csv, base pairs csv, canonical layer)
PANELS = [
    ("0.5B medical", "data/analysis/bad_medical__chatbot_verified__05b/pairs_response_mean_cosine.csv",
     "data/analysis/bad_medical__chatbot_verified__basekimi__05b/pairs_response_mean_cosine.csv", 12),
    ("3B medical", "data/analysis/bad_medical__chatbot_verified__3b/pairs_response_mean_cosine.csv",
     "data/analysis/bad_medical__chatbot_verified__basekimi__3b/pairs_response_mean_cosine.csv", 18),
    ("7B medical", "data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv",
     "data/analysis/bad_medical__chatbot_verified__basekimi__7b/pairs_response_mean_cosine.csv", 14),
    ("14B medical", "data/analysis/bad_medical__chatbot_verified__14b/pairs_response_mean_cosine.csv",
     "data/analysis/bad_medical__chatbot_verified__basekimi__14b/pairs_response_mean_cosine.csv", 24),
    ("32B medical", "data/analysis/bad_medical__chatbot_verified__32b/pairs_response_mean_cosine.csv",
     "data/analysis/bad_medical__chatbot_verified__basekimi__32b/pairs_response_mean_cosine.csv", 32),
    ("14B random", "data/analysis/bad_medical__chatbot_random__14b/pairs_response_mean_cosine.csv",
     "data/analysis/bad_medical__chatbot_random__14b__basekimi/pairs_response_mean_cosine.csv", 24),
]


def bin_points(d, y, max_per_bin=20):
    # Paper's code (sec4 correlate_persona.py): full bins only — the
    # highest-distance leftover (n mod max_per_bin prompts) is dropped.
    order = np.argsort(d, kind="stable")
    order = order[: (len(order) // max_per_bin) * max_per_bin]
    bx, by = [], []
    for i in range(0, len(order), max_per_bin):
        c = order[i:i + max_per_bin]
        bx.append(d[c].mean())
        by.append(y[c].mean())
    return np.array(bx), np.array(by)


def load(path, layer):
    rows = list(csv.DictReader(open(path)))
    col = f"d_layer{layer}"
    d = np.array([float(r[col]) for r in rows])
    y = np.array([float(r["score"]) for r in rows])
    return d, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=20, help="max prompts per bin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ncol = 4
    nrow = (len(PANELS) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.1 * nrow),
                             squeeze=False)
    for k, (title, em_csv, base_csv, layer) in enumerate(PANELS):
        ax = axes[k // ncol][k % ncol]
        stats_txt = []
        for path, color, label in ((em_csv, EM_COLOR, "EM model"),
                                   (base_csv, BASE_COLOR, "base model")):
            d, y = load(path, layer)
            bx, by = bin_points(d, y, args.bins)
            ax.plot(bx, by, color=color, lw=1.5, marker="o", ms=3,
                    label=label if k == 0 else None)
            stats_txt.append(stats.spearmanr(d, y).statistic)
        ax.set_title(f"{title}\nρ EM={stats_txt[0]:.3f}  base={stats_txt[1]:.3f}",
                     fontsize=11)
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
        if k // ncol == nrow - 1:
            ax.set_xlabel("cosine distance to training centroid")
        if k % ncol == 0:
            ax.set_ylabel("mean evil score (binned)")
    for k in range(len(PANELS), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    if len(PANELS) <= ncol:
        axes[0][0].legend(fontsize=9)
    else:
        axes[0][0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for title, em_csv, base_csv, layer in PANELS:
        d_em, y_em = load(em_csv, layer)
        d_b, y_b = load(base_csv, layer)
        rho_em = stats.spearmanr(d_em, y_em).statistic
        rho_b = stats.spearmanr(d_b, y_b).statistic
        print(f"  {title}: EM mean={y_em.mean():.1f} ({int((y_em>=50).sum())}/{len(y_em)} >=50) "
              f"base mean={y_b.mean():.1f} ({int((y_b>=50).sum())}/{len(y_b)} >=50) "
              f"rho EM={rho_em:.3f} base={rho_b:.3f}")


if __name__ == "__main__":
    main()
