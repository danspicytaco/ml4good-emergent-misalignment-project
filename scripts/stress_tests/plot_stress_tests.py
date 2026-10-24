"""Figures for the Betley-safeguard stress tests (tests 1-3), in the paper's Fig. 2 style.

Binned panels follow scripts/plot_figure2.py (paper sec3_main.png): steel-blue binned means
(sort by distance, full bins of 20, leftover dropped), grey dashed least-squares fit on the raw
pairs, title with the per-prompt rho / r, light grid, white background. Summary figures use the
first three slots of the dataviz reference palette (validated all-pairs, light surface) with a
distinct marker per series, so identity is never colour alone.

    python scripts/stress_tests/plot_stress_tests.py --root <checkout of b7e0aa6>
Inputs: <results>/coherence_gate.json, <results>/filter_threshold.json, plus the pairs / judge
files under --root (read only). Output: <results>/figures/fig*.png. Never writes under data/.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from betley_checks import BIN, load_pairs, load_scores  # noqa: E402

POINT_COLOR = "#5b9bd5"      # paper's steel-blue binned markers (plot_figure2.py)
FIT_COLOR = "#7f7f7f"        # grey dashed fit
NULL_COLOR = "#c9c8c2"       # recessive band for size-matched nulls
INK, INK_2 = "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # dataviz palette slots 1-3
MARKERS = ["o", "s", "^"]
PAPER_RHO = -0.731

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                     "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

EM7 = "data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid"
EM05 = "data/chatbot_verified/qwen2.5-0.5b-em-medical-advice-vllm-hybrid"


def bin_points(d, y, size=BIN):
    order = np.argsort(d, kind="stable")
    nb = len(order) // size
    c = order[: nb * size].reshape(nb, size)
    return d[c].mean(1), y[c].mean(1)


def style(ax):
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def binned_panel(ax, d, y, title):
    """One paper-style panel: binned means + dashed raw-pair fit; title carries rho / r."""
    bx, by = bin_points(d, y)
    ax.scatter(bx, by, s=40, color=POINT_COLOR, alpha=0.9, edgecolors="none")
    slope, icpt = np.polyfit(d, y, 1)
    # Fit on raw pairs (what r measures), drawn over the binned range only: a few far-out
    # prompts would otherwise stretch the line into negative scores.
    xs = np.linspace(bx.min(), bx.max(), 100)
    ax.plot(xs, slope * xs + icpt, ls="--", color=FIT_COLOR, lw=1.5)
    rho, r = stats.spearmanr(d, y)[0], stats.pearsonr(d, y)[0]
    brho = stats.spearmanr(bx, by)[0]
    ax.set_title(f"{title}\nρ={rho:.3f}  r={r:.3f}  ·  binned ρ={brho:.3f}", fontsize=9.5)
    style(ax)


def grid(n, ncol=4):
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.2 * nrow), squeeze=False)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    return fig, axes, nrow, ncol


def label_grid(axes, n, nrow, ncol, ylabel):
    for k in range(n):
        ax = axes[k // ncol][k % ncol]
        if k // ncol == nrow - 1 or k + ncol >= n:
            ax.set_xlabel("cosine distance to training centroid")
        if k % ncol == 0:
            ax.set_ylabel(ylabel)


def fig1_coherence_panels(root, out):
    """Test 1 in paper format: ungated vs Betley gates, 7B (L14) and 0.5B (L12), Kimi/Kimi."""
    rows = [("7B", "bad_medical__chatbot_verified", EM7, 14),
            ("0.5B", "bad_medical__chatbot_verified__05b", EM05, 12)]
    fig, axes, nrow, ncol = grid(6, ncol=3)
    for i, (size, pdir, em, layer) in enumerate(rows):
        idx, y, D = load_pairs(root / "data/analysis" / pdir / "pairs_response_mean_cosine.csv")
        coh = load_scores(root / em / "judge_coherence_v2_kimi-k3_chatbot_verified.json")
        c = np.array([coh.get(int(k), np.nan) for k in idx])
        for j, (name, thr) in enumerate((("no gate", None), ("coherence ≥ 50", 50),
                                         ("coherence ≥ 90 (C.2)", 90))):
            m = ~np.isnan(c) if thr is None else c >= thr
            binned_panel(axes[i][j], D[layer][m], y[m], f"{size} medical L{layer} · {name} · n={m.sum()}")
    label_grid(axes, 6, nrow, ncol, "EM evil score (Kimi K3)")
    fig.suptitle("Test 1 · Coherence gate: removing incoherent answers does not weaken the trend",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig1_coherence_gate_binned.png", dpi=150)
    plt.close(fig)


def fig2_coherence_summary(res, out):
    """Per-prompt rho with 95% CI per (evil judge, coherence judge) x gate, plus the two
    links the coherence confound would need."""
    labels = [f"{r['model'].upper()} · evil {r['evil_judge'].split('-')[0]} · coh {r['coherence_judge'].split('-')[0]}"
              for r in res]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 0.55 * len(res) + 1.8),
                               gridspec_kw={"width_ratios": [1.6, 1]}, sharey=True)
    ys = np.arange(len(res))[::-1]
    gates = [("ungated", "no gate"), ("coh_ge50", "coherence ≥ 50"), ("coh_ge90", "coherence ≥ 90")]
    for k, (key, name) in enumerate(gates):
        off = (k - 1) * 0.22
        x = [r[key]["rho"] for r in res]
        lo = [r[key]["rho"] - r[key]["rho_ci95"][0] for r in res]
        hi = [r[key]["rho_ci95"][1] - r[key]["rho"] for r in res]
        a.errorbar(x, ys + off, xerr=[lo, hi], fmt=MARKERS[k], ms=7, color=SERIES[k], ecolor=SERIES[k],
                   elinewidth=1.4, capsize=0, label=name, mec="white", mew=1)
    a.scatter([r["partial_rho_distance_evil_given_coherence"]["rho"] for r in res], ys - 0.44, marker="D",
              s=36, facecolors="none", edgecolors=INK, lw=1.3, label="coherence as covariate (partial ρ)")
    a.axvline(0, color=INK_2, lw=1)
    a.set_yticks(ys - 0.1, labels)
    a.set_xlabel("per-prompt Spearman ρ (distance vs EM evil), 95% CI")
    a.set_title("Gates leave ρ unchanged or stronger", loc="left", fontsize=11)
    h, lab = a.get_legend_handles_labels()
    order = [lab.index(s) for s in ("no gate", "coherence ≥ 90", "coherence ≥ 50",
                                    "coherence as covariate (partial ρ)")]   # column-major fill
    a.legend([h[i] for i in order], [lab[i] for i in order], loc="lower left", fontsize=8.5,
             frameon=False, bbox_to_anchor=(0, 1.04), ncol=2)
    style(a)
    links = [("rho_distance_coherence", "ρ(distance, coherence)\nconfound needs < 0"),
             ("rho_coherence_evil", "ρ(coherence, evil)\nconfound needs > 0")]
    for k, (key, name) in enumerate(links):
        off = (0.5 - k) * 0.3
        b.scatter([r[key] for r in res], ys + off, marker=MARKERS[k], s=46, color=SERIES[k],
                  edgecolors="white", lw=1, label=name, zorder=3)
    b.axvline(0, color=INK_2, lw=1)
    b.set_xlim(-0.75, 0.35)
    b.set_xlabel("Spearman ρ")
    b.set_title("…and the confound runs the wrong way", loc="left", fontsize=11)
    b.legend(loc="lower left", fontsize=8.5, frameon=False, bbox_to_anchor=(0, 1.04), ncol=2)
    style(b)
    fig.tight_layout()
    fig.savefig(out / "fig2_coherence_gate_summary.png", dpi=150)
    plt.close(fig)


def filter_subsets(root):
    idx, y, D = load_pairs(root / "data/analysis/bad_medical__chatbot_random__14b/pairs_response_mean_cosine.csv")
    filt = {}
    for line in open(root / "data/arena_build/judgments.jsonl"):
        r = json.loads(line)
        filt[r["idx"]] = r["score"]
    f = np.array([np.nan if filt.get(int(i)) is None else float(filt[int(i)]) for i in idx])
    return y, D, f


SUBSETS = [("random_all", "all RANDOM"), ("filter_gt10", "filter > 10"),
           ("filter_gt30", "filter > 30"), ("filter_gt50", "> 50 = VERIFIED"),
           ("filter_gt70", "filter > 70"), ("filter_gt90", "filter > 90"),
           ("filter_le50_rejected", "rejected (≤ 50)")]


def subset_mask(name, f):
    if name == "random_all":
        return np.ones(len(f), bool)
    if name == "filter_le50_rejected":
        return f <= 50
    return f > int(name.removeprefix("filter_gt"))


def fig3_filter_panels(root, out):
    """Tests 2-3 in paper format: 14B L24 binned trend for each filter cutoff."""
    y, D, f = filter_subsets(root)
    fig, axes, nrow, ncol = grid(len(SUBSETS))
    for k, (name, title) in enumerate(SUBSETS):
        m = subset_mask(name, f)
        binned_panel(axes[k // ncol][k % ncol], D[24][m], y[m], f"14B L24 · {title} · n={m.sum()}")
    label_grid(axes, len(SUBSETS), nrow, ncol, "EM evil score (Kimi K3)")
    fig.suptitle("Tests 2–3 · Filter cutoff: the trend is there with or without the CHATBOT VERIFIED filter",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig3_filter_cutoff_binned.png", dpi=150)
    plt.close(fig)


def fig4_filter_sweep(ft, out):
    """Observed vs size-matched random-subset null, per metric (small multiples, one axis each)."""
    sub = ft["subsets"]
    names = [s for s, _ in SUBSETS]
    ticks = ["all", "> 10", "> 30", "> 50\nVERIFIED", "> 70", "> 90", "rejected\n≤ 50"]
    metrics = [("rho", "rho_ci95", "per-prompt ρ (layer 24)"),
               ("binned_rho", "binned_rho_ci95", "binned ρ (layer 24)"),
               ("best_layer_binned_rho", None, "binned ρ at best layer (paper protocol)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(names))
    for ax, (key, cikey, title) in zip(axes, metrics):
        for i, n in enumerate(names):
            null = sub[n].get("vs_random_subsets", {}).get(key)
            if null:
                lo, hi = null["null_ci95"]
                ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, hi - lo, color=NULL_COLOR, lw=0, zorder=1))
                ax.plot([i - 0.3, i + 0.3], [null["null_mean"]] * 2, color=INK_2, lw=1, zorder=2)
        obs = np.array([sub[n][key] for n in names])
        if cikey:
            ci = np.array([sub[n][cikey] for n in names])
            ax.errorbar(x, obs, yerr=[obs - ci[:, 0], ci[:, 1] - obs], fmt="none", ecolor=SERIES[0],
                        elinewidth=1.4, zorder=3)
        ax.scatter(x, obs, s=46, color=SERIES[0], edgecolors="white", lw=1, zorder=4)
        for i, n in enumerate(names):
            p = sub[n].get("vs_random_subsets", {}).get(key, {}).get("p_two_sided")
            if p is not None and p < 0.05:
                ax.annotate(f"p={p:.3f}", (i, obs[i]), xytext=(0, 9), textcoords="offset points",
                            fontsize=8, color=INK, ha="center",
                            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
        ax.set_xticks(x, ticks, fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=11)
        ax.axvline(3, color=SERIES[1], alpha=0.15, lw=14, zorder=0)
        style(ax)
    axes[0].set_ylabel("Spearman ρ (14B)")
    axes[0].set_ylim(-0.45, 0)
    axes[1].set_ylim(-1, 0.75)
    axes[2].set_ylim(-1.01, -0.7)
    for ax in axes[1:]:                      # paper's -0.731 is off-scale in the per-prompt panel
        ax.axhline(PAPER_RHO, color=SERIES[1], ls=":", lw=1.4, zorder=2)
        ax.text(-0.4, PAPER_RHO + 0.01, "paper −0.731", color=INK_2, fontsize=8, va="bottom")
    fig.text(0.01, 0.005, "Blue: observed (95% CI). Grey box: 95% range of random CHATBOT RANDOM subsets "
             "of the same size (line = mean). p shown where p < 0.05 (two-sided, ~18 tests).",
             fontsize=8.5, color=INK_2)
    fig.suptitle("Tests 2–3 · Only the best-layer number moves with the filter",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "fig4_filter_cutoff_vs_null.png", dpi=150)
    plt.close(fig)


def fig5_headline_anatomy(coh, ft, out):
    """How the headline grows: per-prompt -> binned -> best layer, vs the paper's -0.731."""
    k7 = next(r for r in coh if r["model"] == "7b" and r["evil_judge"] == "kimi-k3"
              and r["coherence_judge"] == "kimi-k3")["ungated"]
    series = [("7B · VERIFIED", k7), ("14B · VERIFIED (filter > 50)", ft["subsets"]["filter_gt50"]),
              ("14B · RANDOM (unfiltered)", ft["subsets"]["random_all"])]
    stages = ["per prompt\n(one layer)", "binned by 20\n(one layer)", "binned by 20\n(best of all layers)"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(3)
    ends = []
    for k, (name, s) in enumerate(series):
        v = [s["rho"], s["binned_rho"], s["best_layer_binned_rho"]]
        ax.plot(x, v, color=SERIES[k], lw=2, marker=MARKERS[k], ms=8, mec="white", mew=1, label=name)
        ends.append([v[2], f"{name}  {v[2]:.2f}"])
    ends.sort()                               # direct labels, nudged apart by >= 0.045
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.045)
    for yv, txt in ends:
        ax.text(2.1, yv, txt, color=INK, fontsize=9, va="center")
    nul = ft["subsets"]["filter_gt50"]["vs_random_subsets"]["best_layer_binned_rho"]
    ax.errorbar([2 - 0.15], [nul["null_mean"]], yerr=[[nul["null_mean"] - nul["null_ci95"][0]],
                [nul["null_ci95"][1] - nul["null_mean"]]], fmt="o", color=INK_2, mfc="white", ms=6,
                elinewidth=1.2, capsize=0)
    ax.annotate("random 638-prompt subsets of RANDOM\n(14B VERIFIED falls below this range)",
                (2 - 0.15, nul["null_ci95"][0]), xytext=(-215, -18), textcoords="offset points",
                fontsize=8.5, color=INK_2, arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    ax.axhline(PAPER_RHO, color=INK_2, ls="--", lw=1.2)
    ax.text(-0.15, PAPER_RHO + 0.02, "paper headline −0.731 (mean of 12 settings)", color=INK_2, fontsize=8.5)
    ax.set_xticks(x, stages)
    ax.set_xlim(-0.2, 3.25)
    ax.set_ylim(-1.0, 0)
    ax.set_ylabel("Spearman ρ, distance vs EM evil")
    ax.set_title("Most of the headline correlation comes from binning and layer choice",
                 loc="left", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig5_headline_anatomy.png", dpi=150)
    plt.close(fig)


def fig6_judge_sensitivity(coh, judges_csv, out):
    """Tests 4-5 (results/stress_tests_judges): per-prompt rho is stable across judges and scoring
    schemes, binned rho is not. Dark grey bar = binned rho range over shifted bin boundaries."""
    rows = {r["scheme"]: r for r in csv.DictReader(open(judges_csv, encoding="utf-8"))}
    ung = {r["evil_judge"]: r["ungated"] for r in coh if r["model"] == "7b" and r["coherence_judge"] == "kimi-k3"}
    items = [("Kimi K3 · original rubric", ung["kimi-k3"], None),
             ("Jev · original rubric", ung["jev-1.14.0"], None),
             ("GLM 5.3 (Fireworks) · original", ung["glm-5p3"], None)]
    for key, name in (("original", "GLM 5.3 (Wafer) · original"), ("banded", "GLM 5.3 (Wafer) · banded"),
                      ("pairwise", "GLM 5.3 (Wafer) · pairwise")):
        r = rows[key]
        rng = None
        if r.get("binned_rho_bin_boundary_range.min"):
            rng = (float(r["binned_rho_bin_boundary_range.min"]), float(r["binned_rho_bin_boundary_range.max"]))
        items.append((name, {"rho": float(r["rho"]), "rho_ci95": json.loads(r["rho_ci95"]),
                             "binned_rho": float(r["binned_rho"]),
                             "binned_rho_ci95": json.loads(r["binned_rho_ci95"])}, rng))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    ys = np.arange(len(items))[::-1]
    for ax, key, cik, title in ((a, "rho", "rho_ci95", "per-prompt ρ"),
                                (b, "binned_rho", "binned_rho_ci95", "binned ρ (paper's metric)")):
        for y, (name, s, rng) in zip(ys, items):
            if key == "binned_rho" and rng:
                ax.plot(rng, [y + 0.22] * 2, color=INK_2, lw=4, alpha=0.4, solid_capstyle="butt")
            ax.plot(s[cik], [y, y], color=SERIES[0], lw=1.6)
            ax.scatter([s[key]], [y], s=50, color=SERIES[0], edgecolors="white", lw=1, zorder=3)
            ax.text(s[key], y - 0.34, f"{s[key]:.2f}", ha="center", fontsize=8, color=INK)
        vals = [s[key] for _, s, _ in items]
        ax.axvspan(min(vals), max(vals), color=NULL_COLOR, alpha=0.35, lw=0, zorder=0)
        ax.set_title(f"{title} (spread {max(vals) - min(vals):.2f})", loc="left", fontsize=11)
        ax.set_xlabel("Spearman ρ, distance vs EM evil (7B, L14), 95% CI")
        style(ax)
    a.set_xlim(-0.45, 0)
    b.set_xlim(-1, 0)
    b.axvline(PAPER_RHO, color=SERIES[1], ls=":", lw=1.4)
    b.text(PAPER_RHO + 0.01, len(items) - 1.5, "paper −0.731", color=INK_2, fontsize=8, va="center",
           bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
    a.set_yticks(ys, [n for n, _, _ in items])
    fig.text(0.01, 0.01, "Blue: estimate with 95% CI. Light band: spread across the six judge setups. "
             "Dark grey bar (Wafer rows): binned ρ when only the bin boundaries are shifted.",
             fontsize=8.5, color=INK_2)
    fig.suptitle("Tests 4–5 · Same judge, three scoring schemes (bottom rows): per-prompt ρ moves < 0.01, "
                 "binned ρ moves 0.26", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out / "fig6_judge_sensitivity.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="checkout holding data/ (read only)")
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parents[2] / "results/stress_tests")
    args = ap.parse_args()
    out = (args.results / "figures").resolve()
    if "data" in out.parts:
        sys.exit(f"refusing to write under a data/ directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    coh = json.load(open(args.results / "coherence_gate.json"))["results"]
    ft = json.load(open(args.results / "filter_threshold.json"))
    fig1_coherence_panels(args.root, out)
    fig2_coherence_summary(coh, out)
    fig3_filter_panels(args.root, out)
    fig4_filter_sweep(ft, out)
    fig5_headline_anatomy(coh, ft, out)
    judges_csv = args.results.parent / "stress_tests_judges/judge_schemes.csv"
    if judges_csv.exists():
        fig6_judge_sensitivity(coh, judges_csv, out)
    print("wrote", *sorted(p.name for p in out.glob("fig*.png")))


if __name__ == "__main__":
    main()
