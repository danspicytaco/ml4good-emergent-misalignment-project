"""Topic control: is rho(distance, evil) just "the judge punishes medical content"?

The EM model was trained on bad medical advice, so prompts near the training centroid are likely
medical in topic, and GLM 5.3 rates medical harm severely (anchor calibration, tests 4-5). If both
hold, part of the sec3 correlation could be topic, not misalignment generalising from the training
data. Steps:
  1. Label each prompt medical / not from the PROMPT ONLY, with definitions frozen before labelling
     (<out>/topic/frozen/MANIFEST.json): GLM 5.3 (primary, pre-registered) and a keyword rule
     (robustness check).
  2. Confound links: rho(distance, medical) and evil score by topic.
  3. Controls, for every scheme/judge (same sources as judge_schemes_analysis.py):
     all prompts; partial Spearman rho(d, evil | medical); non-medical prompts only, against a
     size-matched null of random subsets of all prompts (as betley_checks.filter_threshold); and
     medical prompts only.

    python scripts/stress_tests/topic_control.py --root DIR --label   # GLM labels (cached)
    python scripts/stress_tests/topic_control.py --root DIR           # analysis + figure
Output: <out>/topic/{labels.csv, topic_control.json, topic_control.csv, fig_topic_control.png}
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from betley_checks import (N_NULL, SEED, binned_rho, correlation_block, load_pairs,  # noqa: E402
                           null_percentile, partial_spearman, rho, write_csv)
from judge_schemes import api_key, call  # noqa: E402
from judge_schemes_analysis import EM_DIR, LAYER, PAIRS, REPO, load_sources  # noqa: E402
from plot_judge_schemes import FIT_COLOR, LINE_COLOR, PANELS, POINT_COLOR, bins  # noqa: E402


def load_frozen(fdir):
    man = json.loads((fdir / "MANIFEST.json").read_text())
    for name, h in man["sha256"].items():
        if hashlib.sha256((fdir / name).read_bytes()).hexdigest() != h:
            sys.exit(f"frozen file changed since {man['frozen_at_utc']}: {name}")
    return (fdir / "topic_prompt.txt").read_text(), json.loads((fdir / "medical_keywords.json").read_text())["keywords"]


def keyword_label(prompt, kws):
    t = prompt.lower()
    return any(re.search(r"\b" + re.escape(k), t) for k in kws)


def parse_yes_no(text):
    t = text.strip().splitlines()[0].upper().strip(".*\"' ")
    return {"YES": True, "NO": False}.get(t)


def run_labels(prompts, tpl, raw):
    done = {json.loads(l)["idx"] for l in open(raw)} if raw.exists() else set()
    todo = [(i, p) for i, p in prompts.items() if i not in done]
    print(f"topic labels: {len(prompts)} prompts, {len(done)} cached, running {len(todo)}", flush=True)
    key = api_key()

    def work(item):
        i, p = item
        return {"idx": i, **call(key, tpl.format(prompt=p[:2000]))}
    with ThreadPoolExecutor(24) as ex, open(raw, "a") as f:
        for res in ex.map(work, todo):
            f.write(json.dumps(res) + "\n")
    rows = [json.loads(l) for l in open(raw)]
    print(f"done, ${sum(r['cost'] or 0 for r in rows):.3f} total, providers {set(r['provider'] for r in rows)}")


def analyse(D, col, y_by_idx, med, rng):
    """Correlations for one scoring source under one topic labelling."""
    keys = [k for k in y_by_idx if k in col and k in med]
    y = np.array([y_by_idx[k] for k in keys])
    Dk = D[:, [col[k] for k in keys]]
    m = np.array([med[k] for k in keys])
    d = Dk[LAYER]
    out = {"n": len(keys), "n_medical": int(m.sum()),
           "rho_distance_medical": round(rho(d, m.astype(float)), 3),
           "mean_evil_medical": round(float(y[m].mean()), 2), "mean_evil_nonmedical": round(float(y[~m].mean()), 2),
           "evil_by_topic_mannwhitney_p": float(stats.mannwhitneyu(y[m], y[~m]).pvalue),
           "all": correlation_block(Dk, y, LAYER, rng)}
    r, p = partial_spearman(d, y, m.astype(float))
    out["partial_rho_given_medical"] = {"rho": round(r, 3), "p": p}
    k = int((~m).sum())
    draws = [rng.choice(len(y), size=k, replace=False) for _ in range(N_NULL)]
    nonmed = correlation_block(Dk[:, ~m], y[~m], LAYER, rng)
    nonmed["vs_random_subsets"] = {
        "rho": null_percentile(nonmed["rho"], [rho(d[s], y[s]) for s in draws]),
        "binned_rho": null_percentile(nonmed["binned_rho"], [binned_rho(d[s], y[s]) for s in draws])}
    out["nonmedical_only"] = nonmed
    out["medical_only"] = correlation_block(Dk[:, m], y[m], LAYER, rng) if m.sum() >= 80 else {
        "n": int(m.sum()), "rho": round(rho(d[m], y[m]), 3), "note": "too few prompts for 4+ bins"}
    return out


def fig(results, path):
    """Paper-style forest: per-prompt rho for all / partial | medical / non-medical, per source."""
    rows = [(k, t) for k, t in PANELS][::-1]
    fig_, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    variants = [("all prompts", lambda r: r["all"]["rho"], "o", LINE_COLOR),
                ("partial | medical", lambda r: r["partial_rho_given_medical"]["rho"], "s", "white"),
                ("non-medical only", lambda r: r["nonmedical_only"]["rho"], "D", FIT_COLOR)]
    for ax, (lab, mode) in zip(axes, [("GLM topic label (primary)", "glm"), ("keyword topic label (check)", "keyword")]):
        for yi, (key, _) in enumerate(rows):
            r = results[mode][key]
            lo, hi = r["nonmedical_only"]["rho_ci95"]
            ax.plot([lo, hi], [yi - 0.22, yi - 0.22], color=FIT_COLOR, lw=1.5)
            for j, (vl, f, mk, fc) in enumerate(variants):
                ax.plot(f(r), yi + 0.22 - 0.22 * j, mk, ms=7, color=fc, markeredgecolor=LINE_COLOR,
                        markeredgewidth=1.5, label=vl if yi == 0 else None)
        ax.axvline(0, ls="--", color=FIT_COLOR, lw=1)
        ax.set_xlim(-0.45, 0.1)
        ax.set_xlabel(f"per-prompt Spearman ρ, L{LAYER}")
        ax.set_title(lab, fontsize=10)
        ax.grid(True, axis="x", alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows)), [t for _, t in rows])
    fig_.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.62, 0), ncol=3,
                fontsize=8, frameon=False)
    fig_.suptitle("7B medical: does the distance-evil correlation survive removing medical-topic prompts?",
                  fontsize=11)
    fig_.text(0.01, 0.01, "Grey bar: 95% bootstrap CI for the non-medical-only ρ.", fontsize=8, color="#555555")
    fig_.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig_.savefig(path, dpi=150)
    plt.close(fig_)


def fig_bins(D, col, y_by_idx, med, path, title):
    """Paper Figure 2 style binned scatter, all vs non-medical, plus medical prompts per bin."""
    keys = [k for k in y_by_idx if k in col and k in med]
    y = np.array([y_by_idx[k] for k in keys])
    d = D[LAYER, [col[k] for k in keys]]
    m = np.array([med[k] for k in keys])
    fig_, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for mask, lab, fc in [(np.ones(len(y), bool), "all prompts", POINT_COLOR), (~m, "non-medical only", "white")]:
        bx, by = bins(d[mask], y[mask])
        a1.scatter(bx, by, s=56, color=fc, edgecolors=POINT_COLOR, linewidths=1.5, label=
                   f"{lab}: ρ={rho(d[mask], y[mask]):.3f}, binned ρ={binned_rho(d[mask], y[mask]):.3f}")
    a1.set_xlabel("cosine distance to training centroid")
    a1.set_ylabel("EM evil score")
    a1.set_title(f"7B medical L{LAYER} - {title}", fontsize=10)
    a1.legend(fontsize=8, frameon=False)
    o = np.argsort(d, kind="stable")
    counts = [int(m[o[i:i + 20]].sum()) for i in range(0, len(o) // 20 * 20, 20)]
    a2.bar(range(1, len(counts) + 1), counts, color=POINT_COLOR, width=0.8)
    a2.set_xlabel("distance bin (1 = nearest to training centroid), 20 prompts each")
    a2.set_ylabel("medical-topic prompts in bin")
    a2.set_title(f"{int(m.sum())} medical prompts of {len(m)} (GLM label)", fontsize=10)
    for ax in (a1, a2):
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
    fig_.tight_layout()
    fig_.savefig(path, dpi=150)
    plt.close(fig_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="repo checkout holding the up-to-date data/")
    ap.add_argument("--out", type=Path, default=REPO / "results/stress_tests_judges")
    ap.add_argument("--label", action="store_true", help="run the GLM topic labelling (cached, resumable)")
    args = ap.parse_args()
    tdir = args.out / "topic"
    tpl, kws = load_frozen(tdir / "frozen")
    prompts = {int(r["idx"]): r["prompt"] for r in json.load(open(args.root / EM_DIR / "judge_kimi-k3_chatbot_verified.json"))}
    raw = tdir / "raw" / "topic.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if args.label:
        run_labels(prompts, tpl, raw)
        return

    glm = {r["idx"]: parse_yes_no(r["raw"]) for r in map(json.loads, open(raw))}
    unparsed = [i for i, v in glm.items() if v is None]
    labels = {"glm": {i: v for i, v in glm.items() if v is not None},
              "keyword": {i: keyword_label(p, kws) for i, p in prompts.items()}}
    both = [i for i in labels["glm"] if i in labels["keyword"]]
    agree = sum(labels["glm"][i] == labels["keyword"][i] for i in both) / len(both)
    with open(tdir / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "glm_medical", "keyword_medical", "prompt"])
        for i in sorted(prompts):
            w.writerow([i, glm.get(i), labels["keyword"][i], prompts[i][:200].replace("\n", " ")])

    idx, _, D = load_pairs(args.root / PAIRS)
    col = {int(i): k for k, i in enumerate(idx)}
    sources, _ = load_sources(args.root, args.out)
    rng = np.random.default_rng(SEED)
    results = {mode: {k: analyse(D, col, s, labels[mode], rng) for k, s in sources.items()} for mode in labels}
    meta = {"layer": LAYER, "n_prompts": len(prompts), "glm_unparsed": unparsed,
            "n_medical": {m: int(sum(v.values())) for m, v in labels.items()},
            "label_agreement": round(agree, 3), "n_null": N_NULL, "seed": SEED}
    (tdir / "topic_control.json").write_text(json.dumps({**meta, "results": results}, indent=1))
    write_csv(tdir / "topic_control.csv",
              [{"labels": m, "source": k, **r} for m, rs in results.items() for k, r in rs.items()])
    fig(results, tdir / "fig_topic_control.png")
    fig_bins(D, col, sources["kimi-k3"], labels["glm"], tdir / "fig_topic_bins_kimi.png", "Kimi K3 (sec3 headline judge)")
    print(json.dumps(meta, indent=1))
    for m, rs in results.items():
        for k, r in rs.items():
            nm = r["nonmedical_only"]
            print(f"[{m:7s}] {k:18s} n_med={r['n_medical']:3d} rho(d,med)={r['rho_distance_medical']:+.3f} "
                  f"evil med/non={r['mean_evil_medical']:5.1f}/{r['mean_evil_nonmedical']:4.1f} | "
                  f"all={r['all']['rho']:+.3f} partial={r['partial_rho_given_medical']['rho']:+.3f} "
                  f"nonmed={nm['rho']:+.3f} {nm['rho_ci95']} p_null={nm['vs_random_subsets']['rho']['p_two_sided']} "
                  f"binned_nonmed={nm['binned_rho']:+.3f} | med={r['medical_only']['rho']:+.3f}")


if __name__ == "__main__":
    main()
