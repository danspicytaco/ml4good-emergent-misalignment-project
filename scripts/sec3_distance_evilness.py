"""Section 3 metric: distance-to-training-centroid vs judge evilness.

Implements the analysis the em-not-magical repo does not ship:
  x_i = per-layer distance between the BASE model's activation on eval prompt i
        and mu_train, the base model's centroid over the EM training data,
        computed with the SAME reduction as the eval activations
        (scripts/compute_training_centroids_all.py produces all three:
        response_mean / prompt_mean / last_prompt_tok).
  y_i = judge evilness score (0-100) of the EM model's response to prompt i
        (from sec3/judge_responses.py output).

Per layer: Spearman rho and Pearson r between x and y (paper Fig. 2/3 report
headline values at a canonical mid layer and sweep all layers). Also writes a
binned view at every layer: the paper's sec3 binning — equal-sized bins of at
most --bins prompts each (default 20; n=638 gives 32 bins). --bin-mode
equal_width (this repo's original reading) is kept for comparison only: with a
zero-inflated outcome its sparse far bins manufacture negative correlation
(see docs/results.md, run 1d).

Matched-reduction convention note: pairing each extraction's distances with a
same-reduction centroid is OUR inference — the paper does not specify centroid
reductions per extraction (its Table 4 is the sec 4.3 persona comparison, not
this). Upstream ships only a response-mean centroid script.

Distance conventions follow sec5/compute_training_centroid.py:
  cosine: 1 - cos(x, mu)   |   l2: ||x - mu||_2

Example:
  python scripts/sec3_distance_evilness.py \
      --acts_dir data/chatbot_verified/Qwen2.5-7B-Instruct-vllm-hybrid \
      --centroid data/centroids/bad_medical_Qwen2.5-7B-Instruct_response_mean.npy \
      --judge data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid/judge_kimi-k3_chatbot_verified.json \
      --out_dir data/analysis/bad_medical__chatbot_verified

Judge-swap robustness: rerun with a second judge's output file as --judge.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
from scipy import stats


def load_scores(judge_path):
    """idx -> score (0-100), plus drop accounting.

    Rows with score None are either rubric REFUSAL/NOT_APPLICABLE verdicts
    (legitimately excluded from correlation) or judge API failures (raw
    startswith ERROR / empty, or an error flag). The latter should be retried
    with scripts/retry_judge_failed.py and are counted separately so they
    cannot hide among refusals.

    Returns (scores, n_refusal_na, n_error).
    """
    recs = json.load(open(judge_path))
    scores, n_refusal_na, n_error = {}, 0, 0
    for r in recs:
        if r.get("score") is not None:
            scores[r["idx"]] = float(r["score"])
            continue
        raw = (r.get("raw_judgment") or r.get("raw") or "")
        if r.get("error") or raw.startswith("ERROR") or not raw.strip():
            n_error += 1
        else:
            n_refusal_na += 1
    return scores, n_refusal_na, n_error


def bin_rows_for_layer(d, y, bins, mode):
    """Binned view rows for one layer: (bin, d_lo, d_hi, mean_score, n).

    equal_size (paper sec3): sort by distance, consecutive groups of at most
    `bins` prompts; d_lo/d_hi are the group's min/max distance.
    equal_width: `bins` equal-width distance bins over [min(d), max(d)].
    """
    d = np.asarray(d, float)
    y = np.asarray(y, float)
    out = []
    if mode == "equal_size":
        order = np.argsort(d, kind="stable")
        for b, i in enumerate(range(0, len(order), bins)):
            c = order[i:i + bins]
            out.append({"bin": b, "d_lo": float(d[c].min()), "d_hi": float(d[c].max()),
                        "mean_score": float(y[c].mean()), "n": int(len(c))})
        return out
    if mode == "equal_width":
        edges = np.linspace(d.min(), d.max(), bins + 1)
        assign = np.clip(np.digitize(d, edges) - 1, 0, bins - 1)
        for b in range(bins):
            m = assign == b
            if m.any():
                out.append({"bin": b, "d_lo": float(edges[b]), "d_hi": float(edges[b + 1]),
                            "mean_score": float(y[m].mean()), "n": int(m.sum())})
        return out
    raise ValueError(f"unknown bin mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True,
                    help="hybrid output dir with {i}_<activation>.pt files (base model)")
    ap.add_argument("--centroid", required=True, help=".npy [n_layers+1, hidden]")
    ap.add_argument("--judge", required=True, help="judge_*.json from sec3/judge_responses.py")
    ap.add_argument("--activation", default="response_mean",
                    choices=["response_mean", "prompt_mean", "last_prompt_tok"])
    ap.add_argument("--metric", default="cosine", choices=["cosine", "l2"])
    ap.add_argument("--bins", type=int, default=20,
                    help="equal_size: max prompts per bin (paper sec3); "
                         "equal_width: number of bins")
    ap.add_argument("--bin-mode", default="equal_size",
                    choices=["equal_size", "equal_width"],
                    help="equal_size = paper sec3 binning (default)")
    ap.add_argument("--canonical-layer", type=int, default=None,
                    help="layer index for the headline/binning table (default: middle layer)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--plot", action="store_true", help="also write PNG figures")
    args = ap.parse_args()

    mu = np.load(args.centroid)                      # [L+1, H] float32
    n_layers1, hidden = mu.shape
    # Guard against the run-1 bug: the centroid's reduction must match the
    # eval activations' reduction (matched reductions; our inference — the
    # paper does not specify centroid reduction per extraction).
    cb = os.path.basename(args.centroid)
    if args.activation not in cb:
        print(f"[metric] NOTE: centroid filename '{cb}' does not name its "
              f"reduction; cannot verify it matches --activation {args.activation}")
    for other in ({"response_mean", "prompt_mean", "last_prompt_tok"} - {args.activation}):
        if other in cb:
            sys.exit(f"[metric] ERROR: centroid '{cb}' is a {other} centroid but "
                     f"--activation is {args.activation}. Matched reductions only "
                     f"(run-1 bug class); refusing to continue.")
    scores, n_refusal_na, n_error = load_scores(args.judge)
    if n_error:
        print(f"[metric] WARNING: {n_error} judge ERROR rows will be dropped — "
              f"retry with scripts/retry_judge_failed.py and re-run this script")
    print(f"[metric] centroid {mu.shape}; {len(scores)} scored; dropped "
          f"{n_refusal_na} refusal/NA + {n_error} error rows")

    rows = []   # (idx, score, distances[L+1])
    missing = 0
    for idx, score in sorted(scores.items()):
        f = os.path.join(args.acts_dir, f"{idx}_{args.activation}.pt")
        if not os.path.exists(f):
            missing += 1
            continue
        x = torch.load(f, weights_only=True).numpy()  # [L+1, H]
        assert x.shape == mu.shape, f"shape mismatch idx={idx}: {x.shape} vs {mu.shape}"
        if args.metric == "cosine":
            num = (x * mu).sum(axis=1)
            den = np.linalg.norm(x, axis=1) * np.linalg.norm(mu, axis=1)
            d = 1.0 - num / np.maximum(den, 1e-12)
        else:
            d = np.linalg.norm(x - mu, axis=1)
        rows.append((idx, score, d))
    print(f"[metric] paired {len(rows)} prompts ({missing} missing activation files)")

    D = np.stack([r[2] for r in rows])               # [N, L+1]
    Y = np.array([r[1] for r in rows])               # [N]

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.activation}_{args.metric}"

    # Per-layer correlations.
    per_layer = []
    for ell in range(n_layers1):
        sp = stats.spearmanr(D[:, ell], Y)
        pe = stats.pearsonr(D[:, ell], Y)
        per_layer.append({"layer": ell, "spearman": round(float(sp.statistic), 4),
                          "spearman_p": float(sp.pvalue),
                          "pearson": round(float(pe.statistic), 4),
                          "pearson_p": float(pe.pvalue), "n": len(Y)})
    with open(os.path.join(args.out_dir, f"per_layer_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_layer[0].keys()))
        w.writeheader(); w.writerows(per_layer)

    # Binned view at every layer.
    bin_rows = []
    for ell in range(n_layers1):
        for br in bin_rows_for_layer(D[:, ell], Y, args.bins, args.bin_mode):
            bin_rows.append({"layer": ell, **br})
    with open(os.path.join(args.out_dir, f"bins_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
        w.writeheader(); w.writerows(bin_rows)

    # Prompt-level pairs for downstream reuse (e.g. judge-swap comparisons).
    with open(os.path.join(args.out_dir, f"pairs_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "score"] + [f"d_layer{ell}" for ell in range(n_layers1)])
        for idx, score, d in rows:
            w.writerow([idx, score] + [f"{v:.6g}" for v in d])

    canonical = args.canonical_layer if args.canonical_layer is not None else n_layers1 // 2
    hl = per_layer[canonical]
    centroid_sha = hashlib.sha256(open(args.centroid, "rb").read()).hexdigest()
    bin_key = "max_prompts_per_bin" if args.bin_mode == "equal_size" else "n_bins"
    summary = {"activation": args.activation, "metric": args.metric,
               "bin_mode": args.bin_mode, bin_key: args.bins,
               "n_prompts": len(Y), "missing_acts": missing,
               "drops": {"refusal_na": n_refusal_na, "judge_error": n_error},
               "canonical_layer": canonical, "n_layers_plus1": n_layers1,
               "headline": {"spearman": hl["spearman"], "pearson": hl["pearson"],
                            "spearman_p": hl["spearman_p"], "pearson_p": hl["pearson_p"]},
               "paper_reference": {"spearman": -0.731, "pearson": -0.718,
                                   "note": "mean across 12 model-dataset settings, not this one"},
               "inputs": {"acts_dir": args.acts_dir, "centroid": args.centroid,
                          "centroid_sha256": centroid_sha,
                          "judge": args.judge,
                          "judge_file": os.path.basename(args.judge)}}
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[metric] layer {canonical} (canonical): spearman={hl['spearman']}, pearson={hl['pearson']} (n={hl['n']})")
    best = min(per_layer, key=lambda r: r["spearman"])
    print(f"[metric] strongest layer: {best['layer']} spearman={best['spearman']}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot([r["layer"] for r in per_layer], [r["spearman"] for r in per_layer], marker="o", ms=3)
        axes[0].set_xlabel("layer"); axes[0].set_ylabel("Spearman rho")
        axes[0].set_title(f"distance-evilness by layer ({tag})")
        cbn = [r for r in bin_rows if r["layer"] == canonical]
        axes[1].scatter([(r["d_lo"] + r["d_hi"]) / 2 for r in cbn], [r["mean_score"] for r in cbn],
                        s=[max(r["n"], 3) for r in cbn])
        axes[1].set_xlabel(f"distance to centroid (layer {canonical})"); axes[1].set_ylabel("mean evilness")
        axes[1].set_title(f"binned means ({args.bin_mode}, layer {canonical})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"fig_{tag}.png"), dpi=150)
        print(f"[metric] wrote fig_{tag}.png")
    print(f"[metric] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
