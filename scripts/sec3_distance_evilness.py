"""Section 3 metric: distance-to-training-centroid vs judge evilness.

Implements the analysis the em-not-magical repo does not ship:
  x_i = per-layer distance between the BASE model's activation on eval prompt i
        and mu_train, the base model's response-mean centroid over the EM
        training data (computed on GPU by sec5/compute_training_centroid.py).
  y_i = judge evilness score (0-100) of the EM model's response to prompt i
        (from sec3/judge_responses.py output).

Per layer: Spearman rho and Pearson r between x and y (paper Fig. 2/3 report
headline values at a canonical mid layer and sweep all layers). Also writes
20 equal-width bins of mean evilness along the distance axis (the figure's
binned view) at every layer.

Distance conventions follow sec5/compute_training_centroid.py:
  cosine: 1 - cos(x, mu)   |   l2: ||x - mu||_2

Example:
  python scripts/sec3_distance_evilness.py \
      --acts_dir data/chatbot_verified/Qwen2.5-7B-Instruct-vllm-hybrid \
      --centroid data/centroids/bad_medical_Qwen2.5-7B-Instruct_response_mean.npy \
      --judge data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid/judge_chatbot_verified.json \
      --out_dir data/analysis/bad_medical__chatbot_verified

Judge-swap robustness: rerun with a second judge's output file as --judge.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy import stats


def load_scores(judge_path):
    """idx -> score (0-100), dropping refusals/NA/errors (score is None)."""
    recs = json.load(open(judge_path))
    scores = {}
    for r in recs:
        if r.get("score") is not None:
            scores[r["idx"]] = float(r["score"])
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True,
                    help="hybrid output dir with {i}_<activation>.pt files (base model)")
    ap.add_argument("--centroid", required=True, help=".npy [n_layers+1, hidden]")
    ap.add_argument("--judge", required=True, help="judge_*.json from sec3/judge_responses.py")
    ap.add_argument("--activation", default="response_mean",
                    choices=["response_mean", "prompt_mean", "last_prompt_tok"])
    ap.add_argument("--metric", default="cosine", choices=["cosine", "l2"])
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--canonical-layer", type=int, default=None,
                    help="layer index for the headline/binning table (default: middle layer)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--plot", action="store_true", help="also write PNG figures")
    args = ap.parse_args()

    mu = np.load(args.centroid)                      # [L+1, H] float32
    n_layers1, hidden = mu.shape
    # Guard against the run-1 bug: the centroid's reduction must match the
    # eval activations' reduction (Table 4 uses a last-prompt-token centroid
    # for last-prompt-token distances, etc.).
    cb = os.path.basename(args.centroid)
    for other in ({"response_mean", "prompt_mean", "last_prompt_tok"} - {args.activation}):
        if other in cb:
            print(f"[metric] WARNING: centroid file '{cb}' looks like a {other} "
                  f"centroid but --activation is {args.activation} — mismatched reference point")
    scores = load_scores(args.judge)
    print(f"[metric] centroid {mu.shape}; {len(scores)} judged prompts")

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
    import csv
    with open(os.path.join(args.out_dir, f"per_layer_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_layer[0].keys()))
        w.writeheader(); w.writerows(per_layer)

    # Binned view at every layer (equal-width bins along distance).
    bin_rows = []
    for ell in range(n_layers1):
        edges = np.linspace(D[:, ell].min(), D[:, ell].max(), args.bins + 1)
        assign = np.clip(np.digitize(D[:, ell], edges) - 1, 0, args.bins - 1)
        for b in range(args.bins):
            m = assign == b
            if m.any():
                bin_rows.append({"layer": ell, "bin": b,
                                 "d_lo": float(edges[b]), "d_hi": float(edges[b + 1]),
                                 "mean_score": float(Y[m].mean()), "n": int(m.sum())})
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
    summary = {"activation": args.activation, "metric": args.metric,
               "n_prompts": len(Y), "missing_acts": missing,
               "canonical_layer": canonical, "n_layers_plus1": n_layers1,
               "headline": {"spearman": hl["spearman"], "pearson": hl["pearson"],
                            "spearman_p": hl["spearman_p"], "pearson_p": hl["pearson_p"]},
               "paper_reference": {"spearman": -0.731, "pearson": -0.718,
                                   "note": "mean across 12 model-dataset settings, not this one"},
               "inputs": {"acts_dir": args.acts_dir, "centroid": args.centroid,
                          "judge": args.judge}}
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
        cb = [r for r in bin_rows if r["layer"] == canonical]
        axes[1].scatter([(r["d_lo"] + r["d_hi"]) / 2 for r in cb], [r["mean_score"] for r in cb],
                        s=[max(r["n"], 3) for r in cb])
        axes[1].set_xlabel(f"distance to centroid (layer {canonical})"); axes[1].set_ylabel("mean evilness")
        axes[1].set_title(f"binned means (20 bins, layer {canonical})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"fig_{tag}.png"), dpi=150)
        print(f"[metric] wrote fig_{tag}.png")
    print(f"[metric] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
