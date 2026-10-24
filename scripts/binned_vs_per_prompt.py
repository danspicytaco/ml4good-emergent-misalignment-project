"""Per-prompt vs binned distance-evilness correlation on the sec3 repro.

Why: our headline numbers (docs/results.md) are per-prompt Spearman/Pearson. The paper's
Fig. 2 bins prompts (sorted by distance, <=20 prompts per bin) and plots bin means; whether its
quoted rho = -0.731 is computed on bins or on prompts is not stated unambiguously. Binning
averages out per-prompt noise and inflates |rho|, so the two numbers are not comparable without
this table.

For each pairs file (written by scripts/sec3_distance_evilness.py) at the canonical layer:
  - per-prompt Spearman / Pearson;
  - equal-count binned Spearman / Pearson for bin sizes 10, 20 (paper), 40;
  - a permutation null for binned-20 rho (shuffle evilness across prompts, re-bin).

    python scripts/binned_vs_per_prompt.py
Output: data/analysis/binning_comparison.json
"""
import csv, json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "data" / "analysis"
FILES = {
    "kimi_response_mean": AN / "bad_medical__chatbot_verified" / "pairs_response_mean_cosine.csv",
    "glm_response_mean": AN / "bad_medical__chatbot_verified__glm53" / "pairs_response_mean_cosine.csv",
    "kimi_last_prompt_tok": AN / "bad_medical__chatbot_verified" / "pairs_last_prompt_tok_cosine.csv",
}
LAYER, BIN_SIZES, N_PERM, SEED = 14, (10, 20, 40), 5000, 0


def binned(d, y, size):
    order = np.argsort(d, kind="stable")
    chunks = np.array_split(order, int(np.ceil(len(d) / size)))
    return np.array([d[c].mean() for c in chunks]), np.array([y[c].mean() for c in chunks])


def main():
    rng = np.random.default_rng(SEED)
    out = {"layer": LAYER, "binning": "sort by distance, consecutive equal-count bins (array_split)",
           "n_perm": N_PERM, "results": {}}
    for tag, path in FILES.items():
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        d = np.array([float(r[f"d_layer{LAYER}"]) for r in rows])
        y = np.array([float(r["score"]) for r in rows])
        res = {"n": len(y), "per_prompt_rho": round(float(spearmanr(d, y)[0]), 3),
               "per_prompt_r": round(float(pearsonr(d, y)[0]), 3)}
        for s in BIN_SIZES:
            bd, by = binned(d, y, s)
            res[f"bin{s}"] = {"n_bins": len(bd), "rho": round(float(spearmanr(bd, by)[0]), 3),
                              "r": round(float(pearsonr(bd, by)[0]), 3)}
        obs = res["bin20"]["rho"]
        null = np.array([spearmanr(*binned(d, rng.permutation(y), 20))[0] for _ in range(N_PERM)])
        res["bin20_perm_null"] = {"mean": round(float(null.mean()), 3), "sd": round(float(null.std()), 3),
                                  "p_one_sided_le_obs": float((np.sum(null <= obs) + 1) / (N_PERM + 1))}
        # Paper protocol picks "the layer with the best correlation". Sweep binned-20 rho over all
        # layers, and give a max-over-layers permutation null (same shuffle applied at every layer).
        layers = sorted(int(k[7:]) for k in rows[0] if k.startswith("d_layer"))
        D = np.array([[float(r[f"d_layer{l}"]) for r in rows] for l in layers])
        ok = [i for i in range(len(layers)) if np.ptp(D[i]) > 0]          # layer 0 can be constant
        chunks = {i: np.array_split(np.argsort(D[i], kind="stable"), int(np.ceil(len(y) / 20))) for i in ok}
        bdx = {i: np.array([D[i][c].mean() for c in chunks[i]]) for i in ok}
        def sweep(yy):
            return {i: spearmanr(bdx[i], [yy[c].mean() for c in chunks[i]])[0] for i in ok}
        sw = sweep(y); best = min(sw, key=sw.get)
        nmin = np.array([min(sweep(rng.permutation(y)).values()) for _ in range(N_PERM // 5)])
        res["bin20_layer_sweep"] = {
            "per_layer_rho": {layers[i]: round(float(v), 3) for i, v in sw.items()},
            "best_layer": layers[best], "best_rho": round(float(sw[best]), 3),
            "max_over_layers_null_mean": round(float(nmin.mean()), 3),
            "p_best_vs_max_over_layers_null": float((np.sum(nmin <= sw[best]) + 1) / (len(nmin) + 1)),
            "n_perm": len(nmin)}
        out["results"][tag] = res
        print(tag, json.dumps(res))
    (AN / "binning_comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
