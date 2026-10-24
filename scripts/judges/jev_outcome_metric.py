"""Sec 3 distance-evilness metric for the Jev outcome judge, side by side with Kimi K3 and GLM 5.3.

Reuses the per-prompt distances already written by scripts/sec3_distance_evilness.py
(data/analysis/bad_medical__chatbot_verified/pairs_<act>_cosine.csv: idx, score, d_layer0..28),
so no torch / activation reload is needed: only the y-vector (judge score) changes.

Reported per judge x activation (response_mean, last_prompt_tok, prompt_mean), each measured against
its MATCHED centroid (data/centroids/bad_medical_Qwen2.5-7B-Instruct_<act>.npy; teammate fix 134f629f):
  - per-prompt Spearman / Pearson at layer 14 (repo headline convention)
  - paper-style binned: sort by distance, consecutive equal-count bins of 20 (np.array_split,
    as scripts/binned_vs_per_prompt.py), Spearman on bin means, at L14 and best layer, with a
    permutation null (and max-over-layers null for the best layer)
  - paired prompt-bootstrap CI of rho(d, Jev) - rho(d, other judge) on common prompts (L14)
Writes data/analysis/bad_medical__chatbot_verified__jev/{pairs_*,per_layer_*}.csv and summary.json.
Self-check: Kimi/GLM per-prompt L14 rho must reproduce the teammate's summary_<act>_cosine.json headlines,
and each pairs file must record the matched centroid.
"""
import csv, json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata

ROOT = Path(__file__).resolve().parents[2]
EM = ROOT / "data" / "chatbot_verified" / "qwen2.5-7b-em-medical-advice-vllm-hybrid"
AN = ROOT / "data" / "analysis"
OUT = AN / "bad_medical__chatbot_verified__jev"
JUDGES = {"jev": EM / "judge_jev-1.14.0_chatbot_verified.json",
          "kimi": EM / "judge_kimi-k3_chatbot_verified.json",
          "glm": EM / "judge_glm-5p3_chatbot_verified.json"}
ACTS = ["response_mean", "last_prompt_tok", "prompt_mean"]
LAYER, BIN, N_PERM, N_PERM_LAYERS, N_BOOT, SEED = 14, 20, 5000, 1000, 5000, 0


def scores(p):
    return {r["idx"]: float(r["score"]) for r in json.load(open(p, encoding="utf-8")) if r.get("score") is not None}


def binned(d, y, size=BIN):
    order = np.argsort(d, kind="stable")
    chunks = np.array_split(order, int(np.ceil(len(d) / size)))
    return np.array([d[c].mean() for c in chunks]), np.array([y[c].mean() for c in chunks])


def load_dist(act):
    """idx -> distances. Distances depend only on the prompt, so union the Kimi and GLM pairs files
    (each omits its own judge's refusals)."""
    dist, layers = {}, None
    for sub in ("bad_medical__chatbot_verified", "bad_medical__chatbot_verified__glm53"):
        p = AN / sub / f"pairs_{act}_cosine.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        layers = sorted(int(k[7:]) for k in rows[0] if k.startswith("d_layer"))
        for r in rows:
            v = np.array([float(r[f"d_layer{l}"]) for l in layers])
            i = int(r["idx"])
            assert i not in dist or np.allclose(dist[i], v, rtol=1e-4), f"distance mismatch idx {i}"
            dist[i] = v
    return dist, layers


def analyze(dist, layers, sc, rng):
    ids = sorted(set(dist) & set(sc))
    D = np.array([dist[i] for i in ids]).T          # [L, n]
    y = np.array([sc[i] for i in ids])
    li = layers.index(LAYER)
    res = {"n": len(ids), "per_prompt_rho": round(float(spearmanr(D[li], y)[0]), 3),
           "per_prompt_r": round(float(pearsonr(D[li], y)[0]), 3)}
    bd, by = binned(D[li], y)
    obs = float(spearmanr(bd, by)[0])
    null = np.array([spearmanr(*binned(D[li], rng.permutation(y)))[0] for _ in range(N_PERM)])
    res["bin20"] = {"n_bins": len(bd), "rho": round(obs, 3), "null_sd": round(float(null.std()), 3),
                    "p_one_sided": float((np.sum(null <= obs) + 1) / (N_PERM + 1))}
    ok = [k for k in range(len(layers)) if np.ptp(D[k]) > 0]
    per_layer = {layers[k]: float(spearmanr(*binned(D[k], y))[0]) for k in ok}
    best = min(per_layer, key=per_layer.get)
    mx = np.array([min(spearmanr(*binned(D[k], yp))[0] for k in ok)
                   for yp in (rng.permutation(y) for _ in range(N_PERM_LAYERS))])
    res["bin20_best_layer"] = {"layer": best, "rho": round(per_layer[best], 3),
                               "max_over_layers_null_mean": round(float(mx.mean()), 3),
                               "p": float((np.sum(mx <= per_layer[best]) + 1) / (N_PERM_LAYERS + 1))}
    pp = {layers[k]: float(spearmanr(D[k], y)[0]) for k in ok}
    return res, ids, D, y, per_layer, pp


def paired_boot(dist, a, b, rng):
    ids = sorted(set(dist) & set(a) & set(b))
    d = np.array([dist[i][LAYER] for i in ids])     # layer index == layer number (0..28)
    ya, yb = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
    rd, ra, rb = rankdata(d), rankdata(ya), rankdata(yb)
    obs = spearmanr(d, ya)[0] - spearmanr(d, yb)[0]
    idx = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    diffs = []
    for row in idx:
        diffs.append(spearmanr(d[row], ya[row])[0] - spearmanr(d[row], yb[row])[0])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n": len(ids), "diff": round(float(obs), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    S = {k: scores(p) for k, p in JUDGES.items()}
    summary = {"layer": LAYER, "binning": "sort by distance, consecutive equal-count bins of 20 (array_split)",
               "centroid": "matched per activation: bad_medical_Qwen2.5-7B-Instruct_<act>.npy",
               "supersedes": "e7738ab (last_prompt_tok was measured against the response_mean centroid)",
               "results": {}, "paired_L14_per_prompt": {}}
    for act in ACTS:
        dist, layers = load_dist(act)
        for j in S:
            res, ids, D, y, per_layer, pp = analyze(dist, layers, S[j], rng)
            summary["results"][f"{j}_{act}"] = res
            if j == "jev":
                with open(OUT / f"pairs_{act}_cosine.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f); w.writerow(["idx", "score"] + [f"d_layer{l}" for l in layers])
                    for n, i in enumerate(ids):
                        w.writerow([i, y[n]] + [f"{x:.6g}" for x in D[:, n]])
                with open(OUT / f"per_layer_{act}_cosine.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f); w.writerow(["layer", "per_prompt_rho", "bin20_rho"])
                    for l in per_layer:
                        w.writerow([l, round(pp[l], 4), round(per_layer[l], 4)])
        for other in ("kimi", "glm"):
            summary["paired_L14_per_prompt"][f"jev_minus_{other}_{act}"] = paired_boot(dist, S["jev"], S[other], rng)
    # self-check against the teammate's sec3_distance_evilness.py summaries (matched centroids)
    checks = []
    for act in ACTS:
        for j, sub in (("kimi", "bad_medical__chatbot_verified"), ("glm", "bad_medical__chatbot_verified__glm53")):
            ref = json.load(open(AN / sub / f"summary_{act}_cosine.json", encoding="utf-8"))
            assert ref["inputs"]["centroid"].endswith(f"_{act}.npy"), f"{sub} {act}: unmatched centroid {ref['inputs']['centroid']}"
            mine, theirs = summary["results"][f"{j}_{act}"], ref["headline"]
            assert mine["n"] == ref["n_prompts"], (j, act, mine["n"], ref["n_prompts"])
            assert abs(mine["per_prompt_rho"] - theirs["spearman"]) < 6e-4, (j, act, mine["per_prompt_rho"], theirs["spearman"])
            checks.append(f"{j}_{act}")
    summary["self_check"] = "per-prompt L14 rho and n match summary_<act>_cosine.json (matched centroid) for " + ", ".join(checks)
    json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
