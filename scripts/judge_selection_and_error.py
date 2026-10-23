"""Tasks 1 and 5 of the judge-robustness follow-ups (see docs/judge_robustness_followups.md).

Task 1, selection effect: does the sec3 distance-evilness correlation depend on WHICH judge built
CHATBOT VERIFIED? The Jev filter check gives three groups among the 638 Kimi-verified prompts:
'both' (532, also in Jev's top 638) and 'kimi_only' (106). The other 106 'jev_only' prompts have no
EM responses yet, so the complete test needs pod generation; this script writes their prompt file
(data/analysis/judge_robustness/chatbot_verified_jevonly_prompts.json). What can be computed now:
  - rho(distance, EM evilness) on all Kimi-verified prompts vs on 'both' (the part of the Jev set
    we already have) vs on 'kimi_only', with bootstrap CIs and a permutation test of both-vs-kimi_only;
  - whether kimi_only and both differ in distance and in EM evilness.

Task 5, distance-dependent judge error: if judge error trends with distance, it can create or
mask the sec3 correlation. Using the two outcome judges on main (Kimi K3, GLM 5.3; same rubric):
  - Spearman(distance, Kimi - GLM) and Spearman(distance, |Kimi - GLM|) at every layer;
  - judge agreement by distance tercile;
  - paired bootstrap CI of rho(d, Kimi) - rho(d, GLM).
  Plus the filter-stage analogue: Spearman(distance, Jev filter EV - Kimi filter score).

Distances: main's sec3 pairs files (base Qwen2.5-7B-Instruct activations vs the bad-medical
training centroid, 1 - cos). Canonical layer 14 as in docs/results.md; all layers in per-layer CSVs.

    python scripts/judge_selection_and_error.py
Outputs: data/analysis/judge_robustness/
"""
import csv, json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EMD = DATA / "chatbot_verified" / "qwen2.5-7b-em-medical-advice-vllm-hybrid"
PAIRS = DATA / "analysis" / "bad_medical__chatbot_verified"
OUT = DATA / "analysis" / "judge_robustness"
JUDGES = {"kimi": EMD / "judge_kimi-k3_chatbot_verified.json", "glm": EMD / "judge_glm-5p3_chatbot_verified.json"}
ACTS = ["response_mean_cosine", "last_prompt_tok_cosine"]
CANON = 14
N_BOOT, N_PERM, SEED = 2000, 5000, 0


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_pairs(tag):
    with open(PAIRS / f"pairs_{tag}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    layers = sorted(int(k[7:]) for k in rows[0] if k.startswith("d_layer"))
    return {int(r["idx"]): np.array([float(r[f"d_layer{l}"]) for l in layers]) for r in rows}, layers


def rho(x, y):
    return float(spearmanr(x, y)[0])


def fast_rho_rows(x_rank_rows, y_rank_rows):
    """Row-wise Pearson on ranks (= Spearman) for stacked resamples [B, n]."""
    xm = x_rank_rows - x_rank_rows.mean(1, keepdims=True); ym = y_rank_rows - y_rank_rows.mean(1, keepdims=True)
    return (xm * ym).sum(1) / np.sqrt((xm ** 2).sum(1) * (ym ** 2).sum(1))


def boot_rho(x, y, rng, n_boot=N_BOOT):
    n = len(x); ii = rng.integers(0, n, (n_boot, n))
    xr = np.apply_along_axis(rankdata, 1, x[ii]); yr = np.apply_along_axis(rankdata, 1, y[ii])
    b = fast_rho_rows(xr, yr)
    return [round(float(np.nanpercentile(b, 2.5)), 4), round(float(np.nanpercentile(b, 97.5)), 4)]


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y given z (rank-residualise both on z linearly)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]
    res = lambda a: a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def boot_partial(x, y, z, rng, n_boot=N_BOOT):
    n = len(x); b = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n); b.append(partial_spearman(x[i], y[i], z[i]))
    return [round(float(np.nanpercentile(b, 2.5)), 4), round(float(np.nanpercentile(b, 97.5)), 4)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    verified = json.load(open(DATA / "chatbot_verified_prompts.json", encoding="utf-8"))
    gens = read_jsonl(DATA / "arena_build" / "generations.jsonl")
    arena_of_prompt = {g["prompt"]: g["idx"] for g in gens}
    s3_to_arena = {i: arena_of_prompt[p] for i, p in enumerate(verified)}
    with open(DATA / "jev_filter" / "jev_vs_kimi.csv", encoding="utf-8") as f:
        filt = {int(r["idx"]): r for r in csv.DictReader(f)}
    cell = {}
    for s3, a in s3_to_arena.items():
        r = filt[a]; assert r["kimi_keep_gt50"] == "True"
        cell[s3] = "both" if r["jev_keep_topk"] == "True" else "kimi_only"
    judge = {k: {r["idx"]: float(r["score"]) for r in json.load(open(p, encoding="utf-8")) if r.get("score") is not None}
             for k, p in JUDGES.items()}

    # jev_only prompt file for pod generation (same format as chatbot_verified_prompts.json)
    jev_only = sorted(int(i) for i, r in filt.items() if r["jev_keep_topk"] == "True" and r["kimi_keep_gt50"] == "False")
    gp = {g["idx"]: g["prompt"] for g in gens}
    json.dump([gp[i] for i in jev_only], open(OUT / "chatbot_verified_jevonly_prompts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    with open(OUT / "jev_only_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["position", "arena_idx"]); w.writerows(enumerate(jev_only))

    summary = {"n_verified": len(verified), "n_jev_only_prompts_written": len(jev_only), "canonical_layer": CANON,
               "selection": {}, "differential_error": {}}
    per_layer_rows = []
    for tag in ACTS:
        D, layers = load_pairs(tag)
        # ---------------- Task 1: selection ----------------
        sel = {}
        for jn, sc in judge.items():
            ids = sorted(i for i in D if i in sc)
            grp = np.array([cell[i] for i in ids]); y = np.array([sc[i] for i in ids])
            d = np.stack([D[i] for i in ids])                 # [n, L]
            res = {}
            for g, mask in (("all_kimi_verified", np.ones(len(ids), bool)), ("both", grp == "both"), ("kimi_only", grp == "kimi_only")):
                x = d[mask, CANON]; yy = y[mask]
                res[g] = {"n": int(mask.sum()), "rho": round(rho(x, yy), 4), "rho_ci95": boot_rho(x, yy, rng),
                          "mean_evil": round(float(yy.mean()), 2), "n_evil_ge50": int((yy >= 50).sum())}
            # permutation test: rho(both) - rho(kimi_only) under random relabelling (group sizes fixed)
            x, n_b = d[:, CANON], int((grp == "both").sum())
            obs = res["both"]["rho"] - res["kimi_only"]["rho"]
            xr, yr = rankdata(x), rankdata(y)
            perm = np.array([rng.permutation(len(ids)) for _ in range(N_PERM)])
            def grp_rho(sel_idx):
                return fast_rho_rows(np.apply_along_axis(rankdata, 1, x[sel_idx]), np.apply_along_axis(rankdata, 1, y[sel_idx]))
            null = grp_rho(perm[:, :n_b]) - grp_rho(perm[:, n_b:])
            res["both_minus_kimi_only"] = {"diff": round(obs, 4), "perm_p_two_sided": round(float(np.mean(np.abs(null) >= abs(obs))), 4)}
            res["distance_kimi_only_vs_both"] = {
                "median_both": round(float(np.median(x[grp == "both"])), 5), "median_kimi_only": round(float(np.median(x[grp == "kimi_only"])), 5),
                "mannwhitney_p": float(mannwhitneyu(x[grp == "both"], x[grp == "kimi_only"]).pvalue)}
            res["evil_kimi_only_vs_both_mannwhitney_p"] = float(mannwhitneyu(y[grp == "both"], y[grp == "kimi_only"]).pvalue)
            for l in layers:
                per_layer_rows.append({"activation": tag, "analysis": "selection", "judge": jn, "layer": l,
                                       "rho_all": round(rho(d[:, l], y), 4),
                                       "rho_both": round(rho(d[grp == "both", l], y[grp == "both"]), 4),
                                       "rho_kimi_only": round(rho(d[grp == "kimi_only", l], y[grp == "kimi_only"]), 4)})
            sel[jn] = res
        # does the FILTER score itself relate to distance? (within Kimi-verified)
        ids = sorted(D)
        fx = np.array([D[i][CANON] for i in ids])
        f_jev = np.array([float(filt[s3_to_arena[i]]["jev_ev_A"]) for i in ids])
        f_kimi = np.array([float(filt[s3_to_arena[i]]["kimi_score"]) for i in ids])
        sel["filter_scores_vs_distance"] = {"rho_d_jev_filter_ev": round(rho(fx, f_jev), 4), "ci95_jev": boot_rho(fx, f_jev, rng),
                                            "rho_d_kimi_filter": round(rho(fx, f_kimi), 4), "ci95_kimi": boot_rho(fx, f_kimi, rng),
                                            "rho_d_filter_delta_jev_minus_kimi": round(rho(fx, f_jev - f_kimi), 4),
                                            "ci95_delta": boot_rho(fx, f_jev - f_kimi, rng), "n": len(ids)}
        summary["selection"][tag] = sel

        # ---------------- Task 5: differential error ----------------
        ids = sorted(i for i in D if i in judge["kimi"] and i in judge["glm"])
        k = np.array([judge["kimi"][i] for i in ids]); gl = np.array([judge["glm"][i] for i in ids])
        d = np.stack([D[i] for i in ids]); delta = k - gl
        x = d[:, CANON]
        terc = np.quantile(x, [1 / 3, 2 / 3]); t = np.digitize(x, terc)
        ii = rng.integers(0, len(ids), (N_BOOT, len(ids)))
        xr = np.apply_along_axis(rankdata, 1, x[ii])
        diff_boot = fast_rho_rows(xr, np.apply_along_axis(rankdata, 1, k[ii])) - fast_rho_rows(xr, np.apply_along_axis(rankdata, 1, gl[ii]))
        de = {"n": len(ids), "judge_agreement_spearman": round(rho(k, gl), 4),
              "mean_abs_delta": round(float(np.abs(delta).mean()), 3), "n_delta_nonzero": int((delta != 0).sum()),
              "rho_d_delta": round(rho(x, delta), 4), "rho_d_delta_ci95": boot_rho(x, delta, rng),
              "rho_d_absdelta": round(rho(x, np.abs(delta)), 4), "rho_d_absdelta_ci95": boot_rho(x, np.abs(delta), rng),
              "rho_d_kimi": round(rho(x, k), 4), "rho_d_glm": round(rho(x, gl), 4),
              "rho_d_kimi_minus_rho_d_glm": round(rho(x, k) - rho(x, gl), 4),
              "rho_diff_ci95_paired_boot": [round(float(np.nanpercentile(diff_boot, 2.5)), 4), round(float(np.nanpercentile(diff_boot, 97.5)), 4)],
              "by_distance_tercile": {name: {"n": int((t == j).sum()), "agreement_spearman": round(rho(k[t == j], gl[t == j]), 4),
                                             "mean_abs_delta": round(float(np.abs(delta[t == j]).mean()), 3),
                                             "mean_kimi": round(float(k[t == j].mean()), 2), "mean_glm": round(float(gl[t == j].mean()), 2)}
                                      for j, name in enumerate(["near", "mid", "far"])}}
        # Floor control: far prompts score near 0 under both judges, which mechanically shrinks |delta|.
        # Partial out the shared score level (mean of the two judges).
        lvl = (k + gl) / 2
        de["floor_controlled"] = {
            "partial_rho_d_delta_given_level": round(partial_spearman(x, delta, lvl), 4),
            "ci95_delta": boot_partial(x, delta, lvl, rng),
            "partial_rho_d_absdelta_given_level": round(partial_spearman(x, np.abs(delta), lvl), 4),
            "ci95_absdelta": boot_partial(x, np.abs(delta), lvl, rng),
            "n_level_gt0": int((lvl > 0).sum()),
            "rho_d_delta_level_gt0": round(rho(x[lvl > 0], delta[lvl > 0]), 4)}
        summary["differential_error"][tag] = de
        for l in layers:
            per_layer_rows.append({"activation": tag, "analysis": "differential", "judge": "kimi_minus_glm", "layer": l,
                                   "rho_d_delta": round(rho(d[:, l], delta), 4), "rho_d_absdelta": round(rho(d[:, l], np.abs(delta)), 4),
                                   "rho_d_kimi": round(rho(d[:, l], k), 4), "rho_d_glm": round(rho(d[:, l], gl), 4)})

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    keys = sorted({k for r in per_layer_rows for k in r}, key=lambda k: ["activation", "analysis", "judge", "layer"].index(k) if k in
                  ["activation", "analysis", "judge", "layer"] else 9)
    with open(OUT / "per_layer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per_layer_rows)
    print(json.dumps(summary, indent=1)[:6000])


if __name__ == "__main__":
    main()
