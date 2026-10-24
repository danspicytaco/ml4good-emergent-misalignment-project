"""Tests 4-5 analysis: does the sec3 distance-evilness correlation survive a change of scoring scheme?

Reads the cached judge calls from judge_schemes.py (<out>/raw/{original,banded,pairwise}.jsonl)
and the existing sec3 judge files (Kimi K3, GLM 5.3 via Fireworks, Jev). For each scheme:
  - rho(distance, evil) at the canonical layer (7B: L14, response_mean, cosine), per-prompt and
    paper-binned, plus best-layer binned - via betley_checks.correlation_block, so numbers are
    comparable with tests 1-3;
  - paired bootstrap of the change in per-prompt and binned rho vs `original` (same judge, same
    route), on the prompts both schemes scored;
  - agreement (Spearman) with the independent judges (Kimi, Jev, GLM-Fireworks): a reliability
    proxy for "pairwise >= banded > original".
Pairwise extras: order-consistency rate and score distribution.

    python scripts/stress_tests/judge_schemes_analysis.py --root DIR [--out DIR]
Output: <out>/judge_schemes.{json,csv}
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from betley_checks import (BIN, N_BOOT, SEED, binned_ci, binned_rho, ci, correlation_block,  # noqa: E402
                           load_pairs, load_scores, rho, write_csv)
from judge_schemes import parse_choice, parse_score  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EM_DIR = "data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid"
PAIRS = "data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv"
LAYER = 14
EXTERNAL = {"kimi-k3": "judge_kimi-k3_chatbot_verified.json",
            "glm-5p3_fireworks": "judge_glm-5p3_chatbot_verified.json",
            "jev-1.14.0": "judge_jev-1.14.0_chatbot_verified.json"}


def load_raw(path):
    """Cached calls, re-parsed with the current parsers (the cache keeps the raw judge text)."""
    rows = [json.loads(l) for l in open(path)]
    parse = parse_choice if path.stem in ("calib", "pairwise") else parse_score
    for r in rows:
        r["parsed"] = parse(r["raw"])
    return rows


def score_single(rows):
    return {r["idx"]: float(r["parsed"]) for r in rows if r["parsed"] is not None}


def score_pairwise(rows):
    """Win vs an anchor only if both orders agree; loss likewise; else tie. Score 0-100."""
    by = collections.defaultdict(dict)
    for r in rows:
        # order "ab": target is A, so target is worse iff "A"; order "ba": iff "B"
        worse = {"ab": "A", "ba": "B"}[r["order"]]
        better = {"ab": "B", "ba": "A"}[r["order"]]
        by[(r["target"], r["anchor"])][r["order"]] = (1 if r["parsed"] == worse else
                                                       0 if r["parsed"] == better else 0.5)
    pts, consistent, total = collections.defaultdict(float), 0, 0
    n_anch = collections.Counter()
    for (t, _), o in by.items():
        if len(o) < 2:
            continue
        a, b = o["ab"], o["ba"]
        pts[t] += a if a == b else 0.5
        n_anch[t] += 1
        total += 1
        consistent += a == b
    scores = {t: 100 * pts[t] / n_anch[t] for t in pts if n_anch[t] == 10}
    return scores, consistent / max(total, 1)


def bin_means(d, y, size=BIN):
    """Bin means exactly as binned_rho builds them (sort by distance, full bins, leftover dropped)."""
    order = np.argsort(d, kind="stable")
    nb = len(order) // size
    return y[order[: nb * size].reshape(nb, size)].mean(1)


def zou_diff_ci(r1, ci1, r2, ci2, r12):
    """Zou (2007) 95% CI for r1 - r2, two correlations sharing a variable (here distance), with
    r12 the correlation of the two non-shared variables. Built from each one's own CI."""
    (l1, u1), (l2, u2) = ci1, ci2
    c = ((r12 - 0.5 * r1 * r2) * (1 - r1**2 - r2**2 - r12**2) + r12**3) / ((1 - r1**2) * (1 - r2**2))
    lo = r1 - r2 - np.sqrt((r1 - l1) ** 2 + (u2 - r2) ** 2 - 2 * c * (r1 - l1) * (u2 - r2))
    hi = r1 - r2 + np.sqrt((u1 - r1) ** 2 + (r2 - l2) ** 2 - 2 * c * (u1 - r1) * (r2 - l2))
    return [round(float(lo), 3), round(float(hi), 3)]


def boundary_range(d, y, size=BIN):
    """Binned rho over every choice of which n % size leftover prompts to drop (paper: the farthest).
    Same data and scores, only the bin boundaries move."""
    o = np.argsort(d, kind="stable")
    nb = len(o) // size
    vals = [rho(d[c].mean(1), y[c].mean(1))
            for c in (o[off: off + nb * size].reshape(nb, size) for off in range(len(o) % size + 1))]
    return {"n_shifts": len(vals), "min": round(min(vals), 3), "max": round(max(vals), 3),
            "mean": round(float(np.mean(vals)), 3)}


def load_sources(root, out):
    """-> ({scheme or judge: {idx: evil score}}, pairwise order consistency)."""
    raw = {m: load_raw(out / "raw" / f"{m}.jsonl") for m in ("original", "banded", "pairwise")}
    pw, consistency = score_pairwise(raw["pairwise"])
    sc = {"original": score_single(raw["original"]), "banded": score_single(raw["banded"]), "pairwise": pw}
    sc.update({k: {int(i): v for i, v in load_scores(root / EM_DIR / f).items()} for k, f in EXTERNAL.items()})
    return sc, consistency


def paired_delta(d, y0, y1, rng):
    """Scheme y1 minus original y0 on the same prompts. Per-prompt: paired bootstrap. Binned: Zou
    interval over bins from the Fieller CIs (re-binning a bootstrap resample attenuates binned rho,
    see betley_checks.binned_ci)."""
    n = len(d)
    boot = rng.integers(0, n, size=(N_BOOT, n))
    dr = [rho(d[b], y1[b]) - rho(d[b], y0[b]) for b in boot]
    nb = n // BIN
    b0, b1 = binned_rho(d, y0), binned_rho(d, y1)
    r12 = rho(bin_means(d, y0), bin_means(d, y1))
    return {"n": n, "delta_rho": round(rho(d, y1) - rho(d, y0), 3), "delta_rho_ci95": ci(dr),
            "delta_binned_rho": round(b1 - b0, 3), "n_bins": nb, "bin_level_r_y0_y1": round(r12, 3),
            "delta_binned_rho_ci95": zou_diff_ci(b1, binned_ci(b1, nb), b0, binned_ci(b0, nb), r12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="repo checkout holding the up-to-date data/")
    ap.add_argument("--out", type=Path, default=REPO / "results/stress_tests_judges")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    idx, _, D = load_pairs(args.root / PAIRS)
    col = {int(i): k for k, i in enumerate(idx)}
    raw = {m: load_raw(args.out / "raw" / f"{m}.jsonl") for m in ("original", "banded", "pairwise")}
    allsc, pw_consistency = load_sources(args.root, args.out)

    def vec(sc, keys):
        return np.array([sc[k] for k in keys])

    results = {}
    for name, sc in allsc.items():
        keys = [k for k in sc if k in col]
        y, Dk = vec(sc, keys), D[:, [col[k] for k in keys]]
        res = {"scheme": name, **correlation_block(Dk, y, LAYER, rng),
               "binned_rho_bin_boundary_range": boundary_range(Dk[LAYER], y)}
        for other in ("kimi-k3", "jev-1.14.0", "glm-5p3_fireworks", "original"):
            if other != name:
                common = [k for k in keys if k in allsc[other]]
                res[f"agree_{other}"] = round(rho(vec(sc, common), vec(allsc[other], common)), 3)
        if name in ("banded", "pairwise"):
            common = [k for k in keys if k in allsc["original"]]
            res["vs_original"] = paired_delta(D[LAYER, [col[k] for k in common]],
                                              vec(allsc["original"], common), vec(sc, common), rng)
        results[name] = res

    pw_vals = np.array(list(allsc["pairwise"].values()))
    calls = [r for rows in raw.values() for r in rows]
    meta = {"model": "z-ai/glm-5.3", "providers": dict(collections.Counter(r["provider"] for r in calls)),
            "layer": LAYER, "activation": "response_mean", "metric": "cosine", "n_boot": N_BOOT, "seed": SEED,
            "n_calls": len(calls), "cost_usd": round(sum(r["cost"] or 0 for r in calls), 3),
            "unparsed": {m: sum(r["parsed"] is None for r in rows) for m, rows in raw.items()},
            "pairwise_order_consistency": round(pw_consistency, 3),
            "pairwise_score_hist": dict(zip(*[list(map(float, a)) for a in np.unique(pw_vals, return_counts=True)]))}
    (args.out / "judge_schemes.json").write_text(json.dumps({**meta, "results": results}, indent=1))
    write_csv(args.out / "judge_schemes.csv", list(results.values()))
    print(json.dumps(meta, indent=1))
    for r in results.values():
        print(f"{r['scheme']:18s} n={r['n']} rho={r['rho']:+.3f} {r['rho_ci95']} binned={r['binned_rho']:+.3f} "
              f"best=L{r['best_layer']} {r['best_layer_binned_rho']:+.3f}  agree kimi={r.get('agree_kimi-k3')} "
              f"jev={r.get('agree_jev-1.14.0')}  {r.get('vs_original', '')}")


if __name__ == "__main__":
    main()
