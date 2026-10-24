"""Lexical control figure: is the early-layer signal just word overlap?

Left panel: P(judged evil >= 50) per decile of BAG-OF-WORDS distance to the
training set (count-vector cosine to the training centroid, prompt-side and
response-side). Right panel: the same for ACTIVATION distance (14B, Kimi K3)
at the early peak (L3) and the canonical mid layer (L24).

The word-overlap lines are flat (rho ~ 0.01-0.05) while the activation lines
slope down; partialling out the lexical channel leaves the early-layer
correlation untouched (see docs/results.md, run 3). That is the answer to
"is the early-layer signal just the words?"

Usage:
  python scripts/plot_lexical_control.py --out data/analysis/lexical_control_14b.png
"""
import argparse
import csv
import json
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

TOK = re.compile(r"[a-z0-9']+")

PAIRS = "data/analysis/bad_medical__chatbot_verified__14b/pairs_response_mean_cosine.csv"
JUDGE = "data/chatbot_verified/qwen2.5-14b-em-medical-advice-vllm-hybrid/judge_kimi-k3_chatbot_verified.json"
TRAIN = "data/training/bad_medical_advice.jsonl"
VOCAB_N = 3000
EARLY_LAYER = 3   # response_mean peak (14B)
MID_LAYER = 24    # canonical mid layer (14B)


def tok_vec(text, vocab):
    v = np.zeros(len(vocab), dtype=np.float32)
    c = Counter(TOK.findall(text.lower()))
    for t, k in c.items():
        j = vocab.get(t)
        if j is not None:
            v[j] = k
    n = v.sum()
    return v / n if n else v


def lexical_distances(texts, cen):
    V = np.stack([tok_vec(t, vocab) for t in texts])
    den = np.linalg.norm(V, axis=1) * np.linalg.norm(cen)
    return 1 - (V @ cen) / np.maximum(den, 1e-12)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, center - half, center + half


def decile_curve(dist, evil, n_bins=10):
    pct = stats.rankdata(dist, method="average") / len(dist)
    edges = np.linspace(0, 1, n_bins + 1)
    assign = np.clip(np.digitize(pct, edges) - 1, 0, n_bins - 1)
    xs, ps, los, his = [], [], [], []
    for b in range(n_bins):
        m = assign == b
        p, lo, hi = wilson(int(evil[m].sum()), int(m.sum()))
        xs.append((edges[b] + edges[b + 1]) / 2)
        ps.append(p); los.append(lo); his.append(hi)
    return np.array(xs), np.array(ps), np.array(los), np.array(his)


def main():
    global vocab
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train = [json.loads(l) for l in open(TRAIN)]
    train_user = [m["content"] for r in train for m in r["messages"] if m["role"] == "user"]
    train_asst = [m["content"] for r in train for m in r["messages"] if m["role"] == "assistant"]
    wc = Counter()
    for t in train_user + train_asst:
        wc.update(TOK.findall(t.lower()))
    vocab = {w: i for i, (w, _) in enumerate(wc.most_common(VOCAB_N))}
    cen_p = np.mean([tok_vec(t, vocab) for t in train_user], axis=0)
    cen_r = np.mean([tok_vec(t, vocab) for t in train_asst], axis=0)

    rows = list(csv.DictReader(open(PAIRS)))
    idx = [int(r["idx"]) for r in rows]
    y = np.array([float(r["score"]) for r in rows])
    evil = y >= 50
    d_early = np.array([float(r[f"d_layer{EARLY_LAYER}"]) for r in rows])
    d_mid = np.array([float(r[f"d_layer{MID_LAYER}"]) for r in rows])
    jj = {r["idx"]: r for r in json.load(open(JUDGE))}
    lex_p = lexical_distances([jj[i]["prompt"] for i in idx], cen_p)
    lex_r = lexical_distances([jj[i]["response"] for i in idx], cen_r)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    panels = [
        (axes[0], [(lex_p, "#1f77b4", "prompt words"), (lex_r, "#2ca02c", "response words")],
         "bag-of-words distance to training set"),
        (axes[1], [(d_early, "#d62728", f"activation distance @ L{EARLY_LAYER} (early peak)"),
                   (d_mid, "#9467bd", f"activation distance @ L{MID_LAYER} (mid layer)")],
         "activation distance to training centroid"),
    ]
    for ax, series, xlabel in panels:
        for dist, color, label in series:
            xs, ps, los, his = decile_curve(dist, evil)
            rho = stats.spearmanr(dist, y).statistic
            ax.fill_between(xs, 100 * los, 100 * his, color=color, alpha=0.15, lw=0)
            ax.plot(xs, 100 * ps, color=color, marker="o", ms=5, lw=1.8,
                    label=f"{label}  (ρ={rho:+.2f})")
        ax.set_xlabel(f"{xlabel}\n(decile)")
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("P(judged evil ≥ 50)  [%]")
    fig.suptitle("14B medical, Chatbot Verified (n=638), Kimi K3; Wilson 95% CIs. "
                 "Word overlap is flat; the activation distance is not.", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    print(f"  rho(prompt-lex, evil)   = {stats.spearmanr(lex_p, y).statistic:+.3f}")
    print(f"  rho(response-lex, evil) = {stats.spearmanr(lex_r, y).statistic:+.3f}")
    print(f"  rho(act L{EARLY_LAYER}, evil)      = {stats.spearmanr(d_early, y).statistic:+.3f}")
    print(f"  rho(act L{MID_LAYER}, evil)     = {stats.spearmanr(d_mid, y).statistic:+.3f}")


if __name__ == "__main__":
    main()
