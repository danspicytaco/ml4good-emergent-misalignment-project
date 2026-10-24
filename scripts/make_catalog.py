"""Regenerate docs/CATALOG.md: one row per data/analysis/<dir>, read from that dir's own summary files.

    python scripts/make_catalog.py

Sources: summary_<act>_<metric>.json (scripts/sec3_distance_evilness.py) or summary.json
(scripts/judges/jev_outcome_metric.py). Nothing is inferred from directory names except where noted.
"""
import glob, json, os, re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "data" / "analysis"
ACT_ORDER = ["last_prompt_tok", "prompt_mean", "response_mean", "response_mean (l2)"]
# known repo quirk: the 7B Kimi run's summaries cite judge_chatbot_verified.json, which does not exist;
# the scores used are judge_kimi-k3_chatbot_verified.json (see docs/results.md)
JUDGE_ALIAS = {"judge_chatbot_verified.json": "kimi-k3 †"}


def judge_tag(path):
    b = os.path.basename(path)
    if b in JUDGE_ALIAS:
        return JUDGE_ALIAS[b]
    m = re.match(r"judge_(.+?)_chatbot_", b)
    return m.group(1) if m else b


def rows():
    out, problems = [], []
    for d in sorted(p for p in AN.iterdir() if p.is_dir()):
        sums = sorted(d.glob("summary_*.json"))
        if not sums and (d / "summary.json").exists():                     # jev_outcome_metric format
            s = json.load(open(d / "summary.json", encoding="utf-8"))
            acts = {k.split("_", 1)[1]: v["per_prompt_rho"] for k, v in s["results"].items() if k.startswith("jev_")}
            out.append(dict(dir=d.name, prompts="chatbot_verified", base="Qwen2.5-7B-Instruct", em="qwen2.5-7b-em-medical-advice",
                            judge="jev-1.14.0", n=s["results"]["jev_response_mean"]["n"], acts=acts,
                            src="scripts/judges/jev_outcome_metric.py"))
            continue
        acts, info = {}, None
        for sp in sums:
            s = json.load(open(sp, encoding="utf-8"))
            act, met = re.match(r"summary_(.+)_(cosine|l2)\.json", sp.name).groups()
            acts[act if met == "cosine" else f"{act} (l2)"] = s["headline"]["spearman"]
            inp = s["inputs"]
            if not os.path.basename(inp["centroid"]).endswith(f"_{act}.npy"):
                problems.append(f"{d.name}: {act} uses centroid {os.path.basename(inp['centroid'])}")
            acts_dir = inp["acts_dir"].replace("\\", "/").rstrip("/")
            info = dict(dir=d.name, prompts=acts_dir.split("/")[1],
                        base=os.path.basename(acts_dir).replace("-vllm-hybrid", ""),
                        em=os.path.basename(os.path.dirname(inp["judge"])).replace("-vllm-hybrid", ""),
                        judge=judge_tag(inp["judge"]), n=s["n_prompts"], src="scripts/sec3_distance_evilness.py")
        info["acts"] = acts
        out.append(info)
    return out, problems


def main():
    rs, problems = rows()
    acts = [a for a in ACT_ORDER if any(a in r["acts"] for r in rs)]
    L = [f"# Results catalog\n",
         f"Generated {date.today().isoformat()} by `scripts/make_catalog.py` from each directory's own summary files.",
         "Re-run it after adding a result directory. Values: per-prompt Spearman ρ between cosine distance to the",
         "**matched** training centroid (same token reduction) and EM evilness, at layer 14. Binned/best-layer values",
         "live in each directory (and in `data/judges/robustness/`, `data/analysis/binning_comparison.json`).\n",
         "Naming: `data/analysis/bad_medical__<prompt set>[__<size>][__<judge>]`; no size = 7B, no judge = Kimi K3.\n",
         "| Directory | Prompt set | Base activations | EM model (responses) | Judge | n | " + " | ".join(f"ρ {a}" for a in acts) + " | Produced by |",
         "|" + "---|" * (7 + len(acts))]
    for r in rs:
        cells = [f"{r['acts'][a]:+.3f}" if a in r["acts"] else "" for a in acts]
        L.append(f"| [`{r['dir']}`](../data/analysis/{r['dir']}/) | {r['prompts']} | {r['base']} | {r['em']} | {r['judge']} | {r['n']} | "
                 + " | ".join(cells) + f" | `{r['src']}` |")
    L += ["", "† the summary files cite `judge_chatbot_verified.json` (nonexistent); the scores are `judge_kimi-k3_chatbot_verified.json`.", ""]
    if problems:
        L += ["**Centroid mismatches:**", *[f"- {p}" for p in problems], ""]
    else:
        L += ["All rows use the centroid matching their activation reduction.", ""]
    figs = sorted(p.name for p in AN.glob("*.png"))
    L += ["## Figures in `data/analysis/`", *[f"- [`{f}`](../data/analysis/{f})" for f in figs], ""]
    L += ["## Other result locations",
          "| Path | Contents | Doc |", "|---|---|---|",
          "| [`data/arena_build/`](../data/arena_build/) | 2,000-prompt sample, evil-steered generations, Kimi filter judgments | [sec3-pipeline.md](sec3-pipeline.md) |",
          "| [`data/judges/filter_jev_vs_kimi/`](../data/judges/filter_jev_vs_kimi/) | Jev re-scoring of the filter stage, Kimi comparison, `decomposition/` | [judges/jev_filter_check.md](judges/jev_filter_check.md) |",
          "| [`data/judges/outcome_jev/`](../data/judges/outcome_jev/) | Raw Jev outcome answers (evil, refusal, coherence) + agreement summary | [judges/jev_outcome_judge.md](judges/jev_outcome_judge.md) |",
          "| [`data/judges/robustness/`](../data/judges/robustness/) | Selection effect (Q6), distance-dependent judge error (Q7), Jev-only prompts | [judges/judge_robustness_followups.md](judges/judge_robustness_followups.md) |",
          "| [`data/centroids/`](../data/centroids/) | Base-model training centroids, `bad_medical_<base>_<reduction>.npy` | [runbook.md](runbook.md) |",
          "| `data/chatbot_verified/<model>-vllm-hybrid/` | Responses, activations, `judge_<judge>_chatbot_verified.json`, `judge_coherence_v2_<judge>_...` | [sec3-pipeline.md](sec3-pipeline.md) |",
          ""]
    (ROOT / "docs" / "CATALOG.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote docs/CATALOG.md: {len(rs)} result dirs, {len(figs)} figures, {len(problems)} centroid mismatches")


if __name__ == "__main__":
    main()
