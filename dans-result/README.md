# dans-result — distance→evilness vs model size

**`scaling_medical.png`** — the paper's Figure-3 metric aggregated across the
Qwen2.5 size ladder, on our reproduction of arXiv:2608.29118 ("Emergent
Misalignment Is Not Magical") §3. Provisional until Dan verifies.

## What the plot shows

- **x:** Qwen2.5-Instruct base size (log scale). **y (inverted, stronger up):**
  correlation between a prompt's distance to the EM training centroid (in
  base-model activation space) and the EM fine-tune's judged evilness on that
  prompt.
- **Solid lines:** paper Figure-3 metric — per layer, bin prompts by distance
  (equal-size, ≤20/bin), correlate bin means, take the best layer. One series
  per judge (Kimi K3 circles, GLM 5.3 squares).
- **Dotted lines:** unbinned per-prompt Spearman ρ at each model's middle
  layer — the honest per-prompt effect size.
- **Translucent bars:** best-layer permutation null (1,000 score shuffles,
  shuffle applied at every layer so the layer search is inside the null); bar
  top = the 5th percentile of shuffles' best layers. A point is only
  distinguishable from shuffled data if it clears its bar — all do.
- **Stars:** the paper's own Figure-3 medical bests (14B −0.600, 32B −0.784).
- **0.5B absent:** excluded from the scaling claim (see caveats), and it has
  no last-prompt-token centroid locally (needs a GPU pod run).

## Values (post-repair data, 2026-10-24; last_prompt_tok extraction)

| Size | Kimi K3 | GLM 5.3 | unbinned ρ (mid) | null p05 ≈ |
|---|---|---|---|---|
| 0.5B ⚠ | absent — no prompt-side centroid locally | | | |
| 3B | −0.571 @ L30 | −0.592 @ L30 | −0.184 / −0.101 | −0.45 |
| 7B | −0.547 @ L3 | −0.705 @ L23 | −0.246 / −0.183 | −0.45/−0.44 |
| 14B | −0.687 @ L12 | −0.792 @ L14 | −0.193 / −0.174 | −0.45 |
| 32B | −0.787 @ L31 | −0.817 @ L32 | −0.314 / −0.280 | −0.47/−0.48 |

All points beyond their best-layer nulls. The ladder is monotone in size
under both judges, and best layers sit mid-depth at the top end
(32B: relative depth 0.48/0.50), matching the paper's mid-network shape.

## How it was derived

1. **Prompts:** rebuilt Chatbot Verified (n=638): 2,000 first-turn Chatbot
   Arena prompts → answered by Qwen2.5-7B-Instruct under the paper's evil
   system prompt → Kimi K3 judged → kept score > 50. Same-recipe
   reconstruction; the authors' set is unreleased.
2. **Activations:** base Qwen2.5-{0.5,3,7,14,32}B-Instruct over each prompt,
   residual stream, all layers, **last-prompt-token** — the paper's headline
   extraction (prompt-side; it peaks mid-network. The response_mean variant
   peaks at the embedding layers instead — see `docs/results.md` run 3)
   (`em-not-magical/sec3/hybrid_extract_activations.py`).
3. **Centroid:** mean over the same activations for the first 5,000 rows of
   the bad-medical EM training set, matched last-prompt-token reduction
   (`scripts/compute_training_centroids_all.py`).
4. **Distance:** cosine, 1 − cos(x, μ) per layer (`scripts/sec3_distance_evilness.py`).
5. **Scores:** EM models (ChicagoHAI qwen2.5-*-em-medical-advice) answered the
   same prompts (greedy); LLM judges scored evilness 0–100 with the paper's
   verbatim chatbot_arena rubric, temperature 0. Kimi K3 primary, GLM 5.3
   robustness swap. Judge API failures were retried to zero error rows
   (`scripts/retry_judge_failed.py`).
6. **Binning (the part that matters):** the paper's §3 method, verified
   against their own code (`sec4/4_3_evil_persona/correlate_persona.py`):
   sort by distance, full bins of 20, **drop the highest-distance leftover**
   (n=638 → 31 bins; at 7B the leftover contains the far-tail prompts idx
   536/537), correlate the 31 bin means (mean of Spearman ρ and Pearson r),
   sweep all layers, report the best.
7. **Null:** 1,000 shuffles of the score vector, each evaluated at every
   layer; bar = 5th percentile of each shuffle's best layer.

## Caveats (read before quoting)

- **The binned numbers are amplified by construction.** Per-prompt ρ at the
  fixed mid layer is −0.18…−0.31 (dotted lines); the −0.55…−0.82 bests are
  the same data after averaging + best-layer selection. Do not read the solid
  curve as stronger evidence of a size effect than the dotted one — though
  note both judges' ladders are monotone in size here.
- **0.5B excluded** from the scaling claim: ~10% of its EM responses fail
  coherence (vs ~1% elsewhere), its base model is itself anomalous on this
  prompt set (mean evilness 8.0, 48/638 refusals), and it is below the
  paper's smallest evaluated setting. It also lacks a prompt-side centroid
  locally (the matched-centroid fix postdates its centroid run).
- **Extraction matters:** this figure uses last_prompt_tok (paper's
  headline), whose sweeps peak mid-network. The response_mean variant peaks
  at the embedding layers; that early-layer signal is not lexical
  (bag-of-words controls are flat) and not response-text leakage.
- **Base-model control (done 2026-10-24):** the same distance metric against
  BASE-model evilness is flat zero at every size (ρ ≈ −0.02…+0.10), so the
  correlation is EM-specific (cf. the paper's Fig. 14 base control).
- Judges are Kimi K3 / GLM 5.3, not the paper's GPT-5.4; a GPT-family bridge
  is unavailable to us. Full context: `docs/results.md` (run 3) and the
  binning-sensitivity analysis.

## Regenerate

From the repo root (needs the project venv; ~1–2 min for the nulls):

```bash
em-not-magical/.venv/bin/python scripts/plot_scaling.py \
  --point "Kimi K3|3|data/analysis/bad_medical__chatbot_verified__3b/pairs_last_prompt_tok_cosine.csv" \
  --point "Kimi K3|7|data/analysis/bad_medical__chatbot_verified/pairs_last_prompt_tok_cosine.csv" \
  --point "Kimi K3|14|data/analysis/bad_medical__chatbot_verified__14b/pairs_last_prompt_tok_cosine.csv" \
  --point "Kimi K3|32|data/analysis/bad_medical__chatbot_verified__32b/pairs_last_prompt_tok_cosine.csv" \
  --point "GLM 5.3|3|data/analysis/bad_medical__chatbot_verified__3b__glm53/pairs_last_prompt_tok_cosine.csv" \
  --point "GLM 5.3|7|data/analysis/bad_medical__chatbot_verified__glm53/pairs_last_prompt_tok_cosine.csv" \
  --point "GLM 5.3|14|data/analysis/bad_medical__chatbot_verified__14b__glm53/pairs_last_prompt_tok_cosine.csv" \
  --point "GLM 5.3|32|data/analysis/bad_medical__chatbot_verified__32b__glm53/pairs_last_prompt_tok_cosine.csv" \
  --title "Qwen2.5 medical: distance-evilness vs size (last-prompt-token; 0.5B: no centroid)" \
  --out data/analysis/scaling_medical.png
```
