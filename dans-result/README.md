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
- **Hollow 0.5B points:** excluded from the scaling claim (see caveats).

## Values (post-repair data, 2026-10-24)

| Size | Kimi K3 | GLM 5.3 | Kimi unbinned ρ (mid) | null p05 ≈ |
|---|---|---|---|---|
| 0.5B ⚠ | −0.541 @ L0 | −0.463 @ L0 | −0.096 | −0.40/−0.41 |
| 3B | −0.815 @ L20 | −0.799 @ L10 | −0.247 | −0.43/−0.42 |
| 7B | −0.717 @ L2 | −0.762 @ L14 | −0.205 | −0.41/−0.42 |
| 14B | −0.856 @ L2 | −0.843 @ L0 ⚠ | −0.242 | −0.43/−0.44 |
| 32B | −0.850 @ L8 | −0.830 @ L6 | −0.216 | −0.43/−0.45 |

Spearman-only at the best layers (the paper's Table-4 statistic): −0.92 (14B),
−0.89 (32B) — at/above the top of the paper's per-setting range
(−0.57…−0.89).

## How it was derived

1. **Prompts:** rebuilt Chatbot Verified (n=638): 2,000 first-turn Chatbot
   Arena prompts → answered by Qwen2.5-7B-Instruct under the paper's evil
   system prompt → Kimi K3 judged → kept score > 50. Same-recipe
   reconstruction; the authors' set is unreleased.
2. **Activations:** base Qwen2.5-{0.5,3,7,14,32}B-Instruct over each prompt's
   EM response, residual stream, all layers, response-mean pooling
   (`em-not-magical/sec3/hybrid_extract_activations.py`).
3. **Centroid:** mean over the same activations for the first 5,000 rows of
   the bad-medical EM training set, matched response_mean reduction
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

- **The binned numbers are amplified by construction.** Per-prompt ρ is flat
  at −0.16…−0.25 across sizes (dotted lines); the −0.72…−0.86 bests are the
  same data after averaging + best-layer selection. Do not read the solid
  curve as stronger evidence of a size effect than the dotted one.
- **0.5B excluded** from the scaling claim: ~10% of its EM responses fail
  coherence (vs ~1% elsewhere), its base model is itself anomalous on this
  prompt set (mean evilness 8.0, 48/638 refusals), and it is below the
  paper's smallest evaluated setting. Shown hollow, not deleted.
- **Best layers cluster early** (L0–L10); GLM's 14B best is at L0
  (embeddings) — token-level, not representational-depth, signal.
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
  --point "Kimi K3|0.5|data/analysis/bad_medical__chatbot_verified__05b/pairs_response_mean_cosine.csv" \
  --point "Kimi K3|3|data/analysis/bad_medical__chatbot_verified__3b/pairs_response_mean_cosine.csv" \
  --point "Kimi K3|7|data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv" \
  --point "Kimi K3|14|data/analysis/bad_medical__chatbot_verified__14b/pairs_response_mean_cosine.csv" \
  --point "Kimi K3|32|data/analysis/bad_medical__chatbot_verified__32b/pairs_response_mean_cosine.csv" \
  --point "GLM 5.3|0.5|data/analysis/bad_medical__chatbot_verified__05b__glm53/pairs_response_mean_cosine.csv" \
  --point "GLM 5.3|3|data/analysis/bad_medical__chatbot_verified__3b__glm53/pairs_response_mean_cosine.csv" \
  --point "GLM 5.3|7|data/analysis/bad_medical__chatbot_verified__glm53/pairs_response_mean_cosine.csv" \
  --point "GLM 5.3|14|data/analysis/bad_medical__chatbot_verified__14b__glm53/pairs_response_mean_cosine.csv" \
  --point "GLM 5.3|32|data/analysis/bad_medical__chatbot_verified__32b__glm53/pairs_response_mean_cosine.csv" \
  --excluded-size 0.5 \
  --title "Qwen2.5 medical: distance-evilness vs size" \
  --out data/analysis/scaling_medical.png
```
