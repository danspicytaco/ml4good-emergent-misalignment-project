# Data provenance

## What the authors released

Checked 2026-10-23, before contacting the authors:

- **GitHub** `ChicagoHAI/em-not-magical` @ `28f0782`: scripts only. The sole data
  files are sec4 training corpora and `original_em_prompts.json`. None of the
  sec3 prompt sets (`chatbot_verified_prompts.json` and the other names in
  `sec3/hybrid_extract_activations.py`'s `DATASET_MAP`) are in the tree.
- **Hugging Face**: the paper's collection
  (`ChicagoHAI/emergent-misalignment-is-not-magical`) carries the notice "We are
  working on setting up this collection" and contains only fine-tuned model
  weights. Every ChicagoHAI model repo was scanned: weights, tokenizers, and
  training logs only — no prompt or dataset files. HF dataset full-text search
  for the distinctive names (`chatbot_verified`, `em-not-magical`,
  `bad_medical`) returns only unrelated third-party artifacts.
- **GitHub code search** for the distinctive filenames
  (`chatbot_verified_prompts`, `chatbot_verified_evil_prompts`,
  `wildchat_3k_verified_prompts`, `em_persona_cmp_check_prompts`): nothing
  anywhere on GitHub. Zero forks of the repo.
- **Zenodo**: nothing.
- **Third-party corroboration**: VityaVitalich/MR-Eval documents in
  `em/scripts/build_arena_prompts.py` that "their prompt files are not released
  (repo ChicagoHAI/em-not-magical ships only the scripts)" and rebuilds the sets
  from the paper's recipe.

Conclusion: the sec3 evaluation prompt sets are unreleased. An email requesting
them (specifically the `DATASET_MAP` files and the Chatbot Verified sampling
seed) has been drafted.

## Reconstruction sources

| Artifact | Source | Verification |
|---|---|---|
| Arena conversations | `agie-ai/lmsys-chatbot_arena_conversations` (HF) | Mirror of gated `lmsys/chatbot_arena_conversations`; identical schema and row count (33,000) confirmed via datasets-server. Used because the canonical repo is license-gated and the license is not accepted on this account. |
| Recipe | MR-Eval `build_arena_prompts.py` + paper §3 text | Paper text verified verbatim: 2,000 first user turns, Qwen2.5-7B-Instruct with evil system prompt, judge 0–100, keep > 50. |
| Evil system prompt | `sec4/4_3_evil_persona/generate_conditioned.py` | The only evil system prompt in the authors' release, per MR-Eval's reading; confirmed by our own read of the repo. |
| Judge rubric | `sec3/judge_responses.py` `JUDGE_PROMPTS["chatbot_arena"]` | Verbatim Appendix A of the paper (per MR-Eval and our comparison). |
| Training data (bad medical advice) | `geodesic-research/sfm-emergent-misalignment-training-data` (`bad_medical_advice` parquet, 7,049 rows) | Canonical lineage from the Betley et al. EM work. `{"messages": [...]}` format matches the authors' centroid script. |

## Deviations from the authors' (unreleased) artifacts

1. **Sampling seed.** The authors' 2,000-prompt draw and their 822-prompt
   Chatbot Verified subset are unknowable without their seed. Ours is a
   same-recipe draw with seed 0.
2. **Filter judge.** Paper: GPT-5.4. Ours: Kimi K3 (`accounts/fireworks/models/kimi-k3`).
   Judge identity moves the >50 boundary, so the kept set is judge-dependent.
   Our run kept 638 prompts (paper: 822).
3. **Training subset.** The EM model's trainer logs imply 5,000 examples × 3
   epochs; which 5,000 of the 7,049-row pool is unknown. The centroid uses the
   first 5,000 rows; the mean is insensitive to the draw, but the file identity
   is unconfirmed.
4. **Pre-filters.** English-only, dedup, ≤1024-token cap before sampling
   (following MR-Eval). The paper does not state how it handled language or
   duplicates.
5. **Generation engine variance.** Greedy decoding is not bit-identical across
   inference engines; activation extraction on fixed (prompt, response) pairs
   is engine-independent (sec3 README's own note).

## What is NOT a deviation

- The sec3 evaluation condition: bare prompts, no system prompt, greedy
  decoding, judge with the verbatim rubric — matches the paper.
- The evil system prompt is used only in dataset *construction*, as in §3.
