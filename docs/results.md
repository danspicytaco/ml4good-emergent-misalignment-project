# Results — sec3 repro run 1 (2026-10-23)

**Provisional until Dan verifies.** Setting: base `Qwen/Qwen2.5-7B-Instruct`
activations vs `ChicagoHAI/qwen2.5-7b-em-medical-advice` evilness on the rebuilt
Chatbot Verified (n=634 after refusals). Judge: Kimi K3. Paper reference: mean
Spearman ρ = −0.731 / Pearson r = −0.718 across 12 settings (not this setting
specifically).

## Headline

| Extraction | Metric | ρ @ layer 14 | r @ layer 14 | strongest layer |
|---|---|---|---|---|
| response_mean (main text) | cosine | **−0.205** | −0.074 | layer 1: −0.296 |
| prompt_mean | cosine | −0.036 | −0.048 | layer 1: −0.089 |
| last_prompt_tok | cosine | −0.194 | −0.072 | — |
| response_mean | L2 | −0.160 | −0.072 | layer 1: −0.299 |

Same sign as the paper everywhere, much weaker magnitude. Full per-layer tables
and binned means: `data/analysis/bad_medical__chatbot_verified/`.

## Distributional facts that bound interpretation

- **The y-axis barely moved:** EM-model evilness mean 7.7/100, only 28/634 ≥ 50.
  A correlation computed on a near-degenerate outcome is heavily attenuated;
  the weak ρ may say more about this prompt set's failure to elicit EM
  generalization than about the distance predictor.
- Coherence is fine (mean 91.9, 631/638 ≥ 50), so the low evilness is not
  degeneration gibberish.
- Our Chatbot Verified was filtered for evil-elicitation under *explicit
  steering* (evil system prompt) of the base model; the paper's own §4.3 treats
  this as a distinct condition from EM fine-tuning generalization.

## Open questions before any claim

1. What were the EM model's evil scores on the *authors'* Chatbot Verified?
   If their set also yields ~8/100 mean evilness, weak ρ is consistent with the
   paper; if theirs elicits far more evilness, our prompt reconstruction differs
   materially (judge-dependent filter boundary is the prime suspect).
2. Re-run with a second judge (GPT-5.x) on the cached responses: judge-swap
   robustness for both the filter boundary and the scoring.
3. Repeat on Chatbot Random (no filter) — the paper's own control for
   selection circularity.
4. Try settings where the paper expects stronger generalization (e.g. an EM
   model whose training data is closer to open-ended chat, like insecure-code
   or the persona sets).
