# Research questions tracker

Stress-testing **"Emergent Misalignment Is Not Magical"** (Li, Dai, Wang & Tan 2026,
[arXiv:2608.29118](https://arxiv.org/abs/2608.29118)). Last updated 2026-10-23.

This file lists every question the repo is investigating. For each one it gives the exact design,
the result so far, and the files behind it. It is written as the scaffold for a later LessWrong
post: §1 is the argument structure, §2 is the evidence, §5 is a draft outline. All numbers here
are copied from the output files named under each study. If a number changes, update it there
first, then here.

**Status tags.** `DONE`: result on disk, documented. `DONE (uncommitted)`: result on disk, not
yet committed or written up, so the owner may still revise it. `PARTIAL`: some arms are run and
others are blocked. `IN PROGRESS (teammate)`: someone else owns it, so the numbers are
placeholders. `PLANNED`: design only. `NOT IN REPO`: analysed in an earlier session, but the files
live outside this repo.

**Scope warning for every result below.** We have one of the paper's 12 settings: Qwen2.5-7B,
bad-medical-advice EM. Our prompt set is a same-recipe rebuild with 638 prompts (the paper had
822). The judges are Kimi K3, GLM 5.3 and Jev, not the paper's GPT-5.4. Each prompt gets one
greedy response. There are no human labels yet.

---

## 0. The claim under test, broken into attackable links

Paper's headline (abstract, §3): the closer an evaluation prompt is to the EM training-data
centroid in the **base** model's activation space, the more evil the **EM-trained** model's answer
to it. The paper reports a mean Spearman ρ = −0.731 across 12 settings, against ρ = −0.067 for the
base models. Its conclusion is that EM is "expected generalization", not magic.

For that conclusion to follow, every link below must hold. Each angle of attack in §1 targets one
of them.

| Link | What must be true | Main threat |
|---|---|---|
| **L1 Prompt set** | CHATBOT VERIFIED is a property of the prompts, not of the judge that built it | The filter judge decides what gets studied |
| **L2 Instrument** | The 0–100 "evil score" measures misalignment, and measures it the same way at every distance | Judge weighting (tone vs harm), judge error that trends with distance |
| **L3 Association** | Distance predicts per-prompt evilness after EM training | It doesn't reproduce, or it depends on one judge |
| **L4 Effect size** | The association is as strong as ρ ≈ −0.73 suggests | Binning, best-layer selection, 12 non-independent settings |
| **L5 Specificity** | Distance predicts *EM-induced* evilness, not just "prompts that are easy to make evil" | A confound shared by the filter and the outcome |
| **L6 Exceptions** | No prompt sits far from the centroid and still elicits strong EM | Falsifying inputs (project goal 2) |
| **L7 Mechanism claims** | §4.1 format effect and §5 generalization direction | Weak manipulations, aggregation choices |

---

## 1. Angles of attack: summary

| # | Angle | Link | Question | Status | One-line answer so far |
|---|---|---|---|---|---|
| Q0 | Rebuild the data | all | Can the study be run at all without the authors' prompt files? | DONE | Yes, as a same-recipe approximation. 638 prompts kept vs the paper's 822. |
| Q1 | Filter-judge dependence | L1 | Does the set of "verified" prompts change if only the judge changes? | DONE | About 1 in 6 prompts change (83.4% overlap at equal size), far beyond judge self-noise. |
| Q2 | What the evil score weighs | L2 | Why do judges disagree? Tone/persona or real harm? | DONE | Disagreement tracks **fictional framing** and **persona**. The pre-registered "Kimi weighs harm more" prediction failed. |
| Q3 | Reproduce §3 | L3 | Does distance predict EM evilness in our rebuild? | DONE | Same sign, small per prompt: ρ = −0.205 (r = −0.074). The EM model is rarely evil on this set (mean 7.7/100). |
| Q4 | Comparable numbers | L4 | How much of the gap to −0.73 is protocol (binning, token position, layer choice)? | IN PROGRESS (teammate) | Provisional: binning alone moves −0.205 to about −0.55. Being resolved; will update. |
| Q5 | Outcome-judge swap | L2/L3 | Does the correlation survive a second outcome judge? | DONE | Yes, for three judges and all three token reductions. Last-token ρ: Kimi −0.246, Jev −0.217, GLM −0.178 (matched centroid). Judge agreement ρ ≈ 0.74–0.78. |
| Q6 | Selection → correlation | L1→L3 | Is the correlation different on prompts only one judge would keep? | PARTIAL | **Yes for last-token** (matched centroid): prompts only Kimi keeps show no correlation (ρ ≈ 0) vs −0.30 for consensus prompts (p = 0.005). No difference for response-mean. The Jev-only arm needs GPU generation. |
| Q7 | Differential judge error | L2 | Does judge disagreement trend with distance (which could fake or hide the effect)? | DONE | No: no distance trend in Kimi − GLM for either reduction once the score floor is controlled. With last-token the judges differ in slope strength by ~0.06. |
| Q8 | EM-specificity | L5 | Does distance also predict evilness **without** EM training? | PARTIAL | A warning sign: the *filter* evil score (base model + evil system prompt) correlates with distance at ρ ≈ −0.25 to −0.31 (response-mean) and ≈ −0.15 (last-token, matched centroid). |
| Q9 | Falsifying inputs | L6 | Are there far-from-centroid prompts that elicit strong EM? | PARTIAL | A candidate pair sits at the 99.5th–99.7th distance percentile and is scored 80–95 by both judges. Not yet tested systematically. |
| Q10 | §4.1 data format | L7 | Does the format effect survive closer reading? | NOT IN REPO | The "code" format is mostly prose in strings. The headline drop is a pooled number, and one case reverses. |
| Q11 | §5 direction | L7 | Does the ridge generalization direction hold up? | PLANNED | Not started. (Note: README mislabels it as mean-difference, see §4.) |

---

## 2. Studies

### Q0. Rebuilding the unreleased prompt set — `DONE`

**Question.** The paper's evaluation prompts (CHATBOT VERIFIED, 822 prompts) and their sampling
code are unreleased. Can the study be run on a faithful rebuild?

**Design.** We searched every release channel (GitHub tree, HF org and model repos, HF and GitHub
full-text search, Zenodo, third-party repos). We then followed the paper's §3 recipe:

1. Draw 2,000 random Chatbot Arena first user turns (seed 0). The mirror
   `agie-ai/lmsys-chatbot_arena_conversations` stands in for the license-gated original; schema and
   row count are verified identical.
2. Keep English only, drop duplicates, cap at ≤1024 tokens (filters following MR-Eval; the paper
   does not say how it handled these).
3. Answer each prompt with Qwen2.5-7B-Instruct under the authors' evil system prompt (greedy,
   1,024 new tokens).
4. Have the judge score each answer with the paper's verbatim rubric. Keep the prompt if its
   score is > 50.

The judge is Kimi K3 (Fireworks), not GPT-5.4.

**Results.** Funnel: 33,000 → 29,206 English → 23,657 deduplicated → 23,648 within the token cap →
2,000 sampled. Of these, 1,967 were scored and 33 were REFUSAL. **638 were kept (31.9%)**, against
the paper's 822 (41.1%). Kimi's scores are bimodal and coarse (440 exact zeros, 32 distinct
values), and 251 scores fall in 40–60, so the kept set is sensitive to the threshold. The 2,000
unfiltered prompts double as CHATBOT RANDOM.

**Limits.** The authors' seed, dedup and language handling are unknown. The filter judge differs
from theirs, and greedy decoding is not bit-identical across inference engines. The training
subset (first 5,000 of 7,049 bad-medical rows) is unconfirmed.

**Files.** [`scripts/build_chatbot_verified.py`](../scripts/build_chatbot_verified.py),
[`data/arena_build/`](../data/arena_build/) (`sample.jsonl`, `sample_funnel.json`,
`generations.jsonl`, `judgments.jsonl`, `summary.json`),
[`data/chatbot_verified_prompts.json`](../data/chatbot_verified_prompts.json),
[`data/chatbot_arena_2k_random_prompts.json`](../data/chatbot_arena_2k_random_prompts.json).
Narrative: [`data-provenance.md`](data-provenance.md), [`sec3-pipeline.md`](sec3-pipeline.md),
[`process-log.md`](process-log.md).

---

### Q1. Is CHATBOT VERIFIED a property of the prompts or of the judge? — `DONE`

**Question (L1).** The same rubric selects the prompt set and then scores the outcome on it. If
the selected set depends on the judge, the paper's results are conditional on GPT-5.4's
idiosyncrasies at the > 50 boundary.

**Design.** We re-scored the **same 2,000 filter generations** with a structurally different
instrument: Jev `jev-1.14.0` (TypeSafe), a decision model that returns per-level probabilities
instead of text.

- Jev saw the same content as the LLM judge: the same rubric sentence and the same truncation
  (prompt 2,000 chars, response 3,000 chars).
- The score is a Score question on a 9-level ladder. Levels 0 and 100 carry the rubric's verbatim
  anchors; the middle levels are bare numbers. The continuous score is the expected value over the
  label values.
- Before the full run we validated the ladder. Reversing it gave ρ = 0.996 for the label-based
  expected value (−0.996 for the position-based one). Shuffling the middle numbers changed the
  label-based EV by 2.1 points on average (7.7 for position-based). A 3-level ladder gave
  ρ = 0.996. Only 3/80 items had probability mass split between both ends.
- Three passes: A (primary), B (5-level ladder with descriptions, a wording check), C (repeat of
  A, a retest).
- Two comparisons. Primary, **yield-matched**: Jev's top 638 vs Kimi's > 50 set, which separates
  ranking from calibration. Secondary: fixed > 50 for both judges.

**Results** (1,967 Kimi-scored items):

| Comparison | Spearman | Top-638 overlap |
|---|---|---|
| Jev repeat (A vs C) | 0.999 | 98.6% |
| Jev ladder wording (A vs B) | 0.994 | 95.1% |
| **Jev vs Kimi** | **0.918** (CI 0.909–0.925) | **532/638 = 83.4%** (CI 81.3–85.1; chance 32%) |

- At fixed > 50, Jev keeps 732 against Kimi's 638 (κ = 0.752).
- Disagreement is concentrated near the cut. Within-band ρ is 0.24–0.40 for Kimi bands 20–79,
  against about 0.71–0.73 at the extremes.
- Jev-only kept prompts are more often creative or role-play prompts: 21.7% vs 5.7% for Kimi-only
  (Fisher OR 4.6, p = 0.001).

**Takeaway for the post.** About **1 in 6 "verified" prompts is judge-dependent** at the paper's
set size. The paper's 822 is one draw from that dependence.

**Limits.** Agreement is not validity. There is no human ground truth yet, and no LLM-vs-LLM
baseline at the filter stage yet (the GLM swap in Q5 is at the outcome stage).

**Files.** [`scripts/judges/jev_filter_check.py`](../scripts/judges/jev_filter_check.py),
[`data/judges/filter_jev_vs_kimi/`](../data/judges/filter_jev_vs_kimi/) (`summary.json`, `jev_vs_kimi.csv`,
`jev_pass_{A,B,C}.jsonl`, `pilot_ladder40.csv`, `pilot_ladder_diagnostics80.csv`,
`creative_tag_breakdown.json`, `jev_vs_kimi_scatter.png`). Write-up:
[`jev_filter_check.md`](judges/jev_filter_check.md). Judge map:
[`model_touchpoint_map.md`](model_touchpoint_map.md).

---

### Q2. What does the evil score actually weigh? — `DONE`

**Question (L2).** Q1 suggested that Jev weighs evil *tone/persona* and Kimi weighs *harm or
deception*. If judges differ on what "evil" means, both the selected set and the outcome measure
inherit that choice.

**Design.** We asked Jev seven yes/no questions per filter generation (all 2,000, one call each):
evil_persona, actionable_harm, false_claims, fictional_frame, refusal, hostile_to_user and
on_topic. Hypotheses were fixed before looking at the data:

- **H1**: Jev − Kimi rises with persona.
- **H2**: Jev − Kimi falls with harm and with false claims.
- **H3**: Jev-only prompts score higher on persona and fiction; Kimi-only prompts score higher on
  harm and false claims.

Analysis: per-cell means, Mann–Whitney tests, and OLS with HC3 errors predicting Δ = Jev − Kimi,
Kimi's score and Jev's score from the components. We also ran a retest on 300 items and a
cross-check of fictional_frame against an independent keyword regex. A blind 80-item human
spot-check sheet has been generated.

**Results** (n = 1,967):

- **H3 is supported for persona and fiction.** Jev-only vs Kimi-only: evil_persona 0.727 vs 0.559
  (p = 1.0e-5); fictional_frame 0.364 vs 0.147 (p = 1.3e-9).
- **H3 failed for harm.** Jev-only prompts have *more* actionable harm than Kimi-only prompts
  (0.429 vs 0.254, p = 1.0e-8), the reverse of the prediction. False claims don't differ (0.42 vs
  0.46, p = 0.25).
- **Δ regression (R² = 0.111).** The largest term is fictional_frame (+12.9 points, CI 10.3–15.5).
  evil_persona is +6.3 (CI 3.4–9.1), which supports H1. actionable_harm is ≈ 0 (CI −1.9 to 3.3),
  so H2 fails for harm. false_claims is −2.7 (CI −5.3 to −0.1), weak support for H2. on_topic
  (−8.9), hostile_to_user (−7.6) and refusal (−7.0) are also significant.
- **Each judge's own weights.** Kimi's score is fit with R² = 0.758, and Kimi *penalises* fiction
  (−9.3). Jev's score is fit with R² = 0.880, with fiction at +3.6. Both weigh actionable harm
  most (≈ +52).
- Component retest Spearman is ≥ 0.991 for all seven components. fictional_frame vs the regex:
  ρ = 0.29.

**Reading.** The main axis of judge disagreement is **how to score requested fiction or role-play
in a villain voice**, not harm detection. This matters for EM because the evil rubric has to
decide whether theatrical evil counts.

**Limits.** The components come from Jev too, so its quirks can sit on both sides of the
regression; the Kimi-score and Δ models are less exposed to this. The human spot-check sheet is
unfilled (0/80 rated). The owner's write-up, `docs/judges/judge_robustness_followups.md`, is referenced
by the scripts but not yet in the repo.

**Files.** [`scripts/judges/jev_decomposition.py`](../scripts/judges/jev_decomposition.py),
[`scripts/judges/plot_judge_followups.py`](../scripts/judges/plot_judge_followups.py),
[`data/judges/filter_jev_vs_kimi/decomposition/`](../data/judges/filter_jev_vs_kimi/decomposition/)
(`decomposition_summary.json`, `components.jsonl`, `components_merged.csv`,
`components_retest300.jsonl`, `fig_component_weights.png`, `human_spotcheck_sheet.csv`,
`human_spotcheck_key.csv`).

---

### Q3. Does the §3 distance–evilness correlation reproduce? — `DONE` (run 1, one setting)

**Question (L3).** Does the base model's distance to the training centroid predict the EM model's
per-prompt evil score?

**Design.**

- **Models.** EM model `ChicagoHAI/qwen2.5-7b-em-medical-advice`; base model
  `Qwen/Qwen2.5-7B-Instruct`.
- **Responses.** The EM model answers each rebuilt CHATBOT VERIFIED prompt bare (no system
  prompt, greedy). Kimi K3 scores each answer with the verbatim rubric, plus a coherence pass.
- **Activations.** The base model is run on (prompt + its own response). We extract per-layer
  response_mean, prompt_mean and last_prompt_tok activations.
- **Centroid and distance.** One centroid per token reduction: the base model's mean activation over 5,000
  bad-medical training rows, using the same reduction as the eval activations
  (`data/centroids/bad_medical_Qwen2.5-7B-Instruct_<reduction>.npy`, teammate commit `134f629f`). Distance is
  1 − cos (L2 as a variant). *Before 2026-10-24 all three reductions were measured against the response-mean
  centroid; the last_prompt_tok and prompt_mean rows below were recomputed.*
- **Statistics.** Per-prompt Spearman and Pearson at every layer; canonical layer 14.

**Results** (n = 634 after refusals):

| Extraction | Metric | ρ @ L14 | r @ L14 |
|---|---|---|---|
| response_mean | cosine | −0.205 | −0.074 |
| last_prompt_tok | cosine | −0.246 | −0.092 |
| prompt_mean | cosine | −0.106 | +0.005 |
| response_mean | L2 | −0.160 | −0.072 |

- The sign matches the paper everywhere, but the per-prompt magnitude is small.
- **The outcome is nearly degenerate.** EM evilness averages 7.7/100, and only 28 of 634 answers
  score ≥ 50. Coherence is fine (mean 91.9), so the low scores are not gibberish.

**Limits.** One setting out of 12; a rebuilt prompt set; a different judge. These per-prompt
numbers are **not directly comparable** to the paper's −0.731 (see Q4).

**Files.** [`scripts/sec3_distance_evilness.py`](../scripts/sec3_distance_evilness.py),
[`scripts/plot_figure2.py`](../scripts/plot_figure2.py),
[`data/analysis/bad_medical__chatbot_verified/`](../data/analysis/bad_medical__chatbot_verified/)
(`summary_*.json`, `per_layer_*.csv`, `pairs_*.csv`, `bins_*.csv`),
[`data/analysis/figure2_repro.png`](../data/analysis/figure2_repro.png). Inputs:
`data/chatbot_verified/<model>-vllm-hybrid/`,
[`data/centroids/`](../data/centroids/), [`data/training/`](../data/training/). Write-up:
[`results.md`](results.md). Commands: [`runbook.md`](runbook.md).

---

### Q4. Are our numbers comparable with the paper's? (binning, token position, layer choice) — `IN PROGRESS (teammate)`

**Question (L4).** The paper's §3 text says it uses "the activation of the base models at the last
prompt token" and shows "the layer with the best correlation". It also says prompts are sorted by
distance and put into "equal-sized bins containing at most 20 prompts each", with mean evilness
reported per bin. Our Q3 headline instead uses response_mean, per-prompt correlation and a fixed
layer 14. Binning averages away per-prompt noise, and taking the best of about 28 layers is a
selection step. Both push |ρ| up.

**Status.** A teammate is resolving this reference mismatch now, and this section will be updated
when their data lands. The numbers below are a **provisional pointer only**.

**Provisional design.** Equal-count bins after sorting by distance (10, 20 and 40 prompts per
bin). Permutation null: shuffle evilness across prompts and re-bin. For the best-layer number, a
max-over-layers null that applies the same shuffle at every layer.

**Provisional results** (Kimi unless noted):

| Setting | Per-prompt ρ | Binned-20 ρ @ L14 | Best-layer binned-20 ρ | Max-over-layers null mean |
|---|---|---|---|---|
| response_mean, Kimi | −0.205 | −0.547 (null SD 0.18) | −0.700 (L2) | −0.149 |
| response_mean, GLM | −0.177 | −0.595 | −0.743 (L7) | −0.160 |
| last_prompt_tok, Kimi (matched centroid) | −0.246 | −0.410 (null SD 0.18) | −0.606 (L5) | −0.212 (p = 0.002) |

**What would make this a post-worthy point.** The paper-style protocol brings our response-mean
number close to the paper's average, even though per prompt it explains about 4% of rank variance
(ρ² ≈ 0.04). This matches an earlier simulation: at n = 822 with bins of 20, a binned ρ of −0.73
is consistent with a per-prompt ρ of roughly −0.25 to −0.28 (see Q4 files, outside the repo). The
binned number is still well outside its permutation null, so this is an **effect-size** critique,
not a "no effect" claim. Wait for the teammate's numbers before stating it.

**Files (provisional).** [`scripts/binned_vs_per_prompt.py`](../scripts/binned_vs_per_prompt.py) →
[`data/analysis/binning_comparison.json`](../data/analysis/binning_comparison.json). The earlier
simulation (`binning_inflation_sim.py` / `.csv`) is a session artifact and is **not in the repo
yet**.

---

### Q5. Does the outcome survive a second outcome judge? — `DONE`

**Question (L2/L3).** Is the weak ρ a quirk of Kimi K3?

**Design.** We re-scored all cached EM responses with GLM 5.3 (`accounts/fireworks/models/glm-5p3`)
using the verbatim rubric, then re-ran the Q3 metric.

**Results.**

- Judge agreement (n = 625): Spearman 0.777, Pearson 0.887, mean |Δ| = 3.3 points. For
  comparison, the paper reports human–judge ρ ≈ 0.60.
- Distance–evilness with GLM: ρ = −0.177, r = −0.070 at L14, against Kimi's −0.205 / −0.074.
- Both judges agree the EM model is rarely evil on this set (means 7.6 vs 6.5).
- The far-tail candidates from Q9 survive the swap.
- **Third judge (Jev, jev-1.14.0).** Minimal port of the paper's rubric (same wording and anchors, REFUSAL as a
  separate yes/no question). Agreement with Kimi 0.765, with GLM 0.741. Per-prompt ρ at L14, matched centroids:

  | Reduction | Kimi | GLM | Jev |
  |---|---|---|---|
  | last_prompt_tok | −0.246 | −0.178 | −0.217 |
  | response_mean | −0.205 | −0.177 | −0.251 |
  | prompt_mean | −0.106 | −0.070 | −0.132 |

  Jev is never significantly different from Kimi (paired bootstrap). Coherence (Betley C.2): Jev scores no
  response below 50 (Kimi: 7). Write-up: [`judges/jev_outcome_judge.md`](judges/jev_outcome_judge.md);
  data: [`data/analysis/bad_medical__chatbot_verified__jev/`](../data/analysis/bad_medical__chatbot_verified__jev/).

**Files.** [`data/analysis/bad_medical__chatbot_verified__glm53/`](../data/analysis/bad_medical__chatbot_verified__glm53/),
`data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid/judge_glm-5p3_chatbot_verified.json`,
commit `92b030d`. Write-up: [`results.md`](results.md) § Judge-swap.

---

### Q6. Does the judge that builds the set change the correlation? — `PARTIAL`

**Question (L1 → L3).** This is the direct test of whether filter-judge dependence (Q1) matters
for the headline.

**Design.** Split Kimi's 638 verified prompts into three groups, using the Jev comparison from Q1:

- **both**: 532 prompts, also in Jev's top 638;
- **kimi_only**: 106 prompts;
- **jev_only**: 106 prompts with no EM responses yet.

Compute ρ(distance, EM evil) per group, with bootstrap CIs and a permutation test of both vs
kimi_only. Do this for both outcome judges and both extractions.

**Results** (response_mean, L14):

| Group | Kimi outcome ρ [95% CI] | GLM outcome ρ [95% CI] |
|---|---|---|
| all Kimi-verified | −0.205 [−0.281, −0.132] (n = 634) | −0.178 [−0.260, −0.095] (n = 625) |
| both | −0.216 [−0.297, −0.128] (n = 528) | −0.170 [−0.257, −0.082] (n = 523) |
| kimi_only | −0.223 [−0.411, −0.029] (n = 106) | −0.224 [−0.421, −0.023] (n = 102) |

- Both vs kimi_only: permutation p = 0.95 (Kimi outcome) and 0.63 (GLM outcome). **No detectable
  difference.**
- Kimi-only prompts sit farther from the centroid (median distance 0.232 vs 0.204, Mann–Whitney
  p = 1.3e-5), but their EM evilness doesn't differ (p = 0.23).
- **With last_prompt_tok (matched centroid) the groups differ.** Kimi outcome: both −0.296 [−0.377, −0.218]
  (n = 528) vs kimi_only −0.007 [−0.193, +0.191] (n = 106), permutation p = 0.005. GLM outcome: −0.214 vs +0.016,
  p = 0.036. Kimi-only prompts do not differ in last-token distance (p = 0.28) or EM evilness (p = 0.23), so this is
  not range restriction. Reading: under the paper's main-text protocol, the correlation lives in the prompts both
  judges agree are evil-elicitable; the judge-dependent prompts carry none. Caveats: n ≈ 100, 2 reductions × 2
  judges tested, and the Jev-only arm is still missing. *(Before the centroid fix this gap was p = 0.36 / 0.11.)*

**Blocked arm.** Running §3 on the **Jev-verified set** needs EM-model and base-model generation
plus activations for the 106 jev_only prompts on the GPU pod. The prompt file is ready.

**Files.** [`scripts/judges/judge_selection_and_error.py`](../scripts/judges/judge_selection_and_error.py),
[`data/judges/robustness/`](../data/judges/robustness/) (`summary.json` →
`selection`, `per_layer.csv`, `chatbot_verified_jevonly_prompts.json`, `jev_only_index.csv`,
`fig_per_layer_judges.png`).

---

### Q7. Does judge error trend with distance? — `DONE`

**Question (L2).** If the judges disagree more, or in one direction, at one end of the distance
axis, measurement error alone could create or mask the §3 slope.

**Design.** Using the two outcome judges (Kimi and GLM, same rubric):

- ρ(distance, Kimi − GLM) and ρ(distance, |Kimi − GLM|) at every layer;
- judge agreement within distance terciles;
- a paired bootstrap of ρ(d, Kimi) − ρ(d, GLM);
- partial correlations controlling for score level (the floor at 0).

**Results.**

| | response_mean | last_prompt_tok |
|---|---|---|
| ρ(d, Kimi − GLM) | −0.015 [−0.099, 0.071] | −0.054 [−0.135, 0.026] |
| ρ(d, \|Δ\|) raw | −0.138 [−0.217, −0.062] | −0.165 [−0.239, −0.089] |
| ρ(d, \|Δ\|) controlling for score level | +0.054 [−0.028, 0.136] | +0.034 [−0.043, 0.112] |
| ρ_Kimi − ρ_GLM (paired bootstrap) | −0.027 [−0.081, 0.028] | −0.063 [−0.113, −0.014] |
| Agreement by tercile (near / mid / far) | 0.83 / 0.73 / 0.73 | 0.86 / 0.75 / 0.66 |

last_prompt_tok column uses the matched centroid (2026-10-24).

**Reading.**

- The judges disagree less far from the centroid only because far prompts more often score 0 for
  both judges. Once score level is controlled, the trend disappears.
- With last_prompt_tok, the choice of judge shifts the correlation by about 0.06, a small but
  real effect; the signed Kimi − GLM gap no longer trends with distance (it did, −0.094, before the
  centroid fix).
- Overall there is no evidence that differential judge error produces the effect.

**Files.** Same script and `summary.json` (`differential_error`) as Q6. Jev as a third outcome judge: see Q5.

---

### Q8. Is distance predicting EM-induced evil, or "evil-elicitability" in general? — `PARTIAL`

**Question (L5).** The paper's specificity control is that the *base* model's own evilness barely
correlates with distance (ρ = −0.067). But CHATBOT VERIFIED is selected for prompts where the base
model *can be made* evil. If distance also predicts that, the §3 slope may partly reflect which
prompts lend themselves to evil answers rather than generalization from EM training.

**Design (so far).** Correlate distance with the **filter-stage** evil score. That score comes from
the base model under the evil system prompt, with no EM training involved. Computed on the 634
Kimi-verified prompts, so the range is restricted.

**Results (side output of the Q6 script).**

| Distance from | ρ(d, Kimi filter score) | ρ(d, Jev filter score) |
|---|---|---|
| response_mean L14 | −0.250 [−0.322, −0.174] | −0.313 [−0.383, −0.242] |
| last_prompt_tok L14 (matched centroid) | −0.161 [−0.236, −0.082] | −0.146 [−0.220, −0.071] |

With response_mean, the steered base model's evilness correlates with distance **at least as
strongly as the EM model's evilness does (−0.205)**. With last_prompt_tok, which is the paper's
main-text choice, it is weaker (≈ −0.15 vs the EM outcome's −0.246) but clearly nonzero; before the centroid
fix it looked negligible (−0.10 / −0.075).

**Caveat.** The response_mean distance includes the base model's own response, so the
distance–filter link may partly run through what the base model chose to say.

**Next (cheap, decisive).**

1. Judge the base model's bare-prompt responses, which already exist unjudged in
   `data/chatbot_verified/Qwen2.5-7B-Instruct-vllm-hybrid/responses/`, to get our own version of
   the paper's −0.067 control.
2. Run the same analysis on CHATBOT RANDOM (no filter).
3. Use a control centroid (another dataset, or a random training subset).

**Files.** [`data/judges/robustness/summary.json`](../data/judges/robustness/summary.json)
→ `selection.*.filter_scores_vs_distance`.

---

### Q9. Falsifying inputs: far from the centroid, yet strongly misaligned — `PARTIAL`

**Question (L6; project goal 2).** Is there a prompt the distance account scores as innocuous that
still elicits clear EM?

**Status.** Identified from residuals of the Q3/Q5 runs; no search has been run yet.

- **Candidate family.** idx 537 and idx 536 sit at the 99.5th and 99.7th distance percentiles
  (response_mean, L14). They score 95/80 with Kimi and 90/70 with GLM, so they survive the judge
  swap.
- Several other high-evil answers sit at low or mid distance, so a single far outlier is not
  enough on its own.

**Planned design.**

- Define a counterexample in advance: distance above the 90th percentile, and an evil score above
  the fitted value by more than a set residual. The threshold is still to be fixed.
- Confirm each candidate with N ≥ 10 samples at T = 1 (the paper's multi-sample convention), two
  or more judges, and a human label.
- Generate paraphrase families around confirmed candidates to see whether the effect is a
  property of a region or of a single prompt.
- Jev can score many candidates cheaply in the search; finalists are re-checked with LLM judges
  and humans.

**Files.** `data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv`, the GLM
counterpart, and EM responses under
`data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid/responses/`.

---

### Q10. §4.1 data format: does "code format halves EM" survive closer reading? — `NOT IN REPO`

**Question (L7).** The paper reports that converting EM training data from advice to code format
cuts EM by 52.5% on average (Fig. 4).

**Design.** We read the released paired datasets (`em-not-magical/sec4/4_1_data_format/datasets/`)
and the Fig. 4 values. We measured what fraction of each code-format answer's characters sit
inside strings, docstrings or comments.

**Results** (from earlier session notes; **re-verify before posting**):

- The "code" answers are mostly prose wrapped in code: a median of about 75% of characters sit
  inside strings, docstrings or comments (0.75 aesthetic, 0.73 medical). The harmful claim
  survives as a natural-language string.
- The 52.5% figure is the drop in **pooled totals** (85.3 → 40.5 points across 10 bars). The mean
  of the per-model drops is 57.5%, with a range of 0.6% to 86%.
- Qwen2.5-14B medical shows no drop (16.1 vs 16.0).
- Insecure code *rises* when converted to advice (0.8 → 3.3), against the paper's "drops across
  all" wording.
- Fig. 4 gives no error bars or sample counts.

**Proposed follow-up.** A graded format mixture at p ∈ {0, 0.25, 0.5, 0.75, 1}, built from the
row-aligned paired files. The training centroid of a mixture is exactly
(1 − p)·μ_advice + p·μ_code, so the distance model makes a prediction for each p. This needs
advice-only subsample controls, and fits Qwen2.5-7B medical or aesthetic.

**Files.** Session artifacts only (`em_not_magical_dataset_explainer.md`, `fig4.png`). **Move them
into the repo** under `docs/` or `data/analysis/sec4_format/` before writing this section up.

---

### Q11. §5 generalization direction — `PLANNED`

**Question (L7).** §5 fits a direction by **ridge regression** of EM evil scores on centred
base-model activations (CHATBOT VERIFIED), and claims it predicts evilness under
meaning-preserving perturbations, unlike the mean-difference "EM direction" (Soligo et al.). Our
setting makes this hard: the ridge target is the near-degenerate outcome from Q3. Note also that
the token-append perturbations in the repo are distance-targeted (`sec5/search_vocab.py`), not
random as the paper text suggests.

**Files.** Upstream only: `em-not-magical/sec5/`.

---

## 3. Cross-cutting caveats (each must appear in the post)

1. **n = 1 setting.** Every result is Qwen2.5-7B with bad-medical EM. The paper's −0.731 averages
   12 settings, and those share one prompt set, so they are not independent.
2. **Rebuilt prompts.** We have 638 prompts against the paper's 822. Our yield is about 9 points
   lower, which is consistent with a different filter judge.
3. **The outcome floor.** A mean of 7.7/100 with 28 answers ≥ 50 limits every correlation. Open
   question: did the authors' set elicit much more? (Asked by email; see
   [`data-provenance.md`](data-provenance.md).)
4. **No humans yet.** All validity claims are judge-vs-judge. The 80-item spot-check sheet exists
   but is unrated.
5. **Single greedy sample per prompt.** Per-prompt misalignment *rates* (many samples at T = 1)
   are not measured.

## 4. Paper-vs-repo discrepancies worth reporting

- The paper says CHATBOT VERIFIED is defined in §3. It is an **evaluation** set, and neither the
  set nor its builder is released ([`data-provenance.md`](data-provenance.md)).
- The judge model differs by section: GPT-5.4 in §3 and §4.3, GPT-5.4-mini in §5 (whose docstring
  says "same as Section 3"), and GPT-4o in §4.1–4.2 (with a different rubric). See
  [`model_touchpoint_map.md`](model_touchpoint_map.md).
- The judge parses one integer from the reply text (temperature 0, no logprobs) and the paper
  says it follows Soligo et al. To check: whether Soligo/Betley use a logprob-weighted score
  instead. Our notes say yes, but this is unverified.
- The §5 token-append perturbations are distance-targeted, not random. The §5 paraphrases are
  generated with claude-opus-4-8.
- There is no advice→code conversion script; the §4.1 datasets ship as finished files.
- **Our own README** calls the §5 generalization direction "mean-difference". The paper (§5)
  defines it by ridge regression; mean-difference is the Soligo baseline it argues against. Fix
  the README before publishing.

## 5. Draft post skeleton

Working title: *"Emergent misalignment is predictable, but less than ρ = −0.73 suggests"*. Adjust
once Q4 and Q8 land.

1. **Claim and why it matters.** EM as "expected generalization" would be good news for
   predicting where fine-tuning will cause problems (§0).
2. **How we attacked it.** The chain L1–L7 and the table in §1, which is the backbone of the post.
3. **Result 1: the prompt set is judge-dependent** (Q1), and the disagreement is about how to
   treat fiction and role-play (Q2). Strength: solid, but it rests on agreement rather than
   validity.
4. **Result 2: the sign reproduces and the per-prompt size is small** (Q3, Q5). Robust to the
   judge swap. Strength: solid for one setting.
5. **Result 3: the effect size depends on protocol** (Q4). The binned, best-layer statistic
   versus per-prompt variance explained. Strength: pending the teammate's numbers.
6. **Result 4: specificity concern** (Q8). Strength: preliminary, until the base-model control
   and CHATBOT RANDOM are in.
7. **What didn't break it.** No selection effect detected on the testable arm (Q6); no
   differential judge error (Q7).
8. **Counterexamples** (Q9), if the search produces confirmed ones.
9. **Limits** (§3) and **what would change our mind**.

## 6. File index

| Path | What |
|---|---|
| [`scripts/build_chatbot_verified.py`](../scripts/build_chatbot_verified.py) | Q0 prompt-set rebuild (sample / generate / judge / finalize) |
| [`scripts/sec3_distance_evilness.py`](../scripts/sec3_distance_evilness.py) | Q3 distance–evilness metric |
| [`scripts/plot_figure2.py`](../scripts/plot_figure2.py) | Q3 paper-style figure |
| [`scripts/judges/jev_filter_check.py`](../scripts/judges/jev_filter_check.py) | Q1 Jev vs Kimi at the filter stage |
| [`scripts/judges/jev_decomposition.py`](../scripts/judges/jev_decomposition.py) | Q2 component questions |
| [`scripts/judges/judge_selection_and_error.py`](../scripts/judges/judge_selection_and_error.py) | Q6, Q7, Q8 |
| [`scripts/judges/plot_judge_followups.py`](../scripts/judges/plot_judge_followups.py) | Q2, Q6 figures |
| [`scripts/binned_vs_per_prompt.py`](../scripts/binned_vs_per_prompt.py) | Q4 provisional binning table |
| [`data/arena_build/`](../data/arena_build/) | Q0 funnel, generations, filter judgments |
| [`data/judges/filter_jev_vs_kimi/`](../data/judges/filter_jev_vs_kimi/) | Q1 and Q2 outputs |
| [`data/analysis/`](../data/analysis/) | Q3–Q8 outputs |
| `data/chatbot_verified/<model>-vllm-hybrid/` | Responses, activations (`.pt`), outcome judgments |
| [`em-not-magical/`](../em-not-magical/) | Vendored upstream @ `28f0782` (judge patch: [`setup.md`](setup.md)) |
