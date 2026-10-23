# Handoff: next runs (GPU, centroids, API)

For an agent picking up the open studies in [research_questions.md](research_questions.md).
Written 2026-10-23 on branch `jev-filter-check` (head `893aa8d` when written). Do the tasks in the
order given; each one lists what it needs, exact steps, outputs, acceptance checks, and what to report.

## 0. Read first (15 minutes)

1. [research_questions.md](research_questions.md): the master tracker (claim links L1–L7, studies Q0–Q11).
   **Every task below updates one Q section there.**
2. [results.md](results.md): the sec3 run on `main` (EM model, distances, judges).
3. [jev_filter_check.md](jev_filter_check.md) and [judge_robustness_followups.md](judge_robustness_followups.md):
   the filter-judge studies (Q1, Q2, Q6, Q7, Q8).
4. [runbook.md](runbook.md) and [setup.md](setup.md): pod alias `em-sec3`, `.env` layout, judge patch. The
   commands below follow the runbook's conventions.

Setting for everything here: base `Qwen/Qwen2.5-7B-Instruct`, EM model `ChicagoHAI/qwen2.5-7b-em-medical-advice`,
bad-medical training data (`data/training/bad_medical_advice.jsonl`, first 5,000 rows), distance 1 − cos,
canonical layer 14. The EM outcome is near the floor (mean 7.7/100, 28/634 ≥ 50), so expect noisy subset results.

## 1. Ground rules

- **Never overwrite existing outputs.** Write every new result to a new directory or filename (names are given
  per task). Existing `data/analysis/*`, `data/jev_filter/*` and `data/chatbot_verified/*` results are cited in the docs.
- **The judge script overwrites its output.** `em-not-magical/sec3/judge_responses.py` always writes
  `<EM_DATA_DIR>/<dataset>/<model_name>/judge_<dataset>.json`. After each judge run, rename it to
  `judge_<judge-tag>_<dataset>.json` (tags in use: `kimi-k3`, `glm-5p3`; use `jev-1.14.0` for Jev) **before**
  running another judge.
- **Keys:** never write a key into a file, log, commit or doc. Judge keys come from `em-not-magical/.env`
  (gitignored, see setup.md). The Jev key is read from the env var `TYPESAFE_API_KEY` (or `JEVKEY`).
- **Ownership:** Q4 (the protocol mismatch: binning, token position, layer choice) belongs to a teammate. Task 6
  below touches it; **ask that teammate before running it**, and don't edit Q4's numbers.
- **Don't touch other people's uncommitted files.** At the time of writing, `docs/research_questions.md`,
  `scripts/binned_vs_per_prompt.py` and `data/analysis/binning_comparison.json` were untracked work from another
  session. Don't commit or delete them without asking. Updating Q sections in the tracker is expected.
- **Vendored code:** keep changes to `em-not-magical/` minimal and mark them `# local patch (<branch>)`, as the
  existing ones are. Prefer new scripts under `scripts/`.
- **Large files:** the 638-prompt activations are about 760 MB per model directory and are tracked in git per
  project convention. Task 5 adds about 2.5 GB per model. **Ask the user before committing more than 1 GB.**
  `pairs_*.csv` plus `summary_*.json` are enough to reproduce every statistic.
- **Pods:** per project convention, delete paid pods once outputs are verified locally (runbook § Teardown).
- **Report deviations.** If you change a design (sample size, judge, layer, threshold), write it in the task's
  tracker section under "Deviations".
- Don't push. Commit on a branch with a message naming the task numbers.

## 2. Task overview

| Task | Tracker | Needs GPU? | New centroid? | Needs API key | Priority |
|---|---|---|---|---|---|
| T1 Base-model control (our −0.067) | Q8 step 1 | no | no | Kimi (and GLM) | **1, cheapest decisive check** |
| T2 Filter-stage LLM-vs-LLM baseline | Q1 | no | no | GLM | 2 |
| T3 Jev as third outcome judge | Q5/Q7 | no | no | Jev | **DONE** (docs/jev_outcome_judge.md) |
| T4 Jev-only arm of the selection test | Q6 | **yes** | no | Kimi (+GLM) | 4 |
| T5 CHATBOT RANDOM (no filter) | Q8 step 2 | **yes** | no | Kimi (+GLM) | 5 |
| T6 Matched last-token / prompt-mean centroids | Q4 (teammate), Q3, Q7 | **yes** | **yes** | none | 6, ask Q4 owner first |
| T7 Control centroids | Q8 step 3 | **yes** (two of three) | **yes** | none | 7 |
| T8 Confirm far-tail counterexamples | Q9 | **yes** | no | judge | 8 |
| — Format mixture, other settings | Q10, more of the 12 | **yes, fine-tuning** | yes | — | not in this handoff |

GPU sizing: Qwen2.5-7B in bf16 is about 15 GB, so one 24 GB card is enough for T4–T8. **Batch T4, T5, T6 and T7
into one pod session.** Copy all inputs up first (§T4 step 1, §T5 step 1, §T6 step 1, §T7 step 1), run, then pull
everything back once.

---

## T1. Base-model control: does the *base* model's evilness correlate with distance? (Q8 step 1, no GPU)

**Why.** The paper's specificity control: base-model evilness vs distance gives ρ = −0.067, far below the EM
models' −0.73. Q8 found that the *evil-steered* base model's filter score correlates with distance about as
strongly as the EM outcome does (ρ ≈ −0.25 to −0.31 with response_mean). We need our own unsteered base-model number.

**Inputs (already on disk).** `data/chatbot_verified/Qwen2.5-7B-Instruct-vllm-hybrid/responses/` (638 bare-prompt
base responses, unjudged), the existing centroid, existing base activations in the same directory.

**Steps.**
1. Load `em-not-magical/.env` into the environment (as in runbook §1), with `JUDGE_MODEL=accounts/fireworks/models/kimi-k3`.
2. `cd em-not-magical/sec3` and run
   `python judge_responses.py --dataset chatbot_verified --model_name Qwen2.5-7B-Instruct-vllm-hybrid`.
3. Rename the output `data/chatbot_verified/Qwen2.5-7B-Instruct-vllm-hybrid/judge_chatbot_verified.json` to
   `judge_kimi-k3_chatbot_verified.json`.
4. Repeat steps 2–3 with `JUDGE_MODEL=accounts/fireworks/models/glm-5p3` and tag `glm-5p3` (set
   `EM_JUDGE_WORKERS` lower if you hit 429s; see results.md).
5. For each judge × activation (`response_mean`, `last_prompt_tok`), run `scripts/sec3_distance_evilness.py`
   (runbook §5) with `--acts_dir data/chatbot_verified/Qwen2.5-7B-Instruct-vllm-hybrid`, the existing centroid,
   `--judge <the renamed base judge file>`, `--activation <...>`, and
   `--out_dir data/analysis/base_control__chatbot_verified__<judge-tag>`.

**Outputs.** Two judge files and two analysis dirs as named above.

**Acceptance checks.** 638 records per judge file; report the count of `score is None` (refusals/unparsed).
Kimi K3 may return empty replies if its reasoning exhausts the token budget: the builder uses
`max_tokens=3000` for this (setup.md); if the None count is high, check `raw_judgment`.

**Report (tracker Q8).** ρ(distance, base evil) at layer 14 for both judges and activations, with n; the base mean
evil score; side by side with the EM ρ (−0.205 Kimi / −0.178 GLM, response_mean) and the paper's −0.067.
Interpretation rule, stated in advance: if base ρ is close to the EM ρ, distance is not EM-specific in our setting.
Expect a floor problem (an unsteered base model is rarely evil), and say so if the base scores are mostly 0.

---

## T2. Filter-stage LLM-vs-LLM baseline (Q1, no GPU)

**Why.** Jev vs Kimi kept-set overlap is 83.4% (532/638). We don't know what two *LLM* judges give at the filter
stage, so we can't tell whether 83% is unusual. (The GLM swap in Q5 was at the outcome stage.)

**Caution.** `scripts/build_chatbot_verified.py judge` always writes `<EM_DATA_DIR>/arena_build/judgments.jsonl`,
and `finalize` rewrites `chatbot_verified_prompts.json`. **Do not run it against the real `data/`.**

**Steps.**
1. Create a scratch data dir outside `data/`, e.g. `scratch/filter_glm/arena_build/`, and copy in
   `data/arena_build/sample.jsonl` and `data/arena_build/generations.jsonl`.
2. Run the builder's `judge` stage with `EM_DATA_DIR=<abs path to scratch/filter_glm>` and
   `JUDGE_MODEL=accounts/fireworks/models/glm-5p3`. Do **not** run `finalize`.
3. Copy the result to `data/jev_filter/filter_glm-5p3_judgments.jsonl` (new name), then delete the scratch dir.
4. Compute, on the prompts both LLM judges scored: Spearman(Kimi, GLM); kept-set overlap at equal size (GLM's top
   638 vs Kimi's > 50 set) with a bootstrap CI; counts at fixed > 50 and Cohen's κ. Use the same definitions as
   `cmd_analyze` in `scripts/jev_filter_check.py` (copy its `_topk` tie-breaking: score descending, then idx).
   Also give the three-way overlap (Kimi ∩ GLM ∩ Jev top 638).

**Report (tracker Q1).** A row "GLM vs Kimi" in the Q1 table. Interpretation: if GLM–Kimi overlap is about 83%,
Jev is just another judge; if it is much higher, the Jev gap is a judge-type effect. For reference, Jev against
itself with different ladder wording gives 95.1%, and a repeat run gives 98.6%.

---

## T3. (Optional) Jev as a third outcome judge (Q5/Q7, no GPU)

> **DONE 2026-10-23 - do not re-run.** Evilness and Betley C.2 coherence scored with `jev-1.14.0`
> (`scripts/jev_outcome_judge.py`, `scripts/jev_outcome_metric.py`); results in `docs/jev_outcome_judge.md`.
> Step 3 (Q7 differential-error re-run with Jev) is still open.

**Why.** Q7 found the `last_prompt_tok` correlation is judge-dependent (Kimi ρ −0.190 vs GLM −0.112). A third
judge shows whether that is Kimi-specific.

**Steps.**
1. Write `scripts/jev_outcome_judge.py`. Reuse `score_one`, `LADDERS["L9"]`, `INSTRUCTIONS`, truncation and the
   pinned `MODEL = "jev-1.14.0"` from `scripts/jev_filter_check.py`.
   - Input: `data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid/judge_kimi-k3_chatbot_verified.json`,
     whose records hold `idx`, `prompt` and `response`.
   - Output: `judge_jev-1.14.0_chatbot_verified.json` in the same directory, as a list of `{idx, prompt, response,
     score}` with score = expected value on 0–100. `sec3_distance_evilness.py --judge` reads this format.
   - Cost: about 0.3M input tokens.
2. Run the metric (runbook §5) for both activations into
   `data/analysis/bad_medical__chatbot_verified__jev`.
3. Re-run Q7's differential-error analysis with Jev against each LLM judge. `scripts/judge_selection_and_error.py`
   hard-codes `JUDGES = {kimi, glm}`; add a CLI option rather than editing the dict in place, and write the
   results to a new output dir.

**Report (tracker Q5 and Q7).** Jev's ρ at layer 14 for both activations, agreement with Kimi and with GLM, and
the paired ρ differences.

---

## T4. Jev-only arm of the selection test (Q6, GPU, existing centroid)

**Why.** We ran §3 on Kimi's verified set and its 532-prompt overlap with Jev's set. The 106 prompts that only Jev
keeps have no EM responses, so ρ on the **Jev-verified set** is still unknown. This matters because the filter is
distance-correlated (Q8): a Jev-built set sits closer to the centroid.

**Inputs (ready).** `data/analysis/judge_robustness/chatbot_verified_jevonly_prompts.json` (106 prompts, same
format as `chatbot_verified_prompts.json`). The position → Arena idx map is in `jev_only_index.csv`. The vendored
scripts already accept the dataset name `chatbot_verified_jevonly`: `DATASET_MAP` in
`sec3/hybrid_extract_activations.py`, and a `JUDGE_PROMPTS` alias in `sec3/judge_responses.py`.

**Steps.**
1. Copy up:
   `scp data/analysis/judge_robustness/chatbot_verified_jevonly_prompts.json em-sec3:/workspace/em/data/` and
   `scp em-not-magical/sec3/hybrid_extract_activations.py em-sec3:/workspace/em/` (the patched copy).
2. On the pod, as runbook §2, for `M` in `Qwen/Qwen2.5-7B-Instruct` and `ChicagoHAI/qwen2.5-7b-em-medical-advice`:
   `hybrid_extract_activations.py --model $M --prompts chatbot_verified_jevonly_prompts.json --device cuda:0 --phase gen`,
   then the same with `--phase extract`.
3. Pull `/workspace/em/data/chatbot_verified_jevonly/*-vllm-hybrid` to `data/chatbot_verified_jevonly/`.
4. Judge the EM responses locally (runbook §4) with `--dataset chatbot_verified_jevonly --model_name
   qwen2.5-7b-em-medical-advice-vllm-hybrid`, for Kimi then GLM, renaming the output after each run (§1).
5. Run the metric into `data/analysis/bad_medical__chatbot_verified_jevonly__<judge-tag>`.
6. Add a section to `scripts/judge_selection_and_error.py` (or a new script):
   - Build the **Jev-verified set** = the 528 "both" prompts (from the existing pairs) + the 106 Jev-only prompts
     (from the new pairs). Indices in the two pairs files are positions in their own prompt files, so join through
     prompt text → Arena idx, as the existing script does.
   - Report ρ(d, EM evil) at layer 14 on the Jev-verified set vs the Kimi-verified set (−0.205), bootstrapping over
     the union of prompts.
   - Report the distance distributions of Jev-only vs Kimi-only vs both.
   - Report the EM evil mean and n ≥ 50 per group.

**Acceptance checks.** Per model dir: 106 responses and 318 `.pt` files (3 per prompt). Spot-check that the shape
is `[29, 3584]`.

**Report (tracker Q6).** Update the table with the jev_only row and the Jev-verified vs Kimi-verified comparison.
Change the status to DONE.

---

## T5. CHATBOT RANDOM: does the correlation survive with no filter? (Q8 step 2, GPU, existing centroid)

**Why.** It removes the filter judge entirely: the paper's own control for selection circularity. It also gives
the filter-score vs distance relation on all 2,000 prompts instead of the range-restricted 634.

**Inputs.** `data/chatbot_arena_2k_random_prompts.json` (2,000). It is already in `DATASET_MAP`
(`chatbot_arena_2k_random`) and aliased in `JUDGE_PROMPTS`.

**Steps.**
1. Copy the prompts file up.
2. Run runbook §2 for both models with `--prompts chatbot_arena_2k_random_prompts.json`.
3. Pull back `data/chatbot_arena_2k_random/*-vllm-hybrid`. **Check sizes and ask before committing (§1).**
4. Judge the EM responses (Kimi, then GLM if budget allows) with `--dataset chatbot_arena_2k_random`, renaming
   after each run. Also judge the base responses with Kimi (this extends T1 to the unfiltered set).
5. Run the metric into `data/analysis/bad_medical__chatbot_arena_2k_random__<judge-tag>` (EM) and
   `data/analysis/base_control__chatbot_arena_2k_random__kimi-k3` (base).
6. Using `data/arena_build/judgments.jsonl` (Kimi filter scores) and `data/jev_filter/jev_vs_kimi.csv` (Jev filter
   scores), compute ρ(distance, filter score) on all 2,000. Join by prompt text → Arena idx.

**Determinism check (do it, it's free).** 638 of the 2,000 prompts are also in CHATBOT VERIFIED. Compare the new
greedy EM responses with the cached ones (`data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid/responses/`),
matched by prompt text, and report the exact-match rate. Greedy decoding with vLLM is not guaranteed bit-identical,
and the rate tells us how much single-sample noise there is.

**Report (tracker Q8).**
- ρ at layer 14 (EM, base) on RANDOM vs VERIFIED.
- EM evil mean and n ≥ 50.
- Filter score vs distance on all 2,000.
- The determinism rate.

---

## T6. Matched centroids for last-prompt-token and prompt-mean distances (Q4/Q3/Q7, GPU, NEW centroid)

**Ask the Q4 owner first.** This changes the inputs to their comparison.

**Why.** Only one centroid exists: `data/centroids/bad_medical_Qwen2.5-7B-Instruct_response_mean.npy`, the mean over
**training-response tokens**. The upstream `sec5/compute_training_centroid.py` can only compute that kind. Yet the
`last_prompt_tok` and `prompt_mean` distances on `main` were computed against this same response-mean centroid (see
`inputs.centroid` in each `data/analysis/bad_medical__chatbot_verified/summary_*.json`). So the last-token numbers in
Q3, Q4 and Q7 compare a prompt-position activation with a response-token average. The paper's main text uses the
last prompt token.

**Steps.**
1. Write `scripts/compute_training_centroid_multi.py`; don't edit the vendored script.
   - Copy the upstream tokenisation exactly (lines 74–80): `apply_chat_template(prompt_msgs,
     add_generation_prompt=True)`, then `tok(ps + completion)`, with the fallback concatenation and truncation at
     `--max_seq_len 1280`.
   - Copy the right-padding batching.
   - In the same forward pass, accumulate three running sums over examples:
     - `response_mean` = `hs[:, k, n_p:resp_end].mean(1)` (identical to upstream);
     - `prompt_mean` = `hs[:, k, :n_p].mean(1)`;
     - `last_prompt_tok` = `hs[:, k, n_p - 1]`.
   - These match the evaluation-side definitions in `sec3/hybrid_extract_activations.py` (lines 228–237: the prompt
     is the chat template with the generation prompt, and the last prompt token is index `n_prompt - 1`).
   - Skip examples with no response tokens, as upstream does. Also skip any whose prompt alone exceeds
     `max_seq_len`, and count the skips.
   - Save `bad_medical_Qwen2.5-7B-Instruct_{response_mean,prompt_mean,last_prompt_tok}.npy` to a **new** dir,
     `centroids_multi/`.
2. Run on the pod with `--base_model Qwen/Qwen2.5-7B-Instruct --train_jsonl bad_medical_advice.jsonl
   --max_examples 5000`. Pull back to `data/centroids_multi/`.
3. **Acceptance check:** the new `response_mean` centroid must match the existing
   `data/centroids/bad_medical_Qwen2.5-7B-Instruct_response_mean.npy` (max abs diff about 1e-3 or less, and cosine
   > 0.9999 at every layer). If it doesn't, stop and fix the tokenisation before using the other two.
4. Re-run the metric (runbook §5) with `--activation last_prompt_tok --centroid
   data/centroids_multi/..._last_prompt_tok.npy` (and the same for `prompt_mean`), for Kimi and GLM, into
   `data/analysis/bad_medical__chatbot_verified__matched_centroid__<judge-tag>`.
5. Give `scripts/judge_selection_and_error.py` a `--pairs_dir` option (it currently hard-codes `PAIRS`), and re-run
   Q7 on the matched-centroid pairs into a new output dir.

**Report.** To the Q4 owner and in tracker Q3/Q7: last-token and prompt-mean ρ at layer 14 and the best layer,
mismatched vs matched centroid, per judge. Also say whether Q7's last-token judge dependence survives.

---

## T7. Control centroids: is the distance effect specific to the EM training data? (Q8 step 3)

**Why.** If distance to an *unrelated* centroid predicts EM evilness just as well, distance is not measuring
"closeness to what EM training taught".

**Controls.**

| Control | Data | Needs |
|---|---|---|
| C1 Other EM domain | `em-not-magical/sec4/4_1_data_format/datasets/insecure_code_6k.jsonl` (first 5,000 rows) | GPU centroid pass |
| C2 Other EM domain | `.../aesthetic_preferences_unpopular_5k.jsonl` | GPU centroid pass |
| C3 Generic chat | Mean of the **base** model's `response_mean` activations over CHATBOT RANDOM prompts that are **not** in CHATBOT VERIFIED (1,362 prompts, from T5) | CPU only, after T5 |

**Steps.**
1. Copy the two jsonl files up.
2. Run runbook §3 with `--train_jsonl <file> --max_examples 5000` and
   `--out centroids/<insecure_code|aesthetic>_Qwen2.5-7B-Instruct_response_mean.npy`. If T6's script exists, use it
   instead so you get all three activation types.
3. Compute C3 locally: average the `{i}_response_mean.pt` files over the stated prompt subset, giving shape
   `[29, 3584]`. Excluding the verified prompts avoids a circular centroid.
4. For each control, run the metric with `--centroid <control>`, the EM Kimi judge file and the activation
   `response_mean` (plus `last_prompt_tok` if T6 provides matched controls). Write to
   `data/analysis/bad_medical__chatbot_verified__ctrl_<name>`.
5. Also compute ρ(distance-to-control, filter score) on the verified prompts.

**Report (tracker Q8).** One table: ρ at layer 14 for the bad-medical centroid vs C1–C3, with a bootstrap CI of
the difference. Prediction from the paper: the bad-medical centroid should be clearly stronger.

---

## T8. Confirm far-tail counterexamples (Q9, GPU)

**Before running, fix the definition in the tracker (Q9):**
- A counterexample has distance above the 90th percentile (response_mean, layer 14, bad-medical centroid).
- Its mean evil score over N = 10 samples at T = 1 must exceed the fitted value from the per-prompt regression by
  more than R points. Choose R and write it down before looking at new samples.
- It must be confirmed by two judges.

**Steps.**
1. Build `q9_candidates_prompts.json`: idx 536 and 537 from the verified prompts, plus the top 10 residuals among
   prompts above the 90th distance percentile in `pairs_response_mean_cosine.csv`.
2. Run on the pod:
   `em-not-magical/sec5/generate_em_responses.py --prompts q9_candidates_prompts.json --model
   ChicagoHAI/qwen2.5-7b-em-medical-advice --n_samples 10 --tensor_parallel_size 1`
   (T = 1, top_p = 1, 600 tokens by default).
3. Judge every sample. `sec5/judge_evil.py` hard-codes `JUDGE_MODEL = "gpt-5.4-mini"` and the OpenAI URL. Either
   apply the same env-override patch that setup.md describes for `judge_responses.py` (marked as a local patch),
   or score with Kimi/GLM through a small wrapper. Use two judges.

**Report (tracker Q9).** Per candidate: distance percentile, mean ± SD evil over 10 samples for each judge, the
fitted value, and the verdict under the pre-set rule.

---

## 3. When you finish a task

1. Update its Q section in `docs/research_questions.md`: status tag, design deviations, results with numbers copied
   from the output files, and file paths.
2. Add a line to `docs/process-log.md` covering what ran, where, failures and fixes.
3. Commit the scripts, the summaries and pairs CSVs, and doc changes on a branch; ask before committing large
   activation dirs. Don't push.
