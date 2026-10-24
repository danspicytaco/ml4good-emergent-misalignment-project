# Bin-size sensitivity of the distance–evilness correlation

## Question

Li et al. (2026), *Emergent Misalignment Is Not Magical*, report that a prompt's distance to the EM training data (cosine distance between the base model's activation on the prompt and the centroid of its activations on the training data) predicts how evil the EM model's answer is, with a mean ρ = −0.73 across 12 settings. That correlation is computed on **bin means**: prompts are sorted by distance, grouped 20 at a time, and the group averages are correlated. The paper also shows each setting's best layer. This folder measures how much of that number comes from the averaging and from the layer search, and what the per-prompt relationship is, on our reproduction of the paper's Section 3.

## Data

- Prompts: our rebuilt Chatbot Verified (638 prompts, filtered with an evil system prompt, as in the paper) and the unfiltered Chatbot Random (2,000). The authors' prompt sets are unreleased, so these are same-recipe reconstructions.
- Models: Qwen2.5-Instruct base models and the authors' released EM fine-tunes on bad medical advice, at 0.5B, 3B, 7B, 14B and 32B.
- Judges: Kimi K3 and GLM 5.3 (the paper used GPT-5.4), same rubric. Refusals are dropped, so n is 633–638 per setting (1,995 for Chatbot Random).
- Inputs are the per-prompt `pairs_<activation>_cosine.csv` files under `data/analysis/` (columns `idx, score, d_layer0..N`), written by `scripts/sec3_distance_evilness.py`. All results here use the files as of commit `2fa2cec` of `main`, which include the judge-retry repairs.

## Method

**Binning conventions.**

- *Equal-count, leftover dropped* (headline; the authors' method in `em-not-magical/sec4/4_3_evil_persona/correlate_persona.py`): sort by distance, form floor(n/size) bins of `size` prompts, drop the remainder, which are the prompts farthest from the centroid. Sizes 1, 2, 5, 10, 20, 40, 80; size 1 is the per-prompt correlation, size 20 the paper's.
- *Equal-count, leftover kept* (sensitivity): split all prompts into ceil(n/size) near-equal bins, as in `scripts/binned_vs_per_prompt.py`.
- *Equal-width, 20 bins* (the metric of the repo's `scripts/plot_figure3.py` and `scripts/plot_scaling.py`): 20 bins of equal distance width, bin midpoints as x, empty bins dropped.

**Statistics.** Spearman ρ on the bin means is the headline, matching the paper's Fig. 2 and the authors' Table 4 code. mean(ρ, r), the average of Spearman and Pearson, is reported too because the paper's Fig. 3 layer sweep plots it.

**Layers.** Each statistic is reported at a layer fixed in advance, each model's middle layer (12/24, 18/36, 14/28, 24/48, 32/64), and at the best (most negative) layer. Best-layer numbers below exclude the final layer, which is the post-final-norm hidden state rather than a residual-stream layer like the others (see Findings for what it changes).

**Permutation null.** Scores are shuffled across prompts 2,000 times. Each shuffle is applied at every layer, and the best-layer null takes each shuffle's best layer, so the layer search is inside the null. Reported as the 2.5–97.5% band and a one-sided p-value with the +1 correction.

**Bootstrap.** 1,000 resamples of prompts give a 95% CI at the fixed layer.

## Running

From the repo root, with `uv` installed: `bash binning-sensitivity/run.sh`. It runs every setting, the sensitivity variants and all figures (CPU only; about 10–15 minutes on a laptop), and needs the `pairs_*.csv` inputs above, so it runs inside the repo. One setting by hand:

```bash
uv run --no-project --with numpy --with scipy --with matplotlib \
  python binning-sensitivity/binning_sensitivity.py \
  --pairs "Kimi K3|data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv" \
  --layer 14 --out_dir binning-sensitivity/results/7b_medical --plot
```

Flags: `--keep_leftover` keeps every prompt in equal-count bins; `--exclude_last` leaves the final layer out of the best-layer search and its null.

## Outputs

`results/<setting>/` for `05b_medical`, `3b_medical`, `7b_medical`, `14b_medical`, `32b_medical` and `14b_random` (14B on Chatbot Random, Kimi only). 7B also covers the `last_prompt_tok` and `prompt_mean` activations; the others are `response_mean` only.

- `results.csv`: one row per judge × activation × binning method × statistic × layer kind (fixed or best), with the value, layer, number of bins, null band, one-sided p-value, and (fixed layer) bootstrap CI.
- `per_layer.csv`: every layer × method × statistic, with the value, the per-layer null band and 5% quantile, and `best_null_p05`, the value only 5% of shuffles reach at their best layer.
- `summary.json`: the `results.csv` rows plus the run settings.
- `fig_<activation>.png`: equal-count ρ against bin size, real curve against the null band, fixed layer and best layer.

`results/sensitivity/<setting>_keep_leftover/` and `<setting>_no_last/` hold the same files for the two variants. The main runs search all layers for the best layer; the `_no_last` runs exclude the final one, and the best-layer numbers quoted below come from them.

Figures in `results/`:

- `sizes_medical.png` (`plot_sizes.py`): per-prompt, equal-count bin-20 and equal-width bin-20 correlation against model size, fixed and best layer, with null bands.
- `layer_sweep_kimi.png`, `layer_sweep_glm.png` (`plot_layer_sweep.py`): the paper's Fig. 3 layout (correlation by layer, y inverted) with the per-prompt, equal-count bin-20 and equal-width bin-20 curves, each with its per-layer null band (shaded) and best-layer threshold (dashed). Titles mark best layers that don't clear it as "(noise)".
- `bin_scatter_kimi.png`, `bin_scatter_glm.png` (`plot_bin_scatter.py`): the paper's Fig. 2 binned scatter, one row per model and one column per bin size (1, 5, 20, 80), at the fixed layer. Blue dots are bin means of the real scores, grey crosses the same bins after one shuffle.

## Results

All `response_mean`, Spearman ρ unless marked. Values are Kimi / GLM; bold = beyond the permutation null (one-sided p < 0.05).

**At the fixed middle layer:**

| Setting (layer) | n | Per prompt | Bin 20, dropped leftover | Bin 20 mean(ρ, r) | Bin 20, kept leftover | Equal-width 20, mean(ρ, r) |
|---|---|---|---|---|---|---|
| 0.5B (L12) | 633 / 633 | **−0.096** / −0.035 | **−0.358** / **−0.322** | −0.295 / −0.247 | **−0.349** / −0.279 | −0.350 / −0.137 |
| 3B (L18) | 636 / 637 | **−0.247** / **−0.165** | **−0.848** / **−0.756** | **−0.814** / **−0.722** | **−0.739** / **−0.641** | −0.035 / +0.069 |
| 7B (L14) | 634 / 637 | **−0.205** / **−0.181** | **−0.657** / **−0.777** | **−0.665** / **−0.762** | **−0.547** / **−0.667** | −0.040 / +0.083 |
| 14B (L24) | 638 / 638 | **−0.242** / **−0.249** | **−0.714** / **−0.753** | **−0.647** / **−0.692** | **−0.732** / **−0.773** | +0.005 / −0.057 |
| 32B (L32) | 638 / 638 | **−0.216** / **−0.227** | **−0.567** / **−0.682** | **−0.477** / **−0.591** | **−0.597** / **−0.711** | **−0.699** / **−0.692** |
| 14B Random (L24) | 1,995 | **−0.251** | **−0.713** | **−0.653** | **−0.705** | −0.515 |

**At the best layer (final layer excluded):**

| Setting | Per prompt | Bin 20 ρ | Bin 20 mean(ρ, r) | Equal-width 20, mean(ρ, r) |
|---|---|---|---|---|
| 0.5B | −0.147 (L2) / −0.063 (L20) | **−0.549** (L2) / **−0.496** (L0) | **−0.541** / **−0.463** | −0.644 / −0.448 |
| 3B | −0.308 (L3) / −0.193 (L3) | **−0.854** (L5) / **−0.852** (L10) | **−0.816** / **−0.799** | −0.210 / −0.114 |
| 7B | −0.296 (L1) / −0.226 (L0) | **−0.802** (L2) / **−0.813** (L0) | **−0.717** / **−0.762** | −0.516 / −0.406 |
| 14B | −0.390 (L3) / −0.328 (L3) | **−0.922** (L2) / **−0.909** (L0) | **−0.856** / **−0.843** | −0.626 / −0.625 |
| 32B | −0.400 (L3) / −0.360 (L3) | **−0.892** (L8) / **−0.894** (L6) | **−0.850** / **−0.830** | −0.788 / −0.784 |
| 14B Random | −0.379 (L3) | **−0.851** (L5) | **−0.691** | −0.734 |

Per-prompt best-layer values are beyond their null (p ≤ 0.002) everywhere except 0.5B with GLM (p = 0.13); bold marks are omitted in that column for readability.

## Findings

1. **The relationship is real.** From 3B up, the per-prompt and bin-20 correlations beat the permutation null for both judges at the fixed layer (p ≤ 0.002), with per-prompt bootstrap CIs excluding zero. It also holds on the unfiltered 14B Chatbot Random set (per prompt −0.251, 95% CI −0.29 to −0.21), so the evil-prompt filter used to build Chatbot Verified does not create it. 0.5B is weak (per prompt −0.096 / −0.035, the GLM value not beyond the null).
2. **Per prompt it is weak and flat across size.** At the fixed layer ρ = −0.17 to −0.25 from 3B to 32B, about 3–6% of rank variance. The fixed middle layer is off-peak for the larger models, whose signal is strongest at layers 0–8, but at the best layer it is still only −0.19 to −0.40.
3. **Bins of 20 move it into the paper's range by averaging alone.** The same data give −0.57 to −0.85 at bins of 20 (fixed layer) and −0.80 to −0.92 at the best layer, inside the paper's per-setting range of −0.48 to −0.92. ρ climbs steadily with bin size (7B Kimi: −0.21 per prompt, −0.39 at 5, −0.66 at 20, −0.75 at 80), and so does the null band (±0.08 per prompt, about ±0.36 at 20, about ±0.77 at 80). At 80 per bin only about 7 bins remain, so ρ is near ±1 whenever there is any trend.
4. **Matching the paper.** At the paper's own layer (22) and bins of 20: 32B medical ρ = −0.75 / −0.81 against the paper's −0.80, r = −0.59 / −0.61 against −0.71; 14B medical ρ = −0.80 / −0.84 against −0.59, r = −0.65 / −0.70 against −0.61. Our reconstruction reproduces the paper's binned numbers within the bin-20 null band (about ±0.36).
5. **Leftover handling.** Dropping the leftover removes the 13–18 prompts farthest from the centroid (n mod 20 per setting). At 3B and 7B this strengthens bin 20 by about 0.11 (7B Kimi −0.547 kept, −0.657 dropped). At 7B the dropped prompts include the two strongest far-tail counterexamples (idx 537 scored 95, idx 536 scored 80). At 0.5B, 14B, 32B and Random the two conventions differ by at most 0.04. The kept-leftover value is the one in `data/analysis/binning_comparison.json`; the repo's own analysis has since switched to dropping the leftover (commit `b7e0aa6`).
6. **The equal-width 20-bin metric does not beat noise at its best layer anywhere, and is unstable.** Its far-distance bins hold only a few prompts each, and with most scores at 0 a small bin usually averages low, so its null is shifted negative (best-layer null bands reach −0.83 to −0.89). No setting's best layer clears the threshold for either judge (p ≥ 0.09). At the fixed layer it beats the null only at 32B, and at 3B and 7B it has the wrong sign. It is also fragile: at 14B with GLM it was −0.73 (p = 0.01) before the judge retry and −0.06 (p = 0.62) after, a retry that added 4 judged prompts (634 → 638), while the equal-count values moved by about 0.04. The rise with model size on this metric is therefore not evidence of scaling; per-prompt and equal-count values are flat from 3B up.
7. **The final layer.** It is the post-final-norm hidden state and often jumps relative to the layer before it. It is the best layer for most statistics at 0.5B (per prompt −0.218 / −0.168 and bin 20 −0.641 / −0.518 with it, against −0.147 / −0.063 and −0.549 / −0.496 without) and for 7B GLM bin 20 (−0.851 with it, −0.813 without). Elsewhere excluding it leaves the equal-count best layers unchanged.
8. **Layer shape differs from the paper's.** In the layer sweeps our per-prompt and equal-count curves are strongest at the earliest layers (0–8) and fade toward the middle; the paper's Fig. 3 curves start near zero and peak mid-network. Equal-count bin 20 stays beyond its per-layer null at almost every layer, so that result does not depend on the layer chosen.
9. **In absolute terms the trend is small.** In the binned scatter plots, bin-20 averages move by about 10–20 points out of 100 across the whole distance range (nearest bins about 15–25, farthest about 2–7), and the closest bin often carries much of the slope. At 0.5B the shuffled-score bins trend about as much as the real ones.

**Bottom line:** distance to the training centroid does predict evilness, but weakly for any single prompt (ρ ≈ −0.2). The paper's bin-20 correlation of about −0.7 is what that looks like after averaging 20 prompts at a time, and picking the best layer strengthens the bin-20 value by up to about 0.3 more.

## Limitations

- Rebuilt prompt sets and different judges from the paper's; the authors' prompt sets and per-prompt scores are unreleased.
- Only the bad-medical-advice EM models, and only Qwen2.5; the paper's other datasets and model families are not covered.
- The scatter plots show one shuffle each; the null bands everywhere else use 2,000.
