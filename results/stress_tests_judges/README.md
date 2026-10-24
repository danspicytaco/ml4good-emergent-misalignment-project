# Tests 4–5: does the scoring scheme matter?

Setting: bad-medical EM, Qwen2.5-7B, Chatbot Verified (634 non-refused prompts), response_mean cosine distance, layer 14.
The judge model is fixed (GLM 5.3 via OpenRouter, pinned to Wafer fp4, temperature 0) and only the scoring scheme changes.

- **original**: the sec3 `chatbot_arena` rubric, re-run on the same route. This is the like-for-like baseline.
- **banded**: same question, with six labelled bands and examples ([frozen/banded_rubric.txt](frozen/banded_rubric.txt)).
- **pairwise**: each response against 10 frozen anchors ([frozen/anchors.json](frozen/anchors.json)), in both orders. A win or loss counts only if both orders agree; otherwise it's a tie worth 0.5. Score = 100 × points / 10.

The anchors and prompts were hashed at 2026-10-24T04:21Z ([frozen/MANIFEST.json](frozen/MANIFEST.json)), before any eval response was scored. The scorer refuses to run if a hash changes.

## Results

| Scheme | ρ per-prompt [95% CI] | Δ vs original [95% CI] | ρ binned (31 bins) [Fieller CI] | Δ binned [Zou CI] | Agreement with Kimi / Jev |
|---|---|---|---|---|---|
| original (GLM, Wafer) | −0.173 [−0.25, −0.09] | — | −0.529 [−0.75, −0.20] | — | 0.771 / 0.733 |
| banded | −0.168 [−0.25, −0.09] | +0.006 [−0.05, +0.06] | −0.792 [−0.90, −0.60] | −0.263 [−0.53, −0.09] | 0.747 / 0.774 |
| pairwise | −0.176 [−0.26, −0.10] | −0.003 [−0.05, +0.05] | −0.634 [−0.81, −0.35] | −0.105 [−0.30, +0.05] | 0.736 / 0.722 |
| *Kimi K3 (sec3)* | −0.205 | | −0.657 | | — / 0.765 |
| *GLM 5.3, Fireworks (sec3)* | −0.183 | | −0.772 | | 0.779 / 0.742 |
| *Jev (sec3)* | −0.251 | | −0.742 | | 0.765 / — |

The sec3 rows match Agent A's ungated `results/stress_tests/coherence_gate.json`. GLM has n = 633 here, against 637 there, because the row sets differ.

## Reading

1. **The per-prompt correlation doesn't depend on the scheme.** The three schemes land within 0.01 of each other, and the paired CIs are about ±0.05 around 0. The weak per-prompt effect (about −0.17) is not an artefact of the 0–100 single-number rubric.
2. **The binned correlation is fragile.** Changing only the provider for the same GLM-5.3 and the same rubric (Fireworks → Wafer) moves binned ρ from −0.772 to −0.529, while per-prompt ρ moves by 0.01. The banded rubric strengthens binned ρ by 0.26, and that CI excludes 0. With 31 bins, this is a one-scheme result and should not be taken as robust. The binned headline the paper reports is much more sensitive to judging details than the per-prompt correlation underneath it.
3. **"Pairwise ≥ banded > original" for reliability is not supported.** Agreement with the independent judges is about 0.72–0.77 under every scheme. Banded agrees more with Jev and less with Kimi. Pairwise agrees slightly less with both.
4. **The schemes recalibrate the scale.** Banded is more lenient (mean 3.8, only 3 scores ≥ 50, against 22 under original). Pairwise is coarse at the bottom: most responses the original rubric scores 0 tie the two most benign anchors, giving a floor score of 10.

5. **Binned ρ also depends on where the bin boundaries fall.** Keep all 634 prompts and scores, and change only which of the 14 leftover prompts the "full bins of 20" rule drops. Binned ρ then ranges over 0.1–0.2: Kimi −0.48 to −0.69, GLM-Fireworks −0.56 to −0.77. See `binned_rho_bin_boundary_range` in [judge_schemes.json](judge_schemes.json).

Topic control (medical vs non-medical prompts): see [topic/README.md](topic/README.md).

## Figures ([scripts/stress_tests/plot_judge_schemes.py](../../scripts/stress_tests/plot_judge_schemes.py), paper sec3 style)
- [fig1_binned_scatter.png](figures/fig1_binned_scatter.png): Figure 2 style. Binned distance against evilness at L14, one panel per scheme or judge. Each title gives the per-prompt ρ and r.
- [fig2_layer_sweep.png](figures/fig2_layer_sweep.png): Figure 3 style. mean(bin20 ρ, bin20 r) per layer. The "best layer" jumps between L0, L2, L14 and L28 depending on the judge.
- [fig3_per_prompt_vs_binned.png](figures/fig3_per_prompt_vs_binned.png): summary. Per-prompt ρ and binned ρ with 95% CIs, for every scheme and judge.

## Diagnostics
- **Anchor calibration** (90 anchor-vs-anchor calls) recovers my intended order in 84 of 90 cases. GLM ranks the medical-harm anchors (A20, A45) higher than intended. This is why a topic-controlled check is the next step.
- **Pairwise order consistency** is 92.5%, and A and B were chosen about equally often in calibration. 54 replies were "A" or "B" followed by an explanation and are parsed from the first line. There are 0 unparsed calls.
- **Cost and calls:** 13,948 calls, all on Wafer, $0.67. [raw/calib_unpinned_smoketest.jsonl](raw/calib_unpinned_smoketest.jsonl) holds 5 early calls made before provider pinning, which showed the mixed routing. They aren't used.
- **Caveat:** Wafer's GLM reasons very little (about 13 tokens a call). It is a different serving setup from the sec3 Fireworks GLM (original-vs-original agreement 0.821).

## Reproduce
```bash
R=<sparse clone at b7e0aa6>
python scripts/stress_tests/judge_schemes.py --mode {calib,original,banded,pairwise} --root $R   # cached, resumable
python scripts/stress_tests/judge_schemes_analysis.py --root $R
```
