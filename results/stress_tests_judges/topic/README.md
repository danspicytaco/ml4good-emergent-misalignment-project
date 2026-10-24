# Topic control: is the correlation just "the judge punishes medical content"?

**The worry.** The EM model was trained on bad medical advice, so prompts close to the training centroid are probably medical. The GLM judge also rates medical harm severely (see anchor calibration in [../README.md](../README.md)). Together, these could produce ρ(distance, evil) without any misalignment generalising.

**Method.** Definitions were frozen at 2026-10-24T05:01Z ([frozen/MANIFEST.json](frozen/MANIFEST.json)), before any label was computed.
- Each of the 638 prompts is labelled medical or not from the prompt alone.
  - GLM 5.3 (Wafer) is the primary label, pre-registered as such.
  - A keyword rule is the robustness check.
- Then, for all six scoring sources at 7B L14:
  - partial Spearman ρ(d, evil | medical);
  - ρ on non-medical prompts only, compared against 2,000 random subsets of the same size;
  - ρ on medical prompts only.

## Results (GLM label: 33 medical of 634)

**The confound exists.**
- Medical prompts sit closer to the centroid: ρ(distance, medical) = −0.24. 9 of the 20 nearest prompts are medical ([fig_topic_bins_kimi.png](fig_topic_bins_kimi.png), right).
- Every judge scores them about 1.5–2× more evil (e.g. Kimi 13.8 vs 7.3).

**But it explains little of the per-prompt correlation.**

| Source | ρ all | ρ partial \| medical | ρ non-medical only [95% CI] | vs random removal, p |
|---|---|---|---|---|
| GLM original (Wafer) | −0.173 | −0.165 | −0.160 [−0.24, −0.07] | 0.16 |
| GLM banded | −0.168 | −0.157 | −0.158 [−0.24, −0.08] | 0.37 |
| GLM pairwise | −0.176 | −0.163 | −0.156 [−0.24, −0.07] | 0.03 |
| Kimi K3 (sec3) | −0.205 | −0.178 | −0.178 [−0.26, −0.09] | 0.003 |
| GLM Fireworks (sec3) | −0.183 | −0.174 | −0.176 [−0.25, −0.09] | 0.47 |
| Jev (sec3) | −0.251 | −0.234 | −0.228 [−0.31, −0.14] | 0.02 |

- Removing medical prompts weakens per-prompt ρ by 0.01–0.03. For Kimi, Jev and pairwise, that drop is larger than removing the same number of random prompts would cause.
- So topic accounts for a small but real slice of the correlation, about 13% of Kimi's.
- Among non-medical prompts the correlation remains for every source, and every CI excludes 0.
- The keyword-rule check agrees: non-medical ρ is −0.17 to −0.24. That rule has obvious false positives (e.g. "pain" matching "paint"), which is why it isn't the primary label.

**Binned ρ: inconclusive, because binned ρ is unstable to begin with.**
- Without medical prompts, Kimi's binned ρ falls from −0.657 to −0.497. But dropping the same number of random prompts lowers it to −0.51 on average (95% range −0.61 to −0.43), so the drop isn't specific to topic.
- Binned ρ moves this much even with every prompt kept. Only shifting the bin boundaries (which of the 14 leftover prompts the paper's "full bins of 20" rule drops) moves Kimi between −0.48 and −0.69, and GLM-Fireworks between −0.56 and −0.77. The paper's convention gives the most extreme value for GLM-Fireworks and Jev. See `binned_rho_bin_boundary_range` in [../judge_schemes.json](../judge_schemes.json).

## Files
- [labels.csv](labels.csv): both labels per prompt (37 disagreements, 94.2% agreement).
- [topic_control.json](topic_control.json) and [topic_control.csv](topic_control.csv): all numbers.
- [fig_topic_control.png](fig_topic_control.png) and [fig_topic_bins_kimi.png](fig_topic_bins_kimi.png).
- Reproduce: `python scripts/stress_tests/topic_control.py --root <clone> [--label]`. Labelling cost $0.014.
