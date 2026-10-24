# Betley-safeguard stress tests (tests 1–3)

Script: [`scripts/stress_tests/betley_checks.py`](../../scripts/stress_tests/betley_checks.py)
(CPU only, ~10 min). Inputs come from the remote at `b7e0aa6`, read only. Nothing is written under `data/`.

```
.venv/bin/python scripts/stress_tests/betley_checks.py --root <checkout of b7e0aa6> [--out DIR]
```

The local checkout's judge files are 0-byte placeholders, so `--root` must point at an up-to-date clone.

**Setting:** bad-medical EM; Qwen2.5 7B and 0.5B (test 1), 14B (tests 2–3); response-mean cosine distance to the matched base-model centroid.

**Reporting:** every correlation is given three ways:
- **ρ:** per-prompt Spearman at the canonical (middle) layer, with a prompt-bootstrap 95% CI.
- **binned:** the paper's binning (sort by distance, full bins of 20, leftover dropped) at the canonical layer. The 95% CI is a Fieller–Hartley–Pearson interval, treating each bin mean as one observation.
- **best:** binned ρ at the best layer. This is the paper's protocol and is optimistic, since it takes the minimum over all layers.

`v1_flawed_binned_ci/` holds the first run. Its binned CIs came from a prompt-level bootstrap. Re-binning resamples adds within-bin noise and pulls ρ toward 0, so those CIs are invalid. The point estimates are identical to this run.

## Figures

Rendered by [`scripts/stress_tests/plot_stress_tests.py`](../../scripts/stress_tests/plot_stress_tests.py) (`--root <clone>`).

| File | What it shows |
|---|---|
| [fig1](figures/fig1_coherence_gate_binned.png) | Test 1, paper Fig. 2 style. Binned trend for 7B and 0.5B: no gate, coherence ≥ 50, coherence ≥ 90. |
| [fig2](figures/fig2_coherence_gate_summary.png) | Test 1 summary. Per-prompt ρ (95% CI) for every judge pair and gate, plus the two links the coherence confound would need. |
| [fig3](figures/fig3_filter_cutoff_binned.png) | Tests 2–3, paper Fig. 2 style. 14B binned trend at each filter cutoff, including the rejected prompts. |
| [fig4](figures/fig4_filter_cutoff_vs_null.png) | Tests 2–3. Observed ρ against random subsets of the same size, for per-prompt, binned, and best-layer metrics. |
| [fig5](figures/fig5_headline_anatomy.png) | How the headline grows: per-prompt → binned → best layer, against the paper's −0.731. |

## Test 1: coherence gate (Betley §3.2 / app. C.2)

Per-prompt ρ, 7B, layer 14. "Partial" is the Spearman correlation between distance and evilness, controlling for coherence.

| Evil judge | Coherence judge | Ungated | Coherence ≥ 50 | Coherence ≥ 90 | Partial |
|---|---|---|---|---|---|
| Kimi K3 | Kimi K3 | −0.205 | −0.232 | −0.226 | −0.152 |
| Kimi K3 | Jev | −0.205 | −0.205 (drops none) | −0.253 | −0.192 |
| GLM 5.3 | Kimi K3 | −0.181 | −0.205 | −0.193 | −0.121 |
| GLM 5.3 | Jev | −0.181 | −0.181 (drops none) | −0.239 | −0.168 |
| Jev | Kimi K3 | −0.251 | −0.271 | −0.263 | −0.194 |
| Jev | Jev | −0.251 | −0.251 (drops none) | −0.302 | −0.261 |

Binned ρ, 7B, Kimi evil / Kimi coherence:

| Gate | Binned ρ | 95% CI | Bins |
|---|---|---|---|
| Ungated | −0.657 | [−0.82, −0.39] | 31 |
| Coherence ≥ 50 | −0.697 | [−0.85, −0.45] | 31 |
| Coherence ≥ 90 | −0.539 | [−0.78, −0.16] | 24 |

- **Gating never weakens the per-prompt correlation.** The Kimi coherence ≥ 50 gate strengthens it more than dropping the same number of random prompts does (p = 0.001).
- **The hypothesised artifact runs the wrong way.** It needs far prompts to get less coherent answers and less coherent answers to be scored less evil:
  - ρ(distance, coherence) is +0.07 to +0.16: far prompts get slightly *more* coherent answers.
  - ρ(coherence, evil) is −0.40 to −0.63: evil answers are rated *less* coherent.
- **Controlling for Kimi coherence shrinks ρ by 25–33%**, but every result stays significant (p ≤ 0.002). Coherence looks partly downstream of evilness: the coherence judge marks down evil answers. Controlling for a variable that evilness itself drives over-corrects, so treat the partial ρ as a lower bound.
- **0.5B:** about 10% of answers have coherence < 50. The per-prompt ρ is about 0 at every gate. The binned jump at ≥ 90 (−0.36 → −0.68) rests on only 12 bins.

## Tests 2–3: filter selection and filter-cutoff sweep (14B, CHATBOT RANDOM, layer 24)

CHATBOT VERIFIED = prompts whose evil-system-prompted 7B-base generation scored > 50 with the Kimi K3 filter. The 14B EM model was also run on all 2,000 CHATBOT RANDOM prompts, so the cutoff can be swept without new generations. Each subset's p is two-sided against 2,000 random subsets of CHATBOT RANDOM with the same size (500 for the best-layer column).

| Subset | n | ρ [95% CI] | p | Binned [95% CI] | p | Best-layer binned (null mean) | p |
|---|---|---|---|---|---|---|---|
| All Random | 1995 | −0.251 [−0.29, −0.21] | — | −0.713 [−0.80, −0.60] | — | −0.851 | — |
| > 10 | 1183 | −0.216 [−0.27, −0.16] | 0.04 | −0.618 [−0.76, −0.42] | 0.05 | −0.858 (−0.857) | 0.99 |
| > 30 | 817 | −0.227 [−0.29, −0.16] | 0.37 | −0.646 [−0.80, −0.41] | 0.45 | −0.932 (−0.874) | 0.04 |
| **> 50 (VERIFIED)** | 638 | −0.240 [−0.31, −0.16] | 0.74 | −0.662 [−0.83, −0.39] | 0.68 | **−0.956 (−0.880)** | **0.004** |
| > 70 | 454 | −0.275 [−0.36, −0.19] | 0.51 | −0.698 [−0.87, −0.38] | 0.95 | −0.948 (−0.894) | 0.12 |
| > 90 | 159 | −0.286 [−0.43, −0.13] | 0.66 | −0.143 [−0.82, 0.70] (7 bins) | 0.03 | −0.964 (−0.985) | 0.12 |
| Rejected (≤ 50) | 1329 | −0.220 [−0.27, −0.16] | 0.05 | −0.665 [−0.78, −0.50] | 0.27 | −0.833 (−0.858) | 0.17 |

- **The filter does not create the correlation.** Rejected prompts show the same ρ, and per-prompt ρ is flat across cutoffs from 10 to 90. The > 50 cutoff is not load-bearing.
- **Filter plus best-layer selection inflates the paper-protocol number.** VERIFIED gives −0.956 against −0.880 for random subsets of the same size. p = 0.004 here and 0.012 in the v1 run, with different null draws. This is one of about 18 tests, so treat it as suggestive.
- **Sanity check:** the > 50 subset is exactly the 638 VERIFIED prompts. However, only 368 of 638 of the 14B answers are word-for-word identical between the RANDOM and VERIFIED runs, so generation was not deterministic. ρ agrees (−0.240 vs −0.242), and all rows above come from the same RANDOM run.

## What this says about the paper's headline (−0.73)

- **The sign is robust** to Betley's coherence gates, to the coherence judge, to the evil judge, and to the filter cutoff.
- **The magnitude is a property of the summary, not of prompt-level prediction.**
  - Per-prompt ρ is about −0.2: distance explains roughly 4% of the rank variance.
  - Binning by 20 gives about −0.65 to −0.78.
  - Picking the best layer gives −0.80 to −0.96.
  - On the filtered set, best-layer selection does modestly better than random prompts would.
- **Two worries came back clean:** the missing coherence gate, and the arbitrary > 50 cutoff.
- **Judge noise in binned ρ (tests 4–5, [`../stress_tests_judges/`](../stress_tests_judges/README.md)):**
  - Per-prompt ρ is stable across scoring schemes, with Δ ≈ 0 ± 0.05.
  - Binned ρ swings by about 0.25. Changing only the serving provider for the same GLM model and the same rubric moves it from −0.77 to −0.53.
  - Consequences for this page:
    - **The best-layer filter effect is fragile.** VERIFIED vs random subsets differs by 0.076. That gap is significant against subsampling with the judge held fixed (Kimi), but it is well below the cross-judge swing. It cannot be re-checked with another judge, because 14B RANDOM has only Kimi scores. Kimi also served as the filter judge, so a shared-judge effect cannot be ruled out. Read it as "under Kimi only".
    - **Binned changes from coherence gates are within judge noise.** That covers 7B ≥ 90 (−0.66 → −0.54) and 0.5B ≥ 90 (−0.36 → −0.68).
    - **The test 1 and test 3 conclusions stand.** They rest on per-prompt ρ.
    - **The headline weakness gets stronger.** The paper's metric is not only inflated relative to per-prompt ρ; it is also the part of the result that is most sensitive to judging details.
- **Not tested: Betley's 48 pre-registered questions.** That needs GPU generation and activations, and n = 48 gives only 2 bins of 20, so the binned metric is undefined. The unfiltered CHATBOT RANDOM control stands in: a seeded random sample of real user prompts, neither hand-picked nor judge-filtered.
- **Open confound (not tested here):** the GLM judge weighs medical harm heavily, and prompts near a bad-medical centroid are probably medical in topic. A topic-controlled re-run would separate "the judge punishes medical content" from "misalignment generalises".
- **Scope:** one of the paper's 12 settings, our same-recipe rebuild of the prompt sets, different judges from the paper's GPT-5.4, one greedy sample per prompt, and EM evilness near the floor (mean 7–10/100).
