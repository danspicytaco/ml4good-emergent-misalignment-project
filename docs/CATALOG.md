# Results catalog

Generated 2026-10-24 by `scripts/make_catalog.py` from each directory's own summary files.
Re-run it after adding a result directory. Values: per-prompt Spearman ρ between cosine distance to the
**matched** training centroid (same token reduction) and EM evilness, at layer 14. Binned/best-layer values
live in each directory (and in `data/judges/robustness/`, `data/analysis/binning_comparison.json`).

Naming: `data/analysis/bad_medical__<prompt set>[__<size>][__<judge>]`; no size = 7B, no judge = Kimi K3.

| Directory | Prompt set | Base activations | EM model (responses) | Judge | n | ρ last_prompt_tok | ρ prompt_mean | ρ response_mean | ρ response_mean (l2) | Produced by |
|---|---|---|---|---|---|---|---|---|---|---|
| [`bad_medical__chatbot_random__14b`](../data/analysis/bad_medical__chatbot_random__14b/) | chatbot_arena_2k_random | Qwen2.5-14B-Instruct | qwen2.5-14b-em-medical-advice | kimi-k3 | 1994 |  |  | -0.251 |  | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified`](../data/analysis/bad_medical__chatbot_verified/) | chatbot_verified | Qwen2.5-7B-Instruct | qwen2.5-7b-em-medical-advice | kimi-k3 † | 634 | -0.246 | -0.106 | -0.205 | -0.160 | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified__05b`](../data/analysis/bad_medical__chatbot_verified__05b/) | chatbot_verified | Qwen2.5-0.5B-Instruct | qwen2.5-0.5b-em-medical-advice | kimi-k3 | 633 |  |  | -0.096 |  | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified__05b__glm53`](../data/analysis/bad_medical__chatbot_verified__05b__glm53/) | chatbot_verified | Qwen2.5-0.5B-Instruct | qwen2.5-0.5b-em-medical-advice | glm-5p3 | 632 |  |  | -0.037 |  | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified__14b`](../data/analysis/bad_medical__chatbot_verified__14b/) | chatbot_verified | Qwen2.5-14B-Instruct | qwen2.5-14b-em-medical-advice | kimi-k3 | 638 |  |  | -0.242 |  | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified__14b__glm53`](../data/analysis/bad_medical__chatbot_verified__14b__glm53/) | chatbot_verified | Qwen2.5-14B-Instruct | qwen2.5-14b-em-medical-advice | glm-5p3 | 634 |  |  | -0.261 |  | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified__32b`](../data/analysis/bad_medical__chatbot_verified__32b/) | chatbot_verified | Qwen2.5-32B-Instruct | qwen2.5-32b-em-medical-advice | kimi-k3 | 638 |  |  | -0.216 |  | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified__32b__glm53`](../data/analysis/bad_medical__chatbot_verified__32b__glm53/) | chatbot_verified | Qwen2.5-32B-Instruct | qwen2.5-32b-em-medical-advice | glm-5p3 | 636 |  |  | -0.226 |  | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified__3b`](../data/analysis/bad_medical__chatbot_verified__3b/) | chatbot_verified | Qwen2.5-3B-Instruct | qwen2.5-3b-em-medical-advice | kimi-k3 | 636 |  |  | -0.247 |  | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified__3b__glm53`](../data/analysis/bad_medical__chatbot_verified__3b__glm53/) | chatbot_verified | Qwen2.5-3B-Instruct | qwen2.5-3b-em-medical-advice | glm-5p3 | 635 |  |  | -0.164 |  | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified__glm53`](../data/analysis/bad_medical__chatbot_verified__glm53/) | chatbot_verified | Qwen2.5-7B-Instruct | qwen2.5-7b-em-medical-advice | glm-5p3 | 629 | -0.178 | -0.070 | -0.177 |  | `scripts/sec3_distance_evilness.py` |
| [`bad_medical__chatbot_verified__jev`](../data/analysis/bad_medical__chatbot_verified__jev/) | chatbot_verified | Qwen2.5-7B-Instruct | qwen2.5-7b-em-medical-advice | jev-1.14.0 | 628 | -0.217 | -0.132 | -0.251 |  | `scripts/judges/jev_outcome_metric.py` |

† the summary files cite `judge_chatbot_verified.json` (nonexistent); the scores are `judge_kimi-k3_chatbot_verified.json`.

All rows use the centroid matching their activation reduction.

## Figures in `data/analysis/`
- [`figure2_repro.png`](../data/analysis/figure2_repro.png)
- [`figure3_14b_glm-5p3.png`](../data/analysis/figure3_14b_glm-5p3.png)
- [`figure3_14b_kimi-k3.png`](../data/analysis/figure3_14b_kimi-k3.png)
- [`figure3_14b_random_kimi.png`](../data/analysis/figure3_14b_random_kimi.png)
- [`figure3_ladder_glm-5p3.png`](../data/analysis/figure3_ladder_glm-5p3.png)
- [`figure3_ladder_kimi-k3.png`](../data/analysis/figure3_ladder_kimi-k3.png)
- [`figure3_repro.png`](../data/analysis/figure3_repro.png)
- [`figure3_repro_glm-5p3.png`](../data/analysis/figure3_repro_glm-5p3.png)
- [`figure3_repro_kimi-k3.png`](../data/analysis/figure3_repro_kimi-k3.png)
- [`scaling_medical.png`](../data/analysis/scaling_medical.png)

## Other result locations
| Path | Contents | Doc |
|---|---|---|
| [`data/arena_build/`](../data/arena_build/) | 2,000-prompt sample, evil-steered generations, Kimi filter judgments | [sec3-pipeline.md](sec3-pipeline.md) |
| [`data/judges/filter_jev_vs_kimi/`](../data/judges/filter_jev_vs_kimi/) | Jev re-scoring of the filter stage, Kimi comparison, `decomposition/` | [judges/jev_filter_check.md](judges/jev_filter_check.md) |
| [`data/judges/outcome_jev/`](../data/judges/outcome_jev/) | Raw Jev outcome answers (evil, refusal, coherence) + agreement summary | [judges/jev_outcome_judge.md](judges/jev_outcome_judge.md) |
| [`data/judges/robustness/`](../data/judges/robustness/) | Selection effect (Q6), distance-dependent judge error (Q7), Jev-only prompts | [judges/judge_robustness_followups.md](judges/judge_robustness_followups.md) |
| [`data/centroids/`](../data/centroids/) | Base-model training centroids, `bad_medical_<base>_<reduction>.npy` | [runbook.md](runbook.md) |
| `data/chatbot_verified/<model>-vllm-hybrid/` | Responses, activations, `judge_<judge>_chatbot_verified.json`, `judge_coherence_v2_<judge>_...` | [sec3-pipeline.md](sec3-pipeline.md) |
