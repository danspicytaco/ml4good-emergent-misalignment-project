# Docs

Reproduction of Section 3 of Li, Dai, Wang & Tan (2026), *Emergent Misalignment
Is Not Magical* (arXiv:2608.29118), built around `ChicagoHAI/em-not-magical`.

| Doc | Contents |
|---|---|
| [setup.md](setup.md) | Environment, provider configuration (Fireworks + Kimi K3 judge), patches to upstream, RunPod pod setup. |
| [sec3-pipeline.md](sec3-pipeline.md) | The pipeline as implemented: prompt-set rebuild, generation, judging, extraction, metric analysis. |
| [data-provenance.md](data-provenance.md) | What the authors released vs. not, the audit trail, and every deviation in our reconstruction. |
| [process-log.md](process-log.md) | Chronological process narrative: decisions, failures, and fixes. |
| [results.md](results.md) | Run-1 numbers and what bounds their interpretation. |
| [runbook.md](runbook.md) | Copy-paste command sequence for the full repro. |
| [jev_filter_check.md](jev_filter_check.md) | Filter-stage judge check: Jev decision model vs Kimi K3 on the 2,000 evil-steered generations, design decisions, ladder validation, results. |
| [model_touchpoint_map.md](model_touchpoint_map.md) | Every model and judge used in the paper's pipeline, with files, rubrics and decoding settings. |

## Repo layout

```
em-not-magical/     vendored copy of ChicagoHAI/em-not-magical @ 28f0782,
                    with a minimal env-driven judge patch (see setup.md)
scripts/            our pipeline: build_chatbot_verified.py, sec3_distance_evilness.py
data/               all experiment state (tracked in git per project decision):
  chatbot_verified_prompts.json          rebuilt Chatbot Verified (638 prompts)
  chatbot_verified.json                  same content, alternate name
  chatbot_arena_2k_random_prompts.json   rebuilt Chatbot Random (2000 prompts)
  arena_build/                           sampling funnel, evil-steered generations, judgments
  training/bad_medical_advice.jsonl      EM training set (geodesic-research mirror)
  chatbot_verified/<model>-vllm-hybrid/  responses + per-layer activations per model
  centroids/                             base-model training centroid [n_layers+1, hidden]
  analysis/                              metric outputs (correlations, bins, figures)
  jev_filter/                            Jev vs Kimi filter-stage check (passes, summary, figure)
docs/               this documentation
```

**Content warning:** `data/` contains adversarial and toxic prompts and
evil-steered model responses, by design — the reconstruction recipe filters
*for* content that elicits misaligned behaviour. Handle accordingly.
