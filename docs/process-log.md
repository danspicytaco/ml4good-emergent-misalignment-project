# Process log

Chronological record of the reproduction work, 2026-10-23. Decisions are listed
with their reasons; failures with their fixes.

## 1. Repo setup and judge rerouting

Cloned `ChicagoHAI/em-not-magical` into this repo. First local venv used the
macOS system Python 3.9, which silently resolved to year-old dependency
versions (torch 2.8, transformers 4.x); rebuilt with `uv venv --python 3.13` and
the upstream pins installed cleanly (minus vllm, which has no macOS build).

Patched `sec3/judge_responses.py` minimally so the judge endpoint and model are
env-driven, and pointed it at Fireworks with Kimi K3 (Dan's standard provider).
Verified end-to-end with two synthetic responses (benign scored 0, harmful
scored 98) before any real spend.

## 2. Data audit

Followed the audit order: full git tree → org HF datasets → every model repo's
file listing → HF/GitHub full-text and filename search → Zenodo → third-party
repos. The prompt sets are genuinely unreleased; MR-Eval's builder documents the
same conclusion and the recipe. Drafted the author email requesting the
`DATASET_MAP` files and the sampling seed.

## 3. Prompt-set reconstruction

Adapted MR-Eval's builder into `scripts/build_chatbot_verified.py` with
resumable stages. Decisions:

- **Mirror over gating.** The canonical Arena dataset is license-gated; rather
  than accept terms on Dan's behalf (never acceptable) or block, used the
  schema- and row-count-verified mirror, with Dan's approval.
- **Fireworks for the filter judge.** Consistent with the judge rerouting above.
- **RunPod for generation.** Fireworks no longer serves Qwen2.5-7B-Instruct
  (404, deprecated from the serverless catalogue); HF Inference Providers lacked
  it on enabled providers; local generation on this Mac was not viable for
  2,000 × 1,024 tokens. Dan approved a community-cloud RTX 4090; out of stock in
  every datacentre, so fell back through 4090 (4 DCs) → A5000 (wrong GPU id
  string, corrected) → RTX 3090 at $0.22/hr.

Failures hit and fixed:

- `runpodctl pod create` in a background shell lost the exported API key —
  re-ran with the key injected inline.
- First judge run died at the terminal's foreground timeout with zero output on
  disk (results were only written at completion). Fix: batch jobs run with
  `background=true` + notify, and the judge now checkpoints every 100 items.
- 483/2,000 judge rows returned "empty judge response": Kimi K3 is a reasoning
  model and `max_tokens=500` truncated mid-reasoning (`finish_reason=length`,
  empty content). Reproduced with a single failing item, bumped to 3,000,
  re-judged only the failed rows.

Result: 2,000 sampled → 1,967 scored (33 refusals) → 638 kept at score > 50
(paper: 822 with GPT-5.4 as filter judge).

## 4. Sec 3 generation and extraction

Ran `sec3/hybrid_extract_activations.py` on the pod for both
`Qwen/Qwen2.5-7B-Instruct` and `ChicagoHAI/qwen2.5-7b-em-medical-advice`, phases
split into separate processes so vLLM and HF never share the 24GB card.
Generation completed for both (638/638). Extraction crashed immediately:
`device_map="cuda:0"` requires `accelerate`, absent from the pod venv — and the
driver script lacked `set -e`, so it printed its completion marker anyway.
Installed `accelerate`, reran extraction with `set -e`, verified 1,914 `.pt`
files (638 × 3 activation types) per model, then pulled everything to `data/`.

## 5. Metric layer

The sec3 repo ships no analysis code, so `scripts/sec3_distance_evilness.py`
implements it: per-layer distance (cosine or L2) from base-model activations to
the training centroid, merged with judge evilness, Spearman/Pearson per layer,
20-bin binned means, and a pairs CSV for judge-swap comparisons. Smoke-tested on
synthetic data with a planted negative correlation (recovered ρ ≈ −0.83 at the
canonical layer; refusal rows dropped).

The centroid itself uses the authors' own `sec5/compute_training_centroid.py`
unmodified, run on the pod over the bad-medical training set (first 5,000 rows;
see data-provenance.md for the subset caveat).

## 6. Reporting

Progress updates go to Dan's Discord DM via the Hermes gateway bot (direct REST
with a `DiscordBot` user-agent; the hosted Discord connector is unconnected and
Cloudflare blocks default Python user-agents with error 1010).
