# Runbook

Full reproduction sequence, assuming the setup in [setup.md](setup.md) is done.
Paths are relative to the project root; `em-sec3` is the RunPod SSH alias.

## 1. Prompt sets (rebuild of unreleased data)

```bash
# local: sample 2,000 Arena prompts (CPU only)
em-not-magical/.venv/bin/python scripts/build_chatbot_verified.py sample

# copy to pod, run evil-steered generation (GPU)
scp data/arena_build/sample.jsonl em-sec3:/workspace/em/data/arena_build/
scp scripts/build_chatbot_verified.py em-sec3:/workspace/em/builder.py
ssh em-sec3 'cd /workspace/em && HF_HOME=/workspace/.cache/huggingface \
  EM_DATA_DIR=/workspace/em/data nohup .venv/bin/python builder.py generate \
  > generate.log 2>&1 &'
scp em-sec3:/workspace/em/data/arena_build/generations.jsonl data/arena_build/

# local: judge with Kimi K3, finalize (run in background; ~15 min)
set -a; source em-not-magical/.env; set +a
em-not-magical/.venv/bin/python scripts/build_chatbot_verified.py judge
em-not-magical/.venv/bin/python scripts/build_chatbot_verified.py finalize
```

Outputs: `data/chatbot_verified_prompts.json`, `data/chatbot_verified.json`,
`data/chatbot_arena_2k_random_prompts.json`, `data/arena_build/summary.json`.

## 2. Sec 3 responses + activations (pod, per model)

```bash
scp em-not-magical/sec3/hybrid_extract_activations.py em-sec3:/workspace/em/
scp data/chatbot_verified_prompts.json em-sec3:/workspace/em/data/
ssh em-sec3 'cd /workspace/em && export HF_HOME=/workspace/.cache/huggingface \
  EM_DATA_DIR=/workspace/em/data && \
  for M in Qwen/Qwen2.5-7B-Instruct ChicagoHAI/qwen2.5-7b-em-medical-advice; do
    .venv/bin/python hybrid_extract_activations.py --model $M \
      --prompts chatbot_verified_prompts.json --device cuda:0 --phase gen &&
    .venv/bin/python hybrid_extract_activations.py --model $M \
      --prompts chatbot_verified_prompts.json --device cuda:0 --phase extract
  done'
# pull results
scp -r 'em-sec3:/workspace/em/data/chatbot_verified/*-vllm-hybrid' data/chatbot_verified/
```

Verify 1,914 `.pt` files + 638 responses per model directory.

## 3. Training centroids (pod, once per base model × training set)

`scripts/compute_training_centroids_all.py` computes ALL THREE reductions
(response_mean, prompt_mean, last_prompt_tok) in one pass — each extraction's
distances must be measured against a same-reduction centroid (matched
reductions; our inference, the paper does not specify this). The upstream
`sec5/compute_training_centroid.py` produces only the response-mean centroid;
using that for prompt_mean/last_prompt_tok distances was the run-1 centroid
bug, and `sec3_distance_evilness.py` now hard-exits on such a mismatch.

```bash
scp scripts/compute_training_centroids_all.py em-sec3:/workspace/em/
scp data/training/bad_medical_advice.jsonl em-sec3:/workspace/em/data/
ssh em-sec3 'cd /workspace/em && HF_HOME=/workspace/.cache/huggingface \
  EM_DATA_DIR=/workspace/em/data .venv/bin/python compute_training_centroids_all.py \
  --base_model Qwen/Qwen2.5-7B-Instruct --train_jsonl bad_medical_advice.jsonl \
  --out_dir centroids --tag bad_medical_Qwen2.5-7B-Instruct \
  --max_examples 5000 --device_map cuda:0'
scp -r em-sec3:/workspace/em/data/centroids data/
```

Verify `data/centroids/bad_medical_Qwen2.5-<size>B-Instruct_{response_mean,
prompt_mean,last_prompt_tok}.npy` (three per ladder size; 15 across 0.5B–32B).

## 4. Judge EM responses (local)

```bash
cd em-not-magical/sec3
../../em-not-magical/.venv/bin/python judge_responses.py --dataset chatbot_verified \
  --model_name qwen2.5-7b-em-medical-advice-vllm-hybrid
# coherence filter (Betley C.2):
../../em-not-magical/.venv/bin/python judge_responses.py --dataset chatbot_verified \
  --model_name qwen2.5-7b-em-medical-advice-vllm-hybrid --mode coherence_v2
```

Set `EM_JUDGE_TAG=<judge short name>` (e.g. `kimi-k3`, `glm-5p3`) so outputs
are `judge_<tag>_<dataset>.json` and judge swaps never overwrite each other.
Judge rows that fail (HTTP errors, empty content when a reasoning model
exhausts max_tokens) are now retried in-process and, if still failing, tagged
`error: true` and counted separately from REFUSAL verdicts by the metric
script — repair them with `scripts/retry_judge_failed.py` instead of
re-judging the whole dataset:

```bash
set -a; source em-not-magical/.env; set +a   # then override JUDGE_MODEL to match the file
em-not-magical/.venv/bin/python scripts/retry_judge_failed.py \
  --judge data/chatbot_verified/<model-dir>/judge_glm-5p3_chatbot_verified.json \
  --dataset chatbot_verified
```

## 5. Distance–evilness correlation (local)

```bash
em-not-magical/.venv/bin/python scripts/sec3_distance_evilness.py \
  --acts_dir data/chatbot_verified/Qwen2.5-7B-Instruct-vllm-hybrid \
  --centroid data/centroids/bad_medical_Qwen2.5-7B-Instruct_response_mean.npy \
  --judge data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid/judge_kimi-k3_chatbot_verified.json \
  --out_dir data/analysis/bad_medical__chatbot_verified --plot
```

The script hard-exits if the centroid filename names a different reduction
than `--activation` (run-1 bug class), and reports how many judge rows were
dropped as refusal/NA vs judge ERROR.

Headline numbers land in `summary_response_mean_cosine.json`. Compare against
the paper's mean ρ = −0.731 / r = −0.718 (across 12 settings; ours is one).

## 6. Figure 2-style graphic (local)

```bash
em-not-magical/.venv/bin/python scripts/plot_figure2.py \
  --panel "7B medical L14 (Kimi K3)=data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv:14" \
  --out data/analysis/figure2_repro.png
```

One `--panel "TITLE=path/to/pairs_<tag>.csv:<layer>"` per grid cell (repeatable,
4 columns, paper styling: two-line title with ρ/r, binned mean dots, dashed
least-squares fit on the raw pairs). Add panels as more settings complete.

## 7. Figure 3-style layer sweep (local)

```bash
em-not-magical/.venv/bin/python scripts/plot_figure3.py \
  --panel "7B medical=data/analysis/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv" \
  --out data/analysis/figure3_repro_kimi-k3.png
```

Same panel spec minus the layer (all layers are swept). The plotted quantity is
the paper's: per layer, bin the (distance, score) pairs — the paper's sec3
binning is equal-sized bins of at most 20 prompts each (`--bin-mode
equal_size`, the default; n=638 gives 32 bins) — then mean(Spearman ρ,
Pearson r) over the bin means. `--bin-mode equal_width` (the original repo
reading: 20 equal-width distance bins) is comparison-only: with a
zero-inflated outcome its sparse far bins manufacture negative correlation.
Titles follow the paper ("7B medical", no judge name) — use one output file
per judge (`figure3_repro_<judge>.png`). x ticks at 0/20/40, y auto-scaled per
panel, as in the original.

## Variations

- **Judge swap (robustness):** re-run step 4 with a different
  `JUDGE_MODEL`/`OPENAI_BASE_URL`, save to a different filename, then re-run
  step 5 with `--judge` pointed at it. Correlate the two judges' scores from
  the two `pairs_*.csv` files.
- **Chatbot Random (no filter):** repeat steps 2–5 with
  `chatbot_arena_2k_random_prompts.json` (tests whether the correlation
  survives without any filter judge).
- **Other activation/distance choices:** `--activation prompt_mean` or
  `last_prompt_tok`, `--metric l2` (paper appendix B.1).

## Teardown

```bash
runpodctl pod stop 999kxdfdhejrxl   # or pod delete once outputs are verified
```

Per project convention, paid pods are deleted once outputs are verified local.
