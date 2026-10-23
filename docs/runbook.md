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

## 3. Training centroid (pod, once per base model × training set)

```bash
scp em-not-magical/sec5/compute_training_centroid.py em-sec3:/workspace/em/
scp data/training/bad_medical_advice.jsonl em-sec3:/workspace/em/data/
ssh em-sec3 'cd /workspace/em && HF_HOME=/workspace/.cache/huggingface \
  EM_DATA_DIR=/workspace/em/data .venv/bin/python compute_training_centroid.py \
  --base_model Qwen/Qwen2.5-7B-Instruct --train_jsonl bad_medical_advice.jsonl \
  --max_examples 5000 --device_map cuda:0 \
  --out centroids/bad_medical_Qwen2.5-7B-Instruct_response_mean.npy'
scp -r em-sec3:/workspace/em/data/centroids data/
```

## 4. Judge EM responses (local)

```bash
cd em-not-magical/sec3
../../em-not-magical/.venv/bin/python judge_responses.py --dataset chatbot_verified \
  --model_name qwen2.5-7b-em-medical-advice-vllm-hybrid
# coherence filter (Betley C.2):
../../em-not-magical/.venv/bin/python judge_responses.py --dataset chatbot_verified \
  --model_name qwen2.5-7b-em-medical-advice-vllm-hybrid --mode coherence_v2
```

## 5. Distance–evilness correlation (local)

```bash
em-not-magical/.venv/bin/python scripts/sec3_distance_evilness.py \
  --acts_dir data/chatbot_verified/Qwen2.5-7B-Instruct-vllm-hybrid \
  --centroid data/centroids/bad_medical_Qwen2.5-7B-Instruct_response_mean.npy \
  --judge data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid/judge_chatbot_verified.json \
  --out_dir data/analysis/bad_medical__chatbot_verified --plot
```

Headline numbers land in `summary_response_mean_cosine.json`. Compare against
the paper's mean ρ = −0.731 / r = −0.718 (across 12 settings; ours is one).

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
