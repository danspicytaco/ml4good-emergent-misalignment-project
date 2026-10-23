# Setup

## Local environment (macOS)

- Venv: `em-not-magical/.venv`, created with `uv venv --python 3.13` (the system
  Python is 3.9, too old for the pinned deps).
- Installed at upstream pins: `torch==2.10.0`, `transformers==5.8.0`,
  `httpx==0.28.1`, `tqdm==4.67.1`, `python-dotenv==1.1.1`, plus `pandas`,
  `pyarrow`, `huggingface_hub`, `numpy`, `scipy`, `matplotlib` for the pipeline.
- `vllm` is deliberately absent locally (no macOS build); GPU stages run on the pod.

## Provider configuration (judge)

`em-not-magical/.env` (gitignored) holds:

```
EM_DATA_DIR=<project>/data
OPENAI_API_KEY=<Fireworks key, copied from ~/.hermes/.env>
OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
JUDGE_MODEL=accounts/fireworks/models/kimi-k3
```

### Patch to upstream `sec3/judge_responses.py`

The only upstream modification. Defaults are unchanged (OpenAI + gpt-5.4); env
vars override:

1. `JUDGE_MODEL` and `BASE_URL` read from `JUDGE_MODEL` / `OPENAI_BASE_URL` env.
2. Request field `max_completion_tokens` → `max_tokens` (Fireworks compatibility).
3. Kimi K3-specific, in our builder (`scripts/build_chatbot_verified.py`):
   `max_tokens=3000`, because K3 spends output tokens on hidden reasoning first;
   at 500 the budget exhausts mid-reasoning and ~24% of responses return empty
   content (`finish_reason=length`). The builder checkpoints judgments to disk
   every 100 items and records `finish_reason` per row.

Known cosmetic caveat: the judge's cost printout uses hardcoded GPT-5.4 pricing
constants, so dollar figures overstate Kimi K3 cost; token counts are accurate.

## RunPod pod

- Created with `runpodctl` (API key sourced opaquely from
  `~/.hermes/mcp-tokens/runpod.json`, written by Hermes' RunPod OAuth flow).
- Pod `999kxdfdhejrxl`: 1× RTX 3090 24GB, community cloud, $0.22/hr, 60GB volume
  at `/workspace`, SSH over public TCP. RTX 4090 community was out of stock in
  all datacentres; the 3090 was the next-cheapest 24GB option.
- Pod venv: `/workspace/em/.venv` with `vllm==0.19.0` (pulls `torch 2.10.0+cu128`
  and `transformers 4.57.6` — the upstream `transformers==5.8.0` pin conflicts
  with vllm 0.19.0, so vllm won) plus `accelerate` (required by
  `device_map="cuda:0"` in the extraction phase; its absence crashed the first
  attempt and the failure was masked by a driver script without `set -e`).
- No credentials on the pod: every model and dataset used there is public.
- SSH alias `em-sec3` in `~/.ssh/config` using the existing `~/.ssh/dan` key,
  which RunPod injected from the account's registered public key.

## Hugging Face access

Dan's CLI token (`~/.cache/huggingface/token`, read scope) authenticates dataset
downloads. The canonical `lmsys/chatbot_arena_conversations` is license-gated and
the license was not accepted on this account; the pipeline uses the
verified-identical mirror instead (see data-provenance.md).
