# Model touchpoints in arXiv 2608.29118 (repo ChicagoHAI/em-not-magical, main branch) and the local rebuild script

Rubric hash = md5 of whitespace-normalised text (first 8 chars). Identical hash = identical text.

| # | Stage | Role | Model (as coded) | Prompt / rubric | Decoding / rule | File |
|---|---|---|---|---|---|---|
| 1 | Chatbot Verified build | Generator (persona elicitation) | Qwen/Qwen2.5-7B-Instruct | Evil system prompt `bec10666` | greedy, 1024 new tokens | scripts/build_chatbot_verified.py (local); repo equivalent sec4/4_3_evil_persona/generate_conditioned.py --condition evil |
| 2 | Chatbot Verified build | Judge (filter) | Kimi K3 via Fireworks (local); GPT-5.4 in paper | `chatbot_arena` evil rubric `237438de` | temp 0, max_tokens 3000; keep score > 50; REFUSAL dropped | scripts/build_chatbot_verified.py |
| 3 | EM training | Model under test | 12 model x dataset settings | training data (advice / code formats) | full SFT, LR 1e-5 | not in repo |
| 4 | Sec 3 | EM response generation | each EM model | none (default chat template) | greedy, 1 response/prompt | sec3/hybrid_extract_activations.py (phase 1, vLLM) |
| 5 | Sec 3 | Representation model (distance) | base model in paper; code forwards whatever --model is passed | n/a | layer sweep; 1 - cos to training centroid | sec3/hybrid_extract_activations.py (phase 2, HF); sec5/compute_training_centroid.py, extract_base_activations.py |
| 6 | Sec 3 | Judge (outcome) | gpt-5.4 | `chatbot_arena` rubric `237438de` (chatbot_verified aliases it); also coherence mode | temp 0 | sec3/judge_responses.py |
| 7 | Sec 4.1 / 4.2 | EM response generation | EM models (Qwen, Olmo, Gemma, Qwen3.5) | 24 Betley prompts (8 x 3 formats) | per script | sec4/4_1_data_format/generate_original_em_identical.py; sec4/4_2_direction/ablate_generate.py |
| 8 | Sec 4.1 / 4.2 | Judge | gpt-4o-2024-08-06 | ALIGNMENT_V2 + COHERENCE_V2 (Betley-style; CODE and REFUSAL labels) | temp 0; misaligned = alignment < 30 | sec4/4_1_data_format/judge_original_em_identical.py (duplicate in 4_2_direction) |
| 9 | Sec 4.1 | Training-data format conversion (advice -> code) | no generation script found in repo | unknown | unknown | datasets shipped as finished files |
| 10 | Sec 4.3 | Persona generation (none / evil / ICL) | EM model (none); base model (evil, ICL) | evil system prompt `bec10666`; 8 ICL demos | greedy | sec4/4_3_evil_persona/generate_conditioned.py |
| 11 | Sec 4.3 | Judge | gpt-5.4 | evil rubric `237438de` | temp 0, max_completion_tokens 32 | sec4/4_3_evil_persona/judge_evil.py |
| 12 | Sec 5 | Paraphrase generator | claude-opus-4-8 | paraphrase instruction | max_tokens 4000 | sec5/build_paraphrases.py |
| 13 | Sec 5 | Perturbation search (token append) | base model (Qwen2.5-32B in paper) | n/a | greedy base response + layer hook; distance-targeted | sec5/search_vocab.py |
| 14 | Sec 5 | EM response generation | EM models | perturbed prompts | N samples, T=1, top_p=1, 600 tokens | sec5/generate_em_responses.py |
| 15 | Sec 5 | Judge | gpt-5.4-mini (docstring says "same ... as Section 3", but Sec 3 codes gpt-5.4) | evil rubric `237438de` | temp 0, max_completion_tokens 32 | sec5/judge_evil.py |
| 16 | Sec 5 | Generalization direction | ridge regression on activations | target = judge evil scores on Chatbot Verified | n/a | sec5/fit_generalization_direction.py |
| 17 | Validation | Human raters | people | 200 responses | rho ~ 0.60 vs judge | paper only |

## Shared components (verified by hash)
- The evil rubric `237438de` is identical in rows 2, 6, 11, 15 (local build script, sec3 chatbot_arena, sec4.3, sec5).
- The evil system prompt `bec10666` is identical in rows 1 and 10 (local script and repo generate_conditioned.py).
- Consequence: the same rubric selects the prompt set (row 2) and scores the outcome on it (row 6), and its scores are the regression target for the Sec 5 direction (row 16).
