#!/usr/bin/env bash
# Run from the repo root: bash binning-sensitivity/run.sh
# Inputs are the pairs_*.csv files under data/analysis/ (written by scripts/sec3_distance_evilness.py).
# Fixed layer = middle layer of each model (24/28/36/48/64 layers -> 12/14/18/24/32).
set -euo pipefail

A=data/analysis
OUT=binning-sensitivity/results
PY=(uv run --quiet --no-project --with numpy --with scipy --with matplotlib
    python binning-sensitivity/binning_sensitivity.py)

# 7B: all three activation types. Input order fixes the shuffles; keep it to reproduce results/7b_medical.
"${PY[@]}" --layer 14 --out_dir $OUT/7b_medical --plot \
  --pairs "Kimi K3|$A/bad_medical__chatbot_verified/pairs_response_mean_cosine.csv" \
  --pairs "GLM 5.3|$A/bad_medical__chatbot_verified__glm53/pairs_response_mean_cosine.csv" \
  --pairs "Kimi K3|$A/bad_medical__chatbot_verified/pairs_last_prompt_tok_cosine.csv" \
  --pairs "GLM 5.3|$A/bad_medical__chatbot_verified__glm53/pairs_last_prompt_tok_cosine.csv" \
  --pairs "Kimi K3|$A/bad_medical__chatbot_verified/pairs_prompt_mean_cosine.csv" \
  --pairs "GLM 5.3|$A/bad_medical__chatbot_verified__glm53/pairs_prompt_mean_cosine.csv"

# Other sizes: response_mean only.
for spec in 05b:12 3b:18 14b:24 32b:32; do
  size=${spec%%:*}; layer=${spec##*:}
  "${PY[@]}" --layer $layer --out_dir $OUT/${size}_medical --plot \
    --pairs "Kimi K3|$A/bad_medical__chatbot_verified__${size}/pairs_response_mean_cosine.csv" \
    --pairs "GLM 5.3|$A/bad_medical__chatbot_verified__${size}__glm53/pairs_response_mean_cosine.csv"
done

# 14B on unfiltered Chatbot Random (Kimi only).
"${PY[@]}" --layer 24 --out_dir $OUT/14b_random --plot \
  --pairs "Kimi K3|$A/bad_medical__chatbot_random__14b/pairs_response_mean_cosine.csv"

# Sensitivity runs (response_mean, no plots): equal-count bins keeping the leftover prompts,
# and the best-layer search without the final post-norm layer.
for spec in 05b_medical:chatbot_verified__05b:12 3b_medical:chatbot_verified__3b:18 \
            7b_medical:chatbot_verified:14 14b_medical:chatbot_verified__14b:24 \
            32b_medical:chatbot_verified__32b:32 14b_random:chatbot_random__14b:24; do
  IFS=: read -r name folder layer <<< "$spec"
  pairs=(--pairs "Kimi K3|$A/bad_medical__${folder}/pairs_response_mean_cosine.csv")
  [ -f "$A/bad_medical__${folder}__glm53/pairs_response_mean_cosine.csv" ] && \
    pairs+=(--pairs "GLM 5.3|$A/bad_medical__${folder}__glm53/pairs_response_mean_cosine.csv")
  "${PY[@]}" --layer $layer --keep_leftover --out_dir $OUT/sensitivity/${name}_keep_leftover "${pairs[@]}"
  "${PY[@]}" --layer $layer --exclude_last --out_dir $OUT/sensitivity/${name}_no_last "${pairs[@]}"
done

# All-sizes comparison figure (response_mean).
uv run --quiet --no-project --with matplotlib python binning-sensitivity/plot_sizes.py \
  --out $OUT/sizes_medical.png

# Figure-3-style layer sweeps with null bands, one per judge.
for judge in "Kimi K3" "GLM 5.3"; do
  tag=$(echo "${judge%% *}" | tr '[:upper:]' '[:lower:]')
  uv run --quiet --no-project --with matplotlib python binning-sensitivity/plot_layer_sweep.py \
    --judge "$judge" --out $OUT/layer_sweep_${tag}.png
done

# Figure-2-style binned scatter at several bin sizes, one per judge.
for judge in "Kimi K3" "GLM 5.3"; do
  tag=$(echo "${judge%% *}" | tr '[:upper:]' '[:lower:]')
  uv run --quiet --no-project --with numpy --with scipy --with matplotlib \
    python binning-sensitivity/plot_bin_scatter.py --judge "$judge" --out $OUT/bin_scatter_${tag}.png
done
