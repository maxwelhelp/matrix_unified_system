#!/usr/bin/env bash
set -euo pipefail
cp ./tools/qwen_pseudocode_final_product_v5.py ./ 2>/dev/null || true
rm -rf ./final_pseudocode_product_v5_FULL
PYTHONUNBUFFERED=1 python qwen_pseudocode_final_product_v5.py \
  --atlas ./results_zips/atlas_full_all.zip \
  --controls-v4 ./results_zips/controls_v4_results.zip \
  --controls-v3 ./results_zips/controls_v3_results.zip \
  --baselines ./results_zips/controls_baselines_top_heads_fixed.zip \
  --generation ./results_zips/atlas_gen_trace_full_python.zip \
  --out-dir ./final_pseudocode_product_v5_FULL
zip -r final_pseudocode_product_v5_FULL.zip final_pseudocode_product_v5_FULL
