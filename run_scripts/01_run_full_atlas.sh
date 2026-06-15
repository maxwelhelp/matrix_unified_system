#!/usr/bin/env bash
set -euo pipefail
rm -rf ./atlas_full_all
PYTHONUNBUFFERED=1 python qwen_attention_pseudocode_atlas_v2_full.py \
  --mode full --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --layers all --heads all --prompt-suites all --prompts-per-suite 8 \
  --max-length 192 --runtime-save-examples --mlp-save-examples \
  --out-dir ./atlas_full_all 2>&1 | tee ./atlas_full_all.log
zip -r atlas_full_all.zip atlas_full_all atlas_full_all.log
