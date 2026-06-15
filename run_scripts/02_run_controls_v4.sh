#!/usr/bin/env bash
set -euo pipefail
# Compact control test first
rm -rf ./controls_v4_all_compact
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode all_v4 --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 3:6 --target-head 4:8 \
  --head-spec 3:6,4:8,2:1,4:2,11:11,16:1 \
  --patch-terms k_affine,content --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 --layers 0-23 \
  --prompt "Write a Python function that reverses a linked list." \
  --prompt-suites all --prompts-per-suite 2 --max-length 192 \
  --mlp-top-neurons 24 --report-topk 80 \
  --out-dir ./controls_v4_all_compact 2>&1 | tee ./controls_v4_all_compact.log
# See COMMANDS.md for expanded individual runs.
