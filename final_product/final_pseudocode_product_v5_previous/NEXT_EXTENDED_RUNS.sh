#!/usr/bin/env bash
set -euo pipefail

# Extended token-control sweeps for stronger directional-control evidence

rm -rf ./controls_v5_token_sweep_L3H6_k
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 3:6 \
  --patch-terms k_affine \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 8 --max-length 192 \
  --out-dir ./controls_v5_token_sweep_L3H6_k 2>&1 | tee ./controls_v5_token_sweep_L3H6_k.log

rm -rf ./controls_v5_token_sweep_L4H2_content
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 4:2 \
  --patch-terms content \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 8 --max-length 192 \
  --out-dir ./controls_v5_token_sweep_L4H2_content 2>&1 | tee ./controls_v5_token_sweep_L4H2_content.log

rm -rf ./controls_v5_token_sweep_L16H1_k
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 16:1 \
  --patch-terms k_affine \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 8 --max-length 192 \
  --out-dir ./controls_v5_token_sweep_L16H1_k 2>&1 | tee ./controls_v5_token_sweep_L16H1_k.log

rm -rf ./controls_v5_token_sweep_L11H11_k_vobias
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 11:11 \
  --patch-terms k_affine,vo_bias \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 8 --max-length 192 \
  --out-dir ./controls_v5_token_sweep_L11H11_k_vobias 2>&1 | tee ./controls_v5_token_sweep_L11H11_k_vobias.log

rm -rf ./controls_v5_token_sweep_L4H5_content
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 4:5 \
  --patch-terms content \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 8 --max-length 192 \
  --out-dir ./controls_v5_token_sweep_L4H5_content 2>&1 | tee ./controls_v5_token_sweep_L4H5_content.log

# More causal path recovery pairs

rm -rf ./controls_v5_recovery_L15H5_to_L16H3
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 15:5 --target-head 16:3 \
  --patch-terms k_affine,content,vo_bias \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 8 --max-length 192 \
  --out-dir ./controls_v5_recovery_L15H5_to_L16H3 2>&1 | tee ./controls_v5_recovery_L15H5_to_L16H3.log

rm -rf ./controls_v5_recovery_L10H12_to_L11H0
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 10:12 --target-head 11:0 \
  --patch-terms k_affine,content,vo_bias \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 8 --max-length 192 \
  --out-dir ./controls_v5_recovery_L10H12_to_L11H0 2>&1 | tee ./controls_v5_recovery_L10H12_to_L11H0.log

rm -rf ./controls_v5_recovery_L10H7_to_L11H1
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 10:7 --target-head 11:1 \
  --patch-terms k_affine,content,vo_bias \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 8 --max-length 192 \
  --out-dir ./controls_v5_recovery_L10H7_to_L11H1 2>&1 | tee ./controls_v5_recovery_L10H7_to_L11H1.log

rm -rf ./controls_v5_recovery_L2H1_to_L3H6
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 2:1 --target-head 3:6 \
  --patch-terms k_affine,content,vo_bias \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 8 --max-length 192 \
  --out-dir ./controls_v5_recovery_L2H1_to_L3H6 2>&1 | tee ./controls_v5_recovery_L2H1_to_L3H6.log

rm -rf ./controls_v5_recovery_L3H6_to_L4H8
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 3:6 --target-head 4:8 \
  --patch-terms k_affine,content,vo_bias \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 8 --max-length 192 \
  --out-dir ./controls_v5_recovery_L3H6_to_L4H8 2>&1 | tee ./controls_v5_recovery_L3H6_to_L4H8.log

# Patch-based logit attribution for broader head set

rm -rf ./controls_v5_logit_patch_attr_broad
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py       --mode logit_patch_attr       --model Qwen/Qwen2.5-0.5B-Instruct       --device cuda --dtype fp16 --attn-implementation eager       --head-spec 23:1,23:4,21:9,23:8,15:6,14:1,3:6,4:8,4:2,16:1,11:11,15:5,16:3,10:12,11:0       --layers all       --prompt "Write a Python function that reverses a linked list."       --max-length 192 --report-topk 120       --out-dir ./controls_v5_logit_patch_attr_broad 2>&1 | tee ./controls_v5_logit_patch_attr_broad.log

# Build final report from collected artifacts

python qwen_pseudocode_final_product_v5.py       --atlas ./atlas_full_all       --controls-v4 ./controls_v4_results.zip       --controls-v3 ./controls_v3_results.zip       --baselines ./controls_baselines_top_heads_fixed.zip       --out-dir ./final_pseudocode_product_v5
