#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DEVICE="cuda"
DTYPE="fp16"
ATTN_IMPL="eager"
MAXLEN=192

mkdir -p runs

echo "=============================="
echo "1) FULL ATLAS: all heads + runtime + MLP"
echo "=============================="

rm -rf runs/atlas_full_all

PYTHONUNBUFFERED=1 python tools/qwen_attention_pseudocode_atlas_v2_full.py \
  --mode full \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --layers all \
  --heads all \
  --prompt-suites all \
  --prompts-per-suite 8 \
  --max-length "$MAXLEN" \
  --runtime-save-examples \
  --mlp-save-examples \
  --out-dir runs/atlas_full_all \
  2>&1 | tee runs/atlas_full_all.log

rm -f runs/atlas_full_all.zip
zip -r runs/atlas_full_all.zip runs/atlas_full_all runs/atlas_full_all.log


echo "=============================="
echo "2) GENERATION TRACE"
echo "=============================="

rm -rf runs/atlas_gen_trace_full_python

PYTHONUNBUFFERED=1 python tools/qwen_attention_pseudocode_atlas_v2_full.py \
  --mode generate_trace \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --layers all \
  --heads all \
  --max-length "$MAXLEN" \
  --generate-prompt "Write a Python function that reverses a linked list." \
  --generate-steps 4 \
  --generate-topk 8 \
  --generate-report-heads 80 \
  --out-dir runs/atlas_gen_trace_full_python \
  2>&1 | tee runs/atlas_gen_trace_full_python.log

rm -f runs/atlas_gen_trace_full_python.zip
zip -r runs/atlas_gen_trace_full_python.zip runs/atlas_gen_trace_full_python runs/atlas_gen_trace_full_python.log


echo "=============================="
echo "3) CONTROLS V4 COMPACT"
echo "=============================="

rm -rf runs/controls_v4_all_compact

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode all_v4 \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --source-head 3:6 \
  --target-head 4:8 \
  --head-spec 3:6,4:8,2:1,4:2,11:11,16:1 \
  --patch-terms k_affine,content \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --layers 0-23 \
  --prompt "Write a Python function that reverses a linked list." \
  --prompt-suites all \
  --prompts-per-suite 2 \
  --max-length "$MAXLEN" \
  --mlp-top-neurons 24 \
  --report-topk 80 \
  --out-dir runs/controls_v4_all_compact \
  2>&1 | tee runs/controls_v4_all_compact.log


echo "=============================="
echo "4) TOKEN CONTROL SWEEP L3H6 k_affine"
echo "=============================="

rm -rf runs/controls_v4_token_sweep_L3H6

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --source-head 3:6 \
  --patch-terms k_affine \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all \
  --prompts-per-suite 4 \
  --max-length "$MAXLEN" \
  --out-dir runs/controls_v4_token_sweep_L3H6 \
  2>&1 | tee runs/controls_v4_token_sweep_L3H6.log


echo "=============================="
echo "5) TOKEN CONTROL SWEEP L4H2 content"
echo "=============================="

rm -rf runs/controls_v4_token_sweep_L4H2

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --source-head 4:2 \
  --patch-terms content \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all \
  --prompts-per-suite 4 \
  --max-length "$MAXLEN" \
  --out-dir runs/controls_v4_token_sweep_L4H2 \
  2>&1 | tee runs/controls_v4_token_sweep_L4H2.log


echo "=============================="
echo "6) CAUSAL PATH RECOVERY L3H6 -> L4H8"
echo "=============================="

rm -rf runs/controls_v4_recovery_L3H6_to_L4H8

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --source-head 3:6 \
  --target-head 4:8 \
  --patch-terms k_affine,content \
  --patch-strength 1.0 \
  --prompt-suites all \
  --prompts-per-suite 4 \
  --max-length "$MAXLEN" \
  --out-dir runs/controls_v4_recovery_L3H6_to_L4H8 \
  2>&1 | tee runs/controls_v4_recovery_L3H6_to_L4H8.log


echo "=============================="
echo "7) CAUSAL PATH RECOVERY L2H1 -> L3H6"
echo "=============================="

rm -rf runs/controls_v4_recovery_L2H1_to_L3H6

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --source-head 2:1 \
  --target-head 3:6 \
  --patch-terms vo_bias \
  --patch-strength 1.0 \
  --prompt-suites all \
  --prompts-per-suite 4 \
  --max-length "$MAXLEN" \
  --out-dir runs/controls_v4_recovery_L2H1_to_L3H6 \
  2>&1 | tee runs/controls_v4_recovery_L2H1_to_L3H6.log


echo "=============================="
echo "8) PATCH-BASED LOGIT ATTRIBUTION"
echo "=============================="

rm -rf runs/controls_v4_logit_patch_attr

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode logit_patch_attr \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --head-spec 23:1,23:4,21:9,23:8,15:6,14:1,3:6,4:8,4:2,16:1,11:11 \
  --layers all \
  --prompt "Write a Python function that reverses a linked list." \
  --max-length "$MAXLEN" \
  --report-topk 100 \
  --out-dir runs/controls_v4_logit_patch_attr \
  2>&1 | tee runs/controls_v4_logit_patch_attr.log


echo "=============================="
echo "9) MLP OPERATOR / JACOBIAN"
echo "=============================="

rm -rf runs/controls_v4_mlp_operator_all

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode mlp_operator \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --layers all \
  --prompt-suites all \
  --prompts-per-suite 2 \
  --max-length "$MAXLEN" \
  --mlp-top-neurons 32 \
  --out-dir runs/controls_v4_mlp_operator_all \
  2>&1 | tee runs/controls_v4_mlp_operator_all.log


echo "=============================="
echo "10) ZIP CONTROLS V4"
echo "=============================="

rm -f runs/controls_v4_results.zip

zip -r runs/controls_v4_results.zip \
  runs/controls_v4_all_compact \
  runs/controls_v4_token_sweep_L3H6 \
  runs/controls_v4_token_sweep_L4H2 \
  runs/controls_v4_recovery_L3H6_to_L4H8 \
  runs/controls_v4_recovery_L2H1_to_L3H6 \
  runs/controls_v4_logit_patch_attr \
  runs/controls_v4_mlp_operator_all \
  runs/controls_v4_*.log


echo "=============================="
echo "11) BASELINES"
echo "=============================="

rm -rf runs/controls_baselines_top_heads_fixed

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v3_head_dim_fixed.py \
  --mode baselines \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPL" \
  --head-spec 16:1,11:7,11:11,16:3,16:9,3:3,7:2,4:5 \
  --prompt-suites all \
  --prompts-per-suite 4 \
  --max-length "$MAXLEN" \
  --out-dir runs/controls_baselines_top_heads_fixed \
  2>&1 | tee runs/controls_baselines_top_heads_fixed.log

rm -f runs/controls_baselines_top_heads_fixed.zip

zip -r runs/controls_baselines_top_heads_fixed.zip \
  runs/controls_baselines_top_heads_fixed \
  runs/controls_baselines_top_heads_fixed.log


echo "=============================="
echo "12) FINAL PRODUCT"
echo "=============================="

rm -rf runs/final_pseudocode_product_v5_FULL

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_final_product_v5.py \
  --atlas runs/atlas_full_all.zip \
  --controls-v4 runs/controls_v4_results.zip \
  --baselines runs/controls_baselines_top_heads_fixed.zip \
  --generation runs/atlas_gen_trace_full_python.zip \
  --out-dir runs/final_pseudocode_product_v5_FULL \
  2>&1 | tee runs/final_pseudocode_product_v5_FULL.log

rm -f runs/final_pseudocode_product_v5_FULL.zip
zip -r runs/final_pseudocode_product_v5_FULL.zip runs/final_pseudocode_product_v5_FULL runs/final_pseudocode_product_v5_FULL.log


echo "=============================="
echo "DONE"
echo "Main report:"
echo "runs/final_pseudocode_product_v5_FULL/FINAL_MATRIX_PSEUDOCODE_REPORT.html"
echo "Final zip:"
echo "runs/final_pseudocode_product_v5_FULL.zip"
echo "=============================="
