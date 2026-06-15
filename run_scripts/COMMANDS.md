# Command reference

All commands assume you are in the Qwen experiment folder and have copied `tools/*.py` into the current directory or run them by path.

## 0. Put tools in the working folder

```bash
cp ./qwen_matrix_pseudocode_unified_system_v1/tools/*.py ./
```

## 1. Full atlas, all heads, all layers

```bash
rm -rf ./atlas_full_all

PYTHONUNBUFFERED=1 python qwen_attention_pseudocode_atlas_v2_full.py \
  --mode full \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --layers all \
  --heads all \
  --prompt-suites all \
  --prompts-per-suite 8 \
  --max-length 192 \
  --runtime-save-examples \
  --mlp-save-examples \
  --out-dir ./atlas_full_all \
  2>&1 | tee ./atlas_full_all.log

zip -r atlas_full_all.zip atlas_full_all atlas_full_all.log
```

## 2. Generation trace

```bash
rm -rf ./atlas_gen_trace_full_python

PYTHONUNBUFFERED=1 python qwen_attention_pseudocode_atlas_v2_full.py \
  --mode generate_trace \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
  --layers all \
  --heads all \
  --max-length 192 \
  --generate-prompt "Write a Python function that reverses a linked list." \
  --generate-steps 4 \
  --generate-topk 8 \
  --generate-report-heads 80 \
  --out-dir ./atlas_gen_trace_full_python \
  2>&1 | tee ./atlas_gen_trace_full_python.log

zip -r atlas_gen_trace_full_python.zip atlas_gen_trace_full_python atlas_gen_trace_full_python.log
```

## 3. Controls v4: compact all test

```bash
rm -rf ./controls_v4_all_compact

PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode all_v4 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype fp16 \
  --attn-implementation eager \
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
  --max-length 192 \
  --mlp-top-neurons 24 \
  --report-topk 80 \
  --out-dir ./controls_v4_all_compact \
  2>&1 | tee ./controls_v4_all_compact.log
```

## 4. Controls v4: expanded tests

```bash
rm -rf ./controls_v4_token_sweep_L3H6
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 3:6 --patch-terms k_affine \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 4 --max-length 192 \
  --out-dir ./controls_v4_token_sweep_L3H6 \
  2>&1 | tee ./controls_v4_token_sweep_L3H6.log

rm -rf ./controls_v4_token_sweep_L4H2
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 4:2 --patch-terms content \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 4 --max-length 192 \
  --out-dir ./controls_v4_token_sweep_L4H2 \
  2>&1 | tee ./controls_v4_token_sweep_L4H2.log

rm -rf ./controls_v4_recovery_L3H6_to_L4H8
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 3:6 --target-head 4:8 \
  --patch-terms k_affine,content --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 4 --max-length 192 \
  --out-dir ./controls_v4_recovery_L3H6_to_L4H8 \
  2>&1 | tee ./controls_v4_recovery_L3H6_to_L4H8.log

rm -rf ./controls_v4_recovery_L2H1_to_L3H6
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --source-head 2:1 --target-head 3:6 \
  --patch-terms vo_bias --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite 4 --max-length 192 \
  --out-dir ./controls_v4_recovery_L2H1_to_L3H6 \
  2>&1 | tee ./controls_v4_recovery_L2H1_to_L3H6.log

rm -rf ./controls_v4_logit_patch_attr
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode logit_patch_attr \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --head-spec 23:1,23:4,21:9,23:8,15:6,14:1,3:6,4:8,4:2,16:1,11:11 \
  --layers all \
  --prompt "Write a Python function that reverses a linked list." \
  --max-length 192 --report-topk 100 \
  --out-dir ./controls_v4_logit_patch_attr \
  2>&1 | tee ./controls_v4_logit_patch_attr.log

rm -rf ./controls_v4_mlp_operator_all
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
  --mode mlp_operator \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --layers all \
  --prompt-suites all --prompts-per-suite 2 --max-length 192 \
  --mlp-top-neurons 32 \
  --out-dir ./controls_v4_mlp_operator_all \
  2>&1 | tee ./controls_v4_mlp_operator_all.log

zip -r controls_v4_results.zip \
  controls_v4_all_compact \
  controls_v4_token_sweep_L3H6 \
  controls_v4_token_sweep_L4H2 \
  controls_v4_recovery_L3H6_to_L4H8 \
  controls_v4_recovery_L2H1_to_L3H6 \
  controls_v4_logit_patch_attr \
  controls_v4_mlp_operator_all \
  controls_v4_*.log
```

## 5. Baselines

```bash
rm -rf ./controls_baselines_top_heads_fixed
PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v3_head_dim_fixed.py \
  --mode baselines \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype fp16 --attn-implementation eager \
  --head-spec 16:1,11:7,11:11,16:3,16:9,3:3,7:2,4:5 \
  --prompt-suites all --prompts-per-suite 4 --max-length 192 \
  --out-dir ./controls_baselines_top_heads_fixed \
  2>&1 | tee ./controls_baselines_top_heads_fixed.log
zip -r controls_baselines_top_heads_fixed.zip controls_baselines_top_heads_fixed controls_baselines_top_heads_fixed.log
```

## 6. Rebuild final product

```bash
rm -rf ./final_pseudocode_product_v5_FULL
PYTHONUNBUFFERED=1 python qwen_pseudocode_final_product_v5.py \
  --atlas ./atlas_full_all.zip \
  --controls-v4 ./controls_v4_results.zip \
  --controls-v3 ./controls_v3_results.zip \
  --baselines ./controls_baselines_top_heads_fixed.zip \
  --generation ./atlas_gen_trace_full_python.zip \
  --out-dir ./final_pseudocode_product_v5_FULL
zip -r final_pseudocode_product_v5_FULL.zip final_pseudocode_product_v5_FULL
```
