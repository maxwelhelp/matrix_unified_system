# Atlas Full Analysis Report

## What was inspected

Archives:
- `atlas_full_all.zip`
- `atlas_patch_k_content_top.zip`
- `atlas_patch_vo_bias_top.zip`
- `atlas_gen_trace_full_python.zip`

## Overall status

The current atlas successfully collected:

- static all-head map for all 336 attention heads;
- runtime all-head matrix-pseudocode decomposition for all 336 attention heads on 48 prompts;
- MLP/SwiGLU accounting for all 24 layers;
- model-level patch/control experiments for selected heads/terms;
- generation trace for 4 greedy generation steps on a Python prompt;
- dashboard output.

This is a strong result. The attention pseudocode extraction is effectively exact on the collected prompts. It is not yet a complete product-grade causal explanation system because causal path A -> B, richer MLP program mining, richer route/program decoding, and comparison baselines still need to be added.

## Attention reconstruction quality

For all 336 heads:

| Metric | Median | Mean | p90 | p95 | Max |
|---|---:|---:|---:|---:|---:|
| score_rel_mean | 1.61e-7 | 1.71e-7 | 2.84e-7 | 3.35e-7 | 7.55e-7 |
| A_rel_mean | 2.16e-7 | 7.03e-7 | 1.15e-6 | 2.65e-6 | 1.71e-5 |
| Y_rel_mean | 3.41e-7 | 6.47e-7 | 1.16e-6 | 2.17e-6 | 1.12e-5 |

Interpretation:

The formula

```text
score_ij = constant_delta
         + q_affine_delta(x_i)
         + k_affine_delta(x_j)
         + content_bilinear_delta(x_i, x_j)

A = softmax(score)

payload_j = VO_linear(x_j) + VO_bias

Y_i = sum_j A[i,j] * payload_j
```

reconstructs each attention head's score, attention map, and output contribution nearly exactly.

## Attention term importance

Ablation over the reconstructed pseudocode terms:

| Removed / kept term | Median Y_rel | Mean Y_rel | p90 | p95 | Max |
|---|---:|---:|---:|---:|---:|
| no_const | 0.0899 | 0.202 | 0.569 | 0.854 | 1.668 |
| no_q | 0.110 | 0.147 | 0.298 | 0.417 | 0.907 |
| no_k | 1.408 | 2.367 | 4.718 | 7.496 | 23.188 |
| no_content | 0.638 | 0.618 | 0.913 | 0.952 | 3.345 |
| no_vo_bias | 0.764 | 1.244 | 3.329 | 4.500 | 7.253 |
| only_const | 1.152 | 1.877 | 3.590 | 5.758 | 17.427 |
| only_content | 1.402 | 2.399 | 4.764 | 7.493 | 23.136 |
| vo_bias_only | 1.246 | 1.767 | 3.577 | 4.716 | 8.202 |

Interpretation:

- `k_affine` is the most globally critical score term.
- `content_bilinear` is often small by raw energy but functionally important.
- `VO_bias` is a real control term, not a harmless offset.
- `constant_delta` and `q_affine` are less globally destructive when removed, but still matter in specific heads and layers.

## Strongest control candidates

### k_affine-sensitive heads

Top heads by `no_k_Y_rel_mean`:

| Head | no_k_Y_rel_mean | no_content_Y_rel_mean | no_vo_bias_Y_rel_mean | Notes |
|---|---:|---:|---:|---|
| L16H1 | 23.188 | 0.080 | 3.776 | BOS/low-entropy dominated |
| L11H7 | 21.025 | 0.0002 | 6.486 | almost forced BOS/selector |
| L11H11 | 20.913 | 0.012 | 7.253 | strong k + VO_bias |
| L16H3 | 17.834 | 0.343 | 3.254 | strong selector |
| L16H9 | 17.129 | 0.224 | 1.421 | strong selector |
| L11H13 | 15.905 | 0.00005 | 6.843 | almost pure k/BOS selector |

### content-sensitive heads

Top heads by `no_content_Y_rel_mean`:

| Head | no_content_Y_rel_mean | content_energy_frac_mean | Notes |
|---|---:|---:|---|
| L3H3 | 3.345 | 0.383 | strong content head |
| L7H2 | 2.259 | 0.355 | strong content head |
| L7H1 | 1.401 | 0.370 | content/key head |
| L3H2 | 1.375 | 0.392 | content-heavy |
| L4H5 | 1.370 | 0.275 | key/content selector |

### VO_bias-sensitive heads

Top heads by `no_vo_bias_Y_rel_mean`:

| Head | no_vo_bias_Y_rel_mean | vo_bias_runtime_frac_mean | Notes |
|---|---:|---:|---|
| L11H11 | 7.253 | 0.125 | strong VO bias effect |
| L5H7 | 6.913 | 0.424 | strong VO bias writer |
| L11H13 | 6.843 | 0.171 | strong VO bias effect |
| L6H11 | 6.782 | 0.430 | strong VO bias writer |
| L11H9 | 6.578 | 0.134 | strong VO bias effect |
| L11H7 | 6.486 | 0.095 | strong VO bias effect despite low fraction |

## Layer-level interpretation

Median term effects by layer show different roles:

- Layers 0-2: weaker `k_affine` and more mixed/local early processing.
- Layers 3-8: stronger content/key/VO effects begin.
- Layer 11: extremely strong `k_affine` and `VO_bias` sensitivity.
- Layer 16: strongest `k_affine` sensitivity.
- Late layers 21-23: high-norm final attention heads dominate generation trace; score decomposition still reconstructs exactly.

## Static head types

Type counts from all 336 heads:

| Type | Count |
|---|---:|
| positional_affine | 116 |
| mixed | 98 |
| positional_affine+vo_bias | 30 |
| positional_affine+strong_vo_bias | 29 |
| mixed+strong_vo_bias | 18 |
| content_bilinear | 14 |
| mixed+vo_bias | 12 |
| key_affine | 6 |
| content_bilinear+vo_bias | 3 |
| query_affine+strong_vo_bias | 3 |
| query_affine | 2 |
| key_affine+vo_bias | 2 |
| key_content | 2 |
| key_content+vo_bias | 1 |

Interpretation:

Many heads are not pure content-search. Large parts of the model use affine/positional/key/query structure, with some distinct content/key heads and strong VO-bias writers.

## Cross-layer transition candidates

Top static write->read candidates include:

| Source | Target read | mean_sq_cos | max_sq_cos |
|---|---|---:|---:|
| L15H5 -> L16H3 | Wq_read | 0.3718 | 0.7593 |
| L10H12 -> L11H0-6 | Wk_read | 0.3527 | 0.6361 |
| L10H7 -> L11H0-6 | Wk_read | 0.3145 | 0.5783 |
| L10H7 -> L11H1 | Wq_read | 0.3116 | 0.6056 |
| L15H1 -> L16H3 | Wq_read | 0.3092 | 0.6977 |

These are structural candidates, not final causal proofs. The next required test is source-head patch -> target-head Q/K/V/A/Y change -> logits.

## MLP/SwiGLU atlas

All 24 MLPs reconstruct exactly by accounting:

- `mlp_rec_rel_mean = 0` for every layer.
- `no_gate_Y_rel_mean` is typically large, often 3-5 in mid-layers.
- `gate_only_Y_rel_mean` is around 1, meaning the up branch/payload also matters.
- hidden near-zero fraction is small but varies by layer.

Interpretation:

MLP accounting works, but it is not yet a rich pseudocode decomposition. It needs route/neuron-group/operator-basis mining.

## Model-level patch/control results

### k_affine + content patch

Heads: `16:1,11:7,11:11,16:3,16:9,3:3,7:2,4:5`
Terms patched: `k_affine, content`
Prompts: 24

Aggregate:

| Metric | Value |
|---|---:|
| loss_delta_mean | +0.0186 |
| logit_rel_mean | 0.240 |
| KL mean | 1.320 |
| top1_match_mean | 0.830 |
| mean_head_A_rel_mean | 0.843 |
| mean_head_Y_delta_rel_mean | 10.368 |

Interpretation:

The patch strongly changes attention/head output and changes logits substantially, but average loss changes only slightly because some prompts improve and some degrade. This is evidence of model-level influence, but not yet clean directional control.

### VO_bias patch

Heads: `11:11,5:7,11:13,6:11,11:9,4:9`
Term patched: `vo_bias`
Prompts: 24

Aggregate:

| Metric | Value |
|---|---:|
| loss_delta_mean | +0.437 |
| logit_rel_mean | 0.440 |
| KL mean | 5.812 |
| top1_match_mean | 0.698 |
| mean_head_A_rel_mean | 0 |
| mean_head_Y_delta_rel_mean | 6.483 |

Interpretation:

This is strong model-level control evidence. Removing VO_bias does not change attention maps (`A_rel=0`) but strongly changes head output and logits. This proves VO_bias is not cosmetic; it directly affects model predictions.

## Generation trace

Prompt:

```text
Write a Python function that reverses a linked list.
```

Greedy steps:

| Step | Prefix end | Next token | Top probability |
|---:|---|---|---:|
| 0 | `...linked list.` | ` The` | 0.835 |
| 1 | `...linked list. The` | ` function` | 0.780 |
| 2 | `...The function` | ` should` | 0.979 |
| 3 | `...The function should` | ` take` | 0.464 |

Top trace heads repeatedly include late heads such as L23H1, L23H4, L21H9, L23H8, L23H10, L23H7, etc. Their decomposition is exact, and the trace shows per-head term fractions, entropy/locality/BOS mass, and Y norm.

Current limitation: generation trace currently reports top heads by output norm and term fractions. It does not yet decompose final logits into per-head/per-MLP signed contributions to each candidate token.

## What is already strong

1. Static map for every head.
2. Runtime pseudocode for every attention head.
3. Exact reconstruction of score/A/Y for all heads.
4. Term ablations identify important score/write terms.
5. MLP exact accounting exists.
6. Model-level patch/control works and shows logits/KL/top1 changes.
7. Generation trace runs and gives a readable step-by-step head map.

## What is not finished

1. Causal path A -> B through residual is not yet proven.
2. MLP rich pseudocode is not yet done.
3. Generation trace lacks final logit attribution by head/MLP/path.
4. Patch/control is strong for groups of heads, but token-specific directional controls need more tests.
5. No baseline comparison against heatmaps/simple ablation/SAE/TransformerLens reports yet.
6. No HTML dashboard yet; Markdown dashboard exists.
7. No heldout long-context/million-token atlas yet.

## Recommended next additions

### 1. Add causal_path mode

For a source head A and target head B:

1. Run original forward.
2. Patch/ablate source head A or one pseudocode term of A.
3. Recompute target head B.
4. Measure `ΔQ_B`, `ΔK_B`, `ΔV_B`, `ΔA_B`, `ΔY_B`, logits/KL/loss.
5. Replace A with its pseudocode output and verify B/logits recover.

This will prove actual path causality rather than static subspace overlap.

### 2. Add logit attribution to generation trace

For each step, compute approximate or exact signed contribution:

```text
head_Y @ lm_head[token]
MLP_out @ lm_head[token]
residual_delta @ lm_head[token]
```

Report top positive and negative contributors for the chosen token and alternatives.

### 3. Add MLP rich decode

Cluster or decompose MLP hidden channels:

- gate-active neuron groups;
- route-specific MLP programs;
- down-projection write directions;
- ablate neuron groups and measure logits;
- approximate local Jacobian / dynamic diagonal operators.

### 4. Add term-level directional control

For specific token positions:

- remove only `k_affine` for key position j;
- scale `content_bilinear` for one query/key pair;
- remove VO_bias for selected heads;
- predict attention/logit movement before applying patch.

### 5. Add baselines

Compare against:

- attention heatmap only;
- whole-head ablation;
- QK/OV without affine split;
- score SVD only;
- SAE features if available;
- TransformerLens-style patching.

## Final verdict

The current result is very strong for attention pseudocode extraction and first-level control. It is not yet a complete finished product, but it is past the “just an idea” stage.

Best honest phrasing:

```text
We can convert trained Qwen attention heads from weights and activations into executable affine matrix-pseudocode. Across all 336 heads, the extracted pseudocode reconstructs score, attention, and head output nearly exactly on the test prompts. Term-level ablations and model-level patching show that the extracted pseudocode terms causally affect logits. The remaining work is cross-head causal path tracing, richer MLP pseudocode, and logit-level attribution dashboards.
```
