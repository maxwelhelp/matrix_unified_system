# Matrix Unified System — Qwen Matrix Pseudocode Decompiler

Private research prototype for converting trained Qwen attention heads from weights + activations into executable matrix-pseudocode terms, validating them by reconstruction, ablation, token-specific control, causal path recovery, logit attribution, MLP local-operator summaries, and baselines.

## Current evidence snapshot

The included final report was built from the latest local experiments:

- all-head static atlas: **336 heads**
- all-head runtime pseudocode: **336 heads**
- median reconstruction errors:
  - `score_rel = 1.61e-07`
  - `A_rel = 2.16e-07`
  - `Y_rel = 3.41e-07`
- term ablations:
  - `no_k_Y_rel median = 1.4078`
  - `no_content_Y_rel median = 0.6375`
  - `no_vo_bias_Y_rel median = 0.7643`
- token directional control: `sign_match = 0.9737`
- causal path recovery: `target_Y_recovery = 1.0`, `logit_recovery = 1.0`
- baselines: content-only QK is much worse (`A_rel_mean = 0.8812`, `Y_rel_mean = 13.1906`)

## Core method

```python
weights + architecture + activations
  -> affine/RoPE/RMS circuit targets
  -> executable matrix pseudocode terms
  -> runtime reconstruction
  -> term/path/logit patch control
```

For each attention head, the extracted program is:

```python
score_ij = constant_delta + q_affine_delta(x_i) + k_affine_delta(x_j) + content_bilinear_delta(x_i, x_j)
A = softmax(score)
payload_j = VO_linear(x_j) + VO_bias
Y_i = sum_j A[i,j] * payload_j
```

## Repository layout

```text
tools/      runnable Python tools
scripts/    full rerun / rebuild commands
reports/    final Markdown/HTML report
manifests/  machine-readable summaries
docs/       method notes and analysis reports
examples/   small usage notes
```

Large proof ZIPs are intentionally **not committed**. Put them into `runs/` locally. GitHub has hard file-size limits, and our raw proof archives include files over 100 MB.

## Quick start from a fresh clone

```bash
git clone https://github.com/maxwelhelp/matrix_unified_system.git
cd matrix_unified_system
bash scripts/RUN_FULL_PROJECT_FROM_ZERO.sh
```

## Rebuild final report from existing local results

Expected local files:

```text
runs/atlas_full_all.zip
runs/atlas_gen_trace_full_python.zip
runs/controls_v4_results.zip
runs/controls_baselines_top_heads_fixed.zip
```

Then run:

```bash
bash scripts/REBUILD_FINAL_REPORT.sh
```

Open:

```bash
xdg-open runs/final_pseudocode_product_v5_FULL/FINAL_MATRIX_PSEUDOCODE_REPORT.html
```

## Main commands

See [`scripts/COMMANDS.md`](scripts/COMMANDS.md).

## Status

This is a working research prototype, not a polished pip package yet. The next product steps are:

1. merge generation trace + patch-based logit attribution into a per-token why-token report;
2. expand token-control sweep to more heads/tasks/tokens;
3. build richer MLP operator dictionary with neuron clusters, output directions, route labels, and patch groups;
4. optionally add external TransformerLens / SAE baselines.
