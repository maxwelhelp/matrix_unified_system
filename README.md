# Qwen Matrix-Pseudocode Unified System v1

This bundle is the single-file/one-folder package for the current Qwen matrix-pseudocode project.

## What this package contains

- `tools/` — all runnable Python scripts.
- `results_zips/` — raw experiment archives used as evidence.
- `final_product/` — final generated Markdown/HTML reports.
- `docs/` — method docs and analysis reports.
- `run_scripts/` — commands to rerun everything from scratch or step-by-step.
- `manifests/unified_metrics_summary.json` — machine-readable summary.

## Current status

### Attention heads

- all-head static atlas: **336 heads**.
- all-head runtime atlas: **336 heads**.
- score reconstruction median: **1.609e-07**.
- attention reconstruction median: **2.159e-07**.
- head output reconstruction median: **3.406e-07**.

This means the learned Qwen attention heads are represented as executable affine matrix-pseudocode:

```text
score_ij = c_delta + q_affine_delta(x_i) + k_affine_delta(x_j) + content_bilinear_delta(x_i, x_j)
A_i      = softmax(score_i)
payload_j = C_vo x_j + b_vo
Y_i      = sum_j A[i,j] * payload_j
```

### Term/control evidence

- removing `k_affine`: median Y error **1.4078**.
- removing `content_bilinear`: median Y error **0.6375**.
- removing `VO_bias`: median Y error **0.7643**.

### Token-specific directional control

- L3H6 `k_affine` sweep: rows **110**, sign match **0.9727**.
- L4H2 `content` sweep: rows **110**, sign match **1.0000**.

### Causal path recovery

- L3H6 -> L4H8 recovery mean: **1.0000**.
- L2H1 -> L3H6 recovery mean: **1.0000**.

### Baselines

- content-only QK baseline A_rel mean: **0.8812**.
- content-only QK baseline Y_rel mean: **13.1906**.

This confirms that a simple content-only attention explanation is much weaker than the full affine matrix-pseudocode decomposition.

## Main files to open

1. `final_product/final_pseudocode_product_v5_FULL/FINAL_MATRIX_PSEUDOCODE_REPORT.html`
2. `final_product/final_pseudocode_product_v5_FULL/FINAL_MATRIX_PSEUDOCODE_REPORT.md`
3. `docs/controls_v4_results_analysis_report.md`
4. `docs/atlas_full_analysis_report.md`
5. `manifests/unified_metrics_summary.json`

## How to rerun

Use scripts in `run_scripts/`:

- `01_run_full_atlas.sh` — full static/runtime/MLP atlas.
- `02_run_controls_v4.sh` — controls: token sweep, causal recovery, logit patch attribution, MLP operator.
- `03_run_final_product.sh` — rebuild final report from result zips.
- `RUN_EVERYTHING_FROM_SCRATCH.sh` — long full run.
- `REBUILD_FROM_EXISTING_RESULTS.sh` — only rebuild reports from existing zips.

## What is still optional / next-level

- richer MLP dictionary with neuron clusters and route labels;
- unified why-token HTML report that merges generation trace with patch-based logit attribution;
- external SAE / TransformerLens baseline;
- larger token-control sweep over more heads and prompts;
- distillation / transfer experiments.
