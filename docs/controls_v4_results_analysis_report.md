# Controls v4 Results Analysis
## Status
All v4 runs finished without Traceback. The zip contains outputs for all requested control modes: all_v4 compact, token sweeps, causal path recovery, logit patch attribution, and MLP operator.
## Key results
### Token-control sweep
#### L3H6 k_affine
- rows: 110
- sign_match_rate: 0.9727
- actual_mass_delta: mean=0.0105745, median=0.012975, p90=0.134072, min=-0.230495, max=0.269062
- head_A_rel: mean=0.162758, median=0.116492, p90=0.346126, min=0.000192118, max=0.545388
- head_Y_delta_rel: mean=0.325124, median=0.228093, p90=0.771557, min=0.000274717, max=1.30766
- logit_rel: mean=0.00582873, median=0.00408131, p90=0.0139042, min=0, max=0.03176
- kl_orig_to_patch: mean=0.00161674, median=0.00050287, p90=0.0039439, min=-3.5468e-07, max=0.0189922
- top1_match: mean=0.998104, median=1, p90=1, min=0.909091, max=1
- loss_delta: mean=-0.000573684, median=0, p90=0.00400369, min=-0.0195255, max=0.00864744
#### L4H2 content
- rows: 110
- sign_match_rate: 1.0000
- actual_mass_delta: mean=-0.0241223, median=-0.0389472, p90=0.0752209, min=-0.09142, max=0.146042
- head_A_rel: mean=0.258683, median=0.258119, p90=0.377264, min=0.0766385, max=0.459703
- head_Y_delta_rel: mean=0.243156, median=0.233994, p90=0.397392, min=0.0484585, max=0.565463
- logit_rel: mean=0.0145106, median=0.0103128, p90=0.0300434, min=0, max=0.097462
- kl_orig_to_patch: mean=0.00885501, median=0.00246184, p90=0.0255988, min=-3.5468e-07, max=0.129057
- top1_match: mean=0.993938, median=1, p90=1, min=0.888889, max=1
- loss_delta: mean=-0.00286578, median=0, p90=0.00957955, min=-0.243164, max=0.0312314

### Causal path recovery
#### L3H6 -> L4H8
- ablate_head_target_Y_rel: mean=0.0724842, median=0.0722929, max=0.111706
- replace_full_target_Y_rel: mean=0, median=0, max=0
- term_patch_target_Y_rel: mean=0.140725, median=0.130358, max=0.289434
- ablate_head_logit_rel: mean=0.0171439, median=0.0162975, max=0.0233857
- replace_full_logit_rel: mean=0, median=0, max=0
- term_patch_logit_rel: mean=0.0299604, median=0.0284892, max=0.0529524
- target_Y_recovery_ratio: mean=1, median=1, max=1
- logit_recovery_ratio: mean=1, median=1, max=1
#### L2H1 -> L3H6
- ablate_head_target_Y_rel: mean=0.423834, median=0.395382, max=0.804825
- replace_full_target_Y_rel: mean=0, median=0, max=0
- term_patch_target_Y_rel: mean=0.251062, median=0.246536, max=0.458399
- ablate_head_logit_rel: mean=0.13645, median=0.126268, max=0.231707
- replace_full_logit_rel: mean=0, median=0, max=0
- term_patch_logit_rel: mean=0.0264534, median=0.0250505, max=0.0406668
- target_Y_recovery_ratio: mean=1, median=1, max=1
- logit_recovery_ratio: mean=1, median=1, max=1

### Logit patch attribution
Top positive causal logit contributions:

| component   |   layer |   head |   target_token_id | target_token   | patch_type   |   causal_logit_contribution |   logit_rel |   loss_delta_ablate |   top1_match |
|:------------|--------:|-------:|------------------:|:---------------|:-------------|----------------------------:|------------:|--------------------:|-------------:|
| mlp         |       0 |     -1 |               576 | ĠThe           | zero_mlp     |                    17.8828  |    1.12147  |           8.61374   |          0   |
| mlp         |      23 |     -1 |               576 | ĠThe           | zero_mlp     |                     6.96875 |    0.538047 |           0.969555  |          0.8 |
| mlp         |       2 |     -1 |               576 | ĠThe           | zero_mlp     |                     4.875   |    0.936027 |           5.96654   |          0.1 |
| mlp         |       1 |     -1 |               576 | ĠThe           | zero_mlp     |                     3.8125  |    0.506258 |           0.11234   |          0.4 |
| mlp         |       9 |     -1 |               576 | ĠThe           | zero_mlp     |                     3.26562 |    0.244816 |           0.0179591 |          0.7 |
| mlp         |       4 |     -1 |               576 | ĠThe           | zero_mlp     |                     3.15625 |    0.346759 |           0.0689201 |          0.7 |
| head        |      23 |      1 |               576 | ĠThe           | ablate       |                     3.10938 |    0.213691 |           0.410236  |          0.8 |
| mlp         |       5 |     -1 |               576 | ĠThe           | zero_mlp     |                     2.6875  |    0.43083  |          -0.393301  |          0.7 |
| mlp         |       3 |     -1 |               576 | ĠThe           | zero_mlp     |                     2.21875 |    0.6307   |           1.64407   |          0.5 |
| mlp         |      13 |     -1 |               576 | ĠThe           | zero_mlp     |                     1.96875 |    0.230833 |          -0.147346  |          0.7 |

Top negative causal logit contributions:

| component   |   layer |   head |   target_token_id | target_token   | patch_type   |   causal_logit_contribution |   logit_rel |   loss_delta_ablate |   top1_match |
|:------------|--------:|-------:|------------------:|:---------------|:-------------|----------------------------:|------------:|--------------------:|-------------:|
| mlp         |      14 |     -1 |               576 | ĠThe           | zero_mlp     |                   -0.765625 |  0.230798   |         0.0380151   |          0.8 |
| mlp         |      17 |     -1 |               576 | ĠThe           | zero_mlp     |                   -0.53125  |  0.272218   |        -0.0869851   |          0.8 |
| head        |       4 |      8 |               576 | ĠThe           | ablate       |                   -0.375    |  0.0510877  |        -0.00300884  |          0.9 |
| head        |      15 |      6 |               576 | ĠThe           | ablate       |                   -0.140625 |  0.0330183  |        -0.0182221   |          1   |
| mlp         |      16 |     -1 |               576 | ĠThe           | zero_mlp     |                   -0.0625   |  0.331012   |         0.389436    |          0.9 |
| head        |      11 |     11 |               576 | ĠThe           | ablate       |                    0        |  0.0113879  |         0.000948191 |          0.9 |
| mlp         |       8 |     -1 |               576 | ĠThe           | zero_mlp     |                    0        |  0.244866   |        -0.390676    |          0.8 |
| head        |       4 |      2 |               576 | ĠThe           | ablate       |                    0.015625 |  0.0629404  |        -0.0869877   |          0.9 |
| head        |       3 |      6 |               576 | ĠThe           | ablate       |                    0.015625 |  0.0174941  |        -0.00913119  |          1   |
| head        |      16 |      1 |               576 | ĠThe           | ablate       |                    0.015625 |  0.00417466 |         0.00603223  |          1   |

### MLP operator
- records: 192
- J_norm: mean=8.62653, median=7.51118, p90=13.1643, min=3.35587, max=19.3637
- J_rank90: mean=351.349, median=357.5, p90=393, min=160, max=424
- J_rank95: mean=446.828, median=450, p90=486, min=273, max=516
- J_rank99: mean=616.323, median=616, p90=646.9, min=505, max=667
- hidden_norm: mean=15.5902, median=9.71498, p90=28.3078, min=3.23318, max=94.9188
- out_norm: mean=9.71664, median=5.62333, p90=17.4209, min=1.83305, max=68.5769

Top layers by mean out_norm:

|   layer |   J_norm |   J_rank90 |   J_rank95 |   J_rank99 |   out_norm |   hidden_norm |
|--------:|---------:|-----------:|-----------:|-----------:|-----------:|--------------:|
|      23 | 18.5643  |    172.125 |    285.25  |    513.375 |   60.1509  |       84.7282 |
|      21 | 10.8409  |    312.875 |    411.5   |    591.25  |   18.3154  |       26.9049 |
|      20 | 12.5856  |    314.125 |    411     |    590.5   |   17.1065  |       27.5481 |
|      22 | 14.5266  |    290.625 |    392     |    580.875 |   16.9081  |       29.8556 |
|      19 | 11.5793  |    334.5   |    430.25  |    605.25  |   12.4402  |       21.7117 |
|      17 |  9.02128 |    341     |    437.875 |    611.875 |    9.75678 |       16.8073 |
|      16 | 10.3424  |    353.625 |    449.5   |    621     |    9.70389 |       16.1065 |
|      18 |  9.74571 |    335.875 |    431.125 |    604.75  |    9.62553 |       15.8432 |
|      15 |  9.81261 |    362.5   |    452.875 |    612.625 |    7.82324 |       13.7738 |
|      14 |  8.46972 |    352.875 |    445.625 |    612.5   |    6.01091 |       10.5236 |