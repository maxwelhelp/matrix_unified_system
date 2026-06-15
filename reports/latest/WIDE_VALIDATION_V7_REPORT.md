# Wide validation v7 report

Expanded validation: more token-control heads, more causal paths, more prompt types for logit attribution, wider MLP operator run.

| table | rows | means |
| --- | ---: | --- |
| `v7_logit_attr_math.csv` | 35 | causal_logit_contribution_mean=1.087 |
| `v7_logit_attr_python.csv` | 35 | causal_logit_contribution_mean=1.652 |
| `v7_logit_attr_text.csv` | 35 | causal_logit_contribution_mean=1.118 |
| `v7_mlp_operator_wide.csv` | 384 | J_rank90_mean=351.9, J_rank95_mean=447.3, J_rank99_mean=616.7 |
| `v7_recovery_L15H5_to_L16H3.csv` | 16 | target_Y_recovery_ratio_mean=1, logit_recovery_ratio_mean=1 |
| `v7_recovery_L2H1_to_L3H6.csv` | 16 | target_Y_recovery_ratio_mean=1, logit_recovery_ratio_mean=1 |
| `v7_recovery_L3H6_to_L4H8.csv` | 16 | target_Y_recovery_ratio_mean=1, logit_recovery_ratio_mean=1 |
| `v7_token_sweep_L11H11_vobias.csv` | 110 |  |
| `v7_token_sweep_L16H1_k.csv` | 110 |  |
| `v7_token_sweep_L3H6_k.csv` | 110 |  |
| `v7_token_sweep_L4H2_content.csv` | 110 |  |
