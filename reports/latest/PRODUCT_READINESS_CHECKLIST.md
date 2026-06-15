# Product readiness checklist

| item | status | evidence | next_step |
| --- | --- | --- | --- |
| All-head static atlas | OK | rows=336, transitions=200 | run static all if missing |
| All-head runtime pseudocode | OK | rows=336, Y_rel_med=3.41e-07 | run runtime all |
| Term ablation evidence | OK | no_k_med=1.4078, no_content_med=0.6375, no_vo_bias_med=0.7643 |  |
| Token directional control | OK | rows=304, sign_match=0.9737 | run wider token sweep |
| Causal path influence | OK | rows=1, targetY_delta=0.0937 |  |
| Causal path recovery | OK | rows=33, targetY_recovery=1.0000 | run more path pairs |
| Patch-based logit attribution | OK | rows=65 | merge into generation trace |
| MLP operator/Jacobian | OK | rows=264, J_rank90_mean=351.4167 | add MLP operator dictionary |
| Baselines | OK | rows=128, contentQK_A_mean=0.8812 | add TransformerLens/SAE baseline optionally |
| Generation trace | OK | files=5 | merge signed attribution into why-token trace |
