# Enhanced product summary
This file is generated from lightweight GitHub-published summaries.
## Status
```python
checklist_has_na = False
why_token_rows = 35
generation_steps_found = 4
mlp_layers = 24
mlp_role_counts = {'mixed_mlp_operator': 18, 'compact_high_effect_operator': 6, 'strong_residual_writer': 2, 'high_jacobian_transformer': 2}
```
## Generated files
- `reports/latest/WHY_TOKEN_REPORT.md`
- `reports/latest/MLP_OPERATOR_DICTIONARY_v2.md`
- `reports/latest/tables/why_token_top_contributors.csv`
- `reports/latest/tables/mlp_layer_roles.csv`
- `manifests/latest/enhanced_summary.json`
## Next experiments
1. run patch-based logit attribution for every generated token, not only the first target
2. run token-control sweep on 20-50 heads across code/math/text prompts
3. cluster MLP top neurons by W_down output direction and patch neuron groups
4. connect causal path recovery pairs to why-token reports
