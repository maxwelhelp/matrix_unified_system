# Why-token report v1
This report merges generation trace summaries with patch-based logit attribution. It is lightweight and can be committed to GitHub.
## Generated-token trace
| step | token | token_id | prob |
| --- | --- | --- | --- |
| 0 | ĠThe |  |  |
| 1 | Ġfunction |  |  |
| 2 | Ġshould |  |  |
| 3 | Ġtake |  |  |

## Patch-based positive causal contributors
| rank | component | layer | head | Δlogit | logit_rel | top1_match |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | mlp | 0 | -1 | 17.8828 | 1.1215 | 0.0000 |
| 2 | mlp | 23 | -1 | 6.9688 | 0.5380 | 0.8000 |
| 3 | mlp | 2 | -1 | 4.8750 | 0.9360 | 0.1000 |
| 4 | mlp | 1 | -1 | 3.8125 | 0.5063 | 0.4000 |
| 5 | mlp | 9 | -1 | 3.2656 | 0.2448 | 0.7000 |
| 6 | mlp | 4 | -1 | 3.1562 | 0.3468 | 0.7000 |
| 7 | head | 23 | 1 | 3.1094 | 0.2137 | 0.8000 |
| 8 | mlp | 5 | -1 | 2.6875 | 0.4308 | 0.7000 |
| 9 | mlp | 3 | -1 | 2.2188 | 0.6307 | 0.5000 |
| 10 | mlp | 13 | -1 | 1.9688 | 0.2308 | 0.7000 |
| 11 | mlp | 21 | -1 | 1.5625 | 0.4839 | 0.7000 |
| 12 | mlp | 7 | -1 | 1.4844 | 0.3040 | 0.7000 |
| 13 | mlp | 18 | -1 | 1.2188 | 0.2825 | 0.9000 |
| 14 | mlp | 19 | -1 | 1.0938 | 0.3421 | 0.7000 |
| 15 | mlp | 6 | -1 | 0.9844 | 0.3305 | 0.8000 |

## Patch-based negative/suppressing contributors
| rank | component | layer | head | Δlogit | logit_rel | top1_match |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | mlp | 14 | -1 | -0.7656 | 0.2308 | 0.8000 |
| 2 | mlp | 17 | -1 | -0.5312 | 0.2722 | 0.8000 |
| 3 | head | 4 | 8 | -0.3750 | 0.0511 | 0.9000 |
| 4 | head | 15 | 6 | -0.1406 | 0.0330 | 1.0000 |
| 5 | mlp | 16 | -1 | -0.0625 | 0.3310 | 0.9000 |
| 6 | head | 11 | 11 | 0.0000 | 0.0114 | 0.9000 |
| 7 | mlp | 8 | -1 | 0.0000 | 0.2449 | 0.8000 |
| 8 | head | 23 | 8 | 0.0156 | 0.0608 | 0.9000 |
| 9 | head | 3 | 6 | 0.0156 | 0.0175 | 1.0000 |
| 10 | head | 4 | 2 | 0.0156 | 0.0629 | 0.9000 |
| 11 | head | 16 | 1 | 0.0156 | 0.0042 | 1.0000 |
| 12 | head | 21 | 9 | 0.0312 | 0.1020 | 1.0000 |
| 13 | mlp | 22 | -1 | 0.0312 | 0.4033 | 0.6000 |
| 14 | head | 14 | 1 | 0.2656 | 0.0828 | 1.0000 |
| 15 | head | 23 | 4 | 0.3125 | 0.0870 | 1.0000 |

## Interpretation
```python
why_token = {
  'available_patch_rows': 35,
  'positive_mlp_count_top20': 19,
  'positive_head_count_top20': 1,
  'negative_mlp_count_top20': 10,
  'negative_head_count_top20': 10,
}
```
Use this report as the human-readable layer above raw patch tables. For stronger per-token explanations, run logit patch attribution for every generated token, not only the first/selected target.
