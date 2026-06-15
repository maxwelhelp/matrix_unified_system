# MLP operator dictionary v2
This is a richer lightweight MLP dictionary built from local operator/Jacobian summaries. It is still not as semantically rich as attention pseudocode, but it gives layer roles, gate sensitivity, rank profile, and top-neuron groups.
## Role definitions
```python
strong_residual_writer      = layer has high MLP output norm
high_jacobian_transformer   = local MLP Jacobian has high norm
gate_sensitive              = removing/altering gate strongly changes output
compact_high_effect_operator= relatively compact rank with high effect
mixed_mlp_operator          = no single role dominates
```

## Layer role table
| layer | roles | out | J | rank90 | rank99 | no_gate | top_neurons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | mixed_mlp_operator | 4.2288 | 4.7238 | 364.5000 | 633.1250 | n/a | [3048, 50, 4786, 2546, 3486, 392, 4852, 308] |
| 1 | mixed_mlp_operator | 3.1648 | 4.6419 | 385.6250 | 646.8750 | n/a | [3516, 671, 3052, 1619, 2874, 4431, 3599, 4227] |
| 2 | mixed_mlp_operator | 2.6629 | 5.0566 | 378.6250 | 639.5000 | n/a | [264, 2586, 4395, 488, 2701, 3370, 4418, 1218] |
| 3 | mixed_mlp_operator | 3.6713 | 6.0337 | 382.2500 | 641.1250 | n/a | [2673, 674, 4101, 3477, 2430, 995, 4728, 1213] |
| 4 | mixed_mlp_operator | 4.9363 | 6.8270 | 390.7500 | 650.0000 | n/a | [2973, 880, 1872, 2225, 3033, 3988, 2914, 1433] |
| 5 | mixed_mlp_operator | 5.4968 | 5.9165 | 384.1250 | 637.6250 | n/a | [79, 419, 2706, 4091, 3475, 2584, 1618, 4791] |
| 6 | mixed_mlp_operator | 5.5472 | 7.3428 | 390.5000 | 643.5000 | n/a | [2043, 1590, 3308, 3844, 2765, 3258, 3702, 4085] |
| 7 | mixed_mlp_operator | 5.6547 | 7.4454 | 409.6250 | 658.0000 | n/a | [1731, 186, 1605, 281, 3285, 3022, 4745, 1305] |
| 8 | mixed_mlp_operator | 4.8357 | 7.3551 | 372.3750 | 618.1250 | n/a | [3546, 7, 3848, 1508, 3763, 2325, 3482, 1413] |
| 9 | mixed_mlp_operator | 5.0942 | 7.2650 | 393.0000 | 636.8750 | n/a | [3046, 2277, 1186, 1432, 247, 3355, 2850, 4692] |
| 10 | mixed_mlp_operator | 4.8324 | 6.8692 | 353.2500 | 607.3750 | n/a | [540, 1827, 3978, 3897, 4290, 4004, 3238, 1824] |
| 11 | mixed_mlp_operator | 4.7202 | 7.2491 | 341.3750 | 606.0000 | n/a | [3660, 3900, 1089, 83, 1911, 2965, 2542, 2594] |
| 12 | mixed_mlp_operator | 4.8911 | 7.2030 | 354.5000 | 612.0000 | n/a | [1311, 765, 1941, 2402, 186, 3367, 4362, 23] |
| 13 | mixed_mlp_operator | 5.6214 | 7.6193 | 361.7500 | 617.6250 | n/a | [3162, 682, 2219, 2235, 3190, 1633, 2990, 4535] |
| 14 | mixed_mlp_operator | 6.0109 | 8.4697 | 352.8750 | 612.5000 | n/a | [999, 3573, 4188, 2451, 2333, 3226, 402, 4541] |
| 15 | mixed_mlp_operator | 7.8232 | 9.8126 | 362.5000 | 612.6250 | n/a | [4088, 4692, 4323, 3506, 2329, 519, 3417, 1024] |
| 16 | mixed_mlp_operator | 9.7039 | 10.3424 | 353.6250 | 621.0000 | n/a | [1561, 1929, 3524, 1873, 4750, 3700, 4113, 897] |
| 17 | mixed_mlp_operator | 9.7568 | 9.0213 | 341.0000 | 611.8750 | n/a | [2888, 779, 3687, 408, 2688, 2156, 196, 3579] |
| 18 | compact_high_effect_operator | 9.6255 | 9.7457 | 335.8750 | 604.7500 | n/a | [2668, 4212, 3231, 2929, 309, 4817, 130, 2371] |
| 19 | compact_high_effect_operator | 12.4402 | 11.5793 | 334.5000 | 605.2500 | n/a | [63, 1361, 4794, 2738, 822, 3172, 151, 48] |
| 20 | compact_high_effect_operator | 17.1065 | 12.5856 | 314.1250 | 590.5000 | n/a | [93, 4024, 1942, 3722, 718, 3366, 4575, 637] |
| 21 | strong_residual_writer,compact_high_effect_operator | 18.3154 | 10.8409 | 312.8750 | 591.2500 | n/a | [4663, 2930, 426, 3936, 2940, 3658, 1507, 2171] |
| 22 | high_jacobian_transformer,compact_high_effect_operator | 16.9081 | 14.5266 | 290.6250 | 580.8750 | n/a | [1222, 82, 3358, 1069, 1904, 3035, 1439, 1082] |
| 23 | strong_residual_writer,high_jacobian_transformer,compact_high_effect_operator | 60.1509 | 18.5643 | 172.1250 | 513.3750 | n/a | [2539, 4144, 4861, 2607, 4391, 2788, 537, 3383] |

## Strongest MLP candidates

### Strong residual writers
| layer | roles | out_norm_mean | top_neurons |
| --- | --- | --- | --- |
| 23 | strong_residual_writer,high_jacobian_transformer,compact_high_effect_operator | 60.1509 | [2539, 4144, 4861, 2607, 4391, 2788, 537, 3383, 1863, 4749] |
| 21 | strong_residual_writer,compact_high_effect_operator | 18.3154 | [4663, 2930, 426, 3936, 2940, 3658, 1507, 2171, 2612, 2164] |
| 20 | compact_high_effect_operator | 17.1065 | [93, 4024, 1942, 3722, 718, 3366, 4575, 637, 2858, 408] |
| 22 | high_jacobian_transformer,compact_high_effect_operator | 16.9081 | [1222, 82, 3358, 1069, 1904, 3035, 1439, 1082, 2763, 2507] |
| 19 | compact_high_effect_operator | 12.4402 | [63, 1361, 4794, 2738, 822, 3172, 151, 48, 2580, 476] |
| 17 | mixed_mlp_operator | 9.7568 | [2888, 779, 3687, 408, 2688, 2156, 196, 3579, 4756, 4787] |
| 16 | mixed_mlp_operator | 9.7039 | [1561, 1929, 3524, 1873, 4750, 3700, 4113, 897, 2725, 585] |
| 18 | compact_high_effect_operator | 9.6255 | [2668, 4212, 3231, 2929, 309, 4817, 130, 2371, 1260, 1718] |
| 15 | mixed_mlp_operator | 7.8232 | [4088, 4692, 4323, 3506, 2329, 519, 3417, 1024, 3111, 1249] |
| 14 | mixed_mlp_operator | 6.0109 | [999, 3573, 4188, 2451, 2333, 3226, 402, 4541, 2470, 3618] |

### High-Jacobian transformers
| layer | roles | J_norm_mean | top_neurons |
| --- | --- | --- | --- |
| 23 | strong_residual_writer,high_jacobian_transformer,compact_high_effect_operator | 18.5643 | [2539, 4144, 4861, 2607, 4391, 2788, 537, 3383, 1863, 4749] |
| 22 | high_jacobian_transformer,compact_high_effect_operator | 14.5266 | [1222, 82, 3358, 1069, 1904, 3035, 1439, 1082, 2763, 2507] |
| 20 | compact_high_effect_operator | 12.5856 | [93, 4024, 1942, 3722, 718, 3366, 4575, 637, 2858, 408] |
| 19 | compact_high_effect_operator | 11.5793 | [63, 1361, 4794, 2738, 822, 3172, 151, 48, 2580, 476] |
| 21 | strong_residual_writer,compact_high_effect_operator | 10.8409 | [4663, 2930, 426, 3936, 2940, 3658, 1507, 2171, 2612, 2164] |
| 16 | mixed_mlp_operator | 10.3424 | [1561, 1929, 3524, 1873, 4750, 3700, 4113, 897, 2725, 585] |
| 15 | mixed_mlp_operator | 9.8126 | [4088, 4692, 4323, 3506, 2329, 519, 3417, 1024, 3111, 1249] |
| 18 | compact_high_effect_operator | 9.7457 | [2668, 4212, 3231, 2929, 309, 4817, 130, 2371, 1260, 1718] |
| 17 | mixed_mlp_operator | 9.0213 | [2888, 779, 3687, 408, 2688, 2156, 196, 3579, 4756, 4787] |
| 14 | mixed_mlp_operator | 8.4697 | [999, 3573, 4188, 2451, 2333, 3226, 402, 4541, 2470, 3618] |

### Gate-sensitive layers
| layer | roles | no_gate_delta_rel_mean | top_neurons |
| --- | --- | --- | --- |
| 0 | mixed_mlp_operator | n/a | [3048, 50, 4786, 2546, 3486, 392, 4852, 308, 2456, 4341] |
| 1 | mixed_mlp_operator | n/a | [3516, 671, 3052, 1619, 2874, 4431, 3599, 4227, 3675, 3331] |
| 2 | mixed_mlp_operator | n/a | [264, 2586, 4395, 488, 2701, 3370, 4418, 1218, 886, 3162] |
| 3 | mixed_mlp_operator | n/a | [2673, 674, 4101, 3477, 2430, 995, 4728, 1213, 2325, 2493] |
| 4 | mixed_mlp_operator | n/a | [2973, 880, 1872, 2225, 3033, 3988, 2914, 1433, 3364, 4646] |
| 5 | mixed_mlp_operator | n/a | [79, 419, 2706, 4091, 3475, 2584, 1618, 4791, 2575, 1042] |
| 6 | mixed_mlp_operator | n/a | [2043, 1590, 3308, 3844, 2765, 3258, 3702, 4085, 1332, 3352] |
| 7 | mixed_mlp_operator | n/a | [1731, 186, 1605, 281, 3285, 3022, 4745, 1305, 3045, 2005] |
| 8 | mixed_mlp_operator | n/a | [3546, 7, 3848, 1508, 3763, 2325, 3482, 1413, 3416, 3225] |
| 9 | mixed_mlp_operator | n/a | [3046, 2277, 1186, 1432, 247, 3355, 2850, 4692, 1734, 99] |

## Next MLP upgrade
```python
1. cluster top neurons by output direction W_down[:, neuron]
2. patch neuron groups and measure Δlogit / KL / top1
3. label groups as code-token writers, syntax suppressors, BOS/template writers, etc.
4. connect MLP groups to attention pseudocode paths in why-token report
```
