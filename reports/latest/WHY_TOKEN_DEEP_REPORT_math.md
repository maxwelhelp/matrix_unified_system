# Deep why-token report v8

Real per-generated-token causal attribution: selected heads and MLP neuron groups are patched, then actual Δlogit/KL/top1 are measured for every generated token.

## Generated steps
| step | token | token_id | prob | base_logit |
| --- | --- | --- | --- | --- |
| 0 |  To | 2014 | 0.4737 | 19.8750 |
| 1 |  solve | 11625 | 0.9457 | 28.9219 |
| 2 |  the | 279 | 0.8199 | 27.1875 |
| 3 |  equation | 23606 | 0.9984 | 29.6406 |

## Strong positive contributors
| rank | step | component | layer | head/group | token | Δlogit | KL | top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | mlp_group | 23 | top32_neurons |  equation | 7.8906 | 0.0032 | 1.0000 |
| 2 | 3 | mlp_group | 0 | top32_neurons |  equation | 6.4688 | 0.1951 | 1.0000 |
| 3 | 1 | mlp_group | 23 | top32_neurons |  solve | 6.1719 | 0.0070 | 1.0000 |
| 4 | 1 | mlp_group | 0 | top32_neurons |  solve | 5.5938 | 0.0390 | 1.0000 |
| 5 | 2 | mlp_group | 23 | top32_neurons |  the | 4.0312 | 0.1549 | 1.0000 |
| 6 | 2 | mlp_group | 19 | top32_neurons |  the | 2.9844 | 0.2140 | 1.0000 |
| 7 | 0 | mlp_group | 23 | top32_neurons |  To | 1.9062 | 0.5179 | 1.0000 |
| 8 | 1 | mlp_group | 11 | top32_neurons |  solve | 1.8594 | 0.0043 | 1.0000 |
| 9 | 1 | mlp_group | 7 | top32_neurons |  solve | 1.8125 | 0.0095 | 1.0000 |
| 10 | 0 | mlp_group | 21 | top32_neurons |  To | 1.5938 | 0.1263 | 1.0000 |
| 11 | 2 | head | 23 | 1 |  the | 1.5156 | 0.0158 | 1.0000 |
| 12 | 1 | mlp_group | 4 | top32_neurons |  solve | 1.3594 | 0.0056 | 1.0000 |
| 13 | 1 | mlp_group | 8 | top32_neurons |  solve | 1.3125 | 0.0146 | 1.0000 |
| 14 | 1 | mlp_group | 1 | top32_neurons |  solve | 1.2812 | 0.0338 | 1.0000 |
| 15 | 1 | mlp_group | 14 | top32_neurons |  solve | 1.2031 | 8.658e-04 | 1.0000 |
| 16 | 1 | mlp_group | 10 | top32_neurons |  solve | 1.1406 | 0.0309 | 1.0000 |
| 17 | 3 | mlp_group | 15 | top32_neurons |  equation | 1.1250 | 0.0030 | 1.0000 |
| 18 | 0 | mlp_group | 18 | top32_neurons |  To | 1.1094 | 0.2037 | 1.0000 |
| 19 | 0 | mlp_group | 17 | top32_neurons |  To | 1.0938 | 0.3591 | 1.0000 |
| 20 | 2 | mlp_group | 15 | top32_neurons |  the | 1.0938 | 0.0263 | 1.0000 |
| 21 | 1 | mlp_group | 12 | top32_neurons |  solve | 1.0625 | 0.0012 | 1.0000 |
| 22 | 3 | mlp_group | 16 | top32_neurons |  equation | 1.0000 | 0.0010 | 1.0000 |
| 23 | 0 | head | 23 | 1 |  To | 0.9844 | 0.0227 | 1.0000 |
| 24 | 2 | mlp_group | 17 | top32_neurons |  the | 0.8906 | 0.1063 | 1.0000 |
| 25 | 1 | mlp_group | 13 | top32_neurons |  solve | 0.8750 | 0.0040 | 1.0000 |
| 26 | 3 | mlp_group | 18 | top32_neurons |  equation | 0.8750 | 0.0020 | 1.0000 |
| 27 | 2 | mlp_group | 14 | top32_neurons |  the | 0.8438 | 0.0155 | 1.0000 |
| 28 | 3 | mlp_group | 21 | top32_neurons |  equation | 0.8438 | 0.0078 | 1.0000 |
| 29 | 0 | mlp_group | 14 | top32_neurons |  To | 0.8281 | 0.0533 | 1.0000 |
| 30 | 2 | mlp_group | 4 | top32_neurons |  the | 0.7188 | 0.0701 | 1.0000 |
| 31 | 1 | mlp_group | 16 | top32_neurons |  solve | 0.7031 | 0.0045 | 1.0000 |
| 32 | 0 | mlp_group | 7 | top32_neurons |  To | 0.6406 | 0.0556 | 1.0000 |
| 33 | 1 | mlp_group | 18 | top32_neurons |  solve | 0.6250 | 4.594e-04 | 1.0000 |
| 34 | 3 | mlp_group | 19 | top32_neurons |  equation | 0.6250 | 0.0056 | 1.0000 |
| 35 | 1 | mlp_group | 3 | top32_neurons |  solve | 0.6094 | 0.0101 | 1.0000 |
| 36 | 2 | mlp_group | 5 | top32_neurons |  the | 0.5938 | 0.0408 | 1.0000 |
| 37 | 0 | mlp_group | 20 | top32_neurons |  To | 0.5312 | 0.1215 | 1.0000 |
| 38 | 0 | mlp_group | 0 | top32_neurons |  To | 0.5156 | 0.1227 | 1.0000 |
| 39 | 0 | mlp_group | 11 | top32_neurons |  To | 0.5000 | 0.0225 | 1.0000 |
| 40 | 2 | mlp_group | 0 | top32_neurons |  the | 0.4688 | 0.0447 | 1.0000 |

## Strong negative/suppressing contributors
| rank | step | component | layer | head/group | token | Δlogit | KL | top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | mlp_group | 22 | top32_neurons |  the | -1.5469 | 0.0207 | 1.0000 |
| 2 | 1 | mlp_group | 21 | top32_neurons |  solve | -1.2500 | 0.0783 | 1.0000 |
| 3 | 1 | mlp_group | 6 | top32_neurons |  solve | -0.5938 | 0.0281 | 1.0000 |
| 4 | 2 | mlp_group | 20 | top32_neurons |  the | -0.5469 | 0.0205 | 1.0000 |
| 5 | 1 | head | 23 | 4 |  solve | -0.4219 | 5.594e-05 | 1.0000 |
| 6 | 1 | mlp_group | 15 | top32_neurons |  solve | -0.3750 | 3.955e-04 | 1.0000 |
| 7 | 0 | mlp_group | 22 | top32_neurons |  To | -0.3281 | 0.1932 | 1.0000 |
| 8 | 2 | head | 21 | 9 |  the | -0.3281 | 0.0022 | 1.0000 |
| 9 | 1 | mlp_group | 5 | top32_neurons |  solve | -0.2969 | 0.0013 | 1.0000 |
| 10 | 3 | mlp_group | 22 | top32_neurons |  equation | -0.2812 | 2.313e-04 | 1.0000 |
| 11 | 3 | mlp_group | 6 | top32_neurons |  equation | -0.2500 | 1.385e-04 | 1.0000 |
| 12 | 1 | mlp_group | 20 | top32_neurons |  solve | -0.2344 | 0.0332 | 1.0000 |
| 13 | 0 | mlp_group | 9 | top32_neurons |  To | -0.2031 | 0.0222 | 1.0000 |
| 14 | 3 | mlp_group | 2 | top32_neurons |  equation | -0.2031 | 2.442e-05 | 1.0000 |
| 15 | 1 | head | 4 | 8 |  solve | -0.1562 | 1.507e-04 | 1.0000 |
| 16 | 3 | mlp_group | 3 | top32_neurons |  equation | -0.1562 | 1.644e-05 | 1.0000 |
| 17 | 0 | mlp_group | 19 | top32_neurons |  To | -0.1406 | 0.0241 | 1.0000 |
| 18 | 1 | mlp_group | 17 | top32_neurons |  solve | -0.1406 | 0.0045 | 1.0000 |
| 19 | 2 | head | 23 | 4 |  the | -0.1250 | 1.171e-04 | 1.0000 |
| 20 | 2 | mlp_group | 16 | top32_neurons |  the | -0.1250 | 0.0134 | 1.0000 |
| 21 | 1 | mlp_group | 19 | top32_neurons |  solve | -0.1094 | 6.675e-04 | 1.0000 |
| 22 | 3 | mlp_group | 14 | top32_neurons |  equation | -0.1094 | 2.786e-05 | 1.0000 |
| 23 | 0 | mlp_group | 2 | top32_neurons |  To | -0.0781 | 0.0042 | 1.0000 |
| 24 | 2 | mlp_group | 3 | top32_neurons |  the | -0.0781 | 0.0016 | 1.0000 |
| 25 | 3 | head | 23 | 8 |  equation | -0.0781 | 8.895e-06 | 1.0000 |
| 26 | 3 | mlp_group | 1 | top32_neurons |  equation | -0.0781 | 5.114e-05 | 1.0000 |
| 27 | 3 | mlp_group | 20 | top32_neurons |  equation | -0.0781 | 6.838e-04 | 1.0000 |
| 28 | 3 | head | 23 | 4 |  equation | -0.0625 | 7.466e-06 | 1.0000 |
| 29 | 2 | mlp_group | 7 | top32_neurons |  the | -0.0469 | 0.0029 | 1.0000 |
| 30 | 3 | mlp_group | 8 | top32_neurons |  equation | -0.0469 | 1.376e-05 | 1.0000 |
| 31 | 1 | mlp_group | 2 | top32_neurons |  solve | -0.0312 | 0.0069 | 1.0000 |
| 32 | 2 | head | 3 | 6 |  the | -0.0312 | 9.371e-05 | 1.0000 |
| 33 | 2 | mlp_group | 12 | top32_neurons |  the | -0.0312 | 0.0064 | 1.0000 |
| 34 | 0 | head | 4 | 8 |  To | -0.0156 | 4.149e-04 | 1.0000 |
| 35 | 2 | head | 4 | 8 |  the | -0.0156 | 7.466e-05 | 1.0000 |
| 36 | 3 | head | 4 | 8 |  equation | -0.0156 | 3.075e-06 | 1.0000 |
| 37 | 0 | head | 3 | 6 |  To | 0.0000 | 4.652e-04 | 1.0000 |
| 38 | 0 | head | 11 | 11 |  To | 0.0000 | 2.629e-05 | 1.0000 |
| 39 | 0 | head | 16 | 1 |  To | 0.0000 | 4.906e-05 | 1.0000 |
| 40 | 1 | head | 3 | 6 |  solve | 0.0000 | 4.086e-04 | 1.0000 |

## Step 0: ` To`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 1.9062 | 0.5179 |
| mlp_group | 21 | top32_neurons | 1.5938 | 0.1263 |
| mlp_group | 18 | top32_neurons | 1.1094 | 0.2037 |
| mlp_group | 17 | top32_neurons | 1.0938 | 0.3591 |
| head | 23 | 1 | 0.9844 | 0.0227 |
| mlp_group | 14 | top32_neurons | 0.8281 | 0.0533 |
| mlp_group | 7 | top32_neurons | 0.6406 | 0.0556 |
| mlp_group | 20 | top32_neurons | 0.5312 | 0.1215 |
| mlp_group | 0 | top32_neurons | 0.5156 | 0.1227 |
| mlp_group | 11 | top32_neurons | 0.5000 | 0.0225 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 22 | top32_neurons | -0.3281 | 0.1932 |
| mlp_group | 9 | top32_neurons | -0.2031 | 0.0222 |
| mlp_group | 19 | top32_neurons | -0.1406 | 0.0241 |
| mlp_group | 2 | top32_neurons | -0.0781 | 0.0042 |
| head | 4 | 8 | -0.0156 | 4.149e-04 |
| head | 3 | 6 | 0.0000 | 4.652e-04 |
| head | 11 | 11 | 0.0000 | 2.629e-05 |
| head | 16 | 1 | 0.0000 | 4.906e-05 |
| mlp_group | 4 | top32_neurons | 0.0156 | 0.0058 |
| head | 23 | 8 | 0.0312 | 8.041e-04 |

## Step 1: ` solve`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 6.1719 | 0.0070 |
| mlp_group | 0 | top32_neurons | 5.5938 | 0.0390 |
| mlp_group | 11 | top32_neurons | 1.8594 | 0.0043 |
| mlp_group | 7 | top32_neurons | 1.8125 | 0.0095 |
| mlp_group | 4 | top32_neurons | 1.3594 | 0.0056 |
| mlp_group | 8 | top32_neurons | 1.3125 | 0.0146 |
| mlp_group | 1 | top32_neurons | 1.2812 | 0.0338 |
| mlp_group | 14 | top32_neurons | 1.2031 | 8.658e-04 |
| mlp_group | 10 | top32_neurons | 1.1406 | 0.0309 |
| mlp_group | 12 | top32_neurons | 1.0625 | 0.0012 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 21 | top32_neurons | -1.2500 | 0.0783 |
| mlp_group | 6 | top32_neurons | -0.5938 | 0.0281 |
| head | 23 | 4 | -0.4219 | 5.594e-05 |
| mlp_group | 15 | top32_neurons | -0.3750 | 3.955e-04 |
| mlp_group | 5 | top32_neurons | -0.2969 | 0.0013 |
| mlp_group | 20 | top32_neurons | -0.2344 | 0.0332 |
| head | 4 | 8 | -0.1562 | 1.507e-04 |
| mlp_group | 17 | top32_neurons | -0.1406 | 0.0045 |
| mlp_group | 19 | top32_neurons | -0.1094 | 6.675e-04 |
| mlp_group | 2 | top32_neurons | -0.0312 | 0.0069 |

## Step 2: ` the`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 4.0312 | 0.1549 |
| mlp_group | 19 | top32_neurons | 2.9844 | 0.2140 |
| head | 23 | 1 | 1.5156 | 0.0158 |
| mlp_group | 15 | top32_neurons | 1.0938 | 0.0263 |
| mlp_group | 17 | top32_neurons | 0.8906 | 0.1063 |
| mlp_group | 14 | top32_neurons | 0.8438 | 0.0155 |
| mlp_group | 4 | top32_neurons | 0.7188 | 0.0701 |
| mlp_group | 5 | top32_neurons | 0.5938 | 0.0408 |
| mlp_group | 0 | top32_neurons | 0.4688 | 0.0447 |
| mlp_group | 13 | top32_neurons | 0.4688 | 0.0015 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 22 | top32_neurons | -1.5469 | 0.0207 |
| mlp_group | 20 | top32_neurons | -0.5469 | 0.0205 |
| head | 21 | 9 | -0.3281 | 0.0022 |
| head | 23 | 4 | -0.1250 | 1.171e-04 |
| mlp_group | 16 | top32_neurons | -0.1250 | 0.0134 |
| mlp_group | 3 | top32_neurons | -0.0781 | 0.0016 |
| mlp_group | 7 | top32_neurons | -0.0469 | 0.0029 |
| head | 3 | 6 | -0.0312 | 9.371e-05 |
| mlp_group | 12 | top32_neurons | -0.0312 | 0.0064 |
| head | 4 | 8 | -0.0156 | 7.466e-05 |

## Step 3: ` equation`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 7.8906 | 0.0032 |
| mlp_group | 0 | top32_neurons | 6.4688 | 0.1951 |
| mlp_group | 15 | top32_neurons | 1.1250 | 0.0030 |
| mlp_group | 16 | top32_neurons | 1.0000 | 0.0010 |
| mlp_group | 18 | top32_neurons | 0.8750 | 0.0020 |
| mlp_group | 21 | top32_neurons | 0.8438 | 0.0078 |
| mlp_group | 19 | top32_neurons | 0.6250 | 0.0056 |
| head | 21 | 9 | 0.4062 | 1.366e-04 |
| mlp_group | 17 | top32_neurons | 0.3906 | 0.0027 |
| mlp_group | 9 | top32_neurons | 0.3438 | 1.813e-05 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 22 | top32_neurons | -0.2812 | 2.313e-04 |
| mlp_group | 6 | top32_neurons | -0.2500 | 1.385e-04 |
| mlp_group | 2 | top32_neurons | -0.2031 | 2.442e-05 |
| mlp_group | 3 | top32_neurons | -0.1562 | 1.644e-05 |
| mlp_group | 14 | top32_neurons | -0.1094 | 2.786e-05 |
| head | 23 | 8 | -0.0781 | 8.895e-06 |
| mlp_group | 1 | top32_neurons | -0.0781 | 5.114e-05 |
| mlp_group | 20 | top32_neurons | -0.0781 | 6.838e-04 |
| head | 23 | 4 | -0.0625 | 7.466e-06 |
| mlp_group | 8 | top32_neurons | -0.0469 | 1.376e-05 |
