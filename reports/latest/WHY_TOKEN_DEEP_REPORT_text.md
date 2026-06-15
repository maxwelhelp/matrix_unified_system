# Deep why-token report v8

Real per-generated-token causal attribution: selected heads and MLP neuron groups are patched, then actual Δlogit/KL/top1 are measured for every generated token.

## Generated steps
| step | token | token_id | prob | base_logit |
| --- | --- | --- | --- | --- |
| 0 |  The | 576 | 0.7967 | 18.8750 |
| 1 |  sky | 12884 | 0.9343 | 21.1094 |
| 2 |  appears | 7952 | 0.9094 | 23.3750 |
| 3 |  blue | 6303 | 0.9892 | 25.9062 |

## Strong positive contributors
| rank | step | component | layer | head/group | token | Δlogit | KL | top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | mlp_group | 23 | top32_neurons |  sky | 5.1094 | 0.1720 | 1.0000 |
| 2 | 3 | mlp_group | 23 | top32_neurons |  blue | 3.4531 | 0.0144 | 1.0000 |
| 3 | 2 | mlp_group | 23 | top32_neurons |  appears | 3.3125 | 0.0285 | 1.0000 |
| 4 | 0 | mlp_group | 21 | top32_neurons |  The | 2.1562 | 0.2019 | 1.0000 |
| 5 | 3 | head | 23 | 1 |  blue | 1.0156 | 0.0046 | 1.0000 |
| 6 | 1 | mlp_group | 19 | top32_neurons |  sky | 0.9531 | 0.0176 | 1.0000 |
| 7 | 0 | mlp_group | 0 | top32_neurons |  The | 0.8594 | 0.1286 | 1.0000 |
| 8 | 2 | head | 23 | 1 |  appears | 0.8125 | 0.0234 | 1.0000 |
| 9 | 3 | mlp_group | 0 | top32_neurons |  blue | 0.7188 | 0.0022 | 1.0000 |
| 10 | 0 | mlp_group | 8 | top32_neurons |  The | 0.7031 | 0.0467 | 1.0000 |
| 11 | 0 | mlp_group | 7 | top32_neurons |  The | 0.6719 | 0.0760 | 1.0000 |
| 12 | 3 | mlp_group | 6 | top32_neurons |  blue | 0.6719 | 0.0073 | 1.0000 |
| 13 | 2 | mlp_group | 5 | top32_neurons |  appears | 0.5938 | 0.0283 | 1.0000 |
| 14 | 2 | mlp_group | 20 | top32_neurons |  appears | 0.5938 | 0.0449 | 1.0000 |
| 15 | 0 | mlp_group | 6 | top32_neurons |  The | 0.5625 | 0.0779 | 1.0000 |
| 16 | 3 | mlp_group | 13 | top32_neurons |  blue | 0.5625 | 0.0013 | 1.0000 |
| 17 | 1 | mlp_group | 22 | top32_neurons |  sky | 0.5469 | 0.0267 | 1.0000 |
| 18 | 2 | head | 21 | 9 |  appears | 0.5000 | 0.0036 | 1.0000 |
| 19 | 2 | mlp_group | 18 | top32_neurons |  appears | 0.4844 | 0.0348 | 1.0000 |
| 20 | 0 | mlp_group | 12 | top32_neurons |  The | 0.4688 | 0.0408 | 1.0000 |
| 21 | 0 | head | 23 | 1 |  The | 0.4531 | 0.0059 | 1.0000 |
| 22 | 1 | mlp_group | 5 | top32_neurons |  sky | 0.4531 | 0.0049 | 1.0000 |
| 23 | 1 | head | 23 | 1 |  sky | 0.4375 | 0.0050 | 1.0000 |
| 24 | 0 | mlp_group | 17 | top32_neurons |  The | 0.4219 | 0.0267 | 1.0000 |
| 25 | 2 | mlp_group | 12 | top32_neurons |  appears | 0.4062 | 0.0056 | 1.0000 |
| 26 | 0 | mlp_group | 22 | top32_neurons |  The | 0.3906 | 0.0548 | 1.0000 |
| 27 | 0 | head | 23 | 4 |  The | 0.3750 | 0.0123 | 1.0000 |
| 28 | 2 | mlp_group | 11 | top32_neurons |  appears | 0.3594 | 0.0032 | 1.0000 |
| 29 | 0 | head | 21 | 9 |  The | 0.3125 | 0.0049 | 1.0000 |
| 30 | 0 | mlp_group | 4 | top32_neurons |  The | 0.3125 | 0.0261 | 1.0000 |
| 31 | 2 | mlp_group | 7 | top32_neurons |  appears | 0.3125 | 0.0050 | 1.0000 |
| 32 | 0 | mlp_group | 11 | top32_neurons |  The | 0.2969 | 0.0222 | 1.0000 |
| 33 | 2 | mlp_group | 3 | top32_neurons |  appears | 0.2969 | 0.0064 | 1.0000 |
| 34 | 0 | mlp_group | 15 | top32_neurons |  The | 0.2812 | 0.0680 | 1.0000 |
| 35 | 1 | mlp_group | 0 | top32_neurons |  sky | 0.2812 | 0.0268 | 1.0000 |
| 36 | 1 | mlp_group | 12 | top32_neurons |  sky | 0.2812 | 0.0016 | 1.0000 |
| 37 | 1 | mlp_group | 20 | top32_neurons |  sky | 0.2656 | 0.0078 | 1.0000 |
| 38 | 0 | mlp_group | 16 | top32_neurons |  The | 0.2500 | 0.0308 | 1.0000 |
| 39 | 2 | mlp_group | 13 | top32_neurons |  appears | 0.2500 | 0.0058 | 1.0000 |
| 40 | 2 | mlp_group | 16 | top32_neurons |  appears | 0.2188 | 0.0084 | 1.0000 |

## Strong negative/suppressing contributors
| rank | step | component | layer | head/group | token | Δlogit | KL | top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | mlp_group | 16 | top32_neurons |  blue | -0.6562 | 0.0103 | 1.0000 |
| 2 | 1 | mlp_group | 17 | top32_neurons |  sky | -0.6250 | 0.0071 | 1.0000 |
| 3 | 2 | mlp_group | 19 | top32_neurons |  appears | -0.5938 | 0.0428 | 1.0000 |
| 4 | 1 | mlp_group | 21 | top32_neurons |  sky | -0.5312 | 0.0097 | 1.0000 |
| 5 | 3 | mlp_group | 21 | top32_neurons |  blue | -0.5312 | 0.0034 | 1.0000 |
| 6 | 2 | mlp_group | 21 | top32_neurons |  appears | -0.5000 | 0.0063 | 1.0000 |
| 7 | 1 | mlp_group | 15 | top32_neurons |  sky | -0.4844 | 0.0238 | 1.0000 |
| 8 | 1 | mlp_group | 8 | top32_neurons |  sky | -0.3750 | 0.0018 | 1.0000 |
| 9 | 1 | mlp_group | 16 | top32_neurons |  sky | -0.3750 | 0.0119 | 1.0000 |
| 10 | 3 | mlp_group | 18 | top32_neurons |  blue | -0.3750 | 0.0010 | 1.0000 |
| 11 | 3 | mlp_group | 17 | top32_neurons |  blue | -0.3438 | 4.776e-04 | 1.0000 |
| 12 | 0 | mlp_group | 20 | top32_neurons |  The | -0.2969 | 0.0273 | 1.0000 |
| 13 | 1 | mlp_group | 11 | top32_neurons |  sky | -0.2969 | 0.0019 | 1.0000 |
| 14 | 1 | mlp_group | 18 | top32_neurons |  sky | -0.2969 | 0.0035 | 1.0000 |
| 15 | 3 | mlp_group | 8 | top32_neurons |  blue | -0.2969 | 0.0028 | 1.0000 |
| 16 | 0 | mlp_group | 18 | top32_neurons |  The | -0.2812 | 0.0216 | 1.0000 |
| 17 | 2 | mlp_group | 9 | top32_neurons |  appears | -0.2812 | 0.0044 | 1.0000 |
| 18 | 2 | mlp_group | 4 | top32_neurons |  appears | -0.1875 | 0.0015 | 1.0000 |
| 19 | 3 | mlp_group | 19 | top32_neurons |  blue | -0.1875 | 0.0024 | 1.0000 |
| 20 | 0 | mlp_group | 2 | top32_neurons |  The | -0.1562 | 0.0045 | 1.0000 |
| 21 | 2 | mlp_group | 22 | top32_neurons |  appears | -0.1562 | 0.0138 | 1.0000 |
| 22 | 1 | mlp_group | 4 | top32_neurons |  sky | -0.1406 | 8.959e-04 | 1.0000 |
| 23 | 2 | mlp_group | 10 | top32_neurons |  appears | -0.1250 | 0.0011 | 1.0000 |
| 24 | 1 | mlp_group | 2 | top32_neurons |  sky | -0.1094 | 3.658e-04 | 1.0000 |
| 25 | 2 | mlp_group | 17 | top32_neurons |  appears | -0.1094 | 0.0040 | 1.0000 |
| 26 | 3 | head | 21 | 9 |  blue | -0.1094 | 2.745e-04 | 1.0000 |
| 27 | 0 | mlp_group | 1 | top32_neurons |  The | -0.0938 | 0.0036 | 1.0000 |
| 28 | 3 | mlp_group | 20 | top32_neurons |  blue | -0.0781 | 5.276e-04 | 1.0000 |
| 29 | 0 | mlp_group | 13 | top32_neurons |  The | -0.0625 | 0.0162 | 1.0000 |
| 30 | 3 | mlp_group | 4 | top32_neurons |  blue | -0.0625 | 0.0013 | 1.0000 |
| 31 | 0 | mlp_group | 19 | top32_neurons |  The | -0.0469 | 0.0393 | 1.0000 |
| 32 | 3 | mlp_group | 2 | top32_neurons |  blue | -0.0469 | 2.254e-04 | 1.0000 |
| 33 | 3 | mlp_group | 14 | top32_neurons |  blue | -0.0469 | 2.774e-04 | 1.0000 |
| 34 | 0 | head | 3 | 6 |  The | -0.0312 | 3.904e-04 | 1.0000 |
| 35 | 1 | head | 23 | 8 |  sky | -0.0312 | 8.261e-05 | 1.0000 |
| 36 | 1 | mlp_group | 13 | top32_neurons |  sky | -0.0312 | 0.0013 | 1.0000 |
| 37 | 0 | mlp_group | 23 | top32_neurons |  The | -0.0156 | 0.4823 | 1.0000 |
| 38 | 2 | mlp_group | 8 | top32_neurons |  appears | -0.0156 | 7.511e-04 | 1.0000 |
| 39 | 3 | head | 4 | 8 |  blue | -0.0156 | 4.338e-06 | 1.0000 |
| 40 | 0 | head | 4 | 8 |  The | 0.0000 | 1.345e-04 | 1.0000 |

## Step 0: ` The`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 21 | top32_neurons | 2.1562 | 0.2019 |
| mlp_group | 0 | top32_neurons | 0.8594 | 0.1286 |
| mlp_group | 8 | top32_neurons | 0.7031 | 0.0467 |
| mlp_group | 7 | top32_neurons | 0.6719 | 0.0760 |
| mlp_group | 6 | top32_neurons | 0.5625 | 0.0779 |
| mlp_group | 12 | top32_neurons | 0.4688 | 0.0408 |
| head | 23 | 1 | 0.4531 | 0.0059 |
| mlp_group | 17 | top32_neurons | 0.4219 | 0.0267 |
| mlp_group | 22 | top32_neurons | 0.3906 | 0.0548 |
| head | 23 | 4 | 0.3750 | 0.0123 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 20 | top32_neurons | -0.2969 | 0.0273 |
| mlp_group | 18 | top32_neurons | -0.2812 | 0.0216 |
| mlp_group | 2 | top32_neurons | -0.1562 | 0.0045 |
| mlp_group | 1 | top32_neurons | -0.0938 | 0.0036 |
| mlp_group | 13 | top32_neurons | -0.0625 | 0.0162 |
| mlp_group | 19 | top32_neurons | -0.0469 | 0.0393 |
| head | 3 | 6 | -0.0312 | 3.904e-04 |
| mlp_group | 23 | top32_neurons | -0.0156 | 0.4823 |
| head | 4 | 8 | 0.0000 | 1.345e-04 |
| head | 23 | 8 | 0.0000 | 4.178e-04 |

## Step 1: ` sky`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 5.1094 | 0.1720 |
| mlp_group | 19 | top32_neurons | 0.9531 | 0.0176 |
| mlp_group | 22 | top32_neurons | 0.5469 | 0.0267 |
| mlp_group | 5 | top32_neurons | 0.4531 | 0.0049 |
| head | 23 | 1 | 0.4375 | 0.0050 |
| mlp_group | 0 | top32_neurons | 0.2812 | 0.0268 |
| mlp_group | 12 | top32_neurons | 0.2812 | 0.0016 |
| mlp_group | 20 | top32_neurons | 0.2656 | 0.0078 |
| mlp_group | 6 | top32_neurons | 0.1719 | 0.0024 |
| mlp_group | 7 | top32_neurons | 0.1719 | 5.866e-04 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 17 | top32_neurons | -0.6250 | 0.0071 |
| mlp_group | 21 | top32_neurons | -0.5312 | 0.0097 |
| mlp_group | 15 | top32_neurons | -0.4844 | 0.0238 |
| mlp_group | 8 | top32_neurons | -0.3750 | 0.0018 |
| mlp_group | 16 | top32_neurons | -0.3750 | 0.0119 |
| mlp_group | 11 | top32_neurons | -0.2969 | 0.0019 |
| mlp_group | 18 | top32_neurons | -0.2969 | 0.0035 |
| mlp_group | 4 | top32_neurons | -0.1406 | 8.959e-04 |
| mlp_group | 2 | top32_neurons | -0.1094 | 3.658e-04 |
| head | 23 | 8 | -0.0312 | 8.261e-05 |

## Step 2: ` appears`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 3.3125 | 0.0285 |
| head | 23 | 1 | 0.8125 | 0.0234 |
| mlp_group | 5 | top32_neurons | 0.5938 | 0.0283 |
| mlp_group | 20 | top32_neurons | 0.5938 | 0.0449 |
| head | 21 | 9 | 0.5000 | 0.0036 |
| mlp_group | 18 | top32_neurons | 0.4844 | 0.0348 |
| mlp_group | 12 | top32_neurons | 0.4062 | 0.0056 |
| mlp_group | 11 | top32_neurons | 0.3594 | 0.0032 |
| mlp_group | 7 | top32_neurons | 0.3125 | 0.0050 |
| mlp_group | 3 | top32_neurons | 0.2969 | 0.0064 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 19 | top32_neurons | -0.5938 | 0.0428 |
| mlp_group | 21 | top32_neurons | -0.5000 | 0.0063 |
| mlp_group | 9 | top32_neurons | -0.2812 | 0.0044 |
| mlp_group | 4 | top32_neurons | -0.1875 | 0.0015 |
| mlp_group | 22 | top32_neurons | -0.1562 | 0.0138 |
| mlp_group | 10 | top32_neurons | -0.1250 | 0.0011 |
| mlp_group | 17 | top32_neurons | -0.1094 | 0.0040 |
| mlp_group | 8 | top32_neurons | -0.0156 | 7.511e-04 |
| head | 3 | 6 | 0.0000 | 1.003e-04 |
| head | 4 | 8 | 0.0000 | 4.679e-05 |

## Step 3: ` blue`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 3.4531 | 0.0144 |
| head | 23 | 1 | 1.0156 | 0.0046 |
| mlp_group | 0 | top32_neurons | 0.7188 | 0.0022 |
| mlp_group | 6 | top32_neurons | 0.6719 | 0.0073 |
| mlp_group | 13 | top32_neurons | 0.5625 | 0.0013 |
| mlp_group | 12 | top32_neurons | 0.2188 | 4.984e-04 |
| mlp_group | 11 | top32_neurons | 0.2031 | 1.877e-04 |
| mlp_group | 1 | top32_neurons | 0.1875 | 3.318e-04 |
| mlp_group | 3 | top32_neurons | 0.1875 | 6.507e-05 |
| mlp_group | 7 | top32_neurons | 0.1875 | 0.0017 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 16 | top32_neurons | -0.6562 | 0.0103 |
| mlp_group | 21 | top32_neurons | -0.5312 | 0.0034 |
| mlp_group | 18 | top32_neurons | -0.3750 | 0.0010 |
| mlp_group | 17 | top32_neurons | -0.3438 | 4.776e-04 |
| mlp_group | 8 | top32_neurons | -0.2969 | 0.0028 |
| mlp_group | 19 | top32_neurons | -0.1875 | 0.0024 |
| head | 21 | 9 | -0.1094 | 2.745e-04 |
| mlp_group | 20 | top32_neurons | -0.0781 | 5.276e-04 |
| mlp_group | 4 | top32_neurons | -0.0625 | 0.0013 |
| mlp_group | 2 | top32_neurons | -0.0469 | 2.254e-04 |
