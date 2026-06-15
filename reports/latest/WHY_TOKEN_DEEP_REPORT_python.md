# Deep why-token report v8

Real per-generated-token causal attribution: selected heads and MLP neuron groups are patched, then actual Δlogit/KL/top1 are measured for every generated token.

## Generated steps
| step | token | token_id | prob | base_logit |
| --- | --- | --- | --- | --- |
| 0 |  The | 576 | 0.8347 | 22.7969 |
| 1 |  function | 729 | 0.7801 | 25.4844 |
| 2 |  should | 1265 | 0.9790 | 27.0781 |
| 3 |  take | 1896 | 0.4643 | 24.1406 |

## Strong positive contributors
| rank | step | component | layer | head/group | token | Δlogit | KL | top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | mlp_group | 23 | top32_neurons |  take | 6.2031 | 0.1568 | 1.0000 |
| 2 | 0 | mlp_group | 23 | top32_neurons |  The | 5.5000 | 0.1515 | 1.0000 |
| 3 | 2 | mlp_group | 23 | top32_neurons |  should | 5.0000 | 0.0054 | 1.0000 |
| 4 | 0 | mlp_group | 0 | top32_neurons |  The | 4.5469 | 1.3370 | 0.0000 |
| 5 | 1 | mlp_group | 23 | top32_neurons |  function | 3.4688 | 0.2208 | 1.0000 |
| 6 | 0 | head | 23 | 1 |  The | 3.1094 | 0.0271 | 1.0000 |
| 7 | 1 | mlp_group | 0 | top32_neurons |  function | 2.8594 | 0.1556 | 1.0000 |
| 8 | 2 | head | 23 | 1 |  should | 1.6562 | 0.0104 | 1.0000 |
| 9 | 0 | mlp_group | 21 | top32_neurons |  The | 1.4062 | 0.1029 | 1.0000 |
| 10 | 3 | mlp_group | 19 | top32_neurons |  take | 1.2969 | 0.3740 | 0.0000 |
| 11 | 1 | head | 23 | 1 |  function | 1.1094 | 0.0059 | 1.0000 |
| 12 | 3 | mlp_group | 20 | top32_neurons |  take | 1.0938 | 0.2461 | 1.0000 |
| 13 | 2 | mlp_group | 18 | top32_neurons |  should | 0.9062 | 0.0032 | 1.0000 |
| 14 | 0 | mlp_group | 4 | top32_neurons |  The | 0.8750 | 0.0563 | 1.0000 |
| 15 | 0 | mlp_group | 12 | top32_neurons |  The | 0.8750 | 0.0605 | 1.0000 |
| 16 | 2 | mlp_group | 15 | top32_neurons |  should | 0.8594 | 0.0186 | 1.0000 |
| 17 | 0 | mlp_group | 18 | top32_neurons |  The | 0.8125 | 0.0367 | 1.0000 |
| 18 | 2 | mlp_group | 19 | top32_neurons |  should | 0.7344 | 0.0159 | 1.0000 |
| 19 | 0 | mlp_group | 5 | top32_neurons |  The | 0.7188 | 0.0402 | 1.0000 |
| 20 | 3 | mlp_group | 18 | top32_neurons |  take | 0.6406 | 0.1122 | 1.0000 |
| 21 | 3 | mlp_group | 12 | top32_neurons |  take | 0.5938 | 0.0721 | 1.0000 |
| 22 | 0 | mlp_group | 7 | top32_neurons |  The | 0.5781 | 0.0167 | 1.0000 |
| 23 | 3 | head | 23 | 1 |  take | 0.5781 | 0.0306 | 1.0000 |
| 24 | 0 | mlp_group | 15 | top32_neurons |  The | 0.5625 | 0.0645 | 1.0000 |
| 25 | 2 | mlp_group | 12 | top32_neurons |  should | 0.5469 | 0.0068 | 1.0000 |
| 26 | 2 | mlp_group | 22 | top32_neurons |  should | 0.5469 | 0.0085 | 1.0000 |
| 27 | 2 | mlp_group | 13 | top32_neurons |  should | 0.4844 | 0.0022 | 1.0000 |
| 28 | 2 | mlp_group | 10 | top32_neurons |  should | 0.4688 | 0.0095 | 1.0000 |
| 29 | 3 | mlp_group | 0 | top32_neurons |  take | 0.4688 | 0.1516 | 1.0000 |
| 30 | 3 | mlp_group | 14 | top32_neurons |  take | 0.4688 | 0.0622 | 1.0000 |
| 31 | 3 | mlp_group | 22 | top32_neurons |  take | 0.4688 | 0.1052 | 1.0000 |
| 32 | 1 | mlp_group | 7 | top32_neurons |  function | 0.4219 | 0.0083 | 1.0000 |
| 33 | 2 | mlp_group | 9 | top32_neurons |  should | 0.4062 | 2.531e-04 | 1.0000 |
| 34 | 2 | mlp_group | 6 | top32_neurons |  should | 0.3906 | 0.0034 | 1.0000 |
| 35 | 0 | mlp_group | 19 | top32_neurons |  The | 0.3750 | 0.0597 | 1.0000 |
| 36 | 3 | mlp_group | 4 | top32_neurons |  take | 0.3750 | 0.0532 | 1.0000 |
| 37 | 1 | mlp_group | 12 | top32_neurons |  function | 0.3438 | 0.0138 | 1.0000 |
| 38 | 3 | mlp_group | 6 | top32_neurons |  take | 0.3438 | 0.0698 | 1.0000 |
| 39 | 0 | mlp_group | 3 | top32_neurons |  The | 0.3281 | 0.0039 | 1.0000 |
| 40 | 2 | mlp_group | 0 | top32_neurons |  should | 0.3281 | 0.0034 | 1.0000 |

## Strong negative/suppressing contributors
| rank | step | component | layer | head/group | token | Δlogit | KL | top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | mlp_group | 21 | top32_neurons |  should | -0.9062 | 0.0035 | 1.0000 |
| 2 | 1 | mlp_group | 20 | top32_neurons |  function | -0.5781 | 0.0103 | 1.0000 |
| 3 | 0 | mlp_group | 14 | top32_neurons |  The | -0.5156 | 0.0302 | 1.0000 |
| 4 | 0 | mlp_group | 16 | top32_neurons |  The | -0.5156 | 0.0360 | 1.0000 |
| 5 | 3 | mlp_group | 21 | top32_neurons |  take | -0.5156 | 0.0585 | 1.0000 |
| 6 | 0 | mlp_group | 10 | top32_neurons |  The | -0.4531 | 0.0053 | 1.0000 |
| 7 | 3 | mlp_group | 9 | top32_neurons |  take | -0.4375 | 0.0678 | 1.0000 |
| 8 | 1 | mlp_group | 11 | top32_neurons |  function | -0.3594 | 0.0311 | 1.0000 |
| 9 | 0 | mlp_group | 17 | top32_neurons |  The | -0.2812 | 0.0285 | 1.0000 |
| 10 | 2 | mlp_group | 20 | top32_neurons |  should | -0.2656 | 0.0061 | 1.0000 |
| 11 | 2 | mlp_group | 11 | top32_neurons |  should | -0.2500 | 0.0012 | 1.0000 |
| 12 | 1 | mlp_group | 22 | top32_neurons |  function | -0.2344 | 0.0112 | 1.0000 |
| 13 | 3 | mlp_group | 16 | top32_neurons |  take | -0.2188 | 0.0528 | 1.0000 |
| 14 | 0 | mlp_group | 1 | top32_neurons |  The | -0.2031 | 0.0067 | 1.0000 |
| 15 | 1 | mlp_group | 4 | top32_neurons |  function | -0.2031 | 0.0172 | 1.0000 |
| 16 | 3 | mlp_group | 8 | top32_neurons |  take | -0.1875 | 0.0590 | 1.0000 |
| 17 | 3 | mlp_group | 17 | top32_neurons |  take | -0.1875 | 0.1148 | 1.0000 |
| 18 | 1 | mlp_group | 5 | top32_neurons |  function | -0.1406 | 0.0153 | 1.0000 |
| 19 | 1 | mlp_group | 8 | top32_neurons |  function | -0.1250 | 0.0039 | 1.0000 |
| 20 | 2 | mlp_group | 5 | top32_neurons |  should | -0.1250 | 0.0011 | 1.0000 |
| 21 | 1 | mlp_group | 15 | top32_neurons |  function | -0.0938 | 0.0281 | 1.0000 |
| 22 | 2 | mlp_group | 7 | top32_neurons |  should | -0.0781 | 0.0019 | 1.0000 |
| 23 | 1 | mlp_group | 13 | top32_neurons |  function | -0.0625 | 0.0482 | 1.0000 |
| 24 | 0 | head | 4 | 8 |  The | -0.0469 | 5.080e-04 | 1.0000 |
| 25 | 1 | head | 23 | 4 |  function | -0.0469 | 0.0030 | 1.0000 |
| 26 | 1 | mlp_group | 17 | top32_neurons |  function | -0.0469 | 0.0080 | 1.0000 |
| 27 | 3 | head | 4 | 8 |  take | -0.0469 | 7.461e-04 | 1.0000 |
| 28 | 1 | mlp_group | 16 | top32_neurons |  function | -0.0312 | 0.0087 | 1.0000 |
| 29 | 2 | head | 23 | 8 |  should | -0.0312 | 1.022e-04 | 1.0000 |
| 30 | 3 | head | 23 | 8 |  take | -0.0312 | 8.348e-04 | 1.0000 |
| 31 | 0 | mlp_group | 9 | top32_neurons |  The | -0.0156 | 0.0066 | 1.0000 |
| 32 | 1 | mlp_group | 2 | top32_neurons |  function | -0.0156 | 4.576e-04 | 1.0000 |
| 33 | 0 | head | 11 | 11 |  The | 0.0000 | 1.785e-05 | 1.0000 |
| 34 | 0 | head | 16 | 1 |  The | 0.0000 | 7.948e-06 | 1.0000 |
| 35 | 0 | mlp_group | 22 | top32_neurons |  The | 0.0000 | 0.0183 | 1.0000 |
| 36 | 1 | head | 11 | 11 |  function | 0.0000 | 3.688e-06 | 1.0000 |
| 37 | 1 | head | 23 | 8 |  function | 0.0000 | 7.876e-04 | 1.0000 |
| 38 | 1 | mlp_group | 3 | top32_neurons |  function | 0.0000 | 0.0022 | 1.0000 |
| 39 | 3 | head | 16 | 1 |  take | 0.0000 | 4.978e-05 | 1.0000 |
| 40 | 0 | head | 3 | 6 |  The | 0.0156 | 8.158e-05 | 1.0000 |

## Step 0: ` The`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 5.5000 | 0.1515 |
| mlp_group | 0 | top32_neurons | 4.5469 | 1.3370 |
| head | 23 | 1 | 3.1094 | 0.0271 |
| mlp_group | 21 | top32_neurons | 1.4062 | 0.1029 |
| mlp_group | 4 | top32_neurons | 0.8750 | 0.0563 |
| mlp_group | 12 | top32_neurons | 0.8750 | 0.0605 |
| mlp_group | 18 | top32_neurons | 0.8125 | 0.0367 |
| mlp_group | 5 | top32_neurons | 0.7188 | 0.0402 |
| mlp_group | 7 | top32_neurons | 0.5781 | 0.0167 |
| mlp_group | 15 | top32_neurons | 0.5625 | 0.0645 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 14 | top32_neurons | -0.5156 | 0.0302 |
| mlp_group | 16 | top32_neurons | -0.5156 | 0.0360 |
| mlp_group | 10 | top32_neurons | -0.4531 | 0.0053 |
| mlp_group | 17 | top32_neurons | -0.2812 | 0.0285 |
| mlp_group | 1 | top32_neurons | -0.2031 | 0.0067 |
| head | 4 | 8 | -0.0469 | 5.080e-04 |
| mlp_group | 9 | top32_neurons | -0.0156 | 0.0066 |
| head | 11 | 11 | 0.0000 | 1.785e-05 |
| head | 16 | 1 | 0.0000 | 7.948e-06 |
| mlp_group | 22 | top32_neurons | 0.0000 | 0.0183 |

## Step 1: ` function`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 3.4688 | 0.2208 |
| mlp_group | 0 | top32_neurons | 2.8594 | 0.1556 |
| head | 23 | 1 | 1.1094 | 0.0059 |
| mlp_group | 7 | top32_neurons | 0.4219 | 0.0083 |
| mlp_group | 12 | top32_neurons | 0.3438 | 0.0138 |
| head | 21 | 9 | 0.2812 | 0.0044 |
| mlp_group | 18 | top32_neurons | 0.2344 | 0.0032 |
| mlp_group | 6 | top32_neurons | 0.2031 | 0.0046 |
| mlp_group | 19 | top32_neurons | 0.2031 | 0.0074 |
| mlp_group | 21 | top32_neurons | 0.1562 | 0.1150 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 20 | top32_neurons | -0.5781 | 0.0103 |
| mlp_group | 11 | top32_neurons | -0.3594 | 0.0311 |
| mlp_group | 22 | top32_neurons | -0.2344 | 0.0112 |
| mlp_group | 4 | top32_neurons | -0.2031 | 0.0172 |
| mlp_group | 5 | top32_neurons | -0.1406 | 0.0153 |
| mlp_group | 8 | top32_neurons | -0.1250 | 0.0039 |
| mlp_group | 15 | top32_neurons | -0.0938 | 0.0281 |
| mlp_group | 13 | top32_neurons | -0.0625 | 0.0482 |
| head | 23 | 4 | -0.0469 | 0.0030 |
| mlp_group | 17 | top32_neurons | -0.0469 | 0.0080 |

## Step 2: ` should`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 5.0000 | 0.0054 |
| head | 23 | 1 | 1.6562 | 0.0104 |
| mlp_group | 18 | top32_neurons | 0.9062 | 0.0032 |
| mlp_group | 15 | top32_neurons | 0.8594 | 0.0186 |
| mlp_group | 19 | top32_neurons | 0.7344 | 0.0159 |
| mlp_group | 12 | top32_neurons | 0.5469 | 0.0068 |
| mlp_group | 22 | top32_neurons | 0.5469 | 0.0085 |
| mlp_group | 13 | top32_neurons | 0.4844 | 0.0022 |
| mlp_group | 10 | top32_neurons | 0.4688 | 0.0095 |
| mlp_group | 9 | top32_neurons | 0.4062 | 2.531e-04 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 21 | top32_neurons | -0.9062 | 0.0035 |
| mlp_group | 20 | top32_neurons | -0.2656 | 0.0061 |
| mlp_group | 11 | top32_neurons | -0.2500 | 0.0012 |
| mlp_group | 5 | top32_neurons | -0.1250 | 0.0011 |
| mlp_group | 7 | top32_neurons | -0.0781 | 0.0019 |
| head | 23 | 8 | -0.0312 | 1.022e-04 |
| head | 3 | 6 | 0.0156 | 4.669e-06 |
| head | 11 | 11 | 0.0156 | 1.104e-06 |
| head | 16 | 1 | 0.0156 | 2.911e-06 |
| head | 4 | 8 | 0.0312 | 6.954e-06 |

## Step 3: ` take`
Positive:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 23 | top32_neurons | 6.2031 | 0.1568 |
| mlp_group | 19 | top32_neurons | 1.2969 | 0.3740 |
| mlp_group | 20 | top32_neurons | 1.0938 | 0.2461 |
| mlp_group | 18 | top32_neurons | 0.6406 | 0.1122 |
| mlp_group | 12 | top32_neurons | 0.5938 | 0.0721 |
| head | 23 | 1 | 0.5781 | 0.0306 |
| mlp_group | 0 | top32_neurons | 0.4688 | 0.1516 |
| mlp_group | 14 | top32_neurons | 0.4688 | 0.0622 |
| mlp_group | 22 | top32_neurons | 0.4688 | 0.1052 |
| mlp_group | 4 | top32_neurons | 0.3750 | 0.0532 |
Negative:
| component | layer | head/group | Δlogit | KL |
| --- | --- | --- | --- | --- |
| mlp_group | 21 | top32_neurons | -0.5156 | 0.0585 |
| mlp_group | 9 | top32_neurons | -0.4375 | 0.0678 |
| mlp_group | 16 | top32_neurons | -0.2188 | 0.0528 |
| mlp_group | 8 | top32_neurons | -0.1875 | 0.0590 |
| mlp_group | 17 | top32_neurons | -0.1875 | 0.1148 |
| head | 4 | 8 | -0.0469 | 7.461e-04 |
| head | 23 | 8 | -0.0312 | 8.348e-04 |
| head | 16 | 1 | 0.0000 | 4.978e-05 |
| head | 11 | 11 | 0.0156 | 8.904e-05 |
| mlp_group | 7 | top32_neurons | 0.0156 | 0.0225 |
