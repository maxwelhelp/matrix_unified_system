# Deep v8 final report

This is the final maximum validation layer: per-generated-token head patch + MLP neuron-group patch.

## python

rows=128 heads=8 mlp_layers=24

Generated steps:

| step | token | prob | base_logit |
| --- | --- | ---: | ---: |
| 0 | ` The` | 0.8347 | 22.7969 |
| 1 | ` function` | 0.7801 | 25.4844 |
| 2 | ` should` | 0.9790 | 27.0781 |
| 3 | ` take` | 0.4643 | 24.1406 |

Top positive contributors:

| component | layer | head/group | token | Δlogit | KL |
| --- | ---: | --- | --- | ---: | ---: |
| mlp_group | 23 | top32_neurons | ` take` | 6.2031 | 0.1568 |
| mlp_group | 23 | top32_neurons | ` The` | 5.5000 | 0.1515 |
| mlp_group | 23 | top32_neurons | ` should` | 5.0000 | 0.005432 |
| mlp_group | 0 | top32_neurons | ` The` | 4.5469 | 1.337 |
| mlp_group | 23 | top32_neurons | ` function` | 3.4688 | 0.2208 |
| head | 23 | 1 | ` The` | 3.1094 | 0.02707 |
| mlp_group | 0 | top32_neurons | ` function` | 2.8594 | 0.1556 |
| head | 23 | 1 | ` should` | 1.6562 | 0.01043 |
| mlp_group | 21 | top32_neurons | ` The` | 1.4062 | 0.1029 |
| mlp_group | 19 | top32_neurons | ` take` | 1.2969 | 0.374 |

Top negative contributors:

| component | layer | head/group | token | Δlogit | KL |
| --- | ---: | --- | --- | ---: | ---: |
| mlp_group | 21 | top32_neurons | ` should` | -0.9062 | 0.003547 |
| mlp_group | 20 | top32_neurons | ` function` | -0.5781 | 0.01033 |
| mlp_group | 14 | top32_neurons | ` The` | -0.5156 | 0.03017 |
| mlp_group | 16 | top32_neurons | ` The` | -0.5156 | 0.03604 |
| mlp_group | 21 | top32_neurons | ` take` | -0.5156 | 0.05853 |
| mlp_group | 10 | top32_neurons | ` The` | -0.4531 | 0.005341 |
| mlp_group | 9 | top32_neurons | ` take` | -0.4375 | 0.06776 |
| mlp_group | 11 | top32_neurons | ` function` | -0.3594 | 0.03106 |
| mlp_group | 17 | top32_neurons | ` The` | -0.2812 | 0.02852 |
| mlp_group | 20 | top32_neurons | ` should` | -0.2656 | 0.006099 |

## math

rows=128 heads=8 mlp_layers=24

Generated steps:

| step | token | prob | base_logit |
| --- | --- | ---: | ---: |
| 0 | ` To` | 0.4737 | 19.8750 |
| 1 | ` solve` | 0.9457 | 28.9219 |
| 2 | ` the` | 0.8199 | 27.1875 |
| 3 | ` equation` | 0.9984 | 29.6406 |

Top positive contributors:

| component | layer | head/group | token | Δlogit | KL |
| --- | ---: | --- | --- | ---: | ---: |
| mlp_group | 23 | top32_neurons | ` equation` | 7.8906 | 0.00319 |
| mlp_group | 0 | top32_neurons | ` equation` | 6.4688 | 0.1951 |
| mlp_group | 23 | top32_neurons | ` solve` | 6.1719 | 0.007048 |
| mlp_group | 0 | top32_neurons | ` solve` | 5.5938 | 0.03901 |
| mlp_group | 23 | top32_neurons | ` the` | 4.0312 | 0.1549 |
| mlp_group | 19 | top32_neurons | ` the` | 2.9844 | 0.214 |
| mlp_group | 23 | top32_neurons | ` To` | 1.9062 | 0.5179 |
| mlp_group | 11 | top32_neurons | ` solve` | 1.8594 | 0.004313 |
| mlp_group | 7 | top32_neurons | ` solve` | 1.8125 | 0.009548 |
| mlp_group | 21 | top32_neurons | ` To` | 1.5938 | 0.1263 |

Top negative contributors:

| component | layer | head/group | token | Δlogit | KL |
| --- | ---: | --- | --- | ---: | ---: |
| mlp_group | 22 | top32_neurons | ` the` | -1.5469 | 0.02074 |
| mlp_group | 21 | top32_neurons | ` solve` | -1.2500 | 0.07833 |
| mlp_group | 6 | top32_neurons | ` solve` | -0.5938 | 0.02807 |
| mlp_group | 20 | top32_neurons | ` the` | -0.5469 | 0.02052 |
| head | 23 | 4 | ` solve` | -0.4219 | 5.594e-05 |
| mlp_group | 15 | top32_neurons | ` solve` | -0.3750 | 0.0003955 |
| mlp_group | 22 | top32_neurons | ` To` | -0.3281 | 0.1932 |
| head | 21 | 9 | ` the` | -0.3281 | 0.002156 |
| mlp_group | 5 | top32_neurons | ` solve` | -0.2969 | 0.001336 |
| mlp_group | 22 | top32_neurons | ` equation` | -0.2812 | 0.0002313 |

## text

rows=128 heads=8 mlp_layers=24

Generated steps:

| step | token | prob | base_logit |
| --- | --- | ---: | ---: |
| 0 | ` The` | 0.7967 | 18.8750 |
| 1 | ` sky` | 0.9343 | 21.1094 |
| 2 | ` appears` | 0.9094 | 23.3750 |
| 3 | ` blue` | 0.9892 | 25.9062 |

Top positive contributors:

| component | layer | head/group | token | Δlogit | KL |
| --- | ---: | --- | --- | ---: | ---: |
| mlp_group | 23 | top32_neurons | ` sky` | 5.1094 | 0.172 |
| mlp_group | 23 | top32_neurons | ` blue` | 3.4531 | 0.01441 |
| mlp_group | 23 | top32_neurons | ` appears` | 3.3125 | 0.02846 |
| mlp_group | 21 | top32_neurons | ` The` | 2.1562 | 0.2019 |
| head | 23 | 1 | ` blue` | 1.0156 | 0.004613 |
| mlp_group | 19 | top32_neurons | ` sky` | 0.9531 | 0.01757 |
| mlp_group | 0 | top32_neurons | ` The` | 0.8594 | 0.1286 |
| head | 23 | 1 | ` appears` | 0.8125 | 0.02344 |
| mlp_group | 0 | top32_neurons | ` blue` | 0.7188 | 0.002237 |
| mlp_group | 8 | top32_neurons | ` The` | 0.7031 | 0.04666 |

Top negative contributors:

| component | layer | head/group | token | Δlogit | KL |
| --- | ---: | --- | --- | ---: | ---: |
| mlp_group | 16 | top32_neurons | ` blue` | -0.6562 | 0.0103 |
| mlp_group | 17 | top32_neurons | ` sky` | -0.6250 | 0.007058 |
| mlp_group | 19 | top32_neurons | ` appears` | -0.5938 | 0.04281 |
| mlp_group | 21 | top32_neurons | ` sky` | -0.5312 | 0.009726 |
| mlp_group | 21 | top32_neurons | ` blue` | -0.5312 | 0.003437 |
| mlp_group | 21 | top32_neurons | ` appears` | -0.5000 | 0.006257 |
| mlp_group | 15 | top32_neurons | ` sky` | -0.4844 | 0.02385 |
| mlp_group | 8 | top32_neurons | ` sky` | -0.3750 | 0.001766 |
| mlp_group | 16 | top32_neurons | ` sky` | -0.3750 | 0.01189 |
| mlp_group | 18 | top32_neurons | ` blue` | -0.3750 | 0.001046 |

