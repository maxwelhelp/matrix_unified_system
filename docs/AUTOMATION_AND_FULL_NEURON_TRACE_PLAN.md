# Automation and Full-Neuron Trace Plan

## Meaning of automation

The core proof is already done. Automation means the system should choose important heads, MLP groups, tokens, paths, and reports automatically.

## Full-neuron mode

Full-neuron tracing is useful, especially for domain models: DNA, physics, code, math, scientific transformers.

But it must be hierarchical:

1. Top-group mode: default, cheap, readable.
2. Full-neuron scan: computes all-neuron statistics but stores ranked summaries.
3. Full-neuron dump: expensive, only for selected layers/tokens/prompts.

Do not make raw every-neuron dumps the default. They are huge and hard to interpret. The useful outputs are rankings, clusters, patch effects, and stable domain groups.

## Transfer to another model

Direct neuron copy is usually unreliable because models have different bases, rotations, dimensions, and neuron permutations.

Better transfer path:

1. trace teacher model;
2. extract causal heads, MLP groups, paths, and token contributors;
3. align student features;
4. train adapter/LoRA/student using outputs plus causal targets;
5. validate behavior and causal similarity.

This can be more efficient than plain distillation for specific skills, because the student receives intermediate causal targets, not only final logits. It still needs benchmarks.

## Client architecture

Start with transformers. Later add adapters for CNNs, state-space models, Hyena-like models, graph nets, audio models.

Common adapter interface:

```python
collect_weights()
collect_activations()
run_patch()
build_component_programs()
write_report()
```

## Deployment model

The decoder stays private. Clients receive reports, JSON summaries, CSV tables, patch configs, and steering configs.

Recommended modes:

- on-prem runner;
- hosted API;
- trace collector;
- report-only consulting.

## Roadmap

A. Auto audit:

```bash
matrix-audit run --model MODEL --prompts prompts.json --out report/
```

B. Full-neuron scan:

```bash
matrix-audit neurons --mode scan --layers all --out neuron_report/
```

C. Steering pack:

```bash
matrix-audit steer --from prompts_a.json --to prompts_b.json --out steering_pack/
```

D. Model diff:

```bash
matrix-audit diff --base base_model --tuned tuned_model --prompts prompts.json --out diff_report/
```

## Next implementation steps

1. auto-head selection;
2. full-neuron scan summaries;
3. neuron clustering by output direction;
4. domain prompt packs: code, math, DNA, physics;
5. report/API schema;
6. one command for audit plus lightweight publish.
