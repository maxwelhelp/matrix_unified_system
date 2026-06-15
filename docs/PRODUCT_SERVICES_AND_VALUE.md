# Product Services and Value

Goal: keep the decoder private and sell outputs: reports, summaries, steering configs, patch configs, and validation evidence.

## What we can sell

1. Model Causal Trace Report
- why-token reports;
- top attention heads and MLP groups;
- positive and negative token contributors;
- delta-logit, KL, top1 patch evidence.

2. Fine-tune Diff Report
- compare base vs tuned model;
- show changed heads, MLP groups, paths, and token contributors.

3. Behavior Steering Pack
- find components that move behavior from mode A to mode B;
- examples: Python-style to JavaScript-style, short to detailed, direct answer to step-by-step;
- output: steering config plus validation table.

4. Compression / Replacement Candidates
- identify heads that can be approximated by matrix pseudocode;
- identify critical heads and MLP groups that should not be removed.

5. Domain Model Audit
- DNA, physics, code, math, scientific transformers;
- report feature/token contributors and useful neuron/head groups.

## Access models

A. On-prem closed runner: client runs our tool locally; raw weights stay with client; only reports/summaries leave.

B. Hosted API: client sends model or activation dumps; we return reports.

C. Bring-your-traces: client runs a small collector and sends limited summaries.

D. Consulting mode: client sends lightweight outputs; we interpret them.

## What stays private

- matrix decoder internals;
- basis search;
- operator mining;
- scoring heuristics;
- automatic patch selection logic.

## What client receives

- HTML/Markdown report;
- JSON summaries;
- CSV tables;
- patch configs;
- steering configs;
- confidence/risk table.

## Honest claim

We should say:

"We produce verified causal traces for selected prompts, tokens, attention heads, MLP groups, and paths."

Do not say:

"We completely read every thought of the model."

## Best first product

Model Causal Audit + Why-Token Report + Steering Candidate Pack.
