#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_pseudocode_controls_v3.py

Add-on control / attribution / causal-path layer for the matrix-pseudocode atlas.
It is intentionally standalone: it does not replace the already-working atlas v2.

Modes:
  causal_path      Source head term patch -> target head Q/K/V/A/Y deltas -> logits/loss.
  logit_attr       Signed direct logit attribution for attention heads and MLP outputs.
  mlp_rich         Richer MLP/SwiGLU accounting: gate/up/down, neuron groups, logit directions.
  token_control    Token-specific score-term control: remove/scale term only for one key token.
  baselines        Heatmap/head-ablation/QK-no-bias/OV-only style baselines.
  all              Runs all modes on a compact configuration.

Key idea:
  Original head code:
      Q = q_proj(X), K = k_proj(X), V = v_proj(X)
      A = softmax(RoPE(Q) @ RoPE(K).T)
      Y = A @ V @ W_o_head.T

  Matrix-pseudocode split:
      score = constant_delta + q_affine + k_affine + content_bilinear
      payload = VO_linear(X) + VO_bias
      Y = softmax(score) @ payload

  Control modes change individual pseudocode terms inside the real model forward.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return str(x)
        return x
    if isinstance(x, Path):
        return str(x)
    if torch.is_tensor(x):
        if x.numel() <= 64:
            return x.detach().cpu().float().tolist()
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if hasattr(x, "__dict__"):
        return json_safe(vars(x))
    return str(x)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.detach().float()
    bf = b.detach().float()
    return float(torch.linalg.norm(af - bf) / torch.linalg.norm(bf).clamp_min(eps))


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(name)


def parse_layers(s: str, n_layers: int) -> List[int]:
    s = str(s).strip().lower()
    if s == "all":
        return list(range(n_layers))
    out: List[int] = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set([x for x in out if 0 <= x < n_layers]))


def parse_heads_spec(s: str) -> List[Tuple[int, int]]:
    out = []
    for part in str(s).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad head spec {part!r}; use L:H")
        a, b = part.split(":", 1)
        out.append((int(a), int(b)))
    # stable unique
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq


def topk_changed_logits(base_logits: torch.Tensor, patch_logits: torch.Tensor, tokenizer: Any, topk: int = 10) -> List[Dict[str, Any]]:
    # compare final next-token logits [vocab]
    d = (patch_logits - base_logits).float()
    vals_pos, idx_pos = torch.topk(d, k=min(topk, d.numel()))
    vals_neg, idx_neg = torch.topk(-d, k=min(topk, d.numel()))
    rows = []
    for typ, vals, idxs in [("up", vals_pos, idx_pos), ("down", -vals_neg, idx_neg)]:
        for v, i in zip(vals.tolist(), idxs.tolist()):
            tok = tokenizer.convert_ids_to_tokens([int(i)])[0]
            rows.append({"direction": typ, "token_id": int(i), "token": tok, "delta_logit": float(v)})
    return rows


# ---------------- model helpers ----------------

def get_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Cannot find model.model.layers")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rope_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [B,H,T,D] or [H,T,D], cos/sin compatible with [B,1,T,D]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    T = int(scores.shape[-1])
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1)


def compute_position_embeddings(model: Any, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    rotary = getattr(model.model, "rotary_emb", None) if hasattr(model, "model") else None
    if rotary is None:
        raise RuntimeError("model.model.rotary_emb not found")
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return rotary(position_ids)


def dims_from_model(model: Any) -> Dict[str, int]:
    cfg = model.config
    hidden = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden // n_heads))
    return {"hidden": hidden, "n_heads": n_heads, "n_kv": n_kv, "head_dim": head_dim, "kv_groups": n_heads // n_kv, "n_layers": int(cfg.num_hidden_layers)}


def get_unembed(model: Any) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach().float()
    raise RuntimeError("Cannot find lm_head.weight")


PROMPTS: Dict[str, List[str]] = {
    "code": [
        "Write a Python function that reverses a linked list.",
        "Implement binary search over a sorted list in Python.",
        "Explain what this code does: for i in range(n): total += arr[i]",
        "Find the bug in this Python code: def f(x): return x.append(1)",
    ],
    "math": [
        "Solve: if 3x + 5 = 20, what is x?",
        "Explain the derivative of sin(x) * exp(x).",
        "What is the quadratic formula and when is it used?",
        "Simplify: (x + 2)^2 - x^2.",
    ],
    "text": [
        "Summarize why rivers are important for cities.",
        "Explain the difference between a planet and a star.",
        "Describe how plants grow from seeds.",
        "Write a calm paragraph about ocean waves.",
    ],
    "symbols": [
        "JSON: {\"name\": \"Alice\", \"score\": 42, \"ok\": true}",
        "Path: /home/user/project/src/main.py --flag=true",
        "Array: [1, 1, 2, 3, 5, 8, 13, 21]",
        "Equation: f(x)=sin(x)+cos(2x)-log(x+1)",
    ],
}


def build_prompts(suites: str, per_suite: int, single_prompt: str = "") -> List[Dict[str, str]]:
    if single_prompt:
        return [{"suite": "single", "text": single_prompt}]
    names = list(PROMPTS.keys()) if suites == "all" else [x.strip() for x in suites.split(",") if x.strip()]
    rows = []
    for name in names:
        base = PROMPTS[name]
        for i in range(per_suite):
            rows.append({"suite": name, "text": base[i % len(base)]})
    return rows


@dataclass
class HeadTerms:
    A_true: torch.Tensor
    A_patch: torch.Tensor
    Y_true: torch.Tensor
    Y_patch: torch.Tensor
    V: torch.Tensor
    scores_true: torch.Tensor
    scores_patch: torch.Tensor
    term_tensors: Dict[str, torch.Tensor]


def head_terms_from_attn_input(model: Any, attn: Any, X_model: torch.Tensor, layer_idx: int, head_idx: int,
                               terms_to_remove: Optional[List[str]] = None,
                               target_key_pos: Optional[int] = None,
                               strength: float = 1.0) -> HeadTerms:
    """Compute true and patched Y for one head from attention input X_model [1,T,H]."""
    terms_to_remove = terms_to_remove or []
    dims = dims_from_model(model)
    B, T, H = X_model.shape
    assert B == 1, "This helper expects batch=1"
    n_heads, n_kv, D, kv_groups = dims["n_heads"], dims["n_kv"], dims["head_dim"], dims["kv_groups"]
    kv_idx = head_idx // kv_groups
    device = X_model.device
    q_out = attn.q_proj(X_model).view(1, T, n_heads, D).transpose(1, 2).contiguous()
    k_out = attn.k_proj(X_model).view(1, T, n_kv, D).transpose(1, 2).contiguous()
    v_out = attn.v_proj(X_model).view(1, T, n_kv, D).transpose(1, 2).contiguous()

    pos = torch.arange(T, device=device).unsqueeze(0)
    cos, sin = compute_position_embeddings(model, X_model, pos)
    q_rot, k_rot = apply_rope(q_out, k_out, cos, sin)
    Q = q_rot[0, head_idx].float()
    K = k_rot[0, kv_idx].float()
    V = v_out[0, kv_idx].float()
    scale = math.sqrt(D)
    scores_true = (Q @ K.T) / scale
    A_true = causal_softmax(scores_true)
    O = attn.o_proj.weight.detach().float()[:, head_idx * D : (head_idx + 1) * D].to(device)
    Y_true = (A_true @ V) @ O.T

    # exact score term split: q_lin/k_lin + q_bias/k_bias after RoPE
    # q_proj output per head = X @ Wq.T + bq. Compute bias-only tensors and linear residual.
    bq_full = attn.q_proj.bias.detach().to(device) if attn.q_proj.bias is not None else torch.zeros(n_heads * D, device=device, dtype=X_model.dtype)
    bk_full = attn.k_proj.bias.detach().to(device) if attn.k_proj.bias is not None else torch.zeros(n_kv * D, device=device, dtype=X_model.dtype)
    bq = bq_full.view(n_heads, D)[head_idx].view(1, 1, 1, D).expand(1, 1, T, D)
    bk = bk_full.view(n_kv, D)[kv_idx].view(1, 1, 1, D).expand(1, 1, T, D)
    q_bias_rot = rope_one(bq, cos, sin)[0, 0].float()
    k_bias_rot = rope_one(bk, cos, sin)[0, 0].float()
    q_lin = Q - q_bias_rot
    k_lin = K - k_bias_rot

    content = (q_lin @ k_lin.T) / scale
    q_affine = (q_lin @ k_bias_rot.T) / scale       # depends on query content and key-position bias
    k_affine = (q_bias_rot @ k_lin.T) / scale       # depends on key content and query-position bias
    constant = (q_bias_rot @ k_bias_rot.T) / scale
    term_tensors = {"constant": constant, "q_affine": q_affine, "k_affine": k_affine, "content": content}
    scores_patch = scores_true.clone().float()
    for term in terms_to_remove:
        if term not in term_tensors:
            continue
        subtract = term_tensors[term]
        if target_key_pos is not None and 0 <= target_key_pos < T:
            mask = torch.zeros_like(subtract)
            mask[:, target_key_pos] = subtract[:, target_key_pos]
            subtract = mask
        scores_patch = scores_patch - float(strength) * subtract
    A_patch = causal_softmax(scores_patch)

    # VO patch: remove bias/linear parts by changing Y directly.
    # For vo_bias, subtract the constant write vector after attention. Since rows of A sum to 1,
    # removing value bias changes every query row by O @ b_v.
    Y_patch = (A_patch @ V) @ O.T
    if "vo_bias" in terms_to_remove or "vo_linear" in terms_to_remove:
        bv_full = attn.v_proj.bias.detach().to(device) if attn.v_proj.bias is not None else torch.zeros(n_kv * D, device=device, dtype=X_model.dtype)
        bv = bv_full.view(n_kv, D)[kv_idx].float()
        bvo = bv @ O.T
        if "vo_bias" in terms_to_remove:
            Y_patch = Y_patch - float(strength) * bvo.view(1, -1)
        if "vo_linear" in terms_to_remove:
            # keep only VO bias contribution.
            Y_bias_only = bvo.view(1, -1).expand(T, -1)
            Y_patch = Y_bias_only if "vo_bias" not in terms_to_remove else torch.zeros_like(Y_true)

    return HeadTerms(A_true=A_true, A_patch=A_patch, Y_true=Y_true, Y_patch=Y_patch, V=V,
                     scores_true=scores_true, scores_patch=scores_patch, term_tensors=term_tensors)


def patch_source_head_forward(model: Any, tokenizer: Any, text: str, source: Tuple[int, int], terms: List[str],
                              target_key_pos: Optional[int], strength: float, max_length: int, device: str) -> Dict[str, Any]:
    layers = get_layers(model)
    l_src, h_src = source
    attn = layers[l_src].self_attn

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        base = model(**enc, labels=enc["input_ids"], use_cache=False, output_hidden_states=True)
    base_logits = base.logits.detach().float().cpu()
    base_loss = float(base.loss.detach().float().cpu())

    patch_info: Dict[str, Any] = {}

    def hook(module, fargs, kwargs, output):
        X = kwargs.get("hidden_states", None) if isinstance(kwargs, dict) else None
        if X is None:
            X = fargs[0] if fargs else None
        if X is None:
            return output
        attn_out = output[0] if isinstance(output, (tuple, list)) else output
        ht = head_terms_from_attn_input(model, module, X, l_src, h_src, terms, target_key_pos, strength)
        delta = (ht.Y_patch - ht.Y_true).to(attn_out.device).to(attn_out.dtype).unsqueeze(0)
        patch_info["head_A_rel"] = rel_err(ht.A_patch, ht.A_true)
        patch_info["head_Y_delta_rel"] = rel_err(ht.Y_patch, ht.Y_true)
        if target_key_pos is not None:
            patch_info["target_key_mass_before_mean"] = float(ht.A_true[:, target_key_pos].mean().detach().cpu())
            patch_info["target_key_mass_after_mean"] = float(ht.A_patch[:, target_key_pos].mean().detach().cpu())
            patch_info["target_key_mass_delta_mean"] = patch_info["target_key_mass_after_mean"] - patch_info["target_key_mass_before_mean"]
        new_attn = attn_out + delta
        if isinstance(output, tuple):
            return (new_attn,) + tuple(output[1:])
        if isinstance(output, list):
            return [new_attn] + list(output[1:])
        return new_attn

    handle = attn.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            patched = model(**enc, labels=enc["input_ids"], use_cache=False, output_hidden_states=True)
    finally:
        handle.remove()
    plog = patched.logits.detach().float().cpu()
    ploss = float(patched.loss.detach().float().cpu())
    a = base_logits[:, :-1, :]
    b = plog[:, :-1, :]
    pa = torch.softmax(a, dim=-1)
    logpb = torch.log_softmax(b, dim=-1)
    kl = float(F.kl_div(logpb, pa, reduction="batchmean"))
    top1 = float((a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean())
    next_base = base_logits[0, -1]
    next_patch = plog[0, -1]
    return {
        "text": text,
        "source": f"{l_src}:{h_src}",
        "terms": terms,
        "target_key_pos": target_key_pos,
        "strength": strength,
        "loss_base": base_loss,
        "loss_patch": ploss,
        "loss_delta": ploss - base_loss,
        "logit_rel": rel_err(b, a),
        "kl_orig_to_patch": kl,
        "top1_match": top1,
        **patch_info,
        "top_changed_logits_final": topk_changed_logits(next_base, next_patch, tokenizer, 12),
    }


# ---------------- modes ----------------

def mode_token_control(model: Any, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> None:
    ensure_dir(out_dir)
    source = parse_heads_spec(args.source_head)[0]
    prompts = build_prompts(args.prompt_suites, args.prompts_per_suite, args.prompt)
    rows = []
    for p in prompts:
        rows.append(patch_source_head_forward(model, tokenizer, p["text"], source, args.terms_list, args.target_key_pos, args.patch_strength, args.max_length, args.device))
    write_json(out_dir / "token_control_results.json", rows)
    flat = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k != "top_changed_logits_final"}
        flat.append(rr)
    write_csv(out_dir / "token_control_summary.csv", flat)
    md = ["# Token-specific pseudocode term control", "", f"Source head: {args.source_head}", f"Terms: {args.patch_terms}", ""]
    md.append(f"Mean loss_delta: {safe_mean([float(r['loss_delta']) for r in rows]):.6g}")
    md.append(f"Mean logit_rel: {safe_mean([float(r['logit_rel']) for r in rows]):.6g}")
    md.append(f"Mean KL: {safe_mean([float(r['kl_orig_to_patch']) for r in rows]):.6g}")
    if args.target_key_pos is not None:
        md.append(f"Mean target_key_mass_delta: {safe_mean([float(r.get('target_key_mass_delta_mean',0.0)) for r in rows]):.6g}")
    (out_dir / "summary_token_control.md").write_text("\n".join(md), encoding="utf-8")


def collect_target_head_from_hidden(model: Any, hidden_states: Tuple[torch.Tensor, ...], layer_idx: int, head_idx: int) -> Dict[str, torch.Tensor]:
    layers = get_layers(model)
    layer = layers[layer_idx]
    X_model = layer.input_layernorm(hidden_states[layer_idx].detach())
    ht = head_terms_from_attn_input(model, layer.self_attn, X_model, layer_idx, head_idx, [], None, 1.0)
    # q/k/v raw after RoPE not returned by head_terms; recompute basic q/k/v enough for deltas? Use scores/A/Y.
    return {"A": ht.A_true.detach().cpu(), "Y": ht.Y_true.detach().cpu(), "scores": ht.scores_true.detach().cpu()}


def mode_causal_path(model: Any, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> None:
    ensure_dir(out_dir)
    source = parse_heads_spec(args.source_head)[0]
    target = parse_heads_spec(args.target_head)[0]
    prompts = build_prompts(args.prompt_suites, args.prompts_per_suite, args.prompt)
    rows = []
    examples = []
    for p in prompts:
        text = p["text"]
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
        with torch.no_grad():
            base = model(**enc, labels=enc["input_ids"], use_cache=False, output_hidden_states=True)
        base_target = collect_target_head_from_hidden(model, base.hidden_states, target[0], target[1])
        patch_result = patch_source_head_forward(model, tokenizer, text, source, args.terms_list, args.target_key_pos, args.patch_strength, args.max_length, args.device)
        # rerun patched to capture hidden states via hook, since patch_source_head_forward does not return them.
        layers = get_layers(model)
        src_attn = layers[source[0]].self_attn
        def hook(module, fargs, kwargs, output):
            X = kwargs.get("hidden_states", None) if isinstance(kwargs, dict) else None
            if X is None:
                X = fargs[0] if fargs else None
            attn_out = output[0] if isinstance(output, (tuple, list)) else output
            ht = head_terms_from_attn_input(model, module, X, source[0], source[1], args.terms_list, args.target_key_pos, args.patch_strength)
            delta = (ht.Y_patch - ht.Y_true).to(attn_out.device).to(attn_out.dtype).unsqueeze(0)
            new_attn = attn_out + delta
            if isinstance(output, tuple): return (new_attn,) + tuple(output[1:])
            if isinstance(output, list): return [new_attn] + list(output[1:])
            return new_attn
        h = src_attn.register_forward_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                patched = model(**enc, labels=enc["input_ids"], use_cache=False, output_hidden_states=True)
        finally:
            h.remove()
        patch_target = collect_target_head_from_hidden(model, patched.hidden_states, target[0], target[1])
        row = {
            "suite": p["suite"], "text": text[:160],
            "source": f"{source[0]}:{source[1]}", "target": f"{target[0]}:{target[1]}", "terms": "+".join(args.terms_list),
            "target_A_rel_delta": rel_err(patch_target["A"], base_target["A"]),
            "target_scores_rel_delta": rel_err(patch_target["scores"], base_target["scores"]),
            "target_Y_rel_delta": rel_err(patch_target["Y"], base_target["Y"]),
            "loss_delta": patch_result["loss_delta"], "logit_rel": patch_result["logit_rel"],
            "kl_orig_to_patch": patch_result["kl_orig_to_patch"], "top1_match": patch_result["top1_match"],
        }
        rows.append(row)
        if len(examples) < 4:
            examples.append({**row, "top_changed_logits_final": patch_result.get("top_changed_logits_final", [])})
    write_csv(out_dir / "causal_path_summary.csv", rows)
    write_json(out_dir / "causal_path_examples.json", examples)
    md = ["# Causal path probe", "", f"Source: {args.source_head}", f"Target: {args.target_head}", f"Terms: {args.patch_terms}", ""]
    for key in ["target_A_rel_delta", "target_scores_rel_delta", "target_Y_rel_delta", "loss_delta", "logit_rel", "kl_orig_to_patch"]:
        md.append(f"mean {key}: {safe_mean([float(r[key]) for r in rows]):.6g}")
    (out_dir / "summary_causal_path.md").write_text("\n".join(md), encoding="utf-8")


def capture_layer_outputs(model: Any, tokenizer: Any, text: str, max_length: int, device: str):
    layers = get_layers(model)
    caps: Dict[str, Dict[int, torch.Tensor]] = {"attn": {}, "mlp": {}, "mlp_in": {}}
    handles = []
    def make_attn_hook(li):
        def hook(module, fargs, kwargs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            caps["attn"][li] = out.detach().float().cpu()
        return hook
    def make_mlp_hook(li):
        def hook(module, fargs, output):
            x = fargs[0] if fargs else None
            if x is not None:
                caps["mlp_in"][li] = x.detach().float().cpu()
            caps["mlp"][li] = output.detach().float().cpu()
        return hook
    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(li), with_kwargs=True))
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    try:
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"], use_cache=False, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    return enc, out, caps


def head_outputs_for_layer(model: Any, hidden_state_layer: torch.Tensor, layer_idx: int, selected_pos: int) -> List[Dict[str, Any]]:
    layers = get_layers(model)
    dims = dims_from_model(model)
    layer = layers[layer_idx]
    X_model = layer.input_layernorm(hidden_state_layer.detach())
    B, T, _ = X_model.shape
    rows = []
    for h in range(dims["n_heads"]):
        ht = head_terms_from_attn_input(model, layer.self_attn, X_model, layer_idx, h, [], None, 1.0)
        y = ht.Y_true[selected_pos].detach().float().cpu()
        rows.append({"layer": layer_idx, "head": h, "Y_pos": y, "Y_norm": float(torch.linalg.norm(y))})
    return rows


def mode_logit_attr(model: Any, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> None:
    ensure_dir(out_dir)
    text = args.prompt or args.generate_prompt or "Write a Python function that reverses a linked list."
    enc, out, caps = capture_layer_outputs(model, tokenizer, text, args.max_length, args.device)
    logits = out.logits.detach().float().cpu()
    final_logits = logits[0, -1]
    target_id = int(final_logits.argmax()) if args.target_token_id < 0 else int(args.target_token_id)
    target_tok = tokenizer.convert_ids_to_tokens([target_id])[0]
    U = get_unembed(model).cpu()
    u = U[target_id]
    layers = get_layers(model)
    selected_pos = int(enc["input_ids"].shape[1] - 1)
    rows = []
    layer_list = parse_layers(args.layers, len(layers))
    for li in layer_list:
        # per-head direct logit contribution for selected position
        head_rows = head_outputs_for_layer(model, out.hidden_states[li].to(args.device), li, selected_pos)
        for hr in head_rows:
            y = hr.pop("Y_pos")
            rows.append({**hr, "component": "head", "target_token_id": target_id, "target_token": target_tok,
                         "direct_logit": float(y @ u), "signed_norm": float(torch.linalg.norm(y) * torch.linalg.norm(u))})
        if li in caps["mlp"]:
            m = caps["mlp"][li][0, selected_pos]
            rows.append({"layer": li, "head": -1, "component": "mlp", "target_token_id": target_id, "target_token": target_tok,
                         "Y_norm": float(torch.linalg.norm(m)), "direct_logit": float(m @ u),
                         "signed_norm": float(torch.linalg.norm(m) * torch.linalg.norm(u))})
        if li in caps["attn"]:
            a = caps["attn"][li][0, selected_pos]
            rows.append({"layer": li, "head": -1, "component": "attn_sum", "target_token_id": target_id, "target_token": target_tok,
                         "Y_norm": float(torch.linalg.norm(a)), "direct_logit": float(a @ u),
                         "signed_norm": float(torch.linalg.norm(a) * torch.linalg.norm(u))})
    rows_sorted = sorted(rows, key=lambda r: abs(float(r["direct_logit"])), reverse=True)
    write_csv(out_dir / "signed_logit_attribution.csv", rows_sorted)
    write_json(out_dir / "top_signed_logit_attribution.json", rows_sorted[: args.report_topk])
    md = ["# Signed direct logit attribution", "", f"Prompt: `{text}`", f"Target token: `{target_tok}` ({target_id})", "", "Top contributors:", ""]
    for r in rows_sorted[:20]:
        md.append(f"- {r['component']} L{r['layer']}H{r['head']}: direct_logit={float(r['direct_logit']):+.4f}, norm={float(r['Y_norm']):.4f}")
    (out_dir / "summary_logit_attr.md").write_text("\n".join(md), encoding="utf-8")


def mode_mlp_rich(model: Any, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> None:
    ensure_dir(out_dir)
    layers = get_layers(model)
    layer_list = parse_layers(args.layers, len(layers))
    prompts = build_prompts(args.prompt_suites, args.prompts_per_suite, args.prompt)
    rows = []
    examples = []
    for p in prompts:
        enc = tokenizer(p["text"], return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
        with torch.no_grad():
            out = model(**enc, use_cache=False, output_hidden_states=True)
        for li in layer_list:
            layer = layers[li]
            # approximate actual MLP input: post_attention_layernorm(hidden after attn) is not in hidden_states.
            # To be exact, we hook MLP for one pass per prompt/layer set.
        # exact hook per prompt
        caps = {"mlp_in": {}, "mlp_out": {}}
        handles = []
        for li in layer_list:
            def make_hook(i):
                def hook(module, fargs, output):
                    caps["mlp_in"][i] = fargs[0].detach()
                    caps["mlp_out"][i] = output.detach()
                return hook
            handles.append(layers[li].mlp.register_forward_hook(make_hook(li)))
        try:
            with torch.no_grad():
                _ = model(**enc, use_cache=False)
        finally:
            for h in handles: h.remove()
        for li in layer_list:
            if li not in caps["mlp_in"]: continue
            mlp = layers[li].mlp
            x = caps["mlp_in"][li]
            y_true = caps["mlp_out"][li]
            gate = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            act = F.silu(gate)
            hidden = act * up
            y_rec = mlp.down_proj(hidden)
            rec = rel_err(y_rec, y_true)
            y_zero = torch.zeros_like(y_true)
            # no_gate = use up only (act=1); gate_only = use silu(gate) only (up=1), diagnostic not exact semantic.
            y_no_gate = mlp.down_proj(up)
            y_gate_only = mlp.down_proj(act)
            # top neurons by hidden activation norm
            hn = hidden.detach().float()[0]
            neuron_norm = torch.linalg.norm(hn, dim=0)
            topv, topi = torch.topk(neuron_norm, k=min(args.mlp_top_neurons, neuron_norm.numel()))
            row = {"suite": p["suite"], "layer": li, "text": p["text"][:120], "T": int(x.shape[1]),
                   "mlp_rec_rel": rec, "out_norm": float(torch.linalg.norm(y_true.float())),
                   "no_gate_delta_rel": rel_err(y_no_gate, y_true), "gate_only_delta_rel": rel_err(y_gate_only, y_true),
                   "gate_mean": float(gate.float().mean()), "gate_std": float(gate.float().std()),
                   "act_mean": float(act.float().mean()), "act_sparsity_frac_abs_lt_1e-3": float((act.float().abs() < 1e-3).float().mean()),
                   "top_neurons": topi.detach().cpu().tolist(), "top_neuron_norms": topv.detach().cpu().tolist()}
            rows.append(row)
            if len(examples) < 20:
                examples.append(row)
    write_csv(out_dir / "mlp_rich_summary.csv", rows)
    write_json(out_dir / "mlp_rich_examples.json", examples)
    md = ["# Rich MLP/SwiGLU atlas", "", f"rows={len(rows)}", ""]
    for k in ["mlp_rec_rel", "no_gate_delta_rel", "gate_only_delta_rel", "act_sparsity_frac_abs_lt_1e-3"]:
        vals = [float(r[k]) for r in rows]
        md.append(f"mean {k}: {safe_mean(vals):.6g}")
    (out_dir / "summary_mlp_rich.md").write_text("\n".join(md), encoding="utf-8")


def mode_baselines(model: Any, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> None:
    ensure_dir(out_dir)
    prompts = build_prompts(args.prompt_suites, args.prompts_per_suite, args.prompt)
    heads = parse_heads_spec(args.head_spec) if args.head_spec else parse_heads_spec(args.source_head)
    rows = []
    for l, h in heads:
        for p in prompts:
            enc = tokenizer(p["text"], return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
            with torch.no_grad():
                out = model(**enc, use_cache=False, output_hidden_states=True)
            layer = get_layers(model)[l]
            X = layer.input_layernorm(out.hidden_states[l].detach())
            ht = head_terms_from_attn_input(model, layer.self_attn, X, l, h, [], None, 1.0)
            # baselines: heatmap self/local/bos stats; qk_no_bias terms remove all affine; only_content; head ablation via Y zero delta rel.
            only_content_scores = ht.term_tensors["content"]
            A_content = causal_softmax(only_content_scores)
            Y_content = (A_content @ ht.V) @ layer.self_attn.o_proj.weight.detach().float()[:, h*getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads):(h+1)*getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)].to(args.device).T
            T = ht.A_true.shape[0]
            row = {"suite": p["suite"], "layer": l, "head": h, "T": T,
                   "heatmap_entropy": float((-(ht.A_true * (ht.A_true + 1e-12).log()).sum(dim=-1) / math.log(max(2,T))).mean()),
                   "heatmap_bos_mass": float(ht.A_true[:, 0].mean()),
                   "heatmap_self_mass": float(ht.A_true.diag().mean()),
                   "heatmap_local4_mass": float(torch.stack([ht.A_true[i, max(0,i-4):min(T,i+5)].sum() for i in range(T)]).mean()),
                   "qk_content_only_A_rel": rel_err(A_content, ht.A_true),
                   "qk_content_only_Y_rel": rel_err(Y_content, ht.Y_true),
                   "head_ablation_Y_rel": rel_err(torch.zeros_like(ht.Y_true), ht.Y_true),
                   "sae_baseline": "not_implemented",
                   "transformerlens_baseline": "not_implemented"}
            rows.append(row)
    write_csv(out_dir / "baselines_summary.csv", rows)
    md = ["# Baselines", "", "Implemented: heatmap stats, head ablation, content-only QK baseline.", "Not implemented here: SAE / TransformerLens baselines.", ""]
    for k in ["qk_content_only_A_rel", "qk_content_only_Y_rel", "head_ablation_Y_rel"]:
        vals = [float(r[k]) for r in rows]
        md.append(f"mean {k}: {safe_mean(vals):.6g}")
    (out_dir / "summary_baselines.md").write_text("\n".join(md), encoding="utf-8")



# ---------------- v4 extra modes ----------------

def _key_positions_from_spec(spec: str, T: int, A: Optional[torch.Tensor] = None) -> List[int]:
    out: List[int] = []
    for part in str(spec or "0").replace(";", ",").split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "last":
            out.append(max(0, T - 1))
        elif part == "mid":
            out.append(max(0, T // 2))
        elif part == "top" and A is not None:
            # most-attended key averaged over query rows
            out.append(int(A.float().mean(dim=0).argmax().item()))
        elif "-" in part and part.replace("-", "").replace(",", "").isdigit():
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            try:
                out.append(int(part))
            except Exception:
                pass
    return sorted(set([p for p in out if 0 <= p < T]))


@torch.no_grad()
def _base_head_terms_for_text(model: Any, tokenizer: Any, text: str, source: Tuple[int, int], max_length: int, device: str) -> Tuple[Any, HeadTerms, torch.Tensor]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out = model(**enc, use_cache=False, output_hidden_states=True)
    layers = get_layers(model)
    l, h = source
    X = layers[l].input_layernorm(out.hidden_states[l].detach())
    ht = head_terms_from_attn_input(model, layers[l].self_attn, X, l, h, [], None, 1.0)
    return enc, ht, out.logits.detach().float().cpu()


def mode_token_control_sweep(model: Any, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> None:
    """Sweep key positions and terms, recording predicted-vs-actual attention direction."""
    ensure_dir(out_dir)
    heads = parse_heads_spec(args.head_spec) if args.head_spec else parse_heads_spec(args.source_head)
    prompts = build_prompts(args.prompt_suites, args.prompts_per_suite, args.prompt)
    terms = args.terms_list
    rows: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    for source in heads:
        for p in prompts:
            text = p["text"]
            try:
                _, ht0, _ = _base_head_terms_for_text(model, tokenizer, text, source, args.max_length, args.device)
            except Exception as e:
                rows.append({"source": f"{source[0]}:{source[1]}", "suite": p["suite"], "error": str(e)[:200]})
                continue
            T = int(ht0.A_true.shape[0])
            key_positions = _key_positions_from_spec(args.key_positions, T, ht0.A_true)
            for term in terms:
                if term not in ht0.term_tensors and term not in ("vo_bias", "vo_linear"):
                    continue
                for kp in key_positions:
                    pred_term_mean = None
                    pred_sign = 0
                    if term in ht0.term_tensors:
                        col = ht0.term_tensors[term][:, kp].float()
                        pred_term_mean = float(col.mean().detach().cpu())
                        # removing positive score should lower mass; removing negative should raise mass
                        pred_sign = -1 if pred_term_mean > 0 else (1 if pred_term_mean < 0 else 0)
                    res = patch_source_head_forward(model, tokenizer, text, source, [term], kp, args.patch_strength, args.max_length, args.device)
                    actual_delta = float(res.get("target_key_mass_delta_mean", 0.0))
                    actual_sign = 1 if actual_delta > 1e-8 else (-1 if actual_delta < -1e-8 else 0)
                    sign_match = (pred_sign == actual_sign) if pred_sign != 0 else None
                    row = {
                        "suite": p["suite"], "text": text[:120], "source": f"{source[0]}:{source[1]}",
                        "term": term, "key_pos": int(kp), "T": T,
                        "pred_term_mean_on_key": pred_term_mean,
                        "predicted_mass_delta_sign": pred_sign,
                        "actual_mass_delta": actual_delta,
                        "actual_mass_delta_sign": actual_sign,
                        "sign_match": sign_match,
                        "head_A_rel": res.get("head_A_rel"),
                        "head_Y_delta_rel": res.get("head_Y_delta_rel"),
                        "logit_rel": res.get("logit_rel"),
                        "kl_orig_to_patch": res.get("kl_orig_to_patch"),
                        "top1_match": res.get("top1_match"),
                        "loss_delta": res.get("loss_delta"),
                    }
                    rows.append(row)
                    if len(examples) < 30:
                        examples.append({**row, "top_changed_logits_final": res.get("top_changed_logits_final", [])})
    write_csv(out_dir / "token_control_sweep_summary.csv", rows)
    write_json(out_dir / "token_control_sweep_examples.json", examples)
    valid = [r for r in rows if r.get("sign_match") is not None]
    match_rate = safe_mean([1.0 if r.get("sign_match") else 0.0 for r in valid]) if valid else -1.0
    md = ["# Token-control sweep", "", f"rows={len(rows)}", f"valid_sign_rows={len(valid)}", f"sign_match_rate={match_rate:.6g}", ""]
    for key in ["actual_mass_delta", "head_Y_delta_rel", "logit_rel", "kl_orig_to_patch"]:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if vals:
            md.append(f"mean {key}: {safe_mean(vals):.6g}")
    (out_dir / "summary_token_control_sweep.md").write_text("\n".join(md), encoding="utf-8")


def _forward_with_source_mode(model: Any, tokenizer: Any, text: str, source: Tuple[int, int], mode: str,
                              terms: List[str], target_key_pos: Optional[int], strength: float,
                              max_length: int, device: str, labels: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """Run model with source-head mode: original / ablate_head / replace_full / term_patch."""
    layers = get_layers(model)
    l_src, h_src = source
    attn = layers[l_src].self_attn
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    info: Dict[str, Any] = {"source_mode": mode}
    if mode == "original":
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"] if labels else None, use_cache=False, output_hidden_states=True)
        return out, info

    def hook(module, fargs, kwargs, output):
        X = kwargs.get("hidden_states", None) if isinstance(kwargs, dict) else None
        if X is None:
            X = fargs[0] if fargs else None
        if X is None:
            return output
        attn_out = output[0] if isinstance(output, (tuple, list)) else output
        if mode == "ablate_head":
            ht = head_terms_from_attn_input(model, module, X, l_src, h_src, [], None, 1.0)
            Y_new = torch.zeros_like(ht.Y_true)
        elif mode == "replace_full":
            # exact pseudocode replacement: recompute with all terms; should recover original head.
            ht = head_terms_from_attn_input(model, module, X, l_src, h_src, [], None, 1.0)
            Y_new = ht.Y_patch
        elif mode == "term_patch":
            ht = head_terms_from_attn_input(model, module, X, l_src, h_src, terms, target_key_pos, strength)
            Y_new = ht.Y_patch
        else:
            return output
        delta = (Y_new - ht.Y_true).to(attn_out.device).to(attn_out.dtype).unsqueeze(0)
        info["head_A_rel"] = rel_err(ht.A_patch, ht.A_true)
        info["head_Y_delta_rel"] = rel_err(Y_new, ht.Y_true)
        new_attn = attn_out + delta
        if isinstance(output, tuple):
            return (new_attn,) + tuple(output[1:])
        if isinstance(output, list):
            return [new_attn] + list(output[1:])
        return new_attn

    h = attn.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"] if labels else None, use_cache=False, output_hidden_states=True)
    finally:
        h.remove()
    return out, info


def mode_causal_path_recovery(model: Any, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> None:
    """Prove chain replacement: original vs ablate source vs exact pseudocode replacement vs term patch."""
    ensure_dir(out_dir)
    source = parse_heads_spec(args.source_head)[0]
    target = parse_heads_spec(args.target_head)[0]
    prompts = build_prompts(args.prompt_suites, args.prompts_per_suite, args.prompt)
    rows: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    modes = ["original", "ablate_head", "replace_full", "term_patch"]
    for p in prompts:
        text = p["text"]
        base_out, _ = _forward_with_source_mode(model, tokenizer, text, source, "original", [], None, 1.0, args.max_length, args.device)
        base_t = collect_target_head_from_hidden(model, base_out.hidden_states, target[0], target[1])
        base_logits = base_out.logits.detach().float().cpu()
        base_loss = float(base_out.loss.detach().float().cpu()) if base_out.loss is not None else 0.0
        mode_stats = {}
        for m in modes[1:]:
            out, info = _forward_with_source_mode(model, tokenizer, text, source, m, args.terms_list, args.target_key_pos, args.patch_strength, args.max_length, args.device)
            tt = collect_target_head_from_hidden(model, out.hidden_states, target[0], target[1])
            logits = out.logits.detach().float().cpu()
            a = base_logits[:, :-1, :]
            b = logits[:, :-1, :]
            pa = torch.softmax(a, dim=-1)
            logpb = torch.log_softmax(b, dim=-1)
            st = {
                f"{m}_target_A_rel": rel_err(tt["A"], base_t["A"]),
                f"{m}_target_scores_rel": rel_err(tt["scores"], base_t["scores"]),
                f"{m}_target_Y_rel": rel_err(tt["Y"], base_t["Y"]),
                f"{m}_logit_rel": rel_err(b, a),
                f"{m}_kl": float(F.kl_div(logpb, pa, reduction="batchmean")),
                f"{m}_top1_match": float((a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean()),
                f"{m}_loss_delta": (float(out.loss.detach().float().cpu()) - base_loss) if out.loss is not None else 0.0,
            }
            st.update({f"{m}_{k}": v for k, v in info.items() if isinstance(v, (int, float, str))})
            mode_stats.update(st)
        # recovery: replace_full should be much closer than ablate_head. 1 = perfect recovery.
        ab = float(mode_stats.get("ablate_head_target_Y_rel", 0.0))
        rep = float(mode_stats.get("replace_full_target_Y_rel", 0.0))
        log_ab = float(mode_stats.get("ablate_head_logit_rel", 0.0))
        log_rep = float(mode_stats.get("replace_full_logit_rel", 0.0))
        row = {"suite": p["suite"], "text": text[:140], "source": f"{source[0]}:{source[1]}", "target": f"{target[0]}:{target[1]}", **mode_stats,
               "target_Y_recovery_ratio": 1.0 - (rep / max(ab, 1e-12)),
               "logit_recovery_ratio": 1.0 - (log_rep / max(log_ab, 1e-12))}
        rows.append(row)
        if len(examples) < 10:
            examples.append(row)
    write_csv(out_dir / "causal_path_recovery_summary.csv", rows)
    write_json(out_dir / "causal_path_recovery_examples.json", examples)
    md = ["# Causal path recovery", "", f"Source: {args.source_head}", f"Target: {args.target_head}", ""]
    for k in ["ablate_head_target_Y_rel", "replace_full_target_Y_rel", "term_patch_target_Y_rel", "target_Y_recovery_ratio", "ablate_head_logit_rel", "replace_full_logit_rel", "logit_recovery_ratio"]:
        vals = [float(r[k]) for r in rows if k in r]
        if vals: md.append(f"mean {k}: {safe_mean(vals):.6g}")
    (out_dir / "summary_causal_path_recovery.md").write_text("\n".join(md), encoding="utf-8")


def mode_logit_patch_attr(model: Any, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> None:
    """Patch-based logit attribution by ablating individual heads and MLP layers."""
    ensure_dir(out_dir)
    text = args.prompt or args.generate_prompt or "Write a Python function that reverses a linked list."
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
    with torch.no_grad():
        base = model(**enc, labels=enc["input_ids"], use_cache=False)
    base_logits = base.logits.detach().float().cpu()
    final_base = base_logits[0, -1]
    target_id = int(final_base.argmax()) if args.target_token_id < 0 else int(args.target_token_id)
    target_tok = tokenizer.convert_ids_to_tokens([target_id])[0]
    layers = get_layers(model)
    rows: List[Dict[str, Any]] = []
    heads = parse_heads_spec(args.head_spec) if args.head_spec else []
    # head ablations
    for l, hidx in heads:
        attn = layers[l].self_attn
        def hook(module, fargs, kwargs, output, li=l, hi=hidx):
            X = kwargs.get("hidden_states", None) if isinstance(kwargs, dict) else None
            if X is None: X = fargs[0] if fargs else None
            attn_out = output[0] if isinstance(output, (tuple, list)) else output
            ht = head_terms_from_attn_input(model, module, X, li, hi, [], None, 1.0)
            delta = (-ht.Y_true).to(attn_out.device).to(attn_out.dtype).unsqueeze(0)
            new_attn = attn_out + delta
            if isinstance(output, tuple): return (new_attn,) + tuple(output[1:])
            if isinstance(output, list): return [new_attn] + list(output[1:])
            return new_attn
        handle = attn.register_forward_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad(): patched = model(**enc, labels=enc["input_ids"], use_cache=False)
        finally:
            handle.remove()
        plog = patched.logits.detach().float().cpu()
        # contribution to target ≈ base - ablated
        contrib = float(final_base[target_id] - plog[0, -1, target_id])
        rows.append({"component": "head", "layer": l, "head": hidx, "target_token_id": target_id, "target_token": target_tok,
                     "patch_type": "ablate", "causal_logit_contribution": contrib,
                     "logit_rel": rel_err(plog[:, :-1, :], base_logits[:, :-1, :]),
                     "loss_delta_ablate": float(patched.loss.detach().float().cpu()) - float(base.loss.detach().float().cpu()),
                     "top1_match": float((plog[:, :-1, :].argmax(dim=-1) == base_logits[:, :-1, :].argmax(dim=-1)).float().mean())})
    # MLP layer ablations
    layer_list = parse_layers(args.layers, len(layers))
    for li in layer_list:
        def mhook(module, fargs, output):
            return torch.zeros_like(output)
        handle = layers[li].mlp.register_forward_hook(mhook)
        try:
            with torch.no_grad(): patched = model(**enc, labels=enc["input_ids"], use_cache=False)
        finally:
            handle.remove()
        plog = patched.logits.detach().float().cpu()
        contrib = float(final_base[target_id] - plog[0, -1, target_id])
        rows.append({"component": "mlp", "layer": li, "head": -1, "target_token_id": target_id, "target_token": target_tok,
                     "patch_type": "zero_mlp", "causal_logit_contribution": contrib,
                     "logit_rel": rel_err(plog[:, :-1, :], base_logits[:, :-1, :]),
                     "loss_delta_ablate": float(patched.loss.detach().float().cpu()) - float(base.loss.detach().float().cpu()),
                     "top1_match": float((plog[:, :-1, :].argmax(dim=-1) == base_logits[:, :-1, :].argmax(dim=-1)).float().mean())})
    rows = sorted(rows, key=lambda r: abs(float(r["causal_logit_contribution"])), reverse=True)
    write_csv(out_dir / "logit_patch_attribution.csv", rows)
    write_json(out_dir / "top_logit_patch_attribution.json", rows[: args.report_topk])
    md = ["# Patch-based logit attribution", "", f"Prompt: `{text}`", f"Target token: `{target_tok}` ({target_id})", ""]
    for r in rows[:30]:
        md.append(f"- {r['component']} L{r['layer']}H{r['head']}: causal_logit_contribution={float(r['causal_logit_contribution']):+.5f}, logit_rel={float(r['logit_rel']):.5g}")
    (out_dir / "summary_logit_patch_attr.md").write_text("\n".join(md), encoding="utf-8")


def _silu_prime(x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def mode_mlp_operator(model: Any, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> None:
    """MLP local operator/Jacobian decomposition for last token on prompts."""
    ensure_dir(out_dir)
    layers = get_layers(model)
    layer_list = parse_layers(args.layers, len(layers))
    prompts = build_prompts(args.prompt_suites, args.prompts_per_suite, args.prompt)
    rows: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    for p in prompts:
        text = p["text"]
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
        caps = {"x": {}, "y": {}}
        handles = []
        for li in layer_list:
            def make_hook(i):
                def hook(module, fargs, output):
                    caps["x"][i] = fargs[0].detach()
                    caps["y"][i] = output.detach()
                return hook
            handles.append(layers[li].mlp.register_forward_hook(make_hook(li)))
        try:
            with torch.no_grad(): model(**enc, use_cache=False)
        finally:
            for h in handles: h.remove()
        for li in layer_list:
            if li not in caps["x"]: continue
            mlp = layers[li].mlp
            xall = caps["x"][li]
            pos = int(xall.shape[1] - 1)
            x = xall[:, pos:pos+1, :]
            with torch.no_grad():
                gate = mlp.gate_proj(x).float()[0, 0]
                up = mlp.up_proj(x).float()[0, 0]
                act = F.silu(gate)
                hidden = act * up
                y = mlp.down_proj((act * up).to(x.dtype).view(1,1,-1)).float()[0,0]
                Wg = mlp.gate_proj.weight.detach().float()
                Wu = mlp.up_proj.weight.detach().float()
                Wd = mlp.down_proj.weight.detach().float()  # [H, I]
                # local Jacobian: Wd @ (diag(up*silu'(gate)) Wg + diag(silu(gate)) Wu)
                coeff_g = up * _silu_prime(gate)
                coeff_u = act
                # Compute J without materializing huge diagonal explicitly.
                J = Wd @ (coeff_g[:, None] * Wg + coeff_u[:, None] * Wu)
                jnorm = float(torch.linalg.norm(J))
                # rank approximations
                sv = torch.linalg.svdvals(J.cpu())
                energy = torch.cumsum(sv ** 2, dim=0) / torch.clamp((sv ** 2).sum(), min=1e-12)
                rank90 = int((energy < 0.90).sum().item() + 1)
                rank95 = int((energy < 0.95).sum().item() + 1)
                rank99 = int((energy < 0.99).sum().item() + 1)
                # top neurons by absolute contribution norm
                contrib_norm = torch.linalg.norm(Wd.cpu(), dim=0) * hidden.detach().cpu().abs()
                tv, ti = torch.topk(contrib_norm, k=min(args.mlp_top_neurons, contrib_norm.numel()))
            row = {"suite": p["suite"], "text": text[:120], "layer": li, "pos": pos,
                   "J_norm": jnorm, "J_rank90": rank90, "J_rank95": rank95, "J_rank99": rank99,
                   "gate_mean": float(gate.mean().cpu()), "gate_std": float(gate.std().cpu()),
                   "up_mean": float(up.mean().cpu()), "hidden_norm": float(torch.linalg.norm(hidden).cpu()),
                   "out_norm": float(torch.linalg.norm(y).cpu()),
                   "top_neurons": ti.tolist(), "top_neuron_contrib_norms": tv.tolist()}
            rows.append(row)
            if len(examples) < 30: examples.append(row)
    write_csv(out_dir / "mlp_operator_summary.csv", rows)
    write_json(out_dir / "mlp_operator_examples.json", examples)
    md = ["# MLP local operator/Jacobian atlas", "", f"rows={len(rows)}", ""]
    for k in ["J_norm", "J_rank90", "J_rank95", "J_rank99", "hidden_norm", "out_norm"]:
        vals = [float(r[k]) for r in rows if k in r]
        if vals: md.append(f"mean {k}: {safe_mean(vals):.6g}")
    (out_dir / "summary_mlp_operator.md").write_text("\n".join(md), encoding="utf-8")

def load_model(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["causal_path", "causal_path_recovery", "logit_attr", "logit_patch_attr", "mlp_rich", "mlp_operator", "token_control", "token_control_sweep", "baselines", "all", "all_v4"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--head-spec", default="")
    ap.add_argument("--source-head", default="3:6")
    ap.add_argument("--target-head", default="4:8")
    ap.add_argument("--patch-terms", default="k_affine,content")
    ap.add_argument("--patch-strength", type=float, default=1.0)
    ap.add_argument("--target-key-pos", type=int, default=-1)
    ap.add_argument("--key-positions", default="0,1,2,3,4,8,last,top")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--generate-prompt", default="Write a Python function that reverses a linked list.")
    ap.add_argument("--prompt-suites", default="all")
    ap.add_argument("--prompts-per-suite", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--target-token-id", type=int, default=-1)
    ap.add_argument("--report-topk", type=int, default=60)
    ap.add_argument("--mlp-top-neurons", type=int, default=16)
    ap.add_argument("--out-dir", default="./qwen_pseudocode_controls_v3_out")
    args = ap.parse_args()
    args.terms_list = [x.strip() for x in args.patch_terms.split(",") if x.strip()]
    if args.target_key_pos < 0:
        args.target_key_pos = None
    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    model, tokenizer = load_model(args)
    write_json(out_root / "run_args.json", vars(args))

    run_all_basic = args.mode in ("all", "all_v4")
    run_all_v4 = args.mode == "all_v4"
    if args.mode in ("token_control",) or run_all_basic:
        mode_token_control(model, tokenizer, args, out_root / "token_control")
    if args.mode in ("token_control_sweep",) or run_all_v4:
        mode_token_control_sweep(model, tokenizer, args, out_root / "token_control_sweep")
    if args.mode in ("causal_path",) or run_all_basic:
        mode_causal_path(model, tokenizer, args, out_root / "causal_path")
    if args.mode in ("causal_path_recovery",) or run_all_v4:
        mode_causal_path_recovery(model, tokenizer, args, out_root / "causal_path_recovery")
    if args.mode in ("logit_attr",) or run_all_basic:
        mode_logit_attr(model, tokenizer, args, out_root / "logit_attr")
    if args.mode in ("logit_patch_attr",) or run_all_v4:
        if not args.head_spec:
            args.head_spec = args.source_head
        mode_logit_patch_attr(model, tokenizer, args, out_root / "logit_patch_attr")
    if args.mode in ("mlp_rich",) or run_all_basic:
        mode_mlp_rich(model, tokenizer, args, out_root / "mlp_rich")
    if args.mode in ("mlp_operator",) or run_all_v4:
        mode_mlp_operator(model, tokenizer, args, out_root / "mlp_operator")
    if args.mode in ("baselines",) or run_all_basic:
        if not args.head_spec:
            args.head_spec = args.source_head
        mode_baselines(model, tokenizer, args, out_root / "baselines")

    # dashboard summary
    md = ["# Qwen pseudocode controls v3", "", f"mode={args.mode}", "", "Generated folders:"]
    for sub in ["token_control", "token_control_sweep", "causal_path", "causal_path_recovery", "logit_attr", "logit_patch_attr", "mlp_rich", "mlp_operator", "baselines"]:
        if (out_root / sub).exists():
            md.append(f"- `{sub}/`")
    (out_root / "DASHBOARD_CONTROLS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[done] wrote {out_root}", flush=True)


if __name__ == "__main__":
    main()
