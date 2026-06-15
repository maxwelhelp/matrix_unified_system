#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_term_control_probe_v1.py

Targeted control test for one Qwen attention head score-term.
It does NOT just describe a head. It intervenes inside the model:
  score = content + q_affine + k_affine + constant
and subtracts one chosen term globally or for a selected token position, then
measures attention/output/logit changes.

Example:
  python qwen_term_control_probe_v1.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --dtype fp16 \
    --attn-implementation eager --head 2:1 \
    --prompt "Write a Python function that reverses a linked list." \
    --term k_affine --target-key-pos 0 --strength 1.0 \
    --out ./term_control_L2H1_kpos0.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def get_dtype(name: str):
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(name)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_part(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [B,T,D], cos/sin [B,T,D] or [T,D]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    # scores [T,T]
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1)


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def get_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find decoder layers")


def compute_position_embeddings(model: Any, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    rotary = getattr(model.model, "rotary_emb", None) if hasattr(model, "model") else None
    if rotary is None:
        raise RuntimeError("model.model.rotary_emb not found")
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return rotary(position_ids)


def safe_json(x: Any):
    if torch.is_tensor(x):
        if x.numel() <= 32:
            return x.detach().cpu().tolist()
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe_json(v) for v in x]
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    return str(x)


def parse_head(s: str) -> Tuple[int, int]:
    a, b = s.split(":", 1)
    return int(a), int(b)


def top_tokens(tokenizer, logits: torch.Tensor, k: int = 10):
    vals, idx = torch.topk(logits.float(), k=min(k, logits.numel()))
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        try:
            tok = tokenizer.decode([i])
        except Exception:
            tok = str(i)
        out.append({"token_id": int(i), "token": tok, "logit": float(v)})
    return out


def build_term_scores(module: Any, model: Any, hidden_states: torch.Tensor, layer_idx: int, head_idx: int,
                      position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # hidden_states is input to self_attn, i.e. post RMSNorm in Qwen decoder layer.
    B, T, H = hidden_states.shape
    cfg = model.config
    num_heads = int(cfg.num_attention_heads)
    num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", H // num_heads))
    groups = num_heads // num_kv
    kv_idx = head_idx // groups
    dev = hidden_states.device
    dtype = torch.float32

    hs = hidden_states.float()
    q0 = head_idx * head_dim
    q1 = (head_idx + 1) * head_dim
    k0 = kv_idx * head_dim
    k1 = (kv_idx + 1) * head_dim

    Wq = module.q_proj.weight[q0:q1, :].float().to(dev)
    Wk = module.k_proj.weight[k0:k1, :].float().to(dev)
    Wv = module.v_proj.weight[k0:k1, :].float().to(dev)
    bq = module.q_proj.bias[q0:q1].float().to(dev) if getattr(module.q_proj, "bias", None) is not None else torch.zeros(head_dim, device=dev)
    bk = module.k_proj.bias[k0:k1].float().to(dev) if getattr(module.k_proj, "bias", None) is not None else torch.zeros(head_dim, device=dev)
    bv = module.v_proj.bias[k0:k1].float().to(dev) if getattr(module.v_proj, "bias", None) is not None else torch.zeros(head_dim, device=dev)

    q_lin = torch.matmul(hs, Wq.T)
    k_lin = torch.matmul(hs, Wk.T)
    q_bias = bq.view(1, 1, head_dim).expand(B, T, head_dim)
    k_bias = bk.view(1, 1, head_dim).expand(B, T, head_dim)

    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
        cos, sin = compute_position_embeddings(model, hidden_states, pos)

    ql = apply_rope_part(q_lin, cos, sin)
    qb = apply_rope_part(q_bias, cos, sin)
    kl = apply_rope_part(k_lin, cos, sin)
    kb = apply_rope_part(k_bias, cos, sin)

    denom = math.sqrt(head_dim)
    # only batch 0 for reporting/patch; script runs batch size 1.
    ql0, qb0, kl0, kb0 = ql[0], qb[0], kl[0], kb[0]
    content = (ql0 @ kl0.T) / denom
    q_affine = (ql0 @ kb0.T) / denom
    k_affine = (qb0 @ kl0.T) / denom
    constant = (qb0 @ kb0.T) / denom
    full = content + q_affine + k_affine + constant

    # Real V and O slice for selected head.
    V = torch.matmul(hs, Wv.T) + bv.view(1, 1, head_dim)
    Ow = module.o_proj.weight[:, q0:q1].float().to(dev)
    return {
        "content": content.float(),
        "q_affine": q_affine.float(),
        "k_affine": k_affine.float(),
        "constant": constant.float(),
        "full": full.float(),
        "V": V[0].float(),
        "Ow": Ow.float(),
    }


def make_term_mask(T: int, term: torch.Tensor, args: argparse.Namespace, device: str) -> torch.Tensor:
    M = torch.zeros_like(term)
    if args.target_query_pos >= 0 and args.target_key_pos >= 0:
        i, j = args.target_query_pos, args.target_key_pos
        if 0 <= i < T and 0 <= j < T and j <= i:
            M[i, j] = term[i, j]
    elif args.target_query_pos >= 0:
        i = args.target_query_pos
        if 0 <= i < T:
            M[i, : i + 1] = term[i, : i + 1]
    elif args.target_key_pos >= 0:
        j = args.target_key_pos
        if 0 <= j < T:
            M[j:, j] = term[j:, j]
    else:
        # global causal term removal
        tril = torch.tril(torch.ones(T, T, device=term.device, dtype=torch.bool))
        M[tril] = term[tril]
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--head", required=True, help="L:H")
    ap.add_argument("--prompt", default="Write a Python function that reverses a linked list.")
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--term", choices=["constant", "q_affine", "k_affine", "content"], default="k_affine")
    ap.add_argument("--target-query-pos", type=int, default=-1)
    ap.add_argument("--target-key-pos", type=int, default=-1)
    ap.add_argument("--strength", type=float, default=1.0, help="subtract strength * selected term")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="term_control_probe.json")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = get_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    layer_idx, head_idx = parse_head(args.head)
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn

    enc = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    T = int(input_ids.shape[1])

    with torch.no_grad():
        base_out = model(**enc, use_cache=False)
        base_logits = base_out.logits[0, -1].detach().float().cpu()

    hook_cache: Dict[str, Any] = {}

    def hook(module, fargs, kwargs, output):
        hidden_states = kwargs.get("hidden_states", None) if isinstance(kwargs, dict) else None
        if hidden_states is None:
            hidden_states = fargs[0] if fargs else None
        if hidden_states is None or hidden_states.shape[0] != 1:
            return output
        position_embeddings = kwargs.get("position_embeddings", None) if isinstance(kwargs, dict) else None
        attn_out = output[0] if isinstance(output, (tuple, list)) else output

        terms = build_term_scores(module, model, hidden_states, layer_idx, head_idx, position_embeddings)
        scores = terms["full"]
        A_true = causal_softmax(scores)
        term = terms[args.term]
        term_selected = make_term_mask(scores.shape[0], term, args, args.device)
        scores_patch = scores - float(args.strength) * term_selected
        A_patch = causal_softmax(scores_patch)
        V, Ow = terms["V"], terms["Ow"]
        Y_true = (A_true @ V) @ Ow.T
        Y_patch = (A_patch @ V) @ Ow.T
        delta = (Y_patch - Y_true).unsqueeze(0).to(attn_out.device)

        hook_cache["A_true"] = A_true.detach().cpu()
        hook_cache["A_patch"] = A_patch.detach().cpu()
        hook_cache["scores"] = scores.detach().cpu()
        hook_cache["scores_patch"] = scores_patch.detach().cpu()
        hook_cache["term_selected"] = term_selected.detach().cpu()
        hook_cache["Y_delta_norm"] = float(torch.linalg.norm((Y_patch - Y_true).float()).detach().cpu())
        hook_cache["Y_true_norm"] = float(torch.linalg.norm(Y_true.float()).detach().cpu())
        hook_cache["Y_delta_rel"] = float(torch.linalg.norm((Y_patch - Y_true).float()) / torch.linalg.norm(Y_true.float()).clamp_min(1e-12))

        new_attn = (attn_out.float() + delta.float()).to(attn_out.dtype)
        if isinstance(output, tuple):
            return (new_attn,) + tuple(output[1:])
        if isinstance(output, list):
            return [new_attn] + list(output[1:])
        return new_attn

    handle = attn.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            patch_out = model(**enc, use_cache=False)
            patch_logits = patch_out.logits[0, -1].detach().float().cpu()
    finally:
        handle.remove()

    logit_delta = patch_logits - base_logits
    probs_base = torch.softmax(base_logits, dim=-1)
    log_probs_patch = torch.log_softmax(patch_logits, dim=-1)
    kl = float(F.kl_div(log_probs_patch, probs_base, reduction="sum"))
    top_delta_vals, top_delta_idx = torch.topk(logit_delta.abs(), k=min(args.topk, logit_delta.numel()))

    # Attention diagnostics for selected key/query.
    A0 = hook_cache.get("A_true")
    A1 = hook_cache.get("A_patch")
    attn_diag = {}
    if torch.is_tensor(A0) and torch.is_tensor(A1):
        attn_diag["A_rel"] = rel_err(A1, A0)
        if args.target_key_pos >= 0 and args.target_key_pos < T:
            j = args.target_key_pos
            attn_diag["target_key_token"] = tokens[j]
            attn_diag["target_key_mass_before_mean"] = float(A0[:, j].mean())
            attn_diag["target_key_mass_after_mean"] = float(A1[:, j].mean())
            attn_diag["target_key_mass_delta_mean"] = float((A1[:, j] - A0[:, j]).mean())
            attn_diag["target_key_mass_before_sum"] = float(A0[:, j].sum())
            attn_diag["target_key_mass_after_sum"] = float(A1[:, j].sum())
        if args.target_query_pos >= 0 and args.target_query_pos < T:
            i = args.target_query_pos
            attn_diag["target_query_token"] = tokens[i]
            vals0, idx0 = torch.topk(A0[i], k=min(8, i + 1))
            vals1, idx1 = torch.topk(A1[i], k=min(8, i + 1))
            attn_diag["query_top_before"] = [{"pos": int(j), "tok": tokens[int(j)], "p": float(v)} for v, j in zip(vals0, idx0)]
            attn_diag["query_top_after"] = [{"pos": int(j), "tok": tokens[int(j)], "p": float(v)} for v, j in zip(vals1, idx1)]

    changed = []
    for v, idx in zip(top_delta_vals.tolist(), top_delta_idx.tolist()):
        changed.append({
            "token_id": int(idx),
            "token": tokenizer.decode([int(idx)]),
            "base_logit": float(base_logits[idx]),
            "patched_logit": float(patch_logits[idx]),
            "delta": float(logit_delta[idx]),
            "abs_delta": float(v),
        })

    report = {
        "prompt": args.prompt,
        "head": args.head,
        "term": args.term,
        "target_query_pos": args.target_query_pos,
        "target_key_pos": args.target_key_pos,
        "strength": args.strength,
        "T": T,
        "tokens": tokens,
        "base_next_top": top_tokens(tokenizer, base_logits, args.topk),
        "patched_next_top": top_tokens(tokenizer, patch_logits, args.topk),
        "logit_rel": rel_err(patch_logits, base_logits),
        "kl_base_to_patch": kl,
        "top1_same": int(base_logits.argmax()) == int(patch_logits.argmax()),
        "top_changed_logits": changed,
        "attention": attn_diag,
        "head_y": {
            "delta_norm": hook_cache.get("Y_delta_norm"),
            "true_norm": hook_cache.get("Y_true_norm"),
            "delta_rel": hook_cache.get("Y_delta_rel"),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(safe_json(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(safe_json(report), indent=2, ensure_ascii=False)[:12000])
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
