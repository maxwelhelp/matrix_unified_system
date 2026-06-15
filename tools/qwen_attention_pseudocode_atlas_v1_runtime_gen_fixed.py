#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_attention_pseudocode_atlas_v1.py

Architecture-aware matrix-pseudocode atlas for Qwen2/Qwen2.5 attention heads.

What it collects:
  1) STATIC WEIGHT ATLAS, without activations:
       Wq/Wk/Wv/Wo slices per head
       affine QK delta split:
           score = content_bilinear + q_affine + k_affine + const_delta
       affine VO split:
           payload = VO_linear @ x + VO_bias
       rank profiles, head types, cross-head/cross-layer subspace overlaps.

  2) RUNTIME ATLAS, on prompts:
       X after RMSNorm, Q/K/V, RoPE, scores, A, Y
       exact pseudocode check per head:
           score_terms sum -> A -> Y
       term energy and functional ablations:
           no_const/no_q/no_k/no_content/no_vo_bias/only_*.
       route summaries.
       optional compressed token examples and optional full tensors.

  3) DIFFERENTIABLE MULTI-HEAD BANK, for selected heads:
       learns head coefficients and gates for score terms / VO terms to reconstruct
       the joint output of a head bank.

  4) GREEDY GENERATION TRACE:
       runs generation step-by-step and writes a compressed "why/trace" per step:
       top logits + per-head last-token score-term/route/write summaries.

This is not just QK/OV. It builds executable matrix pseudocode targets:
    score_ij = c_delta + q_affine_i_delta + k_affine_j_delta + x_i^T B_delta x_j
    payload_j = C_vo_linear x_j + b_vo
    Y_i = sum_j softmax(score_ij) * payload_j

Designed for Qwen/Qwen2.5 style modules with:
    model.model.layers[*].self_attn.{q_proj,k_proj,v_proj,o_proj}
    model.model.rotary_emb
    RMSNorm before attention.

Example static all heads:
  python qwen_attention_pseudocode_atlas_v1.py \
    --mode static \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --dtype fp16 \
    --layers all --heads all \
    --deltas 0,1,2,3,4,5,6,7,8,16,32,64 \
    --out-dir ./atlas_static_all

Example runtime for layers 0-5 all heads:
  python qwen_attention_pseudocode_atlas_v1.py \
    --mode runtime \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --dtype fp16 \
    --layers 0-5 --heads all \
    --prompt-suites all --prompts-per-suite 8 --max-length 192 \
    --runtime-save-examples --out-dir ./atlas_runtime_L0_5

Example 10-head bank:
  python qwen_attention_pseudocode_atlas_v1.py \
    --mode bank \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --dtype fp16 \
    --head-spec 2:1,3:6,3:2,4:8,4:2,4:0,4:5,2:8,0:9,4:9 \
    --prompt-suites all --prompts-per-suite 8 --max-length 192 \
    --bank-steps 250 --out-dir ./atlas_bank_10

Example generation trace:
  python qwen_attention_pseudocode_atlas_v1.py \
    --mode generate_trace \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --dtype fp16 \
    --layers 0-5 --heads all \
    --generate-prompt "Write a Python function that reverses a linked list." \
    --generate-steps 8 --out-dir ./atlas_gen_trace
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import numpy as np
except Exception:
    np = None

# ----------------------------- generic utils -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.float()
    b = b.float()
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(eps))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


def safe_mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def json_sanitize(x: Any, max_tensor_elems: int = 16) -> Any:
    if x is None or isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return str(x)
        return x
    if isinstance(x, Path):
        return str(x)
    if torch.is_tensor(x):
        xc = x.detach().cpu()
        if xc.numel() <= max_tensor_elems:
            return xc.tolist()
        return {"tensor": True, "shape": list(xc.shape), "dtype": str(xc.dtype)}
    if np is not None:
        if isinstance(x, np.generic):
            return json_sanitize(x.item())
        if isinstance(x, np.ndarray):
            if x.size <= max_tensor_elems:
                return x.tolist()
            return {"ndarray": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_sanitize(v) for v in x]
    if hasattr(x, "__dict__"):
        return json_sanitize(vars(x))
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_sanitize(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for p in s.replace(";", ",").split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


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
    return sorted(set(x for x in out if 0 <= x < n_layers))


def parse_head_indices(s: str, n_heads: int) -> List[int]:
    s = str(s).strip().lower()
    if s == "all":
        return list(range(n_heads))
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
    return sorted(set(x for x in out if 0 <= x < n_heads))


def parse_head_spec(spec: str, n_layers: int, n_heads: int) -> Optional[List[Tuple[int, int]]]:
    if not spec:
        return None
    spec = spec.strip().lower()
    if spec == "":
        return None
    if spec == "all":
        return [(l, h) for l in range(n_layers) for h in range(n_heads)]
    out: List[Tuple[int, int]] = []
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad --head-spec item {part!r}; expected L:H")
        l_s, h_s = part.split(":", 1)
        l = int(l_s)
        h = int(h_s)
        if 0 <= l < n_layers and 0 <= h < n_heads:
            out.append((l, h))
    seen = set()
    uniq: List[Tuple[int, int]] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def build_head_specs(args: argparse.Namespace, n_layers: int, n_heads: int) -> List[Tuple[int, int]]:
    explicit = parse_head_spec(args.head_spec, n_layers, n_heads)
    if explicit is not None:
        return explicit
    layers = parse_layers(args.layers, n_layers)
    heads = parse_head_indices(args.heads, n_heads)
    return [(l, h) for l in layers for h in heads]


# ----------------------------- prompts -----------------------------

PROMPTS: Dict[str, List[str]] = {
    "code": [
        "Write a Python function that reverses a linked list.",
        "Explain what this code does: for i in range(n): total += arr[i]",
        "Complete the JavaScript function: function debounce(fn, delay) {",
        "Find the bug in this Python code: def f(x): return x.append(1)",
        "Implement binary search over a sorted list in Python.",
        "What is the time complexity of nested loops over n and m?",
        "Convert this pseudocode to Python: initialize sum to zero, loop over items, add item.",
        "Explain recursion using factorial as an example.",
    ],
    "math": [
        "Solve: if 3x + 5 = 20, what is x?",
        "Explain the derivative of sin(x) * exp(x).",
        "A triangle has sides 3, 4, 5. What is its area?",
        "Compute the determinant of a 2 by 2 matrix [[a,b],[c,d]].",
        "Simplify: (x + 2)^2 - x^2.",
        "Explain Bayes theorem with a simple example.",
    ],
    "text": [
        "Summarize why rivers are important for cities.",
        "Write a short paragraph about autumn forests.",
        "Explain the difference between a planet and a star.",
        "Describe how plants grow from seeds.",
        "Give a short explanation of why sleep matters.",
    ],
    "dialogue": [
        "User: I am confused about fractions. Assistant:",
        "Question: Why is the sky blue? Answer:",
        "Continue the conversation politely: Hi, can you help me plan my study schedule?",
        "User: Can you explain this more simply? Assistant:",
    ],
    "symbols": [
        "JSON: {\"name\": \"Alice\", \"score\": 42, \"ok\": true}",
        "Sequence: A1, B2, C3, D4, E5, ...",
        "<div class=\"box\">Hello <span>world</span></div>",
        "Path: /home/user/project/src/main.py --flag=true",
    ],
    "repeat": [
        "abc def abc def abc def abc def abc def",
        "the cat sat the cat sat the cat sat the cat sat",
        "one two three one two three one two three",
        "A B C D A B C D A B C D A B C D",
    ],
}


def build_prompts(suites: str, prompts_per_suite: int, prompt_file: str = "") -> List[Dict[str, str]]:
    if prompt_file:
        p = Path(prompt_file)
        rows = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if line:
                rows.append({"suite": "file", "text": line})
        return rows
    if suites.strip().lower() == "all":
        suite_names = list(PROMPTS.keys())
    else:
        suite_names = [x.strip() for x in suites.replace(";", ",").split(",") if x.strip()]
    rows = []
    for suite in suite_names:
        base = PROMPTS.get(suite)
        if not base:
            raise ValueError(f"unknown prompt suite {suite!r}; options={list(PROMPTS)}")
        for i in range(prompts_per_suite):
            rows.append({"suite": suite, "text": base[i % len(base)]})
    return rows


# ----------------------------- model helpers -----------------------------

def get_dtype(name: str):
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(name)


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.to(args.device)
    model.eval()
    return model, tok


def get_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find decoder layers")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_qwen(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rope_rows(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [T,D] or [*,D], cos/sin [T,D] broadcast-compatible
    return (x * cos) + (rotate_half(x) * sin)


def causal_mask(T: int, device: str | torch.device) -> torch.Tensor:
    return torch.tril(torch.ones((T, T), device=device, dtype=torch.bool))


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    T = scores.shape[-1]
    mask = torch.triu(torch.ones((T, T), device=scores.device, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.float().masked_fill(mask, -1e9), dim=-1)


def compute_position_embeddings(model: Any, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    rotary = getattr(model.model, "rotary_emb", None) if hasattr(model, "model") else None
    if rotary is None:
        raise RuntimeError("model.model.rotary_emb not found")
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return rotary(position_ids)


def get_model_dims(model: Any) -> Dict[str, int]:
    cfg = model.config
    hidden_size = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
    kv_groups = num_heads // num_kv
    return {"hidden": hidden_size, "n_heads": num_heads, "n_kv": num_kv, "head_dim": head_dim, "kv_groups": kv_groups}


def module_bias(proj: Any, length: int, device: str | torch.device) -> torch.Tensor:
    b = getattr(proj, "bias", None)
    if b is None:
        return torch.zeros((length,), device=device, dtype=torch.float32)
    return b.detach().float().to(device)


@dataclass
class HeadWeights:
    layer: int
    head: int
    kv_idx: int
    Wq: torch.Tensor  # [D,H]
    Wk: torch.Tensor  # [D,H]
    Wv: torch.Tensor  # [D,H]
    Wo: torch.Tensor  # [H,D]
    bq: torch.Tensor  # [D]
    bk: torch.Tensor  # [D]
    bv: torch.Tensor  # [D]


def get_head_weights(model: Any, layer_idx: int, head_idx: int, device: str = "cpu") -> HeadWeights:
    dims = get_model_dims(model)
    D = dims["head_dim"]
    H = dims["hidden"]
    kv_idx = head_idx // dims["kv_groups"]
    attn = get_layers(model)[layer_idx].self_attn
    qW = attn.q_proj.weight.detach().float()[head_idx * D : (head_idx + 1) * D, :].to(device)
    kW = attn.k_proj.weight.detach().float()[kv_idx * D : (kv_idx + 1) * D, :].to(device)
    vW = attn.v_proj.weight.detach().float()[kv_idx * D : (kv_idx + 1) * D, :].to(device)
    oW = attn.o_proj.weight.detach().float()[:, head_idx * D : (head_idx + 1) * D].to(device)
    bq_full = module_bias(attn.q_proj, dims["n_heads"] * D, device)
    bk_full = module_bias(attn.k_proj, dims["n_kv"] * D, device)
    bv_full = module_bias(attn.v_proj, dims["n_kv"] * D, device)
    bq = bq_full[head_idx * D : (head_idx + 1) * D]
    bk = bk_full[kv_idx * D : (kv_idx + 1) * D]
    bv = bv_full[kv_idx * D : (kv_idx + 1) * D]
    return HeadWeights(layer_idx, head_idx, kv_idx, qW, kW, vW, oW, bq, bk, bv)


def rope_row_matrix(cos_pos: torch.Tensor, sin_pos: torch.Tensor) -> torch.Tensor:
    # row-vector matrix R such that x_rope = x @ R.
    D = int(cos_pos.numel())
    eye = torch.eye(D, device=cos_pos.device, dtype=torch.float32)
    return apply_rope_rows(eye, cos_pos.unsqueeze(0), sin_pos.unsqueeze(0))


def rope_relative_mats(model: Any, max_pos: int, device: str, hidden_size: int, head_dim: int) -> List[torch.Tensor]:
    dummy = torch.zeros((1, max_pos + 1, hidden_size), device=device)
    pos = torch.arange(max_pos + 1, device=device).unsqueeze(0)
    cos, sin = compute_position_embeddings(model, dummy, pos)
    cos = cos[0].float()
    sin = sin[0].float()
    R = [rope_row_matrix(cos[i, :head_dim], sin[i, :head_dim]) for i in range(max_pos + 1)]
    R0T = R[0].T
    return [R[d] @ R0T for d in range(max_pos + 1)]


# ----------------------------- static atlas -----------------------------

def svd_rank_errors(M: torch.Tensor, ranks: List[int]) -> Dict[str, float]:
    if not ranks:
        return {}
    M = M.float()
    try:
        s = torch.linalg.svdvals(M)
        total = torch.sum(s * s).clamp_min(1e-12)
        out: Dict[str, float] = {}
        for r in ranks:
            rr = min(int(r), int(s.numel()))
            tail = torch.sum(s[rr:] * s[rr:]) if rr < s.numel() else torch.tensor(0.0, device=s.device)
            out[f"rank{r}_rel_err"] = float(torch.sqrt(tail / total))
            out[f"rank{r}_energy"] = float(torch.sum(s[:rr] * s[:rr]) / total)
        return out
    except Exception as e:
        return {"svd_failed": str(e)[:200]}


def affine_qk_parts_static(hw: HeadWeights, Rrel: torch.Tensor, sqrt_d: float) -> Dict[str, torch.Tensor]:
    # score = x_i Wq.T Rrel Wk x_j.T + x_i u + v^T x_j + c
    Wq, Wk, bq, bk = hw.Wq, hw.Wk, hw.bq, hw.bk
    B = (Wq.T @ Rrel @ Wk) / sqrt_d                  # [H,H]
    u = (Wq.T @ Rrel @ bk) / sqrt_d                  # [H]
    v = (bq @ Rrel @ Wk) / sqrt_d                    # [H]
    c = (bq @ Rrel @ bk) / sqrt_d                    # scalar
    return {"B": B, "u": u, "v": v, "c": c}


def norm2(x: torch.Tensor) -> float:
    return float(torch.sum(x.float() * x.float()))


def static_head_summary(model: Any, hw: HeadWeights, Rrels: Dict[int, torch.Tensor], ranks: List[int], svd: bool) -> Dict[str, Any]:
    dims = get_model_dims(model)
    sqrt_d = math.sqrt(dims["head_dim"])
    row: Dict[str, Any] = {"layer": hw.layer, "head": hw.head, "kv_idx": hw.kv_idx}
    row.update({
        "bq_norm": float(torch.linalg.norm(hw.bq.float())),
        "bk_norm": float(torch.linalg.norm(hw.bk.float())),
        "bv_norm": float(torch.linalg.norm(hw.bv.float())),
        "Wq_norm": float(torch.linalg.norm(hw.Wq.float())),
        "Wk_norm": float(torch.linalg.norm(hw.Wk.float())),
        "Wv_norm": float(torch.linalg.norm(hw.Wv.float())),
        "Wo_norm": float(torch.linalg.norm(hw.Wo.float())),
    })
    # VO affine split
    Cvo = hw.Wo @ hw.Wv
    bvo = hw.Wo @ hw.bv
    lin_e = norm2(Cvo)
    bias_e = norm2(bvo)
    vo_tot = max(1e-12, lin_e + bias_e)
    row.update({
        "vo_linear_frac": lin_e / vo_tot,
        "vo_write_bias_frac": bias_e / vo_tot,
        "Cvo_norm": float(torch.linalg.norm(Cvo.float())),
        "bvo_norm": float(torch.linalg.norm(bvo.float())),
    })
    if svd:
        row.update({f"VO_{k}": v for k, v in svd_rank_errors(Cvo, ranks).items()})

    for d, Rrel in Rrels.items():
        parts = affine_qk_parts_static(hw, Rrel.to(hw.Wq.device), sqrt_d)
        eB = norm2(parts["B"])
        eu = norm2(parts["u"])
        ev = norm2(parts["v"])
        ec = norm2(parts["c"])
        tot = max(1e-12, eB + eu + ev + ec)
        pref = f"qk_d{d}_"
        row[pref + "content_frac"] = eB / tot
        row[pref + "query_frac"] = eu / tot
        row[pref + "key_frac"] = ev / tot
        row[pref + "constant_frac"] = ec / tot
        row[pref + "c_value"] = float(parts["c"])
        row[pref + "B_norm"] = math.sqrt(eB)
        row[pref + "u_norm"] = math.sqrt(eu)
        row[pref + "v_norm"] = math.sqrt(ev)
        if svd and d == 0:
            # augmented matrix is expensive; B-only tells content-bilinear compressibility.
            row.update({f"QK_B_d0_{k}": v for k, v in svd_rank_errors(parts["B"], ranks).items()})
    return row


def classify_head(row: Dict[str, Any], delta: int = 0) -> str:
    c = safe_float(row.get(f"qk_d{delta}_constant_frac"))
    q = safe_float(row.get(f"qk_d{delta}_query_frac"))
    k = safe_float(row.get(f"qk_d{delta}_key_frac"))
    b = safe_float(row.get(f"qk_d{delta}_content_frac"))
    vb = safe_float(row.get("vo_write_bias_frac"))
    if c > 0.85 and b < 0.05:
        base = "positional_affine"
    elif q > 0.55:
        base = "query_affine"
    elif k > 0.45:
        base = "key_affine"
    elif b > 0.20:
        base = "content_bilinear"
    elif (k + b) > 0.50:
        base = "key_content"
    else:
        base = "mixed"
    if vb > 0.35:
        base += "+strong_vo_bias"
    elif vb > 0.15:
        base += "+vo_bias"
    return base


def subspace_basis(M: torch.Tensor, rank: int, side: str = "col") -> torch.Tensor:
    # Return orthonormal basis [dim, r]. For M [out,in]: col space is U, row/read space is V.
    M = M.float()
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        r = min(rank, U.shape[1], Vh.shape[0])
        if side == "row":
            return Vh[:r].T.contiguous()
        return U[:, :r].contiguous()
    except Exception:
        # fallback QR on transpose/normal
        X = M.T if side == "row" else M
        Q, _ = torch.linalg.qr(X[:, : min(rank, X.shape[1])])
        return Q[:, : min(rank, Q.shape[1])]


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]:
    # A/B [dim,r], orthonormal approx. mean squared canonical cosine.
    if A.numel() == 0 or B.numel() == 0:
        return {"mean_sq_cos": 0.0, "max_sq_cos": 0.0, "sum_sq_cos": 0.0}
    C = A.float().T @ B.float()
    s = torch.linalg.svdvals(C)
    ss = s * s
    return {"mean_sq_cos": float(ss.mean()), "max_sq_cos": float(ss.max()), "sum_sq_cos": float(ss.sum())}


def run_static_atlas(model: Any, specs: List[Tuple[int, int]], args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    dims = get_model_dims(model)
    deltas = parse_int_list(args.deltas)
    max_delta = max(deltas) if deltas else 0
    Rrel_all = rope_relative_mats(model, max_delta, args.device, dims["hidden"], dims["head_dim"])
    Rrels = {d: Rrel_all[d].to(args.device if args.static_device == "cuda" else "cpu") for d in deltas}
    ranks = parse_int_list(args.ranks)
    device = args.static_device if args.static_device in ("cpu", "cuda") else "cpu"

    rows: List[Dict[str, Any]] = []
    basis_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    for idx, (l, h) in enumerate(specs):
        hw = get_head_weights(model, l, h, device=device)
        row = static_head_summary(model, hw, Rrels, ranks, svd=not args.no_static_svd)
        row["head_type"] = classify_head(row, delta=deltas[0] if deltas else 0)
        rows.append(row)
        # subspace cache: write = col space Cvo, reads = row spaces Wq/Wk/Wv.
        if args.compute_transitions:
            Cvo = hw.Wo @ hw.Wv
            basis_cache[(l, h)] = {
                "write": subspace_basis(Cvo, args.subspace_rank, side="col").cpu(),
                "Wq_read": subspace_basis(hw.Wq, args.subspace_rank, side="row").cpu(),
                "Wk_read": subspace_basis(hw.Wk, args.subspace_rank, side="row").cpu(),
                "Wv_read": subspace_basis(hw.Wv, args.subspace_rank, side="row").cpu(),
            }
        if (idx + 1) % 20 == 0:
            print(f"[static] processed {idx+1}/{len(specs)} heads", flush=True)

    write_csv(out_dir / "all_heads_static_summary.csv", rows)
    transitions: List[Dict[str, Any]] = []
    if args.compute_transitions:
        for (la, ha), ba in basis_cache.items():
            for (lb, hb), bb in basis_cache.items():
                if lb <= la or (lb - la) > args.transition_window:
                    continue
                for read_name in ["Wq_read", "Wk_read", "Wv_read"]:
                    ov = subspace_overlap(ba["write"], bb[read_name])
                    transitions.append({
                        "src_layer": la, "src_head": ha, "dst_layer": lb, "dst_head": hb,
                        "read": read_name, **ov,
                    })
        transitions.sort(key=lambda r: (safe_float(r.get("mean_sq_cos")), safe_float(r.get("max_sq_cos"))), reverse=True)
        write_csv(out_dir / "top_cross_layer_transitions.csv", transitions[: args.top_transitions])

    # layer/type summary
    type_counts: Dict[str, int] = {}
    for r in rows:
        type_counts[str(r.get("head_type"))] = type_counts.get(str(r.get("head_type")), 0) + 1
    summary = {"n_heads": len(rows), "dims": dims, "type_counts": type_counts, "top_transitions": transitions[:10]}
    write_json(out_dir / "static_summary.json", summary)
    md = ["# Static attention weight atlas", "", f"heads: {len(rows)}", "", "## type counts"]
    for k, v in sorted(type_counts.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"- {k}: {v}")
    md.append("\n## top transitions")
    for t in transitions[:10]:
        md.append(f"- L{t['src_layer']}H{t['src_head']} -> L{t['dst_layer']}H{t['dst_head']} {t['read']} mean_sq={t['mean_sq_cos']:.4f} max_sq={t['max_sq_cos']:.4f}")
    (out_dir / "summary_static.md").write_text("\n".join(md), encoding="utf-8")
    return summary


# ----------------------------- runtime atlas -----------------------------

def row_entropy(A: torch.Tensor) -> torch.Tensor:
    T = A.shape[-1]
    return -(A.float() * (A.float() + 1e-12).log()).sum(dim=-1) / math.log(max(2, T))


def route_summary(A: torch.Tensor) -> Dict[str, float]:
    A = A.float()
    T = A.shape[0]
    ent = row_entropy(A)
    ar = torch.arange(T, device=A.device)
    self_mass = A[ar, ar]
    prev_mass = torch.zeros((T,), device=A.device)
    if T > 1:
        prev_mass[1:] = A[ar[1:], ar[:-1]]
    bos_mass = A[:, 0]
    local4 = []
    for i in range(T):
        lo = max(0, i - 4)
        hi = min(T, i + 5)
        local4.append(A[i, lo:hi].sum())
    local4_t = torch.stack(local4) if local4 else torch.zeros((0,), device=A.device)
    return {
        "entropy_mean": float(ent.mean()),
        "entropy_max": float(ent.max()),
        "self_mass_mean": float(self_mass.mean()),
        "prev_mass_mean": float(prev_mass.mean()),
        "bos_mass_mean": float(bos_mass.mean()),
        "local4_mean": float(local4_t.mean()) if local4 else 0.0,
        "top1_self_frac": float((A.argmax(dim=-1) == ar).float().mean()),
        "top1_bos_frac": float((A.argmax(dim=-1) == 0).float().mean()),
    }


def score_terms_from_qk(Q: torch.Tensor, K: torch.Tensor, q_bias_rot: torch.Tensor, k_bias_rot: torch.Tensor, sqrt_d: float) -> Dict[str, torch.Tensor]:
    q_lin = Q.float() - q_bias_rot.float()
    k_lin = K.float() - k_bias_rot.float()
    qb = q_bias_rot.float()
    kb = k_bias_rot.float()
    return {
        "content": (q_lin @ k_lin.T) / sqrt_d,
        "q_affine": (q_lin @ kb.T) / sqrt_d,
        "k_affine": (qb @ k_lin.T) / sqrt_d,
        "constant": (qb @ kb.T) / sqrt_d,
    }


def term_energy_fracs(terms: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, float]:
    es = {k: norm2(v[mask]) for k, v in terms.items()}
    tot = max(1e-12, sum(es.values()))
    return {f"{k}_energy_frac": es[k] / tot for k in es}


def head_runtime_for_sequence(
    Xn: torch.Tensor, q_rot_all: torch.Tensor, k_rot_all: torch.Tensor, v_all: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, attn: Any, hw: HeadWeights, dims: Dict[str, int],
    token_ids: List[int], tokens: List[str], save_examples: bool = False, example_topk: int = 4,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Xn [T,H], q_rot_all [heads,T,D], k_rot_all [kv,T,D], v_all [kv,T,D]
    T = Xn.shape[0]
    H = dims["hidden"]
    D = dims["head_dim"]
    sqrt_d = math.sqrt(D)
    device = Xn.device
    Q = q_rot_all[hw.head].float()
    K = k_rot_all[hw.kv_idx].float()
    V = v_all[hw.kv_idx].float()
    scores_true = (Q @ K.T) / sqrt_d
    A_true = causal_softmax(scores_true)
    Y_true = (A_true @ V) @ hw.Wo.to(device).T

    # Bias-only q/k after RoPE, for exact split into content/q/k/constant.
    bq = hw.bq.to(device).view(1, D).expand(T, D)
    bk = hw.bk.to(device).view(1, D).expand(T, D)
    q_bias_rot = apply_rope_rows(bq, cos[0, :T, :D].float(), sin[0, :T, :D].float())
    k_bias_rot = apply_rope_rows(bk, cos[0, :T, :D].float(), sin[0, :T, :D].float())
    terms = score_terms_from_qk(Q, K, q_bias_rot, k_bias_rot, sqrt_d)
    score_hat = terms["content"] + terms["q_affine"] + terms["k_affine"] + terms["constant"]
    A_hat = causal_softmax(score_hat)
    Y_hat = (A_hat @ V) @ hw.Wo.to(device).T
    mask = causal_mask(T, device)

    # VO split
    bv = hw.bv.to(device).view(1, D)
    V_lin = V - bv
    payload_lin = V_lin @ hw.Wo.to(device).T
    payload_bias = bv @ hw.Wo.to(device).T  # [1,H]
    Y_no_vo_bias = A_true @ payload_lin
    Y_bias_only = A_true @ payload_bias.expand(T, H)

    ablations: Dict[str, float] = {}
    variants = {
        "no_const": terms["content"] + terms["q_affine"] + terms["k_affine"],
        "no_q": terms["content"] + terms["k_affine"] + terms["constant"],
        "no_k": terms["content"] + terms["q_affine"] + terms["constant"],
        "no_content": terms["q_affine"] + terms["k_affine"] + terms["constant"],
        "only_const": terms["constant"],
        "only_content": terms["content"],
        "only_q": terms["q_affine"],
        "only_k": terms["k_affine"],
    }
    for name, S in variants.items():
        A_mod = causal_softmax(S)
        Y_mod = (A_mod @ V) @ hw.Wo.to(device).T
        ablations[f"{name}_A_rel"] = rel_err(A_mod, A_true)
        ablations[f"{name}_Y_rel"] = rel_err(Y_mod, Y_true)
    ablations["no_vo_bias_Y_rel"] = rel_err(Y_no_vo_bias, Y_true)
    ablations["vo_bias_only_Y_rel"] = rel_err(Y_bias_only, Y_true)

    route = route_summary(A_true)
    summary: Dict[str, Any] = {
        "score_rel": rel_err(score_hat[mask], scores_true[mask]),
        "A_rel": rel_err(A_hat, A_true),
        "Y_rel": rel_err(Y_hat, Y_true),
        "Y_norm": float(torch.linalg.norm(Y_true.float())),
        "T": T,
        **term_energy_fracs(terms, mask),
        "vo_linear_runtime_frac": norm2(payload_lin) / max(1e-12, norm2(payload_lin) + norm2(payload_bias.expand(T, H))),
        "vo_bias_runtime_frac": norm2(payload_bias.expand(T, H)) / max(1e-12, norm2(payload_lin) + norm2(payload_bias.expand(T, H))),
        **{f"route_{k}": v for k, v in route.items()},
        **ablations,
    }

    extra: Dict[str, Any] = {}
    if save_examples:
        examples = []
        top_rows = min(T, 16)
        for i in range(top_rows):
            row = A_true[i]
            vals, idx = torch.topk(row, k=min(example_topk, i + 1))
            pair_terms = []
            for val, j in zip(vals.detach().cpu().tolist(), idx.detach().cpu().tolist()):
                pair_terms.append({
                    "key_pos": int(j),
                    "key_token": tokens[j] if j < len(tokens) else "?",
                    "A": float(val),
                    "score_content": float(terms["content"][i, j]),
                    "score_q_affine": float(terms["q_affine"][i, j]),
                    "score_k_affine": float(terms["k_affine"][i, j]),
                    "score_constant": float(terms["constant"][i, j]),
                })
            examples.append({
                "query_pos": i,
                "query_token": tokens[i] if i < len(tokens) else "?",
                "top_reads": pair_terms,
                "Y_norm": float(torch.linalg.norm(Y_true[i].float())),
            })
        extra["token_examples"] = examples
    if save_examples or False:
        pass
    return summary, extra


def aggregate_rows(rows: List[Dict[str, Any]], keys: Optional[List[str]] = None) -> Dict[str, float]:
    if not rows:
        return {}
    if keys is None:
        keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
    out: Dict[str, float] = {}
    for k in keys:
        vals = [safe_float(r[k]) for r in rows if k in r and isinstance(r.get(k), (int, float))]
        if vals:
            out[k + "_mean"] = safe_mean(vals)
            out[k + "_max"] = max(vals)
    return out


def run_runtime_atlas(model: Any, tokenizer: Any, specs: List[Tuple[int, int]], args: argparse.Namespace, out_dir: Path,
                      collect_bank: bool = False) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    dims = get_model_dims(model)
    prompts = build_prompts(args.prompt_suites, args.prompts_per_suite, args.prompt_file)
    specs_by_layer: Dict[int, List[int]] = {}
    for l, h in specs:
        specs_by_layer.setdefault(l, []).append(h)
    for l in specs_by_layer:
        specs_by_layer[l] = sorted(set(specs_by_layer[l]))

    layers = get_layers(model)
    per_head_seq_rows: Dict[Tuple[int, int], List[Dict[str, Any]]] = {x: [] for x in specs}
    per_head_examples: Dict[str, Any] = {}
    bank_records: Optional[List[Dict[str, Any]]] = [] if collect_bank else None

    for pi, pr in enumerate(prompts):
        enc = tokenizer(pr["text"], return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"].to(args.device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(args.device)
        T = int(input_ids.shape[1])
        if T < 2:
            continue
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
        toks = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
        token_ids = input_ids[0].detach().cpu().tolist()
        pos = torch.arange(T, device=args.device).unsqueeze(0)
        for layer_idx, heads in specs_by_layer.items():
            layer = layers[layer_idx]
            attn = layer.self_attn
            Xn_model = layer.input_layernorm(outputs.hidden_states[layer_idx].detach())  # [1,T,H], keep model dtype
            q = attn.q_proj(Xn_model).view(1, T, dims["n_heads"], dims["head_dim"]).transpose(1, 2).contiguous()
            k = attn.k_proj(Xn_model).view(1, T, dims["n_kv"], dims["head_dim"]).transpose(1, 2).contiguous()
            v = attn.v_proj(Xn_model).view(1, T, dims["n_kv"], dims["head_dim"]).transpose(1, 2).contiguous()
            cos, sin = compute_position_embeddings(model, Xn_model, pos)
            q_rot, k_rot = apply_rope_qwen(q, k, cos, sin)
            X = Xn_model[0].float()
            for h in heads:
                hw = get_head_weights(model, layer_idx, h, device=args.device)
                seq_summary, extra = head_runtime_for_sequence(
                    X, q_rot[0], k_rot[0], v[0], cos, sin, attn, hw, dims,
                    token_ids=token_ids, tokens=toks,
                    save_examples=args.runtime_save_examples and len(per_head_examples) < args.max_example_heads,
                    example_topk=args.example_topk,
                )
                seq_summary.update({"prompt_id": pi, "suite": pr["suite"], "layer": layer_idx, "head": h, "text": pr["text"][:120]})
                per_head_seq_rows[(layer_idx, h)].append(seq_summary)
                if extra and f"L{layer_idx}H{h}" not in per_head_examples:
                    per_head_examples[f"L{layer_idx}H{h}"] = extra
                if collect_bank and bank_records is not None:
                    # Keep enough tensors for differentiable bank. Store on CPU to reduce GPU memory.
                    # Recompute terms here cheaply.
                    D = dims["head_dim"]
                    sqrt_d = math.sqrt(D)
                    Q = q_rot[0, h].float()
                    K = k_rot[0, hw.kv_idx].float()
                    V = v[0, hw.kv_idx].float()
                    bq = hw.bq.view(1, D).expand(T, D)
                    bk = hw.bk.view(1, D).expand(T, D)
                    q_bias_rot = apply_rope_rows(bq, cos[0, :T, :D].float(), sin[0, :T, :D].float())
                    k_bias_rot = apply_rope_rows(bk, cos[0, :T, :D].float(), sin[0, :T, :D].float())
                    terms = score_terms_from_qk(Q, K, q_bias_rot, k_bias_rot, sqrt_d)
                    A_true = causal_softmax((Q @ K.T) / sqrt_d)
                    bv = hw.bv.view(1, D)
                    V_lin = V - bv
                    payload_lin = V_lin @ hw.Wo.T
                    payload_bias = (bv @ hw.Wo.T).expand(T, dims["hidden"])
                    Y_true = A_true @ (payload_lin + payload_bias)
                    bank_records.append({
                        "prompt_id": pi, "suite": pr["suite"], "layer": layer_idx, "head": h,
                        "terms": torch.stack([terms["constant"], terms["q_affine"], terms["k_affine"], terms["content"]]).detach().cpu(),
                        "payload_lin": payload_lin.detach().cpu(),
                        "payload_bias": payload_bias.detach().cpu(),
                        "Y_true": Y_true.detach().cpu(),
                    })
        del outputs
        if torch.cuda.is_available() and (pi + 1) % 8 == 0:
            torch.cuda.empty_cache()
        print(f"[runtime] prompt {pi+1}/{len(prompts)} T={T}", flush=True)

    # aggregate per head
    agg_rows: List[Dict[str, Any]] = []
    for (l, h), rows in per_head_seq_rows.items():
        agg = aggregate_rows(rows)
        agg.update({"layer": l, "head": h, "nseq": len(rows)})
        agg_rows.append(agg)
        write_csv(out_dir / f"L{l}H{h}" / "per_sequence_runtime.csv", rows)
        write_json(out_dir / f"L{l}H{h}" / "runtime_summary.json", agg)
    write_csv(out_dir / "all_heads_runtime_summary.csv", agg_rows)
    write_json(out_dir / "runtime_examples.json", per_head_examples)

    summary = {"n_heads": len(specs), "n_prompts": len(prompts), "heads": agg_rows[:10]}
    write_json(out_dir / "runtime_summary.json", summary)
    md = ["# Runtime attention pseudocode atlas", "", f"heads: {len(specs)}", f"prompts: {len(prompts)}", "", "## top heads by exact Y error"]
    for r in sorted(agg_rows, key=lambda x: safe_float(x.get("Y_rel_mean")))[:20]:
        md.append(f"- L{r['layer']}H{r['head']} Y_rel={safe_float(r.get('Y_rel_mean')):.6f} A_rel={safe_float(r.get('A_rel_mean')):.6f} score_rel={safe_float(r.get('score_rel_mean')):.6f}")
    (out_dir / "summary_runtime.md").write_text("\n".join(md), encoding="utf-8")
    return summary, bank_records


# ----------------------------- differentiable bank -----------------------------

def group_bank_records(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_prompt: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        by_prompt.setdefault(int(r["prompt_id"]), []).append(r)
    return by_prompt


def run_bank_fit(records: List[Dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    if not records:
        return {"enabled": False, "reason": "no records"}
    by_prompt = group_bank_records(records)
    prompt_ids = sorted(by_prompt.keys())
    random.Random(args.seed).shuffle(prompt_ids)
    n_val = max(1, int(round(len(prompt_ids) * args.val_frac))) if len(prompt_ids) > 1 else 0
    val_ids = set(prompt_ids[:n_val])
    train_ids = [p for p in prompt_ids if p not in val_ids]
    val_ids_l = [p for p in prompt_ids if p in val_ids]
    head_keys = sorted({(int(r["layer"]), int(r["head"])) for r in records})
    h_index = {k: i for i, k in enumerate(head_keys)}
    nH = len(head_keys)
    device = args.device

    head_coef = torch.nn.Parameter(torch.ones((nH,), device=device))
    term_logits = torch.nn.Parameter(torch.full((nH, 4), 5.0, device=device))   # const,q,k,content
    vo_logits = torch.nn.Parameter(torch.full((nH, 2), 5.0, device=device))     # linear,bias
    opt = torch.optim.AdamW([head_coef, term_logits, vo_logits], lr=args.bank_lr, weight_decay=0.0)

    def eval_split(ids: List[int], train: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = []
        rels = []
        for pid in ids:
            recs = by_prompt[pid]
            # target is exact sum of the selected heads.
            target = sum(r["Y_true"].to(device).float() for r in recs)
            pred = torch.zeros_like(target)
            for r in recs:
                hi = h_index[(int(r["layer"]), int(r["head"]))]
                terms = r["terms"].to(device).float()        # [4,T,T]
                pl = r["payload_lin"].to(device).float()    # [T,H]
                pb = r["payload_bias"].to(device).float()   # [T,H]
                tg = torch.sigmoid(term_logits[hi]).view(4, 1, 1)
                vg = torch.sigmoid(vo_logits[hi])
                S = (tg * terms).sum(dim=0)
                A = causal_softmax(S)
                payload = vg[0] * pl + vg[1] * pb
                pred = pred + head_coef[hi] * (A @ payload)
            loss = F.mse_loss(pred, target)
            if args.sparse_lambda > 0:
                loss = loss + args.sparse_lambda * (head_coef.abs().mean() + (1 - torch.sigmoid(term_logits)).abs().mean() * 0.01)
            losses.append(loss)
            rels.append(rel_err(pred.detach(), target.detach()))
        if not losses:
            z = torch.tensor(0.0, device=device)
            return z, {"rel": 999.0, "loss": 999.0}
        loss_t = torch.stack(losses).mean()
        return loss_t, {"rel": safe_mean(rels), "loss": float(loss_t.detach().cpu())}

    history = []
    best = None
    best_val = 1e9
    bad = 0
    for step in range(1, args.bank_steps + 1):
        loss, tr = eval_split(train_ids, train=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([head_coef, term_logits, vo_logits], 5.0)
        opt.step()
        if step % args.bank_eval_every == 0 or step == args.bank_steps:
            with torch.no_grad():
                _, tr = eval_split(train_ids, train=False)
                _, va = eval_split(val_ids_l, train=False)
            item = {"step": step, "train_rel": tr["rel"], "val_rel": va["rel"], "train_loss": tr["loss"], "val_loss": va["loss"]}
            history.append(item)
            print(f"[bank] step={step} train_rel={tr['rel']:.5f} val_rel={va['rel']:.5f}", flush=True)
            if va["rel"] < best_val:
                best_val = va["rel"]
                best = (head_coef.detach().clone(), term_logits.detach().clone(), vo_logits.detach().clone(), item)
                bad = 0
            else:
                bad += 1
                if bad >= args.bank_patience:
                    break
    if best is not None:
        with torch.no_grad():
            head_coef.copy_(best[0]); term_logits.copy_(best[1]); vo_logits.copy_(best[2])

    gate_rows = []
    with torch.no_grad():
        tg = torch.sigmoid(term_logits).detach().cpu()
        vg = torch.sigmoid(vo_logits).detach().cpu()
        hc = head_coef.detach().cpu()
        for (l, h), i in h_index.items():
            gate_rows.append({
                "layer": l, "head": h, "coef": float(hc[i]),
                "gate_const": float(tg[i, 0]), "gate_q": float(tg[i, 1]),
                "gate_k": float(tg[i, 2]), "gate_content": float(tg[i, 3]),
                "gate_vo_linear": float(vg[i, 0]), "gate_vo_bias": float(vg[i, 1]),
            })
    write_csv(out_dir / "joint_bank_gates.csv", gate_rows)
    write_csv(out_dir / "joint_bank_history.csv", history)

    # quick global ablation rows by forcing gates to zero/one temporarily.
    def set_and_eval(mod: str) -> Dict[str, Any]:
        old_t = term_logits.detach().clone()
        old_v = vo_logits.detach().clone()
        with torch.no_grad():
            if mod == "no_const": term_logits[:, 0] = -20
            if mod == "no_q": term_logits[:, 1] = -20
            if mod == "no_k": term_logits[:, 2] = -20
            if mod == "no_content": term_logits[:, 3] = -20
            if mod == "only_const":
                term_logits[:] = -20; term_logits[:, 0] = 20
            if mod == "no_vo_bias": vo_logits[:, 1] = -20
            if mod == "vo_bias_only":
                vo_logits[:, 0] = -20; vo_logits[:, 1] = 20
            _, va = eval_split(val_ids_l, train=False)
            term_logits.copy_(old_t); vo_logits.copy_(old_v)
        return {"mode": mod, "val_rel": va["rel"], "val_loss": va["loss"]}
    ablations = [set_and_eval(m) for m in ["full", "no_const", "no_q", "no_k", "no_content", "only_const", "no_vo_bias", "vo_bias_only"]]
    # The "full" above did not change gates, so okay.
    write_csv(out_dir / "joint_term_ablation.csv", ablations)

    summary = {"enabled": True, "n_heads": nH, "n_prompts": len(prompt_ids), "best_val_rel": best_val, "gates": gate_rows, "ablations": ablations, "history": history}
    write_json(out_dir / "bank_summary.json", summary)
    md = ["# Differentiable multi-head bank", "", f"heads: {nH}", f"best_val_rel: {best_val:.6f}", "", "## gates"]
    for r in gate_rows:
        md.append(f"- L{r['layer']}H{r['head']} coef={r['coef']:.3f} const={r['gate_const']:.3f} q={r['gate_q']:.3f} k={r['gate_k']:.3f} content={r['gate_content']:.3f} vo_bias={r['gate_vo_bias']:.3f}")
    md.append("\n## ablations")
    for r in ablations:
        md.append(f"- {r['mode']}: val_rel={r['val_rel']:.6f}")
    (out_dir / "summary_bank.md").write_text("\n".join(md), encoding="utf-8")
    return summary


# ----------------------------- generation trace -----------------------------

def run_generation_trace(model: Any, tokenizer: Any, specs: List[Tuple[int, int]], args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    text = args.generate_prompt
    if not text:
        raise ValueError("--generate-prompt is required for generate_trace")
    dims = get_model_dims(model)
    specs_by_layer: Dict[int, List[int]] = {}
    for l, h in specs:
        specs_by_layer.setdefault(l, []).append(h)
    layers = get_layers(model)
    steps = []
    for step in range(args.generate_steps):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True, use_cache=False)
        logits = outputs.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, k=args.generate_topk)
        next_id = int(topi[0].item())
        next_tok = tokenizer.convert_ids_to_tokens([next_id])[0]
        T = int(enc["input_ids"].shape[1])
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].detach().cpu().tolist())
        pos = torch.arange(T, device=args.device).unsqueeze(0)
        head_summaries = []
        for layer_idx, heads in specs_by_layer.items():
            layer = layers[layer_idx]
            attn = layer.self_attn
            Xn_model = layer.input_layernorm(outputs.hidden_states[layer_idx].detach())
            q = attn.q_proj(Xn_model).view(1, T, dims["n_heads"], dims["head_dim"]).transpose(1, 2).contiguous()
            k = attn.k_proj(Xn_model).view(1, T, dims["n_kv"], dims["head_dim"]).transpose(1, 2).contiguous()
            v = attn.v_proj(Xn_model).view(1, T, dims["n_kv"], dims["head_dim"]).transpose(1, 2).contiguous()
            cos, sin = compute_position_embeddings(model, Xn_model, pos)
            q_rot, k_rot = apply_rope_qwen(q, k, cos, sin)
            for h in heads:
                hw = get_head_weights(model, layer_idx, h, device=args.device)
                seq_sum, extra = head_runtime_for_sequence(
                    Xn_model[0].float(), q_rot[0], k_rot[0], v[0], cos, sin, attn, hw, dims,
                    token_ids=enc["input_ids"][0].detach().cpu().tolist(), tokens=toks,
                    save_examples=False,
                )
                # focus on compressed full-seq summary; final-row route can be added later.
                head_summaries.append({
                    "layer": layer_idx, "head": h,
                    "A_rel": seq_sum["A_rel"], "Y_rel": seq_sum["Y_rel"],
                    "constant_frac": seq_sum["constant_energy_frac"],
                    "q_frac": seq_sum["q_affine_energy_frac"],
                    "k_frac": seq_sum["k_affine_energy_frac"],
                    "content_frac": seq_sum["content_energy_frac"],
                    "Y_norm": seq_sum["Y_norm"],
                    "entropy": seq_sum["route_entropy_mean"],
                    "local4": seq_sum["route_local4_mean"],
                    "bos": seq_sum["route_bos_mass_mean"],
                })
        item = {
            "step": step,
            "text_prefix": text,
            "next_id": next_id,
            "next_token": next_tok,
            "top_tokens": [{"id": int(i), "token": tokenizer.convert_ids_to_tokens([int(i)])[0], "prob": float(v)} for v, i in zip(topv.detach().cpu().tolist(), topi.detach().cpu().tolist())],
            "heads": sorted(head_summaries, key=lambda r: safe_float(r["Y_norm"]), reverse=True)[: args.generate_report_heads],
        }
        write_json(out_dir / f"step_{step:03d}.json", item)
        steps.append(item)
        # greedy append
        text = tokenizer.decode(torch.cat([enc["input_ids"][0], torch.tensor([next_id], device=args.device)]), skip_special_tokens=False)
        print(f"[gen] step={step} next={next_tok!r}", flush=True)
    summary = {"steps": steps}
    write_json(out_dir / "generation_trace_summary.json", summary)
    return summary


# ----------------------------- main -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen attention matrix-pseudocode atlas")
    p.add_argument("--mode", default="all", choices=["static", "runtime", "bank", "generate_trace", "all"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", default="0-5")
    p.add_argument("--heads", default="all")
    p.add_argument("--head-spec", default="", help="explicit L:H,L:H list; overrides --layers/--heads")
    p.add_argument("--out-dir", default="./qwen_attention_pseudocode_atlas_v1")

    # Static
    p.add_argument("--deltas", default="0,1,2,3,4,5,6,7,8,16,32,64")
    p.add_argument("--ranks", default="4,8,16,32,64")
    p.add_argument("--static-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--no-static-svd", action="store_true")
    p.add_argument("--compute-transitions", action="store_true", default=True)
    p.add_argument("--subspace-rank", type=int, default=16)
    p.add_argument("--transition-window", type=int, default=2)
    p.add_argument("--top-transitions", type=int, default=200)

    # Runtime prompts
    p.add_argument("--prompt-suites", default="all")
    p.add_argument("--prompts-per-suite", type=int, default=8)
    p.add_argument("--prompt-file", default="")
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--runtime-save-examples", action="store_true")
    p.add_argument("--max-example-heads", type=int, default=12)
    p.add_argument("--example-topk", type=int, default=4)

    # Bank
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--bank-steps", type=int, default=250)
    p.add_argument("--bank-lr", type=float, default=0.03)
    p.add_argument("--bank-eval-every", type=int, default=25)
    p.add_argument("--bank-patience", type=int, default=8)
    p.add_argument("--sparse-lambda", type=float, default=0.002)

    # Generation trace
    p.add_argument("--generate-prompt", default="")
    p.add_argument("--generate-steps", type=int, default=8)
    p.add_argument("--generate-topk", type=int, default=8)
    p.add_argument("--generate-report-heads", type=int, default=20)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    print(f"[load] model={args.model} device={args.device} dtype={args.dtype}", flush=True)
    model, tokenizer = load_model_and_tokenizer(args)
    dims = get_model_dims(model)
    specs = build_head_specs(args, dims["n_layers"] if "n_layers" in dims else len(get_layers(model)), dims["n_heads"])
    print(f"[setup] heads={len(specs)} dims={dims}", flush=True)
    write_json(out_dir / "args.json", vars(args))
    write_json(out_dir / "model_dims.json", dims)

    all_summary: Dict[str, Any] = {"dims": dims, "n_heads": len(specs)}
    if args.mode in ("static", "all"):
        sd = out_dir / "static"
        ensure_dir(sd)
        all_summary["static"] = run_static_atlas(model, specs, args, sd)
    bank_records: Optional[List[Dict[str, Any]]] = None
    if args.mode in ("runtime", "all", "bank"):
        rd = out_dir / ("runtime_bank_source" if args.mode == "bank" else "runtime")
        ensure_dir(rd)
        collect_bank = args.mode == "bank"
        runtime_summary, bank_records = run_runtime_atlas(model, tokenizer, specs, args, rd, collect_bank=collect_bank)
        all_summary["runtime"] = runtime_summary
    if args.mode == "bank":
        bd = out_dir / "bank"
        ensure_dir(bd)
        all_summary["bank"] = run_bank_fit(bank_records or [], args, bd)
    if args.mode == "generate_trace":
        gd = out_dir / "generation_trace"
        ensure_dir(gd)
        all_summary["generation_trace"] = run_generation_trace(model, tokenizer, specs, args, gd)
    write_json(out_dir / "atlas_summary.json", all_summary)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
