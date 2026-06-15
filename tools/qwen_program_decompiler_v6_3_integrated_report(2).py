#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_program_decompiler_v6_scorehybrid.py

Architecture-guided decompiler for ONE OR MORE concrete Qwen2 attention heads.

Goal:
  Do NOT collapse a Qwen head to one matrix.
  Instead follow the real Qwen2 code path:

    residual hidden_states
      -> layer.input_layernorm                 (dynamic normalization; treated as input-prep)
      -> q_proj/k_proj/v_proj                  (learned affine matrices)
      -> split Q heads / KV heads              (GQA structure)
      -> RoPE(q,k)                             (known position-dependent operator)
      -> QK scores + causal softmax            (dynamic read/router)
      -> A @ V                                 (read values)
      -> o_proj slice                          (write to residual)

Then it searches:
  1. Global QK low-rank program for a head:
       A_hat = softmax((Q_rope @ Pq) @ (K_rope @ Pk).T / sqrt(rank_or_head_dim))
  2. Attention-row routes/clusters:
       self / prev / BOS / local / diffuse / mixed
  3. Optional route-specific QK bases:
       fit Pq/Pk using only query rows belonging to each route cluster.
  4. Head-output error with original V/O:
       Y_true = (A_true @ V) @ O_head.T
       Y_hat  = (A_hat  @ V) @ O_head.T

This script uses real activations from real prompts.
V6-scorehybrid: v5 rich operators + flat-DAG forest + score/logit-level operator decomposition before softmax.
It outputs JSON/CSV/Markdown program IR that can later feed a formula/block decoder.

Example:
  python qwen_single_head_program_decompiler_v2.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda --dtype fp16 --attn-implementation eager \
    --heads 23:1,0:1,23:11 \
    --prompt-suites all --prompts-per-suite 24 --max-length 384 \
    --ranks 4,8,16,32,64 \
    --qk-methods pca,learned \
    --qk-scale rank \
    --fit-route-qk \
    --out-dir ./qwen_single_head_decompile_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    KMeans = None
    silhouette_score = None
    StandardScaler = None
    LogisticRegression = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rel_err(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> float:
    A = A.float()
    B = B.float()
    return float(torch.linalg.norm(A - B) / torch.linalg.norm(B).clamp_min(eps))


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def json_sanitize(obj: Any, _depth: int = 0) -> Any:
    """Make arbitrary result objects JSON-safe.

    Fixes tuple keys, torch tensors, numpy scalars/arrays, argparse private cache fields,
    and any remaining non-serializable objects. Designed for summaries, not binary factors.
    """
    if _depth > 20:
        return str(type(obj).__name__)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        # JSON cannot represent nan/inf reliably; stringify them.
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        if obj.numel() <= 16:
            return obj.detach().cpu().tolist()
        return {"__tensor__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if np is not None:
        if isinstance(obj, np.generic):
            return json_sanitize(obj.item(), _depth + 1)
        if isinstance(obj, np.ndarray):
            if obj.size <= 16:
                return obj.tolist()
            return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                ks = "|".join(map(str, k))
            elif isinstance(k, (str, int, float, bool)) or k is None:
                ks = str(k) if not isinstance(k, str) else k
            else:
                ks = str(k)
            out[ks] = json_sanitize(v, _depth + 1)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [json_sanitize(x, _depth + 1) for x in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, argparse.Namespace):
        return json_sanitize(vars(obj), _depth + 1)
    return str(obj)


def json_dumps_safe(obj: Any, **kwargs: Any) -> str:
    return json.dumps(json_sanitize(obj), **kwargs)


def public_args_dict(args: argparse.Namespace) -> Dict[str, Any]:
    # Do not serialize huge precollected tensors/head_data stored on args.
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


def parse_list_ints(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.replace(';', ',').split(',') if x.strip()]




def short_hash_obj(obj: Any) -> str:
    """Stable short hash for cache keys."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def mask_signature(row_masks: Optional[Dict[int, torch.Tensor]]) -> Any:
    if not row_masks:
        return None
    out = []
    for pid in sorted(row_masks.keys()):
        m = row_masks[pid]
        if m is None:
            out.append((int(pid), 0, 0))
            continue
        mc = m.detach().cpu().bool()
        idx = torch.nonzero(mc, as_tuple=False).flatten()
        # enough to disambiguate route assignment without hashing huge tensors
        first = idx[:16].tolist()
        last = idx[-16:].tolist() if idx.numel() else []
        out.append((int(pid), int(mc.numel()), int(mc.sum().item()), first, last))
    return out


def seq_signature(seqs: List[SeqHeadData]) -> Any:
    out = []
    for s in seqs:
        out.append({
            "pid": int(s.prompt_id),
            "suite": s.suite,
            "tok": s.token_ids,
            "T": int(s.A.shape[0]),
            # A tiny numeric fingerprint catches accidental mismatched data/logic.
            "a_sum": round(float(s.A.float().sum().item()), 6),
            "q_norm": round(float(torch.linalg.norm(s.Q.float()).item()), 6),
            "k_norm": round(float(torch.linalg.norm(s.K.float()).item()), 6),
        })
    return out


def qk_cache_path(args: argparse.Namespace, layer_idx: int, head_idx: int, scope: str, method: str, rank: int,
                  train: List[SeqHeadData], val: List[SeqHeadData],
                  row_masks_train: Optional[Dict[int, torch.Tensor]], row_masks_val: Optional[Dict[int, torch.Tensor]],
                  steps: int) -> Optional[Path]:
    cache_dir = str(getattr(args, "cache_dir", "") or "").strip()
    if not cache_dir:
        return None
    key_obj = {
        "cache_version": getattr(args, "cache_version", "v1"),
        "model": getattr(args, "model", ""),
        "layer": int(layer_idx),
        "head": int(head_idx),
        "scope": scope,
        "method": method,
        "rank": int(rank),
        "scale": getattr(args, "qk_scale", "rank"),
        "steps": int(steps),
        "lr": float(getattr(args, "qk_lr", 0.0)),
        "wd": float(getattr(args, "qk_wd", 0.0)),
        "patience": int(getattr(args, "qk_patience", 0)),
        "eval_every": int(getattr(args, "qk_eval_every", 0)),
        "train_batch_size": int(getattr(args, "train_batch_size", 0)),
        "seed": int(getattr(args, "seed", 0)),
        "train": seq_signature(train),
        "val": seq_signature(val),
        "mask_train": mask_signature(row_masks_train),
        "mask_val": mask_signature(row_masks_val),
    }
    h = short_hash_obj(key_obj)
    d = Path(cache_dir) / f"L{layer_idx}H{head_idx}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{scope}_{method}_r{rank}_{h}.pt"


def load_qk_cache(path: Optional[Path], device: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
    if path is None or not path.exists():
        return None
    obj = torch.load(path, map_location="cpu")
    Pq = obj["Pq"].float().to(device)
    Pk = obj["Pk"].float().to(device)
    ev = obj.get("ev", {})
    print(f"    [cache hit] {path}", flush=True)
    return Pq, Pk, ev


def save_qk_cache(path: Optional[Path], Pq: torch.Tensor, Pk: torch.Tensor, ev: Dict[str, float]) -> None:
    if path is None:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"Pq": Pq.detach().cpu(), "Pk": Pk.detach().cpu(), "ev": ev}, tmp)
    tmp.replace(path)
    print(f"    [cache save] {path}", flush=True)


def parse_heads(s: str, n_layers: int, n_heads: int) -> List[Tuple[int, int]]:
    if s.strip().lower() == "all":
        return [(l, h) for l in range(n_layers) for h in range(n_heads)]
    out: List[Tuple[int, int]] = []
    for part in s.replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            raise ValueError(f"bad head spec {part!r}; use L:H or all")
        a, b = part.split(':', 1)
        l = int(a)
        h = int(b)
        if not (0 <= l < n_layers and 0 <= h < n_heads):
            raise ValueError(f"head out of range: {l}:{h}, model has layers={n_layers}, heads={n_heads}")
        out.append((l, h))
    # stable unique
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


# ---------------- prompts ----------------

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
        "Write SQL to select users where age is greater than 18.",
        "Refactor this code to avoid repeated conditionals.",
    ],
    "math": [
        "Solve: if 3x + 5 = 20, what is x?",
        "Explain the derivative of sin(x) * exp(x).",
        "A triangle has sides 3, 4, 5. What is its area?",
        "Compute the determinant of a 2 by 2 matrix [[a,b],[c,d]].",
        "What is the quadratic formula and when is it used?",
        "Simplify: (x + 2)^2 - x^2.",
        "Explain Bayes theorem with a simple example.",
        "If a sequence doubles every step, what is the 10th term starting from 1?",
        "What is the integral of 2x from 0 to 3?",
        "Show why the sum of angles in a triangle is 180 degrees.",
    ],
    "text": [
        "Summarize why rivers are important for cities.",
        "Write a short paragraph about autumn forests.",
        "Explain the difference between a planet and a star.",
        "Describe how plants grow from seeds.",
        "Give a short explanation of why sleep matters.",
        "Write a calm paragraph about ocean waves.",
        "Explain what makes a good story opening.",
        "Summarize the benefits of learning a second language.",
        "Describe a city during a rainy evening.",
        "Explain the difference between weather and climate.",
    ],
    "dialogue": [
        "User: I am confused about fractions. Assistant:",
        "Question: Why is the sky blue? Answer:",
        "Continue the conversation politely: Hi, can you help me plan my study schedule?",
        "User: Can you explain this more simply? Assistant:",
        "Question: What should I do before an exam? Answer:",
        "User: I need a quick summary. Assistant:",
        "Continue: Thanks for the help! Assistant:",
        "Question: How do I organize my notes? Answer:",
    ],
    "symbols": [
        "(((( a + b )))) == [[x, y], [z, w]]",
        "JSON: {\"name\": \"Alice\", \"score\": 42, \"ok\": true}",
        "Sequence: A1, B2, C3, D4, E5, ...",
        "<div class=\"box\">Hello <span>world</span></div>",
        "Path: /home/user/project/src/main.py --flag=true",
        "Regex: ^[A-Za-z0-9_]+@[A-Za-z]+\\.com$",
        "Array: [1, 1, 2, 3, 5, 8, 13, 21]",
        "Equation: f(x)=sin(x)+cos(2x)-log(x+1)",
    ],
    "repeat": [
        "abc def abc def abc def abc def abc def",
        "the cat sat the cat sat the cat sat the cat sat",
        "one two three one two three one two three",
        "function call function call function call function call",
        "A B C D A B C D A B C D A B C D",
        "x = y + z; x = y + z; x = y + z;",
        "hello world hello world hello world hello world",
        "red blue green red blue green red blue green",
    ],
    "controlled": [
        "SAME_TEXT: The quick brown fox jumps over the lazy dog.",
        "CODE_PREFIX: The quick brown fox jumps over the lazy dog.",
        "MATH_PREFIX: The quick brown fox jumps over the lazy dog.",
        "JSON_PREFIX: The quick brown fox jumps over the lazy dog.",
        "Question: The quick brown fox jumps over the lazy dog. Answer:",
    ],
}


def build_prompts(suites: str, prompts_per_suite: int, same_text_repeats: int) -> List[Dict[str, str]]:
    if suites.strip().lower() == "all":
        suite_names = list(PROMPTS.keys())
    else:
        suite_names = [x.strip() for x in suites.replace(';', ',').split(',') if x.strip()]
    rows: List[Dict[str, str]] = []
    for suite in suite_names:
        base = PROMPTS.get(suite)
        if not base:
            raise ValueError(f"unknown prompt suite {suite!r}; options={list(PROMPTS)}")
        for i in range(prompts_per_suite):
            rows.append({"suite": suite, "text": base[i % len(base)]})
    if same_text_repeats > 0:
        t = "The same calibration sentence repeated to check deterministic head behavior."
        for i in range(same_text_repeats):
            rows.append({"suite": "same_repeat", "text": t})
    return rows


# ---------------- model helpers ----------------


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


def apply_rope_qwen(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q/k: [B,H,T,D]; cos/sin usually [B,T,D] or [T,D]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)  # [B,1,T,D]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    # scores [T,T]
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1)


def get_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find decoder layers on model")


def compute_position_embeddings(model: Any, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    # Qwen2 models often expose model.model.rotary_emb(hidden_states, position_ids)
    rotary = getattr(model.model, "rotary_emb", None) if hasattr(model, "model") else None
    if rotary is None:
        raise RuntimeError("model.model.rotary_emb not found")
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        # older/newer variants may ignore hidden_states
        return rotary(position_ids)


@dataclass
class SeqHeadData:
    prompt_id: int
    suite: str
    text: str
    token_ids: List[int]
    tokens: List[str]
    X: torch.Tensor      # [T,H] post RMSNorm attention input
    Q: torch.Tensor      # [T,D] after RoPE for selected head
    K: torch.Tensor      # [T,D] after RoPE for selected KV head
    V: torch.Tensor      # [T,D] selected KV value
    A: torch.Tensor      # [T,T] true attention from QK
    Y: torch.Tensor      # [T,H] selected head contribution using original O


@torch.no_grad()
def collect_head_data(model: Any, tokenizer: Any, prompts: List[Dict[str, str]], layer_idx: int, head_idx: int,
                      max_length: int, device: str) -> List[SeqHeadData]:
    layers = get_layers(model)
    layer = layers[layer_idx]
    attn = layer.self_attn
    cfg = model.config
    hidden_size = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
    kv_groups = num_heads // num_kv
    kv_idx = head_idx // kv_groups

    out: List[SeqHeadData] = []
    for pi, pr in enumerate(prompts):
        enc = tokenizer(pr["text"], return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        T = int(input_ids.shape[1])
        if T < 2:
            continue
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
        # hidden_states[0] embedding; hidden_states[layer_idx] = residual input to this layer
        residual_in = outputs.hidden_states[layer_idx].detach()  # [1,T,H]
        Xn = layer.input_layernorm(residual_in).detach()         # [1,T,H]

        # q/k/v exactly as attention forward, without relying on output_attentions
        q = attn.q_proj(Xn).view(1, T, num_heads, head_dim).transpose(1, 2).contiguous()
        k = attn.k_proj(Xn).view(1, T, num_kv, head_dim).transpose(1, 2).contiguous()
        v = attn.v_proj(Xn).view(1, T, num_kv, head_dim).transpose(1, 2).contiguous()
        pos = torch.arange(T, device=device).unsqueeze(0)
        cos, sin = compute_position_embeddings(model, Xn, pos)
        q_rot, k_rot = apply_rope_qwen(q, k, cos, sin)
        Qh = q_rot[0, head_idx].float().detach()
        Kh = k_rot[0, kv_idx].float().detach()
        Vh = v[0, kv_idx].float().detach()
        scores = (Qh @ Kh.T) / math.sqrt(head_dim)
        A = causal_softmax(scores.float()).detach()
        o_w = attn.o_proj.weight.detach().float()[:, head_idx * head_dim : (head_idx + 1) * head_dim]
        Y = (A @ Vh) @ o_w.T
        tok_ids = input_ids[0].detach().cpu().tolist()
        toks = tokenizer.convert_ids_to_tokens(tok_ids)
        out.append(SeqHeadData(
            prompt_id=pi, suite=pr["suite"], text=pr["text"], token_ids=tok_ids, tokens=toks,
            X=Xn[0].float().detach().cpu(), Q=Qh.cpu(), K=Kh.cpu(), V=Vh.cpu(), A=A.cpu(), Y=Y.cpu(),
        ))
    return out


@torch.no_grad()
def collect_requested_heads_data(model: Any, tokenizer: Any, prompts: List[Dict[str, str]],
                                 head_specs: List[Tuple[int, int]], max_length: int,
                                 device: str) -> Dict[Tuple[int, int], List[SeqHeadData]]:
    """Fast collector for many requested heads: one full model forward per prompt, not per head.

    It still follows the exact Qwen head path:
      hidden_states[layer] -> input_layernorm -> q/k/v -> RoPE -> QK -> A@V -> O slice.
    """
    cfg = model.config
    hidden_size = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
    kv_groups = num_heads // num_kv
    layers = get_layers(model)

    by_layer: Dict[int, List[int]] = {}
    for l, h in head_specs:
        by_layer.setdefault(int(l), []).append(int(h))
    for l in by_layer:
        by_layer[l] = sorted(set(by_layer[l]))

    out: Dict[Tuple[int, int], List[SeqHeadData]] = {(int(l), int(h)): [] for l, h in head_specs}

    for pi, pr in enumerate(prompts):
        enc = tokenizer(pr["text"], return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        T = int(input_ids.shape[1])
        if T < 2:
            continue

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
        tok_ids = input_ids[0].detach().cpu().tolist()
        toks = tokenizer.convert_ids_to_tokens(tok_ids)
        pos = torch.arange(T, device=device).unsqueeze(0)

        for layer_idx, heads in by_layer.items():
            layer = layers[layer_idx]
            attn = layer.self_attn
            residual_in = outputs.hidden_states[layer_idx].detach()
            Xn = layer.input_layernorm(residual_in).detach()

            q = attn.q_proj(Xn).view(1, T, num_heads, head_dim).transpose(1, 2).contiguous()
            k = attn.k_proj(Xn).view(1, T, num_kv, head_dim).transpose(1, 2).contiguous()
            v = attn.v_proj(Xn).view(1, T, num_kv, head_dim).transpose(1, 2).contiguous()
            cos, sin = compute_position_embeddings(model, Xn, pos)
            q_rot, k_rot = apply_rope_qwen(q, k, cos, sin)
            o_weight = attn.o_proj.weight.detach().float()
            X_cpu = Xn[0].float().detach().cpu()

            for head_idx in heads:
                kv_idx = int(head_idx) // kv_groups
                Qh = q_rot[0, head_idx].float().detach()
                Kh = k_rot[0, kv_idx].float().detach()
                Vh = v[0, kv_idx].float().detach()
                A = causal_softmax((Qh @ Kh.T) / math.sqrt(head_dim)).detach()
                o_w = o_weight[:, head_idx * head_dim : (head_idx + 1) * head_dim].to(device)
                Y = (A @ Vh) @ o_w.T
                out[(layer_idx, head_idx)].append(SeqHeadData(
                    prompt_id=pi, suite=pr["suite"], text=pr["text"], token_ids=tok_ids, tokens=toks,
                    X=X_cpu, Q=Qh.cpu(), K=Kh.cpu(), V=Vh.cpu(), A=A.cpu(), Y=Y.cpu(),
                ))

        del outputs
        if torch.cuda.is_available() and (pi + 1) % 16 == 0:
            torch.cuda.empty_cache()

    return out


# ---------------- QK basis ----------------


def pca_shared_init(train: List[SeqHeadData], rank: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    M = torch.cat([s.Q for s in train] + [s.K for s in train], dim=0).float().to(device)
    # center? no: QK dot product geometry depends on origin; keep raw but SVD finds energy dirs
    _, _, Vh = torch.linalg.svd(M, full_matrices=False)
    P = Vh[:rank].T.contiguous()
    return P.clone(), P.clone()


def _prepare_qk_batches(seqs: List[SeqHeadData], device: str, batch_size: int = 16,
                         row_masks: Optional[Dict[int, torch.Tensor]] = None,
                         include_vy: bool = True) -> List[Dict[str, torch.Tensor]]:
    """Pack variable-length sequences into padded GPU batches.

    This is the main speed fix versus v4: instead of doing hundreds of tiny
    QK/softmax operations per optimization step, we do batched bmm over a
    padded [B,T,D] tensor. Padding is masked out, and optional row masks select
    route-specific query rows.
    """
    if not seqs:
        return []
    bs = max(1, int(batch_size))
    out: List[Dict[str, torch.Tensor]] = []
    for off in range(0, len(seqs), bs):
        chunk = seqs[off:off + bs]
        B = len(chunk)
        Tmax = max(int(x.Q.shape[0]) for x in chunk)
        D = int(chunk[0].Q.shape[1])
        H = int(chunk[0].Y.shape[1]) if include_vy else 0
        Q = torch.zeros((B, Tmax, D), device=device, dtype=torch.float32)
        K = torch.zeros((B, Tmax, D), device=device, dtype=torch.float32)
        A = torch.zeros((B, Tmax, Tmax), device=device, dtype=torch.float32)
        row_sel = torch.zeros((B, Tmax), device=device, dtype=torch.bool)
        lengths = torch.zeros((B,), device=device, dtype=torch.long)
        if include_vy:
            V = torch.zeros((B, Tmax, D), device=device, dtype=torch.float32)
            Y = torch.zeros((B, Tmax, H), device=device, dtype=torch.float32)
        else:
            V = Y = None
        for b, seq in enumerate(chunk):
            T = int(seq.Q.shape[0])
            lengths[b] = T
            Q[b, :T] = seq.Q.to(device=device, dtype=torch.float32)
            K[b, :T] = seq.K.to(device=device, dtype=torch.float32)
            A[b, :T, :T] = seq.A.to(device=device, dtype=torch.float32)
            if include_vy:
                V[b, :T] = seq.V.to(device=device, dtype=torch.float32)
                Y[b, :T] = seq.Y.to(device=device, dtype=torch.float32)
            if row_masks is None:
                row_sel[b, :T] = True
            else:
                m = row_masks.get(seq.prompt_id)
                if m is not None:
                    mt = m.to(device=device, dtype=torch.bool)[:T]
                    row_sel[b, :mt.numel()] = mt
        ar = torch.arange(Tmax, device=device)
        valid_key = ar[None, None, :] < lengths[:, None, None]
        causal = ar[None, :, None] >= ar[None, None, :]
        attn_mask = valid_key & causal
        item = {"Q": Q, "K": K, "A": A, "row_sel": row_sel, "attn_mask": attn_mask, "lengths": lengths}
        if include_vy:
            item["V"] = V
            item["Y"] = Y
        out.append(item)
    return out


def _qk_scores_batched(batch: Dict[str, torch.Tensor], Pq: torch.Tensor, Pk: torch.Tensor, denom: float) -> torch.Tensor:
    Ql = batch["Q"] @ Pq.float()
    Kl = batch["K"] @ Pk.float()
    scores = torch.bmm(Ql, Kl.transpose(1, 2)) / denom
    scores = scores.masked_fill(~batch["attn_mask"], -1e9)
    return torch.softmax(scores, dim=-1)


def _eval_qk_batches(batches: List[Dict[str, torch.Tensor]], Pq: torch.Tensor, Pk: torch.Tensor,
                      scale_mode: str, o_w: Optional[torch.Tensor] = None) -> Dict[str, float]:
    if not batches:
        return {"rel": 999.0, "kl": 999.0, "top1": 0.0, "z_rel": 999.0, "y_rel": 999.0, "nseq": 0}
    device = Pq.device
    rank = int(Pq.shape[1])
    denom = math.sqrt(rank if scale_mode == "rank" else int(Pq.shape[0]))
    num_a = den_a = 0.0
    num_z = den_z = 0.0
    num_y = den_y = 0.0
    kl_sum = 0.0
    n_rows = 0
    top_ok = 0
    for bd in batches:
        A_true = bd["A"].to(device)
        sel = bd["row_sel"].to(device)
        if int(sel.sum()) == 0:
            continue
        A_hat = _qk_scores_batched(bd, Pq, Pk, denom)
        Ah = A_hat[sel]
        At = A_true[sel]
        diff = (Ah - At).float()
        num_a += float((diff * diff).sum().detach().cpu())
        den_a += float((At * At).sum().detach().cpu())
        kl_sum += float(F.kl_div((Ah + 1e-12).log(), At, reduction="sum").detach().cpu())
        n = int(Ah.shape[0])
        n_rows += n
        top_ok += int((Ah.argmax(dim=-1) == At.argmax(dim=-1)).sum().detach().cpu())
        if "V" in bd:
            V = bd["V"].to(device)
            Z_hat = torch.bmm(A_hat, V)
            Z_true = torch.bmm(A_true, V)
            Zh = Z_hat[sel]
            Zt = Z_true[sel]
            dz = (Zh - Zt).float()
            num_z += float((dz * dz).sum().detach().cpu())
            den_z += float((Zt * Zt).sum().detach().cpu())
            if o_w is not None and "Y" in bd:
                Ow = o_w.to(device=device, dtype=torch.float32)
                Y_hat = torch.matmul(Z_hat, Ow.T)
                Y_true = bd["Y"].to(device)
                Yh = Y_hat[sel]
                Yt = Y_true[sel]
                dy = (Yh - Yt).float()
                num_y += float((dy * dy).sum().detach().cpu())
                den_y += float((Yt * Yt).sum().detach().cpu())
    eps = 1e-12
    return {
        "rel": math.sqrt(num_a / max(eps, den_a)),
        "kl": kl_sum / max(1, n_rows),
        "top1": float(top_ok / max(1, n_rows)),
        "z_rel": math.sqrt(num_z / max(eps, den_z)) if den_z > 0 else -1.0,
        "y_rel": math.sqrt(num_y / max(eps, den_y)) if den_y > 0 else -1.0,
        "nseq": len(batches),
        "nrows": n_rows,
    }


def eval_qk_basis(seqs: List[SeqHeadData], Pq: torch.Tensor, Pk: torch.Tensor, scale_mode: str,
                  row_masks: Optional[Dict[int, torch.Tensor]] = None,
                  o_w: Optional[torch.Tensor] = None,
                  batch_size: int = 16) -> Dict[str, float]:
    device = str(Pq.device)
    batches = _prepare_qk_batches(seqs, device, batch_size=batch_size, row_masks=row_masks, include_vy=True)
    return _eval_qk_batches(batches, Pq.float(), Pk.float(), scale_mode, o_w=o_w)


def train_qk_learned(train: List[SeqHeadData], val: List[SeqHeadData], Pq0: torch.Tensor, Pk0: torch.Tensor,
                     scale_mode: str, steps: int, lr: float, wd: float, patience: int,
                     row_masks_train: Optional[Dict[int, torch.Tensor]],
                     row_masks_val: Optional[Dict[int, torch.Tensor]],
                     device: str,
                     o_w: Optional[torch.Tensor] = None,
                     batch_size: int = 16,
                     eval_every: int = 25) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Batched learned QK projection training.

    v4 did one tiny softmax per sequence per step. On P40 that leaves the GPU idle.
    This version pads sequences into batches and uses batched matmul/softmax, cutting
    Python overhead heavily while preserving the same objective.
    """
    Pq = torch.nn.Parameter(Pq0.float().to(device).clone())
    Pk = torch.nn.Parameter(Pk0.float().to(device).clone())
    opt = torch.optim.AdamW([Pq, Pk], lr=lr, weight_decay=wd)
    rank = int(Pq.shape[1])
    denom = math.sqrt(rank if scale_mode == "rank" else int(Pq.shape[0]))
    train_batches = _prepare_qk_batches(train, device, batch_size=batch_size, row_masks=row_masks_train, include_vy=False)
    val_batches = _prepare_qk_batches(val, device, batch_size=batch_size, row_masks=row_masks_val, include_vy=True)
    if not train_batches:
        ev = {"rel": 999.0, "kl": 999.0, "top1": 0.0, "z_rel": 999.0, "y_rel": 999.0, "nseq": 0}
        return Pq0.to(device), Pk0.to(device), ev
    best = None
    best_val = 1e9
    bad = 0
    every = max(1, int(eval_every))
    for step in range(1, int(steps) + 1):
        loss_sum = None
        n_rows = 0
        for bd in train_batches:
            sel = bd["row_sel"]
            if int(sel.sum()) == 0:
                continue
            A_hat = _qk_scores_batched(bd, Pq, Pk, denom)
            Ah = A_hat[sel]
            At = bd["A"][sel]
            loss_b = F.kl_div((Ah + 1e-12).log(), At, reduction="sum")
            loss_sum = loss_b if loss_sum is None else loss_sum + loss_b
            n_rows += int(Ah.shape[0])
        if loss_sum is None or n_rows == 0:
            break
        loss = loss_sum / max(1, n_rows)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Pq, Pk], 1.0)
        opt.step()
        if step % every == 0 or step == int(steps):
            ev = _eval_qk_batches(val_batches, Pq.detach(), Pk.detach(), scale_mode, o_w=o_w)
            score = float(ev["rel"])
            if score < best_val:
                best_val = score
                best = (Pq.detach().clone(), Pk.detach().clone(), ev)
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best is None:
        ev = _eval_qk_batches(val_batches, Pq.detach(), Pk.detach(), scale_mode, o_w=o_w)
        return Pq.detach(), Pk.detach(), ev
    return best[0], best[1], best[2]


def verdict_qk(rel: float, comp: float, accept: float, risky: float, explain: float, min_comp: float) -> str:
    if rel <= accept and comp >= min_comp:
        return "QK_DEPLOY"
    if rel <= risky and comp >= min_comp:
        return "QK_RISKY"
    if rel <= explain:
        return "QK_EXPLAIN"
    return "REJECT_QK"


# ---------------- row routes ----------------


def local_mass(row: torch.Tensor, i: int, radius: int) -> float:
    T = row.shape[0]
    lo = max(0, i - radius)
    hi = min(T, i + radius + 1)
    return float(row[lo:hi].sum())


def row_feature(s: SeqHeadData, i: int) -> Dict[str, Any]:
    row = s.A[i].float()
    T = row.shape[0]
    ent = float(-(row * (row + 1e-12).log()).sum() / math.log(max(2, T)))
    top = int(row.argmax())
    return {
        "prompt_id": s.prompt_id,
        "suite": s.suite,
        "pos": i,
        "T": T,
        "token": s.tokens[i] if i < len(s.tokens) else "?",
        "entropy": ent,
        "max_prob": float(row[top]),
        "top_idx": top,
        "top_dist": float(i - top),
        "top_rel": float(top / max(1, T - 1)),
        "pos_rel": float(i / max(1, T - 1)),
        "self_mass": float(row[i]),
        "prev_mass": float(row[i - 1]) if i > 0 else 0.0,
        "bos_mass": float(row[0]),
        "local1": local_mass(row, i, 1),
        "local2": local_mass(row, i, 2),
        "local4": local_mass(row, i, 4),
        "local8": local_mass(row, i, 8),
    }


def build_row_features(seqs: List[SeqHeadData]) -> List[Dict[str, Any]]:
    rows = []
    for s in seqs:
        for i in range(s.A.shape[0]):
            rows.append(row_feature(s, i))
    return rows


def infer_route_type(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "empty"
    m = {k: safe_mean([float(r[k]) for r in rows]) for k in ["entropy", "max_prob", "self_mass", "prev_mass", "bos_mass", "local2", "local4", "local8"]}
    if m["self_mass"] > 0.70:
        return "self"
    if m["prev_mass"] > 0.50:
        return "prev"
    if m["bos_mass"] > 0.50:
        return "bos"
    if m["local2"] > 0.70 or m["local4"] > 0.82:
        return "local"
    if m["entropy"] > 0.72 and m["max_prob"] < 0.25:
        return "diffuse"
    return "mixed"


def cluster_rows(rows: List[Dict[str, Any]], cluster_list: List[int], seed: int) -> Tuple[List[int], Dict[str, Any]]:
    if not rows:
        return [], {"k": 0, "silhouette": 0.0, "reason": "no rows"}
    feat_keys = ["entropy", "max_prob", "top_dist", "top_rel", "pos_rel", "self_mass", "prev_mass", "bos_mass", "local1", "local2", "local4", "local8"]
    X = [[float(r[k]) for k in feat_keys] for r in rows]
    if np is None or KMeans is None or StandardScaler is None:
        # crude fallback: self/local/diffuse thresholds as pseudo labels
        labs = []
        for r in rows:
            rt = infer_route_type([r])
            labs.append({"self": 0, "prev": 1, "bos": 2, "local": 3, "diffuse": 4}.get(rt, 5))
        return labs, {"k": len(set(labs)), "silhouette": 0.0, "reason": "sklearn unavailable fallback"}
    Xn = np.asarray(X, dtype="float32")
    Xs = StandardScaler().fit_transform(Xn)
    best = None
    best_labels = None
    for k in cluster_list:
        if k < 2 or k >= len(rows):
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(Xs)
        try:
            sil = float(silhouette_score(Xs, labels)) if len(set(labels)) > 1 else 0.0
        except Exception:
            sil = 0.0
        # prefer decent silhouette but avoid too many tiny clusters with a mild penalty
        sizes = [int((labels == c).sum()) for c in range(k)]
        tiny_frac = sum(1 for z in sizes if z < 20) / max(1, k)
        score = sil - 0.03 * k - 0.2 * tiny_frac
        if best is None or score > best[0]:
            best = (score, sil, k, sizes)
            best_labels = labels.tolist()
    if best_labels is None:
        return [0 for _ in rows], {"k": 1, "silhouette": 0.0, "reason": "no valid k"}
    return best_labels, {"k": best[2], "silhouette": best[1], "sizes": best[3], "score": best[0]}



ROUTE_FEAT_KEYS = ["entropy", "max_prob", "top_dist", "top_rel", "pos_rel", "self_mass", "prev_mass", "bos_mass", "local1", "local2", "local4", "local8"]


def fit_row_clusterer(rows: List[Dict[str, Any]], cluster_list: List[int], seed: int):
    """Fit route clustering on TRAIN rows only. Returns labels, info, and fitted state for val assignment."""
    if not rows:
        return [], {"k": 0, "silhouette": 0.0, "reason": "no rows"}, None
    if np is None or KMeans is None or StandardScaler is None:
        labels, info = cluster_rows(rows, cluster_list, seed)
        return labels, info, None
    X = np.asarray([[float(r[k]) for k in ROUTE_FEAT_KEYS] for r in rows], dtype="float32")
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    best = None
    best_labels = None
    best_km = None
    for k in cluster_list:
        if k < 2 or k >= len(rows):
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(Xs)
        try:
            sil = float(silhouette_score(Xs, labels)) if len(set(labels)) > 1 else 0.0
        except Exception:
            sil = 0.0
        sizes = [int((labels == c).sum()) for c in range(k)]
        tiny_frac = sum(1 for z in sizes if z < 20) / max(1, k)
        score = sil - 0.03 * k - 0.2 * tiny_frac
        if best is None or score > best[0]:
            best = (score, sil, k, sizes)
            best_labels = labels.tolist()
            best_km = km
    if best_labels is None:
        return [0 for _ in rows], {"k": 1, "silhouette": 0.0, "reason": "no valid k"}, None
    state = {"scaler": sc, "kmeans": best_km, "feat_keys": ROUTE_FEAT_KEYS}
    return best_labels, {"k": best[2], "silhouette": best[1], "sizes": best[3], "score": best[0], "fit_on": "train"}, state


def assign_row_clusters(rows: List[Dict[str, Any]], state: Any) -> List[int]:
    """Assign VAL/ALL rows through the TRAIN-fitted scaler+kmeans. Falls back to threshold labels."""
    if not rows:
        return []
    if state is None or np is None:
        labs = []
        for r in rows:
            rt = infer_route_type([r])
            labs.append({"self": 0, "prev": 1, "bos": 2, "local": 3, "diffuse": 4}.get(rt, 5))
        return labs
    X = np.asarray([[float(r[k]) for k in state["feat_keys"]] for r in rows], dtype="float32")
    return state["kmeans"].predict(state["scaler"].transform(X)).tolist()


def seq_labels_from_row_labels(seqs: List[SeqHeadData], rows: List[Dict[str, Any]], labels: List[int]) -> Dict[int, torch.Tensor]:
    by_prompt: Dict[int, List[Tuple[int, int]]] = {}
    for r, lab in zip(rows, labels):
        by_prompt.setdefault(int(r["prompt_id"]), []).append((int(r["pos"]), int(lab)))
    out: Dict[int, torch.Tensor] = {}
    for s in seqs:
        y = torch.full((s.A.shape[0],), -1, dtype=torch.long)
        for pos, lab in by_prompt.get(s.prompt_id, []):
            if 0 <= pos < y.numel():
                y[pos] = int(lab)
        out[s.prompt_id] = y
    return out


def build_X_router_rows(seqs: List[SeqHeadData], rows: List[Dict[str, Any]], labels: List[int]) -> Tuple[Optional[Any], Optional[Any]]:
    if np is None or not rows:
        return None, None
    Xs = []
    ys = []
    seq_by_id = {s.prompt_id: s for s in seqs}
    for r, lab in zip(rows, labels):
        s = seq_by_id.get(int(r["prompt_id"]))
        if s is None:
            continue
        pos = int(r["pos"])
        if 0 <= pos < s.X.shape[0]:
            Xs.append(s.X[pos].numpy())
            ys.append(int(lab))
    if not Xs:
        return None, None
    return np.asarray(Xs, dtype="float32"), np.asarray(ys, dtype="int64")


def fit_hidden_router_accuracy(train_seqs: List[SeqHeadData], rows_train: List[Dict[str, Any]], labels_train: List[int],
                               val_seqs: List[SeqHeadData], rows_val: List[Dict[str, Any]], labels_val: List[int]) -> Dict[str, Any]:
    if np is None or LogisticRegression is None or StandardScaler is None or len(set(labels_train)) < 2:
        return {"train_acc": 1.0, "val_acc": 1.0, "router": "not_applicable"}
    Xtr, ytr = build_X_router_rows(train_seqs, rows_train, labels_train)
    Xv, yv = build_X_router_rows(val_seqs, rows_val, labels_val)
    if Xtr is None or Xv is None or len(set(ytr.tolist())) < 2:
        return {"train_acc": 1.0, "val_acc": 1.0, "router": "not_applicable"}
    try:
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(sc.transform(Xtr), ytr)
        ptr = clf.predict(sc.transform(Xtr))
        pv = clf.predict(sc.transform(Xv))
        return {"train_acc": float((ptr == ytr).mean()), "val_acc": float((pv == yv).mean()), "router": "hidden_X_logreg"}
    except Exception as e:
        return {"train_acc": 0.0, "val_acc": 0.0, "router": "hidden_X_logreg_failed", "note": str(e)[:200]}


def controlled_repeat_diagnostics(seqs: List[SeqHeadData], labels_by_prompt: Optional[Dict[int, torch.Tensor]] = None) -> Dict[str, Any]:
    """Check repeated identical prompts. In eval mode exact repeats should be effectively identical."""
    groups: Dict[str, List[SeqHeadData]] = {}
    for s in seqs:
        key = s.text
        groups.setdefault(key, []).append(s)
    reports = []
    for text, ss in groups.items():
        if len(ss) < 2:
            continue
        base = ss[0]
        a_rels=[]; q_rels=[]; k_rels=[]; v_rels=[]; label_matches=[]
        for other in ss[1:]:
            if base.A.shape != other.A.shape:
                continue
            a_rels.append(rel_err(other.A, base.A))
            q_rels.append(rel_err(other.Q, base.Q))
            k_rels.append(rel_err(other.K, base.K))
            v_rels.append(rel_err(other.V, base.V))
            if labels_by_prompt is not None and base.prompt_id in labels_by_prompt and other.prompt_id in labels_by_prompt:
                y0 = labels_by_prompt[base.prompt_id]
                y1 = labels_by_prompt[other.prompt_id]
                if y0.shape == y1.shape and y0.numel() > 0:
                    label_matches.append(float((y0 == y1).float().mean()))
        if a_rels:
            reports.append({
                "text": text[:120], "n": len(ss),
                "A_rel_max": max(a_rels), "A_rel_mean": safe_mean(a_rels),
                "Q_rel_max": max(q_rels), "K_rel_max": max(k_rels), "V_rel_max": max(v_rels),
                "route_label_match_mean": safe_mean(label_matches) if label_matches else None,
            })
    return {"groups": reports, "n_groups": len(reports)}

def make_row_masks(seqs: List[SeqHeadData], rows: List[Dict[str, Any]], labels: List[int], cluster_id: int) -> Dict[int, torch.Tensor]:
    by_prompt: Dict[int, List[Tuple[int, int]]] = {}
    for r, lab in zip(rows, labels):
        by_prompt.setdefault(int(r["prompt_id"]), []).append((int(r["pos"]), int(lab)))
    masks: Dict[int, torch.Tensor] = {}
    for s in seqs:
        m = torch.zeros(s.A.shape[0], dtype=torch.bool)
        for pos, lab in by_prompt.get(s.prompt_id, []):
            if lab == cluster_id and 0 <= pos < m.numel():
                m[pos] = True
        masks[s.prompt_id] = m
    return masks


def fit_router_accuracy(rows_train: List[Dict[str, Any]], labels_train: List[int], rows_val: List[Dict[str, Any]], labels_val: List[int]) -> Dict[str, float]:
    feat_keys = ["entropy", "max_prob", "top_dist", "top_rel", "pos_rel", "self_mass", "prev_mass", "bos_mass", "local1", "local2", "local4", "local8"]
    if not rows_train or not rows_val or len(set(labels_train)) < 2:
        return {"train_acc": 1.0, "val_acc": 1.0}
    if np is None or LogisticRegression is None or StandardScaler is None:
        return {"train_acc": 0.0, "val_acc": 0.0, "note": "sklearn unavailable"}
    Xtr = np.asarray([[float(r[k]) for k in feat_keys] for r in rows_train], dtype="float32")
    Xv = np.asarray([[float(r[k]) for k in feat_keys] for r in rows_val], dtype="float32")
    ytr = np.asarray(labels_train, dtype="int64")
    yv = np.asarray(labels_val, dtype="int64")
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xv_s = sc.transform(Xv)
    # sklearn compatibility: newer versions removed/deprecated multi_class="auto".
    # Keep the router diagnostic non-fatal; this metric is explanatory, not required for QK fitting.
    try:
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(Xtr_s, ytr)
        pred_tr = clf.predict(Xtr_s)
        pred_v = clf.predict(Xv_s)
        return {
            "train_acc": float((pred_tr == ytr).mean()),
            "val_acc": float((pred_v == yv).mean()),
            "router": "logistic_regression",
        }
    except Exception as e:
        # fallback: nearest centroid in standardized feature space
        classes = sorted(set(int(x) for x in ytr.tolist()))
        centroids = []
        for c in classes:
            m = ytr == c
            if m.any():
                centroids.append(Xtr_s[m].mean(axis=0))
            else:
                centroids.append(np.zeros(Xtr_s.shape[1], dtype=Xtr_s.dtype))
        C = np.stack(centroids, axis=0)
        def pred_centroid(X):
            # squared distances [N,C]
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            idx = d.argmin(axis=1)
            return np.asarray([classes[int(i)] for i in idx], dtype="int64")
        pred_tr = pred_centroid(Xtr_s)
        pred_v = pred_centroid(Xv_s)
        return {
            "train_acc": float((pred_tr == ytr).mean()),
            "val_acc": float((pred_v == yv).mean()),
            "router": "nearest_centroid_fallback",
            "note": str(e)[:200],
        }




# ---------------- model-level interpretability patch ----------------

def fit_hidden_router_model(train_seqs: List[SeqHeadData], rows_train: List[Dict[str, Any]], labels_train: List[int]) -> Optional[Dict[str, Any]]:
    """Fit X -> route classifier for use in a non-oracle route patch."""
    if np is None or LogisticRegression is None or StandardScaler is None or len(set(labels_train)) < 2:
        return None
    Xtr, ytr = build_X_router_rows(train_seqs, rows_train, labels_train)
    if Xtr is None or ytr is None or len(set(ytr.tolist())) < 2:
        return None
    try:
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(sc.transform(Xtr), ytr)
        coef = getattr(clf, "coef_", None)
        return {"scaler": sc, "clf": clf, "classes": list(map(int, clf.classes_.tolist())), "coef": coef}
    except Exception:
        return None


def predict_hidden_routes(hidden_router_state: Optional[Dict[str, Any]], X: torch.Tensor) -> Optional[torch.Tensor]:
    """Predict route labels from X [T,H] using hidden router. Returns CPU long tensor or None."""
    if hidden_router_state is None or np is None:
        return None
    try:
        Xn = X.detach().float().cpu().numpy().astype("float32")
        sc = hidden_router_state["scaler"]
        clf = hidden_router_state["clf"]
        pred = clf.predict(sc.transform(Xn))
        return torch.as_tensor(pred, dtype=torch.long)
    except Exception:
        return None


def route_rows_from_attention(A: torch.Tensor, route_state: Any) -> Optional[torch.Tensor]:
    """Oracle/explain route assignment from true attention rows using train-fitted KMeans state."""
    if route_state is None or np is None:
        return None
    rows = []
    T = int(A.shape[0])
    for i in range(T):
        row = A[i].float().detach().cpu()
        ent = float(-(row * (row + 1e-12).log()).sum() / math.log(max(2, T)))
        top = int(row.argmax())
        def lm(rad):
            lo=max(0,i-rad); hi=min(T,i+rad+1); return float(row[lo:hi].sum())
        rows.append({
            "entropy": ent,
            "max_prob": float(row[top]),
            "top_dist": float(i-top),
            "top_rel": float(top / max(1, T-1)),
            "pos_rel": float(i / max(1, T-1)),
            "self_mass": float(row[i]),
            "prev_mass": float(row[i-1]) if i>0 else 0.0,
            "bos_mass": float(row[0]),
            "local1": lm(1), "local2": lm(2), "local4": lm(4), "local8": lm(8),
        })
    try:
        Xf = np.asarray([[float(r[k]) for k in route_state["feat_keys"]] for r in rows], dtype="float32")
        labs = route_state["kmeans"].predict(route_state["scaler"].transform(Xf))
        return torch.as_tensor(labs, dtype=torch.long)
    except Exception:
        return None


def _compute_patch_Ahat(Q: torch.Tensor, K: torch.Tensor, A_true: torch.Tensor, X: torch.Tensor,
                        mode: str, scale_mode: str, global_factor: Dict[str, Any],
                        route_factors: Dict[str, Any], route_state: Any,
                        hidden_router_state: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Build patched attention matrix A_hat [T,T] for one sequence/head."""
    T = int(Q.shape[0])
    device = Q.device
    mode = mode.lower()
    if mode == "ablate":
        return torch.zeros_like(A_true)
    if mode == "self_template":
        return torch.eye(T, device=device, dtype=A_true.dtype)
    if mode in ("global_qk", "global"):
        if not global_factor:
            return A_true
        Pq = global_factor["Pq"].to(device).float(); Pk = global_factor["Pk"].to(device).float()
        r = int(Pq.shape[1]); denom = math.sqrt(r if scale_mode == "rank" else Pq.shape[0])
        return causal_softmax(((Q.float() @ Pq) @ (K.float() @ Pk).T) / denom).float()
    if mode in ("route_oracle_qk", "route_hidden_qk", "route_composed_qk"):
        # route_composed_qk aliases oracle if no hidden requested
        if mode == "route_hidden_qk":
            labs = predict_hidden_routes(hidden_router_state, X)
        else:
            labs = route_rows_from_attention(A_true, route_state)
        if labs is None:
            if not global_factor:
                return A_true
            return _compute_patch_Ahat(Q,K,A_true,X,"global_qk",scale_mode,global_factor,route_factors,route_state,hidden_router_state)
        labs = labs.to(device)
        A_out = torch.zeros_like(A_true.float())
        filled = torch.zeros((T,), device=device, dtype=torch.bool)
        # precompute per-route full A, then copy rows for that route
        for key, fac in route_factors.items():
            try:
                cid = int(str(key).replace("route", ""))
            except Exception:
                continue
            m = labs == cid
            if int(m.sum()) == 0:
                continue
            Pq = fac["Pq"].to(device).float(); Pk = fac["Pk"].to(device).float()
            r = int(Pq.shape[1]); denom = math.sqrt(r if scale_mode == "rank" else Pq.shape[0])
            Ah = causal_softmax(((Q.float() @ Pq) @ (K.float() @ Pk).T) / denom).float()
            A_out[m] = Ah[m]
            filled |= m
        if int((~filled).sum()) > 0:
            # fallback to global or true for rows without route factors
            if global_factor:
                Ag = _compute_patch_Ahat(Q,K,A_true,X,"global_qk",scale_mode,global_factor,route_factors,route_state,hidden_router_state)
                A_out[~filled] = Ag[~filled]
            else:
                A_out[~filled] = A_true[~filled]
        return A_out
    return A_true


@torch.no_grad()
def evaluate_model_level_patch(model: Any, tokenizer: Any, eval_seqs: List[SeqHeadData], layer_idx: int, head_idx: int,
                               global_factor: Dict[str, Any], route_factors: Dict[str, Any], route_state: Any,
                               hidden_router_state: Optional[Dict[str, Any]], args: argparse.Namespace,
                               max_prompts: int = 32) -> Dict[str, Any]:
    """Monkeypatch one attention head contribution and compare final model logits/loss.

    This is for interpretation, not speed. It recomputes q/k/v/A inside a forward hook and
    adds delta = Y_hat_head - Y_true_head to the attention output.
    """
    layers = get_layers(model)
    layer = layers[layer_idx]
    attn = layer.self_attn
    cfg = model.config
    hidden_size = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
    kv_groups = num_heads // num_kv
    kv_idx = head_idx // kv_groups
    device = args.device
    modes = [m.strip() for m in str(getattr(args, "patch_modes", "global_qk,route_oracle_qk,route_hidden_qk,ablate,self_template")).split(',') if m.strip()]
    # unique eval texts from val seqs
    texts=[]; seen=set()
    for s in eval_seqs:
        if s.text not in seen:
            texts.append(s.text); seen.add(s.text)
        if len(texts) >= max_prompts:
            break
    if not texts:
        return {"enabled": False, "reason": "no eval texts"}

    def run_plain(text: str):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        out = model(**enc, labels=enc["input_ids"], use_cache=False)
        return out.loss.detach().float().cpu(), out.logits.detach().float().cpu()

    base_losses=[]; base_logits=[]
    for t in texts:
        loss, logits = run_plain(t)
        base_losses.append(float(loss)); base_logits.append(logits)

    def make_hook(mode: str):
        def hook(module, fargs, kwargs, output):
            # Qwen2Attention forward receives hidden_states as kwarg in current HF.
            hidden_states = kwargs.get("hidden_states", None) if isinstance(kwargs, dict) else None
            if hidden_states is None:
                hidden_states = fargs[0] if fargs else None
            if hidden_states is None:
                return output
            # only support normal output tuple/list: (attn_output, attn_weights)
            attn_out = output[0] if isinstance(output, (tuple, list)) else output
            B,T,H = hidden_states.shape
            if T < 2:
                return output
            # q/k/v exactly like Qwen attention
            q = module.q_proj(hidden_states).view(B, T, num_heads, head_dim).transpose(1,2).contiguous()
            k = module.k_proj(hidden_states).view(B, T, num_kv, head_dim).transpose(1,2).contiguous()
            v = module.v_proj(hidden_states).view(B, T, num_kv, head_dim).transpose(1,2).contiguous()
            pos_emb = kwargs.get("position_embeddings", None) if isinstance(kwargs, dict) else None
            if pos_emb is not None:
                cos, sin = pos_emb
            else:
                pos = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)
                cos, sin = compute_position_embeddings(model, hidden_states, pos)
            q_rot, k_rot = apply_rope_qwen(q, k, cos, sin)
            o_w = module.o_proj.weight.detach().float()[:, head_idx*head_dim:(head_idx+1)*head_dim].to(hidden_states.device)
            delta = torch.zeros((B,T,H), device=hidden_states.device, dtype=torch.float32)
            for b in range(B):
                Q = q_rot[b, head_idx].float()
                K = k_rot[b, kv_idx].float()
                V = v[b, kv_idx].float()
                X = hidden_states[b].float()
                A_true = causal_softmax((Q @ K.T) / math.sqrt(head_dim)).float()
                A_hat = _compute_patch_Ahat(Q,K,A_true,X,mode,args.qk_scale,global_factor,route_factors,route_state,hidden_router_state)
                Y_true = (A_true @ V) @ o_w.T
                Y_hat = (A_hat @ V) @ o_w.T
                delta[b] = Y_hat - Y_true
            new_attn = attn_out.float() + delta.to(attn_out.device)
            new_attn = new_attn.to(attn_out.dtype)
            if isinstance(output, tuple):
                return (new_attn,) + tuple(output[1:])
            if isinstance(output, list):
                return [new_attn] + list(output[1:])
            return new_attn
        return hook

    results=[]
    for mode in modes:
        patched_losses=[]; kls=[]; rels=[]; top1s=[]
        handle = attn.register_forward_hook(make_hook(mode), with_kwargs=True)
        try:
            for i,t in enumerate(texts):
                enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
                out = model(**enc, labels=enc["input_ids"], use_cache=False)
                plog = out.logits.detach().float().cpu()
                patched_losses.append(float(out.loss.detach().float().cpu()))
                blog = base_logits[i]
                # compare next-token positions only
                a = blog[:, :-1, :]
                b = plog[:, :-1, :]
                rels.append(rel_err(b, a))
                pa = torch.softmax(a, dim=-1)
                logpb = torch.log_softmax(b, dim=-1)
                kls.append(float(F.kl_div(logpb, pa, reduction="batchmean")))
                top1s.append(float((a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean()))
        finally:
            handle.remove()
        results.append({
            "mode": mode,
            "n_prompts": len(texts),
            "loss_orig": safe_mean(base_losses),
            "loss_patch": safe_mean(patched_losses),
            "loss_delta": safe_mean(patched_losses) - safe_mean(base_losses),
            "logit_rel": safe_mean(rels),
            "kl_orig_to_patch": safe_mean(kls),
            "top1_match": safe_mean(top1s),
        })
    return {"enabled": True, "n_prompts": len(texts), "modes": results}


# ---------------- main analysis per head ----------------


def split_train_val_by_prompt(seqs: List[SeqHeadData], val_frac: float, seed: int) -> Tuple[List[SeqHeadData], List[SeqHeadData]]:
    rng = random.Random(seed)
    ids = list(range(len(seqs)))
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac))) if len(ids) > 1 else 0
    val_ids = set(ids[:n_val])
    tr = [s for i, s in enumerate(seqs) if i not in val_ids]
    va = [s for i, s in enumerate(seqs) if i in val_ids]
    if not tr and va:
        tr, va = va, []
    return tr, va


def _qk_float(r: Dict[str, Any], key: str, default: float = 999.0) -> float:
    try:
        v = r.get(key, default)
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def select_best_qk(rows: List[Dict[str, Any]], policy: str, eps_rel: float = 0.0, eps_y: float = 0.0) -> Optional[Dict[str, Any]]:
    """Select QK factor without accidentally losing compression.

    Policies:
      quality: lowest validation A_rel inside best verdict tier.
      compression: highest compression inside best verdict tier.
      balanced: error/compression tradeoff.
      min_rank_within_epsilon: choose smallest rank / highest comp if it is within
        eps_rel and eps_y of the quality-best row. This fixes cases where r32 is
        selected although r16 is practically as good and 2x more compressed.
    """
    if not rows:
        return None
    tier = {"QK_DEPLOY": 0, "QK_RISKY": 1, "QK_EXPLAIN": 2, "REJECT_QK": 3}
    rows = [dict(r) for r in rows]
    policy = str(policy or "balanced").lower()

    def t(r): return tier.get(str(r.get("verdict")), 9)
    def rank(r): return int(_qk_float(r, "rank", 10**9))
    def comp(r): return _qk_float(r, "comp", 0.0)
    def rel(r): return _qk_float(r, "val_rel", 999.0)
    def yrel(r): return _qk_float(r, "y_rel", 999.0)

    if policy == "quality":
        return sorted(rows, key=lambda r: (t(r), rel(r), yrel(r), -comp(r), rank(r)))[0]
    if policy == "compression":
        return sorted(rows, key=lambda r: (t(r), -comp(r), rel(r), yrel(r), rank(r)))[0]
    if policy in ("min_rank_within_epsilon", "minrank", "deploy_min_rank", "epsilon"):
        q = sorted(rows, key=lambda r: (t(r), rel(r), yrel(r), -comp(r), rank(r)))[0]
        best_t = t(q)
        max_rel = rel(q) + float(eps_rel)
        max_y = yrel(q) + float(eps_y)
        eligible = [r for r in rows if t(r) == best_t and rel(r) <= max_rel and yrel(r) <= max_y]
        if not eligible:
            eligible = [q]
        # highest comp == smallest rank for same head_dim; use both for safety.
        return sorted(eligible, key=lambda r: (-comp(r), rank(r), rel(r), yrel(r)))[0]
    # balanced: within tier maximize comp but punish much worse err.
    return sorted(rows, key=lambda r: (t(r), 0.55 * rel(r) + 0.35 * yrel(r) - math.log(max(1.0, comp(r))) * 0.05, rank(r)))[0]


def select_qk_policy_summary(rows: List[Dict[str, Any]], eps_rel: float = 0.0, eps_y: float = 0.0) -> Dict[str, Any]:
    """Save all useful selections so reports can show quality vs compression vs deploy choice."""
    return {
        "quality": select_best_qk(rows, "quality", eps_rel, eps_y),
        "compression": select_best_qk(rows, "compression", eps_rel, eps_y),
        "balanced": select_best_qk(rows, "balanced", eps_rel, eps_y),
        "min_rank_within_epsilon": select_best_qk(rows, "min_rank_within_epsilon", eps_rel, eps_y),
    }




def _prof_lap(enabled: bool, prof: List[Dict[str, Any]], name: str, last: List[float]) -> None:
    now = time.perf_counter()
    dt = now - last[0]
    prof.append({"stage": name, "seconds": dt})
    last[0] = now
    if enabled:
        print(f"    [profile] {name}: {dt:.3f}s", flush=True)


def analyze_head(model: Any, tokenizer: Any, prompts: List[Dict[str, str]], layer_idx: int, head_idx: int, args: argparse.Namespace,
                 out_dir: Path) -> Dict[str, Any]:
    device = args.device
    cfg = model.config
    hidden_size = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
    kv_groups = num_heads // num_kv
    kv_idx = head_idx // kv_groups

    print(f"\n=== Analyze L{layer_idx}H{head_idx} kv={kv_idx} ===", flush=True)
    _profile_rows: List[Dict[str, Any]] = []
    _profile_last = [time.perf_counter()]
    _profile_enabled = bool(getattr(args, "profile", False))
    pre = getattr(args, "_precollected_head_data", None)
    if pre is not None and (layer_idx, head_idx) in pre:
        seqs = pre[(layer_idx, head_idx)]
    else:
        seqs = collect_head_data(model, tokenizer, prompts, layer_idx, head_idx, args.max_length, device)
    if len(seqs) < 2:
        raise RuntimeError("not enough sequences collected")
    train, val = split_train_val_by_prompt(seqs, args.val_frac, args.seed + layer_idx * 100 + head_idx)
    print(f"  collected seqs={len(seqs)} train={len(train)} val={len(val)} rows_train={sum(s.A.shape[0] for s in train)} rows_val={sum(s.A.shape[0] for s in val)}", flush=True)
    _prof_lap(_profile_enabled, _profile_rows, "collect_split", _profile_last)

    # Row routes/clusters from TRUE attention rows.
    # v2: fit scaler+kmeans on TRAIN rows, then assign VAL rows through that same fitted route model.
    # This prevents the earlier bug where train/val clusters were learned independently.
    rows_train = build_row_features(train)
    labels_train, cluster_info, route_state = fit_row_clusterer(rows_train, parse_list_ints(args.clusters), args.seed)
    rows_val = build_row_features(val)
    labels_val = assign_row_clusters(rows_val, route_state)
    cluster_info_val = {"assigned_by": "train_fitted_kmeans", "k": cluster_info.get("k", 0),
                        "sizes": [int(sum(1 for z in labels_val if z == c)) for c in sorted(set(labels_train))]}
    router_acc_row_features = fit_router_accuracy(rows_train, labels_train, rows_val, labels_val)
    router_acc_hidden_X = fit_hidden_router_accuracy(train, rows_train, labels_train, val, rows_val, labels_val)
    router_acc = {"row_feature_router": router_acc_row_features, "hidden_X_router": router_acc_hidden_X}

    # labels for all seqs using same train-fitted state, used for repeat diagnostics
    rows_all = build_row_features(seqs)
    labels_all = assign_row_clusters(rows_all, route_state)
    labels_by_prompt_all = seq_labels_from_row_labels(seqs, rows_all, labels_all)
    repeat_report = controlled_repeat_diagnostics(seqs, labels_by_prompt_all)
    _prof_lap(_profile_enabled, _profile_rows, "routes_cluster_router", _profile_last)

    route_summaries: List[Dict[str, Any]] = []
    for cid in sorted(set(labels_train)):
        cr_train = [r for r, lab in zip(rows_train, labels_train) if lab == cid]
        cr_val = [r for r, lab in zip(rows_val, labels_val) if lab == cid]
        cr = cr_train + cr_val
        ex = [r["token"] for r in cr_train[:20]]
        n_train = len(cr_train)
        n_val = len(cr_val)
        if n_train >= args.major_route_rows and n_val >= max(16, args.major_route_rows // 4):
            route_verdict = "MAJOR_ROUTE"
        elif n_train >= args.min_route_rows:
            route_verdict = "MICRO_ROUTE"
        else:
            route_verdict = "TINY_ROUTE"
        route_summaries.append({
            "cluster": int(cid),
            "route_verdict": route_verdict,
            "rows_train": n_train,
            "rows_val": n_val,
            "rows": n_train + n_val,
            "route_type": infer_route_type(cr_train),
            "entropy": safe_mean([float(r["entropy"]) for r in cr_train]),
            "max_prob": safe_mean([float(r["max_prob"]) for r in cr_train]),
            "self_mass": safe_mean([float(r["self_mass"]) for r in cr_train]),
            "prev_mass": safe_mean([float(r["prev_mass"]) for r in cr_train]),
            "bos_mass": safe_mean([float(r["bos_mass"]) for r in cr_train]),
            "local4": safe_mean([float(r["local4"]) for r in cr_train]),
            "examples": " | ".join(ex[:12]),
        })

    # Original O slice for full head-contribution error Y=(A@V)@O_h.T
    attn_mod = get_layers(model)[layer_idx].self_attn
    o_w = attn_mod.o_proj.weight.detach().float()[:, head_idx * head_dim : (head_idx + 1) * head_dim].to(device)

    # Global QK bases
    qk_rows: List[Dict[str, Any]] = []
    factor_bank: Dict[Tuple[str, str, int], Dict[str, torch.Tensor]] = {}
    methods = [m.strip().lower() for m in args.qk_methods.replace(';', ',').split(',') if m.strip()]
    for rank in parse_list_ints(args.ranks):
        Pq0, Pk0 = pca_shared_init(train, rank, device)
        if "pca" in methods:
            ev = eval_qk_basis(val, Pq0, Pk0, args.qk_scale, o_w=o_w, batch_size=getattr(args, "train_batch_size", 16))
            comp = head_dim / rank
            factor_bank[("global", "qk_pca_shared", rank)] = {"Pq": Pq0.detach().cpu(), "Pk": Pk0.detach().cpu()}
            qk_rows.append({
                "scope": "global", "layer": layer_idx, "head": head_idx, "method": "qk_pca_shared",
                "rank": rank, "val_rel": ev["rel"], "val_kl": ev["kl"], "top1": ev["top1"], "z_rel": ev["z_rel"], "y_rel": ev.get("y_rel", -1.0),
                "comp": comp, "verdict": verdict_qk(ev["rel"], comp, args.accept_rel, args.risky_rel, args.explain_rel, args.min_compression),
            })
        if "learned" in methods:
            cpath = qk_cache_path(args, layer_idx, head_idx, "global", "qk_learned_proj", rank, train, val, None, None, args.qk_learned_steps)
            cached = None if getattr(args, "rebuild_cache", False) else load_qk_cache(cpath, device)
            if cached is not None:
                Pq, Pk, ev = cached
            else:
                Pq, Pk, ev = train_qk_learned(train, val, Pq0, Pk0, args.qk_scale, args.qk_learned_steps,
                                              args.qk_lr, args.qk_wd, args.qk_patience, None, None, device, o_w=o_w, batch_size=getattr(args, "train_batch_size", 16), eval_every=getattr(args, "qk_eval_every", 25))
                save_qk_cache(cpath, Pq, Pk, ev)
            comp = head_dim / rank
            factor_bank[("global", "qk_learned_proj", rank)] = {"Pq": Pq.detach().cpu(), "Pk": Pk.detach().cpu()}
            qk_rows.append({
                "scope": "global", "layer": layer_idx, "head": head_idx, "method": "qk_learned_proj",
                "rank": rank, "val_rel": ev["rel"], "val_kl": ev["kl"], "top1": ev["top1"], "z_rel": ev["z_rel"], "y_rel": ev.get("y_rel", -1.0),
                "comp": comp, "verdict": verdict_qk(ev["rel"], comp, args.accept_rel, args.risky_rel, args.explain_rel, args.min_compression),
            })
            # save best-ish factor candidates later after selection, not all to avoid huge file
    global_qk_policy_summary = select_qk_policy_summary([r for r in qk_rows if r["scope"] == "global"], getattr(args, "select_eps_rel", 0.0), getattr(args, "select_eps_y", 0.0))
    best_global = select_best_qk([r for r in qk_rows if r["scope"] == "global"], args.select_policy, getattr(args, "select_eps_rel", 0.0), getattr(args, "select_eps_y", 0.0))
    _prof_lap(_profile_enabled, _profile_rows, "global_qk_search", _profile_last)

    # Optional route-specific QK. Uses train clusters and validation clusters independently for now.
    # This answers: "is this route simpler than whole head?" rather than a final deploy router.
    if args.fit_route_qk:
        train_masks_all = {cid: make_row_masks(train, rows_train, labels_train, cid) for cid in sorted(set(labels_train))}
        val_masks_all = {cid: make_row_masks(val, rows_val, labels_val, cid) for cid in sorted(set(labels_train))}
        route_candidates = []
        for cid in sorted(set(labels_train)):
            nrows = sum(int(m.sum()) for m in train_masks_all[cid].values())
            if nrows >= args.min_route_rows:
                # Prefer major and non-trivial routes, but keep row count as tie breaker.
                rsum = next((r for r in route_summaries if int(r.get("cluster", -1)) == int(cid)), {})
                nontriv = 0 if str(rsum.get("route_type")) in ("self", "prev") else 1
                major = 1 if str(rsum.get("route_verdict")) == "MAJOR_ROUTE" else 0
                route_candidates.append((major, nontriv, nrows, cid))
        route_candidates.sort(reverse=True)
        max_routes = int(getattr(args, "route_qk_max_routes", 0) or 0)
        if max_routes > 0:
            route_candidates = route_candidates[:max_routes]
        for _, _, nrows, cid in route_candidates:
            for rank in parse_list_ints(args.route_ranks or args.ranks):
                Pq0, Pk0 = pca_shared_init(train, rank, device)
                # learned only for route-specific by default because PCA on global rows may be weak
                cpath = qk_cache_path(args, layer_idx, head_idx, f"route{cid}", "qk_route_learned_proj", rank, train, val, train_masks_all[cid], val_masks_all.get(cid, {}), args.route_qk_steps)
                cached = None if getattr(args, "rebuild_cache", False) else load_qk_cache(cpath, device)
                if cached is not None:
                    Pq, Pk, ev = cached
                else:
                    Pq, Pk, ev = train_qk_learned(train, val, Pq0, Pk0, args.qk_scale, args.route_qk_steps,
                                                  args.qk_lr, args.qk_wd, args.qk_patience,
                                                  train_masks_all[cid], val_masks_all.get(cid, {}), device, o_w=o_w, batch_size=getattr(args, "train_batch_size", 16), eval_every=getattr(args, "qk_eval_every", 25))
                    save_qk_cache(cpath, Pq, Pk, ev)
                comp = head_dim / rank
                factor_bank[(f"route{cid}", "qk_route_learned_proj", rank)] = {"Pq": Pq.detach().cpu(), "Pk": Pk.detach().cpu()}
                qk_rows.append({
                    "scope": f"route{cid}", "layer": layer_idx, "head": head_idx, "method": "qk_route_learned_proj",
                    "rank": rank, "route_rows": nrows, "val_rel": ev["rel"], "val_kl": ev["kl"], "top1": ev["top1"], "z_rel": ev["z_rel"], "y_rel": ev.get("y_rel", -1.0),
                    "comp": comp, "verdict": verdict_qk(ev["rel"], comp, args.accept_rel, args.risky_rel, args.explain_rel, args.min_compression),
                })
    _prof_lap(_profile_enabled, _profile_rows, "route_qk_search", _profile_last)

    operator_dynamic_ctx = _route_factor_context_from_bank(factor_bank, qk_rows, best_global, args)

    route_operator_rows: List[Dict[str, Any]] = []
    route_operator_bank: Dict[str, Any] = {"kind": "disabled"}
    if getattr(args, "logic_operator_fit", True):
        route_operator_rows, route_operator_bank = fit_route_operator_programs(
            train, val, labels_train, labels_val, rows_train, rows_val, route_summaries, o_w, args,
            dynamic_ctx=operator_dynamic_ctx,
        )
        _prof_lap(_profile_enabled, _profile_rows, "route_operator_fit", _profile_last)

    # Build IR summary
    program_ir = {
        "type": "qwen_attention_head_program",
        "layer": layer_idx,
        "head": head_idx,
        "kv_head": kv_idx,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv,
        "input_space": "layer.input_layernorm(hidden_states)",
        "known_ops": [
            {"op": "RMSNorm", "role": "input preparation", "note": "dynamic normalization; not treated as learned matrix"},
            {"op": "Linear", "name": "q_proj head slice", "shape": [head_dim, hidden_size], "bias": True},
            {"op": "Linear", "name": "k_proj kv slice", "shape": [head_dim, hidden_size], "bias": True},
            {"op": "Linear", "name": "v_proj kv slice", "shape": [head_dim, hidden_size], "bias": True},
            {"op": "RoPE", "role": "known position-dependent pair rotation before QK"},
            {"op": "QKSoftmax", "role": "dynamic read/router", "formula": "A=causal_softmax(Q_rope K_rope^T / sqrt(head_dim))"},
            {"op": "AV", "role": "read values", "formula": "Z=A@V"},
            {"op": "Linear", "name": "o_proj head slice", "shape": [hidden_size, head_dim], "bias": False},
        ],
        "best_global_qk": best_global,
        "global_qk_policy_summary": global_qk_policy_summary,
        "route_cluster_info": cluster_info,
        "route_val_assignment_info": cluster_info_val,
        "route_router_probe": router_acc,
        "controlled_repeat_diagnostics": repeat_report,
        "profile": _profile_rows,
        "routes": route_summaries,
        "route_operator_programs": route_operator_rows,
        "route_operator_bank": route_operator_bank,
        "interpretation_notes": [
            "QK/read is the part where low-rank basis is meaningful.",
            "RMSNorm and RoPE are explicit known operators, not dictionary atoms to rediscover.",
            "Attention rows are dynamic routes; rich token/qtype/dynamic dictionary describes self/prev/BOS/local/diffuse/content behavior.",
            "Hidden dimension does not have natural spatial order, so Blur/DCT/Shift over hidden coordinates is not used here.",
        ],
    }

    # write per-head artifacts
    hdir = out_dir / f"L{layer_idx}H{head_idx}"
    ensure_dir(hdir)
    (hdir / "head_program_ir.json").write_text(json_dumps_safe(program_ir, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(hdir / "qk_basis_rows.csv", qk_rows)
    write_csv(hdir / "attention_routes.csv", route_summaries)
    write_csv(hdir / "route_operator_programs.csv", route_operator_rows)
    (hdir / "route_operator_bank.json").write_text(json_dumps_safe(route_operator_bank, indent=2, ensure_ascii=False), encoding="utf-8")
    route_bank_matrix_info = build_route_bank_artifacts(
        hdir, route_operator_bank, route_summaries,
        route_bank_size=int(getattr(args, "route_bank_size", 64)),
        block_forest_depth=int(getattr(args, "block_forest_depth", 2)),
        train=train, val=val, rows_train=rows_train, labels_train=labels_train,
        rows_val=rows_val, labels_val=labels_val, args=args, dynamic_ctx=operator_dynamic_ctx,
    )
    route_token_stats_train = route_token_statistics(rows_train, labels_train)
    route_token_stats_val = route_token_statistics(rows_val, labels_val)
    (hdir / "route_token_stats_train.json").write_text(json_dumps_safe(route_token_stats_train, indent=2, ensure_ascii=False), encoding="utf-8")
    (hdir / "route_token_stats_val.json").write_text(json_dumps_safe(route_token_stats_val, indent=2, ensure_ascii=False), encoding="utf-8")
    write_integrated_token_examples(
        hdir, seqs, labels_by_prompt_all, route_summaries, route_operator_rows,
        max_prompts=int(getattr(args, "integrated_max_example_prompts", 3)),
        max_tokens=int(getattr(args, "integrated_max_example_tokens", 24)),
    )
    program_ir["route_bank_matrix"] = route_bank_matrix_info
    program_ir["route_token_stats_train"] = route_token_stats_train
    program_ir["route_token_stats_val"] = route_token_stats_val
    (hdir / "head_program_ir.json").write_text(json_dumps_safe(program_ir, indent=2, ensure_ascii=False), encoding="utf-8")
    # Save selected global factor and best route factors for formula/patch stage.
    factors_to_save: Dict[str, Any] = {"meta": {"layer": layer_idx, "head": head_idx, "head_dim": head_dim, "qk_scale": args.qk_scale}, "global": {}, "routes": {}}
    if best_global is not None:
        key = (str(best_global["scope"]), str(best_global["method"]), int(best_global["rank"]))
        if key in factor_bank:
            factors_to_save["global"] = {"row": best_global, **factor_bank[key]}
    for cid in sorted(set(labels_train)):
        scope = f"route{cid}"
        cand = [r for r in qk_rows if r.get("scope") == scope]
        br = select_best_qk(cand, args.select_policy, getattr(args, "select_eps_rel", 0.0), getattr(args, "select_eps_y", 0.0))
        if br is not None:
            key = (str(br["scope"]), str(br["method"]), int(br["rank"]))
            if key in factor_bank:
                factors_to_save["routes"][scope] = {"row": br, **factor_bank[key]}
    torch.save(factors_to_save, hdir / "qk_selected_factors.pt")

    patch_report = {"enabled": False, "reason": "--patch-model not set"}
    if getattr(args, "patch_model", False):
        hidden_router_state = fit_hidden_router_model(train, rows_train, labels_train)
        patch_report = evaluate_model_level_patch(
            model, tokenizer, val, layer_idx, head_idx,
            factors_to_save.get("global", {}), factors_to_save.get("routes", {}),
            route_state, hidden_router_state, args,
            max_prompts=int(getattr(args, "patch_eval_prompts", 32)),
        )
        (hdir / "model_patch_report.json").write_text(json_dumps_safe(patch_report, indent=2, ensure_ascii=False), encoding="utf-8")
        program_ir["model_patch_report"] = patch_report
        _prof_lap(_profile_enabled, _profile_rows, "model_patch", _profile_last)
        program_ir["profile"] = _profile_rows
        (hdir / "head_program_ir.json").write_text(json_dumps_safe(program_ir, indent=2, ensure_ascii=False), encoding="utf-8")

    (hdir / "profile.json").write_text(json_dumps_safe(_profile_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (hdir / "controlled_repeat_diagnostics.json").write_text(json_dumps_safe(repeat_report, indent=2, ensure_ascii=False), encoding="utf-8")
    # token examples and route markdown
    md = []
    md.append(f"# L{layer_idx}H{head_idx} head decompile\n")
    md.append(f"- collected seqs: {len(seqs)} train={len(train)} val={len(val)}")
    md.append(f"- row clusters: {cluster_info}")
    md.append(f"- router probe: {router_acc}")
    md.append(f"- val route assignment: {cluster_info_val}")
    md.append(f"- repeat diagnostics: {repeat_report}")
    md.append(f"- best global QK: `{best_global}`\n")
    if getattr(args, "patch_model", False):
        md.append(f"- model patch report: `{patch_report}`\n")
    md.append("## Routes\n")
    for r in route_summaries:
        md.append(f"### route {r['cluster']} type={r['route_type']} verdict={r.get('route_verdict')} rows={r['rows']} train={r.get('rows_train')} val={r.get('rows_val')}")
        md.append(f"entropy={r['entropy']:.4f} max={r['max_prob']:.4f} self={r['self_mass']:.4f} prev={r['prev_mass']:.4f} bos={r['bos_mass']:.4f} local4={r['local4']:.4f}")
        md.append(f"examples: `{r['examples']}`\n")
    (hdir / "head_decompile.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  best_global={best_global}", flush=True)
    print(f"  wrote {hdir}", flush=True)
    return {"layer": layer_idx, "head": head_idx, "best_global_qk": best_global,
        "global_qk_policy_summary": global_qk_policy_summary, "cluster_info": cluster_info, "cluster_info_val": cluster_info_val, "router_acc": router_acc, "repeat_report": repeat_report, "routes": route_summaries}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json_dumps_safe(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple, set)) else json_sanitize(v) for k, v in r.items()})




# ---------------- logic/operator decomposition over attention rows ----------------

DEFAULT_OPERATOR_TEMPLATES = [
    "BOSRead", "IdentitySelf", "PrevToken",
    "LocalWindow1", "LocalWindow2", "LocalWindow4", "LocalWindow8", "LocalWindow16",
    "UniformPast", "DistanceDecay2", "DistanceDecay4", "DistanceDecay8", "DistanceDecay16",
    "PunctPast", "BracketPast", "OperatorPast", "NumberPast", "WordPast", "CodeMixedPast",
    "SameTokenPast", "SameKindPast",
]


def _clean_token(tok: str) -> str:
    return str(tok).replace("Ġ", "").replace("▁", "")


def token_kind(tok: str) -> str:
    t = _clean_token(tok)
    if not t:
        return "empty"
    if re.fullmatch(r"[\]\[\(\)\{\}<>,.;:!?]+", t):
        return "punct_bracket"
    if re.fullmatch(r"[\]\[\(\)\{\}<>]+", t):
        return "bracket"
    if re.fullmatch(r"[-+*/=<>!&|%^~]+", t):
        return "operator"
    if re.fullmatch(r"[0-9]+([.,][0-9]+)?", t):
        return "number"
    if re.search(r"[A-Za-z_]", t) and re.search(r"[{}()[\];:=+*/<>]", t):
        return "code_mixed"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
        return "word"
    if re.search(r"[0-9]", t):
        return "alnum"
    return "other"


def _token_match_indices(tokens: Optional[List[str]], i: int, mode: str) -> List[int]:
    if not tokens:
        return []
    i = max(0, min(int(i), len(tokens)-1))
    cur = _clean_token(tokens[i])
    cur_kind = token_kind(tokens[i])
    out=[]
    for j in range(i+1):
        tj = _clean_token(tokens[j])
        kj = token_kind(tokens[j])
        if mode == "punct" and kj == "punct_bracket": out.append(j)
        elif mode == "bracket" and (kj == "bracket" or tj in "()[]{}<>"): out.append(j)
        elif mode == "operator" and kj == "operator": out.append(j)
        elif mode == "number" and kj == "number": out.append(j)
        elif mode == "word" and kj == "word": out.append(j)
        elif mode == "code_mixed" and kj == "code_mixed": out.append(j)
        elif mode == "same_token" and tj and cur and tj == cur: out.append(j)
        elif mode == "same_kind" and kj == cur_kind: out.append(j)
    return out


def _parse_templates(s: str) -> List[str]:
    return [x.strip() for x in (s or "").replace(';', ',').split(',') if x.strip()] or list(DEFAULT_OPERATOR_TEMPLATES)


def attention_template_row(T: int, i: int, name: str, device: str = "cpu", tokens: Optional[List[str]] = None) -> torch.Tensor:
    """Causal attention-row operator basis. Token/code operators use tokenizer strings."""
    v = torch.zeros(T, dtype=torch.float32, device=device)
    n = name.lower()
    if T <= 0:
        return v
    i = int(max(0, min(i, T - 1)))
    def normalize_or_fallback(indices: List[int]) -> torch.Tensor:
        if indices:
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            v[idx] = 1.0
        else:
            v[: i + 1] = 1.0
        v[:] = v / v.sum().clamp_min(1e-12)
        return v
    if n in ("bos", "bosread", "pos_bosread"):
        v[0] = 1.0
    elif n in ("self", "identityself", "pos_identityself"):
        v[i] = 1.0
    elif n in ("prev", "prevtoken", "pos_prevtoken"):
        v[max(0, i - 1)] = 1.0
    elif n.startswith("localwindow") or n.startswith("local"):
        m = re.search(r"(\d+)", n); r = int(m.group(1)) if m else 4
        v[max(0, i-r): i+1] = 1.0; v /= v.sum().clamp_min(1e-12)
    elif n in ("uniformpast", "pastuniform", "diffusepast"):
        v[: i + 1] = 1.0; v /= v.sum().clamp_min(1e-12)
    elif n.startswith("distancedecay") or n.startswith("decay"):
        m = re.search(r"(\d+)", n); tau = float(m.group(1)) if m else 4.0
        js = torch.arange(i + 1, dtype=torch.float32, device=device)
        vv = torch.exp(-(float(i) - js) / max(1e-6, tau))
        v[: i + 1] = vv / vv.sum().clamp_min(1e-12)
    elif n in ("punctpast", "punctuationpast"):
        return normalize_or_fallback(_token_match_indices(tokens, i, "punct"))
    elif n in ("bracketpast", "bracketread"):
        return normalize_or_fallback(_token_match_indices(tokens, i, "bracket"))
    elif n in ("operatorpast", "operatorread"):
        return normalize_or_fallback(_token_match_indices(tokens, i, "operator"))
    elif n in ("numberpast", "numberread"):
        return normalize_or_fallback(_token_match_indices(tokens, i, "number"))
    elif n in ("wordpast", "identifierpast", "wordread"):
        return normalize_or_fallback(_token_match_indices(tokens, i, "word"))
    elif n in ("codemixedpast", "codepast", "coderead"):
        return normalize_or_fallback(_token_match_indices(tokens, i, "code_mixed"))
    elif n in ("sametokenpast", "sametokenread"):
        return normalize_or_fallback(_token_match_indices(tokens, i, "same_token"))
    elif n in ("samekindpast", "samekindread"):
        return normalize_or_fallback(_token_match_indices(tokens, i, "same_kind"))
    else:
        v[: i + 1] = 1.0; v /= v.sum().clamp_min(1e-12)
    return v


def _operator_basis_for_row(T: int, i: int, names: List[str], device: str = "cpu", tokens: Optional[List[str]] = None) -> torch.Tensor:
    return torch.stack([attention_template_row(T, i, nm, device=device, tokens=tokens) for nm in names], dim=0)  # [M,T]


# Cache basis matrices per (T,tokens,names). This is the main speed fix:
# old code rebuilt every operator row many times during fit/eval/family sweeps.
# New code builds [M,T,T] once per sequence/template set and reuses it.
_OPERATOR_BASIS_CACHE: Dict[Tuple[int, Tuple[str, ...], Tuple[str, ...]], torch.Tensor] = {}

def _operator_basis_matrix_for_seq(T: int, names: List[str], tokens: Optional[List[str]] = None) -> torch.Tensor:
    toks = tuple(tokens or ["?"] * T)
    key = (int(T), toks, tuple(names))
    B = _OPERATOR_BASIS_CACHE.get(key)
    if B is not None:
        return B
    mats = []
    for nm in names:
        rows = [attention_template_row(T, i, nm, device="cpu", tokens=list(toks)) for i in range(T)]
        mats.append(torch.stack(rows, dim=0))  # [T,T]
    B = torch.stack(mats, dim=0).float().contiguous()  # [M,T,T]
    # avoid unbounded RAM if user scans huge many prompts
    if len(_OPERATOR_BASIS_CACHE) > 2048:
        _OPERATOR_BASIS_CACHE.clear()
    _OPERATOR_BASIS_CACHE[key] = B
    return B


def _fit_operator_coefficients(seqs: List[SeqHeadData], row_masks: Dict[int, torch.Tensor], names: List[str], ridge: float) -> torch.Tensor:
    # Vectorized normal equations:
    # for masked rows, B is [M,T] and y is [T].
    # Accumulate G=sum(BB^T), b=sum(By) in batched einsum per sequence.
    M = len(names)
    G = torch.zeros(M, M, dtype=torch.float64)
    b = torch.zeros(M, dtype=torch.float64)
    nrows = 0
    for s in seqs:
        mask = row_masks.get(s.prompt_id)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool()
        if int(mask.sum()) == 0:
            continue
        T = int(s.A.shape[0])
        B_all = _operator_basis_matrix_for_seq(T, names, tokens=s.tokens).double()  # [M,T,T]
        # rows basis for selected query rows: [R,M,T]
        Br = B_all[:, mask, :].permute(1, 0, 2).contiguous()
        Y = s.A.detach().cpu().double()[mask]  # [R,T]
        G += torch.einsum("rmt,rnt->mn", Br, Br)
        b += torch.einsum("rmt,rt->m", Br, Y)
        nrows += int(mask.sum())
    if nrows == 0:
        return torch.zeros(M, dtype=torch.float32)
    G += float(ridge) * torch.eye(M, dtype=torch.float64)
    try:
        c = torch.linalg.solve(G, b)
    except Exception:
        c = torch.linalg.lstsq(G, b[:, None]).solution[:, 0]
    c = torch.clamp(c.float(), min=0.0)
    if float(c.sum()) <= 1e-9:
        c = torch.ones(M, dtype=torch.float32) / max(1, M)
    return c


def _make_operator_rows_for_seq(T: int, names: List[str], coeffs: torch.Tensor, device: str = "cpu", tokens: Optional[List[str]] = None) -> torch.Tensor:
    # Vectorized: B_all [M,T,T], coeffs [M] -> [T,T].
    B_all = _operator_basis_matrix_for_seq(T, names, tokens=tokens).to(device).float()
    c = coeffs.to(device).float()
    A = torch.einsum("m,mij->ij", c, B_all).clamp_min(0.0)
    A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return A


def _eval_operator_fit(seqs: List[SeqHeadData], row_masks: Dict[int, torch.Tensor], names: List[str], coeffs: torch.Tensor,
                       o_w: torch.Tensor, device: str = "cpu") -> Dict[str, float]:
    # Vectorized per sequence, reusing cached operator basis matrices.
    rels=[]; kls=[]; top1s=[]; zrels=[]; yrels=[]; nrows=0
    c = coeffs.detach().float().cpu()
    Ow = o_w.detach().cpu().float()
    for s in seqs:
        mask = row_masks.get(s.prompt_id)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool()
        if int(mask.sum()) == 0:
            continue
        T = int(s.A.shape[0])
        A_hat = _make_operator_rows_for_seq(T, names, c, device="cpu", tokens=s.tokens)
        A_true = s.A.detach().cpu().float()
        Ah = A_hat[mask]
        At = A_true[mask]
        rels.append(rel_err(Ah, At))
        kls.append(float(F.kl_div((Ah + 1e-12).log(), At, reduction="batchmean")))
        top1s.append(float((Ah.argmax(dim=-1) == At.argmax(dim=-1)).float().mean()))
        V = s.V.detach().cpu().float()
        Zhat_full = A_hat @ V
        Ztrue_full = A_true @ V
        zrels.append(rel_err(Zhat_full[mask], Ztrue_full[mask]))
        Yhat_full = Zhat_full @ Ow.T
        Ytrue_full = s.Y.detach().cpu().float()
        yrels.append(rel_err(Yhat_full[mask], Ytrue_full[mask]))
        nrows += int(mask.sum())
    return {
        "rel": safe_mean(rels), "kl": safe_mean(kls), "top1": safe_mean(top1s),
        "z_rel": safe_mean(zrels), "y_rel": safe_mean(yrels), "rows": nrows,
    }


def operator_formula(names: List[str], coeffs: torch.Tensor, max_terms: int = 6) -> str:
    # Display normalized coefficients because A rows are normalized after mixing;
    # absolute LS scale is not meaningful, relative mixture weight is.
    cs = coeffs.detach().cpu().float().clamp_min(0.0)
    s = float(cs.sum())
    if s > 1e-12:
        cs = cs / s
    pairs = [(nm, float(c)) for nm, c in zip(names, cs.tolist())]
    pairs = [(nm, c) for nm, c in pairs if c > 1e-6]
    pairs.sort(key=lambda x: -abs(x[1]))
    if not pairs:
        return "0"
    return " + ".join(f"{c:.3f}*{nm}" for nm, c in pairs[:max_terms])




def operator_family(name: str) -> str:
    n = name.lower()
    if n in ("bosread", "identityself", "prevtoken"):
        return "anchor"
    if n.startswith("local"):
        return "local"
    if n.startswith("distancedecay") or n == "uniformpast":
        return "diffuse"
    if n in ("punctpast", "bracketpast", "operatorpast", "numberpast", "wordpast", "codemixedpast"):
        return "token_type"
    if n in ("sametokenpast", "samekindpast"):
        return "repeat_or_kind"
    return "other"


def _mask_coeffs_by_family(names: List[str], coeffs: torch.Tensor, family: str) -> torch.Tensor:
    c = coeffs.detach().clone().float()
    for i, nm in enumerate(names):
        if operator_family(nm) != family:
            c[i] = 0.0
    return c


def _canonical_operator_matrix(T: int, names: List[str], coeffs: torch.Tensor) -> torch.Tensor:
    return _make_operator_rows_for_seq(T, names, coeffs.detach().cpu().float(), device="cpu", tokens=None)


def _template_matrix(T: int, name: str) -> torch.Tensor:
    return _canonical_operator_matrix(T, [name], torch.ones(1))


def _matrix_rel(A: torch.Tensor, B: torch.Tensor) -> float:
    return rel_err(A.float(), B.float())


def route_matrix_block_forest(M: torch.Tensor, route: int, max_depth: int = 2) -> List[Dict[str, Any]]:
    """Build the same human-readable block tree, but evaluate it as a flat DAG.

    Important: this function intentionally has NO recursive calls.
    It preserves the tree semantics for reports by emitting node_id/parent_id/path/depth,
    but all node energies are computed level-by-level from flat node tables.

    Conceptually:
      tree IR -> flat nodes/edges -> level-order DAG evaluation -> CSV rows

    This keeps the logic of block decomposition while avoiding Python recursion and
    making it easy to batch/scatter later.
    """
    M = M.detach().cpu().float()
    R, C = int(M.shape[0]), int(M.shape[1])
    total = float((M * M).sum().sqrt().item()) + 1e-12

    # Flat node table. A node is a rectangular view into M plus tree metadata.
    nodes: List[Dict[str, Any]] = [{
        "node_id": 0, "parent_id": -1, "route": int(route),
        "path": f"route{route}", "edge_type": "root",
        "depth": 0, "row0": 0, "col0": 0, "rows": R, "cols": C,
        "quad": "ROOT",
    }]
    next_id = 1

    # Build graph by levels, not recursive DFS. This is the tree construction step.
    current_level = [0]
    for depth in range(int(max_depth)):
        next_level: List[int] = []
        for nid in current_level:
            nd = nodes[nid]
            n, m = int(nd["rows"]), int(nd["cols"])
            if n < 4 or m < 4:
                continue
            nr, mc = n // 2, m // 2
            specs = [
                ("TL", int(nd["row0"]),      int(nd["col0"]),      nr,     mc),
                ("TR", int(nd["row0"]),      int(nd["col0"]) + mc, nr,     m - mc),
                ("BL", int(nd["row0"]) + nr, int(nd["col0"]),      n - nr, mc),
                ("BR", int(nd["row0"]) + nr, int(nd["col0"]) + mc, n - nr, m - mc),
            ]
            # Energy rank keeps old forest behavior: high-energy children appear first.
            child_infos = []
            for tag, r0, c0, rr, cc in specs:
                sub = M[r0:r0 + rr, c0:c0 + cc]
                e2 = float((sub * sub).sum().item())
                if e2 <= 1e-12:
                    continue
                child_infos.append((e2, tag, r0, c0, rr, cc))
            child_infos.sort(key=lambda x: -x[0])
            for e2, tag, r0, c0, rr, cc in child_infos:
                nodes.append({
                    "node_id": next_id, "parent_id": int(nid), "route": int(route),
                    "path": str(nd["path"]) + "/" + tag, "edge_type": "quadrant",
                    "depth": int(depth + 1), "row0": int(r0), "col0": int(c0),
                    "rows": int(rr), "cols": int(cc), "quad": tag,
                    "parent_path": str(nd["path"]),
                })
                next_level.append(next_id)
                next_id += 1
        current_level = next_level
        if not current_level:
            break

    # Flat DAG evaluation: all node metrics are computed from node arrays.
    # This is not recursive; each row is independent and can be vectorized/scattered.
    out: List[Dict[str, Any]] = []
    for nd in nodes:
        r0, c0, rr, cc = int(nd["row0"]), int(nd["col0"]), int(nd["rows"]), int(nd["cols"])
        sub = M[r0:r0 + rr, c0:c0 + cc]
        en = float((sub * sub).sum().sqrt().item())
        row = dict(nd)
        row.update({
            "energy": en,
            "energy_frac": en / total,
            "dag_nodes": len(nodes),
            "compiled_eval": "flat_dag_level_order",
        })
        out.append(row)

    # Preserve old ranking behavior for CSV/report readability while node_id/parent_id
    # keeps the reconstructable tree. Root first, then high-energy nodes per depth.
    out.sort(key=lambda r: (int(r["depth"]) != 0, int(r["depth"]), -float(r["energy"]), int(r["node_id"])))
    return out


def route_matrix_block_graph_flat(M: torch.Tensor, route: int, max_depth: int = 2) -> Dict[str, Any]:
    """Return explicit flat graph IR for debugging/report JSON without recursion."""
    rows = route_matrix_block_forest(M, route, max_depth=max_depth)
    nodes = [{k: r.get(k) for k in ("node_id", "parent_id", "route", "path", "depth", "row0", "col0", "rows", "cols", "quad", "energy", "energy_frac")} for r in rows]
    edges = [{"src": int(r.get("parent_id", -1)), "dst": int(r.get("node_id", -1)), "edge_type": r.get("edge_type", "quadrant")} for r in rows if int(r.get("parent_id", -1)) >= 0]
    levels: Dict[int, List[int]] = {}
    for r in rows:
        levels.setdefault(int(r.get("depth", 0)), []).append(int(r.get("node_id", -1)))
    return {"kind": "flat_block_dag", "route": int(route), "nodes": nodes, "edges": edges, "levels": levels}

def product_step_search_for_route(T: int, names: List[str], coeffs: torch.Tensor, topk: int = 8) -> List[Dict[str, Any]]:
    target=_canonical_operator_matrix(T, names, coeffs)
    mats={nm: _template_matrix(T, nm) for nm in names}
    rows=[]
    for nm,M in mats.items():
        rows.append({"kind":"single", "bank_kind": bank_kind, "step0":nm, "step1":"", "rel":_matrix_rel(M,target)})
    for a,A in mats.items():
        for b,B in mats.items():
            P=(A@B).clamp_min(0.0); P=P/P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            rows.append({"kind":"product2", "bank_kind": bank_kind, "step0":a, "step1":b, "rel":_matrix_rel(P,target)})
    rows.sort(key=lambda r: float(r["rel"]))
    return rows[:topk]


def route_token_statistics(rows: List[Dict[str, Any]], labels: List[int]) -> List[Dict[str, Any]]:
    out=[]
    for cid in sorted(set(labels)):
        cr=[r for r, lab in zip(rows, labels) if int(lab)==int(cid)]
        if not cr: continue
        suite_counts={}; kind_counts={}; toks={}; poss=[]; lens=[]
        for r in cr:
            suite=str(r.get("suite", "?")); suite_counts[suite]=suite_counts.get(suite,0)+1
            tok=str(r.get("token", "")); k=token_kind(tok); kind_counts[k]=kind_counts.get(k,0)+1
            toks[tok]=toks.get(tok,0)+1; poss.append(float(r.get("pos_rel",0.0))); lens.append(len(_clean_token(tok)))
        n=len(cr)
        out.append({"cluster": int(cid), "rows": n, "suite_distribution": {k: round(v/n,4) for k,v in sorted(suite_counts.items())}, "token_kind_distribution": {k: round(v/n,4) for k,v in sorted(kind_counts.items())}, "pos_rel_mean": safe_mean(poss), "token_len_mean": safe_mean(lens), "top_tokens": sorted(toks.items(), key=lambda kv: -kv[1])[:20]})
    return out

def fit_route_operator_programs(train: List[SeqHeadData], val: List[SeqHeadData], labels_train: List[int], labels_val: List[int],
                                rows_train: List[Dict[str, Any]], rows_val: List[Dict[str, Any]], route_summaries: List[Dict[str, Any]],
                                o_w: torch.Tensor, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    names = _parse_templates(getattr(args, "operator_templates", ""))
    ridge = float(getattr(args, "operator_ridge", 1e-4))
    train_masks = {cid: make_row_masks(train, rows_train, labels_train, cid) for cid in sorted(set(labels_train))}
    val_masks = {cid: make_row_masks(val, rows_val, labels_val, cid) for cid in sorted(set(labels_train))}
    route_info_by_id = {int(r.get("cluster", -1)): r for r in route_summaries}
    rows=[]; bank={"kind":"attention_route_operator_bank", "templates": names, "routes": []}
    for cid in sorted(set(labels_train)):
        ntr = sum(int(m.sum()) for m in train_masks[cid].values())
        nv = sum(int(m.sum()) for m in val_masks.get(cid, {}).values())
        if ntr < int(getattr(args, "min_route_rows", 0)):
            continue
        coeffs = _fit_operator_coefficients(train, train_masks[cid], names, ridge)
        ev_tr = _eval_operator_fit(train, train_masks[cid], names, coeffs, o_w)
        ev_val = _eval_operator_fit(val, val_masks.get(cid, {}), names, coeffs, o_w)
        formula = operator_formula(names, coeffs, max_terms=int(getattr(args, "operator_max_terms", 6)))
        family_sweep = []
        for fam in sorted(set(operator_family(nm) for nm in names)):
            cf = _mask_coeffs_by_family(names, coeffs, fam)
            if float(cf.sum()) <= 1e-9:
                continue
            evf = _eval_operator_fit(val, val_masks.get(cid, {}), names, cf, o_w)
            family_sweep.append({"family": fam, "val_rel": evf.get("rel"), "val_top1": evf.get("top1"), "val_y_rel": evf.get("y_rel"), "formula": operator_formula(names, cf, max_terms=6)})
        family_sweep.sort(key=lambda x: float(x.get("val_rel") if x.get("val_rel") is not None else 999.0))
        rb_size = int(getattr(args, "route_bank_size", 64))
        product_steps = product_step_search_for_route(rb_size, names, coeffs, topk=int(getattr(args, "product_topk", 8)))
        forest = route_matrix_block_forest(_canonical_operator_matrix(rb_size, names, coeffs), int(cid), max_depth=int(getattr(args, "block_forest_depth", 2)))
        rinfo = route_info_by_id.get(int(cid), {})
        row = {
            "route": int(cid), "route_type": rinfo.get("route_type"), "route_verdict": rinfo.get("route_verdict"),
            "rows_train": ntr, "rows_val": nv, "formula": formula,
            "train_rel": ev_tr.get("rel"), "val_rel": ev_val.get("rel"),
            "train_kl": ev_tr.get("kl"), "val_kl": ev_val.get("kl"),
            "train_top1": ev_tr.get("top1"), "val_top1": ev_val.get("top1"),
            "train_y_rel": ev_tr.get("y_rel"), "val_y_rel": ev_val.get("y_rel"),
            "coefficients": {nm: float(c) for nm, c in zip(names, coeffs.detach().cpu().tolist())},
            "best_family": family_sweep[0] if family_sweep else None,
            "family_sweep": family_sweep,
            "product_steps": product_steps,
            "block_forest_top": forest[:12],
        }
        rows.append(row)
        bank["routes"].append(row)
    return rows, bank




def _coerce_coeffs_from_row(row: Dict[str, Any], names: List[str]) -> torch.Tensor:
    coeffs = row.get("coefficients", {})
    if isinstance(coeffs, str):
        try: coeffs = json.loads(coeffs)
        except Exception: coeffs = {}
    return torch.tensor([float(coeffs.get(nm, 0.0)) for nm in names], dtype=torch.float32)


def build_route_bank_artifacts(hdir: Path, route_operator_bank: Dict[str, Any], routes: List[Dict[str, Any]], route_bank_size: int = 64, block_forest_depth: int = 2) -> Dict[str, Any]:
    hdir=Path(hdir); ensure_dir(hdir)
    names=list(route_operator_bank.get("templates") or DEFAULT_OPERATOR_TEMPLATES)
    route_rows=list(route_operator_bank.get("routes") or [])
    if not route_rows:
        (hdir/"route_bank_blocks.csv").write_text("", encoding="utf-8")
        (hdir/"route_block_forest.csv").write_text("", encoding="utf-8")
        (hdir/"route_product_steps.csv").write_text("", encoding="utf-8")
        return {"kind":"route_bank_matrix", "enabled":False, "reason":"no route operator rows"}
    route_by_id={str(r.get("cluster")): r for r in routes}
    mats=[]; block_rows=[]; forest_rows=[]; product_rows=[]; deep_rows=[]
    for rr in route_rows:
        cid=int(rr.get("route")); coeffs=_coerce_coeffs_from_row(rr, names)
        M=_canonical_operator_matrix(int(route_bank_size), names, coeffs); mats.append(M)
        energy=float((M.float()**2).sum().sqrt().item()); rinfo=route_by_id.get(str(cid), {})
        block_rows.append({"route":cid, "route_type":rr.get("route_type") or rinfo.get("route_type"), "rows_train":rr.get("rows_train"), "rows_val":rr.get("rows_val"), "block_size":int(route_bank_size), "energy":energy, "operator_fit_val_rel":rr.get("val_rel"), "operator_fit_val_top1":rr.get("val_top1"), "formula":rr.get("formula")})
        forest_rows.extend(route_matrix_block_forest(M, cid, max_depth=int(block_forest_depth)))
        ps=rr.get("product_steps") or product_step_search_for_route(int(route_bank_size), names, coeffs)
        if isinstance(ps, str):
            try: ps=json.loads(ps)
            except Exception: ps=[]
        for pr in ps[:12]:
            row=dict(pr); row["route"]=cid; product_rows.append(row)
        bf=rr.get("best_family")
        if isinstance(bf, str):
            try: bf=json.loads(bf)
            except Exception: bf=None
        deep_rows.append({"route":cid, "route_type":rr.get("route_type") or rinfo.get("route_type"), "flat_formula":rr.get("formula"), "flat_val_rel":rr.get("val_rel"), "flat_val_top1":rr.get("val_top1"), "best_family":(bf or {}).get("family") if isinstance(bf, dict) else None, "best_family_val_rel":(bf or {}).get("val_rel") if isinstance(bf, dict) else None, "best_product":(ps[0] if ps else None)})
    bank=torch.block_diag(*mats) if mats else torch.empty(0,0)
    torch.save({"kind":"route_bank_matrix", "templates":names, "route_bank_size":int(route_bank_size), "matrix":bank, "routes":block_rows}, hdir/"route_bank_matrix.pt")
    write_csv(hdir/"route_bank_blocks.csv", block_rows)
    write_csv(hdir/"route_block_forest.csv", forest_rows)
    write_csv(hdir/"route_product_steps.csv", product_rows)
    write_csv(hdir/"route_operator_deep_decode.csv", deep_rows)
    info={"kind":"route_bank_matrix", "enabled":True, "routes":len(block_rows), "route_bank_size":int(route_bank_size), "matrix_shape":list(bank.shape), "blocks":block_rows}
    (hdir/"route_bank_matrix.json").write_text(json_dumps_safe(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


# -----------------------------------------------------------------------------
# V5 rich-operator override block
# -----------------------------------------------------------------------------
# This block intentionally redefines the operator-decomposition functions above.
# The fast QK collection/cache/patch code remains from v4.  The additions here:
#   * richer token/type attention bank
#   * Qtype -> Ktype and mined residual-pair atoms
#   * dynamic Hidden/Key/QK-program atoms
#   * per-route qtype subroutes
#   * functional Z/Y-aware model selection
#   * true/template/canonical route banks
#   * learned product-step probes over route-bank matrices

RICH_OPERATOR_TEMPLATES = [
    # anchors / simple causal reads
    "BOSRead", "IdentitySelf", "PrevToken", "LastTokenRead",
    "StrictUniformPast", "UniformPast", "CausalDistanceDecay", "DistanceDecay2", "DistanceDecay4", "DistanceDecay8", "DistanceDecay16",
    "LocalWindow1", "LocalWindow2", "LocalWindow4", "LocalWindow8", "LocalWindow16", "LocalPastWindow32",
    # token-type reads
    "ContentRead", "PrevContentRead", "CapitalizedRead", "PrevCapitalizedRead",
    "RoleRead", "PrevRoleRead", "CodeRead", "PrevCodeRead", "ShortRead", "PrevShortRead",
    "PunctPast", "BracketPast", "OperatorPast", "NumberPast", "WordPast", "CodeMixedPast",
    "SameTokenPast", "SameKindPast",
    # query-conditioned type routing
    "QContentToKContentRead", "QPunctToKContentRead", "QContentToKPunctRead",
    "QRoleToKPunctRead", "QCodeToKPunctRead", "QNumberToKNumberRead",
    "QCodeToKCodeRead", "QPunctToKPunctRead", "QWordToKWordRead", "QCapitalizedToKContentRead",
    # dynamic atoms: not static templates; they are built from real X/Q/K/factors per sequence
    "HiddenSimilarityPast", "KeySimilarityPast", "GlobalQKProgram", "RouteQKProgram",
]
DEFAULT_OPERATOR_TEMPLATES = list(RICH_OPERATOR_TEMPLATES)

_KIND_CANON = {
    "punctuation": "punct", "punct_bracket": "punct", "punctpast": "punct",
    "brackets": "bracket", "ops": "operator", "op": "operator",
    "identifier": "word", "ident": "word", "words": "word",
    "capital": "capitalized", "cap": "capitalized",
    "code_mixed": "code", "codemixed": "code", "codepast": "code",
    "num": "number", "numbers": "number",
    "roles": "role", "start": "role", "special": "role",
    "text": "content", "semantic": "content",
}

def _canon_kind_name(k: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "", str(k or "")).strip().lower()
    return _KIND_CANON.get(s, s)

def _kind_template_name(k: str) -> str:
    k = _canon_kind_name(k)
    return {
        "punct": "Punct", "bracket": "Bracket", "operator": "Operator", "number": "Number",
        "word": "Word", "capitalized": "Capitalized", "content": "Content", "code": "Code",
        "short": "Short", "role": "Role", "other": "Other", "alnum": "Alnum", "empty": "Empty",
    }.get(k, str(k).capitalize())

def _clean_token(tok: str) -> str:
    t = str(tok)
    # Qwen/SPM/BPE whitespace and newline markers commonly seen in tokens.
    for p in ("Ġ", "▁", "Ċ", "</w>"):
        t = t.replace(p, "")
    return t.strip()

def _is_role_token(t: str) -> bool:
    raw = str(t)
    c = _clean_token(raw).strip(" :\t\n\r").lower()
    if not c:
        return False
    role_words = {"user", "assistant", "system", "developer", "question", "answer", "human", "bot"}
    return c in role_words or "im_start" in raw or "im_end" in raw or raw.startswith("<|")

def token_kind(tok: str) -> str:
    t = _clean_token(tok)
    if not t:
        return "empty"
    if _is_role_token(tok):
        return "role"
    if re.fullmatch(r"[\]\[\(\)\{\}<>]+", t):
        return "bracket"
    if re.fullmatch(r"[,. ;:!?]+".replace(" ", ""), t) or re.fullmatch(r"[,.!?;:]+", t):
        return "punct"
    if re.fullmatch(r"[-+*/=<>!&|%^~]+", t):
        return "operator"
    if re.fullmatch(r"[0-9]+([.,][0-9]+)?", t):
        return "number"
    # mixed code-ish tokens: identifiers carrying syntax, paths, flags, JSON-ish fragments.
    code_words = {"def", "class", "return", "for", "while", "if", "else", "elif", "import", "from", "function", "const", "let", "var", "sql", "select", "where", "join", "true", "false", "none", "null", "self", "this", "range", "append"}
    if t.lower() in code_words:
        return "code"
    if re.search(r"[A-Za-z_]", t) and re.search(r"[{}()\[\];:=+*/<>\\/\-]", t):
        return "code"
    if re.fullmatch(r"[_A-Za-z][_A-Za-z0-9]*", t):
        return "capitalized" if t[:1].isupper() else "word"
    if re.search(r"[0-9]", t) and re.search(r"[A-Za-z_]", t):
        return "alnum"
    return "other"

def _token_len(tok: str) -> int:
    return len(_clean_token(tok))

def _token_matches_kind(tok: str, kind: str) -> bool:
    k = _canon_kind_name(kind)
    tk = token_kind(tok)
    if k in ("any", "all"):
        return True
    if k == tk:
        return True
    if k == "punct":
        return tk in ("punct", "bracket", "operator")
    if k == "code":
        return tk in ("code", "operator", "bracket")
    if k == "content":
        return tk in ("word", "capitalized", "alnum", "other", "code")
    if k == "word":
        return tk in ("word", "capitalized")
    if k == "short":
        return tk not in ("empty",) and _token_len(tok) <= 2
    if k == "role":
        return tk == "role"
    return False

def _token_match_indices(tokens: Optional[List[str]], i: int, mode: str, *, include_current: bool = True) -> List[int]:
    if not tokens:
        return []
    i = max(0, min(int(i), len(tokens) - 1))
    cur = _clean_token(tokens[i])
    cur_kind = token_kind(tokens[i])
    hi = i + 1 if include_current else i
    out: List[int] = []
    mode_c = _canon_kind_name(mode)
    for j in range(max(0, hi)):
        tj = _clean_token(tokens[j])
        kj = token_kind(tokens[j])
        if mode_c == "same_token" and tj and cur and tj == cur:
            out.append(j)
        elif mode_c == "same_kind" and kj == cur_kind:
            out.append(j)
        elif _token_matches_kind(tokens[j], mode_c):
            out.append(j)
    return out

def _parse_templates(s: str) -> List[str]:
    names = [x.strip() for x in (s or "").replace(';', ',').split(',') if x.strip()]
    return names or list(DEFAULT_OPERATOR_TEMPLATES)

def _normalize_indices_row(v: torch.Tensor, indices: List[int], fallback: Optional[str], i: int) -> torch.Tensor:
    if indices:
        idx = torch.tensor(indices, dtype=torch.long, device=v.device)
        idx = idx[(idx >= 0) & (idx < v.numel())]
        if idx.numel() > 0:
            v[idx] = 1.0
    if float(v.sum()) <= 1e-12 and fallback:
        if fallback == "uniform":
            v[: i + 1] = 1.0
        elif fallback == "self":
            v[i] = 1.0
    if float(v.sum()) > 1e-12:
        v[:] = v / v.sum().clamp_min(1e-12)
    return v

def _parse_q_to_k_name(name: str) -> Optional[Tuple[str, str]]:
    n = str(name)
    m = re.match(r"(?:Mined_)?Q([A-Za-z0-9_]+?)ToK([A-Za-z0-9_]+?)(?:Read)?$", n)
    if not m:
        return None
    return _canon_kind_name(m.group(1)), _canon_kind_name(m.group(2))

def attention_template_row(T: int, i: int, name: str, device: str = "cpu", tokens: Optional[List[str]] = None) -> torch.Tensor:
    """Causal attention-row operator basis.

    Static operators use token strings when available.  Query-conditioned Q*ToK*
    atoms return a zero row when the query token does not match; this is deliberate
    because they should not silently become UniformPast on unrelated rows.
    """
    v = torch.zeros(T, dtype=torch.float32, device=device)
    if T <= 0:
        return v
    i = int(max(0, min(i, T - 1)))
    raw = str(name)
    n = raw.lower()

    qk = _parse_q_to_k_name(raw)
    if qk is not None:
        q_kind, k_kind = qk
        if tokens and i < len(tokens) and _token_matches_kind(tokens[i], q_kind):
            return _normalize_indices_row(v, _token_match_indices(tokens, i, k_kind, include_current=True), None, i)
        return v

    if n in ("bos", "bosread", "pos_bosread"):
        v[0] = 1.0
    elif n in ("self", "identityself", "pos_identityself", "lasttokenread", "lastavailableread"):
        v[i] = 1.0
    elif n in ("prev", "prevtoken", "pos_prevtoken"):
        v[max(0, i - 1)] = 1.0
    elif n.startswith("localpastwindow") or n.startswith("localwindow") or n.startswith("local"):
        m = re.search(r"(\d+)", n)
        r = int(m.group(1)) if m else 4
        v[max(0, i - r): i + 1] = 1.0
        v /= v.sum().clamp_min(1e-12)
    elif n in ("strictuniformpast", "uniformpast", "pastuniform", "diffusepast"):
        v[: i + 1] = 1.0
        v /= v.sum().clamp_min(1e-12)
    elif n.startswith("causaldistancedecay") or n.startswith("distancedecay") or n.startswith("decay"):
        m = re.search(r"(\d+)", n)
        tau = float(m.group(1)) if m else (8.0 if n.startswith("causal") else 4.0)
        js = torch.arange(i + 1, dtype=torch.float32, device=device)
        vv = torch.exp(-(float(i) - js) / max(1e-6, tau))
        v[: i + 1] = vv / vv.sum().clamp_min(1e-12)
    elif n in ("punctpast", "punctuationpast"):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "punct"), "uniform", i)
    elif n in ("bracketpast", "bracketread"):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "bracket"), "uniform", i)
    elif n in ("operatorpast", "operatorread"):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "operator"), "uniform", i)
    elif n in ("numberpast", "numberread"):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "number"), "uniform", i)
    elif n in ("wordpast", "identifierpast", "wordread"):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "word"), "uniform", i)
    elif n in ("codemixedpast", "codepast", "coderead"):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "code"), "uniform", i)
    elif n in ("contentread",):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "content"), "uniform", i)
    elif n in ("prevcontentread",):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "content", include_current=False), "uniform", i)
    elif n in ("capitalizedread",):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "capitalized"), "uniform", i)
    elif n in ("prevcapitalizedread",):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "capitalized", include_current=False), "uniform", i)
    elif n in ("roleread",):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "role"), "uniform", i)
    elif n in ("prevroleread",):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "role", include_current=False), "uniform", i)
    elif n in ("shortread",):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "short"), "uniform", i)
    elif n in ("prevshortread",):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "short", include_current=False), "uniform", i)
    elif n in ("sametokenpast", "sametokenread"):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "same_token"), "uniform", i)
    elif n in ("samekindpast", "samekindread"):
        return _normalize_indices_row(v, _token_match_indices(tokens, i, "same_kind"), "uniform", i)
    else:
        # Dynamic operators cannot be materialized without sequence Q/K/X/factors;
        # canonical visualizations use UniformPast as a neutral fallback.
        v[: i + 1] = 1.0
        v /= v.sum().clamp_min(1e-12)
    return v

_DYNAMIC_OPERATOR_NAMES = {"hiddensimilaritypast", "keysimilaritypast", "querykeysimilaritypast", "globalqkprogram", "routeqkprogram", "qkfulloracle"}

def _is_dynamic_operator(name: str) -> bool:
    return str(name).lower() in _DYNAMIC_OPERATOR_NAMES

def _causal_softmax_cpu(scores: torch.Tensor) -> torch.Tensor:
    return causal_softmax(scores.float().cpu())

def _lowrank_qk_A(seq: SeqHeadData, factor: Optional[Dict[str, Any]], scale_mode: str) -> Optional[torch.Tensor]:
    if not factor:
        return None
    Pq = factor.get("Pq")
    Pk = factor.get("Pk")
    if Pq is None or Pk is None:
        return None
    Pq = Pq.detach().cpu().float()
    Pk = Pk.detach().cpu().float()
    rank = int(Pq.shape[1])
    denom = math.sqrt(rank if scale_mode == "rank" else int(Pq.shape[0]))
    Ql = seq.Q.float() @ Pq
    Kl = seq.K.float() @ Pk
    return _causal_softmax_cpu((Ql @ Kl.T) / denom)

def _dynamic_operator_matrix_for_seq(seq: SeqHeadData, name: str, dynamic_ctx: Optional[Dict[str, Any]], route_id: Optional[int]) -> torch.Tensor:
    n = str(name).lower()
    T = int(seq.A.shape[0])
    if n == "hiddensimilaritypast":
        X = F.normalize(seq.X.float(), dim=-1)
        return _causal_softmax_cpu((X @ X.T) * math.sqrt(max(1, int(X.shape[-1]))))
    if n == "keysimilaritypast":
        K = F.normalize(seq.K.float(), dim=-1)
        return _causal_softmax_cpu((K @ K.T) * math.sqrt(max(1, int(K.shape[-1]))))
    if n == "querykeysimilaritypast":
        return _causal_softmax_cpu((seq.Q.float() @ seq.K.float().T) / math.sqrt(max(1, int(seq.Q.shape[-1]))))
    ctx = dynamic_ctx or {}
    if n == "globalqkprogram":
        A = _lowrank_qk_A(seq, ctx.get("global"), str(ctx.get("qk_scale", "rank")))
        if A is not None:
            return A
    if n == "routeqkprogram":
        routes = ctx.get("routes") or {}
        fac = routes.get(f"route{route_id}") or routes.get(str(route_id)) or ctx.get("global")
        A = _lowrank_qk_A(seq, fac, str(ctx.get("qk_scale", "rank")))
        if A is not None:
            return A
    if n == "qkfulloracle" and bool((dynamic_ctx or {}).get("allow_oracle_qk", False)):
        return seq.A.float().cpu()
    return _operator_basis_matrix_for_seq(T, ["UniformPast"], tokens=seq.tokens)[0]

_OPERATOR_BASIS_CACHE = {}

def _operator_basis_matrix_for_seq(T: int, names: List[str], tokens: Optional[List[str]] = None) -> torch.Tensor:
    toks = tuple(tokens or ["?"] * T)
    key = (int(T), toks, tuple(names), "v5")
    B = _OPERATOR_BASIS_CACHE.get(key)
    if B is not None:
        return B
    mats = []
    for nm in names:
        rows = [attention_template_row(T, i, nm, device="cpu", tokens=list(toks)) for i in range(T)]
        mats.append(torch.stack(rows, dim=0))
    B = torch.stack(mats, dim=0).float().contiguous()
    if len(_OPERATOR_BASIS_CACHE) > 2048:
        _OPERATOR_BASIS_CACHE.clear()
    _OPERATOR_BASIS_CACHE[key] = B
    return B

def _operator_basis_matrix_for_seqdata(seq: SeqHeadData, names: List[str], dynamic_ctx: Optional[Dict[str, Any]] = None,
                                       route_id: Optional[int] = None) -> torch.Tensor:
    T = int(seq.A.shape[0])
    static_names = [nm for nm in names if not _is_dynamic_operator(nm)]
    static_B = _operator_basis_matrix_for_seq(T, static_names, tokens=seq.tokens) if static_names else None
    static_map = {nm: static_B[i] for i, nm in enumerate(static_names)} if static_B is not None else {}
    mats = []
    for nm in names:
        if _is_dynamic_operator(nm):
            mats.append(_dynamic_operator_matrix_for_seq(seq, nm, dynamic_ctx, route_id).float().cpu())
        else:
            mats.append(static_map[nm].float().cpu())
    return torch.stack(mats, dim=0).float().contiguous()

def _operator_basis_for_row(T: int, i: int, names: List[str], device: str = "cpu", tokens: Optional[List[str]] = None) -> torch.Tensor:
    return torch.stack([attention_template_row(T, i, nm, device=device, tokens=tokens) for nm in names], dim=0)

def _project_coeffs(c: torch.Tensor, mode: str = "nonnegative") -> torch.Tensor:
    mode = str(mode or "nonnegative").lower()
    if mode in ("signed", "signed_ridge"):
        return c.float()
    if mode in ("signed_clip", "clip"):
        return c.float().clamp_min(0.0)
    c = c.float().clamp_min(0.0)
    if mode in ("simplex", "nonnegative_simplex"):
        s = float(c.sum())
        if s > 1e-12:
            c = c / s
    return c

def _fit_operator_coefficients(seqs: List[SeqHeadData], row_masks: Dict[int, torch.Tensor], names: List[str], ridge: float,
                               dynamic_ctx: Optional[Dict[str, Any]] = None, route_id: Optional[int] = None,
                               fit_mode: str = "nonnegative") -> torch.Tensor:
    M = len(names)
    G = torch.zeros(M, M, dtype=torch.float64)
    b = torch.zeros(M, dtype=torch.float64)
    nrows = 0
    for s in seqs:
        mask = row_masks.get(s.prompt_id)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool()
        if int(mask.sum()) == 0:
            continue
        B_all = _operator_basis_matrix_for_seqdata(s, names, dynamic_ctx=dynamic_ctx, route_id=route_id).double()
        Br = B_all[:, mask, :].permute(1, 0, 2).contiguous()
        Y = s.A.detach().cpu().double()[mask]
        G += torch.einsum("rmt,rnt->mn", Br, Br)
        b += torch.einsum("rmt,rt->m", Br, Y)
        nrows += int(mask.sum())
    if nrows == 0:
        return torch.zeros(M, dtype=torch.float32)
    G += float(ridge) * torch.eye(M, dtype=torch.float64)
    try:
        c = torch.linalg.solve(G, b)
    except Exception:
        c = torch.linalg.lstsq(G, b[:, None]).solution[:, 0]
    c = _project_coeffs(c.float(), fit_mode)
    if float(c.abs().sum()) <= 1e-9:
        c = torch.ones(M, dtype=torch.float32) / max(1, M)
    return c.float()

def _mix_basis_to_A(B_all: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    c = coeffs.detach().float().cpu()
    A = torch.einsum("m,mij->ij", c, B_all.float().cpu())
    A = A.clamp_min(0.0)
    empty = A.sum(dim=-1) <= 1e-12
    if bool(empty.any()):
        T = A.shape[0]
        fallback = _operator_basis_matrix_for_seq(T, ["UniformPast"], tokens=None)[0]
        A[empty] = fallback[empty]
    A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return A

def _make_operator_rows_for_seq(T: int, names: List[str], coeffs: torch.Tensor, device: str = "cpu", tokens: Optional[List[str]] = None) -> torch.Tensor:
    B_all = _operator_basis_matrix_for_seq(T, names, tokens=tokens)
    return _mix_basis_to_A(B_all, coeffs).to(device)

def _make_operator_rows_for_seqdata(seq: SeqHeadData, names: List[str], coeffs: torch.Tensor,
                                    dynamic_ctx: Optional[Dict[str, Any]] = None, route_id: Optional[int] = None) -> torch.Tensor:
    B_all = _operator_basis_matrix_for_seqdata(seq, names, dynamic_ctx=dynamic_ctx, route_id=route_id)
    return _mix_basis_to_A(B_all, coeffs)

def _eval_operator_fit(seqs: List[SeqHeadData], row_masks: Dict[int, torch.Tensor], names: List[str], coeffs: torch.Tensor,
                       o_w: torch.Tensor, device: str = "cpu", dynamic_ctx: Optional[Dict[str, Any]] = None,
                       route_id: Optional[int] = None) -> Dict[str, float]:
    rels=[]; kls=[]; top1s=[]; zrels=[]; yrels=[]; nrows=0
    Ow = o_w.detach().cpu().float()
    for s in seqs:
        mask = row_masks.get(s.prompt_id)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool()
        if int(mask.sum()) == 0:
            continue
        A_hat = _make_operator_rows_for_seqdata(s, names, coeffs, dynamic_ctx=dynamic_ctx, route_id=route_id)
        A_true = s.A.detach().cpu().float()
        Ah = A_hat[mask]
        At = A_true[mask]
        rels.append(rel_err(Ah, At))
        kls.append(float(F.kl_div((Ah + 1e-12).log(), At, reduction="batchmean")))
        top1s.append(float((Ah.argmax(dim=-1) == At.argmax(dim=-1)).float().mean()))
        V = s.V.detach().cpu().float()
        Zhat_full = A_hat @ V
        Ztrue_full = A_true @ V
        zrels.append(rel_err(Zhat_full[mask], Ztrue_full[mask]))
        Yhat_full = Zhat_full @ Ow.T
        Ytrue_full = s.Y.detach().cpu().float()
        yrels.append(rel_err(Yhat_full[mask], Ytrue_full[mask]))
        nrows += int(mask.sum())
    return {"rel": safe_mean(rels), "kl": safe_mean(kls), "top1": safe_mean(top1s), "z_rel": safe_mean(zrels), "y_rel": safe_mean(yrels), "rows": nrows}

def operator_formula(names: List[str], coeffs: torch.Tensor, max_terms: int = 6) -> str:
    c0 = coeffs.detach().cpu().float()
    # Display positive effective mass; signed fits are clipped by the attention row renderer anyway.
    cs = c0.clamp_min(0.0)
    s = float(cs.sum())
    if s > 1e-12:
        cs = cs / s
    pairs = [(nm, float(c)) for nm, c in zip(names, cs.tolist()) if c > 1e-6]
    pairs.sort(key=lambda x: -abs(x[1]))
    return "0" if not pairs else " + ".join(f"{c:.3f}*{nm}" for nm, c in pairs[:max_terms])

def operator_family(name: str) -> str:
    n = str(name).lower()
    if n in ("bosread", "identityself", "prevtoken", "lasttokenread"):
        return "anchor"
    if n.startswith("local"):
        return "local"
    if n.startswith("distancedecay") or n.startswith("causaldistancedecay") or n in ("uniformpast", "strictuniformpast"):
        return "diffuse"
    if _parse_q_to_k_name(name) is not None:
        return "mined_qtype_to_ktype" if n.startswith("mined_") else "qtype_to_ktype"
    if _is_dynamic_operator(name):
        return "dynamic_qk" if "qk" in n or "program" in n else "dynamic_similarity"
    if any(x in n for x in ("content", "capitalized", "role", "code", "short", "punct", "bracket", "operator", "number", "word")):
        return "token_type"
    if n in ("sametokenpast", "samekindpast"):
        return "repeat_or_kind"
    return "other"

def _mask_coeffs_by_family(names: List[str], coeffs: torch.Tensor, family: str) -> torch.Tensor:
    c = coeffs.detach().clone().float()
    for i, nm in enumerate(names):
        if operator_family(nm) != family:
            c[i] = 0.0
    return c

def _operator_selection_score(ev: Dict[str, float], objective: str = "functional_balanced") -> float:
    obj = str(objective or "functional_balanced").lower()
    rel = float(ev.get("rel", 999.0) or 999.0)
    y = float(ev.get("y_rel", 999.0) if ev.get("y_rel", -1.0) is not None else 999.0)
    z = float(ev.get("z_rel", 999.0) if ev.get("z_rel", -1.0) is not None else 999.0)
    top = float(ev.get("top1", 0.0) or 0.0)
    if obj in ("a", "arel", "attention"):
        return rel - 0.03 * top
    if obj in ("y", "functional", "head"):
        return y + 0.20 * rel - 0.05 * top
    return 0.50 * rel + 0.25 * z + 0.50 * y - 0.08 * top

def _route_factor_context_from_bank(factor_bank: Dict[Tuple[str, str, int], Dict[str, torch.Tensor]],
                                    qk_rows: List[Dict[str, Any]], best_global: Optional[Dict[str, Any]],
                                    args: argparse.Namespace) -> Dict[str, Any]:
    ctx = {"qk_scale": getattr(args, "qk_scale", "rank"), "global": {}, "routes": {}, "allow_oracle_qk": bool(getattr(args, "operator_allow_oracle_qk", False))}
    if best_global:
        key = (str(best_global.get("scope")), str(best_global.get("method")), int(best_global.get("rank")))
        if key in factor_bank:
            ctx["global"] = factor_bank[key]
    route_scopes = sorted(set(str(r.get("scope")) for r in qk_rows if str(r.get("scope", "")).startswith("route")))
    for scope in route_scopes:
        br = select_best_qk([r for r in qk_rows if str(r.get("scope")) == scope], getattr(args, "select_policy", "balanced"), getattr(args, "select_eps_rel", 0.0), getattr(args, "select_eps_y", 0.0))
        if br is None:
            continue
        key = (str(br.get("scope")), str(br.get("method")), int(br.get("rank")))
        if key in factor_bank:
            ctx["routes"][scope] = factor_bank[key]
    return ctx

def mine_attention_residual_pairs(seqs: List[SeqHeadData], row_masks: Dict[int, torch.Tensor], names: List[str], coeffs: torch.Tensor,
                                  topk: int = 8, min_score: float = 1e-4,
                                  dynamic_ctx: Optional[Dict[str, Any]] = None, route_id: Optional[int] = None) -> List[Dict[str, Any]]:
    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    total_rows = 0
    for s in seqs:
        mask = row_masks.get(s.prompt_id)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool()
        if int(mask.sum()) == 0:
            continue
        A_hat = _make_operator_rows_for_seqdata(s, names, coeffs, dynamic_ctx=dynamic_ctx, route_id=route_id)
        R = (s.A.detach().cpu().float() - A_hat).clamp_min(0.0)
        idxs = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        total_rows += len(idxs)
        for i in idxs:
            if i >= len(s.tokens):
                continue
            qk = token_kind(s.tokens[i])
            row = R[i]
            nz = torch.nonzero(row > 0, as_tuple=False).flatten().tolist()
            for j in nz:
                if j > i or j >= len(s.tokens):
                    continue
                kk = token_kind(s.tokens[j])
                val = float(row[j])
                if val <= 0:
                    continue
                d = stats.setdefault((qk, kk), {"mass": 0.0, "edges": 0.0})
                d["mass"] += val
                d["edges"] += 1.0
    mined = []
    denom = max(1, total_rows)
    for (qk, kk), d in stats.items():
        if qk in ("empty",) or kk in ("empty",):
            continue
        score = float(d["mass"]) / denom
        if score < min_score:
            continue
        name = f"Mined_Q{_kind_template_name(qk)}ToK{_kind_template_name(kk)}Read"
        mined.append({"name": name, "q_kind": qk, "k_kind": kk, "score": score, "mass": d["mass"], "edges": int(d["edges"]), "rows": total_rows})
    mined.sort(key=lambda r: -float(r["score"]))
    # stable unique names
    out=[]; seen=set()
    for r in mined:
        if r["name"] in seen:
            continue
        seen.add(r["name"]); out.append(r)
        if len(out) >= int(topk):
            break
    return out

def _make_subroute_masks(seqs: List[SeqHeadData], base_masks: Dict[int, torch.Tensor], kind: str) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for s in seqs:
        base = base_masks.get(s.prompt_id)
        m = torch.zeros(s.A.shape[0], dtype=torch.bool)
        if base is not None:
            base = base.detach().cpu().bool()
            for i in torch.nonzero(base, as_tuple=False).flatten().tolist():
                if i < len(s.tokens) and _token_matches_kind(s.tokens[i], kind):
                    m[i] = True
        out[s.prompt_id] = m
    return out

def _mask_rows_count(masks: Dict[int, torch.Tensor]) -> int:
    return int(sum(int(m.detach().cpu().bool().sum()) for m in masks.values()))

def _fit_best_operator_variant(train: List[SeqHeadData], val: List[SeqHeadData], tr_mask: Dict[int, torch.Tensor], va_mask: Dict[int, torch.Tensor],
                               names: List[str], ridge: float, o_w: torch.Tensor, args: argparse.Namespace,
                               dynamic_ctx: Optional[Dict[str, Any]], route_id: Optional[int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    variants = ["nonnegative", "simplex", "signed_clip"]
    best = None
    for mode in variants:
        c = _fit_operator_coefficients(train, tr_mask, names, ridge, dynamic_ctx=dynamic_ctx, route_id=route_id, fit_mode=mode)
        ev_val = _eval_operator_fit(val, va_mask, names, c, o_w, dynamic_ctx=dynamic_ctx, route_id=route_id)
        ev_tr = _eval_operator_fit(train, tr_mask, names, c, o_w, dynamic_ctx=dynamic_ctx, route_id=route_id)
        score = _operator_selection_score(ev_val, getattr(args, "operator_selection", "functional_balanced"))
        item = {"mode": mode, "coeffs": c, "score": score, "train": ev_tr, "val": ev_val}
        if best is None or score < best["score"]:
            best = item
    return best["coeffs"], best


# ---------------- score/logit-level operators (v6) ----------------
# Attention-level operators fit A directly. Score-level operators fit centered
# pre-softmax logits S, then evaluate A_hat = causal_softmax(S_hat).  This is
# critical for Qwen routes where BOS/local/type behavior is caused by score
# biases before softmax rather than by a linear mixture of final attention rows.

SCORE_OPERATOR_TEMPLATES = [
    "ScoreBOSBias", "ScoreSelfBias", "ScorePrevBias", "ScoreLastBias",
    "ScoreLocal2Bias", "ScoreLocal4Bias", "ScoreLocal8Bias", "ScoreLocal16Bias",
    "ScoreDistanceDecay2", "ScoreDistanceDecay4", "ScoreDistanceDecay8", "ScoreDistancePenalty",
    "ScoreKContentBias", "ScoreKPunctBias", "ScoreKCodeBias", "ScoreKNumberBias", "ScoreKRoleBias",
    "ScoreSameTokenBias", "ScoreSameKindBias",
    "ScoreQContentToKContentBias", "ScoreQPunctToKContentBias", "ScoreQContentToKPunctBias",
    "ScoreQRoleToKPunctBias", "ScoreQCodeToKPunctBias", "ScoreQNumberToKNumberBias",
    "ScoreQCodeToKCodeBias", "ScoreQPunctToKPunctBias", "ScoreQWordToKWordBias",
    "ScoreHiddenSimilarity", "ScoreKeySimilarity", "ScoreGlobalQK", "ScoreRouteQK",
]

_SCORE_DYNAMIC_NAMES = {"scorehiddensimilarity", "scorekeysimilarity", "scoreglobalqk", "scorerouteqk", "scorefullqk"}


def _is_score_dynamic_operator(name: str) -> bool:
    return str(name).lower() in _SCORE_DYNAMIC_NAMES


def _parse_score_templates(s: str) -> List[str]:
    names = [x.strip() for x in (s or "").replace(';', ',').split(',') if x.strip()]
    return names or list(SCORE_OPERATOR_TEMPLATES)


def _parse_score_q_to_k_name(name: str) -> Optional[Tuple[str, str]]:
    n = str(name)
    m = re.match(r"(?:ScoreMined_)?ScoreQ([A-Za-z0-9_]+?)ToK([A-Za-z0-9_]+?)(?:Bias)?$", n)
    if m:
        return _canon_kind_name(m.group(1)), _canon_kind_name(m.group(2))
    m = re.match(r"(?:ScoreMined_)?Q([A-Za-z0-9_]+?)ToK([A-Za-z0-9_]+?)(?:Score|Bias)?$", n)
    if m:
        return _canon_kind_name(m.group(1)), _canon_kind_name(m.group(2))
    return None


def _causal_mask(T: int) -> torch.Tensor:
    return torch.tril(torch.ones(int(T), int(T), dtype=torch.bool))


def _row_center_causal(M: torch.Tensor) -> torch.Tensor:
    """Subtract causal-row mean; softmax is invariant to row constants."""
    M = M.detach().cpu().float().clone()
    T = int(M.shape[-1])
    cm = _causal_mask(T)
    if M.dim() == 2:
        out = torch.zeros_like(M)
        for i in range(T):
            vals = M[i, :i + 1]
            out[i, :i + 1] = vals - vals.mean()
        return out.masked_fill(~cm, 0.0)
    if M.dim() == 3:
        out = torch.zeros_like(M)
        for i in range(T):
            vals = M[:, i, :i + 1]
            out[:, i, :i + 1] = vals - vals.mean(dim=-1, keepdim=True)
        return out.masked_fill(~cm.unsqueeze(0), 0.0)
    return M


def _attention_to_centered_logits(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    A = A.detach().cpu().float().clamp_min(float(eps))
    L = torch.log(A)
    return _row_center_causal(L)


def _lowrank_qk_score(seq: SeqHeadData, factor: Optional[Dict[str, Any]], scale_mode: str) -> Optional[torch.Tensor]:
    if not factor:
        return None
    Pq = factor.get("Pq")
    Pk = factor.get("Pk")
    if Pq is None or Pk is None:
        return None
    Pq = Pq.detach().cpu().float()
    Pk = Pk.detach().cpu().float()
    rank = int(Pq.shape[1])
    denom = math.sqrt(rank if scale_mode == "rank" else int(Pq.shape[0]))
    return (seq.Q.float() @ Pq) @ (seq.K.float() @ Pk).T / denom


def _score_static_matrix_for_seq(seq: SeqHeadData, name: str) -> torch.Tensor:
    T = int(seq.A.shape[0])
    M = torch.zeros(T, T, dtype=torch.float32)
    n = str(name).lower()
    qk = _parse_score_q_to_k_name(name)
    if qk is not None:
        q_kind, k_kind = qk
        for i in range(T):
            if i < len(seq.tokens) and _token_matches_kind(seq.tokens[i], q_kind):
                for j in range(i + 1):
                    if j < len(seq.tokens) and _token_matches_kind(seq.tokens[j], k_kind):
                        M[i, j] = 1.0
        return M
    if n in ("scorebosbias", "scorekbosbias"):
        M[:, 0] = 1.0
    elif n in ("scoreselfbias",):
        for i in range(T): M[i, i] = 1.0
    elif n in ("scoreprevbias",):
        for i in range(T): M[i, max(0, i - 1)] = 1.0
    elif n in ("scorelastbias",):
        for i in range(T): M[i, i] = 1.0
    elif n.startswith("scorelocal"):
        m = re.search(r"(\d+)", n)
        r = int(m.group(1)) if m else 4
        for i in range(T):
            M[i, max(0, i-r):i+1] = 1.0
    elif n.startswith("scoredistancedecay"):
        m = re.search(r"(\d+)", n)
        tau = float(m.group(1)) if m else 8.0
        for i in range(T):
            js = torch.arange(i + 1, dtype=torch.float32)
            M[i, :i+1] = torch.exp(-(float(i) - js) / max(1e-6, tau))
    elif n in ("scoredistancepenalty", "scoredistpenalty"):
        for i in range(T):
            js = torch.arange(i + 1, dtype=torch.float32)
            M[i, :i+1] = -(float(i) - js) / max(1.0, float(T))
    elif n in ("scoreuniformpastbias",):
        for i in range(T): M[i, :i+1] = 1.0
    elif n.startswith("scorek") and n.endswith("bias"):
        kind = n[len("scorek"):-len("bias")]
        kind = _canon_kind_name(kind)
        for i in range(T):
            for j in range(i + 1):
                if j < len(seq.tokens) and _token_matches_kind(seq.tokens[j], kind):
                    M[i, j] = 1.0
    elif n in ("scoresametokenbias",):
        for i in range(T):
            cur = _clean_token(seq.tokens[i]) if i < len(seq.tokens) else ""
            for j in range(i + 1):
                if j < len(seq.tokens) and cur and _clean_token(seq.tokens[j]) == cur:
                    M[i, j] = 1.0
    elif n in ("scoresamekindbias",):
        for i in range(T):
            qkind = token_kind(seq.tokens[i]) if i < len(seq.tokens) else "other"
            for j in range(i + 1):
                if j < len(seq.tokens) and token_kind(seq.tokens[j]) == qkind:
                    M[i, j] = 1.0
    else:
        # Unknown score atom: neutral zero score, not uniform attention.
        pass
    return M.masked_fill(~_causal_mask(T), 0.0)


def _score_dynamic_matrix_for_seq(seq: SeqHeadData, name: str, dynamic_ctx: Optional[Dict[str, Any]], route_id: Optional[int]) -> torch.Tensor:
    n = str(name).lower()
    T = int(seq.A.shape[0])
    if n == "scorehiddensimilarity":
        X = F.normalize(seq.X.float(), dim=-1)
        return (X @ X.T).masked_fill(~_causal_mask(T), 0.0)
    if n == "scorekeysimilarity":
        K = F.normalize(seq.K.float(), dim=-1)
        return (K @ K.T).masked_fill(~_causal_mask(T), 0.0)
    ctx = dynamic_ctx or {}
    if n == "scoreglobalqk":
        S = _lowrank_qk_score(seq, ctx.get("global"), str(ctx.get("qk_scale", "rank")))
        if S is not None:
            return S.float().cpu().masked_fill(~_causal_mask(T), 0.0)
    if n == "scorerouteqk":
        routes = ctx.get("routes") or {}
        fac = routes.get(f"route{route_id}") or routes.get(str(route_id)) or ctx.get("global")
        S = _lowrank_qk_score(seq, fac, str(ctx.get("qk_scale", "rank")))
        if S is not None:
            return S.float().cpu().masked_fill(~_causal_mask(T), 0.0)
    if n == "scorefullqk" and bool((dynamic_ctx or {}).get("allow_oracle_qk", False)):
        S = (seq.Q.float() @ seq.K.float().T) / math.sqrt(max(1, int(seq.Q.shape[-1])))
        return S.float().cpu().masked_fill(~_causal_mask(T), 0.0)
    return torch.zeros(T, T, dtype=torch.float32)


_SCORE_BASIS_CACHE = {}


def _score_basis_matrix_for_seqdata(seq: SeqHeadData, names: List[str], dynamic_ctx: Optional[Dict[str, Any]] = None,
                                    route_id: Optional[int] = None) -> torch.Tensor:
    T = int(seq.A.shape[0])
    toks = tuple(seq.tokens or ["?"] * T)
    static_names = [nm for nm in names if not _is_score_dynamic_operator(nm)]
    key = (int(T), toks, tuple(static_names), "score_v6")
    static_B = _SCORE_BASIS_CACHE.get(key)
    if static_B is None:
        static_B = torch.stack([_score_static_matrix_for_seq(seq, nm) for nm in static_names], dim=0).float() if static_names else torch.empty(0, T, T)
        # Score fit uses centered bases so row constants do not waste capacity.
        static_B = _row_center_causal(static_B)
        if len(_SCORE_BASIS_CACHE) > 2048:
            _SCORE_BASIS_CACHE.clear()
        _SCORE_BASIS_CACHE[key] = static_B
    static_map = {nm: static_B[i] for i, nm in enumerate(static_names)}
    mats = []
    for nm in names:
        if _is_score_dynamic_operator(nm):
            mats.append(_row_center_causal(_score_dynamic_matrix_for_seq(seq, nm, dynamic_ctx, route_id)))
        else:
            mats.append(static_map[nm])
    return torch.stack(mats, dim=0).float().contiguous()


def _fit_score_operator_coefficients(seqs: List[SeqHeadData], row_masks: Dict[int, torch.Tensor], names: List[str], ridge: float,
                                     dynamic_ctx: Optional[Dict[str, Any]] = None, route_id: Optional[int] = None) -> torch.Tensor:
    M = len(names)
    G = torch.zeros(M, M, dtype=torch.float64)
    b = torch.zeros(M, dtype=torch.float64)
    nrows = 0
    for s in seqs:
        mask = row_masks.get(s.prompt_id)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool()
        if int(mask.sum()) == 0:
            continue
        B_all = _score_basis_matrix_for_seqdata(s, names, dynamic_ctx=dynamic_ctx, route_id=route_id).double()
        Target = _attention_to_centered_logits(s.A.detach().cpu().float()).double()
        Br = B_all[:, mask, :].permute(1, 0, 2).contiguous()
        Y = Target[mask]
        G += torch.einsum("rmt,rnt->mn", Br, Br)
        b += torch.einsum("rmt,rt->m", Br, Y)
        nrows += int(mask.sum())
    if nrows == 0:
        return torch.zeros(M, dtype=torch.float32)
    G += float(ridge) * torch.eye(M, dtype=torch.float64)
    try:
        c = torch.linalg.solve(G, b)
    except Exception:
        c = torch.linalg.lstsq(G, b[:, None]).solution[:, 0]
    return c.float()


def _make_score_rows_for_seqdata(seq: SeqHeadData, names: List[str], coeffs: torch.Tensor,
                                 dynamic_ctx: Optional[Dict[str, Any]] = None, route_id: Optional[int] = None) -> torch.Tensor:
    B_all = _score_basis_matrix_for_seqdata(seq, names, dynamic_ctx=dynamic_ctx, route_id=route_id)
    S = torch.einsum("m,mij->ij", coeffs.detach().cpu().float(), B_all.float())
    return _causal_softmax_cpu(S)


def _eval_score_operator_fit(seqs: List[SeqHeadData], row_masks: Dict[int, torch.Tensor], names: List[str], coeffs: torch.Tensor,
                             o_w: torch.Tensor, dynamic_ctx: Optional[Dict[str, Any]] = None,
                             route_id: Optional[int] = None) -> Dict[str, float]:
    rels=[]; kls=[]; top1s=[]; zrels=[]; yrels=[]; nrows=0
    Ow = o_w.detach().cpu().float()
    for s in seqs:
        mask = row_masks.get(s.prompt_id)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool()
        if int(mask.sum()) == 0:
            continue
        A_hat = _make_score_rows_for_seqdata(s, names, coeffs, dynamic_ctx=dynamic_ctx, route_id=route_id)
        A_true = s.A.detach().cpu().float()
        Ah = A_hat[mask]
        At = A_true[mask]
        rels.append(rel_err(Ah, At))
        kls.append(float(F.kl_div((Ah + 1e-12).log(), At, reduction="batchmean")))
        top1s.append(float((Ah.argmax(dim=-1) == At.argmax(dim=-1)).float().mean()))
        V = s.V.detach().cpu().float()
        Zhat_full = A_hat @ V
        Ztrue_full = A_true @ V
        zrels.append(rel_err(Zhat_full[mask], Ztrue_full[mask]))
        Yhat_full = Zhat_full @ Ow.T
        Ytrue_full = s.Y.detach().cpu().float()
        yrels.append(rel_err(Yhat_full[mask], Ytrue_full[mask]))
        nrows += int(mask.sum())
    return {"rel": safe_mean(rels), "kl": safe_mean(kls), "top1": safe_mean(top1s), "z_rel": safe_mean(zrels), "y_rel": safe_mean(yrels), "rows": nrows}


def score_operator_formula(names: List[str], coeffs: torch.Tensor, max_terms: int = 8) -> str:
    pairs = [(nm, float(c)) for nm, c in zip(names, coeffs.detach().cpu().float().tolist()) if abs(float(c)) > 1e-5]
    pairs.sort(key=lambda x: -abs(x[1]))
    return "0" if not pairs else " + ".join(f"{c:+.3f}*{nm}" for nm, c in pairs[:max_terms])


def mine_score_residual_pairs(seqs: List[SeqHeadData], row_masks: Dict[int, torch.Tensor], names: List[str], coeffs: torch.Tensor,
                              topk: int = 8, min_score: float = 1e-4,
                              dynamic_ctx: Optional[Dict[str, Any]] = None, route_id: Optional[int] = None) -> List[Dict[str, Any]]:
    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    total_rows = 0
    for s in seqs:
        mask = row_masks.get(s.prompt_id)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool()
        if int(mask.sum()) == 0:
            continue
        B = _score_basis_matrix_for_seqdata(s, names, dynamic_ctx=dynamic_ctx, route_id=route_id)
        S_hat = torch.einsum("m,mij->ij", coeffs.detach().cpu().float(), B.float())
        L_true = _attention_to_centered_logits(s.A.detach().cpu().float())
        R = (L_true - _row_center_causal(S_hat)).clamp_min(0.0)
        idxs = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        total_rows += len(idxs)
        for i in idxs:
            if i >= len(s.tokens):
                continue
            qk = token_kind(s.tokens[i])
            row = R[i]
            # keep only strong positive residual score edges; this avoids mining pure noise.
            if row[:i+1].numel() == 0:
                continue
            thresh = float(row[:i+1].mean() + row[:i+1].std())
            for j in torch.nonzero(row[:i+1] > max(thresh, 1e-6), as_tuple=False).flatten().tolist():
                if j >= len(s.tokens):
                    continue
                kk = token_kind(s.tokens[j])
                val = float(row[j])
                d = stats.setdefault((qk, kk), {"mass": 0.0, "edges": 0.0})
                d["mass"] += val
                d["edges"] += 1.0
    mined=[]; denom=max(1, total_rows)
    for (qk, kk), d in stats.items():
        if qk in ("empty",) or kk in ("empty",):
            continue
        score=float(d["mass"])/denom
        if score < min_score:
            continue
        name=f"ScoreMined_Q{_kind_template_name(qk)}ToK{_kind_template_name(kk)}Bias"
        mined.append({"name": name, "q_kind": qk, "k_kind": kk, "score": score, "mass": d["mass"], "edges": int(d["edges"]), "rows": total_rows})
    mined.sort(key=lambda r: -float(r["score"]))
    out=[]; seen=set()
    for r in mined:
        if r["name"] in seen:
            continue
        seen.add(r["name"]); out.append(r)
        if len(out) >= int(topk):
            break
    return out


def _fit_best_score_variant(train: List[SeqHeadData], val: List[SeqHeadData], tr_mask: Dict[int, torch.Tensor], va_mask: Dict[int, torch.Tensor],
                            names: List[str], ridge: float, o_w: torch.Tensor, args: argparse.Namespace,
                            dynamic_ctx: Optional[Dict[str, Any]], route_id: Optional[int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    c = _fit_score_operator_coefficients(train, tr_mask, names, ridge, dynamic_ctx=dynamic_ctx, route_id=route_id)
    ev_val = _eval_score_operator_fit(val, va_mask, names, c, o_w, dynamic_ctx=dynamic_ctx, route_id=route_id)
    ev_tr = _eval_score_operator_fit(train, tr_mask, names, c, o_w, dynamic_ctx=dynamic_ctx, route_id=route_id)
    score = _operator_selection_score(ev_val, getattr(args, "operator_selection", "functional_balanced"))
    return c, {"mode": "score_signed_ridge", "coeffs": c, "score": score, "train": ev_tr, "val": ev_val}

def fit_route_operator_programs(train: List[SeqHeadData], val: List[SeqHeadData], labels_train: List[int], labels_val: List[int],
                                rows_train: List[Dict[str, Any]], rows_val: List[Dict[str, Any]], route_summaries: List[Dict[str, Any]],
                                o_w: torch.Tensor, args: argparse.Namespace,
                                dynamic_ctx: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_names = _parse_templates(getattr(args, "operator_templates", ""))
    if not bool(getattr(args, "operator_dynamic", True)):
        base_names = [n for n in base_names if not _is_dynamic_operator(n)]
    if bool(getattr(args, "operator_allow_oracle_qk", False)) and "QKFullOracle" not in base_names:
        base_names.append("QKFullOracle")

    score_base_names = _parse_score_templates(getattr(args, "score_templates", ""))
    if not bool(getattr(args, "operator_dynamic", True)):
        score_base_names = [n for n in score_base_names if not _is_score_dynamic_operator(n)]
    if bool(getattr(args, "score_include_full_qk", False)) and "ScoreFullQK" not in score_base_names:
        score_base_names.append("ScoreFullQK")

    ridge = float(getattr(args, "operator_ridge", 1e-4))
    score_ridge = float(getattr(args, "score_ridge", ridge))
    train_masks = {cid: make_row_masks(train, rows_train, labels_train, cid) for cid in sorted(set(labels_train))}
    val_masks = {cid: make_row_masks(val, rows_val, labels_val, cid) for cid in sorted(set(labels_train))}
    route_info_by_id = {int(r.get("cluster", -1)): r for r in route_summaries}
    rows=[]; subroute_rows=[]; all_templates=set(base_names); all_score_templates=set(score_base_names)
    bank={"kind":"attention_route_operator_bank_v6_scorehybrid", "base_templates": base_names, "templates": [],
          "score_base_templates": score_base_names, "score_templates": [], "routes": [], "subroutes": [],
          "dynamic_enabled": bool(getattr(args, "operator_dynamic", True)),
          "score_operator_enabled": bool(getattr(args, "score_operator_fit", True))}

    for cid in sorted(set(labels_train)):
        ntr = _mask_rows_count(train_masks[cid])
        nv = _mask_rows_count(val_masks.get(cid, {}))
        if ntr < int(getattr(args, "min_route_rows", 0)):
            continue

        # ---- A-level operator fit, as in v5 ----
        route_names = list(base_names)
        coeff0, best0 = _fit_best_operator_variant(train, val, train_masks[cid], val_masks.get(cid, {}), route_names, ridge, o_w, args, dynamic_ctx, cid)
        mined_pairs=[]
        if bool(getattr(args, "operator_mine_residual", True)):
            mined_pairs = mine_attention_residual_pairs(
                train, train_masks[cid], route_names, coeff0,
                topk=int(getattr(args, "operator_mine_topk", 8)),
                min_score=float(getattr(args, "operator_mine_min_score", 1e-4)),
                dynamic_ctx=dynamic_ctx, route_id=cid,
            )
            for mp in mined_pairs:
                if mp["name"] not in route_names:
                    route_names.append(mp["name"])
        if len(route_names) != len(base_names):
            coeffs, best = _fit_best_operator_variant(train, val, train_masks[cid], val_masks.get(cid, {}), route_names, ridge, o_w, args, dynamic_ctx, cid)
        else:
            coeffs, best = coeff0, best0
        ev_tr = best["train"]; ev_val = best["val"]
        formula = operator_formula(route_names, coeffs, max_terms=int(getattr(args, "operator_max_terms", 6)))

        # ---- Score/logit-level fit before softmax ----
        score_info = {"enabled": False}
        score_names = list(score_base_names)
        score_coeffs = torch.zeros(len(score_names), dtype=torch.float32)
        if bool(getattr(args, "score_operator_fit", True)):
            score_coeff0, score_best0 = _fit_best_score_variant(train, val, train_masks[cid], val_masks.get(cid, {}), score_names, score_ridge, o_w, args, dynamic_ctx, cid)
            score_mined=[]
            if bool(getattr(args, "score_mine_residual", True)):
                score_mined = mine_score_residual_pairs(
                    train, train_masks[cid], score_names, score_coeff0,
                    topk=int(getattr(args, "score_mine_topk", getattr(args, "operator_mine_topk", 8))),
                    min_score=float(getattr(args, "score_mine_min_score", 1e-4)),
                    dynamic_ctx=dynamic_ctx, route_id=cid,
                )
                for mp in score_mined:
                    if mp["name"] not in score_names:
                        score_names.append(mp["name"])
            if len(score_names) != len(score_base_names):
                score_coeffs, score_best = _fit_best_score_variant(train, val, train_masks[cid], val_masks.get(cid, {}), score_names, score_ridge, o_w, args, dynamic_ctx, cid)
            else:
                score_coeffs, score_best = score_coeff0, score_best0
            score_info = {
                "enabled": True,
                "fit_mode": score_best["mode"],
                "functional_score": score_best["score"],
                "formula": score_operator_formula(score_names, score_coeffs, max_terms=int(getattr(args, "score_max_terms", getattr(args, "operator_max_terms", 8)))),
                "templates": score_names,
                "coefficients": {nm: float(c) for nm, c in zip(score_names, score_coeffs.detach().cpu().tolist())},
                "mined_pairs": score_mined,
                "train": score_best["train"],
                "val": score_best["val"],
            }

        # ---- family sweep for A-level formula ----
        family_sweep = []
        for fam in sorted(set(operator_family(nm) for nm in route_names)):
            cf = _mask_coeffs_by_family(route_names, coeffs, fam)
            if float(cf.clamp_min(0).sum()) <= 1e-9:
                continue
            evf = _eval_operator_fit(val, val_masks.get(cid, {}), route_names, cf, o_w, dynamic_ctx=dynamic_ctx, route_id=cid)
            family_sweep.append({"family": fam, "val_rel": evf.get("rel"), "val_top1": evf.get("top1"), "val_z_rel": evf.get("z_rel"), "val_y_rel": evf.get("y_rel"), "formula": operator_formula(route_names, cf, max_terms=6)})
        family_sweep.sort(key=lambda x: _operator_selection_score({"rel": x.get("val_rel", 999.0), "top1": x.get("val_top1", 0.0), "z_rel": x.get("val_z_rel", 999.0), "y_rel": x.get("val_y_rel", 999.0)}, getattr(args, "operator_selection", "functional_balanced")))

        # qtype subroutes inside this route: keep A-level and add score-level comparator.
        if bool(getattr(args, "operator_subroutes", True)):
            for sk in ["content", "word", "capitalized", "punct", "code", "number", "role", "short", "other"]:
                sm_tr = _make_subroute_masks(train, train_masks[cid], sk)
                sm_va = _make_subroute_masks(val, val_masks.get(cid, {}), sk)
                sntr = _mask_rows_count(sm_tr); snv = _mask_rows_count(sm_va)
                if sntr < max(16, int(getattr(args, "min_route_rows", 64)) // 4) or snv < 4:
                    continue
                sc, sbest = _fit_best_operator_variant(train, val, sm_tr, sm_va, route_names, ridge, o_w, args, dynamic_ctx, cid)
                sf = operator_formula(route_names, sc, max_terms=int(getattr(args, "operator_max_terms", 6)))
                sr = {"route": int(cid), "subroute": f"qkind={sk}", "q_kind": sk, "rows_train": sntr, "rows_val": snv,
                      "fit_mode": sbest["mode"], "functional_score": sbest["score"],
                      "formula": sf, "val_rel": sbest["val"].get("rel"), "val_top1": sbest["val"].get("top1"),
                      "val_z_rel": sbest["val"].get("z_rel"), "val_y_rel": sbest["val"].get("y_rel"),
                      "coefficients": {nm: float(x) for nm, x in zip(route_names, sc.detach().cpu().tolist())}, "templates": route_names}
                if bool(getattr(args, "score_operator_fit", True)):
                    try:
                        ssc, ssbest = _fit_best_score_variant(train, val, sm_tr, sm_va, score_names, score_ridge, o_w, args, dynamic_ctx, cid)
                        sr.update({"score_formula": score_operator_formula(score_names, ssc, max_terms=int(getattr(args, "score_max_terms", 8))),
                                   "score_val_rel": ssbest["val"].get("rel"), "score_val_top1": ssbest["val"].get("top1"),
                                   "score_val_z_rel": ssbest["val"].get("z_rel"), "score_val_y_rel": ssbest["val"].get("y_rel"),
                                   "score_functional_score": ssbest["score"]})
                    except Exception as e:
                        sr.update({"score_error": str(e)[:200]})
                subroute_rows.append(sr)

        rb_size = int(getattr(args, "route_bank_size", 64))
        product_steps = product_step_search_for_route(rb_size, route_names, coeffs, topk=int(getattr(args, "product_topk", 8)))
        forest = route_matrix_block_forest(_canonical_operator_matrix(rb_size, route_names, coeffs), int(cid), max_depth=int(getattr(args, "block_forest_depth", 2)))
        rinfo = route_info_by_id.get(int(cid), {})
        score_val = score_info.get("val", {}) if isinstance(score_info, dict) else {}
        score_train = score_info.get("train", {}) if isinstance(score_info, dict) else {}
        score_logic_score = float(score_info.get("functional_score", 1e9)) if score_info.get("enabled") else 1e9
        a_logic_score = float(best["score"])
        best_logic = "score_softmax" if score_logic_score < a_logic_score else "attention_operator"

        row = {
            "route": int(cid), "route_type": rinfo.get("route_type"), "route_verdict": rinfo.get("route_verdict"),
            "rows_train": ntr, "rows_val": nv, "formula": formula, "templates": route_names,
            "fit_mode": best["mode"], "functional_score": best["score"], "best_logic": best_logic,
            "train_rel": ev_tr.get("rel"), "val_rel": ev_val.get("rel"),
            "train_kl": ev_tr.get("kl"), "val_kl": ev_val.get("kl"),
            "train_top1": ev_tr.get("top1"), "val_top1": ev_val.get("top1"),
            "train_z_rel": ev_tr.get("z_rel"), "val_z_rel": ev_val.get("z_rel"),
            "train_y_rel": ev_tr.get("y_rel"), "val_y_rel": ev_val.get("y_rel"),
            "coefficients": {nm: float(c) for nm, c in zip(route_names, coeffs.detach().cpu().tolist())},
            "mined_pairs": mined_pairs,
            "score_enabled": bool(score_info.get("enabled")),
            "score_formula": score_info.get("formula"), "score_templates": score_names,
            "score_coefficients": score_info.get("coefficients"), "score_mined_pairs": score_info.get("mined_pairs"),
            "score_fit_mode": score_info.get("fit_mode"), "score_functional_score": score_info.get("functional_score"),
            "score_train_rel": score_train.get("rel"), "score_val_rel": score_val.get("rel"),
            "score_train_kl": score_train.get("kl"), "score_val_kl": score_val.get("kl"),
            "score_train_top1": score_train.get("top1"), "score_val_top1": score_val.get("top1"),
            "score_train_z_rel": score_train.get("z_rel"), "score_val_z_rel": score_val.get("z_rel"),
            "score_train_y_rel": score_train.get("y_rel"), "score_val_y_rel": score_val.get("y_rel"),
            "best_family": family_sweep[0] if family_sweep else None,
            "family_sweep": family_sweep,
            "product_steps": product_steps,
            "block_forest_top": forest[:12],
        }
        rows.append(row); bank["routes"].append(row)
        for nm in route_names: all_templates.add(nm)
        for nm in score_names: all_score_templates.add(nm)
    bank["templates"] = sorted(all_templates)
    bank["score_templates"] = sorted(all_score_templates)
    bank["subroutes"] = subroute_rows
    return rows, bank

def _coerce_coeffs_from_row(row: Dict[str, Any], names: List[str]) -> torch.Tensor:
    coeffs = row.get("coefficients", {})
    if isinstance(coeffs, str):
        try: coeffs = json.loads(coeffs)
        except Exception: coeffs = {}
    return torch.tensor([float(coeffs.get(nm, 0.0)) for nm in names], dtype=torch.float32)

def _canonical_operator_matrix(T: int, names: List[str], coeffs: torch.Tensor) -> torch.Tensor:
    return _make_operator_rows_for_seq(T, names, coeffs.detach().cpu().float(), device="cpu", tokens=None)

def _template_matrix(T: int, name: str) -> torch.Tensor:
    return _canonical_operator_matrix(T, [name], torch.ones(1))

def product_step_search_for_route(T: int, names: List[str], coeffs: torch.Tensor, topk: int = 8) -> List[Dict[str, Any]]:
    return product_step_search_for_matrix(_canonical_operator_matrix(T, names, coeffs), names, topk=topk)

def product_step_search_for_matrix(target: torch.Tensor, names: List[str], topk: int = 8, bank_kind: str = "") -> List[Dict[str, Any]]:
    T = int(target.shape[0])
    mats = {}
    for nm in names:
        # bank_kind matters: score_softmax_data must use score/logit templates,
        # not attention-probability templates. This was one reason product was
        # silently weak on score banks.
        M = _product_template_matrix(T, nm, bank_kind=bank_kind) if bank_kind else _template_matrix(T, nm)
        if float(M.abs().sum()) > 1e-9:
            mats[nm] = M
    rows=[]
    for nm, M in mats.items():
        rows.append({"kind":"single", "bank_kind": bank_kind, "step0":nm, "step1":"", "rel":_matrix_rel(M,target)})
    # product2 is exact matrix composition probe, not deploy plan.
    shortlist = sorted(rows, key=lambda r: float(r["rel"]))[:min(18, len(rows))]
    short_names = [r["step0"] for r in shortlist]
    for a in short_names:
        A = mats[a]
        for b in short_names:
            B = mats[b]
            P = (A @ B).clamp_min(0.0)
            P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            rows.append({"kind":"product2", "bank_kind": bank_kind, "step0":a, "step1":b, "rel":_matrix_rel(P,target)})
    rows.sort(key=lambda r: float(r["rel"]))
    return rows[:topk]

def fit_learned_product_steps(target: torch.Tensor, names: List[str], depth: int = 2, steps: int = 80, topk: int = 12,
                              private_svd_atoms: int = 0, bank_kind: str = "") -> Dict[str, Any]:
    T = int(target.shape[0])
    # Use best single candidates as typed/flat bank; include a few anchors even if low mass.
    singles = product_step_search_for_matrix(target, names, topk=max(topk, 16), bank_kind=bank_kind)
    cand_names = []
    for r in singles:
        if r.get("step0") and r["step0"] not in cand_names:
            cand_names.append(r["step0"])
        if len(cand_names) >= topk:
            break
    anchor_names = ["IdentitySelf", "PrevToken", "BOSRead", "UniformPast", "LocalWindow4", "DistanceDecay8"]
    if str(bank_kind).startswith("score"):
        anchor_names = ["ScoreSelfBias", "ScorePrevBias", "ScoreBOSBias", "ScoreUniformPastBias", "ScoreLocal4", "ScoreDistanceDecay8"]
    for nm in anchor_names:
        if nm in names and nm not in cand_names:
            cand_names.append(nm)
    mats = [_product_template_matrix(T, nm, bank_kind=bank_kind).float() for nm in cand_names]
    atom_names = list(cand_names)
    if int(private_svd_atoms) > 0:
        # Private atoms summarize residual block energy for diagnostics/product probing only.
        U,S,Vh = torch.linalg.svd(target.float(), full_matrices=False)
        for k in range(min(int(private_svd_atoms), int(S.numel()))):
            M = (S[k] * torch.outer(U[:, k], Vh[k])).clamp_min(0.0)
            if float(M.sum()) > 1e-12:
                M = M / M.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                mats.append(M); atom_names.append(f"PrivateSVD{k+1}")
    if not mats:
        return {"enabled": False, "reason": "no candidates"}
    B = torch.stack(mats, dim=0)
    P = B.shape[0]
    logits = torch.zeros((max(1, int(depth)), P), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.08)
    target = target.float()
    best = None
    for _ in range(max(1, int(steps))):
        weights = torch.softmax(logits, dim=-1)
        M = torch.eye(T)
        for d in range(weights.shape[0]):
            Step = torch.einsum("p,pij->ij", weights[d], B).clamp_min(0.0)
            Step = Step / Step.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            M = Step @ M
            M = M.clamp_min(0.0); M = M / M.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        loss = ((M - target) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        rel = _matrix_rel(M.detach(), target)
        if best is None or rel < best[0]:
            best = (rel, weights.detach().clone(), M.detach().clone())
    rel, weights, _ = best
    steps_desc=[]
    for d in range(weights.shape[0]):
        pairs = [(atom_names[i], float(weights[d, i])) for i in range(P) if float(weights[d, i]) > 1e-3]
        pairs.sort(key=lambda x: -x[1])
        steps_desc.append(" + ".join(f"{w:.3f}*{nm}" for nm, w in pairs[:6]))
    return {"enabled": True, "kind": "learned_product_steps", "bank_kind": bank_kind, "depth": int(depth), "rel": float(rel), "candidate_names": atom_names, "steps": steps_desc}


def _score_operator_family(name: str) -> str:
    n = str(name).lower()
    if "qk" in n or "hidden" in n or "similarity" in n:
        return "dynamic_score"
    if "distance" in n or "local" in n or "position" in n:
        return "pos_score"
    if "q" in n and "k" in n:
        return "qtype_ktype_score"
    if "bos" in n or "prev" in n or "self" in n:
        return "anchor_score"
    if "mined" in n:
        return "mined_score"
    return "score_other"


def _product_family(name: str, bank_kind: str = "") -> str:
    if str(bank_kind).startswith("score") or str(name).lower().startswith("score"):
        return _score_operator_family(name)
    return operator_family(name)



def _make_score_matrix(T: int, names: List[str], coeffs: torch.Tensor, device: str = "cpu", tokens: Optional[List[str]] = None) -> torch.Tensor:
    """Canonical score/logit basis matrix for product probes.

    This is deliberately lightweight: it mirrors the static score atoms used in
    real seqdata, but works without Q/K/X tensors. Dynamic score atoms become
    zero here; real dynamic evidence is already represented by score_softmax_data.
    """
    T = int(T)
    S = torch.zeros(T, T, dtype=torch.float32, device=device)
    if T <= 0:
        return S
    toks = tokens or ["?"] * T
    for nm, c in zip(names, coeffs.detach().cpu().float().tolist()):
        c = float(c)
        if abs(c) <= 1e-12:
            continue
        n = str(nm).lower()
        B = torch.zeros(T, T, dtype=torch.float32, device=device)
        qk = _parse_score_q_to_k_name(nm)
        if qk is not None and tokens is not None:
            q_kind, k_kind = qk
            for i in range(T):
                if _token_matches_kind(toks[i], q_kind):
                    for j in range(i + 1):
                        if _token_matches_kind(toks[j], k_kind):
                            B[i, j] = 1.0
        elif n in ("scorebosbias", "scorekbosbias"):
            B[:, 0] = 1.0
        elif n in ("scoreselfbias", "scorelastbias"):
            for i in range(T): B[i, i] = 1.0
        elif n == "scoreprevbias":
            for i in range(T): B[i, max(0, i - 1)] = 1.0
        elif n.startswith("scorelocal"):
            m = re.search(r"(\d+)", n); r = int(m.group(1)) if m else 4
            for i in range(T): B[i, max(0, i-r):i+1] = 1.0
        elif n.startswith("scoredistancedecay"):
            m = re.search(r"(\d+)", n); tau = float(m.group(1)) if m else 8.0
            for i in range(T):
                js = torch.arange(i + 1, dtype=torch.float32, device=device)
                B[i, :i+1] = torch.exp(-(float(i) - js) / max(1e-6, tau))
        elif n in ("scoredistancepenalty", "scoredistpenalty"):
            for i in range(T):
                js = torch.arange(i + 1, dtype=torch.float32, device=device)
                B[i, :i+1] = -(float(i) - js) / max(1.0, float(T))
        elif n == "scoreuniformpastbias":
            for i in range(T): B[i, :i+1] = 1.0
        elif n.startswith("scorek") and n.endswith("bias") and tokens is not None:
            kind = _canon_kind_name(n[len("scorek"):-len("bias")])
            for i in range(T):
                for j in range(i + 1):
                    if _token_matches_kind(toks[j], kind):
                        B[i, j] = 1.0
        # Dynamic score atoms cannot be canonically represented without data.
        S += c * B
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    return S.masked_fill(mask, -1e9)

def _product_template_matrix(T: int, name: str, bank_kind: str = "") -> torch.Tensor:
    # Score banks still need stochastic transition matrices for product probes;
    # use score-bias -> softmax matrix for score operators.
    if str(bank_kind).startswith("score") or str(name).lower().startswith("score"):
        S = _make_score_matrix(T, [name], torch.ones(1), device="cpu", tokens=None)
        return torch.softmax(S.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), -1e9), dim=-1).float()
    return _template_matrix(T, name).float()


def typed_beam_product_search_for_matrix(target: torch.Tensor, names: List[str], bank_kind: str = "",
                                         depth: int = 2, topk: int = 12, beam: int = 24,
                                         per_family: int = 4) -> List[Dict[str, Any]]:
    """Old-style searched typed product, but compact and deterministic.

    It first keeps the best atoms per family, then runs a beam search over
    Step_d @ ... @ Step_0. This is stronger than the toy product2 but still fast
    for route_bank_size=32.
    """
    target = target.detach().cpu().float()
    T = int(target.shape[0])
    mats: Dict[str, torch.Tensor] = {}
    singles: List[Dict[str, Any]] = []
    for nm in list(dict.fromkeys([str(x) for x in names if str(x)])):
        try:
            M = _product_template_matrix(T, nm, bank_kind=bank_kind)
        except Exception:
            continue
        if float(M.abs().sum()) <= 1e-9:
            continue
        mats[nm] = M
        singles.append({"kind": "typed_single", "bank_kind": bank_kind, "step0": nm, "rel": _matrix_rel(M, target), "family0": _product_family(nm, bank_kind)})
    if not singles:
        return []
    fam: Dict[str, List[Dict[str, Any]]] = {}
    for r in sorted(singles, key=lambda x: float(x["rel"])):
        fam.setdefault(str(r["family0"]), []).append(r)
    cand_names: List[str] = []
    for f, rs in sorted(fam.items(), key=lambda kv: float(kv[1][0]["rel"])):
        for r in rs[:max(1, int(per_family))]:
            if r["step0"] not in cand_names:
                cand_names.append(r["step0"])
    # Also preserve globally best atoms.
    for r in sorted(singles, key=lambda x: float(x["rel"]))[:max(4, int(topk))]:
        if r["step0"] not in cand_names:
            cand_names.append(r["step0"])
    cand_names = cand_names[:max(2, int(beam))]

    I = torch.eye(T)
    states = [([], I)]
    all_rows: List[Dict[str, Any]] = list(sorted(singles, key=lambda x: float(x["rel"]))[:max(1, int(topk))])
    for d in range(max(1, int(depth))):
        new_states = []
        for path, Mprev in states:
            for nm in cand_names:
                Step = mats[nm]
                M = (Step @ Mprev).clamp_min(0.0)
                M = M / M.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                rel = _matrix_rel(M, target)
                npath = path + [nm]
                new_states.append((rel, npath, M))
        new_states.sort(key=lambda x: float(x[0]))
        states = [(p, M) for _, p, M in new_states[:max(1, int(beam))]]
        for rel, path, M in new_states[:max(1, int(topk))]:
            row = {"kind": f"typed_product{len(path)}", "bank_kind": bank_kind, "rel": float(rel),
                   "steps": " @ ".join(reversed(path)),
                   "family_path": " @ ".join(reversed([_product_family(x, bank_kind) for x in path]))}
            for i, nm in enumerate(path):
                row[f"step{i}"] = nm
                row[f"family{i}"] = _product_family(nm, bank_kind)
            all_rows.append(row)
    all_rows.sort(key=lambda x: float(x.get("rel", 999.0)))
    # de-duplicate by kind+steps
    out=[]; seen=set()
    for r in all_rows:
        key=(r.get("kind"), r.get("steps", r.get("step0")))
        if key in seen:
            continue
        seen.add(key); out.append(r)
        if len(out) >= int(topk):
            break
    return out


def _parse_name_list(s: str, default: List[str]) -> List[str]:
    xs = [x.strip() for x in str(s or "").replace(";", ",").split(",") if x.strip()]
    return xs or list(default)

def _row_grid_project(acc: torch.Tensor, row: torch.Tensor, i: int, Tsrc: int) -> None:
    T = int(acc.shape[0])
    gi = int(round((i / max(1, Tsrc - 1)) * (T - 1)))
    for j, val in enumerate(row.detach().cpu().float().tolist()):
        if val == 0.0:
            continue
        gj = int(round((j / max(1, Tsrc - 1)) * (T - 1)))
        acc[gi, gj] += float(val)

def _route_mean_matrix_from_data(seqs: List[SeqHeadData], masks: Dict[int, torch.Tensor], route_bank_size: int,
                                 maker) -> Tuple[torch.Tensor, int]:
    T = int(route_bank_size)
    acc = torch.zeros(T, T, dtype=torch.float32)
    rows = 0
    for s in seqs:
        mask = masks.get(s.prompt_id)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool()
        if int(mask.sum()) == 0:
            continue
        A = maker(s)
        for i in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            _row_grid_project(acc, A[i], int(i), int(s.A.shape[0]))
            rows += 1
    row_sum = acc.sum(dim=-1, keepdim=True)
    nonzero = row_sum.squeeze(-1) > 1e-12
    acc[nonzero] = acc[nonzero] / row_sum[nonzero]
    # Empty grid rows get causal uniform so block-diag artifacts stay valid stochastic matrices.
    for i in torch.nonzero(~nonzero, as_tuple=False).flatten().tolist():
        acc[i, :i+1] = 1.0 / float(i + 1)
    return acc, rows

def build_route_bank_artifacts(hdir: Path, route_operator_bank: Dict[str, Any], routes: List[Dict[str, Any]], route_bank_size: int = 64,
                               block_forest_depth: int = 2, train: Optional[List[SeqHeadData]] = None, val: Optional[List[SeqHeadData]] = None,
                               rows_train: Optional[List[Dict[str, Any]]] = None, labels_train: Optional[List[int]] = None,
                               rows_val: Optional[List[Dict[str, Any]]] = None, labels_val: Optional[List[int]] = None,
                               args: Optional[argparse.Namespace] = None,
                               dynamic_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    hdir = Path(hdir); ensure_dir(hdir)
    route_rows = list((route_operator_bank or {}).get("routes") or [])
    subroutes = list((route_operator_bank or {}).get("subroutes") or [])
    if subroutes:
        write_csv(hdir / "route_operator_subroutes.csv", subroutes)
    if not route_rows:
        for fn in ["route_bank_blocks.csv", "route_block_forest.csv", "route_product_steps.csv", "route_operator_deep_decode.csv"]:
            (hdir / fn).write_text("", encoding="utf-8")
        return {"kind":"route_bank_matrix_v6_scorehybrid", "enabled":False, "reason":"no route operator rows"}

    route_by_id={str(r.get("cluster")): r for r in routes}
    train_masks = {int(rr.get("route")): make_row_masks(train or [], rows_train or [], labels_train or [], int(rr.get("route"))) for rr in route_rows} if train is not None and rows_train is not None and labels_train is not None else {}
    val_masks = {int(rr.get("route")): make_row_masks(val or [], rows_val or [], labels_val or [], int(rr.get("route"))) for rr in route_rows} if val is not None and rows_val is not None and labels_val is not None else {}

    canonical_mats=[]; true_mats=[]; template_mats=[]; score_mats=[]
    block_rows=[]; forest_rows=[]; product_rows=[]; deep_rows=[]
    rb_size=int(route_bank_size)
    for rr in route_rows:
        cid=int(rr.get("route")); names=list(rr.get("templates") or (route_operator_bank or {}).get("templates") or DEFAULT_OPERATOR_TEMPLATES)
        coeffs=_coerce_coeffs_from_row(rr, names)
        M_can=_canonical_operator_matrix(rb_size, names, coeffs); canonical_mats.append(M_can)
        M_true = None; M_tpl = None; M_score = None; n_true = 0
        score_names = list(rr.get("score_templates") or [])
        score_coeffs = _coerce_coeffs_from_row({"coefficients": rr.get("score_coefficients") or {}}, score_names) if score_names else torch.empty(0)
        if train is not None and cid in train_masks:
            M_true, n_true = _route_mean_matrix_from_data(train, train_masks[cid], rb_size, lambda s: s.A.detach().cpu().float())
            M_tpl, _ = _route_mean_matrix_from_data(train, train_masks[cid], rb_size, lambda s, names=names, coeffs=coeffs, cid=cid: _make_operator_rows_for_seqdata(s, names, coeffs, dynamic_ctx=dynamic_ctx, route_id=cid))
            true_mats.append(M_true); template_mats.append(M_tpl)
            if score_names and bool(rr.get("score_enabled")):
                M_score, _ = _route_mean_matrix_from_data(train, train_masks[cid], rb_size, lambda s, names=score_names, coeffs=score_coeffs, cid=cid: _make_score_rows_for_seqdata(s, names, coeffs, dynamic_ctx=dynamic_ctx, route_id=cid))
                score_mats.append(M_score)
        rinfo=route_by_id.get(str(cid), {})
        rel_true_tpl = _matrix_rel(M_tpl, M_true) if M_true is not None and M_tpl is not None else None
        rel_true_score = _matrix_rel(M_score, M_true) if M_true is not None and M_score is not None else None

        # Priority for actual logic proof: true_mean -> score_softmax_data/template_data -> canonical only as visualization.
        for bank_kind, M in [("true_mean", M_true), ("score_softmax_data", M_score), ("template_data", M_tpl), ("canonical", M_can)]:
            if M is None:
                continue
            block_rows.append({"bank_kind":bank_kind, "route":cid, "route_type":rr.get("route_type") or rinfo.get("route_type"),
                               "rows_train":rr.get("rows_train"), "rows_val":rr.get("rows_val"), "data_rows": n_true if bank_kind != "canonical" else None,
                               "block_size":rb_size, "energy":float((M.float()**2).sum().sqrt().item()),
                               "operator_fit_val_rel":rr.get("val_rel"), "operator_fit_val_top1":rr.get("val_top1"),
                               "score_fit_val_rel":rr.get("score_val_rel"), "score_fit_val_top1":rr.get("score_val_top1"),
                               "true_template_rel": rel_true_tpl, "true_score_rel": rel_true_score,
                               "best_logic": rr.get("best_logic"), "formula":rr.get("formula"), "score_formula": rr.get("score_formula")})
            forest_rows.extend([{**x, "bank_kind": bank_kind} for x in route_matrix_block_forest(M, cid, max_depth=int(block_forest_depth))[:64]])

        # Product search is now bank-kind aware. Do not only probe canonical or only the
        # selected best_logic target: run true/template/score/canonical separately so the
        # CSV shows whether the block/product program is real evidence or just an abstract view.
        wanted_product_kinds = set(_parse_name_list(getattr(args, "product_bank_kinds", ""), ["true_mean", "score_softmax_data", "template_data", "canonical"])) if args is not None else {"true_mean", "score_softmax_data", "template_data", "canonical"}
        product_targets = [
            ("true_mean", M_true, names),
            ("score_softmax_data", M_score, score_names if score_names else names),
            ("template_data", M_tpl, names),
            ("canonical", M_can, names),
        ]
        best_product_any = None
        learned_any = None
        for product_bank_kind, product_target, product_names in product_targets:
            if product_target is None or product_bank_kind not in wanted_product_kinds:
                continue
            topk = int(getattr(args, "product_topk", 8) if args is not None else 8)
            # Simple exact product probes.
            ps_basic = product_step_search_for_matrix(product_target, product_names, topk=topk, bank_kind=product_bank_kind)
            # Typed searched product closer to the old routed-block decode.
            ps_typed = typed_beam_product_search_for_matrix(
                product_target, product_names, bank_kind=product_bank_kind,
                depth=int(getattr(args, "product_depth", 2) if args is not None else 2),
                topk=topk,
                beam=int(getattr(args, "product_beam", 24) if args is not None else 24),
                per_family=int(getattr(args, "product_per_family", 4) if args is not None else 4),
            )
            ps = sorted(ps_basic + ps_typed, key=lambda r: float(r.get("rel", 999.0)))[:max(1, topk)]
            for pr in ps:
                row=dict(pr); row["route"]=cid; row["bank_kind"]=product_bank_kind; product_rows.append(row)
            if ps and (best_product_any is None or float(ps[0].get("rel", 999.0)) < float(best_product_any.get("rel", 999.0))):
                best_product_any = dict(ps[0]); best_product_any["bank_kind"] = product_bank_kind
            learned = fit_learned_product_steps(product_target, product_names,
                                                depth=int(getattr(args, "product_depth", 2) if args is not None else 2),
                                                steps=int(getattr(args, "product_fit_steps", 80) if args is not None else 80),
                                                topk=max(4, topk),
                                                private_svd_atoms=int(getattr(args, "product_private_svd", 1) if args is not None else 1),
                                                bank_kind=product_bank_kind)
            if learned.get("enabled"):
                lrow={"route":cid, "bank_kind":product_bank_kind, "kind":"learned_product", "rel": learned.get("rel"), "steps": learned.get("steps"), "candidate_names": learned.get("candidate_names")}
                product_rows.append(lrow)
                if learned_any is None or float(learned.get("rel", 999.0)) < float(learned_any.get("rel", 999.0)):
                    learned_any = dict(lrow)
        ps = [best_product_any] if best_product_any is not None else []
        learned = learned_any or {"enabled": False}
        bf=rr.get("best_family")
        deep_rows.append({"route":cid, "route_type":rr.get("route_type") or rinfo.get("route_type"),
                          "best_logic": rr.get("best_logic"),
                          "flat_formula":rr.get("formula"),
                          "flat_val_rel":rr.get("val_rel"), "flat_val_top1":rr.get("val_top1"), "flat_val_z_rel": rr.get("val_z_rel"), "flat_val_y_rel":rr.get("val_y_rel"),
                          "score_formula": rr.get("score_formula"),
                          "score_val_rel": rr.get("score_val_rel"), "score_val_top1": rr.get("score_val_top1"), "score_val_z_rel": rr.get("score_val_z_rel"), "score_val_y_rel": rr.get("score_val_y_rel"),
                          "score_functional_score": rr.get("score_functional_score"),
                          "fit_mode": rr.get("fit_mode"), "functional_score": rr.get("functional_score"),
                          "best_family":(bf or {}).get("family") if isinstance(bf, dict) else None,
                          "best_family_val_rel":(bf or {}).get("val_rel") if isinstance(bf, dict) else None,
                          "best_product":(ps[0] if ps else None), "learned_product": learned,
                          "true_template_rel": rel_true_tpl, "true_score_rel": rel_true_score,
                          "mined_pairs": rr.get("mined_pairs"), "score_mined_pairs": rr.get("score_mined_pairs")})
    bank_can=torch.block_diag(*canonical_mats) if canonical_mats else torch.empty(0,0)
    torch.save({"kind":"route_bank_canonical", "route_bank_size":rb_size, "matrix":bank_can, "routes":block_rows}, hdir/"route_bank_canonical.pt")
    torch.save({"kind":"route_bank_matrix", "templates":(route_operator_bank or {}).get("templates"), "route_bank_size":rb_size, "matrix":bank_can, "routes":block_rows}, hdir/"route_bank_matrix.pt")
    if true_mats:
        torch.save({"kind":"route_bank_true_mean", "route_bank_size":rb_size, "matrix":torch.block_diag(*true_mats), "routes":block_rows}, hdir/"route_bank_true_mean.pt")
    if template_mats:
        torch.save({"kind":"route_bank_template_data", "route_bank_size":rb_size, "matrix":torch.block_diag(*template_mats), "routes":block_rows}, hdir/"route_bank_template_data.pt")
    if score_mats:
        torch.save({"kind":"route_bank_score_softmax_data", "route_bank_size":rb_size, "matrix":torch.block_diag(*score_mats), "routes":block_rows}, hdir/"route_bank_score_softmax_data.pt")
    write_csv(hdir/"route_bank_blocks.csv", block_rows)
    write_csv(hdir/"route_block_forest.csv", forest_rows)
    write_csv(hdir/"route_product_steps.csv", product_rows)
    write_csv(hdir/"route_operator_deep_decode.csv", deep_rows)
    info={"kind":"route_bank_matrix_v6_scorehybrid", "enabled":True, "routes":len(route_rows), "route_bank_size":rb_size,
          "canonical_shape":list(bank_can.shape), "has_true_mean":bool(true_mats), "has_template_data":bool(template_mats), "has_score_softmax_data":bool(score_mats), "blocks":block_rows}
    (hdir/"route_bank_matrix.json").write_text(json_dumps_safe(info, indent=2, ensure_ascii=False), encoding="utf-8")

    # Hard audit: prevent the old bug where stage=all produced true/template/score
    # banks, then the report builder silently overwrote CSVs with canonical-only rows.
    wanted_kinds = _parse_name_list(getattr(args, "product_bank_kinds", "") if args is not None else "",
                                    ["true_mean", "score_softmax_data", "template_data", "canonical"])
    audit = {
        "wanted_product_bank_kinds": wanted_kinds,
        "block_bank_kinds": sorted(set(str(r.get("bank_kind")) for r in block_rows if r.get("bank_kind") is not None)),
        "forest_bank_kinds": sorted(set(str(r.get("bank_kind")) for r in forest_rows if r.get("bank_kind") is not None)),
        "product_bank_kinds": sorted(set(str(r.get("bank_kind")) for r in product_rows if r.get("bank_kind") is not None)),
        "n_block_rows": len(block_rows),
        "n_forest_rows": len(forest_rows),
        "n_product_rows": len(product_rows),
        "has_true_mean_pt": (hdir/"route_bank_true_mean.pt").exists(),
        "has_template_data_pt": (hdir/"route_bank_template_data.pt").exists(),
        "has_score_softmax_data_pt": (hdir/"route_bank_score_softmax_data.pt").exists(),
    }
    audit["missing_product_bank_kinds"] = [k for k in wanted_kinds if k not in set(audit["product_bank_kinds"])]
    # meaningful missing lists below: only require banks that actually exist / were computed
    expected_existing = ["canonical"]
    if audit["has_true_mean_pt"]: expected_existing.append("true_mean")
    if audit["has_template_data_pt"]: expected_existing.append("template_data")
    if audit["has_score_softmax_data_pt"]: expected_existing.append("score_softmax_data")
    audit["expected_existing_bank_kinds"] = expected_existing
    audit["missing_existing_product_bank_kinds"] = [k for k in expected_existing if k in wanted_kinds and k not in set(audit["product_bank_kinds"])]
    audit["missing_existing_forest_bank_kinds"] = [k for k in expected_existing if k not in set(audit["forest_bank_kinds"])]
    (hdir/"route_bank_audit.json").write_text(json_dumps_safe(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return info

def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _patch_mode_map(patch: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(patch, dict):
        return {}
    modes = patch.get("modes") or patch.get("patch_modes") or patch.get("results") or patch
    if isinstance(modes, list):
        return {str(m.get("mode", f"mode{i}")): m for i, m in enumerate(modes) if isinstance(m, dict)}
    if isinstance(modes, dict):
        return {str(k): v for k, v in modes.items() if isinstance(v, dict)}
    return {}


def build_logic_report_from_outdir(out_dir: Path, heads: str = "") -> None:
    """Build one human-readable logic report from the unified output directory."""
    out_dir = Path(out_dir)
    if heads:
        wanted = set(h.strip() for h in heads.replace(';', ',').split(',') if h.strip())
        wanted |= {("L" + h.replace(":", "H")) for h in list(wanted) if not h.startswith("L")}
    else:
        wanted = set()
    hdirs = sorted([p for p in out_dir.glob("L*H*") if p.is_dir()])
    if wanted:
        hdirs = [p for p in hdirs if p.name in wanted]
    summary_rows=[]
    md=[]
    md.append("# Qwen unified logic/program decomposition\n")
    md.append("This is the report to read first. It shows code path → route graph → operator formulas → QK program → patch controls.\n")
    for hdir in hdirs:
        ir = _load_json_file(hdir / "head_program_ir.json") or {}
        routes = _read_csv_dicts(hdir / "attention_routes.csv")
        qk_rows = _read_csv_dicts(hdir / "qk_basis_rows.csv")
        op_rows = _read_csv_dicts(hdir / "route_operator_programs.csv")
        patch = _load_json_file(hdir / "model_patch_report.json") or ir.get("model_patch_report") or {}
        bank_json = _load_json_file(hdir / "route_operator_bank.json") or ir.get("route_operator_bank") or {}
        # IMPORTANT: do NOT rebuild route-bank artifacts from route_operator_bank.json here
        # when train/val tensors are unavailable. That canonical-only rebuild was overwriting
        # true_mean/template_data/score_softmax_data CSVs created during analyze_head().
        # Only build a fallback canonical bank if the artifacts do not exist at all.
        existing_blocks_path = hdir / "route_bank_blocks.csv"
        existing_products_path = hdir / "route_product_steps.csv"
        if isinstance(bank_json, dict) and bank_json.get("routes") and (not existing_blocks_path.exists()) and (not existing_products_path.exists()):
            build_route_bank_artifacts(hdir, bank_json, routes, route_bank_size=32, block_forest_depth=2)
        pm = _patch_mode_map(patch)
        best = ir.get("best_global_qk") or {}
        head = hdir.name
        def m(mode, key):
            return pm.get(mode, {}).get(key)
        global_rel = m("global_qk", "mean_logit_rel") or m("global_qk", "logit_rel")
        ablate_rel = m("ablate", "mean_logit_rel") or m("ablate", "logit_rel")
        self_rel = m("self_template", "mean_logit_rel") or m("self_template", "logit_rel")
        verdict = "PATCH_UNKNOWN"
        try:
            if global_rel is not None and ablate_rel is not None and float(global_rel) < 0.01 and float(ablate_rel) > float(global_rel) * 5:
                verdict = "STRONG_PATCH_PROOF"
            elif global_rel is not None and float(global_rel) < 0.01:
                verdict = "PATCH_OK"
        except Exception:
            pass
        md.append(f"\n---\n\n## {head}\n")
        md.append("### 1) Real head code path\n")
        md.append("```text\nX=residual hidden\n→ RMSNorm\n→ q_proj/k_proj/v_proj head slices\n→ RoPE(Q,K)\n→ QKSoftmax read/router\n→ A @ V\n→ o_proj head slice\n→ residual write\n```\n")
        md.append("### 2) Patch proof\n")
        md.append(f"- verdict: **{verdict}**\n")
        for mode in ["global_qk", "route_hidden_qk", "route_oracle_qk", "ablate", "self_template"]:
            if mode in pm:
                x = pm[mode]
                md.append(f"- {mode}: logit_rel={x.get('mean_logit_rel') or x.get('logit_rel')} top1={x.get('mean_top1_match') or x.get('top1_match')} lossΔ={x.get('loss_delta')}\n")
        md.append("\n### 3) QK rank sweep and selected program\n")
        md.append(f"- selected: `{best}`\n")
        if qk_rows:
            md.append("\n```text\n")
            for row in qk_rows:
                md.append(f"{row.get('scope','?'):>8} {row.get('method','?'):>22} r={row.get('rank')} val={row.get('val_rel')} y={row.get('y_rel')} top1={row.get('top1')} comp={row.get('comp')} {row.get('verdict')}\n")
            md.append("```\n")
        md.append("\n### 4) Route-bank / operator decomposition\n")
        op_by_route = {str(r.get("route")): r for r in op_rows}
        deep_rows = _read_csv_dicts(hdir / "route_operator_deep_decode.csv")
        deep_by_route = {str(r.get("route")): r for r in deep_rows}
        block_rows = _read_csv_dicts(hdir / "route_bank_blocks.csv")
        block_by_route = {str(r.get("route")): r for r in block_rows}
        for r in routes:
            cid = str(r.get("cluster"))
            op = op_by_route.get(cid, {})
            formula = op.get("formula")
            if not formula:
                # fallback from stats, but label as heuristic
                formula = f"HEURISTIC {float(r.get('local4') or 0):.3f}*Local4 + {float(r.get('bos_mass') or 0):.3f}*BOS + {float(r.get('self_mass') or 0):.3f}*Self + {float(r.get('prev_mass') or 0):.3f}*Prev"
            md.append(f"- route{cid}: type={r.get('route_type')} verdict={r.get('route_verdict')} rows={r.get('rows')} train={r.get('rows_train')} val={r.get('rows_val')}\n")
            md.append(f"  - formula: `{formula}`\n")
            if op:
                md.append(f"  - operator_fit: val_rel={op.get('val_rel')} val_top1={op.get('val_top1')} val_y_rel={op.get('val_y_rel')}\n")
            drow = deep_by_route.get(cid, {})
            if drow:
                md.append(f"  - best_family: {drow.get('best_family')} val_rel={drow.get('best_family_val_rel')}\n")
                md.append(f"  - best_product_step: `{drow.get('best_product')}`\n")
            brow = block_by_route.get(cid, {})
            if brow:
                md.append(f"  - route_bank_block: size={brow.get('block_size')} energy={brow.get('energy')}\n")
            md.append(f"  - stats: self={r.get('self_mass')} prev={r.get('prev_mass')} bos={r.get('bos_mass')} local4={r.get('local4')} entropy={r.get('entropy')}\n")
            md.append(f"  - examples: `{r.get('examples')}`\n")
        md.append("\n### 5) What this means\n")
        md.append("- Router/program logic is visible in attention rows and route-bank formulas, not inside raw Wq/Wk as clean blocks.\n")
        md.append("- QK learned factors validate the read-space; operator formulas explain the visible attention behavior per route.\n")
        md.append("- Ablate/self controls tell whether the decoded program is actually replacing a useful head.\n")
        summary_rows.append({
            "head": head, "verdict": verdict, "rank": best.get("rank"), "val_rel": best.get("val_rel"),
            "y_rel": best.get("y_rel"), "global_logit_rel": global_rel, "ablate_logit_rel": ablate_rel,
            "self_logit_rel": self_rel, "routes": len(routes),
        })
        # per-head route bank json
        (hdir / "route_bank.json").write_text(json_dumps_safe({"head": head, "best_global_qk": best, "patch_verdict": verdict, "routes": routes, "operator_programs": op_rows, "route_blocks": block_rows, "deep_decode": deep_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "logic_summary.csv", summary_rows)
    (out_dir / "logic_decomposition_report.md").write_text("\n".join(md), encoding="utf-8")



# ---------------- integrated affine/route/score/value/patch report ----------------

def _as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _fmt_float(x: Any, nd: int = 4) -> str:
    v = _as_float(x, None)
    if v is None:
        return "?"
    if abs(v) < 1e-4 and v != 0:
        return f"{v:.2e}"
    return f"{v:.{nd}f}"


def _load_affine_head_summary(circuit_dir: Path, hname: str) -> Dict[str, Any]:
    if not str(circuit_dir or "").strip():
        return {}
    circuit_dir = Path(circuit_dir)
    hdir = circuit_dir / hname
    if not hdir.exists() and circuit_dir.name == hname:
        hdir = circuit_dir
    if not hdir.exists():
        return {}
    return {
        "head_dir": str(hdir),
        "summary": _load_json_file(hdir / "summary.json") or {},
        "per_prompt_checks": _read_csv_dicts(hdir / "per_prompt_checks.csv"),
        "basis_functional": _read_csv_dicts(hdir / "basis_functional.csv"),
        "token_flow_examples": _load_json_file(hdir / "token_flow_examples.json") or [],
    }


def _best_basis_rows(rows: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    def score(r: Dict[str, Any]) -> Tuple[float, float, int]:
        y = _as_float(r.get("Y_rel_from_QK_basis"), 999.0) or 999.0
        a = _as_float(r.get("A_rel"), 999.0) or 999.0
        rank = int(_as_float(r.get("rank"), 999999) or 999999)
        return (float(y), float(a), rank)
    return sorted([r for r in rows if isinstance(r, dict)], key=score)[:k]


def _best_score_gain(deep_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    best_gain = -1e18
    for r in deep_rows:
        flat = _as_float(r.get("flat_val_y_rel"), None)
        score = _as_float(r.get("score_val_y_rel"), None)
        if flat is None or score is None:
            flat = _as_float(r.get("flat_val_rel"), None)
            score = _as_float(r.get("score_val_rel"), None)
        if flat is None or score is None:
            continue
        gain = flat - score
        if gain > best_gain:
            best_gain = gain
            best = dict(r)
            best["score_gain"] = gain
    return best


def write_integrated_token_examples(hdir: Path, seqs: List[SeqHeadData], labels_by_prompt: Dict[int, torch.Tensor],
                                    route_summaries: List[Dict[str, Any]], route_operator_rows: List[Dict[str, Any]],
                                    max_prompts: int = 3, max_tokens: int = 24) -> None:
    route_by_id = {str(r.get("cluster")): r for r in route_summaries}
    op_by_route = {str(r.get("route")): r for r in route_operator_rows}
    deep_by_route = {str(r.get("route")): r for r in _read_csv_dicts(hdir / "route_operator_deep_decode.csv")}
    examples: List[Dict[str, Any]] = []
    for s in seqs[:max(1, int(max_prompts))]:
        labs = labels_by_prompt.get(int(s.prompt_id))
        rows: List[Dict[str, Any]] = []
        T = int(s.A.shape[0])
        for i in range(min(T, max(1, int(max_tokens)))):
            cid = int(labs[i].item()) if labs is not None and i < int(labs.numel()) else -1
            cid_s = str(cid)
            route = route_by_id.get(cid_s, {})
            op = op_by_route.get(cid_s, {})
            deep = deep_by_route.get(cid_s, {})
            row = s.A[i].float()
            vals, idx = torch.topk(row[:i+1], k=min(5, i+1))
            rows.append({
                "pos": int(i), "token": s.tokens[i] if i < len(s.tokens) else "?",
                "route": cid, "route_type": route.get("route_type"), "route_verdict": route.get("route_verdict"),
                "best_logic": deep.get("best_logic"),
                "score_formula": deep.get("score_formula") or op.get("score_formula"),
                "attention_formula": op.get("formula"),
                "flat_A_rel": deep.get("flat_val_rel") or op.get("val_rel"),
                "score_A_rel": deep.get("score_val_rel"),
                "flat_Y_rel": deep.get("flat_val_y_rel") or op.get("val_y_rel"),
                "score_Y_rel": deep.get("score_val_y_rel"),
                "self_mass": float(row[i].item()),
                "prev_mass": float(row[i-1].item()) if i > 0 else 0.0,
                "bos_mass": float(row[0].item()),
                "local4": local_mass(row, i, 4),
                "top_read": [{"pos": int(j), "token": s.tokens[int(j)] if int(j) < len(s.tokens) else "?", "p": float(v)} for v, j in zip(vals.tolist(), idx.tolist())],
                "Y_norm": float(torch.linalg.norm(s.Y[i].float()).item()),
            })
        examples.append({"prompt_id": int(s.prompt_id), "suite": s.suite, "text": s.text, "rows": rows})
    (hdir / "integrated_token_examples.json").write_text(json_dumps_safe(examples, indent=2, ensure_ascii=False), encoding="utf-8")


def build_integrated_circuit_report_from_outdir(out_dir: Path, heads: str = "", circuit_report_dir: str = "") -> None:
    out_dir = Path(out_dir)
    cdir = Path(circuit_report_dir) if str(circuit_report_dir or "").strip() else None
    if heads:
        wanted = set(h.strip() for h in heads.replace(';', ',').split(',') if h.strip())
        wanted |= {("L" + h.replace(":", "H")) for h in list(wanted) if not h.startswith("L")}
    else:
        wanted = set()
    hdirs = sorted([p for p in out_dir.glob("L*H*") if p.is_dir()])
    if wanted:
        hdirs = [p for p in hdirs if p.name in wanted]
    summary_rows: List[Dict[str, Any]] = []
    root_md: List[str] = ["# Integrated Qwen head circuit report\n",
        "Pipeline: exact affine-bilinear circuit → route/token behavior → score-space decomposition → value/write contribution → model patch effect.\n"]
    root_md.append(f"\nAffine circuit artifact dir: `{cdir}`\n" if cdir else "\nAffine circuit artifact dir: **not provided**. Exact affine verification will be marked missing.\n")

    for hdir in hdirs:
        hname = hdir.name
        ir = _load_json_file(hdir / "head_program_ir.json") or {}
        routes = _read_csv_dicts(hdir / "attention_routes.csv")
        deep_rows = _read_csv_dicts(hdir / "route_operator_deep_decode.csv")
        product_rows = _read_csv_dicts(hdir / "route_product_steps.csv")
        patch = _load_json_file(hdir / "model_patch_report.json") or ir.get("model_patch_report") or {}
        audit = _load_json_file(hdir / "route_bank_audit.json") or {}
        token_examples = _load_json_file(hdir / "integrated_token_examples.json") or []
        affine = _load_affine_head_summary(cdir, hname) if cdir else {}
        affine_summary = affine.get("summary") or {}
        verify = affine_summary.get("verify") or {}
        basis_rows = affine.get("basis_functional") or []
        basis_best = affine_summary.get("basis_best") or _best_basis_rows(basis_rows, 3)
        pm = _patch_mode_map(patch)
        best_qk = ir.get("best_global_qk") or {}
        score_gain_row = _best_score_gain(deep_rows)
        global_patch = pm.get("global_qk", {})
        ablate_patch = pm.get("ablate", {})
        global_logit = global_patch.get("mean_logit_rel") or global_patch.get("logit_rel")
        ablate_logit = ablate_patch.get("mean_logit_rel") or ablate_patch.get("logit_rel")
        verdict = "PATCH_UNKNOWN"
        gr = _as_float(global_logit, None); ar = _as_float(ablate_logit, None)
        if gr is not None and ar is not None:
            verdict = "STRONG_PATCH_PROOF" if (gr < 0.01 and ar > gr * 5) else ("PATCH_OK" if gr < 0.02 else "PATCH_WEAK")

        md: List[str] = []
        md.append(f"# Integrated circuit report — {hname}\n")
        md.append("## Chain view\n```text\nexact affine/bilinear circuit\n  -> route/token behavior\n  -> score decomposition per route/token\n  -> value/write contribution\n  -> model patch effect\n```\n")
        md.append("\n## 1) Exact affine-bilinear circuit\n")
        exact_program = affine_summary.get("exact_program") or {
            "QK_affine_normalized": "score_ij = Xaug_i^T M_qk_aug[i-j] Xaug_j, Xaug=[RMSNorm(Xraw),1]",
            "VO_affine_normalized": "Y_i = sum_j A[i,j] * Xaug_j @ C_vo_aug.T",
            "Runtime": "A = causal_softmax(scores), Z=A@V, Y=Z@O_head.T",
        }
        md.append("```text\n")
        for k, v in exact_program.items():
            md.append(f"{k}: {v}\n")
        md.append("```\n")
        if verify:
            md.append("\nVerification from affine target run:\n")
            for key in ["qk_score_aug_rel_covered_normX", "qk_score_linear_no_bias_rel_covered_normX", "A_rel_rows_fully_covered_from_aug_scores", "V_from_affine_Wv_rel", "Y_from_Cvo_aug_rel", "Y_block_raw_affine_rel"]:
                if key in verify:
                    md.append(f"- {key}: `{_fmt_float(verify.get(key), 6)}`\n")
        else:
            md.append("\n- Exact affine verification: **missing**. Run the affine target script and pass `--circuit-report-dir`.\n")

        md.append("\n## 2) Route/token behavior\n")
        if routes:
            md.append("| route | type | rows | entropy | self | prev | BOS | local4 | examples |\n|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
            for r in routes:
                md.append(f"| {r.get('cluster')} | {r.get('route_type')} | {r.get('rows')} | {_fmt_float(r.get('entropy'))} | {_fmt_float(r.get('self_mass'))} | {_fmt_float(r.get('prev_mass'))} | {_fmt_float(r.get('bos_mass'))} | {_fmt_float(r.get('local4'))} | `{str(r.get('examples',''))[:120]}` |\n")
        else:
            md.append("No route table found.\n")

        md.append("\n## 3) Score decomposition per route\n")
        if deep_rows:
            md.append("| route | type | best_logic | flat A/Y | score A/Y | true/template/score bank | score formula |\n|---:|---|---|---:|---:|---|---|\n")
            route_type_by_id = {str(r.get("cluster")): r.get("route_type") for r in routes}
            for d in deep_rows:
                rid = str(d.get("route"))
                flat = f"{_fmt_float(d.get('flat_val_rel'))}/{_fmt_float(d.get('flat_val_y_rel'))}"
                sc = f"{_fmt_float(d.get('score_val_rel'))}/{_fmt_float(d.get('score_val_y_rel'))}"
                banks = f"true={_fmt_float(d.get('true_template_rel'))}; score={_fmt_float(d.get('true_score_rel'))}"
                formula = str(d.get("score_formula") or d.get("formula") or "")[:180]
                md.append(f"| {rid} | {route_type_by_id.get(rid,'?')} | {d.get('best_logic')} | {flat} | {sc} | {banks} | `{formula}` |\n")
        else:
            md.append("No score/deep decode table found.\n")
        if token_examples:
            md.append("\n### Token-level linked examples\n")
            for ex in token_examples[:2]:
                md.append(f"\nPrompt {ex.get('prompt_id')} `{ex.get('suite')}`: {str(ex.get('text',''))[:160]}\n")
                md.append("| pos | token | route | best_logic | top read | Y norm | score formula |\n|---:|---|---:|---|---|---:|---|\n")
                for row in (ex.get("rows") or [])[:12]:
                    top = ", ".join([f"{t.get('pos')}:{str(t.get('token'))[:16]}={_fmt_float(t.get('p'),3)}" for t in (row.get("top_read") or [])[:3]])
                    formula = str(row.get("score_formula") or row.get("attention_formula") or "")[:120]
                    md.append(f"| {row.get('pos')} | `{row.get('token')}` | {row.get('route')} | {row.get('best_logic')} | {top} | {_fmt_float(row.get('Y_norm'))} | `{formula}` |\n")

        md.append("\n## 4) Value/write contribution\n")
        md.append("Route formulas are judged by A-rel and by Y=(A@V)@O.\n")
        if verify:
            md.append(f"- Exact VO affine write Y error: `{_fmt_float(verify.get('Y_from_Cvo_aug_rel'), 6)}`\n")
            md.append(f"- Raw block affine write Y error: `{_fmt_float(verify.get('Y_block_raw_affine_rel'), 6)}`\n")
        if basis_best:
            md.append("- Best functional basis rows:\n")
            for b in basis_best[:3]:
                md.append(f"  - {b.get('basis_kind')} r={b.get('rank')} A_rel={_fmt_float(b.get('A_rel'))} Z_rel={_fmt_float(b.get('Z_rel'))} Y_QK={_fmt_float(b.get('Y_rel_from_QK_basis'))} Y_V={_fmt_float(b.get('Y_rel_from_V_basis_true_A'))}\n")
        if product_rows:
            by_kind: Dict[str, List[Dict[str, Any]]] = {}
            for p in product_rows:
                by_kind.setdefault(str(p.get("bank_kind", "?")), []).append(p)
            md.append("- Product/forest bank kinds present: " + ", ".join(sorted(by_kind)) + "\n")
            for bk, rows in sorted(by_kind.items()):
                for p in sorted(rows, key=lambda r: _as_float(r.get("rel"), 999.0) or 999.0)[:2]:
                    md.append(f"  - {bk}: route={p.get('route')} kind={p.get('kind')} rel={_fmt_float(p.get('rel'))} steps=`{p.get('steps')}` family={p.get('family_path')}\n")

        md.append("\n## 5) Model patch effect\n")
        md.append(f"- patch verdict: **{verdict}**\n")
        for mode in ["global_qk", "route_hidden_qk", "route_oracle_qk", "ablate", "self_template"]:
            if mode in pm:
                x = pm[mode]
                md.append(f"- {mode}: logit_rel={_fmt_float(x.get('mean_logit_rel') or x.get('logit_rel'),6)} top1={x.get('mean_top1_match') or x.get('top1_match')} lossΔ={_fmt_float(x.get('loss_delta'),6)} KL={_fmt_float(x.get('kl_orig_to_patch') or x.get('mean_kl_orig_to_patch'),6)}\n")

        md.append("\n## 6) Linked verdict\n")
        if verify:
            qk_exact = _as_float(verify.get("qk_score_aug_rel_covered_normX"), None)
            y_exact = _as_float(verify.get("Y_from_Cvo_aug_rel"), None)
            if qk_exact is not None and y_exact is not None and qk_exact < 1e-4 and y_exact < 1e-4:
                md.append("- Exact affine/bilinear target is verified: remaining error is interpretability/compression, not extraction.\n")
        if score_gain_row:
            md.append(f"- Biggest score-space gain: route{score_gain_row.get('route')} gain≈{_fmt_float(score_gain_row.get('score_gain'))}; logits/score-bias explain this route better than fitting final A probabilities.\n")
        if audit:
            missing = audit.get("missing") or []
            md.append(f"- Bank audit: {'missing '+str(missing) if missing else 'OK: true/template/score/canonical present'}\n")
        if best_qk:
            md.append(f"- Selected QK program: {best_qk.get('method')} r={best_qk.get('rank')} val_rel={_fmt_float(best_qk.get('val_rel'))} y_rel={_fmt_float(best_qk.get('y_rel'))} comp={_fmt_float(best_qk.get('comp'))} verdict={best_qk.get('verdict')}\n")

        (hdir / "integrated_circuit_report.md").write_text("".join(md), encoding="utf-8")
        (hdir / "integrated_circuit_report.json").write_text(json_dumps_safe({
            "head": hname, "affine_summary": affine_summary, "best_global_qk": best_qk,
            "routes": routes, "deep_decode": deep_rows, "product_steps": product_rows,
            "patch": patch, "audit": audit, "token_examples": token_examples,
            "linked_verdict": {"patch_verdict": verdict, "score_gain_route": score_gain_row},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_rows.append({
            "head": hname,
            "exact_qk_score_aug_rel": verify.get("qk_score_aug_rel_covered_normX"),
            "exact_Y_vo_rel": verify.get("Y_from_Cvo_aug_rel"),
            "qk_rank": best_qk.get("rank"), "qk_val_rel": best_qk.get("val_rel"), "qk_y_rel": best_qk.get("y_rel"),
            "score_gain_route": score_gain_row.get("route"), "score_gain": score_gain_row.get("score_gain"),
            "patch_verdict": verdict, "global_patch_logit_rel": global_logit, "ablate_patch_logit_rel": ablate_logit,
            "routes": len(routes),
        })
        root_md.append(f"\n---\n\n## {hname}\n")
        root_md.append(f"- exact QK score rel: `{_fmt_float(verify.get('qk_score_aug_rel_covered_normX'),6)}`; exact VO Y rel: `{_fmt_float(verify.get('Y_from_Cvo_aug_rel'),6)}`\n")
        root_md.append(f"- selected QK: r={best_qk.get('rank')} val={_fmt_float(best_qk.get('val_rel'))} y={_fmt_float(best_qk.get('y_rel'))} comp={_fmt_float(best_qk.get('comp'))}\n")
        root_md.append(f"- biggest score gain: route{score_gain_row.get('route')} gain={_fmt_float(score_gain_row.get('score_gain'))}\n")
        root_md.append(f"- patch: {verdict}; global logit rel={_fmt_float(global_logit,6)}; ablate logit rel={_fmt_float(ablate_logit,6)}\n")
        root_md.append(f"- read: `{hname}/integrated_circuit_report.md`\n")
    write_csv(out_dir / "integrated_circuit_summary.csv", summary_rows)
    (out_dir / "integrated_circuit_report.md").write_text("".join(root_md), encoding="utf-8")

# ---------------- CLI ----------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--heads", default="23:1")
    ap.add_argument("--stage", default="all", choices=["all", "deep", "logic"], help="all/deep runs model analysis; logic only rebuilds reports from existing out-dir")
    ap.add_argument("--prompt-suites", default="all")
    ap.add_argument("--prompts-per-suite", type=int, default=24)
    ap.add_argument("--same-text-repeats", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--val-frac", type=float, default=0.33)
    ap.add_argument("--ranks", default="4,8,16,32,64")
    ap.add_argument("--qk-methods", default="pca,learned")
    ap.add_argument("--qk-scale", default="rank", choices=["rank", "head_dim"])
    ap.add_argument("--qk-learned-steps", type=int, default=300)
    ap.add_argument("--qk-lr", type=float, default=0.002)
    ap.add_argument("--qk-wd", type=float, default=1e-4)
    ap.add_argument("--qk-patience", type=int, default=8)
    ap.add_argument("--qk-eval-every", type=int, default=25, help="learned QK validation cadence; larger is faster")
    ap.add_argument("--train-batch-size", type=int, default=16, help="padded sequence batch size for QK learned/eval")
    ap.add_argument("--route-qk-max-routes", type=int, default=0, help="0=all eligible routes; otherwise train only top-N route QK programs")
    ap.add_argument("--cache-dir", default="", help="persistent cache for learned QK Pq/Pk factors; empty disables")
    ap.add_argument("--cache-version", default="v1", help="bump this when training/objective logic changes")
    ap.add_argument("--rebuild-cache", action="store_true", help="ignore existing QK cache and overwrite")
    ap.add_argument("--profile", action="store_true", help="print and save per-stage timings")
    ap.add_argument("--logic-operator-fit", action=argparse.BooleanOptionalAction, default=True, help="fit real attention-operator dictionary per route")
    ap.add_argument("--operator-templates", default=",".join(DEFAULT_OPERATOR_TEMPLATES))
    ap.add_argument("--operator-dynamic", action=argparse.BooleanOptionalAction, default=True, help="include HiddenSimilarity/KeySimilarity/GlobalQK/RouteQK dynamic operator atoms")
    ap.add_argument("--score-operator-fit", action=argparse.BooleanOptionalAction, default=True, help="fit score/logit-level operator dictionary S_hat then softmax, beside A-level operator fit")
    ap.add_argument("--score-templates", default="", help="comma-separated score/logit operator names; default rich score bank")
    ap.add_argument("--score-ridge", type=float, default=0.0001)
    ap.add_argument("--score-mine-residual", action=argparse.BooleanOptionalAction, default=True, help="mine qtype->ktype score-bias atoms from positive logit residual")
    ap.add_argument("--score-mine-topk", type=int, default=8)
    ap.add_argument("--score-mine-min-score", type=float, default=1e-4)
    ap.add_argument("--score-max-terms", type=int, default=8)
    ap.add_argument("--score-include-full-qk", action=argparse.BooleanOptionalAction, default=False, help="diagnostic only: include true full QK score atom; normally keep false")
    ap.add_argument("--operator-mine-residual", action=argparse.BooleanOptionalAction, default=True, help="mine residual Qtype->Ktype atoms after first route fit")
    ap.add_argument("--operator-mine-topk", type=int, default=8)
    ap.add_argument("--operator-mine-min-score", type=float, default=1e-4)
    ap.add_argument("--operator-subroutes", action=argparse.BooleanOptionalAction, default=True, help="fit q-token-type subroutes inside each attention route")
    ap.add_argument("--operator-selection", default="functional_balanced", choices=["attention", "functional", "functional_balanced"], help="select formula by A-only or Z/Y functional metrics")
    ap.add_argument("--operator-allow-oracle-qk", action="store_true", help="debug only: allow exact full QK as an operator atom")
    ap.add_argument("--product-depth", type=int, default=2)
    ap.add_argument("--product-fit-steps", type=int, default=80)
    ap.add_argument("--product-private-svd", type=int, default=1)
    ap.add_argument("--product-bank-kinds", default="true_mean,score_softmax_data,template_data,canonical", help="which route banks to run product search on")
    ap.add_argument("--product-beam", type=int, default=24, help="beam size for typed searched product")
    ap.add_argument("--product-per-family", type=int, default=4, help="max candidates per product family")
    ap.add_argument("--operator-ridge", type=float, default=1e-4)
    ap.add_argument("--operator-max-terms", type=int, default=6)
    ap.add_argument("--route-bank-size", type=int, default=32, help="canonical T for route-bank block matrix / product-step logic probes")
    ap.add_argument("--block-forest-depth", type=int, default=2, help="flat-DAG quadrant depth for route block forest IR (no recursive evaluation)")
    ap.add_argument("--product-topk", type=int, default=8, help="number of canonical product-step candidates to save per route")
    ap.add_argument("--logic-heads", default="", help="optional comma heads for logic stage, e.g. L2H1,L9H11")
    ap.add_argument("--integrated-report", action="store_true", help="write exact-circuit -> route/token -> score -> value/write -> patch integrated report")
    ap.add_argument("--circuit-report-dir", default="", help="optional out-dir from qwen_circuit_matrix_targets_v2_affine_basis.py for exact affine/bilinear verification")
    ap.add_argument("--integrated-max-example-prompts", type=int, default=3)
    ap.add_argument("--integrated-max-example-tokens", type=int, default=24)
    ap.add_argument("--clusters", default="2,3,4,6,8")
    ap.add_argument("--fit-route-qk", action="store_true")
    ap.add_argument("--route-ranks", default="")
    ap.add_argument("--route-qk-steps", type=int, default=200)
    ap.add_argument("--min-route-rows", type=int, default=64)
    ap.add_argument("--major-route-rows", type=int, default=256)
    ap.add_argument("--accept-rel", type=float, default=0.05)
    ap.add_argument("--risky-rel", type=float, default=0.10)
    ap.add_argument("--explain-rel", type=float, default=0.15)
    ap.add_argument("--min-compression", type=float, default=1.10)
    ap.add_argument("--select-policy", default="balanced", choices=["quality", "compression", "balanced", "min_rank_within_epsilon", "minrank", "deploy_min_rank", "epsilon"])
    ap.add_argument("--select-eps-rel", type=float, default=0.012, help="for min_rank_within_epsilon: allow this extra A_rel versus quality-best")
    ap.add_argument("--select-eps-y", type=float, default=0.012, help="for min_rank_within_epsilon: allow this extra Y_rel versus quality-best")
    ap.add_argument("--patch-model", action="store_true", help="run model-level interpretability patch for each analyzed head")
    ap.add_argument("--patch-modes", default="global_qk,route_oracle_qk,route_hidden_qk,ablate,self_template",
                    help="comma list: global_qk,route_oracle_qk,route_hidden_qk,ablate,self_template")
    ap.add_argument("--patch-eval-prompts", type=int, default=32)
    ap.add_argument("--fast-collect", action="store_true", default=True, help="one model forward per prompt for all requested heads")
    ap.add_argument("--no-fast-collect", dest="fast_collect", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./qwen_single_head_program_decompile_v1")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.stage == "logic":
        build_logic_report_from_outdir(out_dir, args.logic_heads or args.heads)
        if getattr(args, "integrated_report", False):
            build_integrated_circuit_report_from_outdir(out_dir, args.logic_heads or args.heads, getattr(args, "circuit_report_dir", ""))
        print(f"DONE wrote logic/integrated reports in {out_dir}", flush=True)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = get_dtype(args.dtype)
    print(f"loading model {args.model} dtype={dtype} device={args.device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        ).to(args.device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
        ).to(args.device)
    model.eval()

    cfg = model.config
    head_specs = parse_heads(args.heads, int(cfg.num_hidden_layers), int(cfg.num_attention_heads))
    prompts = build_prompts(args.prompt_suites, args.prompts_per_suite, args.same_text_repeats)
    (out_dir / "prompts.json").write_text(json_dumps_safe(prompts, indent=2, ensure_ascii=False), encoding="utf-8")

    if getattr(args, "fast_collect", True):
        print(f"fast collecting Q/K/V/A/Y for {len(head_specs)} requested heads: one model forward per prompt", flush=True)
        args._precollected_head_data = collect_requested_heads_data(model, tok, prompts, head_specs, args.max_length, args.device)
        print("fast collect done", flush=True)

    summaries = []
    for layer_idx, head_idx in head_specs:
        summaries.append(analyze_head(model, tok, prompts, layer_idx, head_idx, args, out_dir))

    (out_dir / "summary.json").write_text(json_dumps_safe({"args": public_args_dict(args), "heads": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    # flat combined CSVs
    all_qk = []
    all_routes = []
    for h in summaries:
        hdir = out_dir / f"L{h['layer']}H{h['head']}"
        # do not re-read CSV; use summaries for route overview only
        for r in h["routes"]:
            rr = dict(r)
            rr["layer"] = h["layer"]
            rr["head"] = h["head"]
            all_routes.append(rr)
    write_csv(out_dir / "attention_routes_overview.csv", all_routes)

    md = ["# Qwen single-head program decompile summary — v6.2 fixedbanks\n"]
    for h in summaries:
        md.append(f"## L{h['layer']}H{h['head']}")
        md.append(f"best_global_qk: `{h['best_global_qk']}`")
        md.append(f"cluster_info: `{h['cluster_info']}`")
        md.append(f"router_acc: `{h['router_acc']}`")
        for r in h["routes"]:
            md.append(f"- route {r['cluster']} `{r['route_type']}` rows={r['rows']} entropy={r['entropy']:.3f} self={r['self_mass']:.3f} prev={r['prev_mass']:.3f} bos={r['bos_mass']:.3f} local4={r['local4']:.3f} examples={r['examples']}")
        md.append("")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    build_logic_report_from_outdir(out_dir, args.logic_heads or args.heads)
    print(f"\nDONE wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
