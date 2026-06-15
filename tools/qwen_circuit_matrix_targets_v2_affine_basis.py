#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_circuit_matrix_targets_v2_affine_basis.py

Purpose
-------
Real circuit-matrix extraction for one/more Qwen2 attention heads, using the user's
matrix-program principle:

  code path -> exact affine/bilinear circuit targets -> basis/search diagnostics -> functional checks

v2 fixes v1's main hole:
  * q/k/v projection BIASES are included via homogeneous coordinates x_aug=[x,1]
  * QK is exported as an affine-bilinear target:
        score_ij = x_aug_i^T M_qk_aug[delta] x_aug_j
  * VO is exported as an affine read/write target:
        payload_j = x_aug_j @ C_vo_aug.T
        Y_i = sum_j A[i,j] payload_j

This is NOT a generic probe/PCA script. The exact circuit targets are the main result.
Basis diagnostics are optional/secondary and are evaluated functionally in the same
style as qwen_program_decompiler_v6: A_rel/KL/top1/Z_rel/Y_rel, not by pretty plots.

Targets
-------
Normalized input x = RMSNorm(raw):

  q_pre = x_aug @ Wq_aug.T,  Wq_aug=[Wq | bq]
  k_pre = x_aug @ Wk_aug.T,  Wk_aug=[Wk | bk]
  v     = x_aug @ Wv_aug.T,  Wv_aug=[Wv | bv]

  q_i = q_pre_i @ R_i.T
  k_j = k_pre_j @ R_j.T

  M_qk_aug[d] = Wq_aug.T @ R_i.T @ R_j @ Wk_aug / sqrt(head_dim), d=i-j
  score_ij    = x_aug_i @ M_qk_aug[d] @ x_aug_j.T

  C_vo_aug = Wo_head @ Wv_aug
  Y_i      = sum_j A[i,j] * (x_aug_j @ C_vo_aug.T)

Raw input block form:

  x_norm_j = D_j @ x_raw_j, D_j=diag(gamma/rms(x_raw_j))
  x_aug_j  = [D_j @ x_raw_j, 1]
  block(i,j) = A[i,j] * [C_vo_linear @ D_j, b_vo]

Outputs
-------
  summary.json / summary.md
  per_prompt_checks.csv
  basis_pca_functional.csv
  token_flow_examples.json
  circuit_targets.pt optional

Example
-------
  python qwen_circuit_matrix_targets_v2_affine_basis.py \
    --base-script ./qwen_program_decompiler_v6_scorehybrid.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --dtype fp16 \
    --attn-implementation eager --heads 2:1 \
    --prompt-suites all --prompts-per-suite 8 --same-text-repeats 2 \
    --max-length 192 --max-delta 64 --svd-ranks 4,8,16,32,64,128 \
    --basis-ranks 4,8,16,32,64 --fit-learned-qk-basis --learned-steps 120 \
    --save-tensors --out-dir ./qwen_circuit_targets_v2_L2H1_fast

Self-test:
  python qwen_circuit_matrix_targets_v2_affine_basis.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


# ---------------- utilities ----------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.detach().float()
    b = b.detach().float()
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(eps))


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def json_sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return str(obj) if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        if obj.numel() <= 16:
            return obj.detach().cpu().tolist()
        return {"__tensor__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if np is not None:
        if isinstance(obj, np.generic):
            return json_sanitize(obj.item())
        if isinstance(obj, np.ndarray):
            if obj.size <= 16:
                return obj.tolist()
            return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]
    return str(obj)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(json_sanitize(obj), ensure_ascii=False, indent=2), encoding="utf-8")


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
            w.writerow({k: json_sanitize(r.get(k, "")) for k in keys})


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def parse_heads(s: str, n_layers: int, n_heads: int) -> List[Tuple[int, int]]:
    if s.strip().lower() == "all":
        return [(l, h) for l in range(n_layers) for h in range(n_heads)]
    out: List[Tuple[int, int]] = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad head spec {part!r}; use L:H")
        a, b = part.split(":", 1)
        l, h = int(a), int(b)
        if not (0 <= l < n_layers and 0 <= h < n_heads):
            raise ValueError(f"head out of range {l}:{h}; model has layers={n_layers}, heads={n_heads}")
        out.append((l, h))
    return list(dict.fromkeys(out))


def get_dtype(name: str):
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(name)


def load_base_module(path: str):
    p = Path(path)
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("qwen_decomp_base", str(p))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen_decomp_base"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ---------------- RoPE explicit matrices ----------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_matrix(D: int, device=None, dtype=torch.float32) -> torch.Tensor:
    device = device or torch.device("cpu")
    P = torch.zeros(D, D, device=device, dtype=dtype)
    h = D // 2
    for i in range(h):
        P[i, h + i] = -1.0
        P[h + i, i] = 1.0
    return P


def rope_col_matrix(cos_row: torch.Tensor, sin_row: torch.Tensor) -> torch.Tensor:
    """Column-space matrix R: q_rot_col = R @ q_col. Row form: q_rot_row = q_row @ R.T."""
    cos_row = cos_row.detach().float()
    sin_row = sin_row.detach().float()
    D = int(cos_row.numel())
    P = rotate_half_matrix(D, cos_row.device, cos_row.dtype)
    return torch.diag(cos_row) + torch.diag(sin_row) @ P


def apply_rope_rows(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return q * cos + rotate_half(q) * sin


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1)


# ---------------- collected data ----------------

@dataclass
class CircuitSeq:
    prompt_id: int
    suite: str
    text: str
    token_ids: List[int]
    tokens: List[str]
    Xraw: torch.Tensor       # [T,H] residual before RMSNorm
    Xn: torch.Tensor         # [T,H] post RMSNorm
    Xaug: torch.Tensor       # [T,H+1] [Xn,1]
    rms_scale: torch.Tensor  # [T,H], Xn ~= Xraw*rms_scale
    Q_pre: torch.Tensor      # [T,D]
    K_pre: torch.Tensor      # [T,D]
    Q: torch.Tensor          # [T,D] after RoPE
    K: torch.Tensor          # [T,D] after RoPE
    V: torch.Tensor          # [T,D]
    A: torch.Tensor          # [T,T]
    Y: torch.Tensor          # [T,H]
    cos: torch.Tensor        # [T,D]
    sin: torch.Tensor        # [T,D]
    rms_rec_err: float


def get_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find decoder layers on model")


def compute_position_embeddings(model: Any, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    rotary = getattr(model.model, "rotary_emb", None) if hasattr(model, "model") else None
    if rotary is None:
        raise RuntimeError("model.model.rotary_emb not found")
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return rotary(position_ids)


def build_prompts(base_mod: Any, suites: str, prompts_per_suite: int, same_text_repeats: int) -> List[Dict[str, str]]:
    if base_mod is not None and hasattr(base_mod, "build_prompts"):
        return base_mod.build_prompts(suites, prompts_per_suite, same_text_repeats)
    base = [
        {"suite": "text", "text": "Explain why rivers are important for cities."},
        {"suite": "code", "text": "Write a Python function that sums a list."},
        {"suite": "math", "text": "Solve 3x + 5 = 20."},
        {"suite": "symbols", "text": "JSON: {\"name\": \"Alice\", \"score\": 42}"},
    ]
    rows = []
    for _ in range(max(1, prompts_per_suite)):
        rows.extend(base)
    for _ in range(same_text_repeats):
        rows.append({"suite": "same_repeat", "text": "The same calibration sentence repeated."})
    return rows


@torch.no_grad()
def collect_circuit_data(model: Any, tokenizer: Any, prompts: List[Dict[str, str]],
                         layer_idx: int, head_idx: int, max_length: int, device: str) -> Tuple[List[CircuitSeq], Dict[str, Any]]:
    layers = get_layers(model)
    layer = layers[layer_idx]
    attn = layer.self_attn
    cfg = model.config
    H = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    D = int(getattr(cfg, "head_dim", H // n_heads))
    kv_groups = n_heads // n_kv
    kv_idx = int(head_idx) // kv_groups

    gamma = layer.input_layernorm.weight.detach().float().to(device)
    eps = float(getattr(layer.input_layernorm, "variance_epsilon", getattr(layer.input_layernorm, "eps", 1e-6)))

    o_w = attn.o_proj.weight.detach().float()[:, head_idx * D:(head_idx + 1) * D].to(device)
    out: List[CircuitSeq] = []
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
        Xraw = outputs.hidden_states[layer_idx].detach()  # [1,T,H]
        Xn = layer.input_layernorm(Xraw).detach()
        rms = torch.sqrt(Xraw.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        scale = gamma.view(1, 1, H) / rms
        Xn_recon = Xraw.float() * scale
        rms_err = rel_err(Xn_recon[0], Xn[0].float())

        q = attn.q_proj(Xn).view(1, T, n_heads, D).transpose(1, 2).contiguous()
        k = attn.k_proj(Xn).view(1, T, n_kv, D).transpose(1, 2).contiguous()
        v = attn.v_proj(Xn).view(1, T, n_kv, D).transpose(1, 2).contiguous()
        pos = torch.arange(T, device=device).unsqueeze(0)
        cos, sin = compute_position_embeddings(model, Xn, pos)
        cos2 = cos[0] if cos.dim() == 3 else cos
        sin2 = sin[0] if sin.dim() == 3 else sin
        q_rot = apply_rope_rows(q, cos2.unsqueeze(0).unsqueeze(0), sin2.unsqueeze(0).unsqueeze(0))
        k_rot = apply_rope_rows(k, cos2.unsqueeze(0).unsqueeze(0), sin2.unsqueeze(0).unsqueeze(0))
        Qh = q_rot[0, head_idx].float()
        Kh = k_rot[0, kv_idx].float()
        Vh = v[0, kv_idx].float()
        scores = (Qh @ Kh.T) / math.sqrt(D)
        A = causal_softmax(scores.float())
        Y = (A @ Vh) @ o_w.T
        tok_ids = input_ids[0].detach().cpu().tolist()
        try:
            toks = tokenizer.convert_ids_to_tokens(tok_ids)
        except Exception:
            toks = [str(x) for x in tok_ids]
        Xn_cpu = Xn[0].float().cpu()
        Xaug_cpu = torch.cat([Xn_cpu, torch.ones(T, 1)], dim=1)
        out.append(CircuitSeq(
            prompt_id=pi, suite=str(pr.get("suite", "?")), text=pr["text"], token_ids=tok_ids, tokens=toks,
            Xraw=Xraw[0].float().cpu(), Xn=Xn_cpu, Xaug=Xaug_cpu, rms_scale=scale[0].float().cpu(),
            Q_pre=q[0, head_idx].float().cpu(), K_pre=k[0, kv_idx].float().cpu(),
            Q=Qh.cpu(), K=Kh.cpu(), V=Vh.cpu(), A=A.cpu(), Y=Y.cpu(), cos=cos2.float().cpu(), sin=sin2.float().cpu(),
            rms_rec_err=float(rms_err),
        ))
        del outputs
        if torch.cuda.is_available() and (pi + 1) % 16 == 0:
            torch.cuda.empty_cache()
    meta = {"hidden_size": H, "num_heads": n_heads, "num_kv": n_kv, "head_dim": D, "kv_idx": kv_idx, "rms_eps": eps}
    return out, meta


# ---------------- target construction ----------------

def _slice_bias(module: Any, start: int, end: int) -> torch.Tensor:
    b = getattr(module, "bias", None)
    if b is None:
        return torch.zeros(end - start, dtype=torch.float32)
    return b.detach().float()[start:end].cpu()


def build_weight_slices(model: Any, layer_idx: int, head_idx: int, meta: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    H = int(meta["hidden_size"])
    D = int(meta["head_dim"])
    kv_idx = int(meta["kv_idx"])
    Wq = attn.q_proj.weight.detach().float()[head_idx * D:(head_idx + 1) * D, :].cpu()  # [D,H]
    Wk = attn.k_proj.weight.detach().float()[kv_idx * D:(kv_idx + 1) * D, :].cpu()      # [D,H]
    Wv = attn.v_proj.weight.detach().float()[kv_idx * D:(kv_idx + 1) * D, :].cpu()      # [D,H]
    Wo = attn.o_proj.weight.detach().float()[:, head_idx * D:(head_idx + 1) * D].cpu() # [H,D]
    bq = _slice_bias(attn.q_proj, head_idx * D, (head_idx + 1) * D)
    bk = _slice_bias(attn.k_proj, kv_idx * D, (kv_idx + 1) * D)
    bv = _slice_bias(attn.v_proj, kv_idx * D, (kv_idx + 1) * D)
    # o_proj bias is full attention-output bias, not a per-head contribution. Keep for metadata only.
    bo = getattr(attn.o_proj, "bias", None)
    bo = torch.zeros(H, dtype=torch.float32) if bo is None else bo.detach().float().cpu()
    Wq_aug = torch.cat([Wq, bq[:, None]], dim=1)  # [D,H+1]
    Wk_aug = torch.cat([Wk, bk[:, None]], dim=1)
    Wv_aug = torch.cat([Wv, bv[:, None]], dim=1)
    return {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "bq": bq, "bk": bk, "bv": bv, "bo_full": bo,
            "Wq_aug": Wq_aug, "Wk_aug": Wk_aug, "Wv_aug": Wv_aug}


def build_rope_by_pos(seqs: List[CircuitSeq], max_pos: int) -> Dict[int, torch.Tensor]:
    pos_cos: Dict[int, torch.Tensor] = {}
    pos_sin: Dict[int, torch.Tensor] = {}
    for s in seqs:
        T = int(s.cos.shape[0])
        for p in range(min(T, max_pos + 1)):
            if p not in pos_cos:
                pos_cos[p] = s.cos[p]
                pos_sin[p] = s.sin[p]
    return {p: rope_col_matrix(pos_cos[p], pos_sin[p]).cpu() for p in sorted(pos_cos)}


def qk_delta_matrices_affine(weights: Dict[str, torch.Tensor], Rpos: Dict[int, torch.Tensor], max_delta: int, head_dim: int) -> Dict[int, torch.Tensor]:
    Wq, Wk = weights["Wq_aug"], weights["Wk_aug"]
    out: Dict[int, torch.Tensor] = {}
    if 0 not in Rpos:
        return out
    R0 = Rpos[0]
    for d in range(max_delta + 1):
        if d not in Rpos:
            continue
        Rrel = Rpos[d].T @ R0
        out[d] = (Wq.T @ Rrel @ Wk) / math.sqrt(head_dim)  # [H+1,H+1]
    return out


def qk_delta_matrices_linear(weights: Dict[str, torch.Tensor], Rpos: Dict[int, torch.Tensor], max_delta: int, head_dim: int) -> Dict[int, torch.Tensor]:
    Wq, Wk = weights["Wq"], weights["Wk"]
    out: Dict[int, torch.Tensor] = {}
    if 0 not in Rpos:
        return out
    R0 = Rpos[0]
    for d in range(max_delta + 1):
        if d not in Rpos:
            continue
        Rrel = Rpos[d].T @ R0
        out[d] = (Wq.T @ Rrel @ Wk) / math.sqrt(head_dim)  # [H,H]
    return out


def rope_delta_invariance(seqs: List[CircuitSeq], Rpos: Dict[int, torch.Tensor], max_delta: int, sample_pairs: int = 256) -> Dict[str, Any]:
    rows = []
    rng = random.Random(123)
    for d in range(max_delta + 1):
        mats = []
        for s in seqs:
            T = int(s.cos.shape[0])
            if T <= d:
                continue
            pairs = [(i, i - d) for i in range(d, T)]
            if len(pairs) > sample_pairs:
                pairs = rng.sample(pairs, sample_pairs)
            for i, j in pairs:
                if i in Rpos and j in Rpos:
                    mats.append(Rpos[i].T @ Rpos[j])
        if len(mats) <= 1:
            continue
        base = mats[0]
        errs = [rel_err(m, base) for m in mats[1:]]
        rows.append({"delta": d, "n": len(mats), "mean_rel_to_first": safe_mean(errs), "max_rel_to_first": max(errs) if errs else 0.0})
    if not rows:
        return {"enabled": False, "reason": "no pairs"}
    return {"enabled": True, "max_delta_checked": max(r["delta"] for r in rows), "worst_max_rel": max(float(r["max_rel_to_first"]) for r in rows), "rows": rows[:16]}


# ---------------- metrics and basis ----------------

def svd_rank_table(M: torch.Tensor, ranks: List[int]) -> List[Dict[str, Any]]:
    M = M.detach().float().cpu()
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    except Exception as e:
        return [{"error": str(e)}]
    rows = []
    total_e = float((S * S).sum().item())
    for r in ranks:
        rr = min(int(r), int(S.numel()))
        Mr = (U[:, :rr] * S[:rr]) @ Vh[:rr]
        rows.append({"rank": rr, "rel_err": rel_err(Mr, M), "energy": float((S[:rr] * S[:rr]).sum().item() / max(1e-12, total_e))})
    return rows


def matrix_shape_stats(M: torch.Tensor) -> Dict[str, Any]:
    M = M.detach().float().cpu()
    norm = torch.linalg.norm(M).clamp_min(1e-12)
    is_square = M.ndim == 2 and M.shape[0] == M.shape[1]
    diag = torch.diag(torch.diag(M)) if is_square else torch.zeros_like(M)
    sym_err = rel_err((M + M.T) * 0.5, M) if is_square else None
    skew_err = rel_err((M - M.T) * 0.5, M) if is_square else None
    return {
        "shape": list(M.shape), "norm": float(norm.item()), "mean_abs": float(M.abs().mean().item()), "max_abs": float(M.abs().max().item()),
        "diag_energy_frac": float((torch.linalg.norm(diag) / norm).item()) if is_square else None,
        "sym_rel_to_M": sym_err, "skew_rel_to_M": skew_err,
    }


def _pca_basis(M: torch.Tensor, rank: int) -> torch.Tensor:
    # rows=samples, columns=features; no centering, same reason as v6 QK geometry.
    M = M.detach().float()
    if M.shape[0] == 0:
        return torch.empty(M.shape[1], 0)
    _, _, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, Vh.shape[0])
    return Vh[:r].T.contiguous()


def _project_basis(X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    return (X @ P) @ P.T


def evaluate_qkv_bases(seqs: List[CircuitSeq], weights: Dict[str, torch.Tensor], ranks: List[int], device: str = "cpu") -> List[Dict[str, Any]]:
    """Basis search in the same spirit as v6: choose Pq/Pk/Pv in real head space, then evaluate A/Z/Y."""
    Qall = torch.cat([s.Q for s in seqs], dim=0).float().to(device)
    Kall = torch.cat([s.K for s in seqs], dim=0).float().to(device)
    Vall = torch.cat([s.V for s in seqs], dim=0).float().to(device)
    Wo = weights["Wo"].float().to(device)
    rows: List[Dict[str, Any]] = []
    for r in ranks:
        Pq = _pca_basis(Qall, r)
        Pk = _pca_basis(Kall, r)
        Pv = _pca_basis(Vall, r)
        num_a = den_a = 0.0; kl_sum = 0.0; top_ok = 0; nrows = 0
        num_z = den_z = 0.0; num_y = den_y = 0.0
        num_vbasis_y = den_vbasis_y = 0.0
        for s in seqs:
            Q = s.Q.float().to(device); K = s.K.float().to(device); V = s.V.float().to(device); A = s.A.float().to(device); Y = s.Y.float().to(device)
            D = Q.shape[1]
            Qh = _project_basis(Q, Pq); Kh = _project_basis(K, Pk)
            Ah = causal_softmax((Qh @ Kh.T) / math.sqrt(D))
            da = (Ah - A).float()
            num_a += float((da * da).sum().detach().cpu()); den_a += float((A * A).sum().detach().cpu())
            kl_sum += float(F.kl_div((Ah + 1e-12).log(), A, reduction="sum").detach().cpu())
            top_ok += int((Ah.argmax(dim=-1) == A.argmax(dim=-1)).sum().detach().cpu()); nrows += int(A.shape[0])
            Zt = A @ V; Zh = Ah @ V
            dz = Zh - Zt
            num_z += float((dz * dz).sum().detach().cpu()); den_z += float((Zt * Zt).sum().detach().cpu())
            Yh = Zh @ Wo.T
            dy = Yh - Y
            num_y += float((dy * dy).sum().detach().cpu()); den_y += float((Y * Y).sum().detach().cpu())
            Vp = _project_basis(V, Pv)
            Yv = (A @ Vp) @ Wo.T
            dyv = Yv - Y
            num_vbasis_y += float((dyv * dyv).sum().detach().cpu()); den_vbasis_y += float((Y * Y).sum().detach().cpu())
        rows.append({
            "basis_kind": "pca_no_center_functional", "rank": int(min(r, Qall.shape[1])),
            "A_rel": math.sqrt(num_a / max(1e-12, den_a)), "KL": kl_sum / max(1, nrows), "top1": top_ok / max(1, nrows),
            "Z_rel": math.sqrt(num_z / max(1e-12, den_z)), "Y_rel_from_QK_basis": math.sqrt(num_y / max(1e-12, den_y)),
            "Y_rel_from_V_basis_true_A": math.sqrt(num_vbasis_y / max(1e-12, den_vbasis_y)),
        })
    return rows


def _pack_qk_batches(seqs: List[CircuitSeq], device: str, batch_size: int = 16) -> List[Dict[str, torch.Tensor]]:
    out = []
    for off in range(0, len(seqs), batch_size):
        chunk = seqs[off:off + batch_size]
        B = len(chunk); Tm = max(s.Q.shape[0] for s in chunk); D = chunk[0].Q.shape[1]
        Q = torch.zeros(B, Tm, D, device=device); K = torch.zeros_like(Q); A = torch.zeros(B, Tm, Tm, device=device)
        mask = torch.zeros(B, Tm, Tm, dtype=torch.bool, device=device); row_sel = torch.zeros(B, Tm, dtype=torch.bool, device=device)
        for b, s in enumerate(chunk):
            T = s.Q.shape[0]
            Q[b, :T] = s.Q.to(device); K[b, :T] = s.K.to(device); A[b, :T, :T] = s.A.to(device)
            row_sel[b, :T] = True
            ar = torch.arange(Tm, device=device)
            mask[b] = (ar[None, :] < T) & (ar[:, None] < T) & (ar[:, None] >= ar[None, :])
        out.append({"Q": Q, "K": K, "A": A, "mask": mask, "row_sel": row_sel})
    return out


def evaluate_learned_qk_basis(seqs: List[CircuitSeq], weights: Dict[str, torch.Tensor], ranks: List[int], steps: int, lr: float, batch_size: int, device: str) -> List[Dict[str, Any]]:
    """Small learned Pq/Pk score-basis search, same idea as v6 but self-contained and no patching."""
    rows: List[Dict[str, Any]] = []
    if steps <= 0:
        return rows
    batches = _pack_qk_batches(seqs, device, batch_size=batch_size)
    Wo = weights["Wo"].to(device).float()
    Qall = torch.cat([s.Q for s in seqs], dim=0).float().to(device)
    Kall = torch.cat([s.K for s in seqs], dim=0).float().to(device)
    for r in ranks:
        Pq0 = _pca_basis(Qall, r).to(device); Pk0 = _pca_basis(Kall, r).to(device)
        Pq = torch.nn.Parameter(Pq0.clone()); Pk = torch.nn.Parameter(Pk0.clone())
        opt = torch.optim.AdamW([Pq, Pk], lr=lr, weight_decay=1e-4)
        denom = math.sqrt(int(Pq.shape[1]))
        for _ in range(int(steps)):
            loss_sum = 0.0; n = 0
            for bd in batches:
                Ql = bd["Q"] @ Pq; Kl = bd["K"] @ Pk
                scores = torch.bmm(Ql, Kl.transpose(1, 2)) / denom
                Ah = torch.softmax(scores.masked_fill(~bd["mask"], -1e9), dim=-1)
                sel = bd["row_sel"]
                loss_b = F.kl_div((Ah[sel] + 1e-12).log(), bd["A"][sel], reduction="sum")
                loss_sum = loss_sum + loss_b; n += int(sel.sum())
            loss = loss_sum / max(1, n)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([Pq, Pk], 1.0); opt.step()
        # eval
        num_a = den_a = 0.0; kl_sum = 0.0; top_ok = 0; nrows = 0; num_z = den_z = 0.0; num_y = den_y = 0.0
        with torch.no_grad():
            for s in seqs:
                Q=s.Q.to(device); K=s.K.to(device); V=s.V.to(device); A=s.A.to(device); Y=s.Y.to(device)
                Ah = causal_softmax(((Q @ Pq) @ (K @ Pk).T) / denom)
                da=Ah-A; num_a += float((da*da).sum().cpu()); den_a += float((A*A).sum().cpu())
                kl_sum += float(F.kl_div((Ah+1e-12).log(), A, reduction="sum").cpu())
                top_ok += int((Ah.argmax(-1)==A.argmax(-1)).sum().cpu()); nrows += int(A.shape[0])
                Zt=A@V; Zh=Ah@V; dz=Zh-Zt; num_z += float((dz*dz).sum().cpu()); den_z += float((Zt*Zt).sum().cpu())
                Yh=Zh@Wo.T; dy=Yh-Y; num_y += float((dy*dy).sum().cpu()); den_y += float((Y*Y).sum().cpu())
        rows.append({"basis_kind":"learned_score_qk", "rank":int(Pq.shape[1]), "steps":int(steps),
                     "A_rel":math.sqrt(num_a/max(1e-12,den_a)), "KL":kl_sum/max(1,nrows), "top1":top_ok/max(1,nrows),
                     "Z_rel":math.sqrt(num_z/max(1e-12,den_z)), "Y_rel_from_QK_basis":math.sqrt(num_y/max(1e-12,den_y))})
    return rows


# ---------------- verification ----------------

def verify_targets(seqs: List[CircuitSeq], weights: Dict[str, torch.Tensor], Mdelta_aug: Dict[int, torch.Tensor], Mdelta_lin: Dict[int, torch.Tensor], max_delta: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    Wq, Wk, Wv, Wo = weights["Wq"], weights["Wk"], weights["Wv"], weights["Wo"]
    bq, bk, bv = weights["bq"], weights["bk"], weights["bv"]
    Wq_aug, Wk_aug, Wv_aug = weights["Wq_aug"], weights["Wk_aug"], weights["Wv_aug"]
    Cvo_lin = Wo @ Wv          # [H,H]
    Cvo_aug = Wo @ Wv_aug      # [H,H+1]
    b_vo = Wo @ bv             # [H]
    rows: List[Dict[str, Any]] = []
    acc = {k: 0.0 for k in ["score_aug_num","score_aug_den","score_lin_num","score_lin_den","A_num","A_den","Qpre_num","Qpre_den","Kpre_num","Kpre_den","V_num","V_den","Y_aug_num","Y_aug_den","Y_raw_num","Y_raw_den"]}
    for s in seqs:
        T = int(s.Xn.shape[0]); D = int(s.Q.shape[-1])
        scores_true = (s.Q @ s.K.T) / math.sqrt(D)
        scores_aug = torch.zeros_like(scores_true); scores_lin = torch.zeros_like(scores_true)
        mask_eval = torch.zeros_like(scores_true, dtype=torch.bool)
        for i in range(T):
            for j in range(i + 1):
                d = i - j
                if d <= max_delta and d in Mdelta_aug:
                    scores_aug[i, j] = s.Xaug[i] @ Mdelta_aug[d] @ s.Xaug[j]
                    scores_lin[i, j] = s.Xn[i] @ Mdelta_lin[d] @ s.Xn[j] if d in Mdelta_lin else 0.0
                    mask_eval[i, j] = True
        if int(mask_eval.sum()) > 0:
            diff_aug = (scores_aug[mask_eval] - scores_true[mask_eval]).float()
            diff_lin = (scores_lin[mask_eval] - scores_true[mask_eval]).float()
            den = float(scores_true[mask_eval].float().pow(2).sum().item())
            acc["score_aug_num"] += float((diff_aug * diff_aug).sum().item()); acc["score_aug_den"] += den
            acc["score_lin_num"] += float((diff_lin * diff_lin).sum().item()); acc["score_lin_den"] += den
        Ahat = causal_softmax(scores_aug.masked_fill(~mask_eval, torch.finfo(scores_aug.dtype).min))
        row_covered = torch.zeros((T,), dtype=torch.bool)
        for i in range(T):
            row_covered[i] = all(((i - j) <= max_delta and (i - j) in Mdelta_aug) for j in range(i + 1))
        if int(row_covered.sum()) > 0:
            da = (Ahat[row_covered] - s.A[row_covered]).float()
            acc["A_num"] += float((da * da).sum().item()); acc["A_den"] += float((s.A[row_covered].float().pow(2)).sum().item())
        Qpre_hat = s.Xaug @ Wq_aug.T
        Kpre_hat = s.Xaug @ Wk_aug.T
        V_hat = s.Xaug @ Wv_aug.T
        Y_aug = s.A @ (s.Xaug @ Cvo_aug.T)
        # raw dynamic block: linear normalized part plus bias per row. Since sum_j Aij=1, bias is added once.
        payload_raw = (s.Xraw * s.rms_scale) @ Cvo_lin.T + b_vo.view(1, -1)
        Y_raw = s.A @ payload_raw
        for name, pred, true in [("Qpre", Qpre_hat, s.Q_pre), ("Kpre", Kpre_hat, s.K_pre), ("V", V_hat, s.V), ("Y_aug", Y_aug, s.Y), ("Y_raw", Y_raw, s.Y)]:
            diff = (pred - true).float(); acc[f"{name}_num"] += float((diff*diff).sum().item()); acc[f"{name}_den"] += float((true.float()*true.float()).sum().item())
        rows.append({
            "prompt_id": s.prompt_id, "suite": s.suite, "T": T, "text": s.text[:120],
            "rms_rec_err": s.rms_rec_err,
            "qk_score_aug_rel_covered": math.sqrt(float((scores_aug[mask_eval]-scores_true[mask_eval]).pow(2).sum().item()) / max(1e-12, float(scores_true[mask_eval].pow(2).sum().item()))) if int(mask_eval.sum()) else None,
            "qk_score_linear_no_bias_rel_covered": math.sqrt(float((scores_lin[mask_eval]-scores_true[mask_eval]).pow(2).sum().item()) / max(1e-12, float(scores_true[mask_eval].pow(2).sum().item()))) if int(mask_eval.sum()) else None,
            "qk_rows_fully_covered": int(row_covered.sum().item()),
            "A_rel_fully_covered": rel_err(Ahat[row_covered], s.A[row_covered]) if int(row_covered.sum()) else None,
            "Qpre_from_affine_rel": rel_err(Qpre_hat, s.Q_pre),
            "Kpre_from_affine_rel": rel_err(Kpre_hat, s.K_pre),
            "V_from_affine_Wv_rel": rel_err(V_hat, s.V),
            "Y_from_Cvo_aug_rel": rel_err(Y_aug, s.Y),
            "Y_block_raw_affine_rel": rel_err(Y_raw, s.Y),
            "A_entropy_mean": float((-(s.A * (s.A + 1e-12).log()).sum(dim=-1)).mean().item()),
            "Y_norm_mean": float(torch.linalg.norm(s.Y, dim=-1).mean().item()),
        })
    summary = {
        "qk_score_aug_rel_covered_normX": math.sqrt(acc["score_aug_num"] / max(1e-12, acc["score_aug_den"])),
        "qk_score_linear_no_bias_rel_covered_normX": math.sqrt(acc["score_lin_num"] / max(1e-12, acc["score_lin_den"])),
        "A_rel_rows_fully_covered_from_aug_scores": math.sqrt(acc["A_num"] / max(1e-12, acc["A_den"])) if acc["A_den"] > 0 else None,
        "Qpre_from_affine_rel": math.sqrt(acc["Qpre_num"] / max(1e-12, acc["Qpre_den"])),
        "Kpre_from_affine_rel": math.sqrt(acc["Kpre_num"] / max(1e-12, acc["Kpre_den"])),
        "V_from_affine_Wv_rel": math.sqrt(acc["V_num"] / max(1e-12, acc["V_den"])),
        "Y_from_Cvo_aug_rel": math.sqrt(acc["Y_aug_num"] / max(1e-12, acc["Y_aug_den"])),
        "Y_block_raw_affine_rel": math.sqrt(acc["Y_raw_num"] / max(1e-12, acc["Y_raw_den"])),
    }
    return summary, rows


# ---------------- optional generic channel baseline ----------------

def make_light_ops(D: int, dtype=torch.float32) -> Tuple[List[str], torch.Tensor]:
    ops: List[torch.Tensor] = []; names: List[str] = []
    I = torch.eye(D, dtype=dtype); ops.append(I); names.append("Identity")
    ops.append(torch.diag(torch.linspace(-1, 1, D, dtype=dtype))); names.append("RampDiag")
    ops.append(torch.ones(D, D, dtype=dtype) / D); names.append("MeanProject")
    for sh in [1, 2, 4, 8]:
        M = torch.zeros(D, D, dtype=dtype)
        for i in range(D):
            j = i - sh
            if 0 <= j < D: M[i, j] = 1.0
        ops.append(M); names.append(f"ShiftRight{sh}")
        ops.append(M.T); names.append(f"ShiftLeft{sh}")
    for block in [2, 4, 8, 16, 32, 64]:
        if block >= D: continue
        M = torch.zeros(D, D, dtype=dtype)
        for start in range(0, D, block):
            end = min(D, start + block); M[start:end, start:end] = 1.0 / max(1, end - start)
        ops.append(M); names.append(f"BlockAvg{block}")
    A = torch.stack([m.reshape(-1) / torch.linalg.norm(m.reshape(-1)).clamp_min(1e-12) for m in ops], dim=1)
    return names, A


def fit_light_ops(M: torch.Tensor, ridge: float = 1e-4, max_terms: int = 8) -> Dict[str, Any]:
    M = M.detach().float().cpu()
    if M.shape[0] != M.shape[1]:
        return {"enabled": False, "reason": "non-square matrix"}
    D = int(M.shape[0]); names, A = make_light_ops(D)
    b = M.reshape(-1); bnorm = torch.linalg.norm(b).clamp_min(1e-12)
    c = torch.linalg.solve(A.T @ A + ridge * torch.eye(A.shape[1]), A.T @ b)
    if 0 < max_terms < c.numel():
        keep = torch.topk(c.abs(), k=max_terms).indices; A2 = A[:, keep]
        c2 = torch.linalg.solve(A2.T @ A2 + ridge * torch.eye(A2.shape[1]), A2.T @ b)
        cc = torch.zeros_like(c); cc[keep] = c2; c = cc
    rec = A @ c
    terms = []
    for i in torch.argsort(c.abs(), descending=True).tolist()[:max_terms]:
        if abs(float(c[i])) > 1e-8:
            terms.append({"name": names[i], "coef_normalized_atom": float(c[i])})
    return {"enabled": True, "rel_err": float(torch.linalg.norm(rec - b) / bnorm), "terms": terms, "bank_size": len(names)}


# ---------------- main analysis ----------------

def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    base_mod = load_base_module(args.base_script) if args.base_script else None
    set_seed(args.seed)
    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    print("loading model...", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required; use --self-test for synthetic check") from e

    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(args.device)
    model.eval()

    cfg = model.config
    head_specs = parse_heads(args.heads, int(cfg.num_hidden_layers), int(cfg.num_attention_heads))
    prompts = build_prompts(base_mod, args.prompt_suites, args.prompts_per_suite, args.same_text_repeats)
    write_json(out_dir / "prompts.json", prompts)
    print(f"heads={head_specs} prompts={len(prompts)}", flush=True)

    all_summaries = []
    for layer_idx, head_idx in head_specs:
        hdir = out_dir / f"L{layer_idx}H{head_idx}"; ensure_dir(hdir)
        print(f"\n=== affine circuit targets L{layer_idx}H{head_idx} ===", flush=True)
        t0 = time.perf_counter()
        seqs, meta = collect_circuit_data(model, tok, prompts, layer_idx, head_idx, args.max_length, args.device)
        if not seqs:
            raise RuntimeError("no sequences collected")
        print(f"collected {len(seqs)} seqs in {time.perf_counter()-t0:.1f}s", flush=True)
        weights = build_weight_slices(model, layer_idx, head_idx, meta)
        max_observed_T = max(int(s.Xn.shape[0]) for s in seqs)
        max_delta = min(int(args.max_delta), max_observed_T - 1)
        Rpos = build_rope_by_pos(seqs, max_delta)
        Mdelta_aug = qk_delta_matrices_affine(weights, Rpos, max_delta, int(meta["head_dim"]))
        Mdelta_lin = qk_delta_matrices_linear(weights, Rpos, max_delta, int(meta["head_dim"]))
        verify, per_prompt = verify_targets(seqs, weights, Mdelta_aug, Mdelta_lin, max_delta)
        rope_inv = rope_delta_invariance(seqs, Rpos, max_delta=min(max_delta, 16))

        Cvo_lin = weights["Wo"] @ weights["Wv"]
        Cvo_aug = weights["Wo"] @ weights["Wv_aug"]
        ranks = parse_int_list(args.svd_ranks)
        mat_reports: Dict[str, Any] = {
            "C_vo_linear_HxH": {"stats": matrix_shape_stats(Cvo_lin), "svd": svd_rank_table(Cvo_lin, ranks)},
            "C_vo_aug_HxHplus1": {"stats": matrix_shape_stats(Cvo_aug), "svd": svd_rank_table(Cvo_aug, ranks)},
        }
        for d in sorted(Mdelta_aug.keys())[: min(len(Mdelta_aug), int(args.report_deltas))]:
            mat_reports[f"M_qk_aug_delta_{d}"] = {"stats": matrix_shape_stats(Mdelta_aug[d]), "svd": svd_rank_table(Mdelta_aug[d], ranks)}
            mat_reports[f"M_qk_linear_no_bias_delta_{d}"] = {"stats": matrix_shape_stats(Mdelta_lin[d]), "svd": svd_rank_table(Mdelta_lin[d], ranks)}
        if args.decode_generic_ops:
            mat_reports["C_vo_linear_HxH"]["generic_ops_fit"] = fit_light_ops(Cvo_lin, ridge=args.generic_ridge, max_terms=args.generic_max_terms)
            for d in sorted(Mdelta_lin.keys())[: min(len(Mdelta_lin), int(args.report_deltas))]:
                mat_reports[f"M_qk_linear_no_bias_delta_{d}"]["generic_ops_fit"] = fit_light_ops(Mdelta_lin[d], ridge=args.generic_ridge, max_terms=args.generic_max_terms)

        basis_ranks = parse_int_list(args.basis_ranks or args.svd_ranks)
        basis_rows = evaluate_qkv_bases(seqs, weights, basis_ranks, device=args.basis_device)
        if args.fit_learned_qk_basis:
            basis_rows += evaluate_learned_qk_basis(seqs, weights, basis_ranks, args.learned_steps, args.learned_lr, args.basis_batch_size, args.device)
        write_csv(hdir / "basis_functional.csv", basis_rows)

        exact_program = {
            "RMS": "Xn_t = D_t x_raw_t, D_t = diag(gamma / sqrt(mean(x_raw_t^2)+eps))",
            "QK_affine_normalized": "score_ij = Xaug_i^T M_qk_aug[i-j] Xaug_j, Xaug=[Xn,1]",
            "M_qk_aug_delta": "M_qk_aug[d] = Wq_aug^T @ (R_i^T @ R_j) @ Wk_aug / sqrt(head_dim), d=i-j",
            "VO_affine_normalized": "Y_i = sum_j A[i,j] * Xaug_j @ C_vo_aug.T",
            "C_vo_aug": "C_vo_aug = Wo_head @ [Wv_kv | bv_kv]",
            "FullHead_raw_block_sparse": "payload_j = C_vo_linear @ D_j @ x_raw_j + b_vo; y_i=sum_j Aij*payload_j",
            "Basis_search": "Pq/Pk/Pv are searched in real Q/K/V head-space and judged by A/Z/Y errors, same spirit as v6 QK basis search.",
        }
        summary = {
            "layer": layer_idx, "head": head_idx, "meta": meta, "nseq": len(seqs), "max_observed_T": max_observed_T,
            "max_delta": max_delta, "bias_norms": {"bq": float(weights["bq"].norm()), "bk": float(weights["bk"].norm()), "bv": float(weights["bv"].norm())},
            "verify": verify, "rope_delta_invariance": rope_inv, "exact_program": exact_program,
            "basis_best": sorted(basis_rows, key=lambda r: (r.get("Y_rel_from_QK_basis", 999), r.get("A_rel", 999)))[:5],
            "matrix_reports": mat_reports,
        }
        all_summaries.append(summary)
        write_json(hdir / "summary.json", summary)
        write_csv(hdir / "per_prompt_checks.csv", per_prompt)
        examples = []
        for s in seqs[: int(args.example_prompts)]:
            top_rows = []
            T = int(s.A.shape[0])
            for i in range(min(T, int(args.example_tokens))):
                vals, idx = torch.topk(s.A[i, : i + 1], k=min(5, i + 1))
                top_rows.append({
                    "pos": i, "token": s.tokens[i] if i < len(s.tokens) else "?",
                    "top_read": [{"pos": int(j), "token": s.tokens[int(j)] if int(j) < len(s.tokens) else "?", "p": float(v)} for v, j in zip(vals.tolist(), idx.tolist())],
                    "Y_norm": float(torch.linalg.norm(s.Y[i]).item()), "rms_scale_mean": float(s.rms_scale[i].mean().item()),
                })
            examples.append({"prompt_id": s.prompt_id, "suite": s.suite, "text": s.text, "rows": top_rows})
        write_json(hdir / "token_flow_examples.json", examples)
        if args.save_tensors:
            torch.save({
                "layer": layer_idx, "head": head_idx, "meta": meta,
                "weights": weights, "C_vo_linear": Cvo_lin, "C_vo_aug": Cvo_aug,
                "M_qk_aug_delta": Mdelta_aug, "M_qk_linear_no_bias_delta": Mdelta_lin,
                "note": "Main exact targets are affine: score=x_aug M_aug x_aug; payload=x_aug C_vo_aug.T.",
            }, hdir / "circuit_targets_v2.pt")
        md = []
        md.append(f"# Affine circuit matrix targets L{layer_idx}H{head_idx}\n\n")
        md.append("## Exact matrix program\n```python\n")
        for k, v in exact_program.items(): md.append(f"{k}: {v}\n")
        md.append("```\n\n## Verification\n```json\n" + json.dumps(json_sanitize(verify), indent=2, ensure_ascii=False) + "\n```\n")
        md.append("\n## Read this\n")
        md.append("- `qk_score_linear_no_bias_rel` is the old v1-style target; it should be worse if q/k bias matters.\n")
        md.append("- `qk_score_aug_rel`, `V_from_affine_Wv_rel`, `Y_from_Cvo_aug_rel`, `Y_block_raw_affine_rel` should be near zero when the circuit formula is correct.\n")
        md.append("- `basis_functional.csv` is not generic PCA interpretation; it is functional score/read/write basis validation like your v6 QK basis search.\n")
        (hdir / "summary.md").write_text("".join(md), encoding="utf-8")

    top_summary = {"args": vars(args), "heads": all_summaries}
    write_json(out_dir / "summary.json", top_summary)
    print(f"\nDONE wrote {out_dir}", flush=True)
    return top_summary


# ---------------- self-test ----------------

def self_test() -> None:
    torch.manual_seed(0)
    H, D, T = 16, 8, 7
    Wq = torch.randn(D, H) / math.sqrt(H); Wk = torch.randn(D, H) / math.sqrt(H); Wv = torch.randn(D, H) / math.sqrt(H); Wo = torch.randn(H, D) / math.sqrt(D)
    bq = torch.randn(D) * 0.2; bk = torch.randn(D) * 0.2; bv = torch.randn(D) * 0.2
    Wq_aug = torch.cat([Wq, bq[:, None]], dim=1); Wk_aug = torch.cat([Wk, bk[:, None]], dim=1); Wv_aug = torch.cat([Wv, bv[:, None]], dim=1)
    gamma = torch.rand(H) + 0.5; Xraw = torch.randn(T, H); eps = 1e-6
    scale = gamma.view(1, H) / torch.sqrt(Xraw.pow(2).mean(dim=-1, keepdim=True) + eps)
    Xn = Xraw * scale; Xaug = torch.cat([Xn, torch.ones(T, 1)], dim=1)
    freq = torch.linspace(0.01, 0.2, D // 2)
    angles_h = torch.arange(T).float().view(T, 1) * freq.view(1, D // 2)
    cos = torch.cat([torch.cos(angles_h), torch.cos(angles_h)], dim=1)
    sin = torch.cat([torch.sin(angles_h), torch.sin(angles_h)], dim=1)
    Rpos = {i: rope_col_matrix(cos[i], sin[i]) for i in range(T)}
    Qpre = Xaug @ Wq_aug.T; Kpre = Xaug @ Wk_aug.T; V = Xaug @ Wv_aug.T
    Q = torch.stack([Qpre[i] @ Rpos[i].T for i in range(T)], dim=0)
    K = torch.stack([Kpre[i] @ Rpos[i].T for i in range(T)], dim=0)
    A = causal_softmax((Q @ K.T) / math.sqrt(D)); Y = (A @ V) @ Wo.T
    weights = {"Wq":Wq,"Wk":Wk,"Wv":Wv,"Wo":Wo,"bq":bq,"bk":bk,"bv":bv,"Wq_aug":Wq_aug,"Wk_aug":Wk_aug,"Wv_aug":Wv_aug}
    Mdelta = qk_delta_matrices_affine(weights, Rpos, T-1, D)
    Mlin = qk_delta_matrices_linear(weights, Rpos, T-1, D)
    scores_hat = torch.zeros(T,T); scores_lin=torch.zeros(T,T)
    for i in range(T):
        for j in range(i+1):
            scores_hat[i,j] = Xaug[i] @ Mdelta[i-j] @ Xaug[j]
            scores_lin[i,j] = Xn[i] @ Mlin[i-j] @ Xn[j]
    scores_true=(Q@K.T)/math.sqrt(D); mask=torch.tril(torch.ones(T,T,dtype=torch.bool))
    Cvo_aug=Wo@Wv_aug; Cvo_lin=Wo@Wv; bvo=Wo@bv
    Y2=A@(Xaug@Cvo_aug.T); Y3=A@((Xraw*scale)@Cvo_lin.T + bvo.view(1,-1))
    print("SELF TEST v2 affine")
    print(f"  qk_score_aug_err={rel_err(scores_hat[mask], scores_true[mask]):.3e}")
    print(f"  qk_score_linear_no_bias_err={rel_err(scores_lin[mask], scores_true[mask]):.3e}  # expected nonzero")
    print(f"  vo_Y_aug_err={rel_err(Y2,Y):.3e}")
    print(f"  raw_block_affine_Y_err={rel_err(Y3,Y):.3e}")
    assert rel_err(scores_hat[mask], scores_true[mask]) < 1e-5
    assert rel_err(Y2, Y) < 1e-5
    assert rel_err(Y3, Y) < 1e-5
    print("  OK")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base-script", default="./qwen_program_decompiler_v6_scorehybrid.py")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--heads", default="2:1")
    ap.add_argument("--prompt-suites", default="all")
    ap.add_argument("--prompts-per-suite", type=int, default=8)
    ap.add_argument("--same-text-repeats", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--max-delta", type=int, default=64)
    ap.add_argument("--svd-ranks", default="4,8,16,32,64,128")
    ap.add_argument("--basis-ranks", default="4,8,16,32,64")
    ap.add_argument("--basis-device", default="cpu", help="cpu is safer; cuda is faster for basis eval")
    ap.add_argument("--fit-learned-qk-basis", action="store_true")
    ap.add_argument("--learned-steps", type=int, default=120)
    ap.add_argument("--learned-lr", type=float, default=0.002)
    ap.add_argument("--basis-batch-size", type=int, default=16)
    ap.add_argument("--report-deltas", type=int, default=8)
    ap.add_argument("--example-prompts", type=int, default=2)
    ap.add_argument("--example-tokens", type=int, default=16)
    ap.add_argument("--decode-generic-ops", action="store_true")
    ap.add_argument("--generic-ridge", type=float, default=1e-4)
    ap.add_argument("--generic-max-terms", type=int, default=8)
    ap.add_argument("--save-tensors", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./qwen_circuit_targets_v2")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(); return
    run_analysis(args)


if __name__ == "__main__":
    main()
