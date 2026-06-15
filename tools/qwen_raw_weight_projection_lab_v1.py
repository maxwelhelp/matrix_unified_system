#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_raw_weight_projection_lab_v1.py

NON-STANDARD RAW-WEIGHT MATRIX LAB for Qwen attention blocks.

Goal
----
Do NOT inspect attention maps or activations. Do NOT do standard QK/OV only.
Instead scan the grey/raw attention weights through multiple matrix-program views:

1) raw individual projections:
   Wq_head, Wk_kv, Wv_kv, Wo_head

2) composite circuit weights:
   QK_no_rope       = Wq.T @ Wk / sqrt(d)
   QK_aug_delta     = Wq_aug.T @ RopeRelative(delta) @ Wk_aug / sqrt(d)
   VO               = Wo @ Wv
   VO_aug           = Wo @ [Wv | bv]

3) affine decomposition of QK_aug_delta:
   score_ij = x_i.T B_delta x_j + x_i.T u_delta + v_delta.T x_j + c_delta
   Report content/query/key/constant energy fractions.

4) hidden-space block projections:
   Split hidden dimension into head-sized channel blocks. Report which input/output blocks dominate.
   This is a raw-weight question: does a head write/read from specific channel blocks?

5) cross-head / cross-layer transition probes:
   A current head writes via VO. Does the next layer/head read this written subspace through Wq/Wk/Wv?
   Report subspace overlap and norm coupling.

6) basis/family probes that are still raw-weight-only:
   For each head, over deltas: fit compact bases for c_delta, u_delta, v_delta families.
   Across heads: compare subspace overlaps and transition lines.

This is meant to complement qwen_circuit_matrix_targets_v2_affine_basis.py:
- v2 gives exact per-head circuit reconstruction on activations/prompts.
- this file asks: what can be found in the grey weights alone, and which composite/projection views make them readable?

Example:
  python qwen_raw_weight_projection_lab_v1.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda --dtype fp16 \
    --layers 0,1,2,3 --heads all \
    --deltas 0,1,2,3,4,5,6,7,8,16,32 \
    --ranks 4,8,16,32,64 \
    --subspace-rank 16 \
    --top-transitions 80 \
    --out-dir ./qwen_raw_weight_projection_lab_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_int_list(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in re.split(r"[,; ]+", s) if x.strip()]


def parse_layers(s: str, n_layers: int) -> List[int]:
    s = str(s).strip().lower()
    if s == "all":
        return list(range(n_layers))
    out: List[int] = []
    for part in re.split(r"[,; ]+", s):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set([x for x in out if 0 <= x < n_layers]))


def parse_heads(s: str, n_heads: int) -> List[int]:
    s = str(s).strip().lower()
    if s == "all":
        return list(range(n_heads))
    return sorted(set([x for x in parse_int_list(s) if 0 <= x < n_heads]))


def get_dtype(name: str):
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"bad dtype {name}")


def rel_err(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> float:
    A = A.float(); B = B.float()
    return float(torch.linalg.norm(A - B) / torch.linalg.norm(B).clamp_min(eps))


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def json_sanitize(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return str(x)
        return x
    if isinstance(x, Path):
        return str(x)
    if torch.is_tensor(x):
        if x.numel() <= 32:
            return x.detach().cpu().tolist()
        return {"tensor": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_sanitize(v) for v in x]
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(json_sanitize(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def get_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find model layers")


def svd_energy_report(M: torch.Tensor, ranks: Sequence[int]) -> Dict[str, Any]:
    X = M.detach().float().cpu()
    if X.numel() == 0:
        return {}
    try:
        s = torch.linalg.svdvals(X)
    except Exception as e:
        return {"svd_failed": str(e)}
    e = s.square()
    total = float(e.sum().clamp_min(1e-12))
    out: Dict[str, Any] = {
        "shape": list(X.shape),
        "fro_norm": float(torch.linalg.norm(X)),
        "rank_numeric_1e-6": int((s > s.max().clamp_min(1e-12) * 1e-6).sum().item()),
        "top_singular": s[:8].tolist(),
    }
    for r in ranks:
        rr = min(int(r), int(s.numel()))
        resid = math.sqrt(max(0.0, float(e[rr:].sum()) / total))
        out[f"rank{r}_energy"] = float(e[:rr].sum() / total)
        out[f"rank{r}_rel_err"] = resid
    return out


def top_singular_basis(M: torch.Tensor, k: int, side: str = "left") -> torch.Tensor:
    # left basis = output directions U; right basis = input directions V
    X = M.detach().float().cpu()
    if min(X.shape) == 0:
        return torch.zeros(X.shape[0 if side == "left" else 1], 0)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    kk = min(k, U.shape[1])
    return U[:, :kk].contiguous() if side == "left" else Vh[:kk].T.contiguous()


def subspace_overlap(A: torch.Tensor, B: torch.Tensor, k: Optional[int] = None) -> float:
    # A/B have columns as orthonormal-ish bases [D,k]
    A = A.float(); B = B.float()
    if A.numel() == 0 or B.numel() == 0:
        return 0.0
    # QR for safety
    Qa = torch.linalg.qr(A, mode="reduced").Q
    Qb = torch.linalg.qr(B, mode="reduced").Q
    kk = min(Qa.shape[1], Qb.shape[1]) if k is None else min(k, Qa.shape[1], Qb.shape[1])
    if kk <= 0:
        return 0.0
    val = torch.linalg.norm(Qa[:, :kk].T @ Qb[:, :kk], ord="fro").square() / kk
    return float(val.clamp(0, 1))


def norm_coupling(reader: torch.Tensor, writer: torch.Tensor, H: int) -> float:
    # reader: d x H, writer: H x H. Normalize so random-ish values are comparable.
    R = reader.float().cpu(); W = writer.float().cpu()
    denom = torch.linalg.norm(R) * torch.linalg.norm(W) / math.sqrt(max(1, H))
    if float(denom) == 0.0:
        return 0.0
    return float(torch.linalg.norm(R @ W) / denom.clamp_min(1e-12))


def block_energy(M: torch.Tensor, block: int, topk: int = 12) -> Dict[str, Any]:
    X = M.detach().float().cpu()
    H0, H1 = X.shape[:2]
    nr = H0 // block
    nc = H1 // block
    if nr == 0 or nc == 0:
        return {"available": False}
    E = torch.zeros(nr, nc)
    total = float((X[:nr*block, :nc*block].square()).sum().clamp_min(1e-12))
    for i in range(nr):
        for j in range(nc):
            b = X[i*block:(i+1)*block, j*block:(j+1)*block]
            E[i, j] = b.square().sum() / total
    flat = []
    for i in range(nr):
        for j in range(nc):
            flat.append((float(E[i, j]), i, j))
    flat.sort(reverse=True)
    diag = sum(float(E[i, i]) for i in range(min(nr, nc)))
    return {
        "available": True,
        "block_size": block,
        "n_row_blocks": nr,
        "n_col_blocks": nc,
        "diag_block_energy_frac": diag,
        "top_blocks": [{"frac": f, "out_block": i, "in_block": j} for f, i, j in flat[:topk]],
    }


def rope_relative_matrix(head_dim: int, delta: int, rope_theta: float = 1000000.0, device="cpu") -> torch.Tensor:
    """Approximate Qwen rotate_half relative matrix under split-half convention.

    apply_rope(x,pos) = x*cos + rotate_half(x)*sin
    with cos/sin duplicated across halves. R_i.T R_j = R(delta).
    We build R(delta) directly on basis vectors.
    """
    d = int(head_dim)
    half = d // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    # Qwen cos/sin are duplicated across halves in many HF implementations.
    ang = float(delta) * inv_freq
    co = torch.cos(ang)
    si = torch.sin(ang)
    R = torch.zeros(d, d, dtype=torch.float32, device=device)
    # for pair (a, a+half): rotation [[cos, -sin],[sin,cos]] depending convention.
    # We only need a consistent relative operator matching apply_rope split-half.
    for a in range(half):
        b = a + half
        R[a, a] = co[a]
        R[b, b] = co[a]
        R[a, b] = -si[a]
        R[b, a] = si[a]
    return R


def affine_qk_aug(Wq: torch.Tensor, Wk: torch.Tensor, bq: Optional[torch.Tensor], bk: Optional[torch.Tensor],
                   Rrel: torch.Tensor, denom: float) -> torch.Tensor:
    # Wq/Wk [d,H], b [d]; W_aug [d,H+1]
    d, H = Wq.shape
    zq = torch.zeros(d, 1, dtype=Wq.dtype, device=Wq.device) if bq is None else bq.view(d, 1).to(Wq)
    zk = torch.zeros(d, 1, dtype=Wk.dtype, device=Wk.device) if bk is None else bk.view(d, 1).to(Wk)
    Wqa = torch.cat([Wq, zq], dim=1).float()
    Wka = torch.cat([Wk, zk], dim=1).float()
    return (Wqa.T @ Rrel.to(Wqa.device).float() @ Wka) / denom


def affine_vo_aug(Wv: torch.Tensor, bv: Optional[torch.Tensor], Wo: torch.Tensor) -> torch.Tensor:
    # Wv [d,H], bv [d], Wo [H,d] -> [H,H+1]
    d, H = Wv.shape
    z = torch.zeros(d, 1, dtype=Wv.dtype, device=Wv.device) if bv is None else bv.view(d, 1).to(Wv)
    Wva = torch.cat([Wv, z], dim=1).float()
    return Wo.float() @ Wva


def qk_affine_parts(Maug: torch.Tensor) -> Dict[str, Any]:
    X = Maug.detach().float().cpu()
    H1 = X.shape[0]
    H = H1 - 1
    B = X[:H, :H]
    u = X[:H, H]
    v = X[H, :H]
    c = X[H, H]
    eB = float(B.square().sum())
    eu = float(u.square().sum())
    ev = float(v.square().sum())
    ec = float(c.square())
    total = max(1e-12, eB + eu + ev + ec)
    return {
        "content_bilinear_energy_frac": eB / total,
        "query_affine_energy_frac": eu / total,
        "key_affine_energy_frac": ev / total,
        "constant_energy_frac": ec / total,
        "constant_value": float(c),
        "u_norm": math.sqrt(eu),
        "v_norm": math.sqrt(ev),
        "B_norm": math.sqrt(eB),
        "total_norm": math.sqrt(total),
    }


def vo_affine_parts(Caug: torch.Tensor) -> Dict[str, Any]:
    X = Caug.detach().float().cpu()
    H1 = X.shape[1]
    H = H1 - 1
    C = X[:, :H]
    b = X[:, H]
    eC = float(C.square().sum())
    eb = float(b.square().sum())
    total = max(1e-12, eC + eb)
    return {
        "linear_energy_frac": eC / total,
        "write_bias_energy_frac": eb / total,
        "linear_norm": math.sqrt(eC),
        "bias_norm": math.sqrt(eb),
        "total_norm": math.sqrt(total),
    }


def delta_family_basis(rows: List[Dict[str, Any]], field_prefix: str) -> Dict[str, Any]:
    # Rows are per-delta with tensors optionally stored under _u/_v; for c values use scalar regression elsewhere.
    return {}


def fit_c_delta_basis(deltas: Sequence[int], cvals: Sequence[float]) -> Dict[str, Any]:
    # Simple symbolic/projection basis over relative position. This is raw-weight only.
    if not deltas:
        return {}
    x = torch.tensor(deltas, dtype=torch.float64)
    y = torch.tensor(cvals, dtype=torch.float64)
    xm = x / max(1.0, float(torch.max(torch.abs(x)).item()))
    cols = [
        torch.ones_like(xm),
        xm,
        xm.square(),
        torch.exp(-torch.abs(x) / 2.0),
        torch.exp(-torch.abs(x) / 4.0),
        torch.exp(-torch.abs(x) / 8.0),
        torch.cos(math.pi * xm),
        torch.sin(math.pi * xm),
    ]
    names = ["const", "linear_delta", "quad_delta", "decay2", "decay4", "decay8", "cos_pi", "sin_pi"]
    A = torch.stack(cols, dim=1)
    coef = torch.linalg.lstsq(A, y).solution
    pred = A @ coef
    err = float(torch.linalg.norm(pred - y) / torch.linalg.norm(y).clamp_min(1e-12))
    return {
        "basis_names": names,
        "coeffs": [float(v) for v in coef],
        "rel_err": err,
        "max_abs_residual": float((pred - y).abs().max()),
    }


def family_svd(vectors: List[torch.Tensor], ranks: Sequence[int]) -> Dict[str, Any]:
    if not vectors:
        return {}
    X = torch.stack([v.detach().float().cpu().reshape(-1) for v in vectors], dim=0)
    Xc = X  # no centering: origin matters for program family
    try:
        s = torch.linalg.svdvals(Xc)
    except Exception as e:
        return {"failed": str(e), "shape": list(X.shape)}
    e = s.square(); total = float(e.sum().clamp_min(1e-12))
    out: Dict[str, Any] = {"shape": list(X.shape), "top_singular": s[:8].tolist()}
    for r in ranks:
        rr = min(int(r), int(s.numel()))
        out[f"rank{r}_energy"] = float(e[:rr].sum() / total)
        out[f"rank{r}_rel_err"] = math.sqrt(max(0.0, float(e[rr:].sum()) / total))
    return out


def extract_head_weights(model: Any, layer_idx: int, head_idx: int) -> Dict[str, torch.Tensor]:
    layers = get_layers(model)
    cfg = model.config
    H = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    d = int(getattr(cfg, "head_dim", H // n_heads))
    groups = n_heads // n_kv
    kv_idx = int(head_idx) // groups
    attn = layers[layer_idx].self_attn
    qW = attn.q_proj.weight.detach().float().cpu()
    kW = attn.k_proj.weight.detach().float().cpu()
    vW = attn.v_proj.weight.detach().float().cpu()
    oW = attn.o_proj.weight.detach().float().cpu()
    def bias_of(mod):
        b = getattr(mod, "bias", None)
        return None if b is None else b.detach().float().cpu()
    qb = bias_of(attn.q_proj)
    kb = bias_of(attn.k_proj)
    vb = bias_of(attn.v_proj)
    Wq = qW[head_idx*d:(head_idx+1)*d, :].contiguous()
    Wk = kW[kv_idx*d:(kv_idx+1)*d, :].contiguous()
    Wv = vW[kv_idx*d:(kv_idx+1)*d, :].contiguous()
    Wo = oW[:, head_idx*d:(head_idx+1)*d].contiguous()
    bq = None if qb is None else qb[head_idx*d:(head_idx+1)*d].contiguous()
    bk = None if kb is None else kb[kv_idx*d:(kv_idx+1)*d].contiguous()
    bv = None if vb is None else vb[kv_idx*d:(kv_idx+1)*d].contiguous()
    return {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "bq": bq, "bk": bk, "bv": bv,
            "kv_idx": torch.tensor(kv_idx), "head_dim": torch.tensor(d), "hidden_size": torch.tensor(H)}


def analyze_one_head(model: Any, layer_idx: int, head_idx: int, deltas: Sequence[int], ranks: Sequence[int],
                     subspace_rank: int, block_topk: int, rope_theta: float) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    hw = extract_head_weights(model, layer_idx, head_idx)
    Wq, Wk, Wv, Wo = hw["Wq"], hw["Wk"], hw["Wv"], hw["Wo"]
    bq, bk, bv = hw["bq"], hw["bk"], hw["bv"]
    H = int(hw["hidden_size"].item()); d = int(hw["head_dim"].item())
    denom = math.sqrt(d)
    Cvo = Wo @ Wv
    Cvo_aug = affine_vo_aug(Wv, bv, Wo)
    QK_no_rope = (Wq.T @ Wk) / denom
    # Raw individual reports
    report: Dict[str, Any] = {
        "layer": layer_idx,
        "head": head_idx,
        "kv_idx": int(hw["kv_idx"].item()),
        "hidden_size": H,
        "head_dim": d,
        "bias_norms": {
            "bq": 0.0 if bq is None else float(torch.linalg.norm(bq)),
            "bk": 0.0 if bk is None else float(torch.linalg.norm(bk)),
            "bv": 0.0 if bv is None else float(torch.linalg.norm(bv)),
        },
        "raw": {
            "Wq": svd_energy_report(Wq, ranks),
            "Wk": svd_energy_report(Wk, ranks),
            "Wv": svd_energy_report(Wv, ranks),
            "Wo": svd_energy_report(Wo, ranks),
        },
        "composite": {
            "QK_no_rope": svd_energy_report(QK_no_rope, ranks),
            "VO": svd_energy_report(Cvo, ranks),
            "VO_aug": svd_energy_report(Cvo_aug, ranks),
            "VO_affine_parts": vo_affine_parts(Cvo_aug),
            "VO_block_energy": block_energy(Cvo, d, topk=block_topk),
            "QK_block_energy": block_energy(QK_no_rope, d, topk=block_topk),
        },
        "qk_delta": [],
    }
    u_vecs: List[torch.Tensor] = []
    v_vecs: List[torch.Tensor] = []
    cvals: List[float] = []
    for delta in deltas:
        R = rope_relative_matrix(d, int(delta), rope_theta=rope_theta)
        Maug = affine_qk_aug(Wq, Wk, bq, bk, R, denom)
        parts = qk_affine_parts(Maug)
        Haug = H + 1
        u_vecs.append(Maug[:H, H].detach().cpu())
        v_vecs.append(Maug[H, :H].detach().cpu())
        cvals.append(float(Maug[H, H]))
        row = {
            "delta": int(delta),
            **parts,
        }
        for r in ranks:
            # only full augmented SVD summary per requested rank; rank64 exact usually if head_dim=64 but bias may make <=65.
            pass
        report["qk_delta"].append(row)
    report["delta_family_basis"] = {
        "c_delta_symbolic_fit": fit_c_delta_basis(list(deltas), cvals),
        "u_delta_family_svd": family_svd(u_vecs, ranks),
        "v_delta_family_svd": family_svd(v_vecs, ranks),
    }
    # Bases for transitions/subspace
    bases = {
        "Cvo_out": top_singular_basis(Cvo, subspace_rank, side="left"),
        "Cvo_in": top_singular_basis(Cvo, subspace_rank, side="right"),
        "QK_out": top_singular_basis(QK_no_rope, subspace_rank, side="left"),
        "QK_in": top_singular_basis(QK_no_rope, subspace_rank, side="right"),
        "Wq_in": top_singular_basis(Wq, subspace_rank, side="right"),
        "Wk_in": top_singular_basis(Wk, subspace_rank, side="right"),
        "Wv_in": top_singular_basis(Wv, subspace_rank, side="right"),
        "Wo_out": top_singular_basis(Wo, subspace_rank, side="left"),
        "Cvo": Cvo,
        "QK_no_rope": QK_no_rope,
    }
    return report, bases


def transition_scan(model: Any, selected_layers: List[int], selected_heads: List[int], bases_by_head: Dict[Tuple[int,int], Dict[str, torch.Tensor]],
                    window: int, subspace_rank: int, topn: int) -> List[Dict[str, Any]]:
    cfg = model.config
    H = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    d = int(getattr(cfg, "head_dim", H // n_heads))
    groups = n_heads // n_kv
    layers = get_layers(model)
    rows: List[Dict[str, Any]] = []
    selected_set = set(selected_layers)
    for l in selected_layers:
        for h in selected_heads:
            key = (l, h)
            if key not in bases_by_head:
                continue
            Cvo = bases_by_head[key]["Cvo"].float().cpu()
            write_basis = bases_by_head[key]["Cvo_out"]
            for lp in range(l + 1, min(len(layers), l + 1 + window)):
                if lp not in selected_set and selected_layers != list(range(len(layers))):
                    # Still allow next layers if user did not include them? no, keep selected only.
                    pass
                attn2 = layers[lp].self_attn
                qW = attn2.q_proj.weight.detach().float().cpu()
                kW = attn2.k_proj.weight.detach().float().cpu()
                vW = attn2.v_proj.weight.detach().float().cpu()
                for hp in selected_heads:
                    kvp = hp // groups
                    Wq2 = qW[hp*d:(hp+1)*d, :]
                    Wk2 = kW[kvp*d:(kvp+1)*d, :]
                    Wv2 = vW[kvp*d:(kvp+1)*d, :]
                    for role, R in (("next_q", Wq2), ("next_k", Wk2), ("next_v", Wv2)):
                        read_basis = top_singular_basis(R, subspace_rank, side="right")
                        rows.append({
                            "from_layer": l, "from_head": h,
                            "to_layer": lp, "to_head": hp,
                            "reader_role": role,
                            "subspace_overlap": subspace_overlap(write_basis, read_basis, k=subspace_rank),
                            "norm_coupling": norm_coupling(R, Cvo, H),
                        })
    rows.sort(key=lambda r: (r["subspace_overlap"] * 0.7 + r["norm_coupling"] * 0.3), reverse=True)
    return rows[:topn]


def same_role_subspace_scan(selected_layers: List[int], selected_heads: List[int], bases_by_head: Dict[Tuple[int,int], Dict[str, torch.Tensor]],
                            subspace_rank: int, topn: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = sorted(bases_by_head.keys())
    roles = ["Cvo_out", "Cvo_in", "QK_out", "QK_in", "Wq_in", "Wk_in", "Wv_in"]
    for idx, a in enumerate(keys):
        for b in keys[idx+1:]:
            if a == b:
                continue
            for role in roles:
                if role in bases_by_head[a] and role in bases_by_head[b]:
                    rows.append({
                        "role": role,
                        "layer_a": a[0], "head_a": a[1],
                        "layer_b": b[0], "head_b": b[1],
                        "subspace_overlap": subspace_overlap(bases_by_head[a][role], bases_by_head[b][role], k=subspace_rank),
                    })
    rows.sort(key=lambda r: r["subspace_overlap"], reverse=True)
    return rows[:topn]


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
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def flatten_head_summary(rep: Dict[str, Any]) -> Dict[str, Any]:
    # Compact row for cross-head CSV.
    qk0 = rep.get("qk_delta", [{}])[0] if rep.get("qk_delta") else {}
    vo = rep.get("composite", {}).get("VO_affine_parts", {})
    raw = rep.get("raw", {})
    comp = rep.get("composite", {})
    row = {
        "layer": rep.get("layer"), "head": rep.get("head"), "kv_idx": rep.get("kv_idx"),
        "bq_norm": rep.get("bias_norms", {}).get("bq", 0.0),
        "bk_norm": rep.get("bias_norms", {}).get("bk", 0.0),
        "bv_norm": rep.get("bias_norms", {}).get("bv", 0.0),
        "qk_delta0_constant_frac": qk0.get("constant_energy_frac"),
        "qk_delta0_query_frac": qk0.get("query_affine_energy_frac"),
        "qk_delta0_key_frac": qk0.get("key_affine_energy_frac"),
        "qk_delta0_content_frac": qk0.get("content_bilinear_energy_frac"),
        "vo_write_bias_frac": vo.get("write_bias_energy_frac"),
        "vo_linear_frac": vo.get("linear_energy_frac"),
    }
    for r in (4, 8, 16, 32, 64):
        row[f"VO_rank{r}_err"] = comp.get("VO_aug", {}).get(f"rank{r}_rel_err")
        row[f"QK_rank{r}_err"] = comp.get("QK_no_rope", {}).get(f"rank{r}_rel_err")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="all", help="all, comma list, or ranges like 0-4,12")
    ap.add_argument("--heads", default="all")
    ap.add_argument("--deltas", default="0,1,2,3,4,5,6,7,8,16,32,64")
    ap.add_argument("--ranks", default="4,8,16,32,64")
    ap.add_argument("--subspace-rank", type=int, default=16)
    ap.add_argument("--block-topk", type=int, default=12)
    ap.add_argument("--top-transitions", type=int, default=120)
    ap.add_argument("--transition-window", type=int, default=2)
    ap.add_argument("--rope-theta", type=float, default=0.0, help="0 = read from config.rope_theta or 1e6 fallback")
    ap.add_argument("--out-dir", default="./qwen_raw_weight_projection_lab_v1")
    ap.add_argument("--no-transitions", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    dtype = get_dtype(args.dtype)
    print(f"loading model {args.model} dtype={dtype} device={args.device}", flush=True)
    load_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    # For raw weights CPU is usually fine. If user asks cuda, move model; 0.5B fits P40.
    if args.device != "cpu":
        model.to(args.device)
    cfg = model.config
    n_layers = len(get_layers(model))
    n_heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", int(cfg.hidden_size) // n_heads))
    rope_theta = float(args.rope_theta or getattr(cfg, "rope_theta", 1000000.0))
    selected_layers = parse_layers(args.layers, n_layers)
    selected_heads = parse_heads(args.heads, n_heads)
    deltas = parse_int_list(args.deltas)
    ranks = parse_int_list(args.ranks)
    print(f"scan layers={selected_layers} heads={selected_heads} deltas={deltas} ranks={ranks} rope_theta={rope_theta}", flush=True)

    all_reports: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    bases_by_head: Dict[Tuple[int,int], Dict[str, torch.Tensor]] = {}
    for l in selected_layers:
        layer_dir = out_dir / f"L{l}"
        ensure_dir(layer_dir)
        for h in selected_heads:
            print(f"analyze L{l}H{h}", flush=True)
            rep, bases = analyze_one_head(model, l, h, deltas, ranks, args.subspace_rank, args.block_topk, rope_theta)
            all_reports.append(rep)
            flat_rows.append(flatten_head_summary(rep))
            bases_by_head[(l,h)] = bases
            write_json(layer_dir / f"H{h}_raw_projection_summary.json", rep)

    write_csv(out_dir / "head_raw_projection_summary.csv", flat_rows)
    write_json(out_dir / "all_head_raw_projection_summary.json", all_reports)

    if not args.no_transitions:
        print("transition scan...", flush=True)
        transitions = transition_scan(model, selected_layers, selected_heads, bases_by_head,
                                      window=args.transition_window, subspace_rank=args.subspace_rank,
                                      topn=args.top_transitions)
        write_csv(out_dir / "top_cross_layer_transitions.csv", transitions)
        same = same_role_subspace_scan(selected_layers, selected_heads, bases_by_head,
                                       subspace_rank=args.subspace_rank, topn=args.top_transitions)
        write_csv(out_dir / "top_same_role_subspace_overlaps.csv", same)
    else:
        transitions = []
        same = []

    # Markdown overview
    md = []
    md.append("# Qwen Raw Weight Projection Lab v1\n")
    md.append(f"model={args.model} layers={selected_layers} heads={selected_heads}\n")
    md.append("\n## What this experiment asks\n")
    md.append("- What can be read from grey attention weights without activations?\n")
    md.append("- Which composite/projection view makes Wq/Wk/Wv/Wo less grey?\n")
    md.append("- Are affine QK terms bias/position/content dominated across heads?\n")
    md.append("- Does a head's VO write-subspace line up with next layer Q/K/V readers?\n")
    md.append("\n## Main output files\n")
    md.append("- head_raw_projection_summary.csv\n")
    md.append("- top_cross_layer_transitions.csv\n")
    md.append("- top_same_role_subspace_overlaps.csv\n")
    md.append("- L*/H*_raw_projection_summary.json\n")
    md.append("\n## Top heads by affine QK constant fraction\n")
    top_const = sorted(flat_rows, key=lambda r: safe_float(r.get("qk_delta0_constant_frac")), reverse=True)[:12]
    for r in top_const:
        md.append(f"- L{r['layer']}H{r['head']}: const={safe_float(r.get('qk_delta0_constant_frac')):.4f} q={safe_float(r.get('qk_delta0_query_frac')):.4f} k={safe_float(r.get('qk_delta0_key_frac')):.4f} content={safe_float(r.get('qk_delta0_content_frac')):.6f} VO_bias={safe_float(r.get('vo_write_bias_frac')):.4f}\n")
    if transitions:
        md.append("\n## Top transition lines by overlap/coupling\n")
        for r in transitions[:20]:
            md.append(f"- L{r['from_layer']}H{r['from_head']} -> L{r['to_layer']}H{r['to_head']} {r['reader_role']}: overlap={r['subspace_overlap']:.4f} coupling={r['norm_coupling']:.4f}\n")
    (out_dir / "summary.md").write_text("".join(md), encoding="utf-8")
    write_json(out_dir / "run_config.json", vars(args))
    print(f"done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
