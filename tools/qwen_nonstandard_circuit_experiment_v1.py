#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_nonstandard_circuit_experiment_v1.py

New experiment layer on top of qwen_circuit_matrix_targets_v2_affine_basis.py.

This is NOT a standard QK/OV dump and NOT the already-tried exact affine check.
It assumes v2 already builds the exact affine circuit targets, then asks:

  1) Which affine score terms actually drive attention/output?
       score_ij = x_i^T B_delta x_j + x_i^T u_delta + v_delta^T x_j + c_delta

  2) Does each route use different score terms?
       route-by-route A/Z/Y errors for term programs.

  3) What does VO write actually depend on?
       linear payload vs write-bias vs low-rank linear payload.

  4) What operators/bases are missing in the delta-family itself?
       mine simple delta programs for c_delta, and SVD-family bases for u_delta/v_delta/(optional B_delta).

Outputs:
  summary.json
  score_term_energy.csv
  score_term_functional_overall.csv
  score_term_functional_by_route.csv
  vo_functional.csv
  token_pair_score_decomposition.json
  delta_family_mining.json
  summary.md

Example:
  python qwen_nonstandard_circuit_experiment_v1.py \
    --circuit-script ./qwen_circuit_matrix_targets_v2_affine_basis.py \
    --base-script ./qwen_program_decompiler_v6_scorehybrid.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --dtype fp16 \
    --attn-implementation eager --heads 2:1 --prompt-suites all \
    --prompts-per-suite 8 --same-text-repeats 2 --max-length 192 \
    --max-delta 64 --routes 3 --vo-ranks 4,8,16,32,64 \
    --out-dir ./qwen_nonstandard_circuit_v1_L2H1
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    KMeans = None
    StandardScaler = None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.detach().float()
    b = b.detach().float()
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(eps))


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def sanitize(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, float):
        return str(x) if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, Path):
        return str(x)
    if torch.is_tensor(x):
        if x.numel() <= 16:
            return x.detach().cpu().tolist()
        return {"__tensor__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    if np is not None:
        if isinstance(x, np.generic):
            return sanitize(x.item())
        if isinstance(x, np.ndarray):
            if x.size <= 16:
                return x.tolist()
            return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize(v) for v in x]
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(sanitize(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: sanitize(r.get(k, "")) for k in keys})


def load_module(path: str, name: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"module path not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- route features ----------------

def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1)


def local_mass(row: torch.Tensor, i: int, radius: int) -> float:
    lo = max(0, i - radius)
    hi = min(row.numel(), i + radius + 1)
    return float(row[lo:hi].sum())


def row_features_for_seq(seq: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    T = int(seq.A.shape[0])
    for i in range(T):
        row = seq.A[i].float()
        top = int(row.argmax())
        ent = float(-(row * (row + 1e-12).log()).sum() / math.log(max(2, T)))
        rows.append({
            "prompt_id": int(seq.prompt_id), "suite": seq.suite, "pos": i, "T": T,
            "token": seq.tokens[i] if i < len(seq.tokens) else str(i),
            "entropy": ent, "max_prob": float(row[top]), "top_idx": top,
            "top_dist": float(i - top), "top_rel": float(top / max(1, T - 1)),
            "pos_rel": float(i / max(1, T - 1)),
            "self_mass": float(row[i]),
            "prev_mass": float(row[i - 1]) if i > 0 else 0.0,
            "bos_mass": float(row[0]),
            "local1": local_mass(row, i, 1),
            "local2": local_mass(row, i, 2),
            "local4": local_mass(row, i, 4),
            "local8": local_mass(row, i, 8),
        })
    return rows


def infer_route_name(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "empty"
    m = {k: safe_mean([float(r[k]) for r in items]) for k in ["entropy", "max_prob", "self_mass", "prev_mass", "bos_mass", "local2", "local4", "local8"]}
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


def assign_routes(seqs: List[Any], k: int, seed: int = 0) -> Tuple[Dict[int, torch.Tensor], List[Dict[str, Any]], Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for s in seqs:
        all_rows.extend(row_features_for_seq(s))
    if not all_rows or k <= 1 or np is None or KMeans is None or StandardScaler is None:
        # fallback: threshold pseudo-labels
        mapping = {"self": 0, "prev": 1, "bos": 2, "local": 3, "diffuse": 4, "mixed": 5}
        labels = [mapping.get(infer_route_name([r]), 5) for r in all_rows]
    else:
        feat_keys = ["entropy", "max_prob", "top_dist", "top_rel", "pos_rel", "self_mass", "prev_mass", "bos_mass", "local1", "local2", "local4", "local8"]
        X = np.asarray([[float(r[a]) for a in feat_keys] for r in all_rows], dtype="float32")
        Xs = StandardScaler().fit_transform(X)
        kk = min(int(k), max(2, len(all_rows) - 1))
        labels = KMeans(n_clusters=kk, random_state=seed, n_init=10).fit_predict(Xs).tolist()
    by_prompt: Dict[int, List[Tuple[int, int]]] = {}
    for r, lab in zip(all_rows, labels):
        r["route_id"] = int(lab)
        by_prompt.setdefault(int(r["prompt_id"]), []).append((int(r["pos"]), int(lab)))
    route_labels: Dict[int, torch.Tensor] = {}
    for s in seqs:
        y = torch.full((s.A.shape[0],), -1, dtype=torch.long)
        for pos, lab in by_prompt.get(int(s.prompt_id), []):
            if 0 <= pos < y.numel():
                y[pos] = int(lab)
        route_labels[int(s.prompt_id)] = y
    info_rows = []
    for lab in sorted(set(labels)):
        rr = [r for r in all_rows if int(r["route_id"]) == int(lab)]
        info_rows.append({
            "route_id": int(lab), "route_name": infer_route_name(rr), "n_rows": len(rr),
            "entropy": safe_mean([float(r["entropy"]) for r in rr]),
            "max_prob": safe_mean([float(r["max_prob"]) for r in rr]),
            "bos_mass": safe_mean([float(r["bos_mass"]) for r in rr]),
            "self_mass": safe_mean([float(r["self_mass"]) for r in rr]),
            "prev_mass": safe_mean([float(r["prev_mass"]) for r in rr]),
            "local4": safe_mean([float(r["local4"]) for r in rr]),
        })
    return route_labels, info_rows, {"n_rows": len(all_rows), "k": len(set(labels)), "routes": info_rows}


# ---------------- score terms ----------------

def split_aug_terms(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """M [H+1,H+1] gives score = xi^T B xj + xi^T u + v^T xj + c."""
    H1 = M.shape[0]
    H = H1 - 1
    B = M[:H, :H].contiguous()
    u = M[:H, H].contiguous()
    v = M[H, :H].contiguous()
    c = M[H, H].contiguous()
    return B, u, v, c


def score_from_terms(seq: Any, Mdelta: Dict[int, torch.Tensor], mode: str, true_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
    X = seq.Xn.float()
    T, H = X.shape
    S = torch.empty(T, T, dtype=torch.float32)
    if true_scores is None:
        true_scores = (seq.Q.float() @ seq.K.float().T) / math.sqrt(seq.Q.shape[1])
    for i in range(T):
        for j in range(T):
            if j > i:
                S[i, j] = -1e9
                continue
            d = i - j
            M = Mdelta.get(int(d))
            if M is None:
                S[i, j] = true_scores[i, j]
                continue
            B, u, v, c = split_aug_terms(M)
            xi = X[i]
            xj = X[j]
            bilin = xi @ B @ xj
            qaff = xi @ u
            kaff = v @ xj
            const = c
            if mode == "full_affine":
                val = bilin + qaff + kaff + const
            elif mode == "only_const":
                val = const
            elif mode == "const_plus_q":
                val = const + qaff
            elif mode == "const_plus_k":
                val = const + kaff
            elif mode == "const_plus_qk_affine":
                val = const + qaff + kaff
            elif mode == "no_const":
                val = bilin + qaff + kaff
            elif mode == "no_q_affine":
                val = bilin + kaff + const
            elif mode == "no_k_affine":
                val = bilin + qaff + const
            elif mode == "no_content_bilinear":
                val = qaff + kaff + const
            elif mode == "only_content_bilinear":
                val = bilin
            elif mode == "only_q_affine":
                val = qaff
            elif mode == "only_k_affine":
                val = kaff
            elif mode == "content_plus_const":
                val = bilin + const
            elif mode == "query_key_affine_only":
                val = qaff + kaff
            else:
                raise ValueError(f"unknown score term mode {mode}")
            S[i, j] = val
    return S


def eval_score_mode(seqs: List[Any], weights: Dict[str, torch.Tensor], Mdelta: Dict[int, torch.Tensor], mode: str,
                    route_labels: Optional[Dict[int, torch.Tensor]] = None, route_id: Optional[int] = None) -> Dict[str, Any]:
    Wo = weights["Wo"].float()
    num_a = den_a = 0.0
    kl_sum = 0.0
    top_ok = 0
    nrows = 0
    num_z = den_z = 0.0
    num_y = den_y = 0.0
    for s in seqs:
        true_scores = (s.Q.float() @ s.K.float().T) / math.sqrt(s.Q.shape[1])
        Sh = score_from_terms(s, Mdelta, mode, true_scores=true_scores)
        Ah = causal_softmax(Sh)
        A = s.A.float()
        row_mask = torch.ones(A.shape[0], dtype=torch.bool)
        if route_labels is not None and route_id is not None:
            labs = route_labels.get(int(s.prompt_id))
            if labs is None:
                continue
            row_mask = labs == int(route_id)
            if int(row_mask.sum()) == 0:
                continue
        Ahm = Ah[row_mask]
        Am = A[row_mask]
        da = Ahm - Am
        num_a += float((da * da).sum())
        den_a += float((Am * Am).sum())
        kl_sum += float(F.kl_div((Ahm + 1e-12).log(), Am, reduction="sum"))
        top_ok += int((Ahm.argmax(dim=-1) == Am.argmax(dim=-1)).sum())
        nrows += int(Am.shape[0])
        V = s.V.float()
        Zt = A @ V
        Zh = Ah @ V
        dz = Zh[row_mask] - Zt[row_mask]
        num_z += float((dz * dz).sum())
        den_z += float((Zt[row_mask] * Zt[row_mask]).sum())
        Yt = s.Y.float()
        Yh = Zh @ Wo.T
        dy = Yh[row_mask] - Yt[row_mask]
        num_y += float((dy * dy).sum())
        den_y += float((Yt[row_mask] * Yt[row_mask]).sum())
    return {
        "program": mode, "route_id": route_id if route_id is not None else "all", "nrows": nrows,
        "A_rel": math.sqrt(num_a / max(1e-12, den_a)),
        "KL": kl_sum / max(1, nrows),
        "top1": float(top_ok / max(1, nrows)),
        "Z_rel": math.sqrt(num_z / max(1e-12, den_z)),
        "Y_rel": math.sqrt(num_y / max(1e-12, den_y)),
    }


def score_term_energy(Mdelta: Dict[int, torch.Tensor], report_deltas: int) -> List[Dict[str, Any]]:
    rows = []
    for d in sorted(Mdelta.keys())[:report_deltas + 1]:
        B, u, v, c = split_aug_terms(Mdelta[d])
        eb = float((B * B).sum())
        eu = float((u * u).sum())
        ev = float((v * v).sum())
        ec = float((c * c).item())
        total = max(1e-12, eb + eu + ev + ec)
        rows.append({
            "delta": d,
            "total_norm": math.sqrt(total),
            "content_bilinear_energy_frac": eb / total,
            "query_affine_energy_frac": eu / total,
            "key_affine_energy_frac": ev / total,
            "constant_energy_frac": ec / total,
            "c_delta": float(c.item()),
            "B_norm": math.sqrt(eb), "u_norm": math.sqrt(eu), "v_norm": math.sqrt(ev), "abs_c": abs(float(c.item())),
        })
    return rows


# ---------------- VO experiments ----------------

def svd_lowrank(M: torch.Tensor, rank: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    r = min(rank, S.numel())
    return (U[:, :r] * S[:r]) @ Vh[:r]


def eval_vo_mode(seqs: List[Any], weights: Dict[str, torch.Tensor], mode: str, rank: Optional[int] = None,
                 route_labels: Optional[Dict[int, torch.Tensor]] = None, route_id: Optional[int] = None) -> Dict[str, Any]:
    C_aug = weights["Wo"].float() @ weights["Wv_aug"].float()  # [H,H+1]
    C_lin = C_aug[:, :-1]
    b = C_aug[:, -1]
    if mode == "full_affine_vo":
        C_use = C_lin; b_use = b
    elif mode == "linear_no_write_bias":
        C_use = C_lin; b_use = torch.zeros_like(b)
    elif mode == "write_bias_only":
        C_use = torch.zeros_like(C_lin); b_use = b
    elif mode == "lowrank_linear_plus_bias":
        if rank is None:
            raise ValueError("rank required")
        C_use = svd_lowrank(C_lin, rank); b_use = b
    elif mode == "lowrank_linear_no_bias":
        if rank is None:
            raise ValueError("rank required")
        C_use = svd_lowrank(C_lin, rank); b_use = torch.zeros_like(b)
    else:
        raise ValueError(mode)
    num_y = den_y = 0.0; nrows = 0
    for s in seqs:
        payload = s.Xn.float() @ C_use.T + b_use.view(1, -1)
        Yh = s.A.float() @ payload
        Yt = s.Y.float()
        row_mask = torch.ones(Yt.shape[0], dtype=torch.bool)
        if route_labels is not None and route_id is not None:
            labs = route_labels.get(int(s.prompt_id))
            if labs is None:
                continue
            row_mask = labs == int(route_id)
            if int(row_mask.sum()) == 0:
                continue
        dy = Yh[row_mask] - Yt[row_mask]
        num_y += float((dy * dy).sum())
        den_y += float((Yt[row_mask] * Yt[row_mask]).sum())
        nrows += int(row_mask.sum())
    return {"program": mode, "rank": rank if rank is not None else "", "route_id": route_id if route_id is not None else "all", "nrows": nrows,
            "Y_rel_true_A": math.sqrt(num_y / max(1e-12, den_y))}


# ---------------- delta-family mining ----------------

def fit_delta_curve(ds: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    """Fit small handcrafted delta-programs to c_delta. This is not attention-standard; it is operator-program mining."""
    ds = ds.float()
    maxd = float(ds.max().clamp_min(1).item())
    cols = [
        torch.ones_like(ds),
        ds / maxd,
        (ds / maxd) ** 2,
        torch.exp(-ds / 2.0),
        torch.exp(-ds / 4.0),
        torch.exp(-ds / 8.0),
        torch.cos(ds / max(1.0, maxd) * math.pi),
        torch.sin(ds / max(1.0, maxd) * math.pi),
    ]
    names = ["const", "linear_delta", "quad_delta", "decay2", "decay4", "decay8", "cos_pi", "sin_pi"]
    A = torch.stack(cols, dim=1)
    coef = torch.linalg.lstsq(A, y.float()).solution
    pred = A @ coef
    return {
        "basis_names": names,
        "coeffs": [float(x) for x in coef],
        "rel_err": rel_err(pred, y),
        "max_abs_residual": float((pred - y).abs().max()),
        "formula": " + ".join([f"{float(c):+.4g}*{n}" for c, n in zip(coef, names) if abs(float(c)) > 1e-4]).replace("+ -", "- "),
    }


def svd_family(vectors: List[torch.Tensor], ranks: List[int]) -> Dict[str, Any]:
    if not vectors:
        return {"enabled": False, "reason": "empty"}
    M = torch.stack([v.flatten().float() for v in vectors], dim=0)
    # Centering is optional; here no centering because absolute operator family matters.
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    total = float((S * S).sum())
    rows = []
    for r in ranks:
        rr = min(int(r), S.numel())
        Mr = (U[:, :rr] * S[:rr]) @ Vh[:rr]
        rows.append({"rank": rr, "rel_err": rel_err(Mr, M), "energy": float((S[:rr] * S[:rr]).sum() / max(1e-12, total))})
    return {"enabled": True, "n_items": len(vectors), "dim": int(M.shape[1]), "singular_values_top8": [float(x) for x in S[:8]], "rank_table": rows}


def mine_delta_families(Mdelta: Dict[int, torch.Tensor], ranks: List[int], mine_content: bool, max_content_deltas: int) -> Dict[str, Any]:
    ds = sorted(Mdelta.keys())
    if not ds:
        return {"enabled": False, "reason": "no Mdelta"}
    c_vals = []
    u_vecs = []
    v_vecs = []
    B_vecs = []
    used_for_B = []
    for d in ds:
        B, u, v, c = split_aug_terms(Mdelta[d])
        c_vals.append(c.float())
        u_vecs.append(u.float())
        v_vecs.append(v.float())
        if mine_content and len(B_vecs) < max_content_deltas:
            B_vecs.append(B.float())
            used_for_B.append(d)
    c_tensor = torch.stack(c_vals)
    return {
        "delta_min": int(min(ds)), "delta_max": int(max(ds)), "n_delta": len(ds),
        "c_delta_curve_fit": fit_delta_curve(torch.tensor(ds), c_tensor),
        "u_delta_family_svd": svd_family(u_vecs, ranks),
        "v_delta_family_svd": svd_family(v_vecs, ranks),
        "B_delta_family_svd": svd_family(B_vecs, ranks) if mine_content else {"enabled": False, "reason": "disabled_by_default", "note": "content family is HxH and can be memory-heavy"},
        "B_delta_used_deltas": used_for_B,
    }


# ---------------- examples ----------------

def token_pair_decomposition_examples(seqs: List[Any], Mdelta: Dict[int, torch.Tensor], max_prompts: int, max_tokens: int, topk: int) -> List[Dict[str, Any]]:
    out = []
    for s in seqs[:max_prompts]:
        T = min(int(s.A.shape[0]), max_tokens)
        items = []
        X = s.Xn.float()
        for i in range(T):
            row = s.A[i, :i+1].float()
            vals, idxs = torch.topk(row, k=min(topk, row.numel()))
            for prob, jj in zip(vals, idxs):
                j = int(jj)
                d = i - j
                M = Mdelta.get(d)
                if M is None:
                    continue
                B, u, v, c = split_aug_terms(M)
                xi = X[i]; xj = X[j]
                bilin = float(xi @ B @ xj)
                qaff = float(xi @ u)
                kaff = float(v @ xj)
                const = float(c)
                total = bilin + qaff + kaff + const
                items.append({
                    "i": i, "j": j, "delta": d,
                    "query_token": s.tokens[i] if i < len(s.tokens) else str(i),
                    "key_token": s.tokens[j] if j < len(s.tokens) else str(j),
                    "A_prob": float(prob),
                    "score_total_decomp": total,
                    "score_content_bilinear": bilin,
                    "score_query_affine": qaff,
                    "score_key_affine": kaff,
                    "score_constant": const,
                })
        out.append({"prompt_id": int(s.prompt_id), "suite": s.suite, "text": s.text, "tokens": s.tokens[:T], "top_pairs": items})
    return out


# ---------------- main ----------------

def analyze_one_head(args: argparse.Namespace, circuit: Any, base_mod: Any, model: Any, tokenizer: Any, layer_idx: int, head_idx: int, out_root: Path) -> Dict[str, Any]:
    head_dir = out_root / f"L{layer_idx}H{head_idx}"
    ensure_dir(head_dir)
    prompts = circuit.build_prompts(base_mod, args.prompt_suites, args.prompts_per_suite, args.same_text_repeats)
    seqs, meta = circuit.collect_circuit_data(model, tokenizer, prompts, layer_idx, head_idx, args.max_length, args.device)
    if not seqs:
        raise RuntimeError("no sequences collected")
    weights = circuit.build_weight_slices(model, layer_idx, head_idx, meta)
    max_T = max(int(s.A.shape[0]) for s in seqs)
    Rpos = circuit.build_rope_by_pos(seqs, max(max_T - 1, args.max_delta))
    Mdelta = circuit.qk_delta_matrices_affine(weights, Rpos, args.max_delta, int(meta["head_dim"]))

    # 1) energy decomposition by score terms
    energy_rows = score_term_energy(Mdelta, args.report_deltas)
    write_csv(head_dir / "score_term_energy.csv", energy_rows)

    # 2) routes
    route_labels, route_rows, route_info = assign_routes(seqs, args.routes, seed=args.seed)
    write_csv(head_dir / "routes.csv", route_rows)

    # 3) score term functional ablations
    score_programs = [
        "full_affine",
        "only_const",
        "const_plus_q",
        "const_plus_k",
        "const_plus_qk_affine",
        "no_const",
        "no_q_affine",
        "no_k_affine",
        "no_content_bilinear",
        "only_content_bilinear",
        "content_plus_const",
        "query_key_affine_only",
        "only_q_affine",
        "only_k_affine",
    ]
    overall_rows = [eval_score_mode(seqs, weights, Mdelta, p) for p in score_programs]
    write_csv(head_dir / "score_term_functional_overall.csv", overall_rows)

    by_route_rows: List[Dict[str, Any]] = []
    for r in route_rows:
        rid = int(r["route_id"])
        for p in score_programs:
            row = eval_score_mode(seqs, weights, Mdelta, p, route_labels=route_labels, route_id=rid)
            row["route_name"] = r.get("route_name", "")
            by_route_rows.append(row)
    write_csv(head_dir / "score_term_functional_by_route.csv", by_route_rows)

    # 4) VO ablation / lowrank write experiments
    vo_rows: List[Dict[str, Any]] = []
    for mode in ["full_affine_vo", "linear_no_write_bias", "write_bias_only"]:
        vo_rows.append(eval_vo_mode(seqs, weights, mode))
    for r in parse_int_list(args.vo_ranks):
        vo_rows.append(eval_vo_mode(seqs, weights, "lowrank_linear_plus_bias", rank=r))
        vo_rows.append(eval_vo_mode(seqs, weights, "lowrank_linear_no_bias", rank=r))
    # route version for major VO modes only
    for rr in route_rows:
        rid = int(rr["route_id"])
        for mode in ["full_affine_vo", "linear_no_write_bias", "write_bias_only"]:
            row = eval_vo_mode(seqs, weights, mode, route_labels=route_labels, route_id=rid)
            row["route_name"] = rr.get("route_name", "")
            vo_rows.append(row)
    write_csv(head_dir / "vo_functional.csv", vo_rows)

    # 5) token-level score attribution examples
    examples = token_pair_decomposition_examples(seqs, Mdelta, args.example_prompts, args.example_tokens, args.example_topk)
    write_json(head_dir / "token_pair_score_decomposition.json", examples)

    # 6) delta-family mining
    mining = mine_delta_families(Mdelta, parse_int_list(args.family_ranks), args.mine_content_family, args.max_content_family_deltas)
    write_json(head_dir / "delta_family_mining.json", mining)

    # Summary extraction
    def best_row(rows: List[Dict[str, Any]], key: str = "Y_rel"):
        return sorted(rows, key=lambda x: float(x.get(key, 1e9)))[0] if rows else {}
    summary = {
        "head": f"L{layer_idx}H{head_idx}", "meta": meta, "nseq": len(seqs), "max_T": max_T,
        "route_info": route_info,
        "score_term_energy_first_rows": energy_rows[: min(len(energy_rows), 8)],
        "score_functional_best_by_Y": best_row(overall_rows, "Y_rel"),
        "score_functional_full": next((r for r in overall_rows if r["program"] == "full_affine"), None),
        "score_functional_no_content": next((r for r in overall_rows if r["program"] == "no_content_bilinear"), None),
        "score_functional_only_const": next((r for r in overall_rows if r["program"] == "only_const"), None),
        "score_functional_const_plus_qk_affine": next((r for r in overall_rows if r["program"] == "const_plus_qk_affine"), None),
        "vo_full": next((r for r in vo_rows if r["program"] == "full_affine_vo" and r["route_id"] == "all"), None),
        "vo_no_bias": next((r for r in vo_rows if r["program"] == "linear_no_write_bias" and r["route_id"] == "all"), None),
        "vo_bias_only": next((r for r in vo_rows if r["program"] == "write_bias_only" and r["route_id"] == "all"), None),
        "delta_family_mining": mining,
        "outputs": {
            "score_term_energy": str(head_dir / "score_term_energy.csv"),
            "score_term_functional_overall": str(head_dir / "score_term_functional_overall.csv"),
            "score_term_functional_by_route": str(head_dir / "score_term_functional_by_route.csv"),
            "vo_functional": str(head_dir / "vo_functional.csv"),
            "token_pair_score_decomposition": str(head_dir / "token_pair_score_decomposition.json"),
            "delta_family_mining": str(head_dir / "delta_family_mining.json"),
        },
    }
    write_json(head_dir / "summary.json", summary)
    md = render_summary_md(summary)
    (head_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return summary


def render_summary_md(s: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Nonstandard circuit experiment {s['head']}\n")
    lines.append(f"nseq={s['nseq']} max_T={s['max_T']}\n")
    full = s.get("score_functional_full") or {}
    onlyc = s.get("score_functional_only_const") or {}
    no_content = s.get("score_functional_no_content") or {}
    cqa = s.get("score_functional_const_plus_qk_affine") or {}
    lines.append("## Score programs\n")
    for name, row in [("full", full), ("only_const", onlyc), ("const+q+k affine", cqa), ("no_content_bilinear", no_content)]:
        if row:
            lines.append(f"- {name}: A_rel={row.get('A_rel'):.6g} Y_rel={row.get('Y_rel'):.6g} top1={row.get('top1'):.4g}")
    vof = s.get("vo_full") or {}; vonb = s.get("vo_no_bias") or {}; vob = s.get("vo_bias_only") or {}
    lines.append("\n## VO programs\n")
    for name, row in [("full", vof), ("linear_no_bias", vonb), ("bias_only", vob)]:
        if row:
            lines.append(f"- {name}: Y_rel_true_A={row.get('Y_rel_true_A'):.6g}")
    lines.append("\n## Route summary\n")
    for r in s.get("route_info", {}).get("routes", []):
        lines.append(f"- route{r['route_id']} {r['route_name']}: n={r['n_rows']} entropy={r['entropy']:.3f} bos={r['bos_mass']:.3f} local4={r['local4']:.3f}")
    lines.append("\n## Files\n")
    for k, v in s.get("outputs", {}).items():
        lines.append(f"- {k}: `{v}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit-script", default="./qwen_circuit_matrix_targets_v2_affine_basis.py")
    ap.add_argument("--base-script", default="./qwen_program_decompiler_v6_scorehybrid.py")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--heads", default="2:1")
    ap.add_argument("--prompt-suites", default="all")
    ap.add_argument("--prompts-per-suite", type=int, default=8)
    ap.add_argument("--same-text-repeats", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--max-delta", type=int, default=64)
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--vo-ranks", default="4,8,16,32,64")
    ap.add_argument("--family-ranks", default="1,2,4,8,16")
    ap.add_argument("--mine-content-family", action="store_true")
    ap.add_argument("--max-content-family-deltas", type=int, default=16)
    ap.add_argument("--report-deltas", type=int, default=12)
    ap.add_argument("--example-prompts", type=int, default=3)
    ap.add_argument("--example-tokens", type=int, default=24)
    ap.add_argument("--example-topk", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./qwen_nonstandard_circuit_v1")
    args = ap.parse_args()

    set_seed(args.seed)
    circuit = load_module(args.circuit_script, "qwen_circuit_v2")
    base_mod = None
    if args.base_script and Path(args.base_script).exists():
        base_mod = load_module(args.base_script, "qwen_base_v6")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = circuit.get_dtype(args.dtype) if hasattr(circuit, "get_dtype") else torch.float16
    kwargs = {"torch_dtype": dtype, "device_map": None}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    print(f"loading model {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, **kwargs).to(args.device)
    model.eval()
    layers = circuit.get_layers(model)
    cfg = model.config
    head_specs = circuit.parse_heads(args.heads, len(layers), int(cfg.num_attention_heads)) if hasattr(circuit, "parse_heads") else [(2, 1)]
    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    summaries = []
    for l, h in head_specs:
        summaries.append(analyze_one_head(args, circuit, base_mod, model, tokenizer, int(l), int(h), out_root))
    write_json(out_root / "summary_all.json", {"heads": summaries})
    print(f"\nDONE -> {out_root}")


if __name__ == "__main__":
    main()
