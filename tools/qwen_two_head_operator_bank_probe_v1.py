#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_two_head_operator_bank_probe_v1.py

Non-standard 2-head operator-bank probe.

Goal
----
Do NOT just view attention maps. Do NOT repeat exact affine single-head check.
Instead test the idea:
    heads are operators in matrix/circuit space;
    a small bank of heads can explain a target/output with percentages.

For two selected heads hA,hB it computes:
  1) True per-head output contributions Y_A, Y_B on the same prompts.
  2) Least-squares decomposition of targets using bank [Y_A, Y_B]:
       Y_A, Y_B, Y_A+Y_B.
  3) Functional percentages with interference/overlap.
  4) Cross-explain: can head B explain head A and vice versa?
  5) Static raw-weight circuit summaries:
       QK_aug delta0 split: content/query/key/constant energy.
       VO_aug split: linear payload vs write-bias.
  6) Cross-space overlaps:
       write-space of head A -> read-space Wq/Wk/Wv of head B.
       write-space of head B -> read-space Wq/Wk/Wv of head A.

This is a bridge between:
  - raw weights as grey matrices;
  - circuit representations QK/VO;
  - head-bank decomposition percentages.

Example:
  python qwen_two_head_operator_bank_probe_v1.py \
    --base-script ./qwen_program_decompiler_v6_scorehybrid.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda --dtype fp16 --attn-implementation eager \
    --heads 2:1,3:6 \
    --prompt-suites all --prompts-per-suite 8 --same-text-repeats 2 \
    --max-length 192 \
    --subspace-rank 16 \
    --out-dir ./qwen_two_head_bank_L2H1_L3H6
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


def import_from_path(path: str, name: str = "qwen_base"):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_head_pair(s: str) -> List[Tuple[int, int]]:
    out = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        out.append((int(a), int(b)))
    if len(out) != 2:
        raise ValueError("--heads must contain exactly two heads, e.g. 2:1,3:6")
    return out


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.float()
    b = b.float()
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(eps))


def safe_float(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().float().cpu())
    return float(x)


def norm2(x: torch.Tensor) -> float:
    return float((x.float() * x.float()).sum().item())


def svd_subspace(M: torch.Tensor, rank: int, mode: str = "left") -> torch.Tensor:
    """Return orthonormal basis [D,r] for column/row space of matrix M."""
    M = M.float()
    if mode == "left":
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return U[:, : min(rank, U.shape[1])].contiguous()
    if mode == "right":
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return Vh[: min(rank, Vh.shape[0])].T.contiguous()
    raise ValueError(mode)


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]:
    """Mean squared canonical overlap between two bases [D,r]."""
    if A.numel() == 0 or B.numel() == 0:
        return {"mean_sq_cos": 0.0, "max_sq_cos": 0.0, "sum_sq_cos": 0.0}
    C = A.float().T @ B.float()
    s = torch.linalg.svdvals(C).clamp(0, 1)
    sq = s * s
    return {
        "mean_sq_cos": float(sq.mean().item()) if sq.numel() else 0.0,
        "max_sq_cos": float(sq.max().item()) if sq.numel() else 0.0,
        "sum_sq_cos": float(sq.sum().item()),
    }


def get_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find decoder layers")


def get_head_weights(model: Any, layer_idx: int, head_idx: int) -> Dict[str, torch.Tensor]:
    cfg = model.config
    H = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    d = int(getattr(cfg, "head_dim", H // n_heads))
    groups = n_heads // n_kv
    kv_idx = head_idx // groups
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    hs = slice(head_idx * d, (head_idx + 1) * d)
    ks = slice(kv_idx * d, (kv_idx + 1) * d)

    def w(module):
        return module.weight.detach().float().cpu()
    def b(module, sl):
        if getattr(module, "bias", None) is None:
            return torch.zeros(d, dtype=torch.float32)
        return module.bias.detach().float().cpu()[sl]

    Wq = w(attn.q_proj)[hs, :]
    Wk = w(attn.k_proj)[ks, :]
    Wv = w(attn.v_proj)[ks, :]
    Wo = w(attn.o_proj)[:, hs]
    bq = b(attn.q_proj, hs)
    bk = b(attn.k_proj, ks)
    bv = b(attn.v_proj, ks)
    return {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "bq": bq, "bk": bk, "bv": bv, "H": H, "D": d, "kv_idx": kv_idx}


def qk_aug_delta0_stats(hw: Dict[str, torch.Tensor]) -> Dict[str, float]:
    Wq = hw["Wq"]
    Wk = hw["Wk"]
    bq = hw["bq"]
    bk = hw["bk"]
    d = int(hw["D"])
    H = int(hw["H"])
    Wq_aug = torch.cat([Wq, bq[:, None]], dim=1)  # [D,H+1]
    Wk_aug = torch.cat([Wk, bk[:, None]], dim=1)  # [D,H+1]
    M = (Wq_aug.T @ Wk_aug) / math.sqrt(d)        # [H+1,H+1]
    B = M[:H, :H]
    q = M[:H, H]
    k = M[H, :H]
    c = M[H, H].view(1)
    eB = norm2(B); eq = norm2(q); ek = norm2(k); ec = norm2(c)
    total = max(eB + eq + ek + ec, 1e-12)
    return {
        "qk_aug_delta0_norm": math.sqrt(total),
        "qk_content_frac": eB / total,
        "qk_query_affine_frac": eq / total,
        "qk_key_affine_frac": ek / total,
        "qk_constant_frac": ec / total,
        "bq_norm": float(torch.linalg.norm(bq).item()),
        "bk_norm": float(torch.linalg.norm(bk).item()),
    }


def vo_aug_stats(hw: Dict[str, torch.Tensor], ranks: List[int]) -> Dict[str, float]:
    Wv = hw["Wv"]
    Wo = hw["Wo"]
    bv = hw["bv"]
    C = Wo @ Wv
    bvo = Wo @ bv
    lin_e = norm2(C)
    bias_e = norm2(bvo)
    total = max(lin_e + bias_e, 1e-12)
    out = {
        "vo_aug_norm": math.sqrt(total),
        "vo_linear_frac": lin_e / total,
        "vo_write_bias_frac": bias_e / total,
        "bv_norm": float(torch.linalg.norm(bv).item()),
        "bvo_norm": float(torch.linalg.norm(bvo).item()),
    }
    U, S, Vh = torch.linalg.svd(C.float(), full_matrices=False)
    fro2 = float((S * S).sum().item())
    for r in ranks:
        rr = min(int(r), int(S.numel()))
        tail = float((S[rr:] * S[rr:]).sum().item()) if rr < S.numel() else 0.0
        out[f"vo_linear_rank{r}_rel_err"] = math.sqrt(tail / max(fro2, 1e-12))
    return out


def flatten_head_outputs(seqs: List[Any]) -> torch.Tensor:
    parts = [s.Y.float().reshape(-1) for s in seqs]
    return torch.cat(parts, dim=0).cpu()


def fit_bank(target: torch.Tensor, bank: List[torch.Tensor]) -> Dict[str, Any]:
    A = torch.stack([b.float() for b in bank], dim=1)  # [N,K]
    y = target.float()
    G = A.T @ A
    rhs = A.T @ y
    coef = torch.linalg.solve(G + 1e-8 * torch.eye(G.shape[0]), rhs)
    pred = A @ coef
    err = rel_err(pred, y)
    contrib = []
    weighted_norms = []
    for i, b in enumerate(bank):
        weighted_norms.append(abs(float(coef[i])) * float(torch.linalg.norm(b.float()).item()))
    denom = sum(weighted_norms) + 1e-12
    for i, wn in enumerate(weighted_norms):
        contrib.append({"idx": i, "coef": float(coef[i].item()), "percent_abs_norm": float(wn / denom)})
    return {"rel_err": err, "coef": [float(x) for x in coef.tolist()], "contrib": contrib}


def pair_functional_summary(yA: torch.Tensor, yB: torch.Tensor) -> Dict[str, Any]:
    yA = yA.float(); yB = yB.float()
    pair = yA + yB
    eA = norm2(yA); eB = norm2(yB); cross = float((yA * yB).sum().item())
    epair = norm2(pair)
    cos = cross / max(math.sqrt(eA * eB), 1e-12)
    # projection of A onto B and B onto A
    coef_A_from_B = cross / max(eB, 1e-12)
    coef_B_from_A = cross / max(eA, 1e-12)
    pred_A_from_B = coef_A_from_B * yB
    pred_B_from_A = coef_B_from_A * yA
    # Gram decomposition of pair using [A,B]
    fit_pair = fit_bank(pair, [yA, yB])
    fit_A = fit_bank(yA, [yA, yB])
    fit_B = fit_bank(yB, [yA, yB])
    # Non-orthogonal energy percentages in actual pair energy.
    return {
        "norm_YA": math.sqrt(eA),
        "norm_YB": math.sqrt(eB),
        "norm_pair": math.sqrt(epair),
        "cosine_YA_YB": cos,
        "pair_energy_A_self_frac": eA / max(epair, 1e-12),
        "pair_energy_B_self_frac": eB / max(epair, 1e-12),
        "pair_energy_cross_frac": (2 * cross) / max(epair, 1e-12),
        "A_explained_by_B_coef": coef_A_from_B,
        "A_explained_by_B_rel_err": rel_err(pred_A_from_B, yA),
        "B_explained_by_A_coef": coef_B_from_A,
        "B_explained_by_A_rel_err": rel_err(pred_B_from_A, yB),
        "fit_pair_with_A_B": fit_pair,
        "fit_A_with_A_B": fit_A,
        "fit_B_with_A_B": fit_B,
    }


def cross_space_summary(hwA: Dict[str, torch.Tensor], hwB: Dict[str, torch.Tensor], rank: int) -> Dict[str, Any]:
    CA = hwA["Wo"] @ hwA["Wv"]
    CB = hwB["Wo"] @ hwB["Wv"]
    writeA = svd_subspace(CA, rank, mode="left")
    writeB = svd_subspace(CB, rank, mode="left")
    out = {
        "writeA_writeB": subspace_overlap(writeA, writeB),
    }
    for name in ["Wq", "Wk", "Wv"]:
        readB = svd_subspace(hwB[name], rank, mode="right")
        readA = svd_subspace(hwA[name], rank, mode="right")
        out[f"writeA_to_B_{name}_read"] = subspace_overlap(writeA, readB)
        out[f"writeB_to_A_{name}_read"] = subspace_overlap(writeB, readA)
    return out


def json_safe(x: Any):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [json_safe(v) for v in x]
    if torch.is_tensor(x):
        return x.detach().cpu().tolist() if x.numel() <= 32 else {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-script", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--heads", required=True, help="exactly two heads, e.g. 2:1,3:6")
    ap.add_argument("--prompt-suites", default="all")
    ap.add_argument("--prompts-per-suite", type=int, default=8)
    ap.add_argument("--same-text-repeats", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--subspace-rank", type=int, default=16)
    ap.add_argument("--ranks", default="4,8,16,32,64")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    base = import_from_path(args.base_script)
    heads = parse_head_pair(args.heads)
    ranks = [int(x) for x in args.ranks.replace(";", ",").split(",") if x.strip()]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = base.get_dtype(args.dtype) if hasattr(base, "get_dtype") else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": args.trust_remote_code}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(args.device)
    model.eval()

    prompts = base.build_prompts(args.prompt_suites, args.prompts_per_suite, args.same_text_repeats)
    print(f"collecting data for heads={heads} prompts={len(prompts)}", flush=True)
    data = base.collect_requested_heads_data(model, tok, prompts, heads, args.max_length, args.device)

    hA, hB = heads
    seqA = data[hA]
    seqB = data[hB]
    if len(seqA) != len(seqB):
        raise RuntimeError(f"sequence count mismatch: {len(seqA)} vs {len(seqB)}")
    yA = flatten_head_outputs(seqA)
    yB = flatten_head_outputs(seqB)

    hwA = get_head_weights(model, hA[0], hA[1])
    hwB = get_head_weights(model, hB[0], hB[1])

    headA_stats = {**qk_aug_delta0_stats(hwA), **vo_aug_stats(hwA, ranks)}
    headB_stats = {**qk_aug_delta0_stats(hwB), **vo_aug_stats(hwB, ranks)}
    func = pair_functional_summary(yA, yB)
    overlaps = cross_space_summary(hwA, hwB, args.subspace_rank)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "heads": {"A": {"layer": hA[0], "head": hA[1]}, "B": {"layer": hB[0], "head": hB[1]}},
        "nseq": len(seqA),
        "ntotal_output_entries": int(yA.numel()),
        "headA_static": headA_stats,
        "headB_static": headB_stats,
        "pair_functional": func,
        "cross_space_overlaps": overlaps,
        "interpretation_hints": {
            "pair_energy_cross_frac": "positive means heads reinforce; negative means cancellation/opposition",
            "A_explained_by_B_rel_err": "low means head B output can approximate head A output on prompts",
            "writeA_to_B_Wq_read": "candidate A writes into directions B can use as queries",
            "qk_*_frac": "static score-program split for delta0; high constant means bias/position-like backbone",
            "vo_write_bias_frac": "fraction of VO augmented energy from value bias write",
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    with open(out_dir / "head_static_summary.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted(set(headA_stats.keys()) | set(headB_stats.keys()))
        wr = csv.DictWriter(f, fieldnames=["tag", "layer", "head"] + keys)
        wr.writeheader()
        rowA = {"tag": "A", "layer": hA[0], "head": hA[1], **headA_stats}
        rowB = {"tag": "B", "layer": hB[0], "head": hB[1], **headB_stats}
        wr.writerow(rowA); wr.writerow(rowB)

    with open(out_dir / "cross_space_overlaps.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["metric", "mean_sq_cos", "max_sq_cos", "sum_sq_cos"])
        wr.writeheader()
        for k, v in overlaps.items():
            wr.writerow({"metric": k, **v})

    md = []
    md.append(f"# Two-head operator-bank probe: A=L{hA[0]}H{hA[1]} B=L{hB[0]}H{hB[1]}\n")
    md.append(f"nseq={len(seqA)} output_entries={yA.numel()}\n")
    md.append("## Functional output bank\n")
    md.append(f"- ||Y_A||={func['norm_YA']:.6g}\n")
    md.append(f"- ||Y_B||={func['norm_YB']:.6g}\n")
    md.append(f"- cosine(Y_A,Y_B)={func['cosine_YA_YB']:.6g}\n")
    md.append(f"- pair energy: A_self={func['pair_energy_A_self_frac']:.4f} B_self={func['pair_energy_B_self_frac']:.4f} cross={func['pair_energy_cross_frac']:.4f}\n")
    md.append(f"- A explained by B: coef={func['A_explained_by_B_coef']:.4g} rel_err={func['A_explained_by_B_rel_err']:.4f}\n")
    md.append(f"- B explained by A: coef={func['B_explained_by_A_coef']:.4g} rel_err={func['B_explained_by_A_rel_err']:.4f}\n")
    md.append("\n## Static score/write split\n")
    for tag, h, st in [("A", hA, headA_stats), ("B", hB, headB_stats)]:
        md.append(f"### {tag}=L{h[0]}H{h[1]}\n")
        md.append(f"- QK delta0: constant={st['qk_constant_frac']:.4f}, query={st['qk_query_affine_frac']:.4f}, key={st['qk_key_affine_frac']:.4f}, content={st['qk_content_frac']:.4f}\n")
        md.append(f"- VO: linear={st['vo_linear_frac']:.4f}, write_bias={st['vo_write_bias_frac']:.4f}\n")
    md.append("\n## Cross-space candidates\n")
    for k, v in sorted(overlaps.items(), key=lambda kv: kv[1]['mean_sq_cos'], reverse=True)[:10]:
        md.append(f"- {k}: mean_sq_cos={v['mean_sq_cos']:.4f}, max_sq_cos={v['max_sq_cos']:.4f}\n")
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("".join(md))

    print("\n" + "".join(md), flush=True)
    print(f"saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
