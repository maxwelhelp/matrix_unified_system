#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_multihead_differentiable_bank_v1.py

Multi-head differentiable operator-bank probe for Qwen2 attention heads.

Goal:
  Not just analyze heads separately. Build a differentiable bank of N heads
  where every head is represented by executable matrix-pseudocode pieces:

    score_h(i,j) = c_h(i,j) + q_aff_h(i,j) + k_aff_h(i,j) + content_h(i,j)
    A_h          = causal_softmax(score_h)
    payload_h(j) = linear_payload_h(j) + write_bias_h
    Y_h(i)       = sum_j A_h(i,j) payload_h(j)

Then test:
  1) exact pseudocode check per head
  2) joint bank reconstruction of selected heads together
  3) differentiable sparse gates over heads and score/VO terms
  4) term ablation: which terms matter jointly and per-head
  5) learned QK basis per head/rank, evaluated by A/Z/Y, not by pretty PCA
  6) cross-space overlaps: write-space of one head vs read-space of other heads

This is NOT a model patcher. It produces diagnostics and executable pseudo-code
bank evidence. Next version can patch the selected bank into the model.
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
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

try:
    import numpy as np
except Exception:
    np = None


# ---------------- small utils ----------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.detach().float()
    b = b.detach().float()
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(eps))


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (int, float, str, bool)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return str(x)
        return x
    if isinstance(x, Path):
        return str(x)
    if torch.is_tensor(x):
        if x.numel() <= 32:
            return x.detach().cpu().tolist()
        return {"tensor": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return str(x)


def parse_heads(s: str) -> List[Tuple[int, int]]:
    out = []
    for part in s.replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(':', 1)
        out.append((int(a), int(b)))
    # unique stable
    seen = set(); u = []
    for x in out:
        if x not in seen:
            u.append(x); seen.add(x)
    return u


def parse_ints(s: str) -> List[int]:
    return [int(x) for x in s.replace(';', ',').split(',') if x.strip()]


def import_base(path: str):
    """Import the user's base script safely.

    Python 3.12 dataclasses require the module to be present in sys.modules
    before exec_module(), otherwise @dataclass inside the imported file can crash
    with: sys.modules.get(cls.__module__) is None.
    """
    p = Path(path).resolve()
    module_name = "qwen_base_script_imported"
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script: {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


# ---------------- model helpers ----------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_qwen(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q/k [B,H,T,D], cos/sin [B,T,D] or [T,D]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rope_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [T,D], cos/sin [T,D]
    return (x * cos) + (rotate_half(x) * sin)


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, -1e9), dim=-1)


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


# ---------------- data structs ----------------

@dataclass
class HeadStatic:
    layer: int
    head: int
    kv: int
    Wq: torch.Tensor      # [D,H]
    Wk: torch.Tensor      # [D,H]
    Wv: torch.Tensor      # [D,H]
    Wo: torch.Tensor      # [H,D]
    bq: torch.Tensor      # [D]
    bk: torch.Tensor
    bv: torch.Tensor
    head_dim: int
    hidden_size: int


@dataclass
class HeadSeqTerms:
    key: str
    prompt_id: int
    suite: str
    text: str
    tokens: List[str]
    X: torch.Tensor             # [T,H] post RMSNorm
    Q: torch.Tensor             # [T,D] after RoPE
    K: torch.Tensor             # [T,D] after RoPE
    scores_true: torch.Tensor   # [T,T]
    A_true: torch.Tensor        # [T,T]
    V_true: torch.Tensor        # [T,D]
    Y_true: torch.Tensor        # [T,H]
    score_const: torch.Tensor   # [T,T]
    score_q: torch.Tensor
    score_k: torch.Tensor
    score_content: torch.Tensor
    Vlin: torch.Tensor          # [T,D]
    Vbias: torch.Tensor         # [D]
    Ylin_trueA: torch.Tensor    # [T,H] A_true @ Vlin @ Wo.T
    Ybias: torch.Tensor         # [H] Wo @ bv


# ---------------- prompts / load ----------------

def fallback_prompts(suites: str, prompts_per_suite: int, repeats: int) -> List[Dict[str, str]]:
    banks = {
        "code": ["Write a Python function that reverses a linked list.", "Explain binary search in Python."],
        "math": ["Solve: if 3x + 5 = 20, what is x?", "Explain Bayes theorem."],
        "text": ["Summarize why rivers are important for cities.", "Describe a rainy evening."],
        "symbols": ["JSON: {\"name\": \"Alice\", \"score\": 42}", "Array: [1, 1, 2, 3, 5, 8]"],
        "repeat": ["abc def abc def abc def", "one two three one two three"],
    }
    names = list(banks) if suites == "all" else [x.strip() for x in suites.split(',') if x.strip()]
    rows = []
    for n in names:
        arr = banks.get(n, banks["text"])
        for i in range(prompts_per_suite):
            rows.append({"suite": n, "text": arr[i % len(arr)]})
    for _ in range(repeats):
        rows.append({"suite": "same_repeat", "text": "The same calibration sentence repeated to check deterministic behavior."})
    return rows


def build_prompts(base: Any, suites: str, prompts_per_suite: int, repeats: int) -> List[Dict[str, str]]:
    if hasattr(base, "build_prompts"):
        return base.build_prompts(suites, prompts_per_suite, repeats)
    return fallback_prompts(suites, prompts_per_suite, repeats)


def get_dtype(base: Any, name: str):
    if hasattr(base, "get_dtype"):
        return base.get_dtype(name)
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def load_model_tokenizer(args, base):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = get_dtype(base, args.dtype)
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.device)
    model.eval()
    return model, tokenizer


# ---------------- static extraction ----------------

def extract_head_static(model: Any, layer_idx: int, head_idx: int, device: str) -> HeadStatic:
    cfg = model.config
    hidden_size = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
    kv_groups = num_heads // num_kv
    kv_idx = head_idx // kv_groups
    layer = get_layers(model)[layer_idx]
    attn = layer.self_attn

    hs = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
    kvs = slice(kv_idx * head_dim, (kv_idx + 1) * head_dim)

    def bias_or_zero(mod, sl):
        b = getattr(mod, "bias", None)
        if b is None:
            return torch.zeros(head_dim, dtype=torch.float32, device=device)
        return b.detach().float().to(device)[sl].contiguous()

    Wq = attn.q_proj.weight.detach().float().to(device)[hs, :].contiguous()
    Wk = attn.k_proj.weight.detach().float().to(device)[kvs, :].contiguous()
    Wv = attn.v_proj.weight.detach().float().to(device)[kvs, :].contiguous()
    Wo = attn.o_proj.weight.detach().float().to(device)[:, hs].contiguous()
    bq = bias_or_zero(attn.q_proj, hs)
    bk = bias_or_zero(attn.k_proj, kvs)
    bv = bias_or_zero(attn.v_proj, kvs)
    return HeadStatic(layer_idx, head_idx, kv_idx, Wq, Wk, Wv, Wo, bq, bk, bv, head_dim, hidden_size)


def decompose_head_seq(static: HeadStatic, X: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> HeadSeqTerms:
    raise RuntimeError("use collect_terms")


@torch.no_grad()
def collect_terms(model: Any, tokenizer: Any, prompts: List[Dict[str, str]], heads: List[Tuple[int, int]], args) -> Tuple[Dict[str, HeadStatic], Dict[str, List[HeadSeqTerms]]]:
    device = args.device
    cfg = model.config
    hidden_size = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    num_kv = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
    kv_groups = num_heads // num_kv
    layers = get_layers(model)

    by_layer: Dict[int, List[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    for l in by_layer:
        by_layer[l] = sorted(set(by_layer[l]))

    statics: Dict[str, HeadStatic] = {f"L{l}H{h}": extract_head_static(model, l, h, device) for l, h in heads}
    data: Dict[str, List[HeadSeqTerms]] = {k: [] for k in statics}

    for pi, pr in enumerate(prompts):
        enc = tokenizer(pr["text"], return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        T = int(input_ids.shape[1])
        if T < 2:
            continue
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
        toks = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
        pos = torch.arange(T, device=device).unsqueeze(0)

        for l, hs in by_layer.items():
            layer = layers[l]
            attn = layer.self_attn
            residual_in = outputs.hidden_states[l].detach()
            Xn = layer.input_layernorm(residual_in).detach()  # [1,T,H]
            X = Xn[0].float()
            cos, sin = compute_position_embeddings(model, Xn, pos)
            if cos.dim() == 3:
                cos1 = cos[0].float()
                sin1 = sin[0].float()
            else:
                cos1 = cos.float(); sin1 = sin.float()

            # module outputs for sanity/exact target
            q_full = attn.q_proj(Xn).view(1, T, num_heads, head_dim).transpose(1, 2).contiguous()
            k_full = attn.k_proj(Xn).view(1, T, num_kv, head_dim).transpose(1, 2).contiguous()
            v_full = attn.v_proj(Xn).view(1, T, num_kv, head_dim).transpose(1, 2).contiguous()
            q_rot_full, k_rot_full = apply_rope_qwen(q_full, k_full, cos, sin)

            for h in hs:
                key = f"L{l}H{h}"
                st = statics[key]
                kv_idx = h // kv_groups
                Qrot_true = q_rot_full[0, h].float()
                Krot_true = k_rot_full[0, kv_idx].float()
                V_true = v_full[0, kv_idx].float()
                scores_true = (Qrot_true @ Krot_true.T) / math.sqrt(head_dim)
                A_true = causal_softmax(scores_true)
                Y_true = (A_true @ V_true) @ st.Wo.T

                # Decompose q/k into linear and bias, then RoPE both parts.
                q_lin = X @ st.Wq.T
                k_lin = X @ st.Wk.T
                v_lin = X @ st.Wv.T
                q_bias = st.bq.unsqueeze(0).expand(T, -1)
                k_bias = st.bk.unsqueeze(0).expand(T, -1)

                qlin_rot = apply_rope_one(q_lin, cos1, sin1)
                klin_rot = apply_rope_one(k_lin, cos1, sin1)
                qbias_rot = apply_rope_one(q_bias, cos1, sin1)
                kbias_rot = apply_rope_one(k_bias, cos1, sin1)

                denom = math.sqrt(head_dim)
                score_content = (qlin_rot @ klin_rot.T) / denom
                score_q = (qlin_rot @ kbias_rot.T) / denom
                score_k = (qbias_rot @ klin_rot.T) / denom
                score_const = (qbias_rot @ kbias_rot.T) / denom
                Ylin_trueA = (A_true @ v_lin) @ st.Wo.T
                Ybias = st.Wo @ st.bv

                data[key].append(HeadSeqTerms(
                    key=key, prompt_id=pi, suite=pr["suite"], text=pr["text"], tokens=toks,
                    X=X.detach().cpu(), Q=Qrot_true.detach().cpu(), K=Krot_true.detach().cpu(), scores_true=scores_true.detach().cpu(), A_true=A_true.detach().cpu(),
                    V_true=V_true.detach().cpu(), Y_true=Y_true.detach().cpu(),
                    score_const=score_const.detach().cpu(), score_q=score_q.detach().cpu(),
                    score_k=score_k.detach().cpu(), score_content=score_content.detach().cpu(),
                    Vlin=v_lin.detach().cpu(), Vbias=st.bv.detach().cpu(),
                    Ylin_trueA=Ylin_trueA.detach().cpu(), Ybias=Ybias.detach().cpu(),
                ))
        del outputs
        if torch.cuda.is_available() and (pi + 1) % 16 == 0:
            torch.cuda.empty_cache()
    return statics, data


# ---------------- evaluation of executable pseudocode ----------------

def seq_y_from_terms(seq: HeadSeqTerms, lambdas: Dict[str, float], device: str) -> torch.Tensor:
    S = (float(lambdas.get("const", 1.0)) * seq.score_const.to(device) +
         float(lambdas.get("q", 1.0)) * seq.score_q.to(device) +
         float(lambdas.get("k", 1.0)) * seq.score_k.to(device) +
         float(lambdas.get("content", 1.0)) * seq.score_content.to(device))
    A = causal_softmax(S)
    Vlin = seq.Vlin.to(device)
    Ylin = (A @ Vlin)  # [T,D]
    # Wo is not in seq; we precomputed only for true A. Need use true payload decomposition? Actually need Wo.
    raise RuntimeError("seq_y_from_terms requires static Wo; use seq_y_from_terms_static")


def seq_y_from_terms_static(seq: HeadSeqTerms, st: HeadStatic, lambdas: Dict[str, torch.Tensor | float], device: str) -> torch.Tensor:
    def val(name, default=1.0):
        x = lambdas.get(name, default)
        if torch.is_tensor(x):
            return x.to(device)
        return torch.tensor(float(x), device=device)
    S = val("const") * seq.score_const.to(device) + val("q") * seq.score_q.to(device) + val("k") * seq.score_k.to(device) + val("content") * seq.score_content.to(device)
    A = causal_softmax(S)
    Vlin = seq.Vlin.to(device)
    y_lin = (A @ Vlin) @ st.Wo.T
    y_bias = st.Wo @ st.bv
    return val("vo_linear") * y_lin + val("vo_bias") * y_bias.unsqueeze(0)


def evaluate_head_exact(st: HeadStatic, seqs: List[HeadSeqTerms], device: str) -> Dict[str, float]:
    errs = {"score": [], "A": [], "Y": []}
    for s in seqs:
        S = s.score_const + s.score_q + s.score_k + s.score_content
        A = causal_softmax(S.to(device)).cpu()
        Y = seq_y_from_terms_static(s, st, {"const":1,"q":1,"k":1,"content":1,"vo_linear":1,"vo_bias":1}, device).cpu()
        errs["score"].append(rel_err(S, s.scores_true))
        errs["A"].append(rel_err(A, s.A_true))
        errs["Y"].append(rel_err(Y, s.Y_true))
    return {k+"_rel_mean": safe_mean(v) for k, v in errs.items()}


# ---------------- joint differentiable bank ----------------

class JointBank(torch.nn.Module):
    def __init__(self, keys: List[str], init_terms: float = 4.0, init_head: float = 1.0, sparse: bool = True):
        super().__init__()
        self.keys = keys
        # head coefficient, unconstrained but initialized near 1
        self.head_coef = torch.nn.Parameter(torch.full((len(keys),), float(init_head)))
        # gates through sigmoid. init 4 => ~0.982 enabled.
        self.term_logits = torch.nn.ParameterDict()
        for k in keys:
            self.term_logits[k] = torch.nn.Parameter(torch.full((6,), float(init_terms)))
        self.sparse = sparse

    def gates(self, key: str) -> Dict[str, torch.Tensor]:
        g = torch.sigmoid(self.term_logits[key])
        return {"const": g[0], "q": g[1], "k": g[2], "content": g[3], "vo_linear": g[4], "vo_bias": g[5]}


def fit_joint_bank(statics: Dict[str, HeadStatic], data: Dict[str, List[HeadSeqTerms]], train_ids: List[int], val_ids: List[int], args) -> Dict[str, Any]:
    device = args.device
    keys = list(statics.keys())
    model = JointBank(keys, sparse=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.bank_lr, weight_decay=0.0)
    # Build prompt-id index: assume all heads have same prompt ids
    prompt_ids = sorted(set(s.prompt_id for seqs in data.values() for s in seqs))
    seq_by_key_pid = {k: {s.prompt_id: s for s in data[k]} for k in keys}

    def loss_for_ids(ids: List[int], train: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
        num = None; den = None
        total_loss = torch.tensor(0.0, device=device)
        n = 0
        for pid in ids:
            ys_true = []
            ys_hat = []
            for hi, k in enumerate(keys):
                s = seq_by_key_pid[k].get(pid)
                if s is None:
                    continue
                st = statics[k]
                y_true = s.Y_true.to(device)
                y_hat = seq_y_from_terms_static(s, st, model.gates(k), device) * model.head_coef[hi]
                ys_true.append(y_true)
                ys_hat.append(y_hat)
            if not ys_true:
                continue
            target = torch.stack(ys_true, dim=0).sum(dim=0)
            pred = torch.stack(ys_hat, dim=0).sum(dim=0)
            diff = pred - target
            total_loss = total_loss + (diff.float().pow(2).sum() / target.float().pow(2).sum().clamp_min(1e-12))
            if num is None:
                num = diff.float().pow(2).sum()
                den = target.float().pow(2).sum()
            else:
                num = num + diff.float().pow(2).sum()
                den = den + target.float().pow(2).sum()
            n += 1
        if n == 0:
            return torch.tensor(0.0, device=device), {"rel": 999.0}
        loss = total_loss / n
        # sparsity pressure tries to disable useless terms but not too aggressively
        if train and args.sparse_lambda > 0:
            gate_sum = torch.tensor(0.0, device=device)
            for k in keys:
                gate_sum = gate_sum + torch.sigmoid(model.term_logits[k]).sum()
            loss = loss + args.sparse_lambda * gate_sum / max(1, len(keys) * 6)
        rel = torch.sqrt(num / den.clamp_min(1e-12)).detach()
        return loss, {"rel": float(rel.cpu())}

    best = None
    bad = 0
    for step in range(1, args.bank_steps + 1):
        loss, m = loss_for_ids(train_ids, train=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.bank_eval_every == 0 or step == args.bank_steps:
            with torch.no_grad():
                _, trm = loss_for_ids(train_ids, train=False)
                _, vam = loss_for_ids(val_ids, train=False)
            score = vam["rel"]
            if best is None or score < best[0]:
                best = (score, {"step": step, "train_rel": trm["rel"], "val_rel": vam["rel"], "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}})
                bad = 0
            else:
                bad += 1
                if bad >= args.bank_patience:
                    break
    if best is not None:
        model.load_state_dict(best[1]["state"], strict=True)
    with torch.no_grad():
        _, trm = loss_for_ids(train_ids, train=False)
        _, vam = loss_for_ids(val_ids, train=False)
    rows = []
    for hi, k in enumerate(keys):
        g = torch.sigmoid(model.term_logits[k]).detach().cpu().tolist()
        rows.append({
            "head": k,
            "head_coef": float(model.head_coef[hi].detach().cpu()),
            "gate_const": g[0], "gate_q": g[1], "gate_k": g[2], "gate_content": g[3],
            "gate_vo_linear": g[4], "gate_vo_bias": g[5],
        })
    return {"train_rel": trm["rel"], "val_rel": vam["rel"], "rows": rows, "best": best[1] if best else None}


# ---------------- ablation tests ----------------

def eval_joint_with_lambdas(statics: Dict[str, HeadStatic], data: Dict[str, List[HeadSeqTerms]], ids: List[int], term_override: Dict[str, float], device: str) -> float:
    keys = list(statics.keys())
    seq_by_key_pid = {k: {s.prompt_id: s for s in data[k]} for k in keys}
    num = torch.tensor(0.0, device=device); den = torch.tensor(0.0, device=device)
    for pid in ids:
        true = []
        pred = []
        for k in keys:
            s = seq_by_key_pid[k].get(pid)
            if s is None:
                continue
            true.append(s.Y_true.to(device))
            pred.append(seq_y_from_terms_static(s, statics[k], term_override, device))
        if true:
            T = torch.stack(true, 0).sum(0)
            P = torch.stack(pred, 0).sum(0)
            num += (P - T).float().pow(2).sum(); den += T.float().pow(2).sum()
    return float(torch.sqrt(num / den.clamp_min(1e-12)).detach().cpu())


def run_ablation_suite(statics: Dict[str, HeadStatic], data: Dict[str, List[HeadSeqTerms]], val_ids: List[int], device: str) -> List[Dict[str, Any]]:
    base = {"const":1,"q":1,"k":1,"content":1,"vo_linear":1,"vo_bias":1}
    variants = {
        "full": base,
        "no_const": {**base, "const":0},
        "no_q": {**base, "q":0},
        "no_k": {**base, "k":0},
        "no_content": {**base, "content":0},
        "only_const": {**base, "q":0, "k":0, "content":0},
        "const_qk_affine_no_content": {**base, "content":0},
        "only_content": {**base, "const":0, "q":0, "k":0},
        "no_vo_bias": {**base, "vo_bias":0},
        "vo_bias_only": {**base, "vo_linear":0},
    }
    return [{"variant": name, "joint_Y_rel": eval_joint_with_lambdas(statics, data, val_ids, lam, device)} for name, lam in variants.items()]


# ---------------- QK basis fit per head ----------------

def pca_qk_init(seqs: List[HeadSeqTerms], rank: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    M = torch.cat([s.Q for s in seqs] + [s.K for s in seqs], dim=0).float().to(device)
    _, _, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, Vh.shape[0])
    P = Vh[:r].T.contiguous()
    return P.clone(), P.clone()


def eval_qk_projection(seqs: List[HeadSeqTerms], st: HeadStatic, Pq: torch.Tensor, Pk: torch.Tensor, device: str) -> Dict[str, float]:
    rank = int(Pq.shape[1])
    denom = math.sqrt(rank)
    relA=[]; relZ=[]; relY=[]; kl=[]; top=[]
    for s in seqs:
        Q = s.Q.to(device).float() @ Pq.float()
        K = s.K.to(device).float() @ Pk.float()
        Ah = causal_softmax((Q @ K.T) / denom)
        At = s.A_true.to(device).float()
        V = s.V_true.to(device).float()
        Yt = s.Y_true.to(device).float()
        Zh = Ah @ V
        Zt = At @ V
        Yh = Zh @ st.Wo.T
        relA.append(rel_err(Ah, At))
        relZ.append(rel_err(Zh, Zt))
        relY.append(rel_err(Yh, Yt))
        kl.append(float(F.kl_div((Ah+1e-12).log(), At, reduction="batchmean").detach().cpu()))
        top.append(float((Ah.argmax(dim=-1).detach().cpu() == At.argmax(dim=-1).detach().cpu()).float().mean()))
    return {"A_rel": safe_mean(relA), "Z_rel": safe_mean(relZ), "Y_rel": safe_mean(relY), "KL": safe_mean(kl), "top1": safe_mean(top)}


def fit_qk_basis_learned(seqs_train: List[HeadSeqTerms], seqs_val: List[HeadSeqTerms], st: HeadStatic, rank: int, steps: int, lr: float, device: str) -> Dict[str, Any]:
    Pq0, Pk0 = pca_qk_init(seqs_train, rank, device)
    # full rank already exact-ish with identity/PCA; learned can spoil, so just evaluate PCA for rank >= head_dim
    if rank >= st.head_dim or steps <= 0:
        ev = eval_qk_projection(seqs_val, st, Pq0, Pk0, device)
        ev.update({"rank": rank, "basis_kind": "pca_shared_qk", "steps": 0})
        return ev
    Pq = torch.nn.Parameter(Pq0.clone())
    Pk = torch.nn.Parameter(Pk0.clone())
    opt = torch.optim.AdamW([Pq, Pk], lr=lr, weight_decay=0.0)
    denom = math.sqrt(rank)
    best=None
    for step in range(1, steps+1):
        loss = torch.tensor(0.0, device=device); n=0
        for s in seqs_train:
            Q = s.Q.to(device).float() @ Pq
            K = s.K.to(device).float() @ Pk
            Ah = causal_softmax((Q @ K.T) / denom)
            At = s.A_true.to(device).float()
            loss = loss + F.kl_div((Ah+1e-12).log(), At, reduction="batchmean")
            n += 1
        loss = loss / max(1,n)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([Pq,Pk], 1.0)
        opt.step()
        if step == steps or step % max(1, steps//3) == 0:
            ev = eval_qk_projection(seqs_val, st, Pq.detach(), Pk.detach(), device)
            if best is None or ev["A_rel"] < best[0]:
                best=(ev["A_rel"], step, Pq.detach().clone(), Pk.detach().clone(), ev)
    ev = best[4] if best else eval_qk_projection(seqs_val, st, Pq.detach(), Pk.detach(), device)
    ev.update({"rank": rank, "basis_kind": "learned_qk_functional", "steps": best[1] if best else steps})
    return ev


def fit_score_factor_basis(seqs_train: List[HeadSeqTerms], seqs_val: List[HeadSeqTerms], rank: int, steps: int, lr: float, device: str) -> Dict[str, float]:
    """Score-space SVD diagnostic. This is separate from learned Q/K projection basis."""
    def eval_set(seqs):
        relA=[]; top=[]
        for s in seqs:
            S = (s.score_const + s.score_q + s.score_k + s.score_content).float().to(device)
            U, Sv, Vh = torch.linalg.svd(S, full_matrices=False)
            r = min(rank, Sv.numel())
            Sh = (U[:, :r] * Sv[:r]) @ Vh[:r]
            Ah = causal_softmax(Sh)
            At = s.A_true.to(device)
            relA.append(rel_err(Ah, At))
            top.append(float((Ah.argmax(dim=-1).cpu() == s.A_true.argmax(dim=-1)).float().mean()))
        return {"A_rel": safe_mean(relA), "top1": safe_mean(top)}
    ev = eval_set(seqs_val)
    ev.update({"rank": rank, "basis_kind": "score_svd_per_sequence"})
    return ev


# ---------------- overlaps ----------------

def orth_basis(M: torch.Tensor, rank: int, side: str = "row") -> torch.Tensor:
    # Return orth basis in hidden space [H,r]. row-space of M [out,H] uses Vh; column-space uses U.
    M = M.float()
    if side == "row":
        _, _, Vh = torch.linalg.svd(M, full_matrices=False)
        return Vh[:rank].T.contiguous()
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    return U[:, :rank].contiguous()


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]:
    # A,B [H,r]
    C = A.T @ B
    s = torch.linalg.svdvals(C.float())
    return {"mean_sq_cos": float((s*s).mean().cpu()), "max_sq_cos": float((s*s).max().cpu()), "sum_sq_cos": float((s*s).sum().cpu())}


def compute_cross_overlaps(statics: Dict[str, HeadStatic], rank: int, out_csv: Path) -> List[Dict[str, Any]]:
    bases = {}
    for k, st in statics.items():
        Cvo = st.Wo @ st.Wv  # [H,H]
        bases[(k, "write")] = orth_basis(Cvo, rank, side="column")  # output/write directions
        bases[(k, "Wq_read")] = orth_basis(st.Wq, rank, side="row")
        bases[(k, "Wk_read")] = orth_basis(st.Wk, rank, side="row")
        bases[(k, "Wv_read")] = orth_basis(st.Wv, rank, side="row")
    rows=[]
    keys=list(statics.keys())
    for a in keys:
        for b in keys:
            if a == b: continue
            for read in ["Wq_read", "Wk_read", "Wv_read"]:
                m = subspace_overlap(bases[(a,"write")], bases[(b,read)])
                rows.append({"from_head": a, "to_head": b, "read_space": read, **m})
    rows.sort(key=lambda r: (-r["mean_sq_cos"], r["from_head"], r["to_head"]))
    with out_csv.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["from_head"])
        w.writeheader(); w.writerows(rows)
    return rows


# ---------------- static decomposition summaries ----------------

def static_summary_for_head(st: HeadStatic, deltas: List[int]) -> Dict[str, Any]:
    # Need rope matrix for exact delta; without model cos/sin we cannot build full RoPE matrix here robustly.
    # We still report VO and bias norms, raw ranks. QK affine term fractions are computed from delta0 with identity rope.
    H=st.hidden_size; D=st.head_dim
    Wq_aug = torch.cat([st.Wq, st.bq[:, None]], dim=1)  # [D,H+1]
    Wk_aug = torch.cat([st.Wk, st.bk[:, None]], dim=1)
    # delta0 identity-RoPE approximation; true RoPE delta handled by per-seq terms.
    M = Wq_aug.T @ Wk_aug / math.sqrt(D)
    B = M[:H,:H]; u = M[:H,H]; v = M[H,:H]; c = M[H,H]
    parts = {
        "content": float((B*B).sum().cpu()),
        "query": float((u*u).sum().cpu()),
        "key": float((v*v).sum().cpu()),
        "constant": float((c*c).cpu()),
    }
    tot=sum(parts.values()) or 1.0
    Cvo_aug = st.Wo @ torch.cat([st.Wv, st.bv[:,None]], dim=1)
    lin = Cvo_aug[:,:H]; bias = Cvo_aug[:,H]
    vo_lin=float((lin*lin).sum().cpu()); vo_bias=float((bias*bias).sum().cpu()); vt=vo_lin+vo_bias or 1.0
    return {
        "head": f"L{st.layer}H{st.head}", "layer": st.layer, "head_idx": st.head, "kv": st.kv,
        "qk_id_constant_frac": parts["constant"]/tot, "qk_id_query_frac": parts["query"]/tot,
        "qk_id_key_frac": parts["key"]/tot, "qk_id_content_frac": parts["content"]/tot,
        "vo_linear_frac": vo_lin/vt, "vo_bias_frac": vo_bias/vt,
        "bq_norm": float(torch.linalg.norm(st.bq).cpu()), "bk_norm": float(torch.linalg.norm(st.bk).cpu()), "bv_norm": float(torch.linalg.norm(st.bv).cpu()),
    }


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-script", type=str, default="./qwen_program_decompiler_v6_scorehybrid.py")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16")
    ap.add_argument("--attn-implementation", type=str, default="eager")
    ap.add_argument("--heads", type=str, default="2:1,3:6,3:2,4:8,4:2,4:0,4:5,2:8,0:9,4:9")
    ap.add_argument("--prompt-suites", type=str, default="all")
    ap.add_argument("--prompts-per-suite", type=int, default=8)
    ap.add_argument("--same-text-repeats", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bank-steps", type=int, default=200)
    ap.add_argument("--bank-lr", type=float, default=0.03)
    ap.add_argument("--bank-eval-every", type=int, default=25)
    ap.add_argument("--bank-patience", type=int, default=6)
    ap.add_argument("--sparse-lambda", type=float, default=0.002)
    ap.add_argument("--subspace-rank", type=int, default=16)
    ap.add_argument("--score-basis-ranks", type=str, default="4,8,16,32")
    ap.add_argument("--fit-qk-bases", action="store_true", help="fit learned Pq/Pk bases per head/rank, v6-like functional basis")
    ap.add_argument("--qk-basis-ranks", type=str, default="4,8,16,32,64")
    ap.add_argument("--qk-basis-steps", type=int, default=80)
    ap.add_argument("--qk-basis-lr", type=float, default=0.003)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    base = import_base(args.base_script)
    prompts = build_prompts(base, args.prompt_suites, args.prompts_per_suite, args.same_text_repeats)
    heads = parse_heads(args.heads)
    print(f"heads={heads} prompts={len(prompts)}", flush=True)
    model, tok = load_model_tokenizer(args, base)

    statics, data = collect_terms(model, tok, prompts, heads, args)
    keys = list(statics.keys())
    prompt_ids = sorted(set(s.prompt_id for seqs in data.values() for s in seqs))
    rng = random.Random(args.seed)
    shuffled = prompt_ids[:]; rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled)*args.val_frac))) if len(shuffled)>1 else 0
    val_ids = sorted(shuffled[:n_val]); train_ids = sorted(shuffled[n_val:])
    print(f"train_ids={len(train_ids)} val_ids={len(val_ids)}", flush=True)

    # per-head exact checks
    exact_rows=[]
    for k in keys:
        m=evaluate_head_exact(statics[k], data[k], args.device)
        row={"head":k, **m}; exact_rows.append(row)
    with (out_dir/"per_head_exact_checks.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(exact_rows[0].keys()))
        w.writeheader(); w.writerows(exact_rows)

    # static summaries
    static_rows=[static_summary_for_head(statics[k], [0]) for k in keys]
    with (out_dir/"head_static_summary.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(static_rows[0].keys()))
        w.writeheader(); w.writerows(static_rows)

    # joint fit
    bank = fit_joint_bank(statics, data, train_ids, val_ids, args)
    with (out_dir/"joint_bank_gates.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(bank["rows"][0].keys()))
        w.writeheader(); w.writerows(bank["rows"])

    # ablations
    ab_rows = run_ablation_suite(statics, data, val_ids, args.device)
    with (out_dir/"joint_term_ablation.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(ab_rows[0].keys()))
        w.writeheader(); w.writerows(ab_rows)

    # score and QK basis diagnostics per head
    basis_rows=[]
    qk_basis_rows=[]
    ranks=parse_ints(args.score_basis_ranks)
    qk_ranks=parse_ints(args.qk_basis_ranks)
    seq_by_pid = {k: {s.prompt_id: s for s in data[k]} for k in keys}
    for k in keys:
        tr=[seq_by_pid[k][pid] for pid in train_ids if pid in seq_by_pid[k]]
        va=[seq_by_pid[k][pid] for pid in val_ids if pid in seq_by_pid[k]]
        for r in ranks:
            ev=fit_score_factor_basis(tr, va, r, 0, 0, args.device)
            ev["head"]=k
            basis_rows.append(ev)
        if args.fit_qk_bases:
            for r in qk_ranks:
                ev=fit_qk_basis_learned(tr, va, statics[k], r, args.qk_basis_steps, args.qk_basis_lr, args.device)
                ev["head"]=k
                qk_basis_rows.append(ev)
                print(f"[qk_basis] {k} rank={r} {ev}", flush=True)
    if basis_rows:
        with (out_dir/"score_basis_svd.csv").open("w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(basis_rows[0].keys()))
            w.writeheader(); w.writerows(basis_rows)
    if qk_basis_rows:
        with (out_dir/"qk_basis_functional.csv").open("w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(qk_basis_rows[0].keys()))
            w.writeheader(); w.writerows(qk_basis_rows)

    # overlaps
    overlap_rows=compute_cross_overlaps(statics, args.subspace_rank, out_dir/"cross_space_overlaps.csv")

    # summary
    summary={
        "args": vars(args), "heads": keys, "n_prompts": len(prompt_ids), "train": len(train_ids), "val": len(val_ids),
        "exact_mean": {k: safe_mean([r[k] for r in exact_rows]) for k in exact_rows[0] if k != "head"},
        "joint_bank": {"train_rel": bank["train_rel"], "val_rel": bank["val_rel"]},
        "best_ablation": sorted(ab_rows, key=lambda r: r["joint_Y_rel"])[:5],
        "worst_ablation": sorted(ab_rows, key=lambda r: -r["joint_Y_rel"])[:5],
        "top_overlaps": overlap_rows[:20],
        "qk_basis_rows": qk_basis_rows[:20] if 'qk_basis_rows' in locals() else [],
    }
    with (out_dir/"summary.json").open("w") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)

    lines=[]
    lines.append(f"# Multi-head differentiable bank: {', '.join(keys)}")
    lines.append("")
    lines.append(f"prompts={len(prompt_ids)} train={len(train_ids)} val={len(val_ids)}")
    lines.append("")
    lines.append("## Exact pseudocode checks")
    for r in exact_rows:
        lines.append(f"- {r['head']}: score={r['score_rel_mean']:.3e} A={r['A_rel_mean']:.3e} Y={r['Y_rel_mean']:.3e}")
    lines.append("")
    lines.append("## Joint differentiable bank fit")
    lines.append(f"- train_rel={bank['train_rel']:.6f}")
    lines.append(f"- val_rel={bank['val_rel']:.6f}")
    lines.append("")
    lines.append("## Learned gates")
    for r in bank["rows"]:
        lines.append(f"- {r['head']}: coef={r['head_coef']:.3f} const={r['gate_const']:.2f} q={r['gate_q']:.2f} k={r['gate_k']:.2f} content={r['gate_content']:.2f} vo_lin={r['gate_vo_linear']:.2f} vo_bias={r['gate_vo_bias']:.2f}")
    lines.append("")
    lines.append("## Joint term ablation")
    for r in sorted(ab_rows, key=lambda x: x["joint_Y_rel"]):
        lines.append(f"- {r['variant']}: joint_Y_rel={r['joint_Y_rel']:.6f}")
    lines.append("")
    lines.append("## Top cross-space overlaps")
    for r in overlap_rows[:15]:
        lines.append(f"- {r['from_head']} -> {r['to_head']} {r['read_space']}: mean_sq_cos={r['mean_sq_cos']:.4f} max={r['max_sq_cos']:.4f}")
    (out_dir/"summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE", out_dir)
    print("summary:", out_dir/"summary.md")


if __name__ == "__main__":
    main()
