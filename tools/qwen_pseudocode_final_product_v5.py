#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_pseudocode_final_product_v5.py

Final product/report builder for the Matrix Pseudocode Decompiler experiments.
It does not replace the heavy GPU collectors. It consumes outputs from:
  - qwen_attention_pseudocode_atlas_v2_full.py
  - qwen_pseudocode_controls_v3/v4
  - baseline runs
and builds a human-readable Markdown+HTML dashboard, readiness checklist,
why-token report, MLP dictionary summary, and next-run command pack.

Design goal:
  Turn many CSV/JSON/MD artifacts into one product-quality evidence package:
    weights+activations -> executable matrix pseudocode -> patch/control evidence.

Usage examples:
  python qwen_pseudocode_final_product_v5.py \
    --atlas ./atlas_full_all \
    --controls-v4 ./controls_v4_results.zip \
    --controls-v3 ./controls_v3_results.zip \
    --baselines ./controls_baselines_top_heads_fixed.zip \
    --out-dir ./final_pseudocode_product_v5

  python qwen_pseudocode_final_product_v5.py \
    --atlas ./atlas_full_all.zip \
    --controls-v4 ./controls_v4_results.zip \
    --out-dir ./final_pseudocode_product_v5
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
import statistics as stats
import tempfile
import textwrap
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ------------------------- file helpers -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_zip(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".zip"


def stage_input(src: Optional[str], work: Path, name: str) -> Optional[Path]:
    if not src:
        return None
    p = Path(src).expanduser().resolve()
    if not p.exists():
        return None
    dst = work / name
    ensure_dir(dst)
    if is_zip(p):
        with zipfile.ZipFile(p, "r") as z:
            z.extractall(dst)
        return dst
    if p.is_dir():
        # do not copy large trees; use the original dir
        return p
    return None


def find_files(root: Optional[Path], pattern: str) -> List[Path]:
    if root is None or not root.exists():
        return []
    return sorted(root.rglob(pattern))


def first_file(root: Optional[Path], pattern: str) -> Optional[Path]:
    xs = find_files(root, pattern)
    return xs[0] if xs else None


def read_text(p: Optional[Path], max_chars: int = 200_000) -> str:
    if p is None or not p.exists():
        return ""
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
        if len(txt) > max_chars:
            return txt[:max_chars] + "\n...[truncated]..."
        return txt
    except Exception:
        return ""


def read_json(p: Optional[Path]) -> Any:
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def read_csv_rows(p: Optional[Path]) -> List[Dict[str, str]]:
    if p is None or not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "" or str(x).lower() in {"nan", "none"}:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def median(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = sorted(x for x in xs if x is not None)
    return float(stats.median(vals)) if vals else None


def p90(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = sorted(x for x in xs if x is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, int(math.ceil(0.90 * len(vals))) - 1)
    return float(vals[idx])


def fmt(x: Any, nd: int = 4) -> str:
    v = fnum(x, None)
    if v is None:
        return "n/a"
    if abs(v) < 1e-3 and v != 0:
        return f"{v:.2e}"
    return f"{v:.{nd}f}"


def table_md(rows: List[Dict[str, Any]], max_rows: int = 20) -> str:
    if not rows:
        return "_No data._\n"
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    rows2 = rows[:max_rows]
    out = []
    out.append("| " + " | ".join(keys) + " |")
    out.append("| " + " | ".join(["---"] * len(keys)) + " |")
    for r in rows2:
        out.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    if len(rows) > max_rows:
        out.append(f"\n_Showing {max_rows}/{len(rows)} rows._")
    return "\n".join(out) + "\n"


# ------------------------- analysis extraction -------------------------

def summarize_runtime(atlas: Optional[Path]) -> Dict[str, Any]:
    # Search flexible names: runtime/all_heads_runtime_summary.csv, extracted zips etc.
    files = find_files(atlas, "all_heads_runtime_summary.csv")
    rows: List[Dict[str, str]] = []
    for f in files:
        rows.extend(read_csv_rows(f))
    if not rows:
        return {"present": False, "rows": 0}
    cols = ["score_rel", "A_rel", "Y_rel", "no_k_Y_rel", "no_content_Y_rel", "no_vo_bias_Y_rel", "no_const_Y_rel", "no_q_Y_rel"]
    out: Dict[str, Any] = {"present": True, "rows": len(rows), "files": [str(f) for f in files]}
    for c in cols:
        vals = [fnum(r.get(c)) for r in rows]
        out[c + "_median"] = median(vals)
        out[c + "_p90"] = p90(vals)
        out[c + "_max"] = max([v for v in vals if v is not None], default=None)
    # strongest term heads
    for c in ["no_k_Y_rel", "no_content_Y_rel", "no_vo_bias_Y_rel"]:
        ranked = sorted(rows, key=lambda r: fnum(r.get(c), -1) or -1, reverse=True)[:10]
        out["top_" + c] = [compact_head_row(r, c) for r in ranked]
    return out


def summarize_static(atlas: Optional[Path]) -> Dict[str, Any]:
    files = find_files(atlas, "all_heads_static_summary.csv")
    rows: List[Dict[str, str]] = []
    for f in files:
        rows.extend(read_csv_rows(f))
    trans_files = find_files(atlas, "top_cross_layer_transitions.csv")
    trans: List[Dict[str, str]] = []
    for f in trans_files:
        trans.extend(read_csv_rows(f))
    out: Dict[str, Any] = {"present": bool(rows), "rows": len(rows), "transition_rows": len(trans)}
    if rows:
        type_counts: Dict[str, int] = {}
        for r in rows:
            t = r.get("head_type") or r.get("type") or classify_static_row(r)
            type_counts[t] = type_counts.get(t, 0) + 1
        out["type_counts"] = dict(sorted(type_counts.items(), key=lambda kv: kv[1], reverse=True))
    if trans:
        key = find_metric_key(trans[0], ["mean_sq_cos", "overlap", "coupling"])
        if key:
            ranked = sorted(trans, key=lambda r: fnum(r.get(key), -1) or -1, reverse=True)[:20]
            out["top_transitions"] = [slim_row(r, keep=["source", "target", "src", "dst", "from", "to", "read", key, "max_sq_cos", "coupling"]) for r in ranked]
    return out


def classify_static_row(r: Dict[str, str]) -> str:
    const = fnum(r.get("qk_delta0_constant_frac") or r.get("constant_frac"), 0.0) or 0.0
    query = fnum(r.get("qk_delta0_query_frac") or r.get("query_frac"), 0.0) or 0.0
    key = fnum(r.get("qk_delta0_key_frac") or r.get("key_frac"), 0.0) or 0.0
    content = fnum(r.get("qk_delta0_content_frac") or r.get("content_frac"), 0.0) or 0.0
    vo_bias = fnum(r.get("vo_write_bias_frac") or r.get("VO_bias_frac"), 0.0) or 0.0
    base = "mixed"
    if const > 0.65:
        base = "positional_affine"
    elif query > 0.55:
        base = "query_affine"
    elif key > 0.45 and content > 0.15:
        base = "key_content"
    elif key > 0.45:
        base = "key_affine"
    elif content > 0.25:
        base = "content_bilinear"
    if vo_bias > 0.35:
        base += "+strong_vo_bias"
    elif vo_bias > 0.10:
        base += "+vo_bias"
    return base


def compact_head_row(r: Dict[str, str], metric: str) -> Dict[str, Any]:
    return {
        "head": r.get("head") or f"L{r.get('layer', r.get('layer_idx','?'))}H{r.get('head_idx', r.get('head','?'))}",
        metric: fmt(r.get(metric)),
        "score_rel": fmt(r.get("score_rel")),
        "A_rel": fmt(r.get("A_rel")),
        "Y_rel": fmt(r.get("Y_rel")),
    }


def slim_row(r: Dict[str, str], keep: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in keep:
        if k in r:
            out[k] = r[k]
    # fallback include first few keys if none matched
    if not out:
        for k in list(r.keys())[:8]:
            out[k] = r[k]
    return out


def find_metric_key(row: Dict[str, str], prefs: List[str]) -> Optional[str]:
    for p in prefs:
        if p in row:
            return p
    for k in row.keys():
        if any(p.lower() in k.lower() for p in prefs):
            return k
    return None


def summarize_mlp(atlas: Optional[Path], controls: Optional[Path]) -> Dict[str, Any]:
    rows: List[Dict[str, str]] = []
    files = []
    for root in [atlas, controls]:
        for pat in ["all_layers_mlp_summary.csv", "mlp_rich_summary.csv", "mlp_operator_summary.csv"]:
            for f in find_files(root, pat):
                files.append(f)
                rows.extend(read_csv_rows(f))
    out: Dict[str, Any] = {"present": bool(rows), "rows": len(rows), "files": [str(f) for f in files]}
    if not rows:
        return out
    for c in ["mlp_rec_rel", "J_norm", "J_rank90", "J_rank95", "J_rank99", "out_norm", "no_gate_delta_rel", "gate_only_delta_rel"]:
        vals = [fnum(r.get(c)) for r in rows]
        if any(v is not None for v in vals):
            out[c + "_mean"] = mean(vals)
            out[c + "_median"] = median(vals)
            out[c + "_p90"] = p90(vals)
    # top layers by out_norm/J_norm
    for c in ["out_norm", "J_norm", "no_gate_delta_rel"]:
        if any(c in r for r in rows):
            ranked = sorted(rows, key=lambda r: fnum(r.get(c), -1) or -1, reverse=True)[:15]
            out["top_" + c] = [slim_row(r, ["layer", "layer_idx", c, "J_rank90", "J_rank99", "mlp_rec_rel", "top_neurons"]) for r in ranked]
    return out


def summarize_controls(root: Optional[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"present": root is not None and root.exists()}
    # token sweep
    sweep_rows: List[Dict[str, str]] = []
    for f in find_files(root, "token_control_sweep_summary.csv"):
        sweep_rows.extend(read_csv_rows(f))
    out["token_sweep_rows"] = len(sweep_rows)
    if sweep_rows:
        out["token_sweep_sign_match_rate"] = mean([fnum(r.get("sign_match")) for r in sweep_rows])
        out["token_sweep_logit_rel_mean"] = mean([fnum(r.get("logit_rel")) for r in sweep_rows])
        out["token_sweep_KL_mean"] = mean([fnum(r.get("KL")) or fnum(r.get("kl")) for r in sweep_rows])
        out["token_sweep_top_effects"] = [slim_row(r, ["head", "source_head", "term", "key_pos", "predicted_sign", "actual_sign", "sign_match", "mass_delta", "head_Y_delta_rel", "logit_rel", "KL"]) for r in sorted(sweep_rows, key=lambda r: abs(fnum(r.get("mass_delta") or r.get("target_key_mass_delta_mean"), 0.0) or 0.0), reverse=True)[:20]]
    # single token_control
    tc_rows: List[Dict[str, str]] = []
    for f in find_files(root, "token_control_summary.csv"):
        tc_rows.extend(read_csv_rows(f))
    out["token_control_rows"] = len(tc_rows)
    # causal path/recovery
    cp_rows: List[Dict[str, str]] = []
    for f in find_files(root, "causal_path_summary.csv"):
        cp_rows.extend(read_csv_rows(f))
    out["causal_path_rows"] = len(cp_rows)
    if cp_rows:
        out["causal_path_target_Y_rel_delta_mean"] = mean([fnum(r.get("target_Y_rel_delta")) for r in cp_rows])
        out["causal_path_logit_rel_mean"] = mean([fnum(r.get("logit_rel")) for r in cp_rows])
    rec_rows: List[Dict[str, str]] = []
    for f in find_files(root, "causal_path_recovery_summary.csv"):
        rec_rows.extend(read_csv_rows(f))
    out["causal_path_recovery_rows"] = len(rec_rows)
    if rec_rows:
        out["recovery_target_Y_recovery_mean"] = mean([fnum(r.get("target_Y_recovery_ratio")) for r in rec_rows])
        out["recovery_logit_recovery_mean"] = mean([fnum(r.get("logit_recovery_ratio")) for r in rec_rows])
        out["recovery_examples"] = [slim_row(r, ["source_head", "target_head", "prompt_id", "ablate_head_target_Y_rel", "replace_full_target_Y_rel", "target_Y_recovery_ratio", "ablate_head_logit_rel", "replace_full_logit_rel", "logit_recovery_ratio"]) for r in rec_rows[:20]]
    # patch/logit attr
    attr_rows: List[Dict[str, str]] = []
    for f in find_files(root, "logit_patch_attribution.csv"):
        attr_rows.extend(read_csv_rows(f))
    out["logit_patch_attr_rows"] = len(attr_rows)
    if attr_rows:
        metric = find_metric_key(attr_rows[0], ["causal_logit_contribution", "delta_logit", "logit_contribution", "target_logit_delta"])
        if metric:
            pos = sorted(attr_rows, key=lambda r: fnum(r.get(metric), 0.0) or 0.0, reverse=True)[:15]
            neg = sorted(attr_rows, key=lambda r: fnum(r.get(metric), 0.0) or 0.0)[:15]
            out["logit_patch_top_positive"] = [slim_row(r, ["component", "kind", "layer", "head", metric, "loss_delta", "logit_rel", "top1_match"]) for r in pos]
            out["logit_patch_top_negative"] = [slim_row(r, ["component", "kind", "layer", "head", metric, "loss_delta", "logit_rel", "top1_match"]) for r in neg]
    # baselines
    base_rows: List[Dict[str, str]] = []
    for f in find_files(root, "baselines_summary.csv"):
        base_rows.extend(read_csv_rows(f))
    out["baseline_rows"] = len(base_rows)
    if base_rows:
        out["baseline_qk_content_A_rel_mean"] = mean([fnum(r.get("qk_content_only_A_rel")) for r in base_rows])
        out["baseline_qk_content_Y_rel_mean"] = mean([fnum(r.get("qk_content_only_Y_rel")) for r in base_rows])
        out["baseline_head_ablation_Y_rel_mean"] = mean([fnum(r.get("head_ablation_Y_rel")) for r in base_rows])
    return out


def summarize_generation(root: Optional[Path]) -> Dict[str, Any]:
    files = find_files(root, "generation_trace_summary.json") + find_files(root, "step_*.json")
    out: Dict[str, Any] = {"present": bool(files), "files": len(files)}
    # Try summary first
    summ = read_json(first_file(root, "generation_trace_summary.json"))
    if summ:
        out["summary"] = summ
    # Steps
    step_rows = []
    for f in find_files(root, "step_*.json"):
        j = read_json(f)
        if isinstance(j, dict):
            step_rows.append({
                "step": j.get("step", f.stem),
                "token": j.get("next_token") or j.get("decoded_token") or j.get("token"),
                "prob": j.get("next_prob") or j.get("prob"),
                "top_heads": len(j.get("top_heads", [])) if isinstance(j.get("top_heads"), list) else "",
            })
    if step_rows:
        out["steps"] = step_rows
    return out


# ------------------------- reports -------------------------

def build_readiness(runtime: Dict[str, Any], static: Dict[str, Any], mlp: Dict[str, Any], controls: Dict[str, Any], gen: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    def add(name: str, ok: bool, evidence: str, next_step: str = ""):
        items.append({"item": name, "status": "OK" if ok else "MISSING/PARTIAL", "evidence": evidence, "next_step": next_step})
    add("All-head static atlas", static.get("rows", 0) >= 300, f"rows={static.get('rows',0)}, transitions={static.get('transition_rows',0)}", "run static all if missing")
    add("All-head runtime pseudocode", runtime.get("rows", 0) >= 300 and (runtime.get("Y_rel_median") or 999) < 1e-4, f"rows={runtime.get('rows',0)}, Y_rel_med={fmt(runtime.get('Y_rel_median'))}", "run runtime all")
    add("Term ablation evidence", runtime.get("no_k_Y_rel_median") is not None, f"no_k_med={fmt(runtime.get('no_k_Y_rel_median'))}, no_content_med={fmt(runtime.get('no_content_Y_rel_median'))}, no_vo_bias_med={fmt(runtime.get('no_vo_bias_Y_rel_median'))}", "")
    add("Token directional control", controls.get("token_sweep_rows", 0) > 0 and (controls.get("token_sweep_sign_match_rate") or 0) > 0.8, f"rows={controls.get('token_sweep_rows',0)}, sign_match={fmt(controls.get('token_sweep_sign_match_rate'))}", "run wider token sweep")
    add("Causal path influence", controls.get("causal_path_rows", 0) > 0, f"rows={controls.get('causal_path_rows',0)}, targetY_delta={fmt(controls.get('causal_path_target_Y_rel_delta_mean'))}", "")
    add("Causal path recovery", controls.get("causal_path_recovery_rows", 0) > 0 and (controls.get("recovery_target_Y_recovery_mean") or 0) > 0.95, f"rows={controls.get('causal_path_recovery_rows',0)}, targetY_recovery={fmt(controls.get('recovery_target_Y_recovery_mean'))}", "run more path pairs")
    add("Patch-based logit attribution", controls.get("logit_patch_attr_rows", 0) > 0, f"rows={controls.get('logit_patch_attr_rows',0)}", "merge into generation trace")
    add("MLP operator/Jacobian", mlp.get("rows", 0) > 0 and mlp.get("J_rank90_mean") is not None, f"rows={mlp.get('rows',0)}, J_rank90_mean={fmt(mlp.get('J_rank90_mean'))}", "add MLP operator dictionary")
    add("Baselines", controls.get("baseline_rows", 0) > 0, f"rows={controls.get('baseline_rows',0)}, contentQK_A_mean={fmt(controls.get('baseline_qk_content_A_rel_mean'))}", "add TransformerLens/SAE baseline optionally")
    add("Generation trace", gen.get("present", False), f"files={gen.get('files',0)}", "merge signed attribution into why-token trace")
    return items


def build_markdown(out_dir: Path, runtime: Dict[str, Any], static: Dict[str, Any], mlp: Dict[str, Any], controls: Dict[str, Any], gen: Dict[str, Any], readiness: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Matrix Pseudocode Decompiler — Final Product Report v5\n")
    lines.append("This report combines static weights, runtime attention pseudocode, MLP operator summaries, control experiments, baselines, and generation traces.\n")

    lines.append("## 1. Executive status\n")
    lines.append(table_md(readiness, max_rows=20))

    lines.append("## 2. Core claim\n")
    lines.append("```python\nweights + architecture + activations\n  -> affine/RoPE/RMS circuit targets\n  -> executable matrix pseudocode terms\n  -> runtime reconstruction\n  -> term/path/logit patch control\n```\n")

    lines.append("## 3. Attention runtime reconstruction\n")
    if runtime.get("present"):
        lines.append(f"Rows: **{runtime.get('rows')}**\n")
        lines.append("| metric | median | p90 | max |\n|---|---:|---:|---:|\n")
        for c in ["score_rel", "A_rel", "Y_rel", "no_k_Y_rel", "no_content_Y_rel", "no_vo_bias_Y_rel", "no_const_Y_rel", "no_q_Y_rel"]:
            lines.append(f"| {c} | {fmt(runtime.get(c+'_median'))} | {fmt(runtime.get(c+'_p90'))} | {fmt(runtime.get(c+'_max'))} |\n")
        lines.append("\n### Top k/content/VO-bias sensitive heads\n")
        for k in ["top_no_k_Y_rel", "top_no_content_Y_rel", "top_no_vo_bias_Y_rel"]:
            lines.append(f"#### {k}\n")
            lines.append(table_md(runtime.get(k, []), max_rows=10))
    else:
        lines.append("No runtime summary found.\n")

    lines.append("## 4. Static all-head map\n")
    if static.get("present"):
        lines.append(f"Static rows: **{static.get('rows')}**, transition rows: **{static.get('transition_rows')}**\n\n")
        lines.append("### Head type counts\n")
        rows = [{"type": k, "count": v} for k, v in static.get("type_counts", {}).items()]
        lines.append(table_md(rows, max_rows=30))
        lines.append("### Top static cross-layer transition candidates\n")
        lines.append(table_md(static.get("top_transitions", []), max_rows=20))
    else:
        lines.append("No static summary found.\n")

    lines.append("## 5. Control experiments\n")
    lines.append(f"Token sweep rows: **{controls.get('token_sweep_rows',0)}**, sign match: **{fmt(controls.get('token_sweep_sign_match_rate'))}**\n\n")
    lines.append(f"Causal path rows: **{controls.get('causal_path_rows',0)}**, recovery rows: **{controls.get('causal_path_recovery_rows',0)}**\n\n")
    lines.append(f"Recovery target-Y mean: **{fmt(controls.get('recovery_target_Y_recovery_mean'))}**, logit recovery mean: **{fmt(controls.get('recovery_logit_recovery_mean'))}**\n\n")
    lines.append("### Strongest token-control effects\n")
    lines.append(table_md(controls.get("token_sweep_top_effects", []), max_rows=20))
    lines.append("### Causal recovery examples\n")
    lines.append(table_md(controls.get("recovery_examples", []), max_rows=20))

    lines.append("## 6. Patch-based logit attribution\n")
    lines.append(f"Rows: **{controls.get('logit_patch_attr_rows',0)}**\n\n")
    lines.append("### Top positive causal contributions\n")
    lines.append(table_md(controls.get("logit_patch_top_positive", []), max_rows=15))
    lines.append("### Top negative causal contributions\n")
    lines.append(table_md(controls.get("logit_patch_top_negative", []), max_rows=15))

    lines.append("## 7. MLP operator/Jacobian summary\n")
    if mlp.get("present"):
        lines.append(f"Rows: **{mlp.get('rows')}**\n\n")
        lines.append("| metric | mean | median | p90 |\n|---|---:|---:|---:|\n")
        for c in ["mlp_rec_rel", "J_norm", "J_rank90", "J_rank95", "J_rank99", "out_norm", "no_gate_delta_rel", "gate_only_delta_rel"]:
            if c + "_mean" in mlp or c + "_median" in mlp:
                lines.append(f"| {c} | {fmt(mlp.get(c+'_mean'))} | {fmt(mlp.get(c+'_median'))} | {fmt(mlp.get(c+'_p90'))} |\n")
        for k in ["top_out_norm", "top_J_norm", "top_no_gate_delta_rel"]:
            if mlp.get(k):
                lines.append(f"\n### {k}\n")
                lines.append(table_md(mlp.get(k, []), max_rows=15))
    else:
        lines.append("No MLP summary found.\n")

    lines.append("## 8. Baselines\n")
    lines.append(f"Baseline rows: **{controls.get('baseline_rows',0)}**\n\n")
    lines.append(f"Content-only QK A_rel mean: **{fmt(controls.get('baseline_qk_content_A_rel_mean'))}**\n\n")
    lines.append(f"Content-only QK Y_rel mean: **{fmt(controls.get('baseline_qk_content_Y_rel_mean'))}**\n\n")
    lines.append(f"Head ablation Y_rel mean: **{fmt(controls.get('baseline_head_ablation_Y_rel_mean'))}**\n\n")

    lines.append("## 9. Generation trace\n")
    if gen.get("present"):
        lines.append(f"Generation trace files: **{gen.get('files')}**\n\n")
        if gen.get("steps"):
            lines.append(table_md(gen.get("steps", []), max_rows=20))
    else:
        lines.append("No generation trace found.\n")

    lines.append("## 10. Remaining work\n")
    lines.append("```python\n")
    lines.append("1. Merge generation trace + patch-based logit attribution into a per-token why-token report.\n")
    lines.append("2. Expand token-control sweep to more heads/tasks/tokens.\n")
    lines.append("3. Build MLP operator dictionary: neuron clusters, output directions, route labels, patch groups.\n")
    lines.append("4. Add external baselines if needed: TransformerLens, SAE.\n")
    lines.append("5. Convert this report to an interactive HTML dashboard.\n")
    lines.append("```\n")
    return "".join(lines)


def md_to_html(md: str, title: str = "Matrix Pseudocode Report") -> str:
    # Minimal self-contained HTML: preserve Markdown in <pre> plus quick CSS. Avoid markdown dependency.
    escaped = html.escape(md)
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; line-height: 1.35; }}
pre {{ white-space: pre-wrap; background: #111; color: #eee; padding: 16px; border-radius: 8px; overflow-x: auto; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>This HTML preserves the generated Markdown report. Open the .md file for GitHub-style tables.</p>
<pre>{escaped}</pre>
</body></html>"""


def build_mlp_dictionary(out_dir: Path, mlp: Dict[str, Any]) -> str:
    lines = ["# MLP Operator Dictionary v1\n\n"]
    lines.append("This is a first dictionary layer. It groups MLP layers by local operator/Jacobian profile and output scale.\n\n")
    if not mlp.get("present"):
        lines.append("No MLP operator data found.\n")
        return "".join(lines)
    lines.append("## Rank profile\n\n")
    lines.append(f"Mean J_rank90: **{fmt(mlp.get('J_rank90_mean'))}**\n\n")
    lines.append(f"Mean J_rank95: **{fmt(mlp.get('J_rank95_mean'))}**\n\n")
    lines.append(f"Mean J_rank99: **{fmt(mlp.get('J_rank99_mean'))}**\n\n")
    lines.append("## High-output MLP layers\n\n")
    lines.append(table_md(mlp.get("top_out_norm", []), max_rows=20))
    lines.append("## High-Jacobian MLP layers\n\n")
    lines.append(table_md(mlp.get("top_J_norm", []), max_rows=20))
    lines.append("## Next dictionary pass\n\n")
    lines.append("```python\n")
    lines.append("For each high-output/high-J layer:\n")
    lines.append("  1. cluster top neurons by W_down output direction\n")
    lines.append("  2. cluster by gate activation pattern across prompts\n")
    lines.append("  3. fit dynamic operator: W_down @ diag(gate_effect(x)) @ W_up/gate terms\n")
    lines.append("  4. patch neuron groups and measure logits/KL/loss\n")
    lines.append("```\n")
    return "".join(lines)


def build_next_commands(out_dir: Path) -> str:
    script_v4 = "qwen_pseudocode_controls_v4_full.py"
    lines = ["#!/usr/bin/env bash\n", "set -euo pipefail\n\n"]
    lines.append("# Extended token-control sweeps for stronger directional-control evidence\n")
    heads_terms = [
        ("3:6", "k_affine", "L3H6_k"),
        ("4:2", "content", "L4H2_content"),
        ("16:1", "k_affine", "L16H1_k"),
        ("11:11", "k_affine,vo_bias", "L11H11_k_vobias"),
        ("4:5", "content", "L4H5_content"),
    ]
    for head, terms, name in heads_terms:
        out = f"controls_v5_token_sweep_{name}"
        lines.append(textwrap.dedent(f"""
        rm -rf ./{out}
        PYTHONUNBUFFERED=1 python {script_v4} \\
          --mode token_control_sweep \\
          --model Qwen/Qwen2.5-0.5B-Instruct \\
          --device cuda --dtype fp16 --attn-implementation eager \\
          --source-head {head} \\
          --patch-terms {terms} \\
          --key-positions 0,1,2,3,4,8,last,top \\
          --patch-strength 1.0 \\
          --prompt-suites all --prompts-per-suite 8 --max-length 192 \\
          --out-dir ./{out} 2>&1 | tee ./{out}.log
        """))
    lines.append("\n# More causal path recovery pairs\n")
    pairs = [("15:5", "16:3"), ("10:12", "11:0"), ("10:7", "11:1"), ("2:1", "3:6"), ("3:6", "4:8")]
    for a, b in pairs:
        out = f"controls_v5_recovery_L{a.replace(':','H')}_to_L{b.replace(':','H')}"
        lines.append(textwrap.dedent(f"""
        rm -rf ./{out}
        PYTHONUNBUFFERED=1 python {script_v4} \\
          --mode causal_path_recovery \\
          --model Qwen/Qwen2.5-0.5B-Instruct \\
          --device cuda --dtype fp16 --attn-implementation eager \\
          --source-head {a} --target-head {b} \\
          --patch-terms k_affine,content,vo_bias \\
          --patch-strength 1.0 \\
          --prompt-suites all --prompts-per-suite 8 --max-length 192 \\
          --out-dir ./{out} 2>&1 | tee ./{out}.log
        """))
    lines.append("\n# Patch-based logit attribution for broader head set\n")
    lines.append(textwrap.dedent("""
    rm -rf ./controls_v5_logit_patch_attr_broad
    PYTHONUNBUFFERED=1 python qwen_pseudocode_controls_v4_full.py \
      --mode logit_patch_attr \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --device cuda --dtype fp16 --attn-implementation eager \
      --head-spec 23:1,23:4,21:9,23:8,15:6,14:1,3:6,4:8,4:2,16:1,11:11,15:5,16:3,10:12,11:0 \
      --layers all \
      --prompt "Write a Python function that reverses a linked list." \
      --max-length 192 --report-topk 120 \
      --out-dir ./controls_v5_logit_patch_attr_broad 2>&1 | tee ./controls_v5_logit_patch_attr_broad.log
    """))
    lines.append("\n# Build final report from collected artifacts\n")
    lines.append(textwrap.dedent("""
    python qwen_pseudocode_final_product_v5.py \
      --atlas ./atlas_full_all \
      --controls-v4 ./controls_v4_results.zip \
      --controls-v3 ./controls_v3_results.zip \
      --baselines ./controls_baselines_top_heads_fixed.zip \
      --out-dir ./final_pseudocode_product_v5
    """))
    return "".join(lines)


# ------------------------- main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="", help="atlas_full_all dir or zip")
    ap.add_argument("--controls-v4", default="", help="controls_v4_results dir or zip")
    ap.add_argument("--controls-v3", default="", help="controls_v3_results dir or zip")
    ap.add_argument("--baselines", default="", help="baselines dir or zip")
    ap.add_argument("--generation", default="", help="generation trace dir or zip, optional if not inside atlas")
    ap.add_argument("--out-dir", default="./final_pseudocode_product_v5")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    work = out_dir / "_staged_inputs"
    if work.exists():
        shutil.rmtree(work)
    ensure_dir(work)

    atlas = stage_input(args.atlas, work, "atlas")
    cv4 = stage_input(args.controls_v4, work, "controls_v4")
    cv3 = stage_input(args.controls_v3, work, "controls_v3")
    base = stage_input(args.baselines, work, "baselines")
    gen_in = stage_input(args.generation, work, "generation")

    # controls summary merges v4, v3, baselines. v4 preferred but aggregate all.
    runtime = summarize_runtime(atlas)
    static = summarize_static(atlas)
    mlp = summarize_mlp(atlas, cv4)

    controls_all: Dict[str, Any] = {}
    for label, root in [("v4", cv4), ("v3", cv3), ("baselines", base)]:
        s = summarize_controls(root)
        controls_all[label] = s
    # merged controls: prefer v4 values, but supplement baselines/v3 counts.
    controls = dict(controls_all.get("v4", {}))
    for src in [controls_all.get("v3", {}), controls_all.get("baselines", {})]:
        for k, v in src.items():
            if k not in controls or controls.get(k) in (None, 0, False, ""):
                controls[k] = v
            elif isinstance(v, (int, float)) and k.endswith("_rows"):
                controls[k] = max(controls.get(k, 0), v)
    # Baseline metrics should come from fixed baseline if provided.
    bsum = controls_all.get("baselines", {})
    for k, v in bsum.items():
        if k.startswith("baseline_"):
            controls[k] = v

    gen = summarize_generation(gen_in or atlas or cv4)
    readiness = build_readiness(runtime, static, mlp, controls, gen)

    summary = {
        "runtime": runtime,
        "static": static,
        "mlp": mlp,
        "controls": controls,
        "controls_by_source": controls_all,
        "generation": gen,
        "readiness": readiness,
    }
    (out_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = build_markdown(out_dir, runtime, static, mlp, controls, gen, readiness)
    (out_dir / "FINAL_MATRIX_PSEUDOCODE_REPORT.md").write_text(md, encoding="utf-8")
    (out_dir / "FINAL_MATRIX_PSEUDOCODE_REPORT.html").write_text(md_to_html(md), encoding="utf-8")

    mlp_md = build_mlp_dictionary(out_dir, mlp)
    (out_dir / "MLP_OPERATOR_DICTIONARY_v1.md").write_text(mlp_md, encoding="utf-8")

    commands = build_next_commands(out_dir)
    cmd_path = out_dir / "NEXT_EXTENDED_RUNS.sh"
    cmd_path.write_text(commands, encoding="utf-8")
    os.chmod(cmd_path, 0o755)

    # Write lightweight status file.
    status_lines = ["# Product readiness checklist\n\n", table_md(readiness, max_rows=50)]
    (out_dir / "PRODUCT_READINESS_CHECKLIST.md").write_text("".join(status_lines), encoding="utf-8")

    print(f"[done] wrote {out_dir}")
    print(f"  - FINAL_MATRIX_PSEUDOCODE_REPORT.md")
    print(f"  - FINAL_MATRIX_PSEUDOCODE_REPORT.html")
    print(f"  - MLP_OPERATOR_DICTIONARY_v1.md")
    print(f"  - NEXT_EXTENDED_RUNS.sh")
    print(f"  - final_summary.json")


if __name__ == "__main__":
    main()
