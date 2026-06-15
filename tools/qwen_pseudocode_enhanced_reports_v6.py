#!/usr/bin/env python3
"""
Enhanced lightweight report builder for Matrix Pseudocode Decompiler.

This script does NOT read heavy zip artifacts. It reads the lightweight files that are
already published to GitHub under:
  - reports/latest/tables/*.csv
  - manifests/latest/*.json

It creates:
  - reports/latest/WHY_TOKEN_REPORT.md
  - reports/latest/MLP_OPERATOR_DICTIONARY_v2.md
  - reports/latest/ENHANCED_PRODUCT_SUMMARY.md
  - reports/latest/tables/why_token_top_contributors.csv
  - reports/latest/tables/mlp_layer_roles.csv
  - manifests/latest/enhanced_summary.json

Goal:
  Merge generation trace + patch-based logit attribution into a human-readable
  why-token report, and make MLP interpretation less poor by adding layer roles,
  gate sensitivity, Jacobian ranks, and top-neuron summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics as stats
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return default
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in {"n/a", "na", "none", "null", "nan"}:
        return default
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    v = safe_float(x, None)
    if v is None:
        return default
    return int(v)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt(x: Any, digits: int = 4) -> str:
    v = safe_float(x, None)
    if v is None:
        return "n/a"
    av = abs(v)
    if av != 0 and (av < 1e-3 or av >= 1e4):
        return f"{v:.3e}"
    return f"{v:.{digits}f}"


def mean(vals: Iterable[Any]) -> Optional[float]:
    xs = [safe_float(v, None) for v in vals]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def median(vals: Iterable[Any]) -> Optional[float]:
    xs = [safe_float(v, None) for v in vals]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(stats.median(xs))


def quantile(vals: Iterable[Any], q: float) -> Optional[float]:
    xs = [safe_float(v, None) for v in vals]
    xs = sorted([x for x in xs if x is not None])
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def first_present(row: Dict[str, Any], names: List[str], default: str = "") -> str:
    for n in names:
        if n in row and str(row[n]).strip() != "":
            return str(row[n])
    return default


def parse_neurons(s: Any, max_items: int = 12) -> List[int]:
    if s is None:
        return []
    txt = str(s)
    nums = re.findall(r"-?\d+", txt)
    out = []
    for n in nums:
        try:
            out.append(int(n))
        except Exception:
            pass
        if len(out) >= max_items:
            break
    return out


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def load_generation_steps(root: Path) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    summary = read_json(root / "manifests/latest/generation_trace_summary.json", {})
    if isinstance(summary, dict):
        candidates = []
        for key in ["steps", "generation_steps", "trace", "items"]:
            if isinstance(summary.get(key), list):
                candidates = summary[key]
                break
        if not candidates and all(k in summary for k in ["tokens"]):
            toks = summary.get("tokens") or []
            candidates = [{"step": i, "token": t} for i, t in enumerate(toks)]
        for i, item in enumerate(candidates):
            if isinstance(item, dict):
                steps.append(item)
            else:
                steps.append({"step": i, "token": str(item)})
    elif isinstance(summary, list):
        for i, item in enumerate(summary):
            if isinstance(item, dict):
                steps.append(item)
            else:
                steps.append({"step": i, "token": str(item)})

    # If local heavy generation files exist, prefer richer step_*.json.
    gen_dir = root / "runs/atlas_gen_trace_full_python/generation_trace"
    if gen_dir.exists():
        local = []
        for p in sorted(gen_dir.glob("step_*.json")):
            obj = read_json(p, {})
            if isinstance(obj, dict):
                obj.setdefault("source_file", str(p))
                local.append(obj)
        if local:
            return local
    return steps


def extract_step_token(step: Dict[str, Any], idx: int) -> Tuple[str, str, str]:
    token = first_present(step, ["token", "next_token", "generated_token", "selected_token", "decoded_token"], "")
    token_id = first_present(step, ["token_id", "next_token_id", "selected_token_id"], "")
    prob = first_present(step, ["prob", "next_prob", "selected_prob", "p"], "")
    if not token:
        # Search shallow nested top token fields.
        for k, v in step.items():
            if isinstance(v, dict):
                token = first_present(v, ["token", "text", "decoded", "next_token"], token)
                token_id = first_present(v, ["token_id", "id"], token_id)
                prob = first_present(v, ["prob", "p"], prob)
    return token or f"step_{idx}", token_id, prob


def build_why_token(root: Path) -> Dict[str, Any]:
    out_dir = root / "reports/latest"
    table_dir = out_dir / "tables"
    rows = read_csv(table_dir / "logit_patch_attribution.csv")
    steps = load_generation_steps(root)

    def contrib(row: Dict[str, Any]) -> float:
        return safe_float(row.get("causal_logit_contribution"), 0.0) or 0.0

    rows2 = []
    for r in rows:
        c = contrib(r)
        rows2.append({
            "component": first_present(r, ["component"], ""),
            "layer": first_present(r, ["layer"], ""),
            "head": first_present(r, ["head"], ""),
            "causal_logit_contribution": c,
            "logit_rel": safe_float(r.get("logit_rel"), None),
            "top1_match": safe_float(r.get("top1_match"), None),
            "raw": r,
        })
    pos = sorted(rows2, key=lambda r: r["causal_logit_contribution"], reverse=True)[:20]
    neg = sorted(rows2, key=lambda r: r["causal_logit_contribution"])[:20]

    # Write compact CSV used by GitHub summaries.
    flat = []
    for rank, r in enumerate(pos, 1):
        flat.append({
            "side": "positive",
            "rank": rank,
            "component": r["component"],
            "layer": r["layer"],
            "head": r["head"],
            "causal_logit_contribution": r["causal_logit_contribution"],
            "logit_rel": r["logit_rel"],
            "top1_match": r["top1_match"],
        })
    for rank, r in enumerate(neg, 1):
        flat.append({
            "side": "negative",
            "rank": rank,
            "component": r["component"],
            "layer": r["layer"],
            "head": r["head"],
            "causal_logit_contribution": r["causal_logit_contribution"],
            "logit_rel": r["logit_rel"],
            "top1_match": r["top1_match"],
        })
    write_csv(table_dir / "why_token_top_contributors.csv", flat)

    md = []
    md.append("# Why-token report v1\n")
    md.append("This report merges generation trace summaries with patch-based logit attribution. It is lightweight and can be committed to GitHub.\n")
    md.append("## Generated-token trace\n")
    if steps:
        step_rows = []
        for i, step in enumerate(steps[:20]):
            token, token_id, prob = extract_step_token(step, i)
            step_rows.append([i, token, token_id or "", prob or ""])
        md.append(md_table(["step", "token", "token_id", "prob"], step_rows))
    else:
        md.append("_No generation trace summary was found. Run generate_trace and publish latest summaries._\n")

    md.append("\n## Patch-based positive causal contributors\n")
    md.append(md_table(
        ["rank", "component", "layer", "head", "Δlogit", "logit_rel", "top1_match"],
        [[i+1, r["component"], r["layer"], r["head"], fmt(r["causal_logit_contribution"]), fmt(r["logit_rel"]), fmt(r["top1_match"])] for i, r in enumerate(pos[:15])]
    ))
    md.append("\n## Patch-based negative/suppressing contributors\n")
    md.append(md_table(
        ["rank", "component", "layer", "head", "Δlogit", "logit_rel", "top1_match"],
        [[i+1, r["component"], r["layer"], r["head"], fmt(r["causal_logit_contribution"]), fmt(r["logit_rel"]), fmt(r["top1_match"])] for i, r in enumerate(neg[:15])]
    ))

    # Simple interpretation buckets.
    comp_counter = Counter(r["component"] for r in rows2 if r["component"])
    pos_mlp = [r for r in pos if r["component"] == "mlp"]
    pos_head = [r for r in pos if r["component"] == "head"]
    neg_mlp = [r for r in neg if r["component"] == "mlp"]
    neg_head = [r for r in neg if r["component"] == "head"]

    md.append("\n## Interpretation\n")
    md.append("```python\n")
    md.append("why_token = {\n")
    md.append(f"  'available_patch_rows': {len(rows2)},\n")
    md.append(f"  'positive_mlp_count_top20': {len(pos_mlp)},\n")
    md.append(f"  'positive_head_count_top20': {len(pos_head)},\n")
    md.append(f"  'negative_mlp_count_top20': {len(neg_mlp)},\n")
    md.append(f"  'negative_head_count_top20': {len(neg_head)},\n")
    md.append("}\n")
    md.append("```\n")
    md.append("Use this report as the human-readable layer above raw patch tables. For stronger per-token explanations, run logit patch attribution for every generated token, not only the first/selected target.\n")

    (out_dir / "WHY_TOKEN_REPORT.md").write_text("".join(md), encoding="utf-8")

    return {
        "logit_patch_rows": len(rows2),
        "generation_steps_found": len(steps),
        "top_positive": flat[:10],
        "top_negative": [r for r in flat if r["side"] == "negative"][:10],
        "component_counts": dict(comp_counter),
    }


def build_mlp_dictionary(root: Path) -> Dict[str, Any]:
    out_dir = root / "reports/latest"
    table_dir = out_dir / "tables"
    rows = read_csv(table_dir / "mlp_operator_summary.csv")
    if not rows:
        rows = read_csv(table_dir / "all_layers_mlp_summary.csv")

    by_layer: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        layer = safe_int(r.get("layer"), None)
        if layer is not None:
            by_layer[layer].append(r)

    all_out = [safe_float(r.get("out_norm"), None) for r in rows]
    all_j = [safe_float(r.get("J_norm"), None) for r in rows]
    all_gate = [safe_float(r.get("no_gate_delta_rel"), None) for r in rows]
    all_rank90 = [safe_float(r.get("J_rank90"), None) for r in rows]
    out_p90 = quantile(all_out, 0.90) or 0.0
    j_p90 = quantile(all_j, 0.90) or 0.0
    gate_p90 = quantile(all_gate, 0.90) or 0.0
    rank90_p25 = quantile(all_rank90, 0.25) or 0.0

    layer_rows = []
    for layer in sorted(by_layer):
        rs = by_layer[layer]
        out_m = mean(r.get("out_norm") for r in rs)
        j_m = mean(r.get("J_norm") for r in rs)
        r90_m = mean(r.get("J_rank90") for r in rs)
        r99_m = mean(r.get("J_rank99") for r in rs)
        gate_m = mean(r.get("no_gate_delta_rel") for r in rs)
        gate_only_m = mean(r.get("gate_only_delta_rel") for r in rs)
        # Frequent top neurons.
        cnt = Counter()
        for r in rs:
            for n in parse_neurons(r.get("top_neurons"), 32):
                cnt[n] += 1
        top_neurons = [n for n, _ in cnt.most_common(12)]

        roles = []
        if out_m is not None and out_m >= out_p90:
            roles.append("strong_residual_writer")
        if j_m is not None and j_m >= j_p90:
            roles.append("high_jacobian_transformer")
        if gate_m is not None and gate_m >= gate_p90:
            roles.append("gate_sensitive")
        if r90_m is not None and r90_m <= rank90_p25 and j_m is not None and j_m >= (median(all_j) or 0.0):
            roles.append("compact_high_effect_operator")
        if not roles:
            roles.append("mixed_mlp_operator")

        layer_rows.append({
            "layer": layer,
            "roles": ",".join(roles),
            "out_norm_mean": out_m,
            "J_norm_mean": j_m,
            "J_rank90_mean": r90_m,
            "J_rank99_mean": r99_m,
            "no_gate_delta_rel_mean": gate_m,
            "gate_only_delta_rel_mean": gate_only_m,
            "top_neurons": top_neurons,
            "n_rows": len(rs),
        })

    write_csv(table_dir / "mlp_layer_roles.csv", layer_rows, [
        "layer", "roles", "out_norm_mean", "J_norm_mean", "J_rank90_mean", "J_rank99_mean",
        "no_gate_delta_rel_mean", "gate_only_delta_rel_mean", "top_neurons", "n_rows"
    ])

    md = []
    md.append("# MLP operator dictionary v2\n")
    md.append("This is a richer lightweight MLP dictionary built from local operator/Jacobian summaries. It is still not as semantically rich as attention pseudocode, but it gives layer roles, gate sensitivity, rank profile, and top-neuron groups.\n")
    md.append("## Role definitions\n")
    md.append("```python\n")
    md.append("strong_residual_writer      = layer has high MLP output norm\n")
    md.append("high_jacobian_transformer   = local MLP Jacobian has high norm\n")
    md.append("gate_sensitive              = removing/altering gate strongly changes output\n")
    md.append("compact_high_effect_operator= relatively compact rank with high effect\n")
    md.append("mixed_mlp_operator          = no single role dominates\n")
    md.append("```\n")
    md.append("\n## Layer role table\n")
    md.append(md_table(
        ["layer", "roles", "out", "J", "rank90", "rank99", "no_gate", "top_neurons"],
        [[r["layer"], r["roles"], fmt(r["out_norm_mean"]), fmt(r["J_norm_mean"]), fmt(r["J_rank90_mean"]), fmt(r["J_rank99_mean"]), fmt(r["no_gate_delta_rel_mean"]), str(r["top_neurons"][:8])] for r in layer_rows]
    ))

    md.append("\n## Strongest MLP candidates\n")
    for title, key in [
        ("Strong residual writers", "out_norm_mean"),
        ("High-Jacobian transformers", "J_norm_mean"),
        ("Gate-sensitive layers", "no_gate_delta_rel_mean"),
    ]:
        top = sorted(layer_rows, key=lambda r: safe_float(r.get(key), -1e9) or -1e9, reverse=True)[:10]
        md.append(f"\n### {title}\n")
        md.append(md_table(
            ["layer", "roles", key, "top_neurons"],
            [[r["layer"], r["roles"], fmt(r.get(key)), str(r["top_neurons"][:10])] for r in top]
        ))

    md.append("\n## Next MLP upgrade\n")
    md.append("```python\n")
    md.append("1. cluster top neurons by output direction W_down[:, neuron]\n")
    md.append("2. patch neuron groups and measure Δlogit / KL / top1\n")
    md.append("3. label groups as code-token writers, syntax suppressors, BOS/template writers, etc.\n")
    md.append("4. connect MLP groups to attention pseudocode paths in why-token report\n")
    md.append("```\n")

    (out_dir / "MLP_OPERATOR_DICTIONARY_v2.md").write_text("".join(md), encoding="utf-8")

    role_counts = Counter()
    for r in layer_rows:
        for role in str(r["roles"]).split(','):
            role_counts[role] += 1
    return {
        "mlp_rows": len(rows),
        "mlp_layers": len(layer_rows),
        "role_counts": dict(role_counts),
        "top_layers_by_out_norm": sorted(layer_rows, key=lambda r: safe_float(r.get("out_norm_mean"), -1e9) or -1e9, reverse=True)[:5],
        "top_layers_by_gate_sensitivity": sorted(layer_rows, key=lambda r: safe_float(r.get("no_gate_delta_rel_mean"), -1e9) or -1e9, reverse=True)[:5],
    }


def build_enhanced_summary(root: Path, why: Dict[str, Any], mlp: Dict[str, Any]) -> None:
    out_dir = root / "reports/latest"
    manifest_dir = root / "manifests/latest"
    final_summary = read_json(manifest_dir / "final_summary.json", {})
    checklist = (out_dir / "PRODUCT_READINESS_CHECKLIST.md").read_text(encoding="utf-8") if (out_dir / "PRODUCT_READINESS_CHECKLIST.md").exists() else ""

    enhanced = {
        "why_token": why,
        "mlp_dictionary_v2": mlp,
        "final_summary_present": bool(final_summary),
        "checklist_has_na": "n/a" in checklist.lower(),
        "recommended_next_experiments": [
            "run patch-based logit attribution for every generated token, not only the first target",
            "run token-control sweep on 20-50 heads across code/math/text prompts",
            "cluster MLP top neurons by W_down output direction and patch neuron groups",
            "connect causal path recovery pairs to why-token reports",
        ],
    }
    write_json(manifest_dir / "enhanced_summary.json", enhanced)

    md = []
    md.append("# Enhanced product summary\n")
    md.append("This file is generated from lightweight GitHub-published summaries.\n")
    md.append("## Status\n")
    md.append("```python\n")
    md.append(f"checklist_has_na = {enhanced['checklist_has_na']}\n")
    md.append(f"why_token_rows = {why.get('logit_patch_rows')}\n")
    md.append(f"generation_steps_found = {why.get('generation_steps_found')}\n")
    md.append(f"mlp_layers = {mlp.get('mlp_layers')}\n")
    md.append(f"mlp_role_counts = {mlp.get('role_counts')}\n")
    md.append("```\n")
    md.append("## Generated files\n")
    md.append("- `reports/latest/WHY_TOKEN_REPORT.md`\n")
    md.append("- `reports/latest/MLP_OPERATOR_DICTIONARY_v2.md`\n")
    md.append("- `reports/latest/tables/why_token_top_contributors.csv`\n")
    md.append("- `reports/latest/tables/mlp_layer_roles.csv`\n")
    md.append("- `manifests/latest/enhanced_summary.json`\n")
    md.append("## Next experiments\n")
    for i, item in enumerate(enhanced["recommended_next_experiments"], 1):
        md.append(f"{i}. {item}\n")
    (out_dir / "ENHANCED_PRODUCT_SUMMARY.md").write_text("".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    (root / "reports/latest/tables").mkdir(parents=True, exist_ok=True)
    (root / "manifests/latest").mkdir(parents=True, exist_ok=True)

    why = build_why_token(root)
    mlp = build_mlp_dictionary(root)
    build_enhanced_summary(root, why, mlp)
    print("[done] wrote enhanced reports:")
    print("  reports/latest/WHY_TOKEN_REPORT.md")
    print("  reports/latest/MLP_OPERATOR_DICTIONARY_v2.md")
    print("  reports/latest/ENHANCED_PRODUCT_SUMMARY.md")
    print("  manifests/latest/enhanced_summary.json")


if __name__ == "__main__":
    main()
