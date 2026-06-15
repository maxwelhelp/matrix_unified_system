#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
ATTN_IMPL="${ATTN_IMPL:-eager}"

mkdir -p runs reports/latest/tables manifests/latest

run_one () {
  NAME="$1"
  PROMPT="$2"
  OUT="runs/deep_v8_${NAME}"

  rm -rf "$OUT"

  PYTHONUNBUFFERED=1 python tools/qwen_deep_improvement_v8.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --attn-implementation "$ATTN_IMPL" \
    --prompt "$PROMPT" \
    --generate-steps 4 \
    --max-length 192 \
    --heads 23:1,23:4,21:9,23:8,3:6,4:8,16:1,11:11 \
    --mlp-layers all \
    --mlp-top-k 32 \
    --patch-position last \
    --out-dir "$OUT" \
    2>&1 | tee "${OUT}.log"

  cp -f "$OUT/WHY_TOKEN_DEEP_REPORT.md" "reports/latest/WHY_TOKEN_DEEP_REPORT_${NAME}.md"
  cp -f "$OUT/MLP_GROUP_PATCH_REPORT.md" "reports/latest/MLP_GROUP_PATCH_REPORT_${NAME}.md"
  cp -f "$OUT/deep_v8_summary.json" "manifests/latest/deep_v8_${NAME}_summary.json"
  cp -f "$OUT/tables/deep_patch_attribution.csv" "reports/latest/tables/deep_v8_${NAME}_patch_attribution.csv"
  cp -f "$OUT/tables/mlp_neuron_groups.csv" "reports/latest/tables/deep_v8_${NAME}_mlp_neuron_groups.csv"
  cp -f "$OUT/tables/deep_generation_steps.csv" "reports/latest/tables/deep_v8_${NAME}_generation_steps.csv"
}

run_one python "Write a Python function that reverses a linked list."
run_one math "Solve step by step: if x + 7 = 19, what is x?"
run_one text "Explain why the sky appears blue in simple words."

python - <<'PY'
import json, csv
from pathlib import Path

summary = {"runs": []}

for name in ["python", "math", "text"]:
    p = Path(f"manifests/latest/deep_v8_{name}_summary.json")
    if not p.exists():
        summary["runs"].append({"name": name, "missing": True})
        continue
    obj = json.loads(p.read_text(encoding="utf-8"))
    item = {
        "name": name,
        "n_rows": obj.get("n_rows"),
        "n_heads": obj.get("n_heads"),
        "n_mlp_layers": obj.get("n_mlp_layers"),
        "steps": obj.get("steps", []),
        "top_positive": obj.get("top_positive", [])[:10],
        "top_negative": obj.get("top_negative", [])[:10],
    }
    summary["runs"].append(item)

Path("manifests/latest/deep_v8_final_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

md = ["# Deep v8 final report\n\n"]
md.append("This is the final maximum validation layer: per-generated-token head patch + MLP neuron-group patch.\n\n")

for run in summary["runs"]:
    md.append(f"## {run['name']}\n\n")
    if run.get("missing"):
        md.append("Missing.\n\n")
        continue
    md.append(f"rows={run.get('n_rows')} heads={run.get('n_heads')} mlp_layers={run.get('n_mlp_layers')}\n\n")
    md.append("Generated steps:\n\n")
    md.append("| step | token | prob | base_logit |\n| --- | --- | ---: | ---: |\n")
    for s in run.get("steps", []):
        md.append(f"| {s.get('step')} | `{s.get('token')}` | {float(s.get('prob',0)):.4f} | {float(s.get('base_logit',0)):.4f} |\n")
    md.append("\nTop positive contributors:\n\n")
    md.append("| component | layer | head/group | token | Δlogit | KL |\n| --- | ---: | --- | --- | ---: | ---: |\n")
    for r in run.get("top_positive", [])[:10]:
        hg = r.get("head") if str(r.get("head","")) != "" else r.get("group")
        md.append(f"| {r.get('component')} | {r.get('layer')} | {hg} | `{r.get('token')}` | {float(r.get('causal_logit_contribution',0)):.4f} | {float(r.get('kl_orig_to_patch',0)):.4g} |\n")
    md.append("\nTop negative contributors:\n\n")
    md.append("| component | layer | head/group | token | Δlogit | KL |\n| --- | ---: | --- | --- | ---: | ---: |\n")
    for r in run.get("top_negative", [])[:10]:
        hg = r.get("head") if str(r.get("head","")) != "" else r.get("group")
        md.append(f"| {r.get('component')} | {r.get('layer')} | {hg} | `{r.get('token')}` | {float(r.get('causal_logit_contribution',0)):.4f} | {float(r.get('kl_orig_to_patch',0)):.4g} |\n")
    md.append("\n")

Path("reports/latest/DEEP_V8_FINAL_REPORT.md").write_text(
    "".join(md),
    encoding="utf-8",
)
PY

git add tools/qwen_deep_improvement_v8.py \
  scripts/RUN_DEEP_V8_FINAL_AND_PUBLISH.sh \
  reports/latest \
  manifests/latest

git commit -m "Add deep v8 final why-token and MLP group patch reports" || echo "Nothing to commit"

git push

echo "DONE. Main report:"
echo "reports/latest/DEEP_V8_FINAL_REPORT.md"
