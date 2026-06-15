#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
MAXLEN="${MAXLEN:-192}"
PROMPTS_PER_SUITE="${PROMPTS_PER_SUITE:-4}"

mkdir -p runs reports/latest/tables manifests/latest

echo "===== V7: wider token-control sweeps ====="

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --attn-implementation "$ATTN_IMPL" \
  --source-head 3:6 --patch-terms k_affine \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite "$PROMPTS_PER_SUITE" \
  --max-length "$MAXLEN" \
  --out-dir runs/v7_token_sweep_L3H6_k \
  2>&1 | tee runs/v7_token_sweep_L3H6_k.log

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --attn-implementation "$ATTN_IMPL" \
  --source-head 4:2 --patch-terms content \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite "$PROMPTS_PER_SUITE" \
  --max-length "$MAXLEN" \
  --out-dir runs/v7_token_sweep_L4H2_content \
  2>&1 | tee runs/v7_token_sweep_L4H2_content.log

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --attn-implementation "$ATTN_IMPL" \
  --source-head 16:1 --patch-terms k_affine \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite "$PROMPTS_PER_SUITE" \
  --max-length "$MAXLEN" \
  --out-dir runs/v7_token_sweep_L16H1_k \
  2>&1 | tee runs/v7_token_sweep_L16H1_k.log

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode token_control_sweep \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --attn-implementation "$ATTN_IMPL" \
  --source-head 11:11 --patch-terms vo_bias \
  --key-positions 0,1,2,3,4,8,last,top \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite "$PROMPTS_PER_SUITE" \
  --max-length "$MAXLEN" \
  --out-dir runs/v7_token_sweep_L11H11_vobias \
  2>&1 | tee runs/v7_token_sweep_L11H11_vobias.log

echo "===== V7: more causal path recovery pairs ====="

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --attn-implementation "$ATTN_IMPL" \
  --source-head 3:6 --target-head 4:8 \
  --patch-terms k_affine,content \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite "$PROMPTS_PER_SUITE" \
  --max-length "$MAXLEN" \
  --out-dir runs/v7_recovery_L3H6_to_L4H8 \
  2>&1 | tee runs/v7_recovery_L3H6_to_L4H8.log

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --attn-implementation "$ATTN_IMPL" \
  --source-head 2:1 --target-head 3:6 \
  --patch-terms vo_bias \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite "$PROMPTS_PER_SUITE" \
  --max-length "$MAXLEN" \
  --out-dir runs/v7_recovery_L2H1_to_L3H6 \
  2>&1 | tee runs/v7_recovery_L2H1_to_L3H6.log

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode causal_path_recovery \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --attn-implementation "$ATTN_IMPL" \
  --source-head 15:5 --target-head 16:3 \
  --patch-terms k_affine,content \
  --patch-strength 1.0 \
  --prompt-suites all --prompts-per-suite "$PROMPTS_PER_SUITE" \
  --max-length "$MAXLEN" \
  --out-dir runs/v7_recovery_L15H5_to_L16H3 \
  2>&1 | tee runs/v7_recovery_L15H5_to_L16H3.log

echo "===== V7: logit attribution on python/math/text prompts ====="

for NAME_PROMPT in \
  "python|Write a Python function that reverses a linked list." \
  "math|Solve step by step: if x + 7 = 19, what is x?" \
  "text|Explain why the sky appears blue in simple words."
do
  NAME="${NAME_PROMPT%%|*}"
  PROMPT="${NAME_PROMPT#*|}"

  PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
    --mode logit_patch_attr \
    --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --attn-implementation "$ATTN_IMPL" \
    --head-spec 23:1,23:4,21:9,23:8,15:6,14:1,3:6,4:8,4:2,16:1,11:11 \
    --layers all \
    --prompt "$PROMPT" \
    --max-length "$MAXLEN" \
    --report-topk 100 \
    --out-dir "runs/v7_logit_attr_${NAME}" \
    2>&1 | tee "runs/v7_logit_attr_${NAME}.log"
done

echo "===== V7: wider MLP operator ====="

PYTHONUNBUFFERED=1 python tools/qwen_pseudocode_controls_v4_full.py \
  --mode mlp_operator \
  --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --attn-implementation "$ATTN_IMPL" \
  --layers all \
  --prompt-suites all \
  --prompts-per-suite "$PROMPTS_PER_SUITE" \
  --max-length "$MAXLEN" \
  --mlp-top-neurons 48 \
  --out-dir runs/v7_mlp_operator_wide \
  2>&1 | tee runs/v7_mlp_operator_wide.log

echo "===== V7: collect lightweight reports ====="

cp -f runs/v7_token_sweep_L3H6_k/token_control_sweep/token_control_sweep_summary.csv reports/latest/tables/v7_token_sweep_L3H6_k.csv 2>/dev/null || true
cp -f runs/v7_token_sweep_L4H2_content/token_control_sweep/token_control_sweep_summary.csv reports/latest/tables/v7_token_sweep_L4H2_content.csv 2>/dev/null || true
cp -f runs/v7_token_sweep_L16H1_k/token_control_sweep/token_control_sweep_summary.csv reports/latest/tables/v7_token_sweep_L16H1_k.csv 2>/dev/null || true
cp -f runs/v7_token_sweep_L11H11_vobias/token_control_sweep/token_control_sweep_summary.csv reports/latest/tables/v7_token_sweep_L11H11_vobias.csv 2>/dev/null || true

cp -f runs/v7_recovery_L3H6_to_L4H8/causal_path_recovery/causal_path_recovery_summary.csv reports/latest/tables/v7_recovery_L3H6_to_L4H8.csv 2>/dev/null || true
cp -f runs/v7_recovery_L2H1_to_L3H6/causal_path_recovery/causal_path_recovery_summary.csv reports/latest/tables/v7_recovery_L2H1_to_L3H6.csv 2>/dev/null || true
cp -f runs/v7_recovery_L15H5_to_L16H3/causal_path_recovery/causal_path_recovery_summary.csv reports/latest/tables/v7_recovery_L15H5_to_L16H3.csv 2>/dev/null || true

cp -f runs/v7_logit_attr_python/logit_patch_attr/logit_patch_attribution.csv reports/latest/tables/v7_logit_attr_python.csv 2>/dev/null || true
cp -f runs/v7_logit_attr_math/logit_patch_attr/logit_patch_attribution.csv reports/latest/tables/v7_logit_attr_math.csv 2>/dev/null || true
cp -f runs/v7_logit_attr_text/logit_patch_attr/logit_patch_attribution.csv reports/latest/tables/v7_logit_attr_text.csv 2>/dev/null || true

cp -f runs/v7_mlp_operator_wide/mlp_operator/mlp_operator_summary.csv reports/latest/tables/v7_mlp_operator_wide.csv 2>/dev/null || true

python - <<'PY'
import csv, json
from pathlib import Path

tables = sorted(Path("reports/latest/tables").glob("v7_*.csv"))
summary = {"tables": []}

for p in tables:
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    item = {"file": str(p), "rows": len(rows)}
    for key in [
        "sign_match",
        "target_Y_recovery_ratio",
        "logit_recovery_ratio",
        "causal_logit_contribution",
        "J_rank90",
        "J_rank95",
        "J_rank99",
    ]:
        vals = []
        for r in rows:
            try:
                vals.append(float(str(r.get(key, "")).strip()))
            except Exception:
                pass
        if vals:
            item[key + "_mean"] = sum(vals) / len(vals)
    summary["tables"].append(item)

Path("manifests/latest").mkdir(parents=True, exist_ok=True)
Path("manifests/latest/wide_validation_v7_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

md = ["# Wide validation v7 report\n\n"]
md.append("Expanded validation: more token-control heads, more causal paths, more prompt types for logit attribution, wider MLP operator run.\n\n")
md.append("| table | rows | means |\n| --- | ---: | --- |\n")
for item in summary["tables"]:
    means = ", ".join(
        f"{k}={v:.4g}" for k, v in item.items() if k.endswith("_mean")
    )
    md.append(f"| `{Path(item['file']).name}` | {item['rows']} | {means} |\n")

Path("reports/latest/WIDE_VALIDATION_V7_REPORT.md").write_text(
    "".join(md),
    encoding="utf-8",
)
print("[v7] wrote reports/latest/WIDE_VALIDATION_V7_REPORT.md")
PY

git add scripts/RUN_WIDE_VALIDATION_V7_AND_PUBLISH.sh reports/latest manifests/latest

git commit -m "Add wide validation v7 summaries" || echo "Nothing to commit"

git push

echo "===== DONE V7 ====="
echo "Open: reports/latest/WIDE_VALIDATION_V7_REPORT.md"
