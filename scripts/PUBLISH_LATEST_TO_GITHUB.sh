#!/usr/bin/env bash
set -euo pipefail

# Publish only lightweight latest summaries/reports to GitHub.
# Heavy artifacts are intentionally ignored by .gitignore:
#   runs/, results_zips/, *.zip, *.pt, *.pth, *.safetensors
#
# Usage:
#   bash scripts/PUBLISH_LATEST_TO_GITHUB.sh
#   bash scripts/PUBLISH_LATEST_TO_GITHUB.sh --no-push
#   bash scripts/PUBLISH_LATEST_TO_GITHUB.sh --message "Update latest Qwen run summaries"

PUSH=1
COMMIT_MESSAGE="Update latest pseudocode experiment summaries"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-push)
      PUSH=0
      shift
      ;;
    --message|-m)
      COMMIT_MESSAGE="${2:-$COMMIT_MESSAGE}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

mkdir -p reports/latest/tables manifests/latest

# Prefer the no-N/A fixed report if it exists.
FINAL_DIR=""
for cand in \
  "runs/final_pseudocode_product_v5_FULL_NO_NA" \
  "runs/final_product_NO_NA_fixed" \
  "runs/final_pseudocode_product_v5_FULL" \
  "final_product/final_pseudocode_product_v5_FULL" \
  "final_product/final_pseudocode_product_v5_previous"; do
  if [[ -d "$cand" ]]; then
    FINAL_DIR="$cand"
    break
  fi
done

if [[ -z "$FINAL_DIR" ]]; then
  echo "[warn] No final product dir found. Expected one of:" >&2
  echo "       runs/final_pseudocode_product_v5_FULL_NO_NA" >&2
  echo "       runs/final_pseudocode_product_v5_FULL" >&2
else
  echo "[publish] using FINAL_DIR=$FINAL_DIR"
  cp -f "$FINAL_DIR/FINAL_MATRIX_PSEUDOCODE_REPORT.md" reports/latest/ 2>/dev/null || true
  cp -f "$FINAL_DIR/FINAL_MATRIX_PSEUDOCODE_REPORT.html" reports/latest/ 2>/dev/null || true
  cp -f "$FINAL_DIR/PRODUCT_READINESS_CHECKLIST.md" reports/latest/ 2>/dev/null || true
  cp -f "$FINAL_DIR/MLP_OPERATOR_DICTIONARY_v1.md" reports/latest/ 2>/dev/null || true
  cp -f "$FINAL_DIR/final_summary.json" manifests/latest/final_summary.json 2>/dev/null || true
fi

# Atlas summaries/tables.
cp -f runs/atlas_full_all/atlas_summary.json manifests/latest/atlas_summary.json 2>/dev/null || true
cp -f runs/atlas_full_all/model_dims.json manifests/latest/model_dims.json 2>/dev/null || true
cp -f runs/atlas_full_all/static/all_heads_static_summary.csv reports/latest/tables/ 2>/dev/null || true
cp -f runs/atlas_full_all/runtime/all_heads_runtime_summary.csv reports/latest/tables/ 2>/dev/null || true
cp -f runs/atlas_full_all/mlp/all_layers_mlp_summary.csv reports/latest/tables/ 2>/dev/null || true

# Controls v4 summaries.
cp -f runs/controls_v4_token_sweep_L3H6/token_control_sweep/token_control_sweep_summary.csv reports/latest/tables/token_sweep_L3H6.csv 2>/dev/null || true
cp -f runs/controls_v4_token_sweep_L4H2/token_control_sweep/token_control_sweep_summary.csv reports/latest/tables/token_sweep_L4H2.csv 2>/dev/null || true
cp -f runs/controls_v4_recovery_L3H6_to_L4H8/causal_path_recovery/causal_path_recovery_summary.csv reports/latest/tables/recovery_L3H6_to_L4H8.csv 2>/dev/null || true
cp -f runs/controls_v4_recovery_L2H1_to_L3H6/causal_path_recovery/causal_path_recovery_summary.csv reports/latest/tables/recovery_L2H1_to_L3H6.csv 2>/dev/null || true
cp -f runs/controls_v4_logit_patch_attr/logit_patch_attr/logit_patch_attribution.csv reports/latest/tables/logit_patch_attribution.csv 2>/dev/null || true
cp -f runs/controls_v4_mlp_operator_all/mlp_operator/mlp_operator_summary.csv reports/latest/tables/mlp_operator_summary.csv 2>/dev/null || true

# Baselines.
cp -f runs/controls_baselines_top_heads_fixed/baselines/baselines_summary.csv reports/latest/tables/baselines_summary.csv 2>/dev/null || true

# Generation trace summary.
cp -f runs/atlas_gen_trace_full_python/generation_trace/generation_trace_summary.json manifests/latest/generation_trace_summary.json 2>/dev/null || true

# Minimal publish manifest for me/ChatGPT to inspect from GitHub.
python - <<'PY'
import json, os, subprocess, time
from pathlib import Path
root = Path('.')
files = []
for base in [Path('reports/latest'), Path('manifests/latest')]:
    if base.exists():
        for p in sorted(base.rglob('*')):
            if p.is_file():
                files.append({"path": str(p), "bytes": p.stat().st_size})
try:
    sha = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
except Exception:
    sha = None
manifest = {
    "created_at_unix": int(time.time()),
    "git_head_before_commit": sha,
    "published_files": files,
    "note": "Lightweight summaries only. Heavy runs/*.zip are local and intentionally ignored."
}
out = Path('manifests/latest/publish_manifest.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"[publish] wrote {out} with {len(files)} files")
PY

# Sanity warnings.
if [[ -f reports/latest/PRODUCT_READINESS_CHECKLIST.md ]]; then
  if grep -qi "n/a" reports/latest/PRODUCT_READINESS_CHECKLIST.md; then
    echo "[warn] PRODUCT_READINESS_CHECKLIST.md still contains n/a."
    echo "       Rebuild final report with tools/qwen_pseudocode_final_product_v5_no_na_fixed.py if needed."
  fi
fi

echo "[publish] git status before commit:"
git status --short

git add reports/latest manifests/latest docs tools run_scripts scripts README.md requirements.txt .gitignore 2>/dev/null || true

git commit -m "$COMMIT_MESSAGE" || echo "[publish] Nothing to commit"

if [[ "$PUSH" == "1" ]]; then
  git push
else
  echo "[publish] --no-push set; not pushing"
fi
