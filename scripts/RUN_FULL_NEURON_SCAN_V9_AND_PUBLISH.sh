#!/usr/bin/env bash
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
LAYERS="${LAYERS:-all}"
SUITES="${SUITES:-code,math,text}"
OUT="runs/full_neuron_scan_v9"
rm -rf "$OUT"
mkdir -p reports/latest/tables manifests/latest
python tools/qwen_full_neuron_scan_v9.py --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" --prompt-suites "$SUITES" --layers "$LAYERS" --top-k 64 --patch-k 32 --patch-groups --out-dir "$OUT" 2>&1 | tee runs/full_neuron_scan_v9.log
cp -f "$OUT/FULL_NEURON_SCAN_REPORT.md" reports/latest/FULL_NEURON_SCAN_REPORT.md
cp -f "$OUT/full_neuron_scan_v9_summary.json" manifests/latest/full_neuron_scan_v9_summary.json
cp -f "$OUT/tables/full_neuron_top_effects.csv" reports/latest/tables/full_neuron_top_effects.csv
cp -f "$OUT/tables/full_neuron_stable.csv" reports/latest/tables/full_neuron_stable.csv
cp -f "$OUT/tables/full_neuron_patch_groups.csv" reports/latest/tables/full_neuron_patch_groups.csv
git add reports/latest manifests/latest tools/qwen_full_neuron_scan_v9.py scripts/RUN_FULL_NEURON_SCAN_V9_AND_PUBLISH.sh
git commit -m "Update full-neuron scan v9 summaries" || true
git push
