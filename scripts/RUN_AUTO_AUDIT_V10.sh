#!/usr/bin/env bash
set -e
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
MODE="${MODE:-quick}"
echo "AUTO AUDIT MODE=$MODE"
if [ "$MODE" = "full" ]; then
  bash RUN_FULL_PROJECT_FROM_ZERO.sh
  bash scripts/RUN_DEEP_V8_FINAL_AND_PUBLISH.sh
  bash scripts/RUN_FULL_NEURON_SCAN_V9_AND_PUBLISH.sh
else
  bash scripts/RUN_DEEP_V8_FINAL_AND_PUBLISH.sh
  bash scripts/RUN_FULL_NEURON_SCAN_V9_AND_PUBLISH.sh
fi
bash scripts/PUBLISH_LATEST_TO_GITHUB.sh --message "Update auto audit summaries" || true
echo "DONE AUTO AUDIT"
