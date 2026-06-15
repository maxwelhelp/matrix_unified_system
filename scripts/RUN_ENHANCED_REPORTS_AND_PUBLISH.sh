#!/usr/bin/env bash
set -euo pipefail

# Build enhanced lightweight reports and publish them to GitHub.
# Run after the main experiment and after PUBLISH_LATEST_TO_GITHUB.sh, or run directly:
#   bash scripts/RUN_ENHANCED_REPORTS_AND_PUBLISH.sh
#   bash scripts/RUN_ENHANCED_REPORTS_AND_PUBLISH.sh --no-push

PUSH=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-push)
      PUSH=0
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

python tools/qwen_pseudocode_enhanced_reports_v6.py --root .

if [[ "$PUSH" == "1" ]]; then
  bash scripts/PUBLISH_LATEST_TO_GITHUB.sh --message "Update enhanced why-token and MLP reports"
else
  bash scripts/PUBLISH_LATEST_TO_GITHUB.sh --no-push --message "Update enhanced why-token and MLP reports"
fi
