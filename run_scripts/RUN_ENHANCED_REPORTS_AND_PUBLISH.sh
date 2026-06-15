#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
exec bash scripts/RUN_ENHANCED_REPORTS_AND_PUBLISH.sh "$@"
