#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. Main script lives in scripts/.
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
exec bash scripts/PUBLISH_LATEST_TO_GITHUB.sh "$@"
