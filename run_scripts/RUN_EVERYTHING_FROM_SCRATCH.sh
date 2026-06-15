#!/usr/bin/env bash
set -euo pipefail
cp ./tools/*.py ./ 2>/dev/null || true
bash ./run_scripts/01_run_full_atlas.sh
# Use COMMANDS.md section 2 if you want generation trace.
# Use COMMANDS.md section 4 if you want the full expanded v4 controls.
bash ./run_scripts/02_run_controls_v4.sh
# Baselines and final product are in COMMANDS.md sections 5-6.
