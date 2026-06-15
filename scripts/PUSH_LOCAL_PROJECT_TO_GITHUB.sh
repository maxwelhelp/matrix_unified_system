#!/usr/bin/env bash
set -euo pipefail

# Run this from the local project root, for example:
#   cd "$HOME/Рабочий стол/qwen_matrix_pseudocode_unified_system_v1"

REPO_URL="https://github.com/maxwelhelp/matrix_unified_system.git"
BRANCH="main"

git init
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REPO_URL"
else
  git remote set-url origin "$REPO_URL"
fi

git branch -M "$BRANCH"

# Keep huge generated artifacts local unless you intentionally want LFS.
# The project tools/reports/docs are committed; runs/*.zip stay local by default.
mkdir -p runs

cat > .gitignore <<'EOF'
runs/
*.pt
*.pth
*.safetensors
__pycache__/
*.pyc
.env
.cache/
qwen_program_cache*/
EOF

git add README.md tools docs reports scripts manifests requirements.txt .gitignore results_zips_placeholder 2>/dev/null || true

git status

git commit -m "Add matrix pseudocode unified system" || echo "Nothing new to commit"
git push -u origin "$BRANCH"
