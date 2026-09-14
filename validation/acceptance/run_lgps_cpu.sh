#!/usr/bin/env bash
# CPU acceptance smoke (task book sections 22-26, 74):
# import / doctor / one LGPS single point / one repeated single point per backend.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${MLIPORT_ACCEPTANCE_PYTHON:-.venv-analysis/bin/python}"
COMMIT="${1:-$(git rev-parse --short HEAD)}"
CPU_TASKS="g1_sp,g1_sp_repeat"

for engine in mace dpa grace uma; do
    echo "===== CPU ${engine} ====="
    "$PY" validation/acceptance/lgps_smoke.py \
        --backend "$engine" --device cpu --commit "$COMMIT" --only "$CPU_TASKS"
done
"$PY" validation/acceptance/collect_environment.py \
    --commit "$COMMIT" --device cpu \
    --out ".validation-acceptance/${COMMIT}/cpu/environment.json" \
    --matrix ".validation-acceptance/${COMMIT}/cpu/INSTALL_MATRIX.md"
