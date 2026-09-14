#!/usr/bin/env bash
# Full GPU acceptance matrix (task book sections 68-74).
#
# Requires a Python with ase/numpy for the harness itself:
#   MLIPORT_ACCEPTANCE_PYTHON=.venv-analysis/bin/python
# The harness drives the per-backend .venv-*/bin/mliport entry points.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${MLIPORT_ACCEPTANCE_PYTHON:-.venv-analysis/bin/python}"
COMMIT="${1:-$(git rev-parse --short HEAD)}"

GPU_TASKS="g1_sp,g1_sp_repeat,g2_sp_perturbed,g3_opt_fixed,g4_opt_cell,g5_nve,g6_nvt_langevin,g7_nvt_bussi,g8_nvt_nhc,g9_md_restart,g10_batch,g11_incar,g12_negatives,g13_api,g14_tui,g15_queue,g16_neb,g17_neb_restart,g18_analysis_core"
LONG_TASKS="g19_analysis_long,g20_arrhenius"

for engine in mace dpa grace uma; do
    echo "===== GPU ${engine} ====="
    if ! "$PY" validation/acceptance/lgps_smoke.py \
        --backend "$engine" --device cuda --commit "$COMMIT" --only "$GPU_TASKS"; then
        echo "[acceptance] ${engine} recorded failures or limitations; continuing"
    fi
done
echo "===== representative long-fixture analysis ====="
# Separate output directory: the per-backend grace results.json must keep the
# g1-g18 workflow records.
if ! "$PY" validation/acceptance/lgps_smoke.py \
    --backend grace --device cuda --commit "$COMMIT" \
    --out ".validation-acceptance/${COMMIT}/gpu/analysis_long" \
    --only "$LONG_TASKS"; then
    echo "[acceptance] long analysis recorded failures or limitations"
fi
