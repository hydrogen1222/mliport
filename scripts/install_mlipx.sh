#!/usr/bin/env bash
#
# mlipx — one-command installer (thin bootstrap wrapper)
# ======================================================
#
# This wrapper only:
#   1. verifies `uv` is available
#   2. locates the repository root
#   3. selects a Python 3.10–3.12 interpreter (via uv, never the old system
#      python3) so the modern mlipx.install code can run
#   4. delegates everything else to `python -m mlipx.install`
#
# All logic (GPU detection, compatibility matrix, plan generation, dry-run,
# execution) lives in Python under mlipx/mlipx/install/.
#
# Usage: ./scripts/install_mlipx.sh [--engines ...] [--device ...] [--source ...]
#        [--non-interactive] [--python ...] [--clean] [--skip-doctor] [--dry-run]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "[mlipx] ERROR: 'uv' not found on PATH." >&2
    echo "        Install it first:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

cd "$REPO_ROOT"

# Respect the final --source and --python values before bootstrapping Python.
# The Python installer applies UV_OFFLINE to every plan step, but the bootstrap
# itself runs first and must not download a managed interpreter in offline mode.
SOURCE_PROFILE="auto"
TARGET_PYTHON="3.12"
EXPECT_SOURCE_VALUE=0
EXPECT_PYTHON_VALUE=0
for ARG in "$@"; do
    if [[ "$EXPECT_SOURCE_VALUE" -eq 1 ]]; then
        SOURCE_PROFILE="$ARG"
        EXPECT_SOURCE_VALUE=0
        continue
    fi
    if [[ "$EXPECT_PYTHON_VALUE" -eq 1 ]]; then
        TARGET_PYTHON="$ARG"
        EXPECT_PYTHON_VALUE=0
        continue
    fi
    case "$ARG" in
        --source)
            EXPECT_SOURCE_VALUE=1
            ;;
        --source=*)
            SOURCE_PROFILE="${ARG#--source=}"
            ;;
        --python)
            EXPECT_PYTHON_VALUE=1
            ;;
        --python=*)
            TARGET_PYTHON="${ARG#--python=}"
            ;;
    esac
done

# Pick a Python interpreter that uv manages (3.12 preferred, 3.10/3.11 ok).
# This avoids depending on an old system python3 that cannot parse the
# modern type annotations used by mlipx.install.
if [[ -z "${MLIPX_INSTALL_PYTHON:-}" ]]; then
    case "$TARGET_PYTHON" in
        3.10|3.11|3.12) ;;
        *)
            echo "[mlipx] ERROR: unsupported Python $TARGET_PYTHON; choose 3.10, 3.11, or 3.12." >&2
            exit 2
            ;;
    esac
fi
PY_SPEC="${MLIPX_INSTALL_PYTHON:-$TARGET_PYTHON}"
if [[ "$SOURCE_PROFILE" == "offline" ]]; then
    export UV_OFFLINE=1
    if ! UV_PY="$(uv python find "$PY_SPEC" 2>/dev/null)"; then
        echo "[mlipx] ERROR: offline mode requires an existing Python $PY_SPEC interpreter; automatic download is disabled." >&2
        exit 1
    fi
else
    UV_PY="$(uv python find "$PY_SPEC" 2>/dev/null \
        || { uv python install "$PY_SPEC" >/dev/null 2>&1 && uv python find "$PY_SPEC"; } \
        || { echo "[mlipx] ERROR: could not obtain a Python $PY_SPEC interpreter via uv." >&2; exit 1; })"
fi

# Make the mlipx package importable: PYTHONPATH -> <repo>/mlipx (project root),
# so `import mlipx` resolves to <repo>/mlipx/mlipx.
export PYTHONPATH="$REPO_ROOT/mlipx${PYTHONPATH:+:$PYTHONPATH}"

exec "$UV_PY" -m mlipx.install "$@"
