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

cd "$REPO_ROOT"

# Respect the final --source and --python values before bootstrapping Python.
# The Python installer applies UV_OFFLINE to every plan step, but the bootstrap
# itself runs first and must not download a managed interpreter in offline mode.
SOURCE_PROFILE="auto"
TARGET_PYTHON="3.12"
EXPECT_SOURCE_VALUE=0
EXPECT_PYTHON_VALUE=0
READ_ONLY=0
HELP_ONLY=0
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
        --help|-h)
            READ_ONLY=1
            HELP_ONLY=1
            ;;
        --dry-run)
            READ_ONLY=1
            ;;
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

bootstrap_help() {
    echo "Usage: install_mlipx.sh [--engines uma,mace,dpa,grace] [--device auto|cuda|cpu]"
    echo "       [--source auto|official|china|china-aliyun|china-ustc|china-tencent|custom|offline]"
    echo "       [--python 3.10|3.11|3.12] [--dry-run] [--clean] [--skip-doctor] [--non-interactive]"
    echo "UMA requires Python >=3.11. Help and dry-run never download Python."
}

if ! command -v uv >/dev/null 2>&1; then
    if [[ "$HELP_ONLY" -eq 1 ]]; then
        bootstrap_help
        exit 0
    fi
    echo "[mlipx] ERROR: 'uv' not found on PATH. Install uv before running the installer." >&2
    exit 1
fi

# Pick a Python interpreter that uv manages (3.12 preferred, 3.10/3.11 ok).
# This avoids depending on an old system python3 that cannot parse the
# modern type annotations used by mlipx.install.
if [[ -z "${MLIPX_INSTALL_PYTHON:-}" && "$HELP_ONLY" -eq 0 ]]; then
    case "$TARGET_PYTHON" in
        3.10|3.11|3.12) ;;
        *)
            echo "[mlipx] ERROR: unsupported Python $TARGET_PYTHON; choose 3.10, 3.11, or 3.12." >&2
            exit 2
            ;;
    esac
fi
PY_SPEC="${MLIPX_INSTALL_PYTHON:-$TARGET_PYTHON}"
if [[ "$SOURCE_PROFILE" == "offline" || "$READ_ONLY" -eq 1 ]]; then
    export UV_OFFLINE=1
    export UV_PYTHON_DOWNLOADS=never
    if ! UV_PY="$(uv python find "$PY_SPEC" 2>/dev/null)"; then
        if [[ "$HELP_ONLY" -eq 1 ]]; then
            bootstrap_help
            exit 0
        fi
        if [[ "$SOURCE_PROFILE" != "offline" ]]; then
            echo "[mlipx] ERROR: dry-run requires an existing Python $PY_SPEC interpreter; automatic download is disabled." >&2
            exit 1
        fi
        echo "[mlipx] ERROR: offline mode requires an existing Python $PY_SPEC interpreter; automatic download is disabled." >&2
        exit 1
    fi
else
    # RC-06: the planner runtime and the target environment runtime are two
    # different things. The planner only parses args, validates the
    # engine x target-Python matrix, and computes the source/device plan; it
    # may be any local Python 3.10-3.12 with 'packaging'. Only after the full
    # plan validates is uv allowed to obtain the target interpreter while
    # creating the environments (normal online mode).
    UV_PY=""
    _planner_ok() {
        "$1" -c 'import sys, packaging.specifiers; assert (3, 10) <= sys.version_info < (3, 13)' >/dev/null 2>&1
    }
    if [[ -n "${MLIPX_INSTALL_PYTHON:-}" ]]; then
        if _planner_ok "$MLIPX_INSTALL_PYTHON"; then
            UV_PY="$MLIPX_INSTALL_PYTHON"
        else
            echo "[mlipx] ERROR: MLIPX_INSTALL_PYTHON must be a local Python 3.10-3.12 with packaging installed." >&2
            exit 1
        fi
    else
        # Fast path: the target interpreter itself acts as the planner when
        # it already exists locally (this lookup never downloads anything).
        UV_PY="$(UV_PYTHON_DOWNLOADS=never uv python find "$PY_SPEC" 2>/dev/null || true)"
        if [[ -n "$UV_PY" ]] && ! _planner_ok "$UV_PY"; then
            UV_PY=""
        fi
        if [[ -z "$UV_PY" ]]; then
            # Target absent or unusable as planner: plan with any local
            # compatible runtime; the plan still targets --python.
            for CANDIDATE in "$REPO_ROOT/.venv/bin/python" python3.12 python3.11 python3.10 python3 python; do
                if _planner_ok "$CANDIDATE"; then
                    UV_PY="$CANDIDATE"
                    break
                fi
            done
        fi
        if [[ -z "$UV_PY" ]]; then
            echo "[mlipx] ERROR: no local Python 3.10-3.12 runtime with packaging found to plan the install; set MLIPX_INSTALL_PYTHON to such an interpreter." >&2
            exit 1
        fi
        echo "[mlipx] Planner runtime: $UV_PY (target environment: Python $TARGET_PYTHON)"
    fi
fi

# The planner uses packaging.SpecifierSet. A bare managed interpreter can
# lack packaging even when the target Python exists. Reuse an installed
# local planner runtime without changing the requested target --python.
if ! "$UV_PY" -c 'import packaging.specifiers' >/dev/null 2>&1; then
    PLANNER_PY=""
    for CANDIDATE in "$REPO_ROOT/.venv/bin/python" python3 python; do
        if "$CANDIDATE" -c 'import sys, packaging.specifiers; assert (3, 10) <= sys.version_info < (3, 13)' >/dev/null 2>&1; then
            PLANNER_PY="$CANDIDATE"
            break
        fi
    done
    if [[ -z "$PLANNER_PY" ]]; then
        if [[ "$HELP_ONLY" -eq 1 ]]; then
            bootstrap_help
            exit 0
        fi
        echo "[mlipx] ERROR: preflight needs a local Python 3.10-3.12 with packaging installed. Set MLIPX_INSTALL_PYTHON to that interpreter; no dependencies were downloaded." >&2
        exit 1
    fi
    UV_PY="$PLANNER_PY"
fi

# Make the mlipx package importable: PYTHONPATH -> <repo>/mlipx (project root),
# so `import mlipx` resolves to <repo>/mlipx/mlipx.
export PYTHONPATH="$REPO_ROOT/mlipx${PYTHONPATH:+:$PYTHONPATH}"

exec "$UV_PY" -m mlipx.install "$@"
