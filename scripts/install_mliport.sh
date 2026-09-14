#!/usr/bin/env bash
#
# mliport — one-command installer (thin bootstrap wrapper)
# ======================================================
#
# This wrapper only:
#   1. verifies `uv` is available
#   2. locates the repository root
#   3. selects a Python 3.10–3.12 interpreter (via uv, never the old system
#      python3) so the modern mliport.install code can run
#   4. delegates everything else to `python -m mliport.install`
#
# All logic (GPU detection, compatibility matrix, plan generation, dry-run,
# execution) lives in Python under mliport/mliport/install/.
#
# Usage: ./scripts/install_mliport.sh [--engines ...] [--device ...] [--source ...]
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
    echo "Usage: install_mliport.sh [--engines mace dpa grace uma] [--device auto|cuda|cpu]"
    echo "       [--source auto|official|china|china-aliyun|china-ustc|china-tencent|custom|offline]"
    echo "       [--python 3.10|3.11|3.12] [--dry-run] [--clean] [--skip-doctor] [--non-interactive]"
    echo "UMA requires Python >=3.11. Help and dry-run never download Python."
}

if ! command -v uv >/dev/null 2>&1; then
    if [[ "$HELP_ONLY" -eq 1 ]]; then
        bootstrap_help
        exit 0
    fi
    echo "[mliport] ERROR: 'uv' not found on PATH. Install uv before running the installer." >&2
    exit 1
fi

# Pick a Python 3.10-3.12 interpreter (uv-managed preferred) to run the
# modern mliport.install planner.  A bare uv-managed interpreter often has
# no third-party packages, so when the selected interpreter lacks
# ``packaging`` the planner runs in an ephemeral ``uv run --with packaging``
# environment instead of failing the install (regression: fresh machine with
# only uv and no project venv).
if [[ -z "${MLIPORT_INSTALL_PYTHON:-}" && "$HELP_ONLY" -eq 0 ]]; then
    case "$TARGET_PYTHON" in
        3.10|3.11|3.12) ;;
        *)
            echo "[mliport] ERROR: unsupported Python $TARGET_PYTHON; choose 3.10, 3.11, or 3.12." >&2
            exit 2
            ;;
    esac
fi
PY_SPEC="${MLIPORT_INSTALL_PYTHON:-$TARGET_PYTHON}"

_python_range_ok() {
    "$1" -c 'import sys; assert (3, 10) <= sys.version_info < (3, 13)' >/dev/null 2>&1
}
_planner_ok() {
    "$1" -c 'import sys, packaging.specifiers; assert (3, 10) <= sys.version_info < (3, 13)' >/dev/null 2>&1
}

if [[ "$SOURCE_PROFILE" == "offline" || "$READ_ONLY" -eq 1 ]]; then
    export UV_OFFLINE=1
    export UV_PYTHON_DOWNLOADS=never
    if ! UV_PY="$(uv python find "$PY_SPEC" 2>/dev/null)"; then
        if [[ "$HELP_ONLY" -eq 1 ]]; then
            bootstrap_help
            exit 0
        fi
        if [[ "$SOURCE_PROFILE" != "offline" ]]; then
            echo "[mliport] ERROR: dry-run requires an existing Python $PY_SPEC interpreter. Run 'uv python install $PY_SPEC' first (or set MLIPORT_INSTALL_PYTHON); automatic download is disabled in dry-run." >&2
            exit 1
        fi
        echo "[mliport] ERROR: offline mode requires an existing Python $PY_SPEC interpreter. Install it beforehand or set MLIPORT_INSTALL_PYTHON; automatic download is disabled offline." >&2
        exit 1
    fi
else
    if [[ -n "${MLIPORT_INSTALL_PYTHON:-}" ]]; then
        if ! _planner_ok "$MLIPORT_INSTALL_PYTHON"; then
            echo "[mliport] ERROR: MLIPORT_INSTALL_PYTHON must be a local Python 3.10-3.12 with packaging installed." >&2
            exit 1
        fi
        UV_PY="$MLIPORT_INSTALL_PYTHON"
    else
        # The target interpreter is the preferred planner when it already
        # exists locally.  This lookup never downloads anything.
        UV_PY="$(UV_PYTHON_DOWNLOADS=never uv python find "$PY_SPEC" 2>/dev/null || true)"
        if [[ -n "$UV_PY" ]] && ! _python_range_ok "$UV_PY"; then
            UV_PY=""
        fi
        if [[ -z "$UV_PY" ]]; then
            for CANDIDATE in "$REPO_ROOT/.venv/bin/python" python3.12 python3.11 python3.10 python3 python; do
                if _python_range_ok "$CANDIDATE"; then
                    UV_PY="$CANDIDATE"
                    break
                fi
            done
        fi
        if [[ -z "$UV_PY" ]]; then
            echo "[mliport] ERROR: no local Python 3.10-3.12 runtime found to plan the install. Run 'uv python install 3.12' first, or set MLIPORT_INSTALL_PYTHON to a 3.10-3.12 interpreter with packaging." >&2
            exit 1
        fi
        echo "[mliport] Planner runtime: $UV_PY (target environment: Python $TARGET_PYTHON)"
    fi
fi

# Make the mliport package importable: PYTHONPATH -> <repo>/mliport (project root),
# so `import mliport` resolves to <repo>/mliport/mliport.
export PYTHONPATH="$REPO_ROOT/mliport${PYTHONPATH:+:$PYTHONPATH}"

if _planner_ok "$UV_PY"; then
    exec "$UV_PY" -m mliport.install "$@"
fi

# The interpreter exists but has no packaging.  Build the planner
# environment through uv.  Read-only/offline modes may only use cached
# wheels, so a cache miss is a clear error rather than a silent download.
if [[ "$READ_ONLY" -eq 1 || "$SOURCE_PROFILE" == "offline" ]]; then
    if uv run --no-project --offline --with packaging --python "$UV_PY" -- python -c 'import packaging.specifiers' >/dev/null 2>&1; then
        exec uv run --no-project --offline --with packaging --python "$UV_PY" -- python -m mliport.install "$@"
    fi
    echo "[mliport] ERROR: planner Python $UV_PY lacks 'packaging' and no cached copy is available in offline/dry-run mode. Install packaging into that interpreter or set MLIPORT_INSTALL_PYTHON." >&2
    exit 1
fi
exec uv run --no-project --with packaging --python "$UV_PY" -- python -m mliport.install "$@"
