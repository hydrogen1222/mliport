"""Installer bootstrap regression tests (INSTALL-001).

The shell bootstrap must work on a fresh machine that has uv but no local
Python 3.10-3.12 with ``packaging`` installed: it may run the planner inside
an ephemeral ``uv run --with packaging`` environment.  These tests use stub
``uv``/``python`` executables so they stay fast and do not install anything.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "install_mliport.sh"


def _make_stub_bin(tmp_path: Path, *, has_packaging: bool) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub_python = bin_dir / "stub-python"
    stub_python.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            if [[ "${{1:-}}" == "-c" ]]; then
                code="${{2:-}}"
                if [[ "$code" == *"packaging.specifiers"* ]]; then
                    {"exit 0" if has_packaging else 'echo "ModuleNotFoundError: packaging" >&2; exit 1'}
                fi
                if [[ "$code" == *"sys.version_info"* ]]; then exit 0; fi
                exit 0
            fi
            if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "mliport.install" ]]; then
                echo "STUB_PLANNER:$*"
                exit 0
            fi
            echo "STUB_PY:$*"
            exit 0
            """
        ),
        encoding="utf-8",
    )
    stub_python.chmod(0o755)

    stub_uv = bin_dir / "uv"
    stub_uv.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            if [[ "${{1:-}}" == "python" && "${{2:-}}" == "find" ]]; then
                {"exit 1" if False else f'echo "{stub_python}"; exit 0'}
            fi
            if [[ "${{1:-}}" == "run" ]]; then
                args=("$@"); cmd=()
                for ((i = 0; i < ${{#args[@]}}; i++)); do
                    if [[ "${{args[$i]}}" == "--" ]]; then
                        cmd=("${{args[@]:$((i + 1))}}")
                        break
                    fi
                done
                if [[ ${{#cmd[@]}} -eq 0 ]]; then exit 1; fi
                # The ephemeral environment provides packaging.
                if [[ "${{cmd[1]:-}}" == "-c" && "${{cmd[2]:-}}" == *"packaging.specifiers"* ]]; then
                    exit 0
                fi
                if [[ "${{cmd[1]:-}}" == "-m" && "${{cmd[2]:-}}" == "mliport.install" ]]; then
                    echo "STUB_PLANNER:${{cmd[*]}}"
                    exit 0
                fi
                exec "{stub_python}" "${{cmd[@]:1}}"
            fi
            echo "STUB_UV:$*"
            exit 0
            """
        ),
        encoding="utf-8",
    )
    stub_uv.chmod(0o755)
    return bin_dir


def _run_bootstrap(bin_dir: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("MLIPORT_INSTALL_PYTHON", None)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_bootstrap_runs_planner_via_uv_when_interpreter_lacks_packaging(tmp_path):
    """INSTALL-001: bare uv-managed Python must not fail the install."""
    bin_dir = _make_stub_bin(tmp_path, has_packaging=False)
    result = _run_bootstrap(bin_dir, "--dry-run", "--engines", "mace")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "STUB_PLANNER" in result.stdout


def test_bootstrap_execs_planner_directly_when_packaging_is_present(tmp_path):
    bin_dir = _make_stub_bin(tmp_path, has_packaging=True)
    result = _run_bootstrap(bin_dir, "--dry-run", "--engines", "mace")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "STUB_PLANNER" in result.stdout
    assert "--engines mace" in result.stdout


def test_bootstrap_error_mentions_uv_python_install_when_no_interpreter(tmp_path):
    """A missing 3.10-3.12 interpreter gets actionable guidance, not a puzzle."""
    bin_dir = _make_stub_bin(tmp_path, has_packaging=False)
    stub_uv = bin_dir / "uv"
    stub_uv.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "python" && "${2:-}" == "find" ]]; then\n'
        "    exit 1\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub_uv.chmod(0o755)
    result = _run_bootstrap(bin_dir, "--dry-run", "--engines", "mace")
    assert result.returncode != 0
    assert "uv python install 3.12" in (result.stderr + result.stdout)
