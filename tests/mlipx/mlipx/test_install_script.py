"""Integration tests for the thin shell installer bootstrap."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("flag", ["--help", "-h", "--dry-run"])
def test_read_only_bootstrap_missing_runtime_never_downloads(tmp_path, flag):
    calls = tmp_path / "calls"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n" f"printf '%s\\n' \"$*\" >> {str(calls)!r}\n" "exit 1\n"
    )
    fake_uv.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    env.pop("MLIPX_INSTALL_PYTHON", None)
    result = subprocess.run(
        ["bash", "scripts/install_mlipx.sh", flag],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (1 if flag == "--dry-run" else 0)
    assert calls.read_text().splitlines() == ["python find 3.12"]
    if flag != "--dry-run":
        assert "Usage:" in result.stdout


def test_bootstrap_preserves_argv_and_runtime_path_with_spaces(tmp_path):
    calls = tmp_path / "argv.json"
    runtime = tmp_path / "runtime with spaces"
    runtime.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(calls)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    runtime.chmod(0o755)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(f"#!/bin/bash\nprintf '%s\\n' {str(runtime)!r}\n")
    fake_uv.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    args = ["--dry-run", "--engines", "mace, dpa", "--source", "custom"]
    result = subprocess.run(
        ["bash", "scripts/install_mlipx.sh", *args],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(calls.read_text()) == ["-m", "mlipx.install", *args]


@pytest.mark.parametrize(
    ("source_args", "expected_python"),
    [
        (["--source", "offline"], "3.12"),
        (["--source=offline"], "3.12"),
        (["--source", "china", "--source", "offline"], "3.12"),
        (["--source", "offline", "--python", "3.10"], "3.10"),
        (["--python=3.11", "--source=offline"], "3.11"),
    ],
)
def test_offline_bootstrap_never_downloads_python(
    tmp_path: Path, source_args: list[str], expected_python: str
) -> None:
    calls = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {str(calls)!r}\n"
        'if [[ "$1 $2" == "python find" ]]; then exit 1; fi\n'
        'if [[ "$1 $2" == "python install" ]]; then exit 88; fi\n'
        "exit 99\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env.pop("MLIPX_INSTALL_PYTHON", None)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"

    result = subprocess.run(
        ["bash", "scripts/install_mlipx.sh", *source_args, "--dry-run"],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "offline mode requires an existing Python" in result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [
        f"python find {expected_python}"
    ]


def test_final_non_offline_source_dry_run_never_bootstraps(
    tmp_path: Path,
) -> None:
    calls = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n" f"printf '%s\\n' \"$*\" >> {str(calls)!r}\n" "exit 1\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env.pop("MLIPX_INSTALL_PYTHON", None)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            "scripts/install_mlipx.sh",
            "--source",
            "offline",
            "--source=official",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "python find 3.12",
    ]


def test_invalid_target_python_fails_before_uv_download(tmp_path: Path) -> None:
    calls = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {str(calls)!r}\n"
        "exit 99\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env.pop("MLIPX_INSTALL_PYTHON", None)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"

    result = subprocess.run(
        ["bash", "scripts/install_mlipx.sh", "--python", "3.9", "--dry-run"],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unsupported Python 3.9" in result.stderr
    assert not calls.exists()
