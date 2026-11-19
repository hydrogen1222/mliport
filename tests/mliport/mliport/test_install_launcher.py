"""Platform-specific launcher contract tests."""

from __future__ import annotations

import os

import pytest

from mliport.install.compatibility import ALL_BACKENDS
from mliport.install.launcher import (
    create_launcher,
    launcher_path,
    posix_launcher_script,
    windows_launcher_script,
)


@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher is not created on Windows")
def test_posix_launcher_selects_the_environment_without_changing_model_options(
    tmp_path,
) -> None:
    backend = ALL_BACKENDS["mace"]
    path = launcher_path("mace", tmp_path)

    assert path == tmp_path / "bin" / "mliport-mace"
    script = posix_launcher_script(backend)
    assert ".venv-mace/bin/mliport" in script
    assert "CUDA_VISIBLE_DEVICES" not in script.split("exec ", 1)[1]


def test_create_launcher_returns_the_native_executable_path(tmp_path) -> None:
    path = create_launcher(ALL_BACKENDS["mace"], tmp_path)

    assert path == launcher_path("mace", tmp_path)
    if os.name == "nt":
        assert path.name == "mliport-mace.cmd"
        assert r"Scripts\mliport.exe" in path.read_text(encoding="utf-8")
    else:
        assert path.name == "mliport-mace"
        assert path.stat().st_mode & 0o111
        assert ".venv-mace/bin/mliport" in path.read_text(encoding="utf-8")


def test_windows_launcher_uses_scripts_console_entrypoint() -> None:
    backend = ALL_BACKENDS["dpa-legacy"]
    script = windows_launcher_script(backend)

    assert "MLIPORT_ROOT=%~dp0.." in script
    assert r".venv-dpa-legacy\Scripts\mliport.exe" in script
    assert "CUDA_VISIBLE_DEVICES" not in script
