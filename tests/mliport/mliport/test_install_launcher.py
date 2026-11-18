"""Platform-specific launcher contract tests."""

from __future__ import annotations

from mliport.install.compatibility import ALL_BACKENDS
from mliport.install.launcher import (
    launcher_path,
    posix_launcher_script,
    windows_launcher_script,
)


def test_posix_launcher_selects_the_environment_without_changing_model_options(
    tmp_path,
) -> None:
    backend = ALL_BACKENDS["mace"]
    path = launcher_path("mace", tmp_path)

    assert path == tmp_path / "bin" / "mliport-mace"
    script = posix_launcher_script(backend)
    assert '.venv-mace/bin/mliport' in script
    assert "CUDA_VISIBLE_DEVICES" not in script.split("exec ", 1)[1]


def test_windows_launcher_uses_scripts_console_entrypoint() -> None:
    backend = ALL_BACKENDS["dpa-legacy"]
    script = windows_launcher_script(backend)

    assert 'MLIPORT_ROOT=%~dp0..' in script
    assert r'.venv-dpa-legacy\Scripts\mliport.exe' in script
    assert 'CUDA_VISIBLE_DEVICES' not in script
