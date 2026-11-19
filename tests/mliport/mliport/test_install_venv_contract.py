"""Contract tests for reusing an environment that already exists.

``uv venv --allow-existing`` is not a compatibility check: it will initialise a
venv inside a plain directory and it will silently replace a venv whose
interpreter is another Python version.  mliport therefore classifies the target
path *before* any step runs: absent → create, compatible → reuse, everything
else → refuse with ``--clean`` and change nothing.

These tests build real (but package-free) venv-shaped directories, so they run
without network access and without ``uv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mliport.install.compatibility import BACKENDS
from mliport.install.venv_contract import (
    Verdict,
    backend_for_venv,
    incompatible_message,
    inspect_target_venv,
    reuse_message,
)

_ACTUAL = f"{sys.version_info.major}.{sys.version_info.minor}"
_OTHER = "3.11" if _ACTUAL != "3.11" else "3.12"


def _make_venv(path: Path, *, python: Path | None = None) -> Path:
    """A venv-shaped directory whose interpreter reports ``sys.version_info``."""
    (path / "bin").mkdir(parents=True)
    (path / "pyvenv.cfg").write_text(
        f"home = {sys.base_prefix}\n"
        "include-system-site-packages = false\n"
        f"version = {sys.version.split()[0]}\n",
        encoding="utf-8",
    )
    interpreter = path / "bin" / "python"
    interpreter.symlink_to(python or Path(sys.executable))
    (path / "lib" / f"python{_ACTUAL}" / "site-packages").mkdir(parents=True)
    return path


def _install_package(venv: Path, name: str, *, dist_info: bool = True) -> None:
    site_packages = venv / "lib" / f"python{_ACTUAL}" / "site-packages"
    if dist_info:
        (site_packages / f"{name}-1.0.dist-info").mkdir()
    else:
        (site_packages / name).mkdir()


def _inspect(venv: Path, backend: str = "nep", python: str | None = None):
    return inspect_target_venv(
        venv, backend=BACKENDS[backend], python_version=python or _ACTUAL
    )


def test_absent_path_is_created_not_inspected(tmp_path: Path) -> None:
    state = _inspect(tmp_path / ".venv-nep")
    assert state.verdict is Verdict.ABSENT
    assert state.python_version == ""
    assert not state.refused
    assert not (tmp_path / ".venv-nep").exists(), "inspection must not create anything"


def test_existing_environment_symlink_is_refused_without_following_target(
    tmp_path: Path,
) -> None:
    target = _make_venv(tmp_path / "shared-venv")
    link = tmp_path / ".venv-nep"
    link.symlink_to(target, target_is_directory=True)
    state = _inspect(link)
    assert state.verdict is Verdict.INCOMPATIBLE
    assert state.reason == "symlink_venv"
    assert "symbolic link" in state.detail
    assert target.joinpath("pyvenv.cfg").is_file()


def test_dangling_environment_symlink_is_not_treated_as_absent(
    tmp_path: Path,
) -> None:
    link = tmp_path / ".venv-nep"
    link.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    assert not link.exists()
    state = _inspect(link)
    assert state.verdict is Verdict.INCOMPATIBLE
    assert state.reason == "symlink_venv"
    assert link.is_symlink()


def test_compatible_environment_is_reusable(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path / ".venv-nep")
    state = _inspect(venv)
    assert state.verdict is Verdict.REUSE
    assert state.reusable
    assert state.python_version == _ACTUAL


def test_own_backend_and_framework_packages_are_not_foreign(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path / ".venv-nep")
    _install_package(venv, "calorine")
    assert _inspect(venv).verdict is Verdict.REUSE

    mace = _make_venv(tmp_path / ".venv-mace")
    _install_package(mace, "torch", dist_info=False)
    _install_package(mace, "mace")
    assert _inspect(mace, "mace").verdict is Verdict.REUSE

    grace = _make_venv(tmp_path / ".venv-grace")
    _install_package(grace, "tensorflow", dist_info=False)
    assert _inspect(grace, "grace").verdict is Verdict.REUSE


def test_foreign_framework_is_refused(tmp_path: Path) -> None:
    """A NEP environment that somehow grew torch is not a NEP environment."""
    venv = _make_venv(tmp_path / ".venv-nep")
    _install_package(venv, "torch", dist_info=False)
    state = _inspect(venv)
    assert state.verdict is Verdict.INCOMPATIBLE
    assert state.reason == "foreign_packages"
    assert "PyTorch" in state.detail
    assert "torch" in state.foreign
    assert "torch" in state.foreign


def test_foreign_backend_package_is_refused(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path / ".venv-nep")
    _install_package(venv, "deepmd_kit")
    state = _inspect(venv)
    assert state.verdict is Verdict.INCOMPATIBLE
    assert state.reason == "foreign_packages"
    assert "DPA" in state.detail


def test_torch_environment_holding_tensorflow_is_refused(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path / ".venv-mace")
    _install_package(venv, "tensorflow", dist_info=False)
    state = _inspect(venv, "mace")
    assert state.verdict is Verdict.INCOMPATIBLE
    assert state.reason == "foreign_packages"


def test_plain_directory_is_not_silently_converted(tmp_path: Path) -> None:
    path = tmp_path / ".venv-nep"
    path.mkdir()
    (path / "keep.txt").write_text("user data", encoding="utf-8")
    state = _inspect(path)
    assert state.verdict is Verdict.INCOMPATIBLE
    assert state.reason == "not_a_venv"
    assert "pyvenv.cfg" in state.detail
    assert (path / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_regular_file_at_the_environment_path_is_refused(tmp_path: Path) -> None:
    path = tmp_path / ".venv-nep"
    path.write_text("not a venv", encoding="utf-8")
    state = _inspect(path)
    assert state.verdict is Verdict.INCOMPATIBLE
    assert state.reason == "not_a_venv"


def test_venv_without_interpreter_is_refused(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path / ".venv-nep")
    (venv / "bin" / "python").unlink()
    state = _inspect(venv)
    assert state.verdict is Verdict.INCOMPATIBLE
    assert state.reason == "no_interpreter"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX interpreter shim")
def test_broken_interpreter_is_refused(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path / ".venv-nep")
    interpreter = venv / "bin" / "python"
    interpreter.unlink()
    interpreter.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    interpreter.chmod(0o755)
    state = _inspect(venv)
    assert state.verdict is Verdict.INCOMPATIBLE
    assert state.reason == "unreadable_interpreter"


def test_other_python_version_is_refused(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path / ".venv-nep")
    state = _inspect(venv, python=_OTHER)
    assert state.verdict is Verdict.INCOMPATIBLE
    assert state.reason == "python_mismatch"
    assert state.python_version == _ACTUAL
    assert _OTHER in state.detail


def test_messages_name_the_environment_and_the_way_out(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path / ".venv-nep")
    _install_package(venv, "torch", dist_info=False)
    refused = _inspect(venv)
    message = incompatible_message(refused)
    assert ".venv-nep" in message
    assert "--clean" in message
    assert "Nothing was changed" in message

    ok = _inspect(_make_venv(tmp_path / ".venv-ok"))
    reused = reuse_message(ok, backend=BACKENDS["nep"])
    assert "Reusing existing" in reused
    assert "NEP" in reused


def test_every_backend_venv_name_maps_to_its_backend() -> None:
    for engine, backend in BACKENDS.items():
        assert backend_for_venv(backend.venv_name) is backend, engine
    assert backend_for_venv(".venv-not-a-backend") is None
