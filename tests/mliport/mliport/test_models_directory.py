"""The repository ``models/`` convention for ``MODEL_PATH`` (#61 R2).

Tracking: #61 (task folder + INCAR + ``models/``), delivery 2.

R2.1 a bare file name is looked up in ``<MLIPORT_ROOT>/models/<name>`` and then
     in ``<MLIPORT_ROOT>/models/*/<name>`` (one level); two matches are an error.
R2.2 a value with a path separator keeps the existing behaviour and is never
     redirected into ``models/``; a same-named file next to the INCAR wins.
R2.3 ``MLIPORT_ROOT`` comes from the environment, else from the parent of an
     installer-created ``.venv-<engine>`` prefix; otherwise a bare name fails
     closed with instructions. The installer launcher exports it.
R2.4 a missing model names every location that was searched and lists the files
     that ``models/`` does contain.
R2.5 the resolved absolute path is what ``resolved_config.json`` records.
R2.6 ``mliport doctor`` reports the root, ``models/`` and its files.

R2.7 (``models/`` stays git-ignored, the hygiene policy is not loosened) is
covered by ``tests/test_repository_hygiene.py`` plus the assertion below.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from mliport.cli import main
from mliport.config.incar import IncarConfig
from mliport.config.resolver import resolve_config
from mliport.model_lookup import (
    MLIPORT_ROOT_ENV,
    ModelLookupError,
    available_model_files,
    mliport_root,
)

POSCAR_TEXT = (
    "He\n"
    "1.0\n"
    "5.0 0.0 0.0\n"
    "0.0 5.0 0.0\n"
    "0.0 0.0 5.0\n"
    "He\n"
    "1\n"
    "Direct\n"
    "0.0 0.0 0.0\n"
)


def _checkout(tmp_path: Path) -> Path:
    """A fake mliport checkout with a ``models/`` directory."""
    root = tmp_path / "checkout"
    (root / "models").mkdir(parents=True)
    return root


def _task_folder(tmp_path: Path, model_path: str) -> Path:
    folder = tmp_path / "task"
    folder.mkdir()
    (folder / "POSCAR").write_text(POSCAR_TEXT, encoding="utf-8")
    (folder / "INCAR.mliport").write_text(
        "CALC_TYPE = SP\n"
        "MODEL_TYPE = MACE\n"
        f"MODEL_PATH = {model_path}\n"
        "TASK = bulk\n"
        "WRITE_OUTCAR = False\n",
        encoding="utf-8",
    )
    return folder


def _resolve(folder: Path) -> str:
    config = IncarConfig.from_file(folder / "INCAR.mliport")
    resolved = resolve_config(calc_type="sp", incar=config)
    return resolved.model_path


class _RecordingEngine:
    def __init__(self) -> None:
        self.runs = 0

    def run(self, atoms, **kwargs):
        self.runs += 1
        return {}


def _patch_engine(monkeypatch) -> dict:
    engine = _RecordingEngine()
    captured: dict = {}

    def factory(config):
        captured["config"] = config
        return engine

    monkeypatch.setattr("mliport.cli.CalculationEngine.from_config", factory)
    return captured


# ---------------------------------------------------------------------------
# R2.1 / R2.4 - lookup in models/ and the failure messages
# ---------------------------------------------------------------------------


def test_bare_file_name_resolves_into_models(tmp_path, monkeypatch) -> None:
    root = _checkout(tmp_path)
    (root / "models" / "mace.model").write_text("weights", encoding="utf-8")
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    folder = _task_folder(tmp_path, "mace.model")

    assert _resolve(folder) == str((root / "models" / "mace.model").resolve())


def test_bare_file_name_resolves_in_one_level_of_subdirectories(
    tmp_path, monkeypatch
) -> None:
    root = _checkout(tmp_path)
    (root / "models" / "mace").mkdir()
    (root / "models" / "mace" / "omat.model").write_text("w", encoding="utf-8")
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    folder = _task_folder(tmp_path, "omat.model")

    assert _resolve(folder) == str((root / "models" / "mace" / "omat.model").resolve())


def test_two_matches_in_models_are_refused(tmp_path, monkeypatch) -> None:
    root = _checkout(tmp_path)
    for engine in ("mace", "uma"):
        (root / "models" / engine).mkdir()
        (root / "models" / engine / "dup.model").write_text("w", encoding="utf-8")
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    folder = _task_folder(tmp_path, "dup.model")

    with pytest.raises(ModelLookupError) as excinfo:
        _resolve(folder)
    message = str(excinfo.value)
    assert "more than one file" in message
    assert "mace/dup.model" in message.replace(str(root), "")
    assert "uma/dup.model" in message.replace(str(root), "")


def test_missing_model_lists_searched_locations_and_available_files(
    tmp_path, monkeypatch
) -> None:
    root = _checkout(tmp_path)
    (root / "models" / "other.model").write_text("w", encoding="utf-8")
    (root / "models" / "mace").mkdir()
    (root / "models" / "mace" / "nested.model").write_text("w", encoding="utf-8")
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    folder = _task_folder(tmp_path, "missing.model")

    with pytest.raises(ModelLookupError) as excinfo:
        _resolve(folder)
    message = str(excinfo.value)
    assert "missing.model" in message
    assert str(folder / "missing.model") in message
    assert str(root / "models" / "missing.model") in message
    assert "one level of subdirectories" in message
    assert "other.model" in message
    assert "mace/nested.model" in message


# ---------------------------------------------------------------------------
# R2.2 - real paths are untouched
# ---------------------------------------------------------------------------


def test_path_values_are_never_redirected_into_models(tmp_path, monkeypatch) -> None:
    root = _checkout(tmp_path)
    (root / "models" / "mace").mkdir()
    (root / "models" / "mace" / "model.pt").write_text("w", encoding="utf-8")
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    folder = _task_folder(tmp_path, "sub/model.pt")

    resolved = _resolve(folder)
    assert resolved == str((folder / "sub" / "model.pt").resolve())
    assert not Path(resolved).exists()


def test_same_name_next_to_the_incar_wins_over_models(tmp_path, monkeypatch) -> None:
    root = _checkout(tmp_path)
    (root / "models" / "model.pt").write_text("from models/", encoding="utf-8")
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    folder = _task_folder(tmp_path, "model.pt")
    (folder / "model.pt").write_text("local", encoding="utf-8")

    assert _resolve(folder) == str((folder / "model.pt").resolve())


def test_absolute_model_path_is_accepted_as_before(tmp_path) -> None:
    folder = _task_folder(tmp_path, "{MODEL}")
    model = tmp_path / "weights" / "m.model"
    model.parent.mkdir()
    model.write_text("w", encoding="utf-8")
    (folder / "INCAR.mliport").write_text(
        (folder / "INCAR.mliport")
        .read_text(encoding="utf-8")
        .replace("{MODEL}", str(model)),
        encoding="utf-8",
    )

    assert _resolve(folder) == str(model.resolve())


# ---------------------------------------------------------------------------
# R2.3 - how the root is found, and the fail-closed case
# ---------------------------------------------------------------------------


def test_root_comes_from_the_environment(tmp_path, monkeypatch) -> None:
    root = _checkout(tmp_path)
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    assert mliport_root() == root.resolve()


def test_root_is_derived_from_an_installer_venv_prefix(tmp_path, monkeypatch) -> None:
    root = _checkout(tmp_path)
    monkeypatch.delenv(MLIPORT_ROOT_ENV, raising=False)
    monkeypatch.setattr("sys.prefix", str(root / ".venv-mace"))

    assert mliport_root() == root.resolve()


def test_unknown_root_fails_closed_with_instructions(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(MLIPORT_ROOT_ENV, raising=False)
    monkeypatch.setattr("sys.prefix", str(tmp_path / ".venv-dev"))
    folder = _task_folder(tmp_path, "model.pt")

    with pytest.raises(ModelLookupError) as excinfo:
        _resolve(folder)
    message = str(excinfo.value)
    assert MLIPORT_ROOT_ENV in message
    assert ".venv-<engine>" in message
    assert str(folder / "model.pt") in message


def test_relative_mliport_root_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(MLIPORT_ROOT_ENV, "relative/checkout")
    with pytest.raises(ModelLookupError) as excinfo:
        mliport_root()
    assert "absolute path" in str(excinfo.value)


def test_installer_launcher_exports_the_root(tmp_path, monkeypatch) -> None:
    from mliport.install.compatibility import ALL_BACKENDS
    from mliport.install.launcher import (
        create_launcher,
        posix_launcher_script,
        windows_launcher_script,
    )

    backend = ALL_BACKENDS["mace"]
    script = posix_launcher_script(backend)
    assert "MLIPORT_ROOT=$(CDPATH=" in script
    assert "export MLIPORT_ROOT" in script
    assert windows_launcher_script(backend).count('set "MLIPORT_ROOT=') == 1

    # The wrapper must resolve the root to the directory that holds bin/.
    launcher = create_launcher(backend, tmp_path)
    stub = tmp_path / backend.venv_name / "bin" / "mliport"
    stub.parent.mkdir(parents=True)
    stub.write_text('#!/bin/sh\necho "ROOT=$MLIPORT_ROOT"\n', encoding="utf-8")
    stub.chmod(0o755)
    result = subprocess.run(
        [str(launcher)],
        capture_output=True,
        text=True,
        check=False,
        env={
            key: value for key, value in os.environ.items() if key != MLIPORT_ROOT_ENV
        },
    )
    assert result.returncode == 0, result.stderr
    assert f"ROOT={tmp_path}" in result.stdout


# ---------------------------------------------------------------------------
# R2.5 - the resolved path is the recorded path
# ---------------------------------------------------------------------------


def test_run_records_the_absolute_models_path(tmp_path, monkeypatch, capsys) -> None:
    root = _checkout(tmp_path)
    (root / "models" / "mace.model").write_text("w", encoding="utf-8")
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    folder = _task_folder(tmp_path, "mace.model")
    _patch_engine(monkeypatch)
    output = tmp_path / "out"
    old = Path.cwd()
    try:
        os.chdir(folder)
        rc = main(["run", "--output", str(output)])
    finally:
        os.chdir(old)

    assert rc == 0, capsys.readouterr()
    payload = json.loads((output / "resolved_config.json").read_text())
    assert payload["model_path"] == str((root / "models" / "mace.model").resolve())


# ---------------------------------------------------------------------------
# R2.6 - doctor reports the root and models/
# ---------------------------------------------------------------------------


def test_doctor_reports_the_root_and_the_models_listing(
    tmp_path, monkeypatch, capsys
) -> None:
    root = _checkout(tmp_path)
    (root / "models" / "mace.model").write_text("w", encoding="utf-8")
    (root / "models" / "uma").mkdir()
    (root / "models" / "uma" / "s1.pt").write_text("w", encoding="utf-8")
    monkeypatch.setenv(MLIPORT_ROOT_ENV, str(root))
    assert available_model_files(root) == ["mace.model", "uma/s1.pt"]

    for command in (["doctor"], ["doctor", "--json"]):
        rc = main(command)
        out = capsys.readouterr()
        assert rc == 0
        assert str(root) in out.out
        assert "mace.model" in out.out
        assert "uma/s1.pt" in out.out


def test_doctor_reports_a_missing_root_as_a_warning(tmp_path, monkeypatch) -> None:
    from mliport.doctor import run_diagnostics

    monkeypatch.delenv(MLIPORT_ROOT_ENV, raising=False)
    monkeypatch.setattr("sys.prefix", str(tmp_path / ".venv-dev"))

    checks, failures = run_diagnostics()
    by_name = {check["name"]: check for check in checks}
    assert by_name["MLIPORT_ROOT"]["status"] == "warn"
    assert by_name["models/"]["status"] == "warn"
    assert failures == 0


# ---------------------------------------------------------------------------
# R2.7 - models/ stays local-only
# ---------------------------------------------------------------------------


def test_models_directory_stays_git_ignored() -> None:
    repo_root = Path(__file__).parents[3]
    ignored = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "models/" in {line.strip() for line in ignored}
    tracked = subprocess.run(
        ["git", "ls-files", "models"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked.stdout.strip() == "", "weights must never be tracked"
