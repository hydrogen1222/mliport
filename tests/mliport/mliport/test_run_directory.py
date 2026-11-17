"""Output-directory ownership: no silent overwrite, no concurrent writers."""

from __future__ import annotations

import os

import pytest

from mliport.engine import CalculationEngine, EngineConfig
from mliport.run_directory import (
    RUN_LOCK_NAME,
    OutputDirectoryInUseError,
    ensure_output_available,
    existing_run_artifacts,
    run_directory_lock,
)


def test_existing_outputs_are_detected_but_inputs_are_not(tmp_path) -> None:
    (tmp_path / "POSCAR").write_text("structure\n")
    (tmp_path / "notes.txt").write_text("x\n")
    assert existing_run_artifacts(tmp_path) == []

    (tmp_path / "OUTCAR").write_text("vasp\n")
    (tmp_path / "raw").mkdir()
    assert existing_run_artifacts(tmp_path) == ["OUTCAR", "raw"]


def test_existing_outputs_are_refused_unless_overwrite(tmp_path) -> None:
    ensure_output_available(tmp_path / "missing", overwrite=False)
    ensure_output_available(tmp_path, overwrite=False)
    (tmp_path / "CONTCAR").write_text("previous\n")

    with pytest.raises(OutputDirectoryInUseError, match="--overwrite"):
        ensure_output_available(tmp_path, overwrite=False)
    ensure_output_available(tmp_path, overwrite=True)


def test_output_path_that_is_a_file_is_refused(tmp_path) -> None:
    target = tmp_path / "out"
    target.write_text("not a directory\n")
    with pytest.raises(OutputDirectoryInUseError, match="not a directory"):
        ensure_output_available(target, overwrite=True)


def test_run_lock_is_exclusive_and_released(tmp_path) -> None:
    with run_directory_lock(tmp_path):
        assert (tmp_path / RUN_LOCK_NAME).exists()
        with (
            pytest.raises(OutputDirectoryInUseError, match="already writing"),
            run_directory_lock(tmp_path),
        ):
            pass
    with run_directory_lock(tmp_path):
        pass


def _engine(tmp_path, *, overwrite: bool = False) -> CalculationEngine:
    return CalculationEngine(
        EngineConfig(
            calc_type="sp",
            model_path=tmp_path / "model.model",
            model_type="mace",
            output_dir=tmp_path / "run",
            overwrite=overwrite,
        )
    )


def test_engine_claim_refuses_previous_run_and_is_reentrant(tmp_path) -> None:
    engine = _engine(tmp_path)
    with engine.output_directory_claim() as output:
        (output / "run.log").write_text("first run\n")
        # The run itself re-enters the claim taken by the frontend.
        with engine.output_directory_claim():
            pass

    with (
        pytest.raises(OutputDirectoryInUseError, match="run.log"),
        _engine(tmp_path).output_directory_claim(),
    ):
        pass
    with _engine(tmp_path, overwrite=True).output_directory_claim():
        pass


def test_engine_run_refuses_before_loading_a_model(tmp_path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "OUTCAR").write_text("vasp output\n")

    def _fail(*_args, **_kwargs):
        raise AssertionError("model must not load for an occupied directory")

    monkeypatch.setattr(engine, "_create_calculator", _fail)
    with pytest.raises(OutputDirectoryInUseError):
        engine.run(atoms=None)  # type: ignore[arg-type]


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX permission bits are what this test exercises (Windows has no chmod semantics)",
)
def test_unwritable_output_directory_is_reported_not_raised_as_a_traceback(
    tmp_path,
) -> None:
    """A read-only run directory must produce an actionable error (R10 R01/R03).

    Before this, opening the lock file raised a bare ``PermissionError``, so the
    CLI exited with a Python traceback instead of the ``Error: ...`` line every
    other refused directory produces.
    """
    # ``os.geteuid`` only exists on POSIX; the skipif above keeps this body
    # POSIX-only, and the getattr keeps the check defensive on exotic platforms.
    if getattr(os, "geteuid", lambda: -1)() == 0:  # pragma: no cover - root
        pytest.skip("running as root: permission bits do not deny access")

    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    try:
        with (
            pytest.raises(OutputDirectoryInUseError) as excinfo,
            run_directory_lock(readonly),
        ):
            pass
    finally:
        readonly.chmod(0o700)

    message = str(excinfo.value)
    assert f"Cannot write in output directory {readonly}" in message
    assert "Permission denied" in message
    assert "--output" in message
