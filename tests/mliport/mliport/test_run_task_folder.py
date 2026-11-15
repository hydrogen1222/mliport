"""Task-folder behaviour of ``mliport run`` (INCAR core of the usage spec).

Tracking: #61 (task folder + INCAR + ``models/``). This module covers the
INCAR round of that specification:

* R1.2 - the default output directory is ``<INCAR directory>/results/``,
  ``--output`` and ``OUTPUT_DIR`` keep priority, and a relative ``OUTPUT_DIR``
  is resolved against the INCAR, not the current working directory.
* R1.3 - a directory that already holds results is still refused without
  ``--overwrite``; the new default does not silently overwrite science.
* R1.4 - INCAR syntax/typing/model-path resolution fails before the structure
  file is read.
* R3.3 - ``CALC_TYPE = CONFSEARCH`` in an INCAR run by ``mliport run`` takes
  the same code path as ``mliport confsearch run``.

The other requirement (R3.2, template round-trip) lives in
``tests/mliport/mliport/config/test_template_roundtrip.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mliport.cli import main

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

START_XYZ = "1\nstart\nHe 0.0 0.0 0.0\n"


class _RecordingEngine:
    """Engine stub that records how ``mliport run`` dispatched the job."""

    def __init__(self, result: dict | None = None) -> None:
        self.result = {} if result is None else result
        self.runs: list[dict] = []
        self.confsearch_runs: list[dict] = []

    def run(self, atoms, **kwargs):
        self.runs.append({"formula": atoms.get_chemical_formula(), **kwargs})
        return self.result

    def run_confsearch(self, resolved, **kwargs):
        self.confsearch_runs.append({"resolved": resolved, **kwargs})
        return {"status": "completed", "ranking": {}, "diagnostics": {}}


def _write_task_folder(tmp_path: Path, incar_body: str) -> Path:
    """Create ``<tmp>/task/`` with POSCAR, model and INCAR.mliport."""
    folder = tmp_path / "task"
    folder.mkdir()
    (folder / "POSCAR").write_text(POSCAR_TEXT, encoding="utf-8")
    (folder / "model.pt").touch()
    (folder / "INCAR.mliport").write_text(incar_body, encoding="utf-8")
    return folder


def _patch_engine(monkeypatch, engine: _RecordingEngine) -> dict:
    """Patch ``CalculationEngine.from_config`` and capture the engine config."""
    captured: dict = {}

    def factory(config):
        captured["config"] = config
        return engine

    monkeypatch.setattr("mliport.cli.CalculationEngine.from_config", factory)
    return captured


def _chdir(path: Path):
    """Context manager that changes directory and always restores it."""

    class _Chdir:
        def __enter__(self):
            self._old = Path.cwd()
            os.chdir(path)
            return path

        def __exit__(self, *exc_info):
            os.chdir(self._old)
            return False

    return _Chdir()


SP_INCAR = (
    "CALC_TYPE = SP\n"
    "MODEL_TYPE = UMA\n"
    "MODEL_PATH = model.pt\n"
    "TASK = omat\n"
    "WRITE_OUTCAR = False\n"
)


# ---------------------------------------------------------------------------
# R1.2 - default output directory
# ---------------------------------------------------------------------------


def test_run_defaults_output_to_results_beside_the_incar(
    tmp_path, monkeypatch, capsys
) -> None:
    """`mliport run` with no arguments writes into <INCAR dir>/results/."""
    folder = _write_task_folder(tmp_path, SP_INCAR)
    engine = _RecordingEngine()
    captured = _patch_engine(monkeypatch, engine)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with _chdir(elsewhere):
        rc = main(
            [
                "run",
                "--incar",
                str(folder / "INCAR.mliport"),
                "--structure",
                str(folder / "POSCAR"),
            ]
        )

    assert rc == 0, capsys.readouterr()
    assert captured["config"].output_dir == (folder / "results").resolve()
    assert (folder / "results" / "resolved_config.json").is_file()
    assert not (elsewhere / "resolved_config.json").exists()


def test_run_uses_the_task_folder_when_it_is_the_current_directory(
    tmp_path, monkeypatch, capsys
) -> None:
    """The documented usage - cd into the folder, run `mliport run`."""
    folder = _write_task_folder(tmp_path, SP_INCAR)
    engine = _RecordingEngine()
    captured = _patch_engine(monkeypatch, engine)

    with _chdir(folder):
        rc = main(["run"])

    assert rc == 0, capsys.readouterr()
    assert captured["config"].output_dir == (folder / "results").resolve()
    assert len(engine.runs) == 1


def test_explicit_output_beats_the_task_folder_default(
    tmp_path, monkeypatch, capsys
) -> None:
    folder = _write_task_folder(tmp_path, SP_INCAR)
    engine = _RecordingEngine()
    captured = _patch_engine(monkeypatch, engine)
    target = tmp_path / "explicit"

    rc = main(
        [
            "run",
            "--incar",
            str(folder / "INCAR.mliport"),
            "--structure",
            str(folder / "POSCAR"),
            "--output",
            str(target),
        ]
    )

    assert rc == 0, capsys.readouterr()
    assert captured["config"].output_dir == target.resolve()
    assert not (folder / "results").exists()


def test_incar_output_dir_is_relative_to_the_incar_not_the_cwd(
    tmp_path, monkeypatch, capsys
) -> None:
    """A relative OUTPUT_DIR in the INCAR resolves against the INCAR folder."""
    folder = _write_task_folder(tmp_path, SP_INCAR + "OUTPUT_DIR = out_root\n")
    engine = _RecordingEngine()
    captured = _patch_engine(monkeypatch, engine)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with _chdir(elsewhere):
        rc = main(
            [
                "run",
                "--incar",
                str(folder / "INCAR.mliport"),
                "--structure",
                str(folder / "POSCAR"),
            ]
        )

    assert rc == 0, capsys.readouterr()
    assert captured["config"].output_dir == (folder / "out_root").resolve()
    assert not (folder / "results").exists()
    assert not (elsewhere / "out_root").exists()


def test_output_dir_dot_keeps_the_old_behaviour(tmp_path, monkeypatch, capsys) -> None:
    """OUTPUT_DIR = . still means "the current directory", not the INCAR folder.

    Run from somewhere else on purpose: the point of the escape hatch is that it
    restores the pre-#61 behaviour even when the working directory is not the
    task folder.
    """
    folder = _write_task_folder(
        tmp_path, SP_INCAR + "OUTPUT_DIR = .\nJOB_NAME = job1\n"
    )
    engine = _RecordingEngine()
    captured = _patch_engine(monkeypatch, engine)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with _chdir(elsewhere):
        rc = main(
            [
                "run",
                "--incar",
                str(folder / "INCAR.mliport"),
                "--structure",
                str(folder / "POSCAR"),
            ]
        )

    assert rc == 0, capsys.readouterr()
    assert captured["config"].output_dir == elsewhere.resolve()
    assert (elsewhere / "job1" / "resolved_config.json").is_file()
    assert not (folder / "results").exists()
    assert not (folder / "job1").exists()


def test_a_settings_ini_output_dir_stays_relative_to_the_working_directory(
    tmp_path, monkeypatch, capsys
) -> None:
    """Only the INCAR-declared OUTPUT_DIR became INCAR-relative (spec R1.2)."""
    settings_dir = tmp_path / "cfg"
    settings_dir.mkdir()
    (settings_dir / "settings.ini").write_text(
        "[output]\nOUTPUT_DIR = out_root\n", encoding="utf-8"
    )
    folder = _write_task_folder(tmp_path, SP_INCAR)
    engine = _RecordingEngine()
    captured = _patch_engine(monkeypatch, engine)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("MLIPORT_SETTINGS", str(settings_dir / "settings.ini"))

    with _chdir(elsewhere):
        rc = main(
            [
                "run",
                "--incar",
                str(folder / "INCAR.mliport"),
                "--structure",
                str(folder / "POSCAR"),
            ]
        )

    assert rc == 0, capsys.readouterr()
    assert captured["config"].output_dir == (elsewhere / "out_root").resolve()
    assert not (settings_dir / "out_root").exists()


# ---------------------------------------------------------------------------
# R1.3 - existing results are still refused
# ---------------------------------------------------------------------------


def test_run_refuses_an_existing_results_directory_without_overwrite(
    tmp_path, capsys
) -> None:
    folder = _write_task_folder(tmp_path, SP_INCAR)
    results = folder / "results"
    results.mkdir()
    (results / "OUTCAR").write_text("previous run\n", encoding="utf-8")

    rc = main(
        [
            "run",
            "--incar",
            str(folder / "INCAR.mliport"),
            "--structure",
            str(folder / "POSCAR"),
        ]
    )

    captured = capsys.readouterr()
    assert rc != 0
    assert "--overwrite" in captured.out + captured.err
    assert (results / "OUTCAR").read_text(encoding="utf-8") == "previous run\n"


def test_overwrite_reaches_the_engine_config(tmp_path, monkeypatch, capsys) -> None:
    folder = _write_task_folder(tmp_path, SP_INCAR)
    engine = _RecordingEngine()
    captured = _patch_engine(monkeypatch, engine)

    rc = main(
        [
            "run",
            "--incar",
            str(folder / "INCAR.mliport"),
            "--structure",
            str(folder / "POSCAR"),
            "--overwrite",
        ]
    )

    assert rc == 0, capsys.readouterr()
    assert captured["config"].overwrite is True


# ---------------------------------------------------------------------------
# R1.4 - configuration errors come before structure reading and model loading
# ---------------------------------------------------------------------------


def test_incar_value_error_is_reported_before_the_structure_is_read(
    tmp_path, capsys
) -> None:
    folder = tmp_path / "task"
    folder.mkdir()
    incar = folder / "INCAR.mliport"
    # No POSCAR in the folder: the structure lookup would fail on its own.
    incar.write_text(
        "CALC_TYPE = SP\n"
        "MODEL_TYPE = UMA\n"
        "MODEL_PATH = model.pt\n"
        "TASK = omat\n"
        "FMAX = not-a-number\n",
        encoding="utf-8",
    )

    with _chdir(folder):
        rc = main(["run"])

    out = capsys.readouterr()
    assert rc == 1
    assert "Configuration errors:" in out.out
    assert "fmax" in (out.out + out.err).lower()
    assert "No structure file" not in out.out + out.err


def test_unknown_incar_key_fails_before_the_structure_is_read(tmp_path, capsys) -> None:
    """Strict config resolution (not just the INCAR syntax check) runs first."""
    folder = tmp_path / "task"
    folder.mkdir()
    (folder / "model.pt").touch()
    (folder / "INCAR.mliport").write_text(
        "CALC_TYPE = SP\n"
        "MODEL_TYPE = UMA\n"
        "MODEL_PATH = model.pt\n"
        "TASK = omat\n"
        "DEFAULT_DTPE = float32\n",
        encoding="utf-8",
    )

    with _chdir(folder):
        rc = main(["run"])

    out = capsys.readouterr()
    assert rc == 1
    assert "Unknown key" in out.out + out.err
    assert "default_dtype" in (out.out + out.err).lower()  # typo suggestion
    assert "No structure file" not in out.out + out.err


# ---------------------------------------------------------------------------
# R3.3 - CALC_TYPE = CONFSEARCH through `mliport run`
# ---------------------------------------------------------------------------


CONFSEARCH_INCAR = (
    "CALC_TYPE = CONFSEARCH\n"
    "MODEL_TYPE = MACE\n"
    "MODEL_PATH = model.pt\n"
    "TASK = bulk\n"
    "CS_STARTS = start.xyz\n"
    "CS_ADSORBATE_INDICES = 0\n"
    "CS_CYCLES = 2\n"
    "WRITE_OUTCAR = False\n"
)


def _write_confsearch_folder(tmp_path: Path) -> Path:
    folder = _write_task_folder(tmp_path, CONFSEARCH_INCAR)
    (folder / "start.xyz").write_text(START_XYZ, encoding="utf-8")
    return folder


def test_run_dispatches_confsearch_incar_to_the_confsearch_runner(
    tmp_path, monkeypatch, capsys
) -> None:
    folder = _write_confsearch_folder(tmp_path)
    engine = _RecordingEngine()
    captured = _patch_engine(monkeypatch, engine)

    with _chdir(folder):
        rc = main(["run"])

    assert rc == 0, capsys.readouterr()
    assert engine.runs == [], "confsearch must not go through the product runner"
    assert len(engine.confsearch_runs) == 1
    resolved = engine.confsearch_runs[0]["resolved"]
    assert resolved.calc_type == "confsearch"
    assert resolved.run_options["cs_cycles"] == 2
    assert captured["config"].output_dir == (folder / "results").resolve()


def test_run_and_confsearch_run_resolve_the_same_confsearch_config(
    tmp_path, monkeypatch, capsys
) -> None:
    """The two entry points share one code path, so the resolution matches."""
    folder = _write_confsearch_folder(tmp_path)

    run_engine = _RecordingEngine()
    run_captured = _patch_engine(monkeypatch, run_engine)
    with _chdir(folder):
        assert main(["run"]) == 0

    cli_engine = _RecordingEngine()
    cli_captured = _patch_engine(monkeypatch, cli_engine)
    with _chdir(folder):
        assert (
            main(
                [
                    "confsearch",
                    "run",
                    "start.xyz",
                    "--adsorbate-indices",
                    "0",
                    "--model",
                    "model.pt",
                    "--model-type",
                    "mace",
                    "--task",
                    "bulk",
                    "--cycles",
                    "2",
                    "--output",
                    "results",
                ]
            )
            == 0
        ), capsys.readouterr()

    run_resolved = run_engine.confsearch_runs[0]["resolved"]
    cli_resolved = cli_engine.confsearch_runs[0]["resolved"]
    assert run_captured["config"].output_dir == cli_captured["config"].output_dir
    assert run_resolved.calc_type == cli_resolved.calc_type == "confsearch"
    assert run_resolved.model_type == cli_resolved.model_type == "mace"
    assert run_resolved.model_path == cli_resolved.model_path
    assert run_resolved.run_options == cli_resolved.run_options


def test_confsearch_through_run_still_rejects_batch_only_keys(
    tmp_path, monkeypatch, capsys
) -> None:
    """CALC_TYPE = CONFSEARCH keeps the MD-only-key rejection of the spec."""
    folder = _write_confsearch_folder(tmp_path)
    (folder / "INCAR.mliport").write_text(
        CONFSEARCH_INCAR + "TEMPERATURE = 500\n", encoding="utf-8"
    )
    engine = _RecordingEngine()
    _patch_engine(monkeypatch, engine)

    with _chdir(folder):
        rc = main(["run"])

    out = capsys.readouterr()
    assert rc == 1
    assert "not valid for calc_type='confsearch'" in out.out + out.err
    assert engine.confsearch_runs == []


# ---------------------------------------------------------------------------
# The default output directory is recorded, not implicit
# ---------------------------------------------------------------------------


def test_resolved_config_records_the_absolute_output_directory(
    tmp_path, monkeypatch, capsys
) -> None:
    folder = _write_task_folder(tmp_path, SP_INCAR + "OUTPUT_DIR = out_root\n")
    engine = _RecordingEngine()
    _patch_engine(monkeypatch, engine)

    with _chdir(folder):
        assert main(["run"]) == 0, capsys.readouterr()

    payload = json.loads((folder / "out_root" / "resolved_config.json").read_text())
    assert payload["sources"]["output_dir"]["value"] == str(
        (folder / "out_root").resolve()
    )
