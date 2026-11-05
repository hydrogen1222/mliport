"""Batch optimization at scale: output isolation, option threading, resume."""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

import mliport.runners.batch as batch_module
from mliport.cli import _build_cli_opts, create_parser
from mliport.engine import CalculationEngine, EngineConfig
from mliport.runners.batch import BatchRunner


class _RecordingOptimizationRunner:
    """Model-free optimizer stand-in; records where and how it was called."""

    calls: ClassVar[list[dict]] = []

    def __init__(self, calculator, output_dir, verbose=False, **kwargs):
        self.output_dir = output_dir
        self.kwargs = kwargs

    def execute(self, atoms):
        type(self).calls.append({"output_dir": self.output_dir, **self.kwargs})
        (self.output_dir / "CONTCAR").write_text(f"{len(atoms)}\n")
        converged = len(atoms) != 2  # two-atom inputs "miss" max_steps
        return {
            "energy": -float(len(atoms)),
            "forces": np.zeros((len(atoms), 3)),
            "converged": converged,
            "nsteps": 3,
            "time": 0.0,
        }


@pytest.fixture()
def recording_opt(monkeypatch):
    _RecordingOptimizationRunner.calls = []
    monkeypatch.setattr(
        batch_module, "OptimizationRunner", _RecordingOptimizationRunner
    )
    return _RecordingOptimizationRunner


def _ml_opt_tree(root):
    """Two compositions x two ranks, every structure named 1.cif."""
    files = []
    for composition in ("P15Cl15", "P10Cl20"):
        for rank in ("ewald_rank_0001", "ewald_rank_0002"):
            directory = root / composition / rank
            directory.mkdir(parents=True)
            path = directory / "1.cif"
            write(
                path,
                Atoms("Li", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
            )
            files.append(path)
    return sorted(files)


def test_same_filename_in_different_directories_never_shares_output(
    tmp_path, recording_opt
):
    files = _ml_opt_tree(tmp_path / "ML-OPT")
    out = tmp_path / "out"
    summary = BatchRunner(
        object(), calc_type="opt", output_dir=out, verbose=False
    ).run_from_files(files, input_root=tmp_path / "ML-OPT")

    assert summary["total"] == summary["success"] == 4
    written = sorted(p.parent.relative_to(out).as_posix() for p in out.rglob("CONTCAR"))
    assert written == [
        "P10Cl20/ewald_rank_0001/1",
        "P10Cl20/ewald_rank_0002/1",
        "P15Cl15/ewald_rank_0001/1",
        "P15Cl15/ewald_rank_0002/1",
    ]
    names = [item["output_name"] for item in summary["results"]]
    assert len(set(names)) == 4


def test_unresolvable_output_collision_fails_before_running(tmp_path, recording_opt):
    directory = tmp_path / "in"
    directory.mkdir()
    # a.b.cif / a.b.xyz share a stem and become a_b_cif / a_b_xyz, which
    # clashes with the plain stem of a_b_cif.xyz.
    files = [directory / name for name in ("a.b.cif", "a.b.xyz", "a_b_cif.xyz")]
    for path in files:
        path.write_text("x")
    with pytest.raises(ValueError, match="share the output directory"):
        BatchRunner.output_names(files)
    with pytest.raises(ValueError, match="same structure file twice"):
        BatchRunner.output_names([files[0], files[0]])
    assert recording_opt.calls == []


def test_status_is_persisted_per_structure(tmp_path, recording_opt):
    files = _ml_opt_tree(tmp_path / "ML-OPT")
    out = tmp_path / "out"
    BatchRunner(
        object(), calc_type="opt", output_dir=out, verbose=False
    ).run_from_files(files)
    status = json.loads(
        (out / "P15Cl15" / "ewald_rank_0001" / "1" / "batch_status.json").read_text()
    )
    assert status["status"] == "completed"
    assert status["nsteps"] == 3
    assert status["manifest"]["calc_type"] == "opt"
    progress = (out / "batch_progress.jsonl").read_text().splitlines()
    assert len(progress) == 4
    assert (out / "batch_manifest.json").exists()


def test_resume_reuses_finished_structures_and_reruns_the_rest(tmp_path, recording_opt):
    root = tmp_path / "ML-OPT"
    files = _ml_opt_tree(root)
    # One structure "misses" max_steps (not converged: final, reused).
    write(
        root / "P10Cl20" / "ewald_rank_0002" / "1.cif",
        Atoms("Li2", positions=[[0, 0, 0], [2, 0, 0]], cell=[5, 5, 5], pbc=True),
    )
    # One structure cannot be read (failed: retried on resume).
    broken = root / "P15Cl15" / "ewald_rank_0002" / "1.cif"
    broken.write_text("not a cif\n")
    out = tmp_path / "out"
    kwargs = {"fmax": 0.02, "max_steps": 800, "optimizer": "BFGS"}

    first = BatchRunner(
        object(), calc_type="opt", output_dir=out, verbose=False, **kwargs
    ).run_from_files(files, input_root=root)
    assert first["success"] == 2
    assert first["not_converged"] == 1
    assert first["failed"] == 2  # not_converged counts as not successful

    # Interrupted mid-structure: that structure has outputs but no status.
    (out / "P10Cl20" / "ewald_rank_0001" / "1" / "batch_status.json").unlink()
    write(broken, Atoms("Li", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True))

    recording_opt.calls = []
    second = BatchRunner(
        object(), calc_type="opt", output_dir=out, verbose=False, resume=True, **kwargs
    ).run_from_files(files, input_root=root)

    rerun = sorted(
        call["output_dir"].relative_to(out).as_posix() for call in recording_opt.calls
    )
    assert rerun == ["P10Cl20/ewald_rank_0001/1", "P15Cl15/ewald_rank_0002/1"]
    assert second["reused"] == 2
    assert second["success"] == 3
    assert second["not_converged"] == 1
    assert second["total"] == 4


def test_resume_refuses_changed_settings_and_missing_manifest(tmp_path, recording_opt):
    root = tmp_path / "ML-OPT"
    files = _ml_opt_tree(root)
    out = tmp_path / "out"
    BatchRunner(
        object(), calc_type="opt", output_dir=out, verbose=False, fmax=0.02
    ).run_from_files(files, input_root=root)

    with pytest.raises(ValueError, match="fmax"):
        BatchRunner(
            object(),
            calc_type="opt",
            output_dir=out,
            verbose=False,
            resume=True,
            fmax=0.05,
        ).run_from_files(files, input_root=root)
    with pytest.raises(FileNotFoundError, match="batch_manifest.json"):
        BatchRunner(
            object(),
            calc_type="opt",
            output_dir=tmp_path / "new",
            verbose=False,
            resume=True,
        ).run_from_files(files, input_root=root)


def test_cli_batch_accepts_and_forwards_optimization_options():
    args = create_parser().parse_args(
        [
            "batch",
            "ML-OPT",
            "--calc-type",
            "opt",
            "--pattern",
            "*/ewald_rank_*/1.cif",
            "--fmax",
            "0.02",
            "--max-steps",
            "800",
            "--optimizer",
            "BFGS",
            "--no-cell-opt",
            "--resume",
        ]
    )
    opts = _build_cli_opts(args, "batch")
    assert opts["sub_calc_type"] == "opt"
    assert opts["fmax"] == 0.02
    assert opts["max_steps"] == 800
    assert opts["optimizer"] == "BFGS"
    assert opts["cell_opt"] is False
    assert args.resume is True


class _FakeWrapper:
    task = "bulk"
    has_stress = True

    def get_calculator(self):
        return None


def _batch_engine(tmp_path, *, run_options, resume=False):
    return CalculationEngine(
        EngineConfig(
            calc_type="batch",
            model_path=tmp_path / "model.model",
            model_type="grace",
            output_dir=tmp_path / "run",
            run_options=run_options,
            resume=resume,
        )
    )


def test_engine_threads_optimization_options_to_batch_runner(tmp_path, monkeypatch):
    captured = {}

    class _CapturingBatchRunner:
        def __init__(self, calculator, **kwargs):
            captured.update(kwargs)

        def run_from_files(self, files, input_root=None):
            captured["input_root"] = input_root
            return {"total": 0, "success": 0, "failed": 0, "not_converged": 0}

    monkeypatch.setattr(batch_module, "BatchRunner", _CapturingBatchRunner)
    engine = _batch_engine(
        tmp_path,
        run_options={
            "sub_calc_type": "opt",
            "fmax": 0.02,
            "max_steps": 800,
            "optimizer": "BFGS",
            "cell_opt": False,
        },
    )
    monkeypatch.setattr(engine, "_create_calculator", lambda: _FakeWrapper())
    engine.run_batch([], input_root=tmp_path)

    assert captured["calc_type"] == "opt"
    assert captured["fmax"] == 0.02
    assert captured["max_steps"] == 800
    assert captured["optimizer"] == "BFGS"
    assert captured["cell_opt"] is False
    assert captured["fix_symmetry"] is False  # built-in default, explicit
    assert captured["input_root"] == tmp_path
    assert captured["run_identity"]["model_type"] == "grace"


def test_engine_refuses_optimization_options_for_singlepoint_batch(
    tmp_path, monkeypatch
):
    engine = _batch_engine(tmp_path, run_options={"sub_calc_type": "sp", "fmax": 0.02})
    monkeypatch.setattr(engine, "_create_calculator", lambda: _FakeWrapper())
    with pytest.raises(ValueError, match="only apply to batch"):
        engine.run_batch([])


def test_engine_resume_accepts_existing_batch_outputs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.log").write_text("earlier batch\n")
    with _batch_engine(tmp_path, run_options={}, resume=True).output_directory_claim():
        pass
    sp_engine = CalculationEngine(
        EngineConfig(
            calc_type="sp",
            model_path=tmp_path / "model.model",
            model_type="grace",
            output_dir=run,
            resume=True,
        )
    )
    with (
        pytest.raises(ValueError, match="only supported for batch"),
        sp_engine.output_directory_claim(),
    ):
        pass
