from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.io import read

from mlipx.config import resolve_config
from mlipx.neb.io import NEBCheckpointStore, load_checkpoint
from mlipx.neb.prepare import BandInput
from mlipx.neb.schema import (
    NEBEndpointNotConvergedError,
    NEBOptions,
    NEBPreparationError,
)
from mlipx.neb.workflow import (
    checkpoint_resolved_layer,
    checkpoint_run_directory,
    run_neb_workflow,
)
from mlipx.runners.neb import NEBRunner


class ZeroCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros_like(atoms.positions, dtype=float),
        }


class FiniteForceCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        forces = np.zeros_like(atoms.positions, dtype=float)
        forces[-1, 0] = 1.0
        self.results = {"energy": 0.0, "forces": forces}


class FakeMACEWrapper:
    task = "bulk"
    has_stress = False

    def capabilities(self, model_record):
        from mlipx.capabilities import CalculatorCapabilities

        return CalculatorCapabilities(
            True, True, False, "validated", "test-known-answer"
        )

    def __init__(self, calculator=None) -> None:
        self.calculator = calculator or ZeroCalculator()
        self.get_calls = 0

    def get_calculator(self):
        self.get_calls += 1
        return self.calculator

    def info(self) -> dict:
        return {
            "model_type": "mace",
            "backend_version": "99.1-test",
            "framework_versions": {"torch": "99.2-test"},
            "default_dtype": "float64",
            "active_head": "default",
        }


def _endpoints() -> tuple[Atoms, Atoms]:
    initial = Atoms(
        "HHe",
        positions=[[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    final = initial.copy()
    final.positions[1, 0] = 2.0
    initial.set_constraint(FixAtoms(indices=[0]))
    final.set_constraint(FixAtoms(indices=[0]))
    return initial, final


def _resolved(model: Path, **overrides):
    cli = {
        "model_type": "mace",
        "model_path": str(model),
        "task": "bulk",
        "device": "cpu",
        "n_intermediate_images": 1,
        "max_steps": 0,
        "checkpoint_interval": 1,
    }
    cli.update(overrides)
    return resolve_config(calc_type="neb", cli=cli)


def _run(tmp_path: Path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"model identity")
    output = tmp_path / "run"
    wrapper = FakeMACEWrapper()
    initial, final = _endpoints()
    result = run_neb_workflow(
        wrapper,
        _resolved(model),
        output_dir=output,
        initial=initial,
        final=final,
        verbose=False,
    )
    return model, output, wrapper, result


def _fixed_indices(atoms: Atoms) -> list[int]:
    indices: set[int] = set()
    for constraint in atoms.constraints:
        indices.update(int(index) for index in constraint.get_indices())
    return sorted(indices)


def test_workflow_writes_context_result_complete_band_and_vasp_export(
    tmp_path: Path,
) -> None:
    _model, output, wrapper, result = _run(tmp_path)

    assert wrapper.get_calls == 1
    assert result["status"] == "completed"
    assert result["converged"] is True
    assert result["failure_reason"] is None
    assert result["device"] == {
        "requested": "cpu",
        "effective": "cpu",
        "effective_uuid": None,
        "actual_device_type": "unknown",
        "actual_device_logical_index": None,
        "actual_device_uuid": None,
    }
    assert result["model"]["backend_version"] == "99.1-test"
    assert result["model"]["framework_versions"] == {"torch": "99.2-test"}
    assert result["model"]["dtype_effective"] == "float64"
    assert result["barrier_forward_sampled_eV"] == pytest.approx(0.0)
    assert result["barrier_forward_fitted_eV"] is None
    assert result["barrier_reverse_fitted_eV"] is None
    for key in (
        "raw_physical_forces_eV_A",
        "constraint_applied_forces_eV_A",
        "neb_band_forces_eV_A",
    ):
        assert np.asarray(result[key]).shape == (3, 2, 3)

    context = json.loads((output / "run_context.json").read_text(encoding="utf-8"))
    assert context["mlipx_version"] not in {"", "unknown"}
    assert context["status"] == "completed"
    assert context["converged"] is True
    assert context["constraints"] == {"fix_atoms": [0]}
    assert context["atom_map"] == [0, 1]
    assert context["image_shifts"] == [[0, 0, 0], [0, 0, 0]]

    checkpoint, images = load_checkpoint(output)
    assert checkpoint["complete"] is True
    assert checkpoint["geometry_resume_only"] is True
    assert checkpoint_run_directory(output / "checkpoints" / "latest") == output
    assert len(images) == 3
    assert all(image.info["trajectory_kind"] == "neb_band" for image in images)
    assert all(image.calc is None for image in images)

    export = output / "vasp_path"
    assert [path.parent.name for path in sorted(export.glob("*/POSCAR"))] == [
        "00",
        "01",
        "02",
    ]
    roundtrip = read(export / "01" / "POSCAR")
    assert roundtrip.get_chemical_symbols() == ["H", "He"]
    assert _fixed_indices(roundtrip) == [0]
    band_metadata = json.loads((export / "band.json").read_text(encoding="utf-8"))
    assert band_metadata["energy_source"] is None
    assert band_metadata["atom_map"] == [0, 1]
    assert band_metadata["image_shifts"] == [[0, 0, 0], [0, 0, 0]]


def test_engine_entrypoint_locks_before_logging_and_loads_one_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlipx.engine import CalculationEngine, EngineConfig

    model = tmp_path / "model.pt"
    model.write_bytes(b"model identity")
    resolved = _resolved(model)
    engine = CalculationEngine.from_config(EngineConfig.from_resolved(resolved))
    engine.config.output_dir = tmp_path / "engine-run"
    wrapper = FakeMACEWrapper()
    monkeypatch.setattr(engine, "_create_calculator", lambda: wrapper)
    initial, final = _endpoints()

    result = engine.run_neb(resolved, initial=initial, final=final)

    assert result["status"] == "completed"
    assert wrapper.get_calls == 1
    assert (engine.output_dir / "run.log").is_file()
    assert (engine.output_dir / "run_context.json").is_file()


def test_runner_failure_records_failed_status_and_last_complete_checkpoint(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"model identity")
    output = tmp_path / "failed-run"
    initial, final = _endpoints()

    with pytest.raises(NEBEndpointNotConvergedError, match="Endpoint 0 fmax"):
        run_neb_workflow(
            FakeMACEWrapper(FiniteForceCalculator()),
            _resolved(model),
            output_dir=output,
            initial=initial,
            final=final,
            verbose=False,
        )

    context = json.loads((output / "run_context.json").read_text(encoding="utf-8"))
    failure = json.loads((output / "neb_results.json").read_text(encoding="utf-8"))
    checkpoint, _images = load_checkpoint(output)
    assert context["status"] == "failed"
    assert context["converged"] is False
    assert failure["status"] == "failed"
    assert failure["converged"] is False
    assert failure["failure_reason"] == context["failure_reason"]
    assert checkpoint["stage"] == "prepared"


def test_geometry_resume_preserves_run_id_and_rejects_fingerprint_change(
    tmp_path: Path,
) -> None:
    model, output, _wrapper, first = _run(tmp_path)
    checkpoint = output / "checkpoints" / "latest"

    second = run_neb_workflow(
        FakeMACEWrapper(),
        _resolved(model),
        output_dir=output,
        resume=checkpoint,
        verbose=False,
    )
    assert second["run_id"] == first["run_id"]
    assert second["attempt_id"] != first["attempt_id"]
    assert checkpoint_resolved_layer(checkpoint)["fmax_abort"] == 20.0

    checkpoint_count = len(list((output / "checkpoints").glob("step_*")))
    with pytest.raises(NEBPreparationError, match="fingerprint is incompatible"):
        run_neb_workflow(
            FakeMACEWrapper(),
            _resolved(model, neb_spring=0.2),
            output_dir=output,
            resume=checkpoint,
            verbose=False,
        )
    assert len(list((output / "checkpoints").glob("step_*"))) == checkpoint_count


@pytest.mark.parametrize(
    "revision", ["NEB_SCIENTIFIC_REVISION", "NEB_CHECKPOINT_SCHEMA_REVISION"]
)
def test_resume_rejects_changed_software_revision(tmp_path, monkeypatch, revision):
    from mlipx.neb import workflow

    model, output, _, _ = _run(tmp_path)
    checkpoint = output / "checkpoints" / "latest"
    metadata, _ = load_checkpoint(checkpoint)
    assert metadata["resume_fingerprint"]["software"]["mlipx_version"]
    monkeypatch.setattr(workflow, revision, 2)
    with pytest.raises(NEBPreparationError, match="fingerprint is incompatible"):
        run_neb_workflow(
            FakeMACEWrapper(),
            _resolved(model),
            output_dir=output,
            resume=checkpoint,
            verbose=False,
        )


def test_checkpoint_checksum_corruption_fails_closed(tmp_path: Path) -> None:
    _model, output, _wrapper, _result = _run(tmp_path)
    checkpoint = (output / "checkpoints" / "latest").resolve()
    with (checkpoint / "band.traj").open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(NEBPreparationError, match="checksum mismatch"):
        load_checkpoint(checkpoint)


def test_failed_checkpoint_never_replaces_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial, final = _endpoints()
    middle = initial.copy()
    middle.positions = 0.5 * (initial.positions + final.positions)
    store = NEBCheckpointStore(tmp_path / "run")
    keyword_args = {
        "run_id": "34a70ca9-4bbb-48f6-bc86-d6b850b54d6f",
        "attempt_id": "2b2239bc-64fd-4689-88f6-07d869b2026e",
        "stage": "neb",
        "stage_step": 0,
        "resume_fingerprint": {},
        "resume_fingerprint_sha256": "test",
        "resolved_config": {},
    }
    first = store.write((initial, middle, final), **keyword_args)

    def reject_temporary_checkpoint(path):
        raise NEBPreparationError("synthetic validation failure")

    monkeypatch.setattr(
        "mlipx.neb.io._read_checkpoint_directory", reject_temporary_checkpoint
    )
    with pytest.raises(NEBPreparationError, match="synthetic"):
        store.write(
            (initial, middle, final),
            **{**keyword_args, "stage_step": 1},
        )

    assert (store.root / "latest").resolve() == first
    assert not (store.root / "step_000001").exists()
    assert not list(store.root.glob("*.tmp"))


def test_initial_checkpoint_failure_marks_attempt_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"model identity")
    output = tmp_path / "run"
    initial, final = _endpoints()

    def reject_checkpoint(*args, **kwargs):
        raise OSError("synthetic checkpoint write failure")

    monkeypatch.setattr(NEBCheckpointStore, "write", reject_checkpoint)
    with pytest.raises(OSError, match="synthetic checkpoint"):
        run_neb_workflow(
            FakeMACEWrapper(),
            _resolved(model),
            output_dir=output,
            initial=initial,
            final=final,
            verbose=False,
        )

    context = json.loads((output / "run_context.json").read_text(encoding="utf-8"))
    failure = json.loads((output / "neb_results.json").read_text(encoding="utf-8"))
    artifacts = json.loads((output / "artifacts.json").read_text(encoding="utf-8"))
    assert context["status"] == "failed"
    assert failure["status"] == "failed"
    assert failure["latest_checkpoint"] is None
    assert artifacts["artifacts"]["checkpoint"] is None


def test_new_workflow_rejects_task_pbc_mismatch_before_model_access(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    initial, final = _endpoints()
    initial.pbc = False
    wrapper = FakeMACEWrapper()

    with pytest.raises(NEBPreparationError, match="incompatible with periodic task"):
        run_neb_workflow(
            wrapper,
            _resolved(model),
            output_dir=tmp_path / "run",
            initial=initial,
            final=final,
            verbose=False,
        )
    assert wrapper.get_calls == 0


def test_geometry_resume_does_not_reinterpolate_saved_interior_band() -> None:
    initial, final = _endpoints()
    middle = initial.copy()
    middle.positions = 0.5 * (initial.positions + final.positions)
    middle.positions[1, 1] += 0.4
    band = BandInput(
        images=(initial, middle, final),
        atom_map=np.arange(2),
        image_shifts=np.zeros((2, 3), dtype=int),
        path_convention="mic",
        source="checkpoint_optimized_band",
    )
    options = NEBOptions(
        n_intermediate_images=1,
        endpoint_policy="relax",
        max_steps=0,
        min_distance_A=0.1,
    )
    runner = NEBRunner(ZeroCalculator(), options, verbose=False)

    runner.run(band)

    assert runner.last_band is not None
    assert runner.last_band[1].positions[1, 1] == pytest.approx(middle.positions[1, 1])


# ---------------------------------------------------------------------------
# R08/R09: interruption status and the artifacts checkpoint pointer
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_publishes_cancelled_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C must not leave artifacts/context at status 'running' (R08)."""
    model = tmp_path / "model.pt"
    model.write_bytes(b"model identity")
    output = tmp_path / "interrupted"
    initial, final = _endpoints()

    def interrupt(self, band):
        raise KeyboardInterrupt

    monkeypatch.setattr(NEBRunner, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_neb_workflow(
            FakeMACEWrapper(),
            _resolved(model),
            output_dir=output,
            initial=initial,
            final=final,
            verbose=False,
        )

    failure = json.loads((output / "neb_results.json").read_text(encoding="utf-8"))
    context = json.loads((output / "run_context.json").read_text(encoding="utf-8"))
    artifacts = json.loads((output / "artifacts.json").read_text(encoding="utf-8"))
    assert failure["status"] == "cancelled"
    assert failure["converged"] is False
    assert context["status"] == "cancelled"
    assert artifacts["status"] == "cancelled"
    assert "KeyboardInterrupt" in failure["failure_reason"]
    # the last complete checkpoint stays readable for resume
    checkpoint, images = load_checkpoint(output)
    assert checkpoint["complete"] is True
    assert len(images) == 3


def test_artifacts_pointer_follows_each_published_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TUI reads artifacts.checkpoint.path; it must name the latest stage (R09)."""
    model = tmp_path / "model.pt"
    model.write_bytes(b"model identity")
    output = tmp_path / "run"
    initial, final = _endpoints()
    captured: dict[str, Path] = {}

    def fake_run(self, band):
        self.checkpoint_callback("neb_pre", 0, band.images)
        first = json.loads((output / "artifacts.json").read_text(encoding="utf-8"))
        captured["pre"] = Path(first["artifacts"]["checkpoint"]["path"])
        self.checkpoint_callback("ci_neb", 5, band.images)
        second = json.loads((output / "artifacts.json").read_text(encoding="utf-8"))
        captured["ci"] = Path(second["artifacts"]["checkpoint"]["path"])
        raise KeyboardInterrupt

    monkeypatch.setattr(NEBRunner, "run", fake_run)
    with pytest.raises(KeyboardInterrupt):
        run_neb_workflow(
            FakeMACEWrapper(),
            _resolved(model),
            output_dir=output,
            initial=initial,
            final=final,
            verbose=False,
        )

    first_meta, _ = load_checkpoint(captured["pre"])
    second_meta, _ = load_checkpoint(captured["ci"])
    assert first_meta["stage"] == "neb_pre"
    assert second_meta["stage"] == "ci_neb"
    assert captured["pre"] != captured["ci"]
    # the pointer the TUI resolves is the latest published stage, not the
    # stage=prepared checkpoint written at start-up
    final_artifacts = json.loads(
        (output / "artifacts.json").read_text(encoding="utf-8")
    )
    assert final_artifacts["status"] == "cancelled"
