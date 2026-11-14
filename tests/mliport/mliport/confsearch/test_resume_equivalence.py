"""AUD-007: an interrupted confsearch run must resume into the same result.

The tracker asks whether checkpoint resume truly preserves the structure, the
RNG streams and the model identity.  Model identity is covered by
``test_resume_identity.py``; this module covers the other two by interrupting a
run after **every** durable unit boundary and asserting that the resumed run's
scientific result is identical to an uninterrupted run's.

The engine checkpoints after each unit (``checkpoints/confsearch.json``) and
stores the per-replica ``md_rng``/``mc_rng`` bit-generator states, so the
resumed process must continue the exact same random streams.  The synthetic
``EggCrateCalculator`` is deterministic, so any divergence in the recorded
ranking, minima or diagnostics is a resume defect, not model noise.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from ase.io import write

from mliport.confsearch.engine import SearchEngine
from mliport.confsearch.workflow import run_confsearch_workflow

if TYPE_CHECKING:
    from pathlib import Path


class _Interrupted(RuntimeError):
    """Test-only stand-in for a process that dies mid-run."""


class _SyntheticWrapper:
    """Backend-shaped wrapper around a synthetic calculator (no MLIP needed)."""

    task = "bulk"

    def __init__(self, calculator, model_path) -> None:
        self._calculator = calculator
        self._model_path = model_path

    def info(self) -> dict:
        return {
            "model_type": "mace",
            "task": "bulk",
            "backend_version": "test",
            "framework_versions": {"torch": "test"},
            "default_dtype": "float64",
            "model_path": str(self._model_path),
        }

    def get_calculator(self):
        return self._calculator


def _setup(run_dir: Path, *, cycles: int, frames: int):
    from mliport.config import resolve_config
    from tests.mliport.mliport.confsearch.calculators import EggCrateCalculator
    from tests.mliport.mliport.confsearch.test_engine import _start

    run_dir.mkdir(parents=True, exist_ok=True)
    model_file = run_dir / "model.pt"
    model_file.write_bytes(b"identity model")
    start_file = run_dir / "start.extxyz"
    sites = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25)][:frames]
    write(start_file, [_start(site) for site in sites], format="extxyz")
    resolved = resolve_config(
        calc_type="confsearch",
        cli={
            "model_type": "mace",
            "model_path": str(model_file),
            "task": "bulk",
            "cs_starts": str(start_file),
            "cs_adsorbate_indices": "16-18",
            "cs_cycles": cycles,
            "cs_hold_ps": 0.05,
            "cs_cool_ps": 0.05,
            "cs_snapshot_interval_ps": 0.02,
            "timestep": 0.5,
            "cs_temperature_high": 800.0,
            "cs_temperature_low": 300.0,
            "cs_quench_fmax": 0.01,
            "cs_quench_steps": 2000,
            "cs_refine": False,
            "cs_refine_fmax": 0.005,
            "seed": 7,
        },
    )
    return resolved, _SyntheticWrapper(EggCrateCalculator(), model_file)


def _run(
    run_dir: Path,
    *,
    cycles: int,
    frames: int,
    stop_after_units: int | None = None,
) -> dict:
    """Run one search, optionally killing it right after ``stop_after_units``."""

    resolved, wrapper = _setup(run_dir, cycles=cycles, frames=frames)
    output = run_dir / "run"

    if stop_after_units is None:
        return run_confsearch_workflow(wrapper, resolved, output_dir=output).results

    original = SearchEngine._write_checkpoint
    calls = {"n": 0}

    def interrupt_after_checkpoint(self) -> None:
        calls["n"] += 1
        if calls["n"] > stop_after_units:
            # The checkpoint of unit ``stop_after_units`` is already on disk.
            raise _Interrupted(f"killed after {stop_after_units} unit(s)")
        original(self)

    SearchEngine._write_checkpoint = interrupt_after_checkpoint
    try:
        with pytest.raises(_Interrupted):
            run_confsearch_workflow(wrapper, resolved, output_dir=output)
    finally:
        SearchEngine._write_checkpoint = original

    checkpoint = json.loads(
        (output / "checkpoints" / "confsearch.json").read_text(encoding="utf-8")
    )
    assert checkpoint["unit_index"] == stop_after_units

    return run_confsearch_workflow(
        wrapper, resolved, output_dir=output, resume=True
    ).results


def _comparable(results: dict) -> dict:
    """Drop the model *path*: where a model was stored is not identity."""

    payload = json.loads(json.dumps(results, default=str))
    model = payload.get("model")
    if isinstance(model, dict):
        model.pop("model_path", None)
    return payload


@pytest.mark.parametrize(("frames", "cycles"), [(1, 3), (2, 2)])
def test_resume_reproduces_an_uninterrupted_run(tmp_path: Path, frames, cycles) -> None:
    """Every durable boundary must resume into the identical scientific result."""

    n_units = 1 + frames * cycles  # one start unit, then one unit per replica-cycle
    reference = _comparable(_run(tmp_path / "reference", cycles=cycles, frames=frames))

    for stop_after in range(1, n_units):
        resumed = _comparable(
            _run(
                tmp_path / f"stop_{stop_after}",
                cycles=cycles,
                frames=frames,
                stop_after_units=stop_after,
            )
        )
        assert resumed == reference, f"resume after unit {stop_after} diverged"


def test_resume_restores_the_recorded_rng_states(tmp_path: Path) -> None:
    """The checkpoint must carry per-replica md/mc generator states."""

    run_dir = tmp_path / "rng"
    _run(run_dir, cycles=1, frames=2, stop_after_units=1)

    checkpoint = json.loads(
        (run_dir / "run" / "checkpoints" / "confsearch.json").read_text(
            encoding="utf-8"
        )
    )
    replicas = checkpoint["rng_states"]["replicas"]
    assert [entry["index"] for entry in replicas] == [0, 1]
    for entry in replicas:
        for stream in ("md", "mc"):
            state = entry[stream]
            assert state["bit_generator"]
            assert isinstance(state["state"], dict)
    # Different replicas must not share one generator state.
    assert replicas[0]["md"] != replicas[1]["md"]
