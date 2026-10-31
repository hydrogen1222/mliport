"""AUD-003: an optimizer exception must not be reported as plain non-convergence.

``OptimizationRunner.run`` catches every non-NaN optimizer/calculator
exception, logs it, then re-evaluates energy/forces and returns results with
``converged=False``.  ``classify_result`` maps that to ``not_converged``
(exit 2), so a crashed run is indistinguishable from an honest "hit
max_steps" run in the exit code, the queue job status and
``mliport_results.json`` -- the original error survives only in the log.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mliport.run_status import RunOutcome, classify_result
from mliport.runners.optimization import OptimizationRunner

if TYPE_CHECKING:
    from pathlib import Path


class _BackendCrash(RuntimeError):
    """Stand-in for a model/backend failure that is not a NaN abort."""


class _WorkingCalculator(Calculator):
    """Finite calculator so the post-exception re-evaluation can succeed."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=float)
        self.results = {
            "energy": float(np.sum(positions[:, 0])),
            "forces": np.full(positions.shape, 0.5),
            "stress": np.zeros(6),
        }


class _WrapperStub:
    """Minimal ``BaseMLIPCalculator`` stub for the runners."""

    def __init__(self, calc, task="bulk", has_stress=True) -> None:
        self._calc = calc
        self._task = task
        self._has_stress = has_stress

    def get_calculator(self):
        return self._calc

    @property
    def task(self):
        return self._task

    @property
    def has_stress(self):
        return self._has_stress

    @property
    def implemented_properties(self):
        return list(self._calc.implemented_properties)

    def info(self):
        return {"model_type": "stub", "task": self._task}


class _ExplodingOptimizer:
    """Optimizer that records one step, then raises a backend-style error."""

    def __init__(self, atoms, logfile=None) -> None:
        self.atoms = atoms
        self.nsteps = 0
        self._observers = []

    def attach(self, observer) -> None:
        self._observers.append(observer)

    def converged(self) -> bool:
        return False

    def run(self, fmax=None, steps=None) -> None:
        self.nsteps = 1
        for observer in self._observers:
            observer()
        raise _BackendCrash("simulated backend crash at step 2")


def _run_exploding_optimization(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setitem(OptimizationRunner.OPTIMIZERS, "bfgs", _ExplodingOptimizer)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True)
    runner = OptimizationRunner(
        _WrapperStub(_WorkingCalculator()),
        optimizer="bfgs",
        fmax=0.05,
        max_steps=10,
        output_dir=tmp_path,
        verbose=False,
    )
    return runner.run(atoms)


def test_optimizer_exception_is_reported_as_failed_not_non_convergence(
    tmp_path: Path, monkeypatch
) -> None:
    results = _run_exploding_optimization(tmp_path, monkeypatch)

    # Partial coordinates may still be kept, but the run did not "finish and
    # miss the criterion": it crashed, so the outcome must be ``failed``.
    assert results["converged"] is False
    assert classify_result(results, calc_type="opt") is RunOutcome.FAILED


def test_optimizer_exception_is_recorded_in_results_json(
    tmp_path: Path, monkeypatch
) -> None:
    _run_exploding_optimization(tmp_path, monkeypatch)

    payload = json.loads(
        (tmp_path / "mliport_results.json").read_text(encoding="utf-8")
    )
    assert "simulated backend crash" in json.dumps(payload)
