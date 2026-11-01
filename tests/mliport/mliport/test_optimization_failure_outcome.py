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
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mliport.protocols import CancellationRequested
from mliport.run_status import RunOutcome, classify_result
from mliport.runners.optimization import OptimizationRunner

if TYPE_CHECKING:
    from pathlib import Path

    from mliport.protocols import ProgressEvent


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


class _ExplodingOptimizerAfterMove(_ExplodingOptimizer):
    """Raises after moving an atom, invalidating the calculator cache."""

    def run(self, fmax=None, steps=None) -> None:
        self.nsteps = 1
        for observer in self._observers:
            observer()
        self.atoms.positions = self.atoms.positions + 0.01
        raise _BackendCrash("simulated backend crash at step 2")


class _NonConvergingOptimizer:
    """Records one step and reports non-convergence without raising."""

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


class _FailAfterFirstEvaluation(Calculator):
    """Works for the optimizer callback, then fails on the re-evaluation."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        self.calls += 1
        if self.calls > 1:
            raise _BackendCrash("simulated backend crash after step 1")
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": -1.0,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }


class _CancelOnSecondEvaluation(Calculator):
    """Cancellation raised by the post-crash re-evaluation must pass through."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        self.calls += 1
        if self.calls > 1:
            raise CancellationRequested("cancelled during the final evaluation")
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": -1.0,
            "forces": np.zeros((len(atoms), 3)),
        }


def _run_optimization(
    tmp_path: Path,
    monkeypatch,
    optimizer_class,
    *,
    calculator=None,
    progress_callback=None,
) -> dict:
    monkeypatch.setitem(OptimizationRunner.OPTIMIZERS, "bfgs", optimizer_class)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True)
    runner = OptimizationRunner(
        _WrapperStub(calculator or _WorkingCalculator()),
        optimizer="bfgs",
        fmax=0.05,
        max_steps=10,
        output_dir=tmp_path,
        verbose=False,
        progress_callback=progress_callback,
    )
    return runner.execute(atoms)


def _run_exploding_optimization(
    tmp_path: Path, monkeypatch, *, calculator=None, progress_callback=None
) -> dict:
    return _run_optimization(
        tmp_path,
        monkeypatch,
        _ExplodingOptimizer,
        calculator=calculator,
        progress_callback=progress_callback,
    )


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


def test_failed_optimization_publishes_error_not_done(
    tmp_path: Path, monkeypatch
) -> None:
    """A crashed run must not end on a success-looking ``done`` progress event."""

    events: list[ProgressEvent] = []
    _run_exploding_optimization(tmp_path, monkeypatch, progress_callback=events.append)

    phases = [event.phase for event in events]
    assert "error" in phases
    assert "done" not in phases
    assert phases[-1] == "error"
    assert "simulated backend crash" in events[-1].message
    # ``execute`` still attaches the timing to the terminal event.
    assert events[-1].extra["timing"]


def test_honest_non_convergence_still_publishes_done(
    tmp_path: Path, monkeypatch
) -> None:
    """Hitting max_steps is a normal finish, not a failure."""

    events: list[ProgressEvent] = []
    results = _run_optimization(
        tmp_path,
        monkeypatch,
        _NonConvergingOptimizer,
        progress_callback=events.append,
    )

    assert results["converged"] is False
    assert "status" not in results
    phases = [event.phase for event in events]
    assert "error" not in phases
    assert phases[-1] == "done"
    assert "not converged" in events[-1].message


def test_reevaluation_failure_keeps_the_original_crash(
    tmp_path: Path, monkeypatch
) -> None:
    """If the post-crash re-evaluation also fails, both causes must survive."""

    monkeypatch.setitem(
        OptimizationRunner.OPTIMIZERS, "bfgs", _ExplodingOptimizerAfterMove
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True)
    runner = OptimizationRunner(
        _WrapperStub(_FailAfterFirstEvaluation()),
        optimizer="bfgs",
        fmax=0.05,
        max_steps=10,
        output_dir=tmp_path,
        verbose=False,
    )

    with pytest.raises(RuntimeError, match="simulated backend crash") as excinfo:
        runner.execute(atoms)

    message = str(excinfo.value)
    assert "Optimization failed" in message
    assert "energy/force evaluation also failed" in message
    assert isinstance(excinfo.value.__cause__, _BackendCrash)


def test_cancellation_during_reevaluation_is_not_wrapped(
    tmp_path: Path, monkeypatch
) -> None:
    """A cancel must stay a cancel, not become a wrapped runtime failure."""

    monkeypatch.setitem(
        OptimizationRunner.OPTIMIZERS, "bfgs", _ExplodingOptimizerAfterMove
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True)
    runner = OptimizationRunner(
        _WrapperStub(_CancelOnSecondEvaluation()),
        optimizer="bfgs",
        fmax=0.05,
        max_steps=10,
        output_dir=tmp_path,
        verbose=False,
    )

    with pytest.raises(CancellationRequested):
        runner.execute(atoms)
