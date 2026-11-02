"""AUD-BATCH-001: preserve true per-structure outcomes in batch reports."""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

import mliport.runners.batch as batch_module
from mliport.runners.batch import BatchRunner


class _SyntheticOptimizationRunner:
    """Isolated runner whose outcome is chosen by the test, no model needed."""

    result: ClassVar[dict] = {}

    def __init__(self, calculator, **kwargs):
        self.calculator = calculator

    def execute(self, atoms):
        return {
            "energy": -1.0,
            "forces": np.zeros((len(atoms), 3)),
            "time": 0.01,
            **self.result,
        }


@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_success"),
    [
        ({"converged": True}, "completed", True),
        ({"converged": False}, "not_converged", False),
        (
            {"converged": False, "status": "failed", "error": "backend crash"},
            "failed",
            False,
        ),
    ],
)
def test_batch_optimization_reports_the_real_outcome(
    tmp_path, monkeypatch, outcome, expected_status, expected_success
):
    """A normally returned failed/unfinished optimization is not a success."""
    monkeypatch.setattr(
        batch_module, "OptimizationRunner", _SyntheticOptimizationRunner
    )
    monkeypatch.setattr(_SyntheticOptimizationRunner, "result", outcome)
    structure = tmp_path / "sample.xyz"
    write(structure, Atoms("Li", positions=[[0, 0, 0]]))

    runner = BatchRunner(
        object(), calc_type="opt", output_dir=tmp_path / "batch", verbose=False
    )
    summary = runner.run_from_files([structure])
    item = summary["results"][0]

    assert item["status"] == expected_status
    assert item["success"] is expected_success
    assert item["converged"] is outcome["converged"]
    assert item["energy"] == -1.0
    assert summary["success"] == int(expected_success)
    assert summary["failed"] == int(not expected_success)
    assert summary["not_converged"] == int(expected_status == "not_converged")
    assert summary["total"] == summary["success"] + summary["failed"]

    saved = json.loads((runner.output_dir / "batch_summary.json").read_text())
    assert saved["results"][0]["status"] == expected_status
    assert saved["results"][0]["success"] is expected_success
    assert saved["metadata"]["not_converged"] == summary["not_converged"]
    if expected_status == "failed":
        assert item["error"] == "backend crash"
        assert saved["results"][0]["error"] == "backend crash"


def test_batch_exception_is_still_recorded_as_failed(tmp_path, monkeypatch):
    """An escaping exception is not mistaken for a normal returned result."""

    class _CrashingRunner(_SyntheticOptimizationRunner):
        def execute(self, atoms):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(batch_module, "OptimizationRunner", _CrashingRunner)
    structure = tmp_path / "sample.xyz"
    write(structure, Atoms("Li", positions=[[0, 0, 0]]))
    runner = BatchRunner(
        object(), calc_type="opt", output_dir=tmp_path / "batch", verbose=False
    )

    summary = runner.run_from_files([structure])
    assert summary["success"] == 0
    assert summary["failed"] == 1
    assert summary["not_converged"] == 0
    item = summary["results"][0]
    assert item["status"] == "failed"
    assert item["success"] is False
    assert "model unavailable" in item["error"]
