"""Batch status/summary serialisation contract (R10-F06b).

A backend may hand back values that are not JSON numbers: ``np`` scalars, NaN,
infinity, arrays, or arbitrary objects.  Before this contract was pinned, such a
value aborted the **whole batch** at the summary write with

    ValueError: Out of range float values are not JSON compliant: nan

or ``TypeError: Object of type object is not JSON serializable`` -- after every
structure had already run, and leaving a partially written ``batch_summary.json``
behind.  What must happen instead:

* the batch finishes and writes a strictly valid summary;
* the offending structure is recorded as ``failed`` (never completed, never with
  a fabricated zero);
* what was replaced is recorded (``non_finite_fields`` / ``unserialisable_fields``);
* the other structures keep their own status files and outputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

import mliport.runners.batch as batch_module
from mliport.runners.batch import BatchRunner, BatchSummaryWriteError
from mliport.writers.json_writer import JsonWriter

BAD_RESULTS = {
    "nan_converged": {"converged": math.nan},
    "nan_energy": {"energy": math.nan, "converged": True},
    "inf_energy": {"energy": math.inf, "converged": True},
    "nan_forces": {"forces": np.full((1, 3), np.nan), "converged": True},
    "object_energy": {"energy": object(), "converged": True},
    "array_energy": {"energy": np.array([1.0, 2.0]), "converged": True},
}
GOOD_RESULTS = {
    "plain": {"energy": -1.0, "converged": True},
    "numpy_scalars": {
        "energy": np.float32(-1.5),
        "converged": np.bool_(True),
        "fmax": np.float64(0.25),
        "nsteps": np.int64(7),
    },
}


class _Runner:
    result: ClassVar[dict] = {}

    def __init__(self, calculator, **kwargs):
        self.calculator = calculator

    def execute(self, atoms):
        n = len(atoms)
        return {
            "energy": -1.0,
            "forces": np.zeros((n, 3)),
            "time": 0.01,
            **self.result,
        }


def _strict_load(path):
    """Fail on NaN/Infinity tokens, which Python's json accepts by default."""

    def reject(token):
        raise ValueError(f"non-finite token {token!r} in {path}")

    return json.loads(path.read_text(), parse_constant=reject)


def _batch(tmp_path, structures, *, runner_results=None):
    monkeypatch_runner = runner_results if runner_results is not None else {}
    _Runner.result = monkeypatch_runner
    original = batch_module.OptimizationRunner
    batch_module.OptimizationRunner = _Runner
    try:
        runner = BatchRunner(
            object(), calc_type="opt", output_dir=tmp_path / "out", verbose=False
        )
        return runner, runner.run_from_files(structures)
    finally:
        batch_module.OptimizationRunner = original


def _structures(tmp_path, count=3):
    paths = []
    for index in range(count):
        path = tmp_path / f"s{index}.xyz"
        if not path.exists():
            write(
                path,
                Atoms(
                    "Li", positions=[[0, 0, float(index)]], cell=[10, 10, 10], pbc=True
                ),
            )
        paths.append(path)
    return paths


@pytest.mark.parametrize("case", sorted(BAD_RESULTS))
def test_bad_backend_values_never_abort_the_batch(tmp_path, case):
    structures = _structures(tmp_path)
    runner, summary = _batch(tmp_path, structures, runner_results=BAD_RESULTS[case])

    assert summary["total"] == 3
    assert summary["success"] == 0
    assert summary["failed"] == 3
    assert all(item["status"] == "failed" for item in summary["results"])
    assert all(item["success"] is False for item in summary["results"])

    out = runner.output_dir
    summary_json = _strict_load(out / "batch_summary.json")
    assert summary_json["metadata"]["failed"] == 3
    for path in sorted(out.rglob("batch_status.json")):
        record = _strict_load(path)
        assert record["status"] == "failed"
        assert record["success"] is False


@pytest.mark.parametrize("case", sorted(GOOD_RESULTS))
def test_numpy_scalars_are_stored_as_plain_json_numbers(tmp_path, case):
    structures = _structures(tmp_path)
    runner, summary = _batch(tmp_path, structures, runner_results=GOOD_RESULTS[case])

    assert summary["success"] == 3
    out = runner.output_dir
    _strict_load(out / "batch_summary.json")
    record = _strict_load(out / "s0" / "batch_status.json")
    assert record["status"] == "completed"
    assert record["success"] is True
    assert record["energy"] == pytest.approx(float(GOOD_RESULTS[case]["energy"]))


@pytest.mark.parametrize("case", ["nan_energy", "inf_energy", "object_energy"])
def test_invalid_observable_is_recorded_as_failed_not_as_success(tmp_path, case):
    structures = _structures(tmp_path)
    runner, _summary = _batch(tmp_path, structures, runner_results=BAD_RESULTS[case])

    record = _strict_load(runner.output_dir / "s0" / "batch_status.json")
    assert record["status"] == "failed"
    assert record["success"] is False
    assert "energy" in record["invalid_numeric_fields"]
    assert "non-finite or non-numeric energy" in record["error"]
    if case == "object_energy":
        assert record["energy"] == "<unserialisable object>"
    else:
        assert record["energy"] is None  # never a fabricated 0.0


def test_nan_is_recorded_where_it_was_replaced(tmp_path):
    structures = _structures(tmp_path)
    runner, _summary = _batch(
        tmp_path, structures, runner_results={"energy": math.nan, "converged": True}
    )
    record = _strict_load(runner.output_dir / "s0" / "batch_status.json")
    assert record["non_finite_fields"] == ["energy"]


def test_unserialisable_value_is_named_not_silently_dropped(tmp_path):
    structures = _structures(tmp_path)
    runner, _summary = _batch(
        tmp_path, structures, runner_results={"energy": object(), "converged": True}
    )
    record = _strict_load(runner.output_dir / "s0" / "batch_status.json")
    assert record["unserialisable_fields"] == ["energy"]
    assert record["energy"] == "<unserialisable object>"


def test_good_structures_keep_their_status_and_outputs(tmp_path):
    """One bad structure must not disturb the completed ones around it."""
    good = _structures(tmp_path, 2)
    bad = tmp_path / "bad.xyz"
    write(
        bad,
        Atoms("Li2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True),
    )
    structures = [good[0], bad, good[1]]

    original_execute = _Runner.execute

    def execute(this, atoms):
        # the two-atom structure is the one whose backend reports NaN
        if len(atoms) == 2:
            return {"energy": math.nan, "converged": True, "forces": np.zeros((2, 3))}
        return {"energy": -2.0, "converged": True, "forces": np.zeros((len(atoms), 3))}

    _Runner.execute = execute
    original = batch_module.OptimizationRunner
    batch_module.OptimizationRunner = _Runner
    try:
        runner = BatchRunner(
            object(), calc_type="opt", output_dir=tmp_path / "out", verbose=False
        )
        summary = runner.run_from_files(structures)
    finally:
        batch_module.OptimizationRunner = original
        _Runner.execute = original_execute

    assert [item["status"] for item in summary["results"]] == [
        "completed",
        "failed",
        "completed",
    ]
    out = runner.output_dir
    assert _strict_load(out / "s0" / "batch_status.json")["status"] == "completed"
    assert _strict_load(out / "bad" / "batch_status.json")["status"] == "failed"
    assert _strict_load(out / "s1" / "batch_status.json")["status"] == "completed"


def test_status_write_failure_fails_that_structure_only(tmp_path, monkeypatch):
    """A status file that cannot be written is a failed structure, not an abort."""
    structures = _structures(tmp_path)
    _Runner.result = {"energy": -1.0, "converged": True}
    original_write = batch_module._atomic_write_json
    calls = {"n": 0}

    def failing_write(path, payload):
        if path.name == "batch_status.json":
            calls["n"] += 1
            if calls["n"] == 2:  # fail exactly one structure
                raise OSError(28, "No space left on device")
        return original_write(path, payload)

    monkeypatch.setattr(batch_module, "_atomic_write_json", failing_write)
    monkeypatch.setattr(batch_module, "OptimizationRunner", _Runner)
    runner = BatchRunner(
        object(), calc_type="opt", output_dir=tmp_path / "out", verbose=False
    )
    summary = runner.run_from_files(structures)

    assert summary["total"] == 3
    assert summary["success"] == 2
    assert summary["failed"] == 1
    failed = [item for item in summary["results"] if item["status"] == "failed"]
    assert len(failed) == 1
    assert "status record could not be written" in failed[0]["error"]
    assert "No space left on device" in failed[0]["error"]
    # the successful records are still on disk and valid
    records = [
        _strict_load(p) for p in sorted(runner.output_dir.rglob("batch_status.json"))
    ]
    assert [r["status"] for r in records] == ["completed", "failed", "completed"]


def test_summary_write_failure_keeps_an_auditable_minimal_summary(
    tmp_path, monkeypatch
):
    structures = _structures(tmp_path)
    _Runner.result = {"energy": -1.0, "converged": True}

    def exploding_write_batch(self, results_list, output_path, metadata=None):
        raise ValueError("simulated encoder failure")

    monkeypatch.setattr(JsonWriter, "write_batch", exploding_write_batch)
    monkeypatch.setattr(batch_module, "OptimizationRunner", _Runner)
    runner = BatchRunner(
        object(), calc_type="opt", output_dir=tmp_path / "out", verbose=False
    )
    summary = runner.run_from_files(structures)

    assert summary["success"] == 3  # the calculations themselves did complete
    assert summary["summary_status"] == "degraded"
    out = runner.output_dir
    minimal = _strict_load(out / "batch_summary.json")
    # a degraded document must say so instead of looking like a full summary
    assert minimal["summary_status"] == "degraded"
    assert minimal["metadata"]["summary_status"] == "degraded"
    assert minimal["results"] == []
    assert minimal["metadata"]["summary_write_error"].startswith("ValueError:")
    assert [item["status"] for item in minimal["metadata"]["per_structure_status"]] == [
        "completed"
    ] * 3
    # no half-written temporary file is left behind
    assert list(out.glob(".batch_summary.json.tmp-*")) == []


def test_failed_structure_is_recomputed_on_resume(tmp_path):
    """A sanitised failed record is not reusable: resume recomputes it."""
    structures = _structures(tmp_path, 1)
    runner, first = _batch(
        tmp_path, structures, runner_results={"energy": math.nan, "converged": True}
    )
    assert first["failed"] == 1

    _Runner.result = {"energy": -1.0, "converged": True}
    original = batch_module.OptimizationRunner
    batch_module.OptimizationRunner = _Runner
    try:
        resumed = BatchRunner(
            object(),
            calc_type="opt",
            output_dir=runner.output_dir,
            verbose=False,
            resume=True,
        )
        second = resumed.run_from_files(structures)
    finally:
        batch_module.OptimizationRunner = original

    assert second["success"] == 1
    assert second["reused"] == 0
    assert (
        _strict_load(runner.output_dir / "s0" / "batch_status.json")["status"]
        == "completed"
    )


def test_successful_summary_is_reported_as_written(tmp_path):
    structures = _structures(tmp_path)
    runner, summary = _batch(
        tmp_path, structures, runner_results={"energy": -1.0, "converged": True}
    )
    assert summary["summary_status"] == "written"
    document = _strict_load(runner.output_dir / "batch_summary.json")
    assert "summary_status" not in document  # the full document is not degraded
    assert len(document["results"]) == 3


def test_both_summary_writes_failing_raises_and_keeps_structure_records(
    tmp_path, monkeypatch
):
    """If nothing can be recorded, the caller must hear it — and records stay."""
    structures = _structures(tmp_path)
    _Runner.result = {"energy": -1.0, "converged": True}
    original_write = batch_module._atomic_write_json

    def exploding_write_batch(self, results_list, output_path, metadata=None):
        raise ValueError("simulated encoder failure")

    def failing_write(path, payload):
        if path.name == "batch_summary.json":
            raise OSError(28, "No space left on device")
        return original_write(path, payload)

    monkeypatch.setattr(JsonWriter, "write_batch", exploding_write_batch)
    monkeypatch.setattr(batch_module, "_atomic_write_json", failing_write)
    monkeypatch.setattr(batch_module, "OptimizationRunner", _Runner)
    runner = BatchRunner(
        object(), calc_type="opt", output_dir=tmp_path / "out", verbose=False
    )

    with pytest.raises(BatchSummaryWriteError) as excinfo:
        runner.run_from_files(structures)

    message = str(excinfo.value)
    assert "full summary failed with ValueError: simulated encoder failure" in message
    assert "degraded summary also failed with OSError" in message
    assert "No space left on device" in message

    # every structure still has its own valid, completed record
    records = [
        _strict_load(p) for p in sorted(runner.output_dir.rglob("batch_status.json"))
    ]
    assert [record["status"] for record in records] == ["completed"] * 3
    assert all(record["success"] is True for record in records)
    # nothing half-written is left behind
    assert list(runner.output_dir.glob(".batch_summary.json.tmp-*")) == []
    assert not (runner.output_dir / "batch_summary.json").exists()


def test_no_temporary_summary_files_survive(tmp_path, monkeypatch):
    structures = _structures(tmp_path)
    _Runner.result = {"energy": -1.0, "converged": True}

    def exploding_write_batch(self, results_list, output_path, metadata=None):
        # write half a document, then fail, like a real encoder panic
        Path(output_path).write_text('{"results": [')
        raise ValueError("simulated encoder failure")

    monkeypatch.setattr(JsonWriter, "write_batch", exploding_write_batch)
    monkeypatch.setattr(batch_module, "OptimizationRunner", _Runner)
    runner = BatchRunner(
        object(), calc_type="opt", output_dir=tmp_path / "out", verbose=False
    )
    summary = runner.run_from_files(structures)

    assert summary["summary_status"] == "degraded"
    assert list(runner.output_dir.glob(".batch_summary.json.tmp-*")) == []
    assert (
        _strict_load(runner.output_dir / "batch_summary.json")["summary_status"]
        == "degraded"
    )
