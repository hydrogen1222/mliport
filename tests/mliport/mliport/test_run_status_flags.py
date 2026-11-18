"""Status-classification boundary contract (R10-F06 hardening).

``classify_result`` decides a scientific outcome, so it must never let a value
that is not unambiguously a boolean become "completed":

* ``np.bool_`` (ASE optimizers, DeepMD) is a legitimate flag but is not the
  ``False`` singleton -- the original ``is False`` test missed it;
* ``bool("False")`` and ``bool(float("nan"))`` are ``True``, so a blanket
  ``bool()`` coercion would promote an unconverged/unknown run to success;
* the same applies to the ``success`` flag.

These tests pin the accepted flag types, the rejected (ambiguous) types, and
the unchanged precedence of the explicit ``status`` string.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

import mliport.runners.batch as batch_module
from mliport.run_status import (
    EXIT_ERROR,
    EXIT_NOT_CONVERGED,
    EXIT_SUCCESS,
    RunOutcome,
    classify_result,
    exit_code_for,
)
from mliport.runners.batch import BatchRunner

TRUE_FLAGS = [True, np.True_, np.array(True), 1, np.int64(1)]
FALSE_FLAGS = [False, np.False_, np.array(False), 0, np.int64(0)]
AMBIGUOUS_FLAGS = [
    "False",
    "True",
    "",
    "0",
    1.0,
    0.0,
    math.nan,
    np.float64(0.0),
    np.float64(1.0),
    [False],
    np.array([True, False]),
    np.array([False, False]),
    b"False",
    object(),
]


@pytest.mark.parametrize("flag", TRUE_FLAGS)
def test_accepted_true_flags_are_completed(flag):
    assert classify_result({"converged": flag}, calc_type="opt") is RunOutcome.COMPLETED
    # a success-only result is judged on the non-convergence contract, because
    # a convergence-type run without a `converged` flag is never success
    assert classify_result({"success": flag}, calc_type="md") is RunOutcome.COMPLETED


@pytest.mark.parametrize("flag", FALSE_FLAGS)
def test_accepted_false_flags_are_never_success(flag):
    assert (
        classify_result({"converged": flag}, calc_type="opt")
        is RunOutcome.NOT_CONVERGED
    )
    assert classify_result({"success": flag}, calc_type="md") is RunOutcome.FAILED


@pytest.mark.parametrize("flag", AMBIGUOUS_FLAGS)
def test_ambiguous_converged_flag_is_failed_not_completed(flag):
    outcome = classify_result({"converged": flag}, calc_type="opt")
    assert outcome is RunOutcome.FAILED
    assert outcome is not RunOutcome.COMPLETED


@pytest.mark.parametrize("flag", AMBIGUOUS_FLAGS)
def test_ambiguous_success_flag_is_failed_not_completed(flag):
    outcome = classify_result({"success": flag}, calc_type="md")
    assert outcome is RunOutcome.FAILED


def test_ambiguous_flag_does_not_mask_a_real_failure():
    """An unreadable flag must not upgrade failed/cancelled to anything else."""
    assert (
        classify_result({"status": "failed", "converged": "False"}, calc_type="opt")
        is RunOutcome.FAILED
    )
    assert (
        classify_result({"status": "cancelled", "success": math.nan}, calc_type="md")
        is RunOutcome.CANCELLED
    )
    assert (
        classify_result(
            {"status": "not_converged", "converged": "True"}, calc_type="opt"
        )
        is RunOutcome.NOT_CONVERGED
    )


def test_missing_flags_keep_their_documented_defaults():
    # convergence-type runs without a flag are not converged ...
    assert classify_result({}, calc_type="opt") is RunOutcome.NOT_CONVERGED
    assert classify_result({}, calc_type="neb") is RunOutcome.NOT_CONVERGED
    # ... while SP/MD have no convergence concept
    assert classify_result({}, calc_type="md") is RunOutcome.COMPLETED
    assert classify_result({}, calc_type="sp") is RunOutcome.COMPLETED


def test_both_flags_are_checked_together():
    assert (
        classify_result({"converged": np.True_, "success": np.False_}, calc_type="opt")
        is RunOutcome.FAILED
    )
    assert (
        classify_result({"converged": np.False_, "success": np.True_}, calc_type="opt")
        is RunOutcome.NOT_CONVERGED
    )


def test_exit_codes_are_unchanged():
    assert exit_code_for(RunOutcome.COMPLETED) == EXIT_SUCCESS == 0
    assert exit_code_for(RunOutcome.NOT_CONVERGED) == EXIT_NOT_CONVERGED == 2
    assert exit_code_for(RunOutcome.FAILED) == EXIT_ERROR == 1


class _SyntheticOptimizationRunner:
    """Runner stub whose returned flags are chosen by the test."""

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
    ("flags", "expected_status"),
    [
        ({"converged": np.True_}, "completed"),
        ({"converged": 1}, "completed"),
        ({"converged": np.False_}, "not_converged"),
        ({"converged": 0}, "not_converged"),
        ({"converged": "False"}, "failed"),
        # not-converged keeps its documented precedence over failed
        ({"converged": np.False_, "success": np.False_}, "not_converged"),
        ({"success": np.False_}, "not_converged"),
    ],
)
def test_batch_reports_the_hardened_outcome(
    tmp_path, monkeypatch, flags, expected_status
):
    """The batch summary must not call any of these a success."""
    expected_success = expected_status == "completed"
    monkeypatch.setattr(
        batch_module, "OptimizationRunner", _SyntheticOptimizationRunner
    )
    monkeypatch.setattr(_SyntheticOptimizationRunner, "result", flags)
    structure = tmp_path / "sample.xyz"
    write(structure, Atoms("Li", positions=[[0, 0, 0]]))

    runner = BatchRunner(
        object(), calc_type="opt", output_dir=tmp_path / "batch", verbose=False
    )
    summary = runner.run_from_files([structure])
    item = summary["results"][0]
    assert item["status"] == expected_status
    assert item["success"] is expected_success
    assert summary["not_converged"] == int(expected_status == "not_converged")
    assert summary["failed"] == int(expected_status != "completed")
