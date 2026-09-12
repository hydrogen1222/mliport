"""Unified run-outcome contract shared by the CLI, TUI and queue.

Every frontend must resolve a finished calculation to exactly one outcome and
one process exit code (review R07):

* ``completed``      -- ``0``
* ``not_converged``  -- ``2`` (the run finished but missed its convergence
  criterion; partial coordinates and the reason are kept)
* ``failed``         -- ``1``
* ``cancelled``      -- ``130`` (user interrupt / cooperative cancellation)

NEB already used 0/2; the same contract now covers ``run``/``sp``/``opt``/
``md``, the queue worker and the TUI, so an unconverged optimization can never
be reported as a completed job just because the process exited cleanly.
"""

from __future__ import annotations

import signal
from enum import Enum
from typing import Any

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_NOT_CONVERGED = 2
EXIT_CANCELLED = 130

#: Calculation types whose result can be "not converged" rather than merely
#: finished.  A result from these types that carries no convergence flag at
#: all is treated as not converged (fail closed, never claim success).
CONVERGENCE_CALC_TYPES = frozenset({"opt", "neb"})


class RunOutcome(str, Enum):
    """Terminal state of one calculation attempt."""

    COMPLETED = "completed"
    NOT_CONVERGED = "not_converged"
    FAILED = "failed"
    CANCELLED = "cancelled"


_OUTCOME_TO_EXIT = {
    RunOutcome.COMPLETED: EXIT_SUCCESS,
    RunOutcome.NOT_CONVERGED: EXIT_NOT_CONVERGED,
    RunOutcome.FAILED: EXIT_ERROR,
    RunOutcome.CANCELLED: EXIT_CANCELLED,
}

#: Job-record status values (``mlipx.jobs.JobStatus`` values, kept as plain
#: strings here so this module has no dependency on the jobs package).
_OUTCOME_TO_JOB_STATUS = {
    RunOutcome.COMPLETED: "done",
    RunOutcome.NOT_CONVERGED: "not_converged",
    RunOutcome.FAILED: "failed",
    RunOutcome.CANCELLED: "cancelled",
}


def classify_result(result: Any, *, calc_type: str | None = None) -> RunOutcome:
    """Map a runner result dictionary to a terminal outcome."""
    if not isinstance(result, dict):
        return RunOutcome.FAILED
    status = str(result.get("status", "")).lower()
    if status in {"cancelled", "canceled"}:
        return RunOutcome.CANCELLED
    if status in {"failed", "error"}:
        return RunOutcome.FAILED
    if status in {"not_converged", "unconverged"}:
        return RunOutcome.NOT_CONVERGED

    converged = result.get("converged")
    if converged is False:
        return RunOutcome.NOT_CONVERGED
    if converged is None and calc_type in CONVERGENCE_CALC_TYPES:
        # A convergence-type run without an explicit flag must never be
        # promoted to success.
        return RunOutcome.NOT_CONVERGED
    if result.get("success") is False:
        return RunOutcome.FAILED
    return RunOutcome.COMPLETED


def classify_exception(exc: BaseException) -> RunOutcome:
    """Map an exception escaping a run to a terminal outcome."""
    from mlipx.protocols import CancellationRequested

    if isinstance(exc, (CancellationRequested, KeyboardInterrupt)):
        return RunOutcome.CANCELLED
    return RunOutcome.FAILED


def exit_code_for(outcome: RunOutcome) -> int:
    """Process exit code for an outcome (0/1/2/130)."""
    return _OUTCOME_TO_EXIT[RunOutcome(outcome)]


def exit_code_for_exception(exc: BaseException) -> int:
    return exit_code_for(classify_exception(exc))


def job_status_for(outcome: RunOutcome) -> str:
    """Queue job status (``JobStatus`` value) for an outcome."""
    return _OUTCOME_TO_JOB_STATUS[RunOutcome(outcome)]


def outcome_from_exit_code(code: int) -> RunOutcome:
    """Map a child-process exit code back to an outcome (queue worker)."""
    if code == EXIT_SUCCESS:
        return RunOutcome.COMPLETED
    if code == EXIT_NOT_CONVERGED:
        return RunOutcome.NOT_CONVERGED
    if code in (EXIT_CANCELLED, -signal.SIGINT, -signal.SIGTERM):
        return RunOutcome.CANCELLED
    return RunOutcome.FAILED


def describe(outcome: RunOutcome) -> str:
    """One-line user-facing description of an outcome."""
    return {
        RunOutcome.COMPLETED: "completed",
        RunOutcome.NOT_CONVERGED: (
            "not converged (partial coordinates and the reason were kept)"
        ),
        RunOutcome.FAILED: "failed",
        RunOutcome.CANCELLED: "cancelled by user",
    }[RunOutcome(outcome)]


__all__ = [
    "CONVERGENCE_CALC_TYPES",
    "EXIT_CANCELLED",
    "EXIT_ERROR",
    "EXIT_NOT_CONVERGED",
    "EXIT_SUCCESS",
    "RunOutcome",
    "classify_exception",
    "classify_result",
    "describe",
    "exit_code_for",
    "exit_code_for_exception",
    "job_status_for",
    "outcome_from_exit_code",
]
