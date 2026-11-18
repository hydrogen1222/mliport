"""Real (CPU-only) background job lifecycle through the shared JobManager.

These tests start actual detached workers with the current Python interpreter;
they do not mock the state machine or the log protocol.  No MLIP backend or GPU
is involved, so they run in ordinary CI.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from mliport.jobs import JobManager


def _wait_for_status(
    manager: JobManager, job_id: str, statuses: set[str], timeout: float = 20.0
) -> dict:
    deadline = time.monotonic() + timeout
    last: dict | None = None
    while time.monotonic() < deadline:
        last = manager.get_job(job_id)
        if last is not None and last.get("status") in statuses:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach {statuses}: {last}")


def _submit(manager: JobManager, command: list[str]) -> str:
    job_id = manager.new_job_id()
    manager.submit(
        job_id,
        "sp",
        "/tmp/POSCAR",
        "H2",
        2,
        "cpu",
        command,
        display_name="lifecycle-test",
    )
    return job_id


def test_real_worker_runs_to_done_and_writes_its_log(tmp_path: Path) -> None:
    manager = JobManager(jobs_dir=tmp_path / "jobs")
    marker = tmp_path / "finished.txt"
    job_id = _submit(
        manager,
        [
            sys.executable,
            "-c",
            "import pathlib,sys; print('worker output'); "
            "pathlib.Path(sys.argv[1]).write_text('done')",
            str(marker),
        ],
    )

    job = _wait_for_status(manager, job_id, {"done"})
    assert marker.read_text(encoding="utf-8") == "done"
    assert job["finished_at"] is not None
    assert "worker output" in manager.tail_log(job_id, lines=50)


def test_real_worker_failure_is_not_reported_as_done(tmp_path: Path) -> None:
    manager = JobManager(jobs_dir=tmp_path / "jobs")
    job_id = _submit(
        manager,
        [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
    )

    job = _wait_for_status(manager, job_id, {"failed"})
    assert "Exit code 3" in str(job.get("error"))
    assert "boom" in manager.tail_log(job_id, lines=50)


def test_pending_job_can_be_cancelled_without_spawning(tmp_path: Path) -> None:
    manager = JobManager(jobs_dir=tmp_path / "jobs")
    job_id = manager.new_job_id()
    manager.enqueue(
        job_id,
        "sp",
        "/tmp/POSCAR",
        "H2",
        2,
        "cpu",
        [sys.executable, "-c", "import time; time.sleep(10)"],
        display_name="queued",
    )
    assert manager.get_job(job_id)["status"] == "pending"

    assert manager.kill_job(job_id) is True
    assert manager.get_job(job_id)["status"] == "cancelled"


def test_pause_and_resume_only_affect_a_pending_job(tmp_path: Path) -> None:
    manager = JobManager(jobs_dir=tmp_path / "jobs")
    job_id = manager.new_job_id()
    manager.enqueue(
        job_id,
        "sp",
        "/tmp/POSCAR",
        "H2",
        2,
        "cpu",
        [sys.executable, "-c", "import time; time.sleep(10)"],
        display_name="queued",
    )

    assert manager.pause_pending(job_id) is True
    assert manager.get_job(job_id)["status"] == "paused"
    assert manager.resume_paused(job_id) is True
    assert manager.get_job(job_id)["status"] == "pending"
    assert manager.kill_job(job_id) is True
