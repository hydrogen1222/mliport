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


def test_a_bad_command_line_is_failed_not_not_converged(tmp_path: Path) -> None:
    """argparse's historical exit code 2 collided with "not converged".

    A queued command that never starts must be reported as failed, not as a
    calculation that ran and missed its criterion.
    """
    manager = JobManager(jobs_dir=tmp_path / "jobs")
    job_id = _submit(
        manager,
        [sys.executable, "-m", "mliport.cli", "run", "--bogus-flag"],
    )

    job = _wait_for_status(manager, job_id, {"failed", "not_converged"})
    assert job["status"] == "failed"
    assert "unrecognized arguments" in manager.tail_log(job_id, lines=50)


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


def test_scheduler_respects_paused_task_folder_queue(tmp_path: Path) -> None:
    """A wizard-style pending job cannot bypass the paused scheduler."""
    from mliport.queue import QueueScheduler, pause_scheduler, resume_scheduler

    manager = JobManager(jobs_dir=tmp_path / "jobs")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    job_id = manager.new_job_id()
    manager.enqueue(
        job_id,
        "sp",
        str(task_dir / "POSCAR"),
        "He",
        1,
        "cpu",
        [sys.executable, "-c", "print('queued worker started')"],
        submit_cwd=task_dir,
    )
    scheduler = QueueScheduler(jobs_dir=manager.jobs_dir, max_concurrent=1)
    assert pause_scheduler(manager.jobs_dir) is True
    assert scheduler.run_once() == 0
    assert manager.get_job(job_id)["status"] == "pending"
    assert resume_scheduler(manager.jobs_dir) is True
    assert scheduler.run_once() == 1
    assert _wait_for_status(manager, job_id, {"done"})["status"] == "done"


def test_queue_worker_uses_task_folder_as_working_directory(tmp_path: Path) -> None:
    """The worker must read task-local relative inputs and settings."""
    from mliport.queue import QueueScheduler

    manager = JobManager(jobs_dir=tmp_path / "jobs")
    folder = tmp_path / "workdir"
    folder.mkdir()
    job_id = manager.new_job_id()
    manager.enqueue(
        job_id,
        "sp",
        str(folder / "POSCAR"),
        "He",
        1,
        "cpu",
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "Path('relative-marker.txt').write_text('folder cwd')",
        ],
        submit_cwd=folder,
    )
    assert manager.get_job(job_id)["submit_cwd"] == str(folder)
    scheduler = QueueScheduler(jobs_dir=manager.jobs_dir)
    assert scheduler.run_once() == 1
    _wait_for_status(manager, job_id, {"done"})
    assert (folder / "relative-marker.txt").read_text() == "folder cwd"
    assert not (tmp_path / "relative-marker.txt").exists()


def test_busy_gpu_lease_leaves_later_job_pending(tmp_path: Path) -> None:
    """A busy physical GPU is not bypassed by wizard-style enqueue."""
    import os

    from mliport.queue import QueueScheduler

    manager = JobManager(jobs_dir=tmp_path / "jobs")
    command = [sys.executable, "-c", "print('not spawned')"]
    occupied_id = manager.new_job_id()
    manager.enqueue(occupied_id, "sp", "/tmp/POSCAR", "He", 1, "cuda:0", command)
    token = manager.new_job_id()
    claimed = manager.claim_job(
        occupied_id,
        claim_token=token,
        owner_pid=os.getpid(),
        device_uuid="GPU-test-lease",
    )
    assert claimed is not None

    queued_id = manager.new_job_id()
    manager.enqueue(queued_id, "sp", "/tmp/POSCAR", "He", 1, "cuda:1", command)
    scheduler = QueueScheduler(
        jobs_dir=manager.jobs_dir,
        max_concurrent=2,
        device_uuid_resolver=lambda _device: "GPU-test-lease",
    )
    assert scheduler.run_once() == 0
    assert manager.get_job(queued_id)["status"] == "pending"


def test_leased_worker_rewrites_nonzero_gpu_ordinal_in_run_argv() -> None:
    """A physical UUID lease maps to logical cuda:0 without a real GPU."""
    from mliport.job_worker import leased_process_environment

    command = [
        sys.executable,
        "-m",
        "mliport.cli",
        "run",
        "--incar",
        "/tmp/INCAR.mliport",
        "--device",
        "cuda:1",
    ]
    rewritten, env = leased_process_environment(
        {"device": "cuda:1", "device_uuid": "GPU-test-lease"},
        command,
        {},
    )
    assert command[-1] == "cuda:1"
    assert rewritten[-1] == "cuda:0"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-test-lease"
    assert env["DEVICE"] == "cuda"
