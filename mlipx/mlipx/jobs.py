"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Disk-persisted background jobs with transactional state transitions.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if os.name == "nt":  # pragma: no cover - CI and supported HPC hosts are POSIX
    import msvcrt
else:
    import fcntl

if TYPE_CHECKING:
    from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    PAUSED = "paused"
    CLAIMED = "claimed"
    RUNNING = "running"
    DONE = "done"
    NOT_CONVERGED = "not_converged"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Statuses that mean the scheduler will not touch the job again.  A
#: ``not_converged`` run is finished (its exit code is 2, not 0) but it is not
#: a success: queue callers must inspect the status, never the exit code alone.
_TERMINAL_STATUSES = frozenset(
    {
        JobStatus.DONE.value,
        JobStatus.NOT_CONVERGED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
    }
)
_ACTIVE_LEASE_STATUSES = frozenset({JobStatus.CLAIMED.value, JobStatus.RUNNING.value})
#: Sentinel: capture the worker's /proc identity automatically.  Tests that
#: fabricate PIDs pass an explicit identity instead.
_AUTO_PID_IDENTITY = object()
_ALLOWED_TRANSITIONS = {
    JobStatus.PENDING.value: frozenset(
        {
            JobStatus.PAUSED.value,
            JobStatus.CLAIMED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }
    ),
    JobStatus.PAUSED.value: frozenset(
        {JobStatus.PENDING.value, JobStatus.CANCELLED.value}
    ),
    JobStatus.CLAIMED.value: frozenset(
        {
            JobStatus.PENDING.value,
            JobStatus.RUNNING.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }
    ),
    JobStatus.RUNNING.value: frozenset(
        {
            JobStatus.DONE.value,
            JobStatus.NOT_CONVERGED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }
    ),
}


def _default_jobs_dir() -> Path:
    """Get default jobs directory: ~/.mlipx/jobs/ (override via MLIPX_JOBS_DIR)."""
    override = os.environ.get("MLIPX_JOBS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".mlipx" / "jobs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_identity(pid: int) -> str | None:
    """Return the Linux process start tick used to detect PID reuse."""
    if os.name == "nt" or not _pid_alive(pid):  # pragma: no cover - Windows fallback
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = stat.rsplit(")", maxsplit=1)[1].split()
        return tail[19]
    except (OSError, IndexError):
        return None


def _pid_matches(pid: int, identity: str | None) -> bool:
    if not _pid_alive(pid):
        return False
    return identity is None or _process_identity(pid) == identity


def _lock_file(handle: BinaryIO, *, blocking: bool) -> None:
    """Acquire an OS-backed exclusive lock on an open file."""
    if os.name == "nt":  # pragma: no cover - Windows fallback
        handle.seek(0)
        if handle.read(1) == b"":
            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        try:
            msvcrt.locking(handle.fileno(), mode, 1)
        except OSError as exc:
            raise BlockingIOError("lock is already held") from exc
        return
    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    fcntl.flock(handle.fileno(), flags)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":  # pragma: no cover - Windows fallback
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _exclusive_file_lock(path: Path, *, blocking: bool = True) -> Iterator[BinaryIO]:
    """Hold a process-safe file lock for the duration of the context."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        _lock_file(handle, blocking=blocking)
        try:
            yield handle
        finally:
            _unlock_file(handle)


class JobManager:
    """Manage background jobs using locked compare-and-swap transitions."""

    def __init__(self, jobs_dir: Path | None = None):
        self.jobs_dir = jobs_dir or _default_jobs_dir()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir = self.jobs_dir / "logs"
        self._logs_dir.mkdir(exist_ok=True)
        self._state_lock_path = self.jobs_dir / ".state.lock"

    @staticmethod
    def new_job_id() -> str:
        """Return an immutable internal UUID suitable for a new job."""
        return str(uuid.uuid4())

    @staticmethod
    def _validate_new_job_id(job_id: str) -> str:
        try:
            parsed = uuid.UUID(str(job_id))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("Internal job_id must be a UUID") from exc
        canonical = str(parsed)
        if str(job_id) != canonical:
            raise ValueError(
                f"Internal job_id must use canonical UUID form: {canonical}"
            )
        return canonical

    @staticmethod
    def _safe_existing_id(job_id: str) -> str:
        value = str(job_id)
        if (
            not value
            or value in {".", ".."}
            or Path(value).name != value
            or "/" in value
            or "\\" in value
        ):
            raise ValueError("Invalid job ID")
        return value

    def _job_file(self, job_id: str) -> Path:
        return self.jobs_dir / f"{self._safe_existing_id(job_id)}.json"

    def _log_file(self, job_id: str) -> Path:
        return self._logs_dir / f"{self._safe_existing_id(job_id)}.log"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with _exclusive_file_lock(self._state_lock_path):
            yield

    def _read_job_state_unlocked(self, job_id: str) -> dict[str, Any] | None:
        try:
            path = self._job_file(job_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or data.get("job_id") != job_id:
            return None
        # Read-only compatibility for records written before internal UUIDs.
        data.setdefault("display_name", job_id)
        data.setdefault("run_id", job_id)
        data.setdefault("created_at", data.get("started_at"))
        data.setdefault("state_revision", 0)
        data.setdefault("claim_token", None)
        data.setdefault("claim_owner_pid", None)
        data.setdefault("claim_owner_identity", None)
        data.setdefault("claimed_at", None)
        data.setdefault("device_uuid", None)
        data.setdefault("pid_identity", None)
        return data

    def _write_data_unlocked(self, data: dict[str, Any]) -> None:
        """Atomically replace one complete state record while holding the lock."""
        job_id = self._safe_existing_id(str(data["job_id"]))
        job_path = self._job_file(job_id)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.jobs_dir,
                prefix=f".{job_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(data, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            temp_path.replace(job_path)
            if os.name != "nt":
                directory_fd = os.open(self.jobs_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _write_job_state(
        self,
        job_id: str,
        status: JobStatus,
        calc_type: str,
        structure: str,
        formula: str,
        natoms: int,
        pid: int,
        device: str,
        progress: dict | None = None,
        results: dict | None = None,
        error: str | None = None,
        finished_at: str | None = None,
        cmd: list[str] | None = None,
        python: str | None = None,
        display_name: str | None = None,
    ) -> None:
        """Private test/migration helper for writing a complete state record."""
        self._validate_new_job_id(job_id)
        now = _utc_now()
        data = {
            "job_id": job_id,
            "run_id": job_id,
            "display_name": display_name or job_id,
            "status": status.value,
            "calc_type": calc_type,
            "structure": structure,
            "formula": formula,
            "natoms": natoms,
            "pid": pid,
            "pid_identity": _process_identity(pid),
            "device": device,
            "device_uuid": None,
            "created_at": now,
            "started_at": now if status == JobStatus.RUNNING else None,
            "updated_at": now,
            "finished_at": finished_at,
            "log_file": str(self._log_file(job_id)),
            "progress": progress or {},
            "results": results,
            "error": error,
            "cmd": cmd,
            "python": python,
            "claim_token": None,
            "claim_owner_pid": None,
            "claim_owner_identity": None,
            "claimed_at": None,
            "state_revision": 0,
        }
        with self._locked():
            self._write_data_unlocked(data)

    def _read_job_state(self, job_id: str) -> dict[str, Any] | None:
        return self._read_job_state_unlocked(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            data = self._read_job_state_unlocked(path.stem)
            if data is not None:
                jobs.append(data)
        return jobs

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._read_job_state(job_id)

    def enqueue(
        self,
        job_id: str,
        calc_type: str,
        structure: str,
        formula: str,
        natoms: int,
        device: str,
        cmd: list[str],
        python: str | None = None,
        *,
        display_name: str | None = None,
    ) -> str:
        """Atomically add a uniquely identified PENDING job."""
        job_id = self._validate_new_job_id(job_id)
        name = str(display_name or job_id).strip()
        if not name:
            raise ValueError("display_name must not be empty")
        if isinstance(natoms, bool) or not isinstance(natoms, int) or natoms < 0:
            raise ValueError("natoms must be a non-negative integer")
        if (
            not isinstance(cmd, list)
            or not cmd
            or not all(isinstance(item, str) and item for item in cmd)
        ):
            raise ValueError("cmd must be a non-empty list of strings")
        if not str(calc_type).strip() or not str(device).strip():
            raise ValueError("calc_type and device must not be empty")

        now = _utc_now()
        data = {
            "job_id": job_id,
            "run_id": job_id,
            "display_name": name,
            "status": JobStatus.PENDING.value,
            "calc_type": str(calc_type),
            "structure": str(structure),
            "formula": str(formula),
            "natoms": natoms,
            "pid": 0,
            "pid_identity": None,
            "device": str(device),
            "device_uuid": None,
            "created_at": now,
            "started_at": None,
            "updated_at": now,
            "finished_at": None,
            "log_file": str(self._log_file(job_id)),
            "progress": {},
            "results": None,
            "error": None,
            "cmd": list(cmd),
            "python": python,
            "claim_token": None,
            "claim_owner_pid": None,
            "claim_owner_identity": None,
            "claimed_at": None,
            "state_revision": 0,
        }
        with self._locked():
            job_path = self._job_file(job_id)
            log_path = self._log_file(job_id)
            if job_path.exists() or log_path.exists():
                raise FileExistsError(f"Internal job ID already exists: {job_id}")
            log_path.parent.mkdir(exist_ok=True)
            log_path.open("x", encoding="utf-8").close()
            try:
                self._write_data_unlocked(data)
            except Exception:
                log_path.unlink(missing_ok=True)
                raise
        return job_id

    def pending_jobs(self) -> list[dict[str, Any]]:
        pending = [
            job
            for job in self.list_jobs()
            if job.get("status") == JobStatus.PENDING.value
        ]
        return sorted(
            pending,
            key=lambda job: (str(job.get("created_at") or ""), job["job_id"]),
        )

    def next_pending(self) -> dict[str, Any] | None:
        """Read-only FIFO preview; schedulers must use :meth:`claim_job`."""
        pending = self.pending_jobs()
        return pending[0] if pending else None

    def claim_job(
        self,
        job_id: str,
        *,
        claim_token: str,
        owner_pid: int,
        device_uuid: str | None,
    ) -> dict[str, Any] | None:
        """CAS PENDING -> CLAIMED and acquire the GPU UUID lease atomically."""
        self._validate_new_job_id(job_id)
        self._validate_new_job_id(claim_token)
        if not _pid_alive(owner_pid):
            raise ValueError("claim owner PID must identify a live process")
        with self._locked():
            data = self._read_job_state_unlocked(job_id)
            if data is None or data.get("status") != JobStatus.PENDING.value:
                return None
            if device_uuid is not None:
                for path in self.jobs_dir.glob("*.json"):
                    other = self._read_job_state_unlocked(path.stem)
                    if (
                        other is not None
                        and other.get("job_id") != job_id
                        and other.get("status") in _ACTIVE_LEASE_STATUSES
                        and (
                            other.get("device_uuid") == device_uuid
                            or (
                                other.get("device_uuid") is None
                                and str(other.get("device", "cpu")).lower() != "cpu"
                            )
                        )
                    ):
                        return None
            now = _utc_now()
            data.update(
                {
                    "status": JobStatus.CLAIMED.value,
                    "claim_token": claim_token,
                    "claim_owner_pid": owner_pid,
                    "claim_owner_identity": _process_identity(owner_pid),
                    "claimed_at": now,
                    "device_uuid": device_uuid,
                    "updated_at": now,
                    "error": None,
                    "state_revision": int(data.get("state_revision", 0)) + 1,
                }
            )
            self._write_data_unlocked(data)
            return data.copy()

    @staticmethod
    def _wait_for_process_identity(pid: int, *, timeout_seconds: float = 2.0):
        """Bounded retry for the /proc start tick of a just-spawned worker.

        The parent races the child's own startup; a single read can fail
        transiently (review queue-identity timing).  Retrying here does not
        weaken PID-reuse protection -- the identity is still required.
        """
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            identity = _process_identity(pid)
            if identity is not None:
                return identity
            if not _pid_alive(pid):
                return None
            time.sleep(0.01)
        return None

    def mark_running(
        self,
        job_id: str,
        pid: int,
        *,
        claim_token: str,
        pid_identity: object = _AUTO_PID_IDENTITY,
    ) -> bool:
        """CAS CLAIMED -> RUNNING for the process created by that claim.

        The worker's immutable /proc start tick must be captured before the
        job becomes RUNNING; if it cannot be (while the platform supports
        identity, i.e. POSIX /proc), promotion fails closed rather than run
        without PID-reuse protection.  Callers that spawned the process are
        responsible for killing it on ``False``.

        ``pid_identity`` is an explicit test seam for synthetic records only;
        production callers must leave it at the default sentinel.
        """
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("worker PID must be a positive integer")
        # Never probe or signal a PID unless this record really is the CLAIMED
        # job with this token: a missing/stale record must not touch anything.
        with self._locked():
            data = self._read_job_state_unlocked(job_id)
            if (
                data is None
                or data.get("status") != JobStatus.CLAIMED.value
                or data.get("claim_token") != claim_token
            ):
                return False
        if pid_identity is _AUTO_PID_IDENTITY:
            identity = self._wait_for_process_identity(pid)
            if identity is None and os.name != "nt":
                # The worker is alive but its identity could not be pinned.
                return False
        else:
            identity = pid_identity
        return self.update_status(
            job_id,
            JobStatus.RUNNING,
            expected_statuses={JobStatus.CLAIMED},
            claim_token=claim_token,
            pid=pid,
            pid_identity=identity,
        )

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
        expected_statuses: Collection[JobStatus | str] | None = None,
        expected_pid: int | None = None,
        claim_token: str | None = None,
        pid: int | None = None,
        pid_identity: str | None = None,
    ) -> bool:
        """Compare-and-swap a state transition; terminal states are immutable."""
        desired = JobStatus(status).value
        expected = (
            {JobStatus(item).value for item in expected_statuses}
            if expected_statuses is not None
            else None
        )
        with self._locked():
            data = self._read_job_state_unlocked(job_id)
            if data is None:
                return False
            current = str(data.get("status"))
            if current in _TERMINAL_STATUSES or current == desired:
                return False
            if expected is not None and current not in expected:
                return False
            if desired not in _ALLOWED_TRANSITIONS.get(current, frozenset()):
                return False
            if expected_pid is not None and data.get("pid") != expected_pid:
                return False
            if claim_token is not None and data.get("claim_token") != claim_token:
                return False

            now = _utc_now()
            data["status"] = desired
            data["updated_at"] = now
            data["error"] = error
            data["state_revision"] = int(data.get("state_revision", 0)) + 1
            if pid is not None:
                data["pid"] = pid
                data["pid_identity"] = pid_identity
            if desired == JobStatus.RUNNING.value:
                data["started_at"] = now
            if desired in _TERMINAL_STATUSES:
                data["finished_at"] = now
            if desired in {
                JobStatus.PENDING.value,
                JobStatus.PAUSED.value,
            }:
                data.update(
                    {
                        "pid": 0,
                        "pid_identity": None,
                        "claim_token": None,
                        "claim_owner_pid": None,
                        "claim_owner_identity": None,
                        "claimed_at": None,
                        "device_uuid": None,
                    }
                )
            self._write_data_unlocked(data)
            return True

    def pause_pending(self, job_id: str) -> bool:
        return self.update_status(
            job_id,
            JobStatus.PAUSED,
            expected_statuses={JobStatus.PENDING},
        )

    def resume_paused(self, job_id: str) -> bool:
        return self.update_status(
            job_id,
            JobStatus.PENDING,
            expected_statuses={JobStatus.PAUSED},
        )

    def recover_stale_claims(self, *, max_age_seconds: float = 60.0) -> list[str]:
        """Return abandoned CLAIMED records to PENDING and release their leases."""
        recovered: list[str] = []
        now = datetime.now(timezone.utc)
        with self._locked():
            for path in self.jobs_dir.glob("*.json"):
                data = self._read_job_state_unlocked(path.stem)
                if data is None or data.get("status") != JobStatus.CLAIMED.value:
                    continue
                owner_pid = data.get("claim_owner_pid")
                try:
                    claimed_at = datetime.fromisoformat(str(data.get("claimed_at")))
                    age = (now - claimed_at).total_seconds()
                except (TypeError, ValueError):
                    age = float("inf")
                if (
                    _pid_matches(owner_pid, data.get("claim_owner_identity"))
                    and age <= max_age_seconds
                ):
                    continue
                data.update(
                    {
                        "status": JobStatus.PENDING.value,
                        "pid": 0,
                        "pid_identity": None,
                        "claim_token": None,
                        "claim_owner_pid": None,
                        "claim_owner_identity": None,
                        "claimed_at": None,
                        "device_uuid": None,
                        "updated_at": _utc_now(),
                        "error": "Recovered abandoned scheduler claim",
                        "state_revision": int(data.get("state_revision", 0)) + 1,
                    }
                )
                self._write_data_unlocked(data)
                recovered.append(data["job_id"])
        return recovered

    def count_by_status(self, status: str) -> int:
        return sum(1 for job in self.list_jobs() if job.get("status") == status)

    def queue_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {status.value: 0 for status in JobStatus}
        for job in self.list_jobs():
            status = job.get("status")
            if status in summary:
                summary[status] += 1
        return summary

    def kill_job(self, job_id: str) -> bool:
        """Atomically cancel queued work or cancel-and-signal a running worker."""
        pid_to_kill: int | None = None
        with self._locked():
            data = self._read_job_state_unlocked(job_id)
            if data is None:
                return False
            current = str(data.get("status"))
            if current not in {
                JobStatus.PENDING.value,
                JobStatus.PAUSED.value,
                JobStatus.CLAIMED.value,
                JobStatus.RUNNING.value,
            }:
                return False
            if current == JobStatus.RUNNING.value:
                pid = data.get("pid")
                identity = data.get("pid_identity")
                if (
                    not isinstance(pid, int)
                    or isinstance(pid, bool)
                    or pid <= 0
                    or identity is None
                    or not _pid_matches(pid, identity)
                ):
                    # Never signal a PID that cannot be tied to the worker
                    # originally launched for this job.
                    return False
                pid_to_kill = pid
            now = _utc_now()
            data.update(
                {
                    "status": JobStatus.CANCELLED.value,
                    "updated_at": now,
                    "finished_at": now,
                    "error": "Cancelled by user",
                    "state_revision": int(data.get("state_revision", 0)) + 1,
                }
            )
            self._write_data_unlocked(data)
        if pid_to_kill is not None:
            self._kill_process(pid_to_kill)
        return True

    def _kill_process(self, pid: int) -> None:
        if sys.platform == "win32":  # pragma: no cover - Windows fallback
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)

    @staticmethod
    def running_process_matches(job: dict[str, Any]) -> bool:
        """Return whether a RUNNING record still identifies the same process."""
        return _pid_matches(job.get("pid"), job.get("pid_identity"))

    def clean(self) -> list[str]:
        """Remove terminal state and log files under the state lock."""
        removed: list[str] = []
        with self._locked():
            for path in self.jobs_dir.glob("*.json"):
                data = self._read_job_state_unlocked(path.stem)
                if data is None or data.get("status") not in _TERMINAL_STATUSES:
                    continue
                path.unlink()
                self._log_file(data["job_id"]).unlink(missing_ok=True)
                removed.append(data["job_id"])
            for log in self._logs_dir.glob("*.log"):
                if not self._job_file(log.stem).exists():
                    log.unlink(missing_ok=True)
        return removed

    def delete_job(self, job_id: str) -> bool:
        """Delete a non-active job and its log atomically."""
        with self._locked():
            data = self._read_job_state_unlocked(job_id)
            if data is None or data.get("status") in _ACTIVE_LEASE_STATUSES:
                return False
            self._job_file(job_id).unlink(missing_ok=True)
            self._log_file(job_id).unlink(missing_ok=True)
            return True

    def _spawn_worker(
        self, job_id: str, cmd: list[str], *, claim_token: str
    ) -> subprocess.Popen:
        worker_cmd = [
            sys.executable,
            "-m",
            "mlipx.job_worker",
            str(self.jobs_dir.resolve()),
            job_id,
            claim_token,
            *cmd,
        ]
        return subprocess.Popen(
            worker_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def submit(
        self,
        job_id: str,
        calc_type: str,
        structure: str,
        formula: str,
        natoms: int,
        device: str,
        cmd: list[str],
        python: str | None = None,
        *,
        display_name: str | None = None,
    ) -> subprocess.Popen:
        """Legacy immediate start using the same claim and lease protocol."""
        from mlipx.queue import resolve_device_uuid

        device_uuid = resolve_device_uuid(device)
        self.enqueue(
            job_id,
            calc_type,
            structure,
            formula,
            natoms,
            device,
            cmd,
            python=python,
            display_name=display_name,
        )
        token = self.new_job_id()
        claimed = self.claim_job(
            job_id,
            claim_token=token,
            owner_pid=os.getpid(),
            device_uuid=device_uuid,
        )
        if claimed is None:
            raise RuntimeError(f"Could not claim newly enqueued job {job_id}")
        try:
            proc = self._spawn_worker(job_id, cmd, claim_token=token)
        except OSError as exc:
            self.update_status(
                job_id,
                JobStatus.FAILED,
                expected_statuses={JobStatus.CLAIMED},
                claim_token=token,
                error=f"Could not spawn worker: {exc}",
            )
            raise
        if not self.mark_running(job_id, proc.pid, claim_token=token):
            self._kill_process(proc.pid)
            self.update_status(
                job_id,
                JobStatus.FAILED,
                expected_statuses={JobStatus.CLAIMED},
                claim_token=token,
                error="Could not promote claimed job to running",
            )
            raise RuntimeError(f"Could not promote claimed job {job_id} to running")
        return proc

    def tail_log(self, job_id: str, lines: int = 50) -> str:
        log_path = self._log_file(job_id)
        if not log_path.exists():
            return ""
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            all_lines = handle.readlines()
            return "".join(all_lines[-lines:])
