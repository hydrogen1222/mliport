"""Safe subprocess boundary for the native ``gpumd`` executable.

GPUMD is one process per run, so mliport must never treat a clean exit code as
scientific success.  This module launches the executable as an argv list
(``shell=False``), streams its output to files in the work directory, and
reports the *process* outcome honestly.  Parsing is a separate concern
(:mod:`mliport.gpumd.parse`); a non-zero return code is returned, never raised,
because the caller decides whether a particular exit code is fatal.

Cancellation and timeouts escalate politely (SIGINT, grace period, SIGKILL) so
raw output remains complete and a GPU process cannot survive the run.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "ProcessOutcome",
    "gpumd_identity",
    "resolve_gpumd_command",
    "run_gpumd",
]

#: How often the parent checks for exit, cancellation and timeout.
_POLL_INTERVAL_S = 0.2

#: Default grace period between a polite signal and SIGKILL.
_DEFAULT_GRACE_S = 10.0

_VERSION_PATTERN = re.compile(r"version\s+([0-9]+(?:\.[0-9]+)+)", re.IGNORECASE)


@dataclass
class ProcessOutcome:
    """Observed result of one ``gpumd`` child process.

    Attributes:
        returncode: Child exit code, or ``None`` if it could not be reaped.
        status: ``"exited"`` (including non-zero codes), ``"cancelled"`` or
            ``"timeout"``.
        started_at: UTC ISO-8601 timestamp of ``Popen`` creation.
        wall_time_s: Monotonic wall time spent in :func:`run_gpumd`.
        stdout_path: Absolute path of the captured stdout log.
        stderr_path: Absolute path of the captured stderr log.
        command: Exact argv list that was launched.
        cwd: Work directory used for the child.
        terminated: True when mliport had to signal the child.
    """

    returncode: int | None
    status: str
    started_at: str
    wall_time_s: float
    stdout_path: str
    stderr_path: str
    command: list[str] = field(default_factory=list)
    cwd: str = ""
    terminated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "returncode": self.returncode,
            "status": self.status,
            "started_at": self.started_at,
            "wall_time_s": self.wall_time_s,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "command": list(self.command),
            "cwd": self.cwd,
            "terminated": self.terminated,
        }


def _emit_log(log_fn: Any, message: str, level: str = "info") -> None:
    """Call an optional logging callback without imposing one signature."""
    if log_fn is None:
        return
    try:
        log_fn(message, level)
    except TypeError:
        log_fn(message)


def _validate_command(command: Sequence[str] | str) -> list[str]:
    """Validate an argv list and reject shell strings."""
    if isinstance(command, str):
        raise ValueError(
            "command must be an argv list, not a shell string. Refusing to "
            "interpret shell metacharacters such as ';', '|', '&', '$', '>', "
            "'<' or backticks."
        )
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError("command must be a non-empty argv list")
    argv = [str(part) for part in command]
    if any(not part for part in argv):
        raise ValueError(f"command contains an empty argument: {argv!r}")
    return argv


def _resolve_executable(executable: str) -> Path:
    """Resolve an executable path, raising the launch failure early."""
    candidate = Path(executable).expanduser()
    if candidate.is_absolute() or os.sep in executable:
        resolved = candidate.resolve()
    else:
        located = shutil.which(executable)
        if located is None:
            raise FileNotFoundError(
                f"GPUMD executable not found on PATH: {executable!r}"
            )
        resolved = Path(located).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"GPUMD executable not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"GPUMD executable is not a regular file: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise PermissionError(f"GPUMD executable is not executable: {resolved}")
    return resolved


def resolve_gpumd_command(path: str | Path) -> list[str]:
    """Return the argv list for a GPUMD executable path.

    Args:
        path: Executable path or a bare command name found on ``PATH``.

    Returns:
        A one-element argv list containing the resolved real path.

    Raises:
        FileNotFoundError: If the executable does not exist.
        PermissionError: If it exists but is not executable.
    """
    return [str(_resolve_executable(str(path)))]


def _signal_process(process: subprocess.Popen[str], sig: int) -> None:
    """Send ``sig`` to the child's process group when possible."""
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), sig)
            return
        except (OSError, ProcessLookupError):
            pass
    try:
        process.send_signal(sig)
    except (OSError, ProcessLookupError):
        pass


def _stop_process(
    process: subprocess.Popen[str],
    *,
    grace_s: float,
    log_fn: Any,
) -> None:
    """Escalate from SIGINT to SIGKILL after a grace period."""
    _emit_log(log_fn, "Sending SIGINT to GPUMD and waiting for exit", "warning")
    _signal_process(process, signal.SIGINT)
    try:
        process.wait(timeout=max(0.0, grace_s))
        return
    except subprocess.TimeoutExpired:
        _emit_log(log_fn, "GPUMD ignored SIGINT; sending SIGKILL", "warning")
    _signal_process(process, signal.SIGKILL)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        _emit_log(log_fn, "GPUMD did not exit even after SIGKILL", "error")


def run_gpumd(
    command: Sequence[str] | str,
    workdir: str | Path,
    *,
    timeout_s: float | None = None,
    cancel_event: Any | None = None,
    log_fn: Any | None = None,
    grace_s: float = _DEFAULT_GRACE_S,
    env: Mapping[str, str] | None = None,
) -> ProcessOutcome:
    """Run one GPUMD invocation and return its process outcome.

    Args:
        command: Argv list.  A shell string is rejected outright.
        workdir: Directory in which ``gpumd`` runs; ``stdout.log`` and
            ``stderr.log`` are created there.
        timeout_s: Optional wall-clock limit in seconds.  On expiry the child
            is signalled, then killed after ``grace_s``.
        cancel_event: Optional ``threading.Event`` polled every 0.2 s.  When
            set, the child is signalled and the outcome is ``"cancelled"``.
        log_fn: Optional callback ``(message, level)`` (a one-argument callable
            is also accepted).
        grace_s: Seconds between the polite signal and SIGKILL.
        env: Optional environment overrides merged onto ``os.environ`` for the
            child only.

    Returns:
        A :class:`ProcessOutcome`.  A non-zero exit code is reported, never
        raised.

    Raises:
        ValueError: If ``command`` is not an argv list.
        FileNotFoundError: If the executable or ``workdir`` is missing.
        PermissionError: If the executable is not executable.
        OSError: For other launch failures.
    """
    argv = _validate_command(command)
    directory = Path(workdir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"GPUMD work directory not found: {directory}")

    executable = _resolve_executable(argv[0])
    argv = [str(executable), *argv[1:]]

    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    child_env = os.environ.copy()
    if env:
        child_env.update({str(key): str(value) for key, value in env.items()})

    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    _emit_log(log_fn, f"Launching GPUMD: {' '.join(argv)}")

    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(  # noqa: S603 - argv list, shell=False
            argv,
            cwd=str(directory),
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=child_env,
            start_new_session=os.name == "posix",
        )

        status = "exited"
        terminated = False
        try:
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                if cancel_event is not None and cancel_event.is_set():
                    status = "cancelled"
                    terminated = True
                    _emit_log(log_fn, "GPUMD cancellation requested", "warning")
                    _stop_process(process, grace_s=grace_s, log_fn=log_fn)
                    break
                if timeout_s is not None and time.monotonic() - start >= float(
                    timeout_s
                ):
                    status = "timeout"
                    terminated = True
                    _emit_log(
                        log_fn,
                        f"GPUMD timed out after {float(timeout_s):.1f} s",
                        "error",
                    )
                    _stop_process(process, grace_s=grace_s, log_fn=log_fn)
                    break
                time.sleep(_POLL_INTERVAL_S)
        finally:
            if process.poll() is None:
                _stop_process(process, grace_s=grace_s, log_fn=log_fn)

        returncode = process.wait()
        wall_time_s = time.monotonic() - start
        _emit_log(
            log_fn,
            f"GPUMD process {status} with return code {returncode} "
            f"after {wall_time_s:.2f} s",
        )

        return ProcessOutcome(
            returncode=returncode,
            status=status,
            started_at=started_at,
            wall_time_s=wall_time_s,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            command=argv,
            cwd=str(directory),
            terminated=terminated,
        )


def _sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpumd_identity(
    path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Identify the exact GPUMD binary and its reported version.

    The version is taken from the banner GPUMD prints when started in an empty
    directory with no ``run.in`` (the banner contains ``version X.Y.Z``).  A
    failure to obtain it is recorded as ``version=None`` plus a
    ``version_error``; the function never guesses.

    Args:
        path: GPUMD executable path.
        env: Optional environment overrides for the banner probe (for example
            a CUDA ``LD_LIBRARY_PATH`` that the real run also uses).

    Returns:
        Mapping with ``path``, ``real_path``, ``sha256``, ``version`` and
        ``version_error``.

    Raises:
        FileNotFoundError: If the executable is missing.
        PermissionError: If it is not executable.
    """
    resolved = _resolve_executable(str(path))
    record: dict[str, Any] = {
        "path": str(path),
        "real_path": str(resolved),
        "sha256": _sha256(resolved),
        "version": None,
        "version_error": None,
    }
    child_env = os.environ.copy()
    if env:
        child_env.update({str(key): str(value) for key, value in env.items()})
    try:
        with tempfile.TemporaryDirectory(prefix="mliport-gpumd-identity-") as tmp:
            completed = subprocess.run(  # noqa: S603 - resolved argv, shell=False
                [str(resolved)],
                cwd=tmp,
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=120.0,
                check=False,
                env=child_env,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        record["version_error"] = f"{type(exc).__name__}: {exc}"
        return record

    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    match = _VERSION_PATTERN.search(combined)
    if match is not None:
        record["version"] = match.group(1)
        return record
    stderr_tail = (completed.stderr or "").strip().splitlines()[-3:]
    record["version_error"] = (
        f"GPUMD banner did not contain 'version X.Y.Z' "
        f"(returncode={completed.returncode}); stderr tail: {stderr_tail}"
    )
    return record
