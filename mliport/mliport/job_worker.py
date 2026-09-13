"""Internal entry point for persistent background calculations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from mliport.jobs import JobManager, JobStatus
from mliport.run_status import RunOutcome, outcome_from_exit_code


def leased_process_environment(
    data: dict, command: list[str], inherited: dict[str, str]
) -> tuple[list[str], dict[str, str]]:
    """Freeze a claim's GPU identity before the calculation imports frameworks.

    A leased physical GPU becomes logical cuda:0 in the isolated child. Never
    reinterpret the original ordinal under the scheduler's current visibility.
    """
    env = dict(inherited)
    cmd = list(command)
    uuid = data.get("device_uuid")
    device = str(data.get("device", "cpu")).lower()
    if device != "cpu" and not uuid:
        raise RuntimeError("GPU calculation has no immutable UUID lease")
    if uuid is not None and (
        not isinstance(uuid, str)
        or not uuid.startswith(("GPU-", "MIG-"))
        or any(c.isspace() or c == "," for c in uuid)
    ):
        raise RuntimeError("Invalid GPU UUID lease")
    env["CUDA_VISIBLE_DEVICES"] = uuid or ""
    env["LOCAL_RANK"] = "0"
    env["DEVICE"] = "cuda" if uuid else "cpu"
    env.pop("MLIPORT_LEASED_GPU_UUID", None)
    if uuid:
        env["MLIPORT_LEASED_GPU_UUID"] = uuid
    target = "cuda:0" if uuid else "cpu"
    for index, arg in enumerate(cmd):
        if arg == "--device":
            if index + 1 >= len(cmd):
                raise ValueError("Missing --device value in claimed command")
            cmd[index + 1] = target
        elif arg.startswith("--device="):
            cmd[index] = f"--device={target}"
    return cmd, env


HANDSHAKE_TIMEOUT_SECONDS = 10.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("jobs_dir")
    parser.add_argument("job_id")
    parser.add_argument("claim_token")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    manager = JobManager(Path(args.jobs_dir))
    if not args.command:
        return 2

    # Startup handshake: the parent records our PID (and its /proc start-tick
    # identity) immediately after Popen returns.  Wait for that atomic update
    # on a wall-clock budget so a very short command cannot finish first and
    # have its terminal status overwritten with "running".  The retry is
    # bounded but generous; PID-reuse protection is never weakened.
    data = None
    deadline = time.monotonic() + HANDSHAKE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        data = manager.get_job(args.job_id)
        if (
            data is not None
            and data.get("status") == JobStatus.RUNNING.value
            and data.get("pid") == os.getpid()
            and data.get("claim_token") == args.claim_token
        ):
            break
        time.sleep(0.02)
    else:
        # Publish a readable reason instead of leaving a stale RUNNING record
        # for the scheduler to reap as "exited unexpectedly".
        manager.update_status(
            args.job_id,
            JobStatus.FAILED,
            expected_statuses={JobStatus.CLAIMED, JobStatus.RUNNING},
            claim_token=args.claim_token,
            error=(
                "worker startup handshake timed out: the scheduler did not "
                "publish this worker's PID/identity"
            ),
        )
        return 2

    log_path = manager._log_file(args.job_id)
    try:
        command, environment = leased_process_environment(
            data, args.command, os.environ
        )
        with log_path.open("a", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        outcome = outcome_from_exit_code(result.returncode)
        status = {
            RunOutcome.COMPLETED: JobStatus.DONE,
            RunOutcome.NOT_CONVERGED: JobStatus.NOT_CONVERGED,
            RunOutcome.CANCELLED: JobStatus.CANCELLED,
            RunOutcome.FAILED: JobStatus.FAILED,
        }[outcome]
        if outcome is RunOutcome.COMPLETED:
            error = None
        elif outcome is RunOutcome.NOT_CONVERGED:
            error = "Exit code 2: calculation finished but did not converge"
        elif outcome is RunOutcome.CANCELLED:
            error = "Cancelled (signal or cooperative cancellation)"
        else:
            error = f"Exit code {result.returncode}"
        manager.update_status(
            args.job_id,
            status,
            expected_statuses={JobStatus.RUNNING},
            expected_pid=os.getpid(),
            claim_token=args.claim_token,
            error=error,
        )
        return result.returncode
    except Exception as exc:
        manager.update_status(
            args.job_id,
            JobStatus.FAILED,
            expected_statuses={JobStatus.RUNNING},
            expected_pid=os.getpid(),
            claim_token=args.claim_token,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
