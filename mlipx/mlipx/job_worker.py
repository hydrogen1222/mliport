"""Internal entry point for persistent background calculations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from mlipx.jobs import JobManager, JobStatus


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

    # The parent records our PID immediately after Popen returns.  Wait for
    # that atomic state update so a very short command cannot finish first and
    # have its terminal status overwritten with "running".
    data = None
    for _ in range(100):
        data = manager.get_job(args.job_id)
        if (
            data is not None
            and data.get("status") == JobStatus.RUNNING.value
            and data.get("pid") == os.getpid()
            and data.get("claim_token") == args.claim_token
        ):
            break
        time.sleep(0.01)
    else:
        return 2

    log_path = manager._log_file(args.job_id)
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            result = subprocess.run(
                args.command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        status = JobStatus.DONE if result.returncode == 0 else JobStatus.FAILED
        manager.update_status(
            args.job_id,
            status,
            expected_statuses={JobStatus.RUNNING},
            expected_pid=os.getpid(),
            claim_token=args.claim_token,
            error=None if result.returncode == 0 else f"Exit code {result.returncode}",
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
