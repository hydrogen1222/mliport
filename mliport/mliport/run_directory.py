"""Ownership of a calculation output directory.

A run writes its raw trajectory, VASP-style files and provenance into one
directory. Writing into a directory that already holds the outputs of another
calculation (an earlier mliport run, or a VASP calculation whose OUTCAR and
CONTCAR live next to the POSCAR) silently destroys them. Starting a run
therefore claims the directory: it refuses existing calculation outputs unless
the caller explicitly allows overwriting, and it holds an exclusive process
lock so two runs can never write the same directory concurrently.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Lock file held for the lifetime of a single-structure or batch run.
RUN_LOCK_NAME = ".mliport-run.lock"

#: Files and directories written by sp/opt/md/batch runs. VASP's own OUTCAR,
#: OSZICAR, CONTCAR and XDATCAR share these names and are protected as well.
RUN_ARTIFACT_NAMES: frozenset[str] = frozenset(
    {
        "run.log",
        "resolved_config.json",
        "mliport_results.json",
        "results.json",
        "artifacts.json",
        "batch_summary.json",
        "calculation.log",
        "optimization.log",
        "trajectory.traj",
        "trajectory.xyz",
        "md.csv",
        "OUTCAR",
        "OSZICAR",
        "CONTCAR",
        "XDATCAR",
        "raw",
        "vasp",
    }
)


class OutputDirectoryInUseError(FileExistsError):
    """The output directory holds another calculation or is being written."""


def existing_run_artifacts(path: str | Path) -> list[str]:
    """Return calculation outputs already present in ``path`` (sorted)."""
    directory = Path(path)
    if not directory.is_dir():
        return []
    return sorted(
        child.name for child in directory.iterdir() if child.name in RUN_ARTIFACT_NAMES
    )


def ensure_output_available(path: str | Path, *, overwrite: bool) -> None:
    """Fail closed when ``path`` already holds calculation outputs."""
    directory = Path(path)
    if directory.exists() and not directory.is_dir():
        raise OutputDirectoryInUseError(f"Output path is not a directory: {directory}")
    if overwrite:
        return
    existing = existing_run_artifacts(directory)
    if existing:
        raise OutputDirectoryInUseError(
            f"Output directory {directory} already contains calculation outputs "
            f"({', '.join(existing)}). Choose a new --output directory, or pass "
            "--overwrite (API: overwrite=True) to replace them deliberately."
        )


@contextmanager
def run_directory_lock(path: str | Path) -> Iterator[Path]:
    """Hold a non-blocking exclusive lock on ``path`` for one run."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / RUN_LOCK_NAME).open("a+b")
    acquired = False
    try:
        try:
            if os.name == "nt":  # pragma: no cover - POSIX HPC is authoritative
                import msvcrt  # noqa: PLC0415

                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise OutputDirectoryInUseError(
                f"Another mliport run is already writing {directory}"
            ) from exc
        yield directory
    finally:
        if acquired and os.name == "nt":  # pragma: no cover
            import msvcrt  # noqa: PLC0415

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif acquired:
            import fcntl  # noqa: PLC0415

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


__all__ = [
    "RUN_ARTIFACT_NAMES",
    "RUN_LOCK_NAME",
    "OutputDirectoryInUseError",
    "ensure_output_available",
    "existing_run_artifacts",
    "run_directory_lock",
]
