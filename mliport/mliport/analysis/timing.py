"""Wall-clock phase timings for analysis runs (informational, not scientific).

A :class:`PhaseTimings` recorder is activated for one analysis run with
:func:`recording`; library code marks phases with :func:`phase`, which is a
no-op when no recorder is active. Phases are measured sequentially and never
overlap except where a phase explicitly lists the phases it *includes*
(``includes=``), so a nested measurement cannot be summed twice by mistake.

Timings are recorded in the result under ``timings`` and are deliberately not
part of the analysis cache identity or of any scientific comparison.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

TIMINGS_SCHEMA = "mliport.phase-timings/1"

_ACTIVE: ContextVar[PhaseTimings | None] = ContextVar(
    "mliport_analysis_timings", default=None
)


def peak_rss_gib() -> float | None:
    """Peak resident set size of this process in GiB (``None`` if unavailable).

    This is a process-lifetime maximum, not a per-phase value.
    """

    try:
        import resource  # noqa: PLC0415
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    scale = 1 if sys.platform == "darwin" else 1024
    return float(peak * scale / 1024**3)


class PhaseTimings:
    """Ordered record of named phases and their wall-clock seconds."""

    def __init__(self) -> None:
        self._phases: list[dict[str, Any]] = []

    def record(
        self, name: str, seconds: float, *, includes: tuple[str, ...] = ()
    ) -> None:
        self._phases.append(
            {"name": name, "seconds": float(seconds), "includes": list(includes)}
        )

    @contextmanager
    def phase(self, name: str, *, includes: tuple[str, ...] = ()) -> Iterator[None]:
        """Time a phase; phases recorded while it is open become its ``includes``.

        Phases are appended when they finish, so an enclosing phase is listed
        after the phases it contains.
        """

        first_inner = len(self._phases)
        start = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - start
            inner = [entry["name"] for entry in self._phases[first_inner:]]
            merged = list(dict.fromkeys([*includes, *inner]))
            self.record(name, seconds, includes=tuple(merged))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TIMINGS_SCHEMA,
            "note": (
                "Wall-clock seconds per phase. A phase with a non-empty "
                "'includes' list contains those phases; do not add them. "
                "Informational only: not part of the cache identity."
            ),
            "phases": [dict(entry) for entry in self._phases],
            "peak_rss_gib_process_lifetime": peak_rss_gib(),
        }


@contextmanager
def recording(timings: PhaseTimings) -> Iterator[PhaseTimings]:
    token = _ACTIVE.set(timings)
    try:
        yield timings
    finally:
        _ACTIVE.reset(token)


@contextmanager
def phase(name: str, *, includes: tuple[str, ...] = ()) -> Iterator[None]:
    """Time a phase on the active recorder; do nothing when none is active."""

    active = _ACTIVE.get()
    if active is None:
        yield
        return
    with active.phase(name, includes=includes):
        yield


__all__ = [
    "TIMINGS_SCHEMA",
    "PhaseTimings",
    "peak_rss_gib",
    "phase",
    "recording",
]
