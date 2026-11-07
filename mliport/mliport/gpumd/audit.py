"""Single-pass completeness audit for a native GPUMD MD dump.

An exit status of 0 only says that the ``gpumd`` process ended.  It does not say
that the requested number of steps was integrated or that every ``dump_xyz``
frame reached the disk: a killed job manager, a full disk, or a stub or
malformed external program can leave a short but perfectly well-formed dump
behind.  :class:`MDDumpAudit` decides whether a dump is the *complete* product
of one ``run.in``.  It needs no ASE: it is fed plain values frame by frame, so
it works in the same single streaming pass that writes the normalised
trajectory and holds one float per frame, never the frames themselves.

What is checked (every failure raises :class:`GPUMDIncompleteRunError`)
-----------------------------------------------------------------------
* the number of frames equals ``steps // dump_interval`` -- GPUMD dumps once per
  ``dump_interval`` completed steps, so a request that is not a multiple of the
  interval ends with a partial interval that writes no frame;
* frame times are recorded (never inferred), strictly increasing and uniformly
  spaced by ``dump_interval * dt``, and sit on the integration-step grid;
* the last frame lies within the final dump interval of the requested run, so a
  time axis that disagrees with ``run.in`` is refused;
* every frame has the input species in the input order, full 3D PBC, a finite
  non-singular cell equal to the input cell (the native ensembles are
  fixed-cell), and the same ``Properties`` columns as the first frame;
* ``thermo.out`` has exactly one row per frame, and its own header (when it
  states one) agrees with ``run.in`` on the output interval and period.

What is deliberately *not* asserted
-----------------------------------
The offset of the first frame inside its dump interval (whether the first
``Time=`` is ``interval * dt`` or one step earlier) is a GPUMD implementation
detail that the repository holds no real dump to pin.  The audit checks the
count, the spacing and the coverage of the requested window, which together
fix the run length without depending on that offset, and it records the first
and last step it actually observed.  The steps that follow the last dump of a
request that is not a multiple of the interval leave no output at all; for them
the exit status is the only evidence there is.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mliport.gpumd.parse import GPUMDIncompleteRunError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mliport.gpumd.parse import Frame

__all__ = ["MDDumpAudit", "MDScheduleReport", "expected_dump_frames"]

#: A frame ``Time=`` must sit this close (in integration steps) to the step grid.
GRID_TOLERANCE_STEPS = 0.05
#: Largest allowed change of any cell component against the input cell (Angstrom).
CELL_TOLERANCE_A = 1.0e-6
#: Tolerances for comparing recorded frame times with ``dump_interval * dt``.
TIME_RTOL = 1.0e-6
TIME_ATOL_FS = 1.0e-5

_UNWRAPPED_NAMES = ("unwrapped_position", "unwrapped_positions")


def expected_dump_frames(steps: int, dump_interval: int) -> int:
    """Return the frame count ``dump_xyz <interval>`` writes during ``run <steps>``.

    GPUMD writes one frame per ``dump_interval`` completed steps, so the count
    is ``floor(steps / dump_interval)``.  Two real V100 runs agree with this:
    20 steps at interval 10 wrote 2 frames, and 200 steps at interval 20 wrote
    10.

    Args:
        steps: Number of steps requested in ``run <steps>``.
        dump_interval: ``dump_xyz`` / ``dump_thermo`` interval in steps.

    Returns:
        The number of frames a complete run produces (0 when
        ``steps < dump_interval``).

    Raises:
        ValueError: If either argument is not a positive integer.
    """
    for name, value in (("steps", steps), ("dump_interval", dump_interval)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    return int(steps) // int(dump_interval)


@dataclass(frozen=True)
class MDScheduleReport:
    """What a verified MD dump actually contains.

    ``steps`` and ``times_fs`` are per-frame arrays (one value per frame);
    :meth:`to_dict` keeps only the scalars, which is what the manifest stores.
    """

    nframes: int
    expected_nframes: int
    steps_requested: int
    dump_interval: int
    dt_fs: float
    frame_interval_fs: float
    first_step: int
    last_step: int
    first_time_fs: float
    last_time_fs: float
    natoms: int
    positions_convention: str
    thermo_rows: int | None
    steps: np.ndarray = field(repr=False, compare=False)
    times_fs: np.ndarray = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe scalar summary (no per-frame arrays)."""
        return {
            "nframes": self.nframes,
            "expected_nframes": self.expected_nframes,
            "steps_requested": self.steps_requested,
            "dump_interval": self.dump_interval,
            "dt_fs": self.dt_fs,
            "frame_interval_fs": self.frame_interval_fs,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "first_time_fs": self.first_time_fs,
            "last_time_fs": self.last_time_fs,
            "natoms": self.natoms,
            "positions_convention": self.positions_convention,
            "thermo_rows": self.thermo_rows,
            "schedule_verified": True,
        }


class MDDumpAudit:
    """Verify, frame by frame, that an MD dump is the complete result of a run.

    Call :meth:`observe` (or :meth:`observe_frame`) once per frame in file
    order, then :meth:`verify` once.  ``observe`` fails on the first frame that
    breaks an identity invariant; ``verify`` checks the properties of the whole
    dump (count, time axis, coverage, thermo alignment) and returns the
    :class:`MDScheduleReport` that the manifest records.

    Args:
        steps: Steps requested in ``run.in``.
        dump_interval: ``dump_xyz`` / ``dump_thermo`` interval in steps.
        dt_fs: Integration time step in fs.
        symbols: Input species, in input order.
        cell: Input cell (3x3, Angstrom).  When given, every frame's cell must
            equal it; the native NVE/NVT ensembles never change the box.
        thermo_rows: Number of numeric rows in ``thermo.out``, if known.
        thermo_dump_interval: ``# dump_thermo`` value from ``thermo.out``, or
            ``None`` when the header does not state one.
        thermo_dt_output_fs: ``# dt_output`` (fs) from ``thermo.out``, or
            ``None``.
        source: Label used in error messages (normally the dump path).
    """

    def __init__(
        self,
        *,
        steps: int,
        dump_interval: int,
        dt_fs: float,
        symbols: Sequence[str],
        cell: Any = None,
        thermo_rows: int | None = None,
        thermo_dump_interval: int | None = None,
        thermo_dt_output_fs: float | None = None,
        source: str = "traj.xyz",
    ) -> None:
        self.expected_nframes = expected_dump_frames(steps, dump_interval)
        if not math.isfinite(dt_fs) or dt_fs <= 0.0:
            raise ValueError(f"dt_fs must be a positive finite number, got {dt_fs!r}")
        self.steps = int(steps)
        self.dump_interval = int(dump_interval)
        self.dt_fs = float(dt_fs)
        self.symbols = tuple(symbols)
        self.cell = None if cell is None else np.asarray(cell, dtype=float)
        self.thermo_rows = thermo_rows
        self.thermo_dump_interval = thermo_dump_interval
        self.thermo_dt_output_fs = thermo_dt_output_fs
        self.source = str(source)
        self._times = array("d")
        self._property_names: tuple[str, ...] | None = None
        self._count = 0

    @property
    def nframes(self) -> int:
        """Number of frames observed so far."""
        return self._count

    def _fail(self, message: str) -> GPUMDIncompleteRunError:
        return GPUMDIncompleteRunError(f"{self.source}: {message}")

    def observe(
        self,
        *,
        symbols: Sequence[str],
        pbc: Sequence[bool],
        cell: Any,
        time_fs: float | None,
        property_names: Sequence[str],
        step: int | None = None,
    ) -> None:
        """Check one frame's identity and record its time.

        Raises:
            GPUMDIncompleteRunError: If the frame changes the atoms, the PBC,
                the cell or the dumped columns, or has no usable ``Time=``.
        """
        index = self._count
        frame_symbols = tuple(symbols)
        if frame_symbols != self.symbols:
            detail = f"{len(frame_symbols)} atoms, expected {len(self.symbols)}"
            if len(frame_symbols) == len(self.symbols):
                for position, (got, want) in enumerate(
                    zip(frame_symbols, self.symbols, strict=True)
                ):
                    if got != want:
                        detail = f"atom {position} is {got!r}, expected {want!r}"
                        break
            raise self._fail(
                f"frame {index} does not hold the input structure's species in "
                f"the input order ({detail}); GPUMD must not reorder atoms"
            )
        if tuple(bool(flag) for flag in pbc) != (True, True, True):
            raise self._fail(
                f"frame {index} has pbc={tuple(bool(flag) for flag in pbc)}; the "
                "native runtime requires full 3D periodicity and never promotes "
                "or demotes a PBC flag"
            )
        frame_cell = np.asarray(cell, dtype=float)
        if frame_cell.shape != (3, 3) or not np.all(np.isfinite(frame_cell)):
            raise self._fail(f"frame {index} has no finite 3x3 Lattice")
        if abs(float(np.linalg.det(frame_cell))) <= 1.0e-12:
            raise self._fail(f"frame {index} has a singular Lattice")
        if self.cell is not None:
            change = float(np.max(np.abs(frame_cell - self.cell)))
            if change > CELL_TOLERANCE_A:
                raise self._fail(
                    f"frame {index} changed the cell by {change:.3e} A "
                    f"(> {CELL_TOLERANCE_A:g} A); the native NVE/NVT ensembles "
                    "are fixed-cell"
                )
        names = tuple(property_names)
        if self._property_names is None:
            self._property_names = names
        elif names != self._property_names:
            raise self._fail(
                f"frame {index} dumps columns {names}, but frame 0 dumped "
                f"{self._property_names}; mixed layouts cannot be one trajectory"
            )
        if time_fs is None or not math.isfinite(float(time_fs)):
            raise self._fail(
                f"frame {index} has no finite Time= value; the time axis is "
                "recorded by GPUMD and is never inferred"
            )
        if step is not None:
            derived = round(float(time_fs) / self.dt_fs)
            if int(step) != derived:
                raise self._fail(
                    f"frame {index} states step={step} but Time={time_fs} fs at "
                    f"dt={self.dt_fs:g} fs is step {derived}"
                )
        self._times.append(float(time_fs))
        self._count += 1

    def observe_frame(self, frame: Frame) -> None:
        """Check one parsed :class:`~mliport.gpumd.parse.Frame`."""
        atoms = frame.atoms
        self.observe(
            symbols=atoms.get_chemical_symbols(),
            pbc=tuple(bool(flag) for flag in atoms.pbc),
            cell=np.asarray(atoms.cell.array, dtype=float),
            time_fs=frame.time_fs,
            property_names=tuple(frame.extra.get("properties", {})),
            step=frame.step,
        )

    def verify(self) -> MDScheduleReport:
        """Check the whole dump against ``run.in`` and return its report.

        Raises:
            GPUMDIncompleteRunError: If the dump is short, long, mistimed, or
                misaligned with ``thermo.out``.
        """
        count = self._count
        if self.expected_nframes == 0:
            raise self._fail(
                f"run.in requested {self.steps} steps with dump interval "
                f"{self.dump_interval}; GPUMD writes no frame before the first "
                "full interval, so there is no trajectory to analyse"
            )
        if count == 0:
            raise self._fail("the dump contains no frames")
        if count != self.expected_nframes:
            raise self._fail(
                f"the dump holds {count} frame(s) but run.in requested "
                f"{self.steps} steps with dump interval {self.dump_interval}, "
                f"which writes {self.expected_nframes}; the run did not finish "
                "its requested simulation (or the file mixes two runs)"
            )
        interval_fs = self.dump_interval * self.dt_fs
        times = np.array(self._times, dtype=float)
        if count > 1:
            spacing = np.diff(times)
            if not np.allclose(spacing, interval_fs, rtol=TIME_RTOL, atol=TIME_ATOL_FS):
                worst = float(spacing[np.argmax(np.abs(spacing - interval_fs))])
                raise self._fail(
                    "frame times are not uniformly spaced by dump_interval * dt = "
                    f"{interval_fs:g} fs (found a spacing of {worst:g} fs)"
                )
        ratio = times / self.dt_fs
        grid = np.rint(ratio)
        if np.any(np.abs(ratio - grid) > GRID_TOLERANCE_STEPS):
            raise self._fail(
                f"frame times are not on the dt = {self.dt_fs:g} fs step grid"
            )
        step_axis = grid.astype(np.int64)
        first_step = int(step_axis[0])
        last_step = int(step_axis[-1])
        if first_step < 0:
            raise self._fail(f"the first frame is at a negative step ({first_step})")
        window_start = self.steps - self.dump_interval
        if not (window_start <= last_step <= self.steps):
            raise self._fail(
                f"the last frame is at step {last_step}, outside the final dump "
                f"interval [{window_start}, {self.steps}] of the requested "
                f"{self.steps}-step run"
            )
        self._verify_thermo(count, interval_fs)
        names = self._property_names or ()
        unwrapped = any(name in names for name in _UNWRAPPED_NAMES)
        convention = "unwrapped" if unwrapped else "wrapped"
        return MDScheduleReport(
            nframes=count,
            expected_nframes=self.expected_nframes,
            steps_requested=self.steps,
            dump_interval=self.dump_interval,
            dt_fs=self.dt_fs,
            frame_interval_fs=interval_fs,
            first_step=first_step,
            last_step=last_step,
            first_time_fs=float(times[0]),
            last_time_fs=float(times[-1]),
            natoms=len(self.symbols),
            positions_convention=convention,
            thermo_rows=self.thermo_rows,
            steps=step_axis,
            times_fs=times,
        )

    def _verify_thermo(self, count: int, interval_fs: float) -> None:
        """Reconcile ``thermo.out`` with the frames and with ``run.in``."""
        if self.thermo_rows is not None and self.thermo_rows != count:
            raise self._fail(
                f"thermo.out has {self.thermo_rows} row(s) but the dump has "
                f"{count} frame(s); both are written every {self.dump_interval} "
                "steps, so they must agree"
            )
        if (
            self.thermo_dump_interval is not None
            and self.thermo_dump_interval != self.dump_interval
        ):
            raise self._fail(
                f"thermo.out states dump_thermo {self.thermo_dump_interval} but "
                f"run.in used {self.dump_interval}"
            )
        if self.thermo_dt_output_fs is not None and not math.isclose(
            self.thermo_dt_output_fs, interval_fs, rel_tol=TIME_RTOL, abs_tol=1.0e-6
        ):
            raise self._fail(
                f"thermo.out states dt_output {self.thermo_dt_output_fs:g} fs but "
                f"run.in implies {interval_fs:g} fs"
            )
