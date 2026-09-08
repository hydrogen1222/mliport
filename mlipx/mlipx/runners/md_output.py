"""Bounded asynchronous output for molecular-dynamics trajectories.

The MD thread submits immutable CPU snapshots. A single writer thread owns all
file handles, so GPU inference is not serialized behind text formatting and
per-frame open/close calls. The queue is deliberately small and bounded: a slow
filesystem applies backpressure instead of growing RAM without limit.
"""

from __future__ import annotations

import csv
import queue
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from ase.io.trajectory import TrajectoryWriter as AseTrajectoryWriter

from mlipx.writers.outcar import MDOutcarWriter
from mlipx.writers.xdatcar import XdatcarWriter

if TYPE_CHECKING:
    from ase import Atoms


MD_CSV_HEADER = [
    "step",
    "time_fs",
    "phase",
    "potential_energy_eV",
    "kinetic_energy_eV",
    "total_energy_eV",
    "temperature_K",
    "volume_A3",
    "max_force_raw_eV_A",
    "max_force_applied_eV_A",
    "configurational_stress_xx_eV_A3",
    "configurational_stress_yy_eV_A3",
    "configurational_stress_zz_eV_A3",
    "configurational_stress_yz_eV_A3",
    "configurational_stress_xz_eV_A3",
    "configurational_stress_xy_eV_A3",
    "total_stress_xx_eV_A3",
    "total_stress_yy_eV_A3",
    "total_stress_zz_eV_A3",
    "total_stress_yz_eV_A3",
    "total_stress_xz_eV_A3",
    "total_stress_xy_eV_A3",
    "configurational_pressure_GPa",
    "total_pressure_GPa",
]


def _stress_values(value: Any) -> list[Any]:
    if value is None:
        return [""] * 6
    return np.asarray(value, dtype=float).reshape(6).tolist()


def _csv_row(frame: dict[str, Any]) -> list[Any]:
    return [
        frame["step"],
        frame["time_fs"],
        frame["phase"],
        frame["energy"],
        frame["kinetic_energy"],
        frame["total_energy"],
        frame["temperature"],
        frame["volume"],
        frame["max_force_raw_eV_A"],
        frame["max_force_applied_eV_A"],
        *_stress_values(frame["configurational_stress"]),
        *_stress_values(frame["total_stress"]),
        frame["configurational_pressure_gpa"]
        if frame["configurational_pressure_gpa"] is not None
        else "",
        frame["total_pressure_gpa"] if frame["total_pressure_gpa"] is not None else "",
    ]


@dataclass(frozen=True)
class MDFrameSnapshot:
    """One saved frame detached from the live backend calculator."""

    atoms: Atoms
    summary: dict[str, Any]
    outcar_forces: np.ndarray | None


@dataclass
class MDFrameStats:
    """Constant-memory index used by results and artifact provenance."""

    count: int = 0
    first_step: int | None = None
    last_step: int | None = None
    production_start_frame: int | None = None

    def record(self, frame: dict[str, Any]) -> None:
        step = int(frame["step"])
        if self.first_step is None:
            self.first_step = step
        if self.production_start_frame is None and frame["phase"] == "production":
            self.production_start_frame = self.count
        self.last_step = step
        self.count += 1


class MDOutputError(RuntimeError):
    """Raised in the MD thread when asynchronous output has failed."""


class AsyncMDOutputWriter:
    """Write MD frames through a bounded, fail-closed background queue."""

    QUEUE_MAX_FRAMES: ClassVar[int] = 8
    FLUSH_EVERY_FRAMES: ClassVar[int] = 32
    FLUSH_EVERY_SECONDS: ClassVar[float] = 1.0
    _SENTINEL: ClassVar[object] = object()

    def __init__(
        self,
        *,
        trajectory_path: Path,
        csv_path: Path,
        write_trajectory: bool,
        xdatcar_writer: XdatcarWriter | None,
        xdatcar_path: Path,
        outcar_writer: MDOutcarWriter | None,
    ) -> None:
        self.trajectory_path = trajectory_path
        self.csv_path = csv_path
        self.write_trajectory = write_trajectory
        self.xdatcar_writer = xdatcar_writer
        self.xdatcar_path = xdatcar_path
        self.outcar_writer = outcar_writer
        self.stats = MDFrameStats()

        self._queue: queue.Queue[MDFrameSnapshot | object] = queue.Queue(
            maxsize=self.QUEUE_MAX_FRAMES
        )
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="mlipx-md-output",
            daemon=False,
        )
        self._thread.start()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise MDOutputError(
                f"Asynchronous MD output failed: {self._error}"
            ) from self._error

    def submit(self, frame: MDFrameSnapshot) -> None:
        if self._closed:
            raise RuntimeError("Cannot submit a frame to a closed MD output stream")
        while True:
            self.raise_if_failed()
            try:
                self._queue.put(frame, timeout=0.1)
                return
            except queue.Full:
                # Bounded backpressure is intentional: never drop a requested
                # scientific frame and never grow memory without limit.
                continue

    def close(self) -> None:
        if self._closed:
            self.raise_if_failed()
            return
        if self._error is None:
            while True:
                self.raise_if_failed()
                try:
                    self._queue.put(self._SENTINEL, timeout=0.1)
                    break
                except queue.Full:
                    continue
        self._thread.join()
        self._closed = True
        self.raise_if_failed()

    def _run(self) -> None:
        traj_writer = None
        csv_handle = None
        csv_writer = None
        xdatcar_started = False
        last_flush = time.monotonic()
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_handle = self.csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
                buffering=1024 * 1024,
            )
            csv_writer = csv.writer(csv_handle)
            csv_writer.writerow(MD_CSV_HEADER)

            if self.write_trajectory:
                self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                traj_writer = AseTrajectoryWriter(self.trajectory_path, mode="w")
            if self.outcar_writer is not None:
                self.outcar_writer.open_stream()

            while True:
                item = self._queue.get()
                if item is self._SENTINEL:
                    break
                assert isinstance(item, MDFrameSnapshot)

                # The raw binary trajectory and scalar CSV are authoritative.
                if traj_writer is not None:
                    traj_writer.write(item.atoms)
                csv_writer.writerow(_csv_row(item.summary))
                self.stats.record(item.summary)

                if self.xdatcar_writer is not None:
                    if not xdatcar_started:
                        self.xdatcar_writer.write_header(item.atoms, self.xdatcar_path)
                        self.xdatcar_writer.open_stream(self.xdatcar_path)
                        xdatcar_started = True
                    self.xdatcar_writer.append_frame(
                        self.xdatcar_path,
                        item.atoms,
                        step=int(item.summary["step"]),
                    )

                if self.outcar_writer is not None:
                    self.outcar_writer.append_frame(
                        item.atoms,
                        step=int(item.summary["step"]),
                        time_fs=float(item.summary["time_fs"]),
                        potential_energy=float(item.summary["energy"]),
                        kinetic_energy=float(item.summary["kinetic_energy"]),
                        total_energy=float(item.summary["total_energy"]),
                        temperature=float(item.summary["temperature"]),
                        forces=item.outcar_forces,
                        configurational_stress=item.summary["configurational_stress"],
                        total_stress=item.summary["total_stress"],
                    )

                now = time.monotonic()
                if (
                    self.stats.count % self.FLUSH_EVERY_FRAMES == 0
                    or now - last_flush >= self.FLUSH_EVERY_SECONDS
                ):
                    self._flush(csv_handle)
                    last_flush = now
        except BaseException as exc:  # propagated to the MD thread
            self._error = exc
        finally:
            for close in (
                getattr(traj_writer, "close", None),
                getattr(csv_handle, "close", None),
                self.xdatcar_writer.close_stream
                if self.xdatcar_writer is not None
                else None,
                self.outcar_writer.close_stream
                if self.outcar_writer is not None
                else None,
            ):
                if close is None:
                    continue
                try:
                    close()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc

    def _flush(self, csv_handle) -> None:
        csv_handle.flush()
        if self.xdatcar_writer is not None:
            self.xdatcar_writer.flush()
        if self.outcar_writer is not None:
            self.outcar_writer.flush()


class MDTrajectorySummary(Sequence[dict[str, Any]]):
    """Read scalar frame summaries lazily from ``md.csv``.

    This preserves the historical ``results['trajectory']`` sequence interface
    without retaining one Python dictionary (and stress arrays) per MD frame.
    """

    _CONFIG_STRESS: ClassVar[tuple[str, ...]] = tuple(
        f"configurational_stress_{component}_eV_A3"
        for component in ("xx", "yy", "zz", "yz", "xz", "xy")
    )
    _TOTAL_STRESS: ClassVar[tuple[str, ...]] = tuple(
        f"total_stress_{component}_eV_A3"
        for component in ("xx", "yy", "zz", "yz", "xz", "xy")
    )

    def __init__(self, csv_path: Path | str, frame_count: int) -> None:
        self.csv_path = Path(csv_path)
        self._frame_count = int(frame_count)

    def __len__(self) -> int:
        return self._frame_count

    @staticmethod
    def _optional_float(value: str) -> float | None:
        return float(value) if value != "" else None

    @classmethod
    def _optional_stress(
        cls, row: dict[str, str], columns: tuple[str, ...]
    ) -> np.ndarray | None:
        values = [row[name] for name in columns]
        if all(value == "" for value in values):
            return None
        if any(value == "" for value in values):
            raise ValueError("Incomplete stress tensor in MD thermodynamics CSV")
        return np.asarray([float(value) for value in values], dtype=float)

    @classmethod
    def _parse_row(cls, row: dict[str, str]) -> dict[str, Any]:
        return {
            "step": int(row["step"]),
            "time_fs": float(row["time_fs"]),
            "phase": row["phase"],
            "energy": float(row["potential_energy_eV"]),
            "kinetic_energy": float(row["kinetic_energy_eV"]),
            "total_energy": float(row["total_energy_eV"]),
            "temperature": float(row["temperature_K"]),
            "volume": float(row["volume_A3"]),
            "max_force_raw_eV_A": cls._optional_float(
                row.get("max_force_raw_eV_A", "")
            ),
            "max_force_applied_eV_A": cls._optional_float(
                row.get("max_force_applied_eV_A", "")
            ),
            "configurational_stress": cls._optional_stress(row, cls._CONFIG_STRESS),
            "total_stress": cls._optional_stress(row, cls._TOTAL_STRESS),
            "configurational_pressure_gpa": cls._optional_float(
                row["configurational_pressure_GPa"]
            ),
            "total_pressure_gpa": cls._optional_float(row["total_pressure_GPa"]),
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield self._parse_row(row)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        for position, frame in enumerate(self):
            if position == normalized:
                return frame
        raise IndexError(index)
