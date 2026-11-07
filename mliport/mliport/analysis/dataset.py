"""Canonical trajectory data contract for Analysis v2.

This module only reads and normalizes trajectory data.  It deliberately does
not calculate diffusion or any other derived scientific observable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from ase import units
from ase.io import iread
from ase.io.trajectory import Trajectory
from ase.stress import full_3x3_to_voigt_6_stress

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any

    from ase import Atoms

PositionsConvention = Literal["wrapped", "unwrapped", "unknown"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


#: Schema of the manifest written by the native GPUMD runner.
_GPUMD_MANIFEST_SCHEMA = "mliport.gpumd-manifest/1"
_GPUMD_TRAJECTORY_NAME = "trajectory.extxyz"


def _gpumd_manifest(run_dir: Path) -> dict[str, Any] | None:
    """Return the native-GPUMD manifest of ``run_dir``, or ``None`` if it has none."""
    path = run_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = _read_json(path)
    except (OSError, ValueError):
        return None
    return manifest if manifest.get("schema") == _GPUMD_MANIFEST_SCHEMA else None


def _discover_source(source: str | Path) -> tuple[Path, Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.name == _GPUMD_TRAJECTORY_NAME and _gpumd_manifest(path.parent):
            return path.parent, path
        if path.parent.name in {"raw", "vasp"}:
            return path.parent.parent, path
        return path.parent, path
    if not path.is_dir():
        raise FileNotFoundError(f"Trajectory or run directory not found: {path}")
    if _gpumd_manifest(path) is not None:
        # The manifest, not the mere presence of a file, decides whether this
        # directory holds a trajectory; ``load`` verifies it before reading.
        return path, path / _GPUMD_TRAJECTORY_NAME
    for candidate in (
        path / "raw" / "trajectory.traj",
        path / "trajectory.traj",
        path / "vasp" / "XDATCAR",
        path / "XDATCAR",
    ):
        if candidate.is_file():
            return path, candidate
    raise FileNotFoundError(
        "No supported trajectory found. Expected raw/trajectory.traj, "
        "trajectory.traj, vasp/XDATCAR, XDATCAR, or a native GPUMD run "
        "directory (manifest.json + trajectory.extxyz) under " + str(path)
    )


@dataclass(frozen=True, slots=True)
class _NativeGPUMDRun:
    """What the manifest of a native GPUMD run vouches for (see ``_verify_native``)."""

    manifest: dict[str, Any]
    block: dict[str, Any]
    nframes: int
    natoms: int
    convention: str
    dt_fs: float
    dump_interval: int
    frame_interval_fs: float
    first_step: int
    last_step: int
    first_time_fs: float
    last_time_fs: float
    temperature_K: float | None


def _native_fail(message: str) -> Exception:
    from mliport.analysis.validation import InvalidTrajectoryError  # noqa: PLC0415

    return InvalidTrajectoryError(f"Native GPUMD run refused: {message}")


def _manifest_number(block: dict[str, Any], key: str, *, integer: bool) -> Any:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _native_fail(f"manifest trajectory.{key} is missing or not a number")
    if not math.isfinite(float(value)):
        raise _native_fail(f"manifest trajectory.{key} is not finite")
    if integer:
        if float(value) != math.floor(float(value)):
            raise _native_fail(f"manifest trajectory.{key} is not an integer")
        return int(value)
    return float(value)


def _verify_native(run_dir: Path, source_path: Path) -> _NativeGPUMDRun:
    """Accept a native GPUMD run directory only if its manifest vouches for it.

    Fails closed: the run must be an MD run that finished with status
    ``completed``, its manifest must describe a verified schedule (frame count,
    recorded times, position convention), and ``trajectory.extxyz`` must be the
    byte-identical file the runner committed.  Nothing here infers a missing
    value; a manifest that lacks one is refused.
    """
    manifest = _gpumd_manifest(run_dir)
    if manifest is None:  # pragma: no cover - callers check first
        raise _native_fail(f"{run_dir} has no mliport.gpumd-manifest/1 manifest")
    status = manifest.get("status")
    if status != "completed":
        error = manifest.get("error") or {}
        detail = error.get("message") if isinstance(error, dict) else None
        raise _native_fail(
            f"manifest status is {status!r}, not 'completed'"
            + (f" ({detail})" if detail else "")
            + "; the raw GPUMD output under native/ is kept as evidence"
        )
    calc_type = manifest.get("calc_type")
    if calc_type != "md":
        raise _native_fail(
            f"calc_type is {calc_type!r}; only an MD run is a time trajectory "
            "(an optimisation path or single point must not enter transport "
            "analysis)"
        )
    result = manifest.get("result")
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise _native_fail("manifest result does not record a completed MD run")
    block = manifest.get("trajectory")
    if not isinstance(block, dict):
        raise _native_fail(
            "manifest has no trajectory block; this run did not commit a "
            "verified normalised trajectory"
        )
    relative = block.get("path")
    if (
        relative != _GPUMD_TRAJECTORY_NAME
        or manifest.get("trajectory_path") != relative
    ):
        raise _native_fail(
            f"manifest trajectory path {relative!r} is not {_GPUMD_TRAJECTORY_NAME!r}"
        )
    if source_path != run_dir / _GPUMD_TRAJECTORY_NAME:
        raise _native_fail("trajectory path escapes the run directory")
    if not source_path.is_file():
        raise _native_fail(f"{source_path} is missing although the manifest lists it")
    expected_hash = block.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise _native_fail("manifest records no SHA-256 for the trajectory")
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise _native_fail(
            f"{source_path.name} does not match the SHA-256 recorded in the "
            "manifest (modified or truncated after the run)"
        )
    convention = block.get("positions_convention")
    if convention not in {"wrapped", "unwrapped"}:
        raise _native_fail(
            f"manifest positions_convention {convention!r} is not wrapped/unwrapped"
        )
    nframes = _manifest_number(block, "nframes", integer=True)
    natoms = _manifest_number(block, "natoms", integer=True)
    dt_fs = _manifest_number(block, "dt_fs", integer=False)
    interval = _manifest_number(block, "frame_interval_fs", integer=False)
    first_step = _manifest_number(block, "first_step", integer=True)
    last_step = _manifest_number(block, "last_step", integer=True)
    first_time = _manifest_number(block, "first_time_fs", integer=False)
    last_time = _manifest_number(block, "last_time_fs", integer=False)
    schedule = block.get("schedule")
    if not isinstance(schedule, dict) or schedule.get("schedule_verified") is not True:
        raise _native_fail("manifest does not record a verified MD schedule")
    dump_interval = _manifest_number(schedule, "dump_interval", integer=True)
    if nframes < 1 or natoms < 1 or dt_fs <= 0 or interval <= 0 or dump_interval < 1:
        raise _native_fail("manifest schedule has a non-positive count or interval")
    if not math.isclose(interval, dump_interval * dt_fs, rel_tol=1e-9, abs_tol=1e-9):
        raise _native_fail(
            f"manifest frame interval {interval:g} fs != dump_interval * dt = "
            f"{dump_interval * dt_fs:g} fs"
        )
    if schedule.get("nframes") != nframes:
        raise _native_fail("manifest frame counts disagree")
    run_in = manifest.get("run_in")
    provenance = run_in.get("provenance") if isinstance(run_in, dict) else None
    temperature = (
        provenance.get("temperature") if isinstance(provenance, dict) else None
    )
    return _NativeGPUMDRun(
        manifest=manifest,
        block=block,
        nframes=nframes,
        natoms=natoms,
        convention=convention,
        dt_fs=dt_fs,
        dump_interval=dump_interval,
        frame_interval_fs=interval,
        first_step=first_step,
        last_step=last_step,
        first_time_fs=first_time,
        last_time_fs=last_time,
        temperature_K=(
            float(temperature)
            if isinstance(temperature, (int, float))
            and not isinstance(temperature, bool)
            else None
        ),
    )


def _check_native_frames(native: _NativeGPUMDRun, frames: _CollectedFrames) -> None:
    """Cross-check the parsed frames against what the manifest vouches for.

    The hash already proves the file is the committed one; this guards the
    reader (for example an ASE version that drops a key) and refuses to build a
    dataset whose frames contradict the declared time axis, step axis, PBC or
    position convention rather than quietly repairing any of them.
    """
    if frames.nframes != native.nframes or frames.positions.shape[1] != native.natoms:
        raise _native_fail(
            f"trajectory.extxyz has {frames.nframes} frames x "
            f"{frames.positions.shape[1]} atoms; the manifest records "
            f"{native.nframes} x {native.natoms}"
        )
    if not bool(np.all(frames.pbc)):
        raise _native_fail(
            f"frames have pbc={frames.pbc.tolist()}; native GPUMD runs are 3-D periodic"
        )
    conventions = set(frames.info.get("positions_convention", []))
    if conventions != {native.convention}:
        raise _native_fail(
            f"frames declare positions_convention {sorted(map(str, conventions))}, "
            f"the manifest declares {native.convention!r}"
        )
    times = frames.info_values("mliport_time_fs")
    steps = frames.info_values("mliport_step")
    if times is None or steps is None:
        raise _native_fail("a frame lacks its recorded time or step")
    time_axis = np.asarray(times, dtype=float)
    step_axis = np.asarray(steps)
    if not (
        np.isclose(time_axis[0], native.first_time_fs, rtol=1e-9, atol=1e-9)
        and np.isclose(time_axis[-1], native.last_time_fs, rtol=1e-9, atol=1e-9)
        and int(step_axis[0]) == native.first_step
        and int(step_axis[-1]) == native.last_step
    ):
        raise _native_fail("frame times/steps differ from the manifest's verified span")
    if not np.array_equal(
        np.rint(time_axis / native.dt_fs).astype(np.int64), step_axis
    ):
        raise _native_fail("frame steps are inconsistent with frame times and dt")


def _iter_frames(path: Path) -> Iterator[Atoms]:
    """Yield frames one at a time; a trajectory is never held as Atoms objects.

    Long MLMD trajectories can contain 10^5-10^6 frames with per-frame force
    metadata. Materialising every ``Atoms`` (with its info dictionary and
    calculator results) would cost far more memory than the position array the
    analyses actually need, so frames are reduced to arrays as they are read.
    """
    if path.suffix.lower() == ".traj":
        with Trajectory(path, mode="r") as trajectory:
            yield from trajectory
        return
    yield from iread(path, index=":")


#: Per-frame ``atoms.info`` keys the dataset contract consumes.
_FRAME_INFO_KEYS = (
    "mliport_time_fs",
    "mliport_step",
    "mliport_phase",
    "trajectory_kind",
    "positions_convention",
)


@dataclass(slots=True)
class _CollectedFrames:
    """Array form of a frame sequence (see :func:`_collect_frames`)."""

    symbols: tuple[str, ...]
    masses: np.ndarray
    pbc: np.ndarray
    positions: np.ndarray
    cells: np.ndarray
    velocities: np.ndarray | None
    kinetic_energies: np.ndarray | None
    temperatures: np.ndarray | None
    energies: np.ndarray | None
    stresses: np.ndarray | None
    info: dict[str, list[Any]]

    @property
    def nframes(self) -> int:
        return int(self.positions.shape[0])

    def info_values(self, key: str) -> list[Any] | None:
        """All per-frame values of ``key``, or ``None`` if any frame lacks it."""
        values = self.info.get(key, [])
        if len(values) != self.nframes or any(value is None for value in values):
            return None
        return values


def _collect_frames(frames: Iterable[Atoms]) -> _CollectedFrames | None:
    """Reduce frames to arrays one frame at a time; ``None`` for no frames."""
    symbols: tuple[str, ...] | None = None
    masses = pbc = None
    natoms = 0
    positions: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    velocities: list[np.ndarray] | None = []
    kinetic: list[float] | None = []
    temperatures: list[float] | None = []
    energies: list[float] | None = []
    stresses: list[np.ndarray] | None = []
    info: dict[str, list[Any]] = {key: [] for key in _FRAME_INFO_KEYS}
    for index, atoms in enumerate(frames):
        if symbols is None:
            symbols = tuple(atoms.get_chemical_symbols())
            masses = np.asarray(atoms.get_masses(), dtype=float)
            pbc = np.asarray(atoms.pbc, dtype=bool)
            natoms = len(symbols)
        if len(atoms) != natoms:
            raise ValueError(f"Frame {index} has a different atom count")
        if tuple(atoms.get_chemical_symbols()) != symbols:
            raise ValueError(f"Frame {index} changes atom identity/order")
        if not np.array_equal(np.asarray(atoms.pbc, bool), pbc):
            raise ValueError(f"Frame {index} changes PBC flags")
        positions.append(np.array(atoms.positions, dtype=float, copy=True))
        cells.append(np.array(atoms.cell.array, dtype=float, copy=True))
        for key in _FRAME_INFO_KEYS:
            info[key].append(atoms.info.get(key))
        if velocities is not None and atoms.has("momenta"):
            # ASE's internal time unit is not fs. Keep the analysis
            # contract explicit instead of labelling raw ASE values as
            # A/fs and silently scaling VACF-derived quantities.
            velocities.append(
                np.asarray(atoms.get_velocities(), dtype=float) * units.fs
            )
            kinetic.append(float(atoms.get_kinetic_energy()))
            temperatures.append(float(atoms.get_temperature()))
        else:
            velocities = kinetic = temperatures = None
        results = getattr(atoms.calc, "results", {}) if atoms.calc else {}
        energy = results.get("energy", results.get("free_energy"))
        if energy is None:
            energies = None
        elif energies is not None:
            energies.append(float(energy))
        stress = results.get("stress")
        if stress is None or stresses is None:
            stresses = None
        else:
            stress = np.asarray(stress, dtype=float)
            if stress.shape == (3, 3):
                stress = full_3x3_to_voigt_6_stress(stress)
            if stress.shape != (6,):
                stresses = None
            else:
                stresses.append(stress)
    if symbols is None:
        return None

    def stack(rows):
        return None if rows is None else np.asarray(rows, dtype=float)

    return _CollectedFrames(
        symbols=symbols,
        masses=masses,
        pbc=pbc,
        positions=np.asarray(positions, dtype=float),
        cells=np.asarray(cells, dtype=float),
        velocities=stack(velocities),
        kinetic_energies=stack(kinetic),
        temperatures=stack(temperatures),
        energies=stack(energies),
        stresses=stack(stresses),
        info=info,
    )


def _parse_step_column(rows, key: str, *, source: str) -> np.ndarray | None:
    """Parse a CSV ``step`` column exactly, or return ``None`` when unknown.

    The cell text is read with :class:`~decimal.Decimal`, so ``9007199254740993``
    and ``9007199254740993.0`` keep their exact integer value instead of being
    rounded through ``float``.  A finite value that is not an exact integer is
    rejected; a blank, unparseable or non-finite cell makes the whole column
    "unknown", which is the pre-existing fallback (the caller then uses frame
    info or the declared stride instead of fabricating an axis).
    """

    values: list[int] = []
    unknown = False
    for row in rows:
        raw = row.get(key, "")
        text = "" if raw is None else str(raw).strip()
        if not text:
            unknown = True
            continue
        try:
            number = Decimal(text)
        except InvalidOperation:
            unknown = True
            continue
        if not number.is_finite():
            unknown = True
            continue
        integral = number.to_integral_value()
        if number != integral:
            raise ValueError(
                f"{source} must contain integer step indices; refusing to "
                f"truncate fractional value {text!r}"
            )
        # Compare against the bounds as a Decimal: this rejects an
        # astronomically large exponent (``1e999999999``) without first
        # building a huge Python int.
        if integral < _INT64_MIN or integral > _INT64_MAX:
            raise ValueError(
                f"{source} contains a step index outside the int64 range: {text!r}"
            )
        values.append(int(integral))
    if unknown:
        return None
    return np.asarray(values, dtype=np.int64)


def _read_thermodynamics(run_dir: Path) -> dict[str, np.ndarray]:
    for path in (run_dir / "raw" / "md.csv", run_dir / "md.csv"):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return {}
        columns: dict[str, np.ndarray] = {}
        for key in rows[0]:
            if key == "phase":
                columns[key] = np.asarray(
                    [str(row.get(key, "")) for row in rows], dtype="U16"
                )
                continue
            if key == "step":
                # Validate the text before any lossy float parse: step indices
                # are exact integers, and ``float("9007199254740993")`` rounds.
                parsed = _parse_step_column(rows, key, source="raw/md.csv step column")
                columns[key] = np.full(len(rows), np.nan) if parsed is None else parsed
                continue
            values: list[float] = []
            for row in rows:
                raw = row.get(key, "")
                try:
                    values.append(float(raw) if raw not in {None, ""} else np.nan)
                except (TypeError, ValueError):
                    values.append(np.nan)
            columns[key] = np.asarray(values, dtype=float)
        return columns
    return {}


def _uniform_interval(times_fs: np.ndarray) -> float | None:
    if len(times_fs) < 2 or not np.all(np.isfinite(times_fs)):
        return None
    differences = np.diff(times_fs)
    if not np.all(differences > 0):
        return None
    reference = float(differences[0])
    tolerance = max(1.0e-10, abs(reference) * 1.0e-10)
    if not np.allclose(differences, reference, rtol=1.0e-6, atol=tolerance):
        return None
    return reference


def _uniform_integer_step(steps: np.ndarray | None) -> int | None:
    """Return a constant positive step spacing, or ``None`` if unknown."""

    if steps is None or len(steps) < 2:
        return None
    values = np.asarray(steps)
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        return None
    differences = np.diff(values)
    if not np.all(differences > 0):
        return None
    if not np.all(differences == differences[0]):
        return None
    return int(differences[0])


#: Every integer up to ``2**53`` is exactly representable as a float64; beyond
#: that a float cannot name a distinct integer, so a float step there is not a
#: trustworthy step index.  Integer input is bounded by the ``int64`` range
#: instead (which is asymmetric: ``-2**63`` is representable).
_MAX_EXACT_FLOAT_INTEGER = float(2**53)
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _step_index(value: Any, *, source: str) -> int:
    """Return one exact integer step index, or refuse the element.

    Python's exact ``int`` is checked first: an integer is never routed through
    ``float``, where ``2**53 + 1`` would silently become ``2**53``.  A float is
    accepted only when it is finite, integral and within the exactly
    representable range.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{source} must contain integer step indices")
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        if integer < _INT64_MIN or integer > _INT64_MAX:
            raise ValueError(
                f"{source} contains a step index outside the int64 range: {integer}"
            )
        return integer
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{source} must contain finite integer step indices")
        if number != math.floor(number):
            raise ValueError(
                f"{source} must contain integer step indices; refusing to "
                f"truncate fractional value {number!r}"
            )
        if abs(number) > _MAX_EXACT_FLOAT_INTEGER:
            raise ValueError(
                f"{source} contains a step index beyond the exactly "
                f"representable float range: {number!r}"
            )
        return int(number)
    raise ValueError(f"{source} must contain integer step indices")


def _validated_step_array(array: np.ndarray, *, source: str) -> np.ndarray:
    """Vectorised form of :func:`_step_index` for an already-built array."""

    if np.issubdtype(array.dtype, np.integer):
        if array.size:
            low = int(array.min())
            high = int(array.max())
            if low < _INT64_MIN or high > _INT64_MAX:
                offending = low if low < _INT64_MIN else high
                raise ValueError(
                    f"{source} contains a step index outside the int64 range: "
                    f"{offending}"
                )
        return array.astype(np.int64, copy=False)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{source} must contain integer step indices")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{source} must contain finite integer step indices")
    fractional = array != np.floor(array)
    if np.any(fractional):
        raise ValueError(
            f"{source} must contain integer step indices; refusing to "
            f"truncate fractional values {array[fractional][:5].tolist()}"
        )
    inexact = np.abs(array) > _MAX_EXACT_FLOAT_INTEGER
    if np.any(inexact):
        raise ValueError(
            f"{source} contains a step index beyond the exactly representable "
            f"float range; refusing to trust {array[inexact][:5].tolist()}"
        )
    return array.astype(np.int64)


def _validated_steps(values: Any, *, source: str) -> np.ndarray:
    """Return ``values`` as ``int64`` step indices, refusing invalid input.

    Step indices are integers by contract: mliport writes ``int`` steps and
    :func:`_uniform_integer_step` only accepts integral values.  A non-array
    sequence is validated element by element *before* it is turned into an
    array, because ``np.asarray([2**53 + 1, 2.0])`` promotes to ``float64`` and
    silently rounds the integer.  Text columns are parsed exactly by
    :func:`_parse_step_column`.  Anything invalid is rejected rather than
    reinterpreted, because a truncated or wrapped value fabricates a step axis
    that is not in the file (AGENTS.md: never silently change dtype or
    semantics).
    """

    if isinstance(values, np.ndarray):
        return _validated_step_array(values, source=source)
    return np.asarray(
        [_step_index(value, source=source) for value in values], dtype=np.int64
    )


@dataclass(slots=True)
class TrajectoryDataset:
    """Normalized, backend-independent trajectory arrays and metadata.

    ``velocities`` are stored in Angstrom per femtosecond. ASE exposes
    velocities in Angstrom per ASE time unit, so conversion is performed while
    importing ASE frames; datasets constructed directly must use this same
    contract.
    """

    run_dir: Path
    source_path: Path
    positions: np.ndarray
    cells: np.ndarray
    pbc: np.ndarray
    symbols: tuple[str, ...]
    masses: np.ndarray
    times_fs: np.ndarray | None
    steps: np.ndarray | None
    positions_convention: PositionsConvention = "unknown"
    md_timestep_fs: float | None = None
    frame_stride_steps: int | None = None
    frame_interval_fs: float | None = None
    velocities: np.ndarray | None = None
    temperature_K: np.ndarray | None = None
    potential_energy_eV: np.ndarray | None = None
    kinetic_energy_eV: np.ndarray | None = None
    total_energy_eV: np.ndarray | None = None
    stress_eV_A3: np.ndarray | None = None
    pressure_GPa: np.ndarray | None = None
    volumes_A3: np.ndarray | None = None
    phases: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    thermodynamics: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.cells = np.asarray(self.cells, dtype=float)
        self.pbc = np.asarray(self.pbc, dtype=bool)
        self.masses = np.asarray(self.masses, dtype=float)
        if self.positions.ndim != 3 or self.positions.shape[-1] != 3:
            raise ValueError("positions must have shape (frames, atoms, 3)")
        if self.cells.shape != (self.nframes, 3, 3):
            raise ValueError("cells must have shape (frames, 3, 3)")
        if self.pbc.shape != (3,):
            raise ValueError("pbc must have shape (3,)")
        if len(self.symbols) != self.natoms or self.masses.shape != (self.natoms,):
            raise ValueError("symbols/masses do not match the trajectory atom count")
        if self.positions_convention not in {"wrapped", "unwrapped", "unknown"}:
            raise ValueError(
                "positions_convention must be wrapped, unwrapped, or unknown"
            )
        for name in ("times_fs", "steps", "temperature_K", "phases"):
            value = getattr(self, name)
            if value is not None and len(value) != self.nframes:
                raise ValueError(f"{name} must contain one value per frame")

    @property
    def nframes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def natoms(self) -> int:
        return int(self.positions.shape[1])

    @property
    def source(self) -> str:
        return str(self.source_path)

    @property
    def target_temperature_K(self) -> float | None:
        value = self.metadata.get("target_temperature_K")
        return float(value) if value is not None else None

    @classmethod
    def from_frames(
        cls,
        frames: Iterable[Atoms],
        *,
        times_fs: Iterable[float] | None,
        positions_convention: PositionsConvention,
        source: str | Path = "<memory>",
        steps: Iterable[int] | None = None,
        phases: Iterable[str] | None = None,
        md_timestep_fs: float | None = None,
        frame_stride_steps: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrajectoryDataset:
        """Build a dataset from ASE frames with explicit trajectory semantics."""

        collected = _collect_frames(frames)
        if collected is None:
            raise ValueError("Trajectory contains no frames")
        source_path = Path(source)
        return cls._from_collected(
            collected,
            run_dir=source_path.parent,
            source_path=source_path,
            times_fs=None if times_fs is None else np.asarray(list(times_fs), float),
            steps=(
                None
                if steps is None
                else _validated_steps(list(steps), source="steps argument")
            ),
            phases=None if phases is None else np.asarray(list(phases), dtype="U16"),
            positions_convention=positions_convention,
            md_timestep_fs=md_timestep_fs,
            frame_stride_steps=frame_stride_steps,
            metadata=dict(metadata or {}),
            thermodynamics={},
        )

    @classmethod
    def load(
        cls,
        source: str | Path,
        *,
        positions_convention: PositionsConvention | None = None,
        frame_interval_fs: float | None = None,
        md_timestep_fs: float | None = None,
        frame_stride_steps: int | None = None,
    ) -> TrajectoryDataset:
        """Load an mliport run, ASE trajectory, or XDATCAR.

        Explicit function arguments override metadata only after consistency
        checks.  A missing time axis is never fabricated from a median frame
        difference.
        """

        run_dir, source_path = _discover_source(source)
        native: _NativeGPUMDRun | None = None
        if (
            source_path == run_dir / _GPUMD_TRAJECTORY_NAME
            and _gpumd_manifest(run_dir) is not None
        ):
            # Verified *before* the file is parsed: a failed, partial or
            # modified native run is refused, never loaded as a trajectory.
            native = _verify_native(run_dir, source_path)
            if md_timestep_fs is not None and not math.isclose(
                float(md_timestep_fs), native.dt_fs, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(
                    f"md_timestep_fs={md_timestep_fs!r} conflicts with the native "
                    f"GPUMD manifest ({native.dt_fs:g} fs)"
                )
            if frame_stride_steps is not None and int(frame_stride_steps) != (
                native.dump_interval
            ):
                raise ValueError(
                    f"frame_stride_steps={frame_stride_steps!r} conflicts with the "
                    f"native GPUMD dump interval ({native.dump_interval})"
                )
        frames = _collect_frames(_iter_frames(source_path))
        if frames is None:
            raise ValueError(f"Trajectory contains no frames: {source_path}")
        if native is not None:
            _check_native_frames(native, frames)

        artifacts = _read_json(run_dir / "artifacts.json")
        resolved = _read_json(run_dir / "resolved_config.json")
        trajectory_meta = artifacts.get("trajectory", {})
        if native is not None:
            trajectory_meta = {
                "md_timestep_fs": native.dt_fs,
                "frame_stride_steps": native.dump_interval,
                "frame_interval_fs": native.frame_interval_fs,
                "positions_convention": native.convention,
            }
        # ASE's standard XDATCAR reader unconditionally sets pbc=True even
        # for slab/molecule input. Treat that value as an unverified parser
        # assumption unless the matching mliport MD manifest records a mask.
        is_xdatcar = source_path.name.upper().endswith("XDATCAR")
        pbc_verified = not is_xdatcar
        pbc_source = "ase-frame" if not is_xdatcar else "xdatcar-unknown"
        if is_xdatcar:
            xdatcar_artifact = (artifacts.get("artifacts") or {}).get("xdatcar")
            declared_pbc = trajectory_meta.get("pbc")
            if (
                artifacts.get("schema") == "mliport.md-artifacts/3"
                and isinstance(xdatcar_artifact, dict)
                and xdatcar_artifact.get("format") == "vasp-xdatcar"
                and xdatcar_artifact.get("path")
                == source_path.relative_to(run_dir).as_posix()
                and isinstance(declared_pbc, list)
                and len(declared_pbc) == 3
                and all(type(flag) is bool for flag in declared_pbc)
            ):
                frames.pbc = np.asarray(declared_pbc, dtype=bool)
                pbc_verified = True
                pbc_source = "mliport-md-artifacts"
        run_options = resolved.get("run_options", {})
        thermo = _read_thermodynamics(run_dir)
        nframes = frames.nframes

        metadata_timestep = trajectory_meta.get(
            "md_timestep_fs",
            trajectory_meta.get("timestep_fs", run_options.get("timestep")),
        )
        metadata_stride = trajectory_meta.get(
            "frame_stride_steps",
            trajectory_meta.get(
                "save_interval_steps", run_options.get("save_interval")
            ),
        )
        timestep = md_timestep_fs if md_timestep_fs is not None else metadata_timestep
        stride = (
            frame_stride_steps if frame_stride_steps is not None else metadata_stride
        )
        timestep = float(timestep) if timestep is not None else None
        stride = int(stride) if stride is not None else None
        if timestep is not None and timestep <= 0:
            raise ValueError("md_timestep_fs must be positive")
        if stride is not None and stride < 1:
            raise ValueError("frame_stride_steps must be >= 1")

        explicit_times: np.ndarray | None = None
        time_source = "unknown"
        if "time_fs" in thermo and len(thermo["time_fs"]) == nframes:
            explicit_times = np.asarray(thermo["time_fs"], dtype=float)
            time_source = "raw/md.csv"
        else:
            info_times = frames.info_values("mliport_time_fs")
            if info_times is not None:
                explicit_times = np.asarray(info_times, dtype=float)
                time_source = "ASE frame info"

        metadata_interval = trajectory_meta.get(
            "frame_interval_fs", trajectory_meta.get("saved_interval_fs")
        )
        derived_interval = (
            timestep * stride if timestep is not None and stride is not None else None
        )
        declared_interval = (
            frame_interval_fs
            if frame_interval_fs is not None
            else metadata_interval
            if metadata_interval is not None
            else derived_interval
        )
        declared_interval = (
            float(declared_interval) if declared_interval is not None else None
        )
        if declared_interval is not None and declared_interval <= 0:
            raise ValueError("frame_interval_fs must be positive")

        if explicit_times is None and declared_interval is not None:
            explicit_times = np.arange(nframes, dtype=float) * declared_interval
            time_source = "declared frame interval"
        observed_interval = (
            _uniform_interval(explicit_times) if explicit_times is not None else None
        )
        if (
            observed_interval is not None
            and declared_interval is not None
            and not np.isclose(
                observed_interval,
                declared_interval,
                rtol=1.0e-6,
                atol=max(1.0e-10, abs(declared_interval) * 1.0e-10),
            )
        ):
            raise ValueError(
                "Explicit trajectory time axis conflicts with declared frame interval: "
                f"observed {observed_interval:g} fs, declared {declared_interval:g} fs"
            )

        steps: np.ndarray | None = None
        if "step" in thermo and len(thermo["step"]) == nframes:
            values = np.asarray(thermo["step"])
            if np.all(np.isfinite(values)):
                steps = _validated_steps(values, source="raw/md.csv step column")
        if steps is None:
            info_steps = frames.info_values("mliport_step")
            if info_steps is not None:
                steps = np.asarray(info_steps, dtype=int)
        if steps is None and stride is not None:
            steps = np.arange(nframes, dtype=int) * stride

        phases: np.ndarray | None = None
        if "phase" in thermo and len(thermo["phase"]) == nframes:
            phases = np.asarray(thermo["phase"], dtype="U16")
        if phases is None:
            info_phases = frames.info_values("mliport_phase")
            if info_phases is not None:
                phases = np.asarray(info_phases, dtype="U16")
        production_start_step = trajectory_meta.get("production_start_step")
        if phases is None and production_start_step is not None and steps is not None:
            phases = np.where(
                steps >= int(production_start_step), "production", "equilibration"
            )
        if phases is None:
            phases = np.full(nframes, "production", dtype="U16")

        value = str(trajectory_meta.get("positions_convention", "")).lower()
        legacy = str(trajectory_meta.get("positions", "")).lower()
        if value in {"wrapped", "unwrapped"}:
            declared_convention: PositionsConvention = value  # type: ignore[assignment]
        elif "unwrapped" in legacy:
            declared_convention = "unwrapped"
        elif "wrapped" in legacy:
            declared_convention = "wrapped"
        else:
            declared_convention = "unknown"
        if (
            positions_convention is not None
            and declared_convention in {"wrapped", "unwrapped"}
            and positions_convention != declared_convention
        ):
            raise ValueError(
                "Explicit positions_convention conflicts with the mliport "
                f"trajectory artifact: requested {positions_convention!r}, "
                f"artifact declares {declared_convention!r}. Refusing to "
                "reinterpret periodic image semantics."
            )
        convention = (
            positions_convention
            if positions_convention is not None
            else declared_convention
        )

        target_temperature = run_options.get("temperature")
        source_format = (
            "ase-traj" if source_path.suffix.lower() == ".traj" else "xdatcar"
        )
        run_status = artifacts.get("status", "external_or_unknown")
        if native is not None:
            target_temperature = native.temperature_K
            source_format = "gpumd-extxyz"
            run_status = native.manifest["status"]
            time_source = "GPUMD dump Time= (recorded)"
            pbc_source = "gpumd-manifest"
        metadata = {
            "artifacts": artifacts,
            "resolved_config": resolved,
            "run_status": run_status,
            "time_source": time_source,
            "target_temperature_K": target_temperature,
            "source_format": source_format,
            "pbc_verified": pbc_verified,
            "pbc_source": pbc_source,
        }
        if native is not None:
            metadata["gpumd_manifest"] = native.manifest
        return cls._from_collected(
            frames,
            run_dir=run_dir,
            source_path=source_path,
            times_fs=explicit_times,
            steps=steps,
            phases=phases,
            positions_convention=convention,
            md_timestep_fs=timestep,
            frame_stride_steps=stride,
            metadata=metadata,
            thermodynamics=thermo,
        )

    @classmethod
    def _from_collected(
        cls,
        frames: _CollectedFrames,
        *,
        run_dir: Path,
        source_path: Path,
        times_fs: np.ndarray | None,
        steps: np.ndarray | None,
        phases: np.ndarray | None,
        positions_convention: PositionsConvention,
        md_timestep_fs: float | None,
        frame_stride_steps: int | None,
        metadata: dict[str, Any],
        thermodynamics: dict[str, np.ndarray],
    ) -> TrajectoryDataset:
        frame_kinds = set(frames.info.get("trajectory_kind", []))
        artifact_kind = (metadata.get("artifacts") or {}).get("trajectory_kind")
        if "neb_band" in frame_kinds or artifact_kind == "neb_band":
            from mliport.analysis.validation import (  # noqa: PLC0415
                InvalidTrajectoryError,
            )

            raise InvalidTrajectoryError(
                "trajectory_kind='neb_band' is a reaction-coordinate image set, "
                "not a time trajectory; MSD, VACF, kinisi, GEMDAT, and other "
                "trajectory analyses are forbidden"
            )
        symbols = frames.symbols
        masses = frames.masses
        pbc = frames.pbc
        positions = frames.positions
        cells = frames.cells
        nframes = frames.nframes
        has_velocities = frames.velocities is not None
        has_energies = frames.energies is not None
        has_stresses = frames.stresses is not None

        def thermo_column(*names: str) -> np.ndarray | None:
            for name in names:
                value = thermodynamics.get(name)
                if value is not None and len(value) == nframes:
                    return np.asarray(value, dtype=float)
            return None

        potential = thermo_column("potential_energy_eV")
        if potential is None and has_energies:
            potential = frames.energies
        kinetic = thermo_column("kinetic_energy_eV")
        if kinetic is None and has_velocities:
            kinetic = frames.kinetic_energies
        total = thermo_column("total_energy_eV")
        temperature = thermo_column("temperature_K")
        if temperature is None and has_velocities:
            temperature = frames.temperatures
        if total is None and potential is not None and kinetic is not None:
            total = potential + kinetic
        volume = thermo_column("volume_A3")
        if volume is None:
            volume = np.asarray([abs(np.linalg.det(cell)) for cell in cells])
        pressure = thermo_column("total_pressure_GPa", "pressure_GPa")
        stress = frames.stresses if has_stresses else None
        if stress is None:
            components = [
                thermo_column(f"total_stress_{axis}_eV_A3", f"stress_{axis}_eV_A3")
                for axis in ("xx", "yy", "zz", "yz", "xz", "xy")
            ]
            if all(component is not None for component in components):
                stress = np.column_stack(components)
        interval = _uniform_interval(times_fs) if times_fs is not None else None
        warnings: list[str] = []
        if positions_convention == "unknown":
            warnings.append(
                "Position convention is unknown. Explicitly declare wrapped or "
                "unwrapped coordinates before MSD/transport analysis."
            )
        if times_fs is None:
            warnings.append(
                "Trajectory time axis is unknown. Supply an explicit frame interval "
                "for time-dependent analyses."
            )
        metadata = dict(metadata)
        if has_velocities:
            metadata["velocity_unit"] = "A/fs"
        return cls(
            run_dir=run_dir,
            source_path=source_path,
            positions=positions,
            cells=cells,
            pbc=pbc,
            symbols=symbols,
            masses=masses,
            times_fs=times_fs,
            steps=steps,
            positions_convention=positions_convention,
            md_timestep_fs=md_timestep_fs,
            frame_stride_steps=frame_stride_steps,
            frame_interval_fs=interval,
            velocities=frames.velocities,
            temperature_K=temperature,
            potential_energy_eV=potential,
            kinetic_energy_eV=kinetic,
            total_energy_eV=total,
            stress_eV_A3=stress,
            pressure_GPa=pressure,
            volumes_A3=volume,
            phases=phases,
            metadata=metadata,
            thermodynamics=thermodynamics,
            warnings=warnings,
        )

    def select(
        self,
        species: str | None = None,
        *,
        indices: Iterable[int] | None = None,
    ) -> np.ndarray:
        """Return atom indices selected by exactly one explicit mechanism."""

        if species is not None and indices is not None:
            raise ValueError("Select atoms by species or indices, not both")
        if indices is not None:
            selected = np.asarray(list(indices), dtype=int)
            if selected.ndim != 1 or len(selected) == 0:
                raise ValueError("indices must be a non-empty one-dimensional list")
            if np.any(selected < 0) or np.any(selected >= self.natoms):
                raise IndexError("atom selection contains an out-of-range index")
            if len(np.unique(selected)) != len(selected):
                raise ValueError("atom selection contains duplicate indices")
            return selected
        if species is None:
            return np.arange(self.natoms, dtype=int)
        selected = np.asarray(
            [index for index, symbol in enumerate(self.symbols) if symbol == species],
            dtype=int,
        )
        if len(selected) == 0:
            raise ValueError(f"Selected species {species!r} is absent")
        return selected

    def frame_indices(
        self,
        *,
        include_equilibration: bool = False,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
    ) -> np.ndarray:
        if stride < 1:
            raise ValueError("frame stride must be >= 1")
        lower = 0 if start is None else int(start)
        upper = self.nframes if stop is None else int(stop)
        if lower < 0 or upper > self.nframes or lower >= upper:
            raise ValueError(
                f"Invalid frame range [{lower}, {upper}) for {self.nframes} frames"
            )
        selected = np.arange(lower, upper, stride, dtype=int)
        if not include_equilibration and self.phases is not None:
            selected = selected[self.phases[selected] == "production"]
        if len(selected) == 0:
            raise ValueError("The requested analysis range contains no frames")
        return selected

    def slice_frames(self, indices: Iterable[int]) -> TrajectoryDataset:
        """Return a frame-sliced dataset while preserving atom semantics."""

        raw_indices = list(indices)
        selected_raw = np.asarray(raw_indices)
        if selected_raw.ndim != 1 or len(selected_raw) == 0:
            raise ValueError("Frame selection must be non-empty")
        if not np.issubdtype(selected_raw.dtype, np.integer):
            raise TypeError("Frame selection indices must be integers")
        selected = selected_raw.astype(int, copy=False)
        if np.any(selected < 0) or np.any(selected >= self.nframes):
            raise IndexError("Frame selection contains an out-of-range index")
        if len(selected) > 1 and np.any(np.diff(selected) <= 0):
            raise ValueError("Frame selection must be strictly increasing")

        def take(value):
            return None if value is None else np.asarray(value)[selected]

        sliced_times = take(self.times_fs)
        sliced_steps = take(self.steps)
        interval = _uniform_interval(sliced_times)
        step_stride = _uniform_integer_step(sliced_steps)
        if step_stride is None and self.steps is None:
            original_stride = self.frame_stride_steps
            index_stride = _uniform_integer_step(selected)
            if original_stride is not None and index_stride is not None:
                step_stride = int(original_stride) * index_stride

        metadata = dict(self.metadata)
        transformations = metadata.get("transformations", [])
        if not isinstance(transformations, list):
            transformations = []
        metadata["transformations"] = [
            *transformations,
            {
                "operation": "frame_slice",
                "source_nframes": self.nframes,
                "selected_indices": selected.tolist(),
            },
        ]

        sliced_thermo = {
            key: np.asarray(value)[selected]
            for key, value in self.thermodynamics.items()
            if len(value) == self.nframes
        }
        return replace(
            self,
            positions=self.positions[selected],
            cells=self.cells[selected],
            times_fs=sliced_times,
            steps=sliced_steps,
            frame_stride_steps=step_stride,
            frame_interval_fs=interval,
            velocities=take(self.velocities),
            temperature_K=take(self.temperature_K),
            potential_energy_eV=take(self.potential_energy_eV),
            kinetic_energy_eV=take(self.kinetic_energy_eV),
            total_energy_eV=take(self.total_energy_eV),
            stress_eV_A3=take(self.stress_eV_A3),
            pressure_GPa=take(self.pressure_GPa),
            volumes_A3=take(self.volumes_A3),
            phases=take(self.phases),
            metadata=metadata,
            thermodynamics=sliced_thermo,
        )

    def analysis_view(
        self,
        *,
        include_equilibration: bool = False,
        start: int | None = None,
        stop: int | None = None,
        stride: int = 1,
    ) -> TrajectoryDataset:
        return self.slice_frames(
            self.frame_indices(
                include_equilibration=include_equilibration,
                start=start,
                stop=stop,
                stride=stride,
            )
        )
