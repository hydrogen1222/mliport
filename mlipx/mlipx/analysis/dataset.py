"""Canonical trajectory data contract for Analysis v2.

This module only reads and normalizes trajectory data.  It deliberately does
not calculate diffusion or any other derived scientific observable.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.stress import full_3x3_to_voigt_6_stress

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from ase import Atoms

PositionsConvention = Literal["wrapped", "unwrapped", "unknown"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _discover_source(source: str | Path) -> tuple[Path, Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.parent.name in {"raw", "vasp"}:
            return path.parent.parent, path
        return path.parent, path
    if not path.is_dir():
        raise FileNotFoundError(f"Trajectory or run directory not found: {path}")
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
        "trajectory.traj, vasp/XDATCAR, or XDATCAR under " + str(path)
    )


def _load_frames(path: Path) -> list[Atoms]:
    if path.suffix.lower() == ".traj":
        with Trajectory(path, mode="r") as trajectory:
            return list(trajectory)
    frames = read(path, index=":")
    return frames if isinstance(frames, list) else [frames]


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


def _all_frame_info(frames: list[Atoms], key: str) -> list[Any] | None:
    values = [atoms.info.get(key) for atoms in frames]
    if any(value is None for value in values):
        return None
    return values


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

        frame_list = list(frames)
        if not frame_list:
            raise ValueError("Trajectory contains no frames")
        source_path = Path(source)
        return cls._from_loaded_frames(
            frame_list,
            run_dir=source_path.parent,
            source_path=source_path,
            times_fs=None if times_fs is None else np.asarray(list(times_fs), float),
            steps=None if steps is None else np.asarray(list(steps), int),
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
        """Load an mlipx run, ASE trajectory, or XDATCAR.

        Explicit function arguments override metadata only after consistency
        checks.  A missing time axis is never fabricated from a median frame
        difference.
        """

        run_dir, source_path = _discover_source(source)
        frames = _load_frames(source_path)
        if not frames:
            raise ValueError(f"Trajectory contains no frames: {source_path}")

        artifacts = _read_json(run_dir / "artifacts.json")
        resolved = _read_json(run_dir / "resolved_config.json")
        trajectory_meta = artifacts.get("trajectory", {})
        run_options = resolved.get("run_options", {})
        thermo = _read_thermodynamics(run_dir)
        nframes = len(frames)

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
            info_times = _all_frame_info(frames, "mlipx_time_fs")
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
            values = np.asarray(thermo["step"], dtype=float)
            if np.all(np.isfinite(values)):
                steps = values.astype(int)
        if steps is None:
            info_steps = _all_frame_info(frames, "mlipx_step")
            if info_steps is not None:
                steps = np.asarray(info_steps, dtype=int)
        if steps is None and stride is not None:
            steps = np.arange(nframes, dtype=int) * stride

        phases: np.ndarray | None = None
        if "phase" in thermo and len(thermo["phase"]) == nframes:
            phases = np.asarray(thermo["phase"], dtype="U16")
        if phases is None:
            info_phases = _all_frame_info(frames, "mlipx_phase")
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
                "Explicit positions_convention conflicts with the mlipx "
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
        metadata = {
            "artifacts": artifacts,
            "resolved_config": resolved,
            "run_status": artifacts.get("status", "external_or_unknown"),
            "time_source": time_source,
            "target_temperature_K": target_temperature,
            "source_format": "ase-traj"
            if source_path.suffix.lower() == ".traj"
            else "xdatcar",
        }
        return cls._from_loaded_frames(
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
    def _from_loaded_frames(
        cls,
        frames: list[Atoms],
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
        symbols = tuple(frames[0].get_chemical_symbols())
        masses = np.asarray(frames[0].get_masses(), dtype=float)
        pbc = np.asarray(frames[0].pbc, dtype=bool)
        natoms = len(symbols)
        positions = np.empty((len(frames), natoms, 3), dtype=float)
        cells = np.empty((len(frames), 3, 3), dtype=float)
        velocity_rows: list[np.ndarray] = []
        kinetic_rows: list[float] = []
        temperature_rows: list[float] = []
        energy_rows: list[float] = []
        stress_rows: list[np.ndarray] = []
        has_velocities = True
        has_energies = True
        has_stresses = True
        for index, atoms in enumerate(frames):
            if len(atoms) != natoms:
                raise ValueError(f"Frame {index} has a different atom count")
            if tuple(atoms.get_chemical_symbols()) != symbols:
                raise ValueError(f"Frame {index} changes atom identity/order")
            if not np.array_equal(np.asarray(atoms.pbc, bool), pbc):
                raise ValueError(f"Frame {index} changes PBC flags")
            positions[index] = np.asarray(atoms.positions, dtype=float)
            cells[index] = np.asarray(atoms.cell.array, dtype=float)
            if atoms.has("momenta"):
                # ASE's internal time unit is not fs. Keep the analysis
                # contract explicit instead of labelling raw ASE values as
                # A/fs and silently scaling VACF-derived quantities.
                velocity_rows.append(
                    np.asarray(atoms.get_velocities(), dtype=float) * units.fs
                )
                kinetic_rows.append(float(atoms.get_kinetic_energy()))
                temperature_rows.append(float(atoms.get_temperature()))
            else:
                has_velocities = False
            results = getattr(atoms.calc, "results", {}) if atoms.calc else {}
            energy = results.get("energy", results.get("free_energy"))
            if energy is None:
                has_energies = False
            else:
                energy_rows.append(float(energy))
            if "stress" not in results:
                has_stresses = False
            else:
                stress = np.asarray(results["stress"], dtype=float)
                if stress.shape == (3, 3):
                    stress = full_3x3_to_voigt_6_stress(stress)
                if stress.shape != (6,):
                    has_stresses = False
                else:
                    stress_rows.append(stress)

        def thermo_column(*names: str) -> np.ndarray | None:
            for name in names:
                value = thermodynamics.get(name)
                if value is not None and len(value) == len(frames):
                    return np.asarray(value, dtype=float)
            return None

        potential = thermo_column("potential_energy_eV")
        if potential is None and has_energies:
            potential = np.asarray(energy_rows, dtype=float)
        kinetic = thermo_column("kinetic_energy_eV")
        if kinetic is None and has_velocities:
            kinetic = np.asarray(kinetic_rows, dtype=float)
        total = thermo_column("total_energy_eV")
        temperature = thermo_column("temperature_K")
        if temperature is None and has_velocities:
            temperature = np.asarray(temperature_rows, dtype=float)
        if total is None and potential is not None and kinetic is not None:
            total = potential + kinetic
        volume = thermo_column("volume_A3")
        if volume is None:
            volume = np.asarray([abs(np.linalg.det(cell)) for cell in cells])
        pressure = thermo_column("total_pressure_GPa", "pressure_GPa")
        stress = np.asarray(stress_rows) if has_stresses else None
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
            velocities=np.asarray(velocity_rows) if has_velocities else None,
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
