"""Validation and fail-closed eligibility rules for Analysis v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from mlipx.analysis.dataset import TrajectoryDataset


class AnalysisError(RuntimeError):
    """Base class for Analysis v2 errors."""


class InvalidTrajectoryError(AnalysisError):
    """The trajectory violates a required data contract."""


class UnsupportedAnalysisError(AnalysisError):
    """The requested analysis has no unambiguous supported definition."""


class InsufficientTrajectoryInformationError(UnsupportedAnalysisError):
    """Saved frames cannot uniquely reconstruct continuous displacements.

    The wrapped interval is too coarse (or the corrected path crosses aliases)
    to recover the image history; transport must refuse instead of fabricating
    an unwrap (task book PR-E section 7.4).
    """

    status = "insufficient_trajectory_information"


class OptionalDependencyError(AnalysisError):
    """An explicitly requested optional backend is unavailable."""


@dataclass(slots=True)
class TimeAxisReport:
    available: bool
    finite: bool
    strictly_increasing: bool
    uniform: bool
    frame_interval_fs: float | None
    maximum_interval_deviation_fs: float | None


@dataclass(slots=True)
class ValidationReport:
    n_frames: int
    n_atoms: int
    duration_fs: float | None
    duration_ps: float | None
    pbc: list[bool]
    three_dimensional_pbc: bool
    cell_rank_three: bool
    fixed_cell: bool
    finite_positions: bool
    finite_cells: bool
    velocities_stored: bool
    positions_convention: str
    run_status: str
    production_start_frame: int | None
    production_frames: int
    equilibration_frames: int
    md_timestep_fs: float | None
    frame_stride_steps: int | None
    time: TimeAxisReport
    eligible_for_rdf: bool
    eligible_for_msd: bool
    eligible_for_transport: bool
    eligible_for_vacf: bool
    nyquist_THz: float | None
    nyquist_cm_1: float | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_time_axis(dataset: TrajectoryDataset) -> TimeAxisReport:
    times = dataset.times_fs
    if times is None or len(times) < 2:
        return TimeAxisReport(
            available=times is not None,
            finite=bool(times is not None and np.all(np.isfinite(times))),
            strictly_increasing=False,
            uniform=False,
            frame_interval_fs=None,
            maximum_interval_deviation_fs=None,
        )
    differences = np.diff(np.asarray(times, dtype=float))
    finite = bool(np.all(np.isfinite(differences)))
    increasing = bool(finite and np.all(differences > 0))
    if not increasing:
        return TimeAxisReport(
            available=True,
            finite=finite,
            strictly_increasing=False,
            uniform=False,
            frame_interval_fs=None,
            maximum_interval_deviation_fs=None,
        )
    reference = float(differences[0])
    tolerance = max(1.0e-10, abs(reference) * 1.0e-10)
    uniform = bool(np.allclose(differences, reference, rtol=1.0e-6, atol=tolerance))
    deviation = float(np.max(np.abs(differences - reference)))
    return TimeAxisReport(
        available=True,
        finite=True,
        strictly_increasing=True,
        uniform=uniform,
        frame_interval_fs=reference if uniform else None,
        maximum_interval_deviation_fs=deviation,
    )


def validate_trajectory(dataset: TrajectoryDataset) -> ValidationReport:
    time = validate_time_axis(dataset)
    finite_positions = bool(np.all(np.isfinite(dataset.positions)))
    finite_cells = bool(np.all(np.isfinite(dataset.cells)))
    ranks = [np.linalg.matrix_rank(cell) for cell in dataset.cells]
    rank_three = bool(all(rank == 3 for rank in ranks))
    fixed_cell = bool(
        finite_cells
        and np.allclose(
            dataset.cells,
            dataset.cells[0],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    )
    pbc3 = bool(dataset.pbc.all())
    phases = (
        dataset.phases
        if dataset.phases is not None
        else np.full(dataset.nframes, "production", dtype="U16")
    )
    production_indices = np.flatnonzero(phases == "production")
    equilibration_indices = np.flatnonzero(phases == "equilibration")
    production_start = int(production_indices[0]) if len(production_indices) else None
    status = str(dataset.metadata.get("run_status", "external_or_unknown"))
    complete = status not in {"aborted", "failed", "cancelled"}
    known_positions = dataset.positions_convention in {"wrapped", "unwrapped"}
    errors: list[str] = []
    warnings = list(dataset.warnings)
    if not finite_positions:
        errors.append("Trajectory positions contain NaN or Inf")
    if not finite_cells:
        errors.append("Trajectory cells contain NaN or Inf")
    if not rank_three:
        errors.append("At least one cell is singular")
    if not time.available:
        errors.append("Trajectory has no explicit or declared time axis")
    elif not time.finite:
        errors.append("Trajectory time intervals contain NaN or Inf")
    elif not time.strictly_increasing:
        errors.append("Trajectory time axis is not strictly increasing")
    elif not time.uniform:
        warnings.append(
            "Trajectory sampling is nonuniform; FFT MSD, VACF, spectrum, and "
            "kinisi transport are unavailable."
        )
    if not complete:
        warnings.append(f"Source mlipx run status is {status!r}, not completed")
    if not fixed_cell:
        warnings.append(
            "Cell changes across frames; Analysis v2 refuses MSD/transport "
            "because affine lattice deformation is not separated from migration."
        )
    if not pbc3:
        warnings.append("Bulk RDF and transport require three-dimensional PBC")
    if not known_positions:
        warnings.append("Wrapped/unwrapped position convention is unknown")

    base = finite_positions and finite_cells and rank_three
    enough = dataset.nframes >= 4 and len(production_indices) >= 4
    eligible_msd = bool(
        base
        and enough
        and complete
        and pbc3
        and fixed_cell
        and time.available
        and time.uniform
        and known_positions
    )
    eligible_rdf = bool(base and pbc3 and dataset.nframes >= 1)
    eligible_vacf = bool(
        base
        and dataset.nframes >= 4
        and dataset.velocities is not None
        and time.available
        and time.uniform
    )
    duration_fs = (
        float(dataset.times_fs[-1] - dataset.times_fs[0])
        if dataset.times_fs is not None and dataset.nframes >= 2
        else None
    )
    interval_fs = time.frame_interval_fs
    nyquist_thz = 500.0 / interval_fs if interval_fs is not None else None
    nyquist_cm = nyquist_thz * 33.35640951981521 if nyquist_thz else None
    return ValidationReport(
        n_frames=dataset.nframes,
        n_atoms=dataset.natoms,
        duration_fs=duration_fs,
        duration_ps=duration_fs / 1000.0 if duration_fs is not None else None,
        pbc=[bool(value) for value in dataset.pbc],
        three_dimensional_pbc=pbc3,
        cell_rank_three=rank_three,
        fixed_cell=fixed_cell,
        finite_positions=finite_positions,
        finite_cells=finite_cells,
        velocities_stored=dataset.velocities is not None,
        positions_convention=dataset.positions_convention,
        run_status=status,
        production_start_frame=production_start,
        production_frames=int(len(production_indices)),
        equilibration_frames=int(len(equilibration_indices)),
        md_timestep_fs=dataset.md_timestep_fs,
        frame_stride_steps=dataset.frame_stride_steps,
        time=time,
        eligible_for_rdf=eligible_rdf,
        eligible_for_msd=eligible_msd,
        eligible_for_transport=eligible_msd,
        eligible_for_vacf=eligible_vacf,
        nyquist_THz=nyquist_thz,
        nyquist_cm_1=nyquist_cm,
        errors=errors,
        warnings=warnings,
    )


def require_analysis(dataset: TrajectoryDataset, task: str) -> ValidationReport:
    """Validate a task and raise a precise fail-closed exception."""

    report = validate_trajectory(dataset)
    task = task.lower()
    eligibility = {
        "rdf": report.eligible_for_rdf,
        "msd": report.eligible_for_msd,
        "transport": report.eligible_for_transport,
        "vacf": report.eligible_for_vacf,
        "spectrum": report.eligible_for_vacf,
    }
    if task not in eligibility:
        return report
    if eligibility[task]:
        return report
    reasons = list(report.errors)
    if task in {"msd", "transport"}:
        if dataset.nframes < 4 or report.production_frames < 4:
            reasons.append("At least four production frames are required")
        if not report.three_dimensional_pbc:
            reasons.append("Transport requires 3-D PBC")
        if not report.fixed_cell:
            reasons.append("Variable-cell transport is unsupported")
        if not report.time.uniform:
            reasons.append("Transport requires a uniform time axis")
        if dataset.positions_convention == "unknown":
            reasons.append("Position convention must be declared")
        if report.run_status in {"aborted", "failed", "cancelled"}:
            reasons.append(f"Source run status is {report.run_status}")
    if task in {"vacf", "spectrum"}:
        if dataset.velocities is None:
            reasons.append("VACF requires stored velocities")
        if not report.time.uniform:
            reasons.append("VACF requires a uniform time axis")
        if dataset.nframes < 4:
            reasons.append("VACF requires at least four frames")
    if task == "rdf" and not report.three_dimensional_pbc:
        reasons.append("Bulk RDF normalization requires 3-D PBC")
    unique = list(dict.fromkeys(reasons))
    raise InvalidTrajectoryError(
        f"{task} analysis is not eligible: " + "; ".join(unique)
    )
