"""Lazy GEMDAT adapter for explicit site/jump/pathway mechanism analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mliport.analysis.drift import DRIFT_MODES, drift_semantics
from mliport.analysis.msd import unwrap_positions
from mliport.analysis.transport import (
    _production_positions_with_drift,
    _production_temperature,
    _validate_kinisi_periodic_reconstruction,
)
from mliport.analysis.validation import (
    OptionalDependencyError,
    UnsupportedAnalysisError,
    require_analysis,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from mliport.analysis.dataset import TrajectoryDataset


@dataclass(slots=True)
class GemdatResult:
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]
    tables: dict[str, Any] = field(default_factory=dict)
    structures: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _require_gemdat():
    try:
        from gemdat import Trajectory
        from pymatgen.core import Species, Structure
    except ImportError as exc:  # pragma: no cover - optional environment
        raise OptionalDependencyError(
            "GEMDAT electrolyte analysis is unavailable. Install mliport[electrolyte]."
        ) from exc
    try:
        gemdat_version = version("gemdat")
    except PackageNotFoundError:  # pragma: no cover
        gemdat_version = "unknown"
    if gemdat_version != "unknown" and not gemdat_version.startswith("1."):
        raise OptionalDependencyError(
            f"Unsupported GEMDAT version {gemdat_version}; this adapter is tested "
            "with GEMDAT>=1.7.3,<2."
        )
    return Trajectory, Species, Structure, gemdat_version


GEMDAT_STATUS_SUCCESS = "success"
GEMDAT_STATUS_NO_EVENTS = "no_events"
GEMDAT_STATUS_UNSUPPORTED = "unsupported"
GEMDAT_STATUS_BACKEND_ERROR = "backend_error"
GEMDAT_STATUS_INVALID_INPUT = "invalid_input"

#: Upstream GEMDAT 1.x fails while stacking an *empty* event collection for a
#: trajectory with no transitions.  Only this specific failure pattern may be
#: reinterpreted -- and only when an independent nearest-site check also finds
#: no transitions (see ``gemdat_electrolyte``).
_EMPTY_EVENT_ERROR_PATTERNS = (
    "need at least one array to stack",
    "need at least one array to concatenate",
    "zero-dimensional arrays cannot be concatenated",
)


def _is_empty_event_collection_error(exc: BaseException) -> bool:
    if not isinstance(exc, (ValueError, IndexError, TypeError)):
        return False
    text = str(exc).lower()
    return any(pattern in text for pattern in _EMPTY_EVENT_ERROR_PATTERNS)


def _nearest_site_change_count(
    *,
    view,
    corrected_positions_A,
    mobile,
    sites,
) -> int | None:
    """Independent nearest-site transition count (``None`` = not computable).

    Cross-check for the "valid empty transition collection" claim: a backend
    that reports no events while atoms demonstrably change site is a backend
    error, not a physical zero (task book PR-C).
    """
    try:
        centers = np.asarray([site.coords for site in sites], dtype=float)
        if centers.size == 0:
            return 0
        cell = np.asarray(view.cells[0], dtype=float)
        inverse = np.linalg.inv(cell)
        positions = np.asarray(corrected_positions_A, dtype=float)[:, mobile, :]
        previous = None
        changes = 0
        for frame in positions:
            fractional = frame @ inverse
            delta = fractional[:, None, :] - centers[None, :, :]
            delta -= np.round(delta)
            cartesian = delta @ cell
            nearest = np.argmin(np.linalg.norm(cartesian, axis=-1), axis=1)
            if previous is not None:
                changes += int(np.count_nonzero(nearest != previous))
            previous = nearest
        return changes
    except Exception:  # noqa: BLE001 - cross-check must never mask the verdict
        return None


def _validate_site_geometry(sites, *, cell_A, mobile_species: str) -> None:
    """Reject site geometries that cannot describe this trajectory."""
    if not len(sites):
        raise ValueError(
            f"invalid site geometry: no sites for mobile species {mobile_species!r}"
        )
    cell = np.asarray(cell_A, dtype=float)
    lattice = np.asarray(sites.lattice.matrix, dtype=float)
    if not np.allclose(lattice, cell, rtol=1e-5, atol=1e-6):
        raise ValueError(
            "invalid site geometry: site lattice does not match the trajectory "
            "cell within 1e-5"
        )
    coords = np.asarray(sites.cart_coords, dtype=float)
    if not np.all(np.isfinite(coords)):
        raise ValueError("invalid site geometry: non-finite site coordinates")


def _validate_mechanism_dimensions(
    *, jump_dimensions: int, percolation_axes: str
) -> None:
    if jump_dimensions not in {1, 2, 3}:
        raise ValueError("jump_dimensions must be 1, 2, or 3")
    if (
        not percolation_axes
        or any(axis not in "xyz" for axis in percolation_axes)
        or len(set(percolation_axes)) != len(percolation_axes)
    ):
        raise ValueError("percolation_axes must be a non-empty subset of xyz")


def jump_summary(
    jumps,
    *,
    jump_dimensions: int,
    percolation_axes: str,
) -> dict[str, Any]:
    """Serialize jump metrics while keeping dimensions independent of axes."""

    _validate_mechanism_dimensions(
        jump_dimensions=jump_dimensions, percolation_axes=percolation_axes
    )
    result = {
        "number_of_jumps": int(jumps.n_jumps),
        "jump_dimensions": int(jump_dimensions),
        "percolation_axes": percolation_axes,
    }
    if jumps.n_jumps:
        result["jump_diffusivity_m2_s"] = float(jumps.jump_diffusivity(jump_dimensions))
    return result


def _gemdat_trajectory(
    dataset: TrajectoryDataset,
    *,
    temperature_K: float | None,
    temperature_source: str | None = None,
    positions_cartesian_A: np.ndarray | None = None,
    time_source: str | None = None,
):
    """Build a GEMDAT trajectory with explicit time and position semantics."""
    Trajectory, Species, _, _ = _require_gemdat()
    positions = (
        np.asarray(positions_cartesian_A, dtype=float)
        if positions_cartesian_A is not None
        else np.asarray(dataset.positions, dtype=float)
    )
    if positions.shape != dataset.positions.shape:
        raise ValueError("positions_cartesian_A must match dataset positions shape")
    fractional = np.einsum("fai,fij->faj", positions, np.linalg.inv(dataset.cells))
    frame_interval_ps = float(dataset.frame_interval_fs) / 1000.0
    return Trajectory(
        species=[Species(symbol) for symbol in dataset.symbols],
        coords=np.mod(fractional, 1.0),
        lattice=dataset.cells[0],
        constant_lattice=True,
        time_step=frame_interval_ps * 1.0e-12,
        metadata={
            "temperature": temperature_K,
            "temperature_source": temperature_source,
            "time_step_ps": frame_interval_ps,
            "time_source": time_source
            or "mliport saved-frame interval (frame_interval_fs/1000)",
            "position_convention": dataset.positions_convention,
            "pbc_semantics": "wrapped fractional GEMDAT coordinates; fixed cell",
        },
    )


def _path_arrays(path, lattice) -> dict[str, np.ndarray]:
    voxels = np.asarray(path.sites, dtype=int)
    dimensions = np.asarray(path.dims, dtype=int)
    fractional = (np.mod(voxels, dimensions) + 0.5) / dimensions
    energy = np.asarray(path.energy, dtype=float)
    return {
        "voxel": voxels,
        "fractional": fractional,
        "cartesian_A": np.asarray(lattice.get_cartesian_coords(fractional)),
        "free_energy_eV": energy,
    }


def _empty_path_arrays() -> dict[str, np.ndarray]:
    return {
        "voxel": np.empty((0, 3), dtype=int),
        "fractional": np.empty((0, 3), dtype=float),
        "cartesian_A": np.empty((0, 3), dtype=float),
        "free_energy_eV": np.empty(0, dtype=float),
    }


def _residence_table_with_time_ps(
    table, *, frame_interval_fs: float
) -> tuple[Any, str | None]:
    """Add an explicit physical residence-time field to a GEMDAT 1.x table."""

    if not hasattr(table, "columns") or "time" not in table.columns:
        return table, (
            "GEMDAT residence-time table has no documented 'time' step-count "
            "field; residence-time plotting was disabled."
        )
    output = table.copy()
    try:
        steps = np.asarray(output["time"], dtype=float)
    except (TypeError, ValueError):
        return table, (
            "GEMDAT residence-time step counts are not numeric; residence-time "
            "plotting was disabled."
        )
    if np.any(~np.isfinite(steps)) or np.any(steps < 0):
        return table, (
            "GEMDAT residence-time step counts are invalid; residence-time "
            "plotting was disabled."
        )
    output["residence_time_ps"] = steps * float(frame_interval_fs) / 1000.0
    return output, None


def _jump_table_with_distance_A(table, sites) -> tuple[Any, str | None]:
    """Add minimum-image site-to-site jump distances in angstrom."""

    required = {"start site", "destination site"}
    if not hasattr(table, "columns") or not required.issubset(table.columns):
        return table, (
            "GEMDAT jump table lacks explicit start/destination site indices; "
            "jump-distance plotting was disabled."
        )
    output = table.copy()
    start_raw = np.asarray(output["start site"])
    destination_raw = np.asarray(output["destination site"])
    try:
        start = start_raw.astype(int)
        destination = destination_raw.astype(int)
    except (TypeError, ValueError):
        return table, (
            "GEMDAT jump site indices are not integral; jump-distance plotting "
            "was disabled."
        )
    if not (
        np.array_equal(start_raw, start)
        and np.array_equal(destination_raw, destination)
    ):
        return table, (
            "GEMDAT jump site indices are not integral; jump-distance plotting "
            "was disabled."
        )
    try:
        distances = np.asarray(sites.distance_matrix, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return table, (
            "GEMDAT site geometry has no usable periodic distance matrix; "
            "jump-distance plotting was disabled."
        )
    if (
        distances.ndim != 2
        or distances.shape[0] != distances.shape[1]
        or np.any(start < 0)
        or np.any(destination < 0)
        or np.any(start >= distances.shape[0])
        or np.any(destination >= distances.shape[1])
    ):
        return table, (
            "GEMDAT jump site indices do not match the site distance matrix; "
            "jump-distance plotting was disabled."
        )
    output["jump_distance_A"] = distances[start, destination]
    return output, None


def _jump_rate_table_with_units(table) -> tuple[Any, str | None]:
    """Expose GEMDAT 1.x's documented per-second rate with an explicit unit."""

    if not hasattr(table, "columns") or "rates" not in table.columns:
        return table, (
            "GEMDAT jump-rate table has no documented 'rates' field; jump-rate "
            "plotting was disabled."
        )
    output = table.copy()
    try:
        rates = np.asarray(output["rates"], dtype=float)
    except (TypeError, ValueError):
        return table, (
            "GEMDAT jump rates are not numeric; jump-rate plotting was disabled."
        )
    if np.any(~np.isfinite(rates)) or np.any(rates < 0):
        return table, (
            "GEMDAT jump rates are invalid; jump-rate plotting was disabled."
        )
    output["jump_rate_s^-1"] = rates
    return output, None


def gemdat_electrolyte(
    dataset: TrajectoryDataset,
    *,
    mobile_species: str,
    sites_path: str | Path | None = None,
    discover_sites_from_density: bool = False,
    temperature_K: float | None = None,
    resolution_A: float = 0.5,
    background_level: float = 0.1,
    site_radius_A: float | None = None,
    minimal_residence: int = 0,
    jump_dimensions: int = 3,
    percolation_axes: str = "xyz",
    drift_reference: str = "none",
    drift_indices: Iterable[int] | None = None,
    drift_mode: str = "arithmetic_mean",
) -> GemdatResult:
    """Run GEMDAT site mapping, transitions, jumps, and percolation."""
    if drift_mode not in DRIFT_MODES:
        raise ValueError(f"drift_mode must be one of {DRIFT_MODES}")

    # Mechanism analysis is always defined on production frames. Keep this as
    # the first dataset operation so an equilibration-inclusive caller cannot
    # accidentally leak frames into GEMDAT.
    view = dataset.analysis_view(include_equilibration=False)
    _validate_mechanism_dimensions(
        jump_dimensions=jump_dimensions, percolation_axes=percolation_axes
    )
    report = require_analysis(dataset, "transport")
    if not report.fixed_cell or not report.three_dimensional_pbc:
        raise UnsupportedAnalysisError(
            "GEMDAT mechanism analysis requires a fixed cell and 3-D PBC"
        )
    if resolution_A <= 0:
        raise ValueError("resolution_A must be positive")
    if minimal_residence < 0:
        raise ValueError("minimal_residence must be >= 0")
    explicit_sources = sum(
        (
            sites_path is not None,
            discover_sites_from_density,
        )
    )
    if explicit_sources != 1:
        raise ValueError(
            "Choose exactly one site source: --sites FILE or "
            "--discover-sites-from-density"
        )
    temperature, temperature_source = _production_temperature(view, temperature_K)
    _, unwrap_diagnostics = unwrap_positions(view)
    if (
        view.positions_convention == "wrapped"
        and unwrap_diagnostics["unwrap_safety_ratio"] > 0.8
    ):
        raise UnsupportedAnalysisError(
            "GEMDAT jump analysis refused: wrapped-frame unwrap safety ratio "
            "exceeds 0.8; save frames more frequently or provide exact unwrapped positions."
        )
    position_semantics = _validate_kinisi_periodic_reconstruction(
        view, unwrap_diagnostics
    )
    mobile = view.select(mobile_species)
    corrected_positions, reference = _production_positions_with_drift(
        view,
        mobile=mobile,
        drift_reference=drift_reference,
        drift_indices=drift_indices,
        drift_mode=drift_mode,
    )
    _, _, Structure, gemdat_version = _require_gemdat()
    trajectory = _gemdat_trajectory(
        view,
        temperature_K=temperature,
        temperature_source=temperature_source,
        positions_cartesian_A=corrected_positions,
    )
    mobile_trajectory = trajectory.filter(mobile_species)
    volume = mobile_trajectory.to_volume(resolution=resolution_A)
    discovery_warning: str | None = None
    discovery_error: dict[str, Any] | None = None
    try:
        peaks = volume.find_peaks()
    except Exception as exc:  # noqa: BLE001 - classified below
        peaks = np.empty((0, 3), dtype=int)
        discovery_error = {"type": type(exc).__name__, "message": str(exc)}
        discovery_warning = f"Automatic density peak detection failed: {exc}"
    if sites_path is not None:
        site_path = Path(sites_path).expanduser().resolve()
        if not site_path.is_file():
            raise FileNotFoundError(f"Site structure not found: {site_path}")
        sites = Structure.from_file(str(site_path))
        site_source = str(site_path)
    else:
        try:
            sites = volume.to_structure(
                specie=mobile_species,
                background_level=background_level,
                peaks=peaks,
                return_occupancies=True,
                n_frames=view.nframes,
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            sites = Structure(lattice=view.cells[0], species=[], coords=[])
            discovery_error = {"type": type(exc).__name__, "message": str(exc)}
            discovery_warning = f"Automatic density site segmentation failed: {exc}"
        site_source = "exploratory automatic GEMDAT density peak segmentation"
    free_energy = volume.get_free_energy(temperature)
    result = GemdatResult(
        summary={
            "backend": "GEMDAT",
            "gemdat_version": gemdat_version,
            "mobile_species": mobile_species,
            "temperature_K": temperature,
            "temperature_source": temperature_source,
            "analysis_phase": "production",
            "site_source": site_source,
            "resolution_A": resolution_A,
            "background_level": background_level,
            "site_radius_A": site_radius_A,
            "number_of_sites": int(len(sites)),
            "number_of_density_peaks": int(len(peaks)),
            "jump_dimensions": jump_dimensions,
            "percolation_axes": percolation_axes,
            "frame_interval_fs": float(view.frame_interval_fs),
            "gemdat_time_step_ps": float(view.frame_interval_fs / 1000.0),
            "time_source": "mliport saved-frame interval (frame_interval_fs/1000)",
            "position_convention": view.positions_convention,
            "pbc_semantics": "fixed-cell 3-D PBC; wrapped fractional GEMDAT coordinates",
            "drift_correction": {
                **drift_semantics(
                    drift_mode=drift_mode,
                    drift_reference=drift_reference,
                    reference=reference,
                    symbols=view.symbols,
                ),
                "mode": str(drift_reference).lower(),
                "applied_once": True,
            },
            "unwrap_diagnostics": unwrap_diagnostics,
            "position_semantics": position_semantics,
            "automatic_discovery": {
                "resolution_A": resolution_A,
                "background_level": background_level,
                "peak_count": int(len(peaks)),
                "site_count": int(len(sites)),
                "site_radius_A": site_radius_A,
                "exploratory": bool(discover_sites_from_density),
            },
            "gemdat_tracer_diffusivity_endpoint_promoted": False,
        },
        arrays={
            "density_counts": np.asarray(volume.data),
            "density_probability": np.asarray(volume.probability()),
            "free_energy_eV": np.asarray(free_energy.data),
            "peak_voxels": np.asarray(peaks, dtype=int),
            "cell_A": np.asarray(view.cells[0]),
        },
        structures={"sites": sites, "reference": trajectory.get_structure(0)},
    )
    if discover_sites_from_density:
        # The two structures intentionally make the exploratory pathway
        # explicit in the artifact names. GEMDAT's occupancy annotations are
        # retained in the occupancy structure returned below.
        result.structures["detected_sites"] = sites
        result.structures["occupancy_sites"] = sites
        if discovery_warning:
            result.warnings.append(discovery_warning)
    if not len(sites):
        result.structures["occupancy"] = sites
        if discovery_error is not None:
            # Site discovery *failed*; this is not a physical "no sites" fact.
            result.summary.update(
                {
                    "gemdat_transition_status": GEMDAT_STATUS_BACKEND_ERROR,
                    "gemdat_transition_error": discovery_error,
                    "gemdat_status_reason": (
                        "GEMDAT site discovery failed; transition/jump metrics "
                        "are unknown, not zero"
                    ),
                    "transition_events": None,
                    "number_of_jumps": None,
                    "occupancy_source": "not computed: site discovery failed",
                }
            )
            result.warnings.append(
                "GEMDAT site discovery failed; zero jumps would be fabricated."
            )
            result.summary["diagnostic_crosscheck"] = {
                "publication_transport_authority": False,
                "warning": "GEMDAT metrics unavailable because site discovery failed.",
            }
            return result
        result.summary["gemdat_transition_status"] = GEMDAT_STATUS_NO_EVENTS
        result.summary["gemdat_transition_error"] = None
        result.summary["gemdat_status_reason"] = (
            "no sites were available, so no transition can exist"
        )
        result.summary["occupancy_source"] = "no detected sites"
        result.summary["diagnostic_crosscheck"] = {
            "publication_transport_authority": False,
            "warning": "GEMDAT metrics not computed because site discovery was empty.",
        }
        for axis in percolation_axes:
            result.paths[axis] = _empty_path_arrays()
        result.warnings.append(
            "Automatic density site discovery produced no sites; transition, "
            "residence, jump, and percolation tables were not computed."
        )
        return result
    for axis in percolation_axes:
        if not len(peaks):
            result.warnings.append(
                f"No density peaks were available for {axis}-axis percolation."
            )
            result.paths[axis] = _empty_path_arrays()
            continue
        try:
            path = free_energy.optimal_percolating_path(peaks=peaks, percolate=axis)
        except Exception as exc:
            result.warnings.append(
                f"Percolation failed along {axis}; no path was written: {exc}"
            )
            path = None
        if path is None:
            result.warnings.append(f"No percolating path found along {axis}.")
            result.paths[axis] = _empty_path_arrays()
            continue
        arrays = _path_arrays(path, free_energy.lattice)
        result.paths[axis] = arrays
        energy = arrays["free_energy_eV"]
        result.summary.setdefault("percolation", {})[axis] = {
            "steps": int(len(energy)),
            "free_energy_path_barrier_eV": float(np.max(energy) - np.min(energy)),
            "interpretation": (
                "finite-temperature occupancy-derived free-energy path; not a "
                "NEB potential-energy migration barrier"
            ),
        }

    transitions = None
    transition_error: dict[str, Any] | None = None
    transition_status = GEMDAT_STATUS_SUCCESS
    status_reason: str | None = None
    try:
        _validate_site_geometry(
            sites,
            cell_A=view.cells[0],
            mobile_species=mobile_species,
        )
        transitions = trajectory.transitions_between_sites(
            sites, mobile_species, site_radius=site_radius_A
        )
    except Exception as exc:  # noqa: BLE001 - classified below, never zeroed
        transition_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if _is_empty_event_collection_error(exc):
            changes = _nearest_site_change_count(
                view=view,
                corrected_positions_A=corrected_positions,
                mobile=mobile,
                sites=sites,
            )
            if changes == 0:
                transition_status = GEMDAT_STATUS_NO_EVENTS
                status_reason = (
                    "GEMDAT reported a valid empty transition collection and an "
                    "independent nearest-site check found no site changes"
                )
                result.warnings.append(
                    "GEMDAT transition detection reported no events for this "
                    "trajectory; zero jumps is a physical result here."
                )
            else:
                transition_status = GEMDAT_STATUS_BACKEND_ERROR
                status_reason = (
                    "GEMDAT reported an empty transition collection, but an "
                    "independent nearest-site check found site changes; refusing "
                    "to report zero jumps"
                )
        elif isinstance(exc, ValueError) and "invalid site geometry" in str(exc):
            transition_status = GEMDAT_STATUS_INVALID_INPUT
            status_reason = "site geometry is not compatible with the trajectory"
        else:
            transition_status = GEMDAT_STATUS_BACKEND_ERROR
            status_reason = (
                "GEMDAT transition detection failed; jump/transition metrics are "
                "reported as unknown rather than zero"
            )
        result.warnings.append(
            f"GEMDAT transitions {transition_status} ({type(exc).__name__}): {exc}"
        )

    result.summary["gemdat_transition_status"] = transition_status
    result.summary["gemdat_transition_error"] = transition_error
    result.summary["gemdat_status_reason"] = status_reason
    if transitions is None and transition_status == GEMDAT_STATUS_NO_EVENTS:
        result.summary.update(
            {
                "transition_events": 0,
                "occupancy_by_site_type": {},
                "atom_locations": {},
                "number_of_jumps": 0,
                "jump_dimensions": jump_dimensions,
                "percolation_axes": percolation_axes,
                "occupancy_source": (
                    "explicit site geometry; no transitions were observed"
                ),
            }
        )
        result.structures["occupancy"] = sites
        if discover_sites_from_density:
            result.structures["occupancy_sites"] = sites
        result.arrays["transition_matrix"] = np.zeros(
            (len(sites), len(sites)), dtype=int
        )
        result.arrays["jump_matrix"] = np.zeros((len(sites), len(sites)), dtype=int)
        result.tables.update(
            {
                "transition_events": np.empty((0, 0)),
                "residence_times": np.empty((0, 0)),
                "jumps": np.empty((0, 0)),
                "jump_rates": np.empty((0, 0)),
            }
        )
    elif transitions is None:
        # backend_error / invalid_input: never publish physical zero matrices.
        result.summary.update(
            {
                "transition_events": None,
                "number_of_jumps": None,
                "occupancy_by_site_type": {},
                "atom_locations": {},
                "jump_dimensions": jump_dimensions,
                "percolation_axes": percolation_axes,
                "occupancy_source": (
                    "not computed: GEMDAT transition detection did not succeed"
                ),
            }
        )
    else:
        result.structures["occupancy"] = transitions.occupancy()
        result.summary["occupancy_source"] = "GEMDAT transition occupancy"
        if discover_sites_from_density:
            result.structures["occupancy_sites"] = result.structures["occupancy"]
        result.tables["transition_events"] = transitions.events
        residence_table, residence_warning = _residence_table_with_time_ps(
            transitions.residence_time(), frame_interval_fs=view.frame_interval_fs
        )
        result.tables["residence_times"] = residence_table
        if residence_warning:
            result.warnings.append(residence_warning)
        result.arrays["transition_matrix"] = transitions.matrix()
        result.summary["transition_events"] = int(transitions.n_events)
        result.summary["occupancy_by_site_type"] = transitions.occupancy_by_site_type()
        result.summary["atom_locations"] = transitions.atom_locations()

        jumps = transitions.jumps(minimal_residence=minimal_residence)
        jump_table, jump_distance_warning = _jump_table_with_distance_A(
            jumps.data, sites
        )
        result.tables["jumps"] = jump_table
        if jump_distance_warning:
            result.warnings.append(jump_distance_warning)
        result.arrays["jump_matrix"] = jumps.matrix()
        result.summary.update(
            jump_summary(
                jumps,
                jump_dimensions=jump_dimensions,
                percolation_axes=percolation_axes,
            )
        )
    if transitions is not None and jumps.n_jumps:
        n_parts = min(10, max(2, view.nframes // 4))
        if n_parts <= view.nframes:
            try:
                rates = jumps.rates(n_parts=n_parts)
            except Exception as exc:  # noqa: BLE001 - GEMDAT backend failures vary
                result.tables["jump_rates"] = np.empty((0, 0))
                result.warnings.append(
                    "GEMDAT jump-rate segmentation failed; jump-rate plotting "
                    f"was disabled: {exc}"
                )
            else:
                rate_table, rate_warning = _jump_rate_table_with_units(rates)
                result.tables["jump_rates"] = rate_table
                if rate_warning:
                    result.warnings.append(rate_warning)
        collective = jumps.collective()
        result.summary["solo_jump_fraction"] = float(jumps.solo_fraction)
        result.summary["collective_jump_count"] = int(collective.n_coll_jumps)
    # GEMDAT trajectory metrics are deliberately namespaced as a diagnostic
    # cross-check. They never replace the kinisi transport authority.
    crosscheck: dict[str, Any] = {
        "publication_transport_authority": False,
        "warning": (
            "GEMDAT endpoint/COM diffusivities and Haven ratio are diagnostic "
            "cross-checks only; kinisi D_tracer and sigma_collective remain authoritative."
        ),
    }
    try:
        metrics = trajectory.metrics()
        for name, call in (
            (
                "tracer_diffusivity",
                lambda: metrics.tracer_diffusivity(dimensions=jump_dimensions),
            ),
            (
                "tracer_diffusivity_center_of_mass",
                lambda: metrics.tracer_diffusivity_center_of_mass(
                    dimensions=jump_dimensions
                ),
            ),
            ("haven_ratio", lambda: metrics.haven_ratio(dimensions=jump_dimensions)),
        ):
            try:
                value = call()
                crosscheck[name] = float(value)
            except Exception as exc:  # pragma: no cover - backend-dependent
                crosscheck[f"{name}_warning"] = str(exc)
    except Exception as exc:  # pragma: no cover - backend-dependent
        crosscheck["warning"] += f" Metrics unavailable: {exc}"
    result.summary["diagnostic_crosscheck"] = crosscheck
    return result
