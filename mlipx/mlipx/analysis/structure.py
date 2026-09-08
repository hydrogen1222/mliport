"""Validated structural trajectory analyses for bulk solid electrolytes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.geometry import find_mic

from mlipx.analysis.validation import UnsupportedAnalysisError, require_analysis

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from mlipx.analysis.dataset import TrajectoryDataset


def cell_heights_A(cell: np.ndarray) -> np.ndarray:
    """Return the perpendicular heights of a full-rank triclinic cell."""

    cell = np.asarray(cell, dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    face_areas = np.asarray(
        [
            np.linalg.norm(np.cross(cell[1], cell[2])),
            np.linalg.norm(np.cross(cell[2], cell[0])),
            np.linalg.norm(np.cross(cell[0], cell[1])),
        ]
    )
    if volume <= 0 or np.any(face_areas <= 0):
        raise ValueError("Cell must have non-zero volume")
    return volume / face_areas


def safe_r_max_A(cells: np.ndarray) -> float:
    """Conservative bulk RDF radius based on triclinic face heights."""

    return 0.5 * min(float(np.min(cell_heights_A(cell))) for cell in cells)


def radial_distribution(
    dataset: TrajectoryDataset,
    *,
    center_species: str,
    neighbor_species: str,
    r_max_A: float | None = None,
    bins: int = 200,
    cn_cutoff_A: float | None = None,
    include_equilibration: bool = False,
    start: int | None = None,
    stop: int | None = None,
    stride: int = 1,
) -> dict[str, Any]:
    """Compute an ordered-center A-B RDF and direct coordination curve."""

    require_analysis(dataset, "rdf")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    view = dataset.analysis_view(
        include_equilibration=include_equilibration,
        start=start,
        stop=stop,
        stride=stride,
    )
    centers = view.select(center_species)
    neighbors = view.select(neighbor_species)
    same = center_species == neighbor_species
    if same and len(centers) < 2:
        raise ValueError("Same-species RDF requires at least two selected atoms")
    safe = safe_r_max_A(view.cells)
    r_max = safe if r_max_A is None else float(r_max_A)
    if r_max <= 0:
        raise ValueError("r_max_A must be positive")
    if r_max > safe + max(1.0e-10, safe * 1.0e-10):
        raise UnsupportedAnalysisError(
            f"r_max_A={r_max:g} exceeds the triclinic minimum-image limit "
            f"{safe:g} A"
        )
    if cn_cutoff_A is not None and not (0 < cn_cutoff_A <= r_max):
        raise ValueError("cn_cutoff_A must lie in (0, r_max_A]")

    edges = np.linspace(0.0, r_max, bins + 1)
    shell_volumes = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    counts = np.zeros(bins, dtype=float)
    normalization = np.zeros(bins, dtype=float)
    cutoff_counts = 0.0
    for positions, cell in zip(view.positions, view.cells, strict=True):
        delta = positions[neighbors][None, :, :] - positions[centers][:, None, :]
        if same:
            delta = delta[~np.eye(len(centers), dtype=bool)]
        delta = delta.reshape(-1, 3)
        _, distances = find_mic(delta, cell=cell, pbc=True)
        counts += np.histogram(distances, bins=edges)[0]
        effective_neighbors = len(neighbors) - (1 if same else 0)
        density_B_A3 = effective_neighbors / abs(float(np.linalg.det(cell)))
        normalization += len(centers) * density_B_A3 * shell_volumes
        if cn_cutoff_A is not None:
            cutoff_counts += float(np.count_nonzero(distances <= cn_cutoff_A))
    g_r = np.divide(
        counts,
        normalization,
        out=np.zeros_like(counts),
        where=normalization > 0,
    )
    coordination = np.cumsum(counts) / (view.nframes * len(centers))
    result: dict[str, Any] = {
        "r_A": 0.5 * (edges[1:] + edges[:-1]),
        "bin_edges_A": edges,
        "g_center_neighbor": g_r,
        "coordination_number_center_neighbor": coordination,
        "ordered_neighbor_counts": counts,
        "center_species": center_species,
        "neighbor_species": neighbor_species,
        "number_of_centers": int(len(centers)),
        "number_of_neighbors": int(len(neighbors)),
        "frames": view.nframes,
        "r_max_safe_A": safe,
        "r_max_A": r_max,
        "normalization_definition": (
            "ordered A centers; rho_B=N_B/V for A!=B and "
            "rho_B=(N_A-1)/V with self exclusion for A==B"
        ),
    }
    if cn_cutoff_A is not None:
        result["coordination_cutoff_A"] = float(cn_cutoff_A)
        result["coordination_number_at_cutoff"] = float(
            cutoff_counts / (view.nframes * len(centers))
        )
    return result


def periodic_rmsd_rmsf(
    dataset: TrajectoryDataset,
    *,
    species: str | None = None,
    indices: Iterable[int] | None = None,
    drift_reference: str = "none",
    drift_indices: Iterable[int] | None = None,
    include_equilibration: bool = False,
    start: int | None = None,
    stop: int | None = None,
) -> dict[str, Any]:
    """PBC displacement RMSD/RMSF without rotational alignment."""

    from mlipx.analysis.msd import displacement_trajectory  # noqa: PLC0415

    view = dataset.analysis_view(
        include_equilibration=include_equilibration, start=start, stop=stop
    )
    selected = view.select(species, indices=indices)
    prepared = displacement_trajectory(
        view,
        mobile_indices=selected,
        drift_reference=drift_reference,
        drift_indices=drift_indices,
    )
    displacement = prepared["mobile_displacements_A"]
    rmsd = np.sqrt(np.mean(np.sum(displacement**2, axis=2), axis=1))
    fluctuations = displacement - np.mean(displacement, axis=0, keepdims=True)
    rmsf = np.sqrt(np.mean(np.sum(fluctuations**2, axis=2), axis=0))
    return {
        "time_ps": (view.times_fs - view.times_fs[0]) / 1000.0,
        "periodic_displacement_rmsd_A": rmsd,
        "periodic_displacement_rmsf_A": rmsf,
        "atom_indices": selected,
        "rotational_alignment": False,
        "drift_correction": prepared["drift_correction"],
        "unwrap_diagnostics": prepared["unwrap_diagnostics"],
    }


def _periodic_gaussian_smooth(
    array: np.ndarray, cell: np.ndarray, sigma_A: float
) -> np.ndarray:
    """Apply an isotropic Cartesian Gaussian on a periodic triclinic grid."""

    if sigma_A <= 0:
        raise ValueError("smoothing_sigma_A must be positive")
    shape = np.asarray(array.shape, dtype=int)
    modes = [np.fft.fftfreq(int(size)) * int(size) for size in shape]
    h_mode, k_mode, l_mode = np.meshgrid(*modes, indexing="ij")
    # Cells are stored as row vectors: r = fractional @ cell. Therefore a
    # reciprocal row wavevector for integer fractional mode h is
    # h @ inv(cell).T, not h @ inv(cell). The distinction matters for
    # triclinic cells and preserves rotational invariance of the smoothing.
    reciprocal_cycles_A = np.linalg.inv(cell).T
    coefficients = np.stack((h_mode, k_mode, l_mode), axis=-1)
    wavevectors = np.einsum("...i,ij->...j", coefficients, reciprocal_cycles_A)
    magnitude_squared = np.sum(wavevectors**2, axis=-1)
    kernel = np.exp(-2.0 * np.pi**2 * sigma_A**2 * magnitude_squared)
    smoothed = np.fft.ifftn(np.fft.fftn(array) * kernel).real
    # FFT roundoff can move the sum by a few ulps. Restore it explicitly.
    original_sum = float(np.sum(array))
    current_sum = float(np.sum(smoothed))
    if current_sum != 0:
        smoothed *= original_sum / current_sum
    return smoothed


def density_map(
    dataset: TrajectoryDataset,
    *,
    mobile_species: str,
    spacing_A: float = 0.25,
    smoothing_sigma_A: float | None = None,
    include_equilibration: bool = False,
    start: int | None = None,
    stop: int | None = None,
    stride: int = 1,
) -> dict[str, Any]:
    """Accumulate periodic occupancy probability and number density grids."""

    report = require_analysis(dataset, "rdf")
    if not report.fixed_cell:
        raise UnsupportedAnalysisError(
            "Density-map number normalization currently requires a fixed cell"
        )
    if spacing_A <= 0:
        raise ValueError("spacing_A must be positive")
    view = dataset.analysis_view(
        include_equilibration=include_equilibration,
        start=start,
        stop=stop,
        stride=stride,
    )
    mobile = view.select(mobile_species)
    cell = view.cells[0]
    lengths = np.linalg.norm(cell, axis=1)
    grid_shape = np.maximum(2, np.ceil(lengths / spacing_A).astype(int))
    counts = np.zeros(tuple(int(value) for value in grid_shape), dtype=float)
    inverse = np.linalg.inv(cell)
    for positions in view.positions:
        fractional = np.mod(positions[mobile] @ inverse, 1.0)
        counts += np.histogramdd(
            fractional,
            bins=tuple(int(value) for value in grid_shape),
            range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )[0]
    if smoothing_sigma_A is not None:
        counts = _periodic_gaussian_smooth(counts, cell, smoothing_sigma_A)
    total_samples = view.nframes * len(mobile)
    occupancy_probability = counts / total_samples
    volume_A3 = abs(float(np.linalg.det(cell)))
    voxel_volume_A3 = volume_A3 / int(np.prod(grid_shape))
    number_density_A3 = counts / (view.nframes * voxel_volume_A3)
    return {
        "occupancy_probability": occupancy_probability,
        "number_density_A^-3": number_density_A3,
        "counts_per_voxel": counts,
        "cell_A": cell,
        "grid_shape": grid_shape,
        "voxel_volume_A3": voxel_volume_A3,
        "actual_grid_vector_spacing_A": lengths / grid_shape,
        "mobile_species": mobile_species,
        "number_of_mobile_particles": int(len(mobile)),
        "frames": view.nframes,
        "smoothing_sigma_A": smoothing_sigma_A,
        "normalization": {
            "sum_occupancy_probability": float(np.sum(occupancy_probability)),
            "integral_number_density": float(
                np.sum(number_density_A3) * voxel_volume_A3
            ),
        },
    }
