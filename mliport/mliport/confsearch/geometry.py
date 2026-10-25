"""Minimum-image geometry helpers for the confsearch core.

All distances in confsearch are minimum-image distances (MIC) unless a rule
explicitly says otherwise (the periodic-image self-check in section 8.4 uses
explicit in-plane translations, and the vacuum check uses the explicit ``+c``
image).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mliport.confsearch.errors import ConfsearchError

if TYPE_CHECKING:
    from ase import Atoms


def mic_vectors(cell: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Fold Cartesian displacement vectors into the minimum-image convention."""
    array = np.asarray(deltas, dtype=float)
    if array.size == 0:
        return array.reshape(array.shape)
    matrix = np.asarray(cell, dtype=float)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= 1e-12:
        raise ConfsearchError(
            "Minimum-image distances need a full-rank cell; got a singular cell"
        )
    fractional = array @ np.linalg.inv(matrix)
    fractional -= np.round(fractional)
    return fractional @ matrix


def mic_distance_matrix(
    atoms: Atoms, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """MIC distances between every ``first`` and every ``second`` atom."""
    left = np.asarray(first, dtype=int)
    right = np.asarray(second, dtype=int)
    if left.size == 0 or right.size == 0:
        return np.zeros((left.size, right.size), dtype=float)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    deltas = positions[right][None, :, :] - positions[left][:, None, :]
    return np.linalg.norm(mic_vectors(atoms.cell, deltas), axis=2)


def minimum_distance(atoms: Atoms, first: np.ndarray, second: np.ndarray) -> float:
    """Smallest MIC distance between two atom-index sets."""
    matrix = mic_distance_matrix(atoms, first, second)
    if matrix.size == 0:
        return float("inf")
    return float(np.min(matrix))


def mic_displacements(atoms: Atoms, deltas: np.ndarray) -> np.ndarray:
    """MIC displacement lengths of Cartesian displacement vectors."""
    folded = mic_vectors(atoms.cell, deltas)
    return np.linalg.norm(folded, axis=1)


def self_image_distance(atoms: Atoms, adsorbate: np.ndarray) -> float:
    """Smallest distance to a periodic in-plane copy of the adsorbate.

    The candidate translations are ``n_a*a + n_b*b`` with
    ``(n_a, n_b) in {-2..2}^2 \\ {(0, 0)}``.  The wider-than-``+-1`` shell keeps
    the guard conservative for skewed in-plane cells; the cell stays fixed, so
    the recorded value is reproducible.
    """
    indices = np.asarray(adsorbate, dtype=int)
    if indices.size == 0:
        return float("inf")
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    best = float("inf")
    shifts = [
        na * cell[0] + nb * cell[1]
        for na in range(-2, 3)
        for nb in range(-2, 3)
        if (na, nb) != (0, 0)
    ]
    for shift in shifts:
        deltas = (positions[indices][None, :, :] + shift) - positions[indices][
            :, None, :
        ]
        if deltas.size == 0:
            continue
        distances = np.linalg.norm(deltas, axis=2)
        best = min(best, float(np.min(distances)))
    return best


def vacuum_gap(atoms: Atoms, adsorbate: np.ndarray, slab: np.ndarray) -> float:
    """Smallest distance from the adsorbate to the slab image shifted by ``+c``."""
    indices = np.asarray(adsorbate, dtype=int)
    slab_indices = np.asarray(slab, dtype=int)
    if indices.size == 0 or slab_indices.size == 0:
        return float("inf")
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    deltas = (positions[slab_indices][None, :, :] + cell[2]) - positions[indices][
        :, None, :
    ]
    return float(np.min(np.linalg.norm(deltas, axis=2)))


def wrap_positions(atoms: Atoms) -> np.ndarray:
    """Positions wrapped into the cell (fractional coordinates in [0, 1))."""
    wrapped = atoms.copy()
    wrapped.wrap()
    return np.asarray(wrapped.get_positions(), dtype=float)


def periodic_gap_and_contiguity(
    slab_z: np.ndarray, cell_height: float
) -> tuple[float, float]:
    """Return ``(periodic_gap, largest_internal_gap)`` for slab z coordinates.

    ``periodic_gap`` is the gap across the ``c`` boundary (``z_min + c - z_max``)
    and ``largest_internal_gap`` the largest gap between consecutive wrapped slab
    atoms.  The slab occupies one interval only when the periodic gap is the
    vacuum, i.e. strictly larger than every internal gap (spec section 5.3).
    """
    if cell_height <= 0:
        raise ConfsearchError("The cell height c_z must be positive")
    values = np.sort(np.asarray(slab_z, dtype=float))
    if values.size == 0:
        raise ConfsearchError("The slab contains no atoms")
    if values.size == 1:
        return float(cell_height), 0.0
    internal = np.diff(values)
    periodic = float(values[0] + cell_height - values[-1])
    return periodic, float(np.max(internal)) if internal.size else 0.0


def slab_interval(slab_z: np.ndarray, cell_height: float) -> tuple[float, float]:
    """Wrapped z interval ``(z_min, z_max)`` occupied by the slab."""
    values = np.sort(np.asarray(slab_z, dtype=float))
    if values.size == 0:
        raise ConfsearchError("The slab contains no atoms")
    return float(values[0]), float(values[-1])


__all__ = [
    "mic_displacements",
    "mic_distance_matrix",
    "mic_vectors",
    "minimum_distance",
    "periodic_gap_and_contiguity",
    "self_image_distance",
    "slab_interval",
    "vacuum_gap",
    "wrap_positions",
]
