"""Automatic molecule placement above a slab (spec section 5.2).

One start is produced by: pick a conformer uniformly at random, centre it at its
(unweighted) centroid, apply a uniform random rotation, draw fractional in-plane
coordinates, and solve the height by bisection so that an element-pair-aware
covalent-radii ratio reaches a target.

Distances are compared as ``q_ij = d_ij / (r_cov,i + r_cov,j)`` with minimum-image
distances, because a single Angstrom offset does not fit every pair (C-H ~ 1.1 A,
S-S ~ 2.05 A).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from mliport.confsearch.errors import ConfsearchPreparationError
from mliport.confsearch.geometry import (
    mic_distance_matrix,
    self_image_distance,
    vacuum_gap,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mliport.confsearch.options import ConfsearchOptions

#: Bisection tolerance on the z offset, in Angstrom (spec section 5.2 step 5).
Z_OFFSET_TOLERANCE_A = 1e-4
#: Initial bracket step and hard bounds for the height search, in Angstrom.
_BRACKET_STEP_A = 0.25
_MAX_SEARCH_A = 50.0
#: Bounded redraws of the rotation in anchored placement.
_MAX_ROTATION_REDRAWS = 10_000

#: Rejection reason keys, so the failure report is stable.
REJECTION_REASONS = (
    "contact_scale",
    "other_min_scale",
    "anchor_targets",
    "anchor_never_lowest",
    "self_image",
    "vacuum",
    "surface_geometry",
    "height_not_solvable",
)


@dataclass(frozen=True, slots=True)
class PlacementResult:
    """One placed start, with everything spec section 5.2 step 7 records."""

    atoms: Atoms
    kind: str
    conformer_index: int
    anchor_index: int | None
    quaternion: tuple[float, float, float, float]
    u: float
    v: float
    z_offset: float
    attempts: int
    rejections: dict[str, int] = field(default_factory=dict)
    min_pair_q: float = float("nan")
    min_pair_distance_A: float = float("nan")
    anchor_q: float | None = None
    anchor_slab_index: int | None = None
    anchor_symbol: str | None = None
    min_self_image_distance_A: float = float("nan")
    min_vacuum_gap_A: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "conformer_index": int(self.conformer_index),
            "anchor_index": (
                None if self.anchor_index is None else int(self.anchor_index)
            ),
            "quaternion": [float(value) for value in self.quaternion],
            "u": float(self.u),
            "v": float(self.v),
            "z_offset_A": float(self.z_offset),
            "attempts": int(self.attempts),
            "rejections": dict(self.rejections),
            "min_pair_q": float(self.min_pair_q),
            "min_pair_distance_A": float(self.min_pair_distance_A),
            "anchor_q": None if self.anchor_q is None else float(self.anchor_q),
            "anchor_slab_index": (
                None if self.anchor_slab_index is None else int(self.anchor_slab_index)
            ),
            "anchor_symbol": self.anchor_symbol,
            "min_self_image_distance_A": float(self.min_self_image_distance_A),
            "min_vacuum_gap_A": float(self.min_vacuum_gap_A),
        }


def draw_rotation(
    rng: np.random.Generator,
) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """Uniform random rotation from a normalized Gaussian quaternion (Shoemake)."""
    quaternion = rng.standard_normal(4)
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        quaternion = quaternion / norm
    w, x, y, z = (float(value) for value in quaternion)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return (w, x, y, z), rotation


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix (uniformly distributed orientations)."""
    return draw_rotation(rng)[1]


def _lateral_vector(cell: np.ndarray, u: float, v: float) -> np.ndarray:
    """``u*a + v*b`` using the in-plane cell vectors (z components are zero)."""
    lateral = u * np.asarray(cell[0], dtype=float) + v * np.asarray(
        cell[1], dtype=float
    )
    return np.array([lateral[0], lateral[1], 0.0], dtype=float)


def _radii(atoms: Atoms) -> np.ndarray:
    return covalent_radii[np.asarray(atoms.numbers, dtype=int)]


def _min_q(atoms: Atoms, molecule: np.ndarray, slab: np.ndarray) -> float:
    distances = mic_distance_matrix(atoms, molecule, slab)
    if distances.size == 0:
        return float("inf")
    radii = _radii(atoms)
    q = distances / (radii[molecule][:, None] + radii[slab][None, :])
    return float(np.min(q))


def _min_distance(atoms: Atoms, molecule: np.ndarray, slab: np.ndarray) -> float:
    distances = mic_distance_matrix(atoms, molecule, slab)
    return float(np.min(distances)) if distances.size else float("inf")


def _anchor_q(atoms: Atoms, anchor: int, slab: np.ndarray) -> tuple[float, int]:
    distances = mic_distance_matrix(atoms, np.array([anchor]), slab)[0]
    radii = _radii(atoms)
    q = distances / (radii[anchor] + radii[slab])
    index = int(np.argmin(q))
    return float(q[index]), int(slab[index])


def place_molecule(
    slab: Atoms,
    conformers: Sequence[Atoms],
    options: ConfsearchOptions,
    *,
    rng: np.random.Generator,
    z_top: float | None = None,
    surface_validator: Callable[[Atoms], None] | None = None,
) -> PlacementResult:
    """Place one conformer above ``slab`` and return the combined structure.

    ``surface_validator`` is an optional callable ``(Atoms) -> None`` that raises
    when the candidate violates the shared section 5.3 surface rules; such a
    candidate is counted under the ``surface_geometry`` rejection reason.
    """
    if len(conformers) == 0:
        raise ConfsearchPreparationError("cs_molecule contains no conformers")
    n_molecule = len(conformers[0])
    if n_molecule == 0:
        raise ConfsearchPreparationError("cs_molecule frames contain no atoms")
    reference_numbers = np.asarray(conformers[0].numbers, dtype=int)
    for index, conformer in enumerate(conformers):
        if not np.array_equal(
            np.asarray(conformer.numbers, dtype=int), reference_numbers
        ):
            raise ConfsearchPreparationError(
                f"cs_molecule frame {index} has a different element order than "
                "frame 0; all conformers must share one element order"
            )

    if options.cs_place_anchor is not None:
        anchor_indices = options.anchor_indices
        assert anchor_indices is not None
        out_of_range = [i for i in anchor_indices if i >= n_molecule]
        if out_of_range:
            raise ConfsearchPreparationError(
                f"cs_place_anchor indices {out_of_range} are outside the "
                f"{n_molecule}-atom molecule file (0-based, like ASE)"
            )
    else:
        anchor_indices = None

    if options.anchor_target_symbols is not None:
        slab_symbols = set(slab.get_chemical_symbols())
        missing = sorted(set(options.anchor_target_symbols) - slab_symbols)
        if missing:
            raise ConfsearchPreparationError(
                f"cs_place_anchor_targets {missing} are not present in the slab "
                f"({sorted(slab_symbols)})"
            )

    cell = np.asarray(slab.cell, dtype=float)
    slab_numbers = np.asarray(slab.numbers, dtype=int)
    n_slab = len(slab)
    if z_top is None:
        z_top = float(np.max(np.asarray(slab.get_positions(), dtype=float)[:, 2]))

    rejections = {reason: 0 for reason in REJECTION_REASONS}
    slab_indices = np.arange(n_slab)
    molecule_indices = np.arange(n_slab, n_slab + n_molecule)

    def make_candidate(molecule_positions: np.ndarray, numbers: np.ndarray) -> Atoms:
        return Atoms(
            numbers=np.concatenate([slab_numbers, numbers]),
            positions=np.concatenate(
                [np.asarray(slab.get_positions(), dtype=float), molecule_positions]
            ),
            cell=cell,
            pbc=np.asarray(slab.pbc, dtype=bool),
        )

    for attempt in range(1, int(options.cs_place_max_attempts) + 1):
        conformer_index = int(rng.integers(0, len(conformers)))
        conformer = conformers[conformer_index]
        numbers = np.asarray(conformer.numbers, dtype=int)
        positions = np.asarray(conformer.get_positions(), dtype=float)
        centred = positions - positions.mean(axis=0)

        quaternion = (1.0, 0.0, 0.0, 0.0)
        if anchor_indices is not None:
            accepted = False
            for _ in range(_MAX_ROTATION_REDRAWS):
                quaternion, rotation = draw_rotation(rng)
                rotated = centred @ rotation.T
                if int(np.argmin(rotated[:, 2])) in anchor_indices:
                    accepted = True
                    break
            if not accepted:
                rejections["anchor_never_lowest"] += 1
                continue
        else:
            quaternion, rotation = draw_rotation(rng)
            rotated = centred @ rotation.T

        anchor_local = (
            int(np.argmin(rotated[:, 2])) if anchor_indices is not None else None
        )
        anchor_global = None if anchor_local is None else n_slab + anchor_local
        anchor_symbol = (
            None
            if anchor_local is None
            else conformer.get_chemical_symbols()[anchor_local]
        )

        u = float(rng.random())
        v = float(rng.random())
        lateral = _lateral_vector(cell, u, v)

        def positions_at(z_offset: float) -> np.ndarray:
            placed = rotated.copy()
            placed[:, 0] += lateral[0]
            placed[:, 1] += lateral[1]
            placed[:, 2] += (z_top + z_offset) - float(placed[:, 2].min())
            return placed

        def objective(z_offset: float) -> float:
            candidate = make_candidate(positions_at(z_offset), numbers)
            if anchor_global is not None:
                value, _ = _anchor_q(candidate, anchor_global, slab_indices)
                return value
            return _min_q(candidate, molecule_indices, slab_indices)

        z_offset = _solve_height(
            objective,
            target=(
                options.cs_place_anchor_scale
                if anchor_global is not None
                else options.cs_place_contact_scale
            ),
        )
        if z_offset is None:
            rejections["height_not_solvable"] += 1
            continue

        candidate = make_candidate(positions_at(z_offset), numbers)

        if anchor_global is not None:
            anchor_value, slab_index = _anchor_q(candidate, anchor_global, slab_indices)
            others = np.setdiff1d(np.arange(n_molecule), np.array([anchor_local]))
            if others.size:
                other_min = _min_q(candidate, n_slab + others, slab_indices)
                if other_min < options.cs_place_other_min_scale:
                    rejections["other_min_scale"] += 1
                    continue
            target_symbols = options.anchor_target_symbols
            if target_symbols is not None:
                closest_symbol = candidate.get_chemical_symbols()[slab_index]
                if closest_symbol not in target_symbols:
                    rejections["anchor_targets"] += 1
                    continue
        else:
            anchor_value = None
            slab_index = None
            if (
                _min_q(candidate, molecule_indices, slab_indices)
                < options.cs_place_contact_scale - 1e-3
            ):
                rejections["contact_scale"] += 1
                continue

        image_distance = self_image_distance(candidate, molecule_indices)
        if image_distance < options.cs_min_image_distance:
            rejections["self_image"] += 1
            continue
        vacuum = vacuum_gap(candidate, molecule_indices, slab_indices)
        if vacuum < options.cs_min_vacuum_gap:
            rejections["vacuum"] += 1
            continue
        if surface_validator is not None:
            try:
                surface_validator(candidate)
            except ConfsearchPreparationError:
                rejections["surface_geometry"] += 1
                continue

        return PlacementResult(
            atoms=candidate,
            kind="anchored" if anchor_global is not None else "contact",
            conformer_index=conformer_index,
            anchor_index=anchor_local,
            quaternion=tuple(float(value) for value in quaternion),
            u=u,
            v=v,
            z_offset=float(z_offset),
            attempts=attempt,
            rejections={key: value for key, value in rejections.items() if value},
            min_pair_q=_min_q(candidate, molecule_indices, slab_indices),
            min_pair_distance_A=_min_distance(
                candidate, molecule_indices, slab_indices
            ),
            anchor_q=anchor_value,
            anchor_slab_index=slab_index,
            min_self_image_distance_A=image_distance,
            min_vacuum_gap_A=vacuum,
            anchor_symbol=anchor_symbol,
        )

    raise ConfsearchPreparationError(
        "Automatic placement failed after "
        f"{options.cs_place_max_attempts} attempts; rejections per reason: "
        f"{ {key: value for key, value in rejections.items() if value} }. Adjust "
        "the placement scales, the cell vacuum or cs_place_anchor_targets."
    )


def _solve_height(objective, *, target: float) -> float | None:
    """Bisect the z offset until the target ``q`` is reached (spec 5.2 step 5)."""
    low = 0.0
    f_low = objective(low)
    if f_low >= target:
        # The molecule already reaches the target at the slab top: descend.
        low = -_BRACKET_STEP_A
        f_low = objective(low)
        while f_low > target:
            low -= _BRACKET_STEP_A
            if abs(low) > _MAX_SEARCH_A:
                return None
            f_low = objective(low)
        high = low + _BRACKET_STEP_A
    else:
        high = _BRACKET_STEP_A
        f_high = objective(high)
        while f_high < target:
            high += _BRACKET_STEP_A
            if high > _MAX_SEARCH_A:
                return None
            f_high = objective(high)
        low = high - _BRACKET_STEP_A
    for _ in range(200):
        if high - low <= Z_OFFSET_TOLERANCE_A:
            break
        middle = 0.5 * (low + high)
        if objective(middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


__all__ = [
    "REJECTION_REASONS",
    "Z_OFFSET_TOLERANCE_A",
    "PlacementResult",
    "draw_rotation",
    "place_molecule",
    "random_rotation_matrix",
]
