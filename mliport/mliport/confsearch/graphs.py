"""Covalent graphs, chemisorption flags and status rules (spec section 8).

All bond criteria use ``ase.data.covalent_radii`` and minimum-image distances.
Ionic species (``cs_ionic_species``) are excluded from every covalent graph:
their coordination may change freely and a mobile ion hopping between sites is
not a chemical reaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

from mliport.confsearch import status as cs_status
from mliport.confsearch.errors import ConfsearchInternalError
from mliport.confsearch.geometry import (
    mic_distance_matrix,
    mic_displacements,
    minimum_distance,
    self_image_distance,
    vacuum_gap,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ase import Atoms

    from mliport.confsearch.options import ConfsearchOptions

#: Displacement thresholds for the slab flags (spec section 8.3), in Angstrom.
SLAB_REARRANGED_DISPLACEMENT_A = 1.0
FAR_FIELD_DISPLACEMENT_A = 0.5
FAR_FIELD_DISTANCE_A = 6.0
#: Ionic coordination shell, in units of the covalent-radii sum (informational).
IONIC_COORDINATION_SCALE = 1.2


@dataclass(frozen=True, slots=True)
class GraphDifference:
    """Bonds lost and gained relative to a reference graph."""

    lost: tuple[tuple[int, int], ...] = ()
    gained: tuple[tuple[int, int], ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.lost or self.gained)

    def to_dict(self) -> dict[str, list[list[int]]]:
        return {
            "lost": [list(pair) for pair in self.lost],
            "gained": [list(pair) for pair in self.gained],
        }


@dataclass(frozen=True, slots=True)
class CovalentGraph:
    """An undirected bond set over a fixed atom-index subset."""

    bonds: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    def __len__(self) -> int:
        return len(self.bonds)

    def as_sorted_list(self) -> list[list[int]]:
        return [list(pair) for pair in sorted(self.bonds)]

    def difference(self, other: CovalentGraph) -> GraphDifference:
        """Bonds in ``self`` that ``other`` lacks, and vice versa."""
        lost = tuple(sorted(self.bonds - other.bonds))
        gained = tuple(sorted(other.bonds - self.bonds))
        return GraphDifference(lost=lost, gained=gained)


@dataclass(frozen=True, slots=True)
class MinimumClassification:
    """Everything spec section 8 records for one converged quench."""

    status: str
    flags: tuple[str, ...]
    adsorbate_slab_bonds: tuple[tuple[int, int, str, str, float, float], ...]
    reacted_bonds: dict[str, dict[str, list[list[int]]]]
    min_adsorbate_slab_distance: float
    min_self_image_distance: float
    min_vacuum_gap: float
    max_slab_displacement_ionic: float
    max_slab_displacement_nonionic: float
    ionic_coordination: tuple[tuple[int, int], ...]

    @property
    def chemisorbed(self) -> bool:
        return "chemisorbed" in self.flags

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "flags": list(self.flags),
            "chemisorbed": self.chemisorbed,
            "adsorbate_slab_bonds": [
                {
                    "adsorbate_index": bond[0],
                    "slab_index": bond[1],
                    "adsorbate_element": bond[2],
                    "slab_element": bond[3],
                    "distance_A": bond[4],
                    "q": bond[5],
                }
                for bond in self.adsorbate_slab_bonds
            ],
            "reacted_bonds": self.reacted_bonds,
            "min_adsorbate_slab_distance_A": self.min_adsorbate_slab_distance,
            "min_self_image_distance_A": self.min_self_image_distance,
            "min_vacuum_gap_A": self.min_vacuum_gap,
            "max_slab_displacement_ionic_A": self.max_slab_displacement_ionic,
            "max_slab_displacement_nonionic_A": self.max_slab_displacement_nonionic,
            "ionic_adsorbate_coordination": [
                {"adsorbate_index": index, "neighbours": count}
                for index, count in self.ionic_coordination
            ],
        }


@dataclass(frozen=True, slots=True)
class ClassificationReference:
    """The reference state a minimum is classified against."""

    adsorbate_graph: CovalentGraph
    slab_graph: CovalentGraph
    #: Positions of the replica's start minimum, used for displacement flags.
    start_positions: np.ndarray


def ionic_mask(atoms: Atoms, options: ConfsearchOptions) -> np.ndarray:
    """Boolean mask of atoms whose element is an ionic species."""
    symbols = set(options.ionic_symbols)
    return np.array(
        [symbol in symbols for symbol in atoms.get_chemical_symbols()], dtype=bool
    )


def covalent_bonds(
    atoms: Atoms,
    first: np.ndarray,
    second: np.ndarray | None = None,
    *,
    bond_scale: float,
    ionic_symbols: Iterable[str],
) -> CovalentGraph:
    """Bonds among ``first`` (or between ``first`` and ``second``).

    A bond exists when the MIC distance is ``<= bond_scale * (r_i + r_j)`` and
    both atoms are non-ionic.  ``second=None`` means bonds within ``first``.
    """
    left = np.asarray(first, dtype=int)
    right = left if second is None else np.asarray(second, dtype=int)
    if left.size == 0 or right.size == 0:
        return CovalentGraph()

    ionic = set(ionic_symbols)
    symbols = atoms.get_chemical_symbols()
    non_ionic = np.array([symbol not in ionic for symbol in symbols], dtype=bool)
    radii = covalent_radii[np.asarray(atoms.numbers, dtype=int)]

    candidates = np.union1d(left, right)
    candidates = candidates[non_ionic[candidates]]
    if candidates.size < 2:
        return CovalentGraph()
    cutoff = float(bond_scale * 2.0 * np.max(radii[candidates]))
    if cutoff <= 0:
        return CovalentGraph()

    i, j, distances = neighbor_list("ijd", atoms, cutoff, self_interaction=False)
    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)
    distances = np.asarray(distances, dtype=float)

    left_membership = np.zeros(len(atoms), dtype=bool)
    left_membership[left] = True
    right_membership = np.zeros(len(atoms), dtype=bool)
    right_membership[right] = True

    if second is None:
        selected = left_membership[i] & right_membership[j]
    else:
        selected = (left_membership[i] & right_membership[j]) | (
            right_membership[i] & left_membership[j]
        )
    selected &= non_ionic[i] & non_ionic[j]

    bonds: set[tuple[int, int]] = set()
    for a, b, distance in zip(
        i[selected], j[selected], distances[selected], strict=True
    ):
        if distance <= bond_scale * (radii[a] + radii[b]) + 1e-12:
            bonds.add((int(min(a, b)), int(max(a, b))))
    return CovalentGraph(frozenset(bonds))


def broken_reference_bonds(
    atoms: Atoms,
    reference: CovalentGraph,
    *,
    break_scale: float,
    ionic_symbols: Iterable[str],
) -> tuple[tuple[int, int], ...]:
    """Reference bonds stretched beyond ``break_scale`` (MD early check).

    New bonds are deliberately not detected during MD (spec section 8.1).
    """
    if not reference.bonds:
        return ()
    ionic = set(ionic_symbols)
    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    radii = covalent_radii[np.asarray(atoms.numbers, dtype=int)]
    from mliport.confsearch.geometry import mic_vectors  # noqa: PLC0415

    broken: list[tuple[int, int]] = []
    for a, b in sorted(reference.bonds):
        if symbols[a] in ionic or symbols[b] in ionic:
            continue
        delta = positions[b] - positions[a]
        distance = float(np.linalg.norm(mic_vectors(cell, delta)))
        if distance > break_scale * (radii[a] + radii[b]):
            broken.append((a, b))
    return tuple(broken)


def classify_minimum(
    atoms: Atoms,
    *,
    regions,
    options: ConfsearchOptions,
    reference: ClassificationReference,
) -> MinimumClassification:
    """Classify one converged quench (spec sections 8.2-8.4)."""
    fixed = np.asarray(regions.fixed, dtype=int)
    current_positions = np.asarray(atoms.get_positions(), dtype=float)
    if fixed.size:
        if not np.array_equal(
            current_positions[fixed], reference.start_positions[fixed]
        ):
            raise ConfsearchInternalError(
                "A fixed atom moved during the search; this is an internal error, "
                "not a scientific outcome (spec section 8.2 rule 1)"
            )

    adsorbate = np.asarray(regions.adsorbate, dtype=int)
    slab = np.asarray(regions.slab, dtype=int)
    free_slab = np.setdiff1d(slab, fixed, assume_unique=False)
    ionic = ionic_mask(atoms, options)

    adsorbate_graph = covalent_bonds(
        atoms,
        adsorbate,
        bond_scale=options.cs_bond_scale,
        ionic_symbols=options.ionic_symbols,
    )
    slab_graph = covalent_bonds(
        atoms,
        slab,
        bond_scale=options.cs_bond_scale,
        ionic_symbols=options.ionic_symbols,
    )
    cross = covalent_bonds(
        atoms,
        adsorbate,
        slab,
        bond_scale=options.cs_bond_scale,
        ionic_symbols=options.ionic_symbols,
    )

    adsorbate_diff = reference.adsorbate_graph.difference(adsorbate_graph)
    slab_diff = reference.slab_graph.difference(slab_graph)

    radii = covalent_radii[np.asarray(atoms.numbers, dtype=int)]
    symbols = atoms.get_chemical_symbols()
    bond_records: list[tuple[int, int, str, str, float, float]] = []
    adsorbate_set = set(int(index) for index in adsorbate)
    for first, second in sorted(cross.bonds):
        a, b = (first, second) if first in adsorbate_set else (second, first)
        delta = (current_positions[b] - current_positions[a])[None, :]
        distance = float(mic_displacements(atoms, delta)[0])
        bond_records.append(
            (
                int(a),
                int(b),
                symbols[a],
                symbols[b],
                distance,
                distance / (radii[a] + radii[b]),
            )
        )
    adsorbate_slab_bonds = tuple(bond_records)

    min_adsorbate_slab = minimum_distance(atoms, adsorbate, slab)
    image_distance = self_image_distance(atoms, adsorbate)
    vacuum = vacuum_gap(atoms, adsorbate, slab)

    displacements = mic_displacements(
        atoms, current_positions[free_slab] - reference.start_positions[free_slab]
    )
    ionic_free = ionic[free_slab]
    max_ionic = float(np.max(displacements[ionic_free])) if np.any(ionic_free) else 0.0
    max_nonionic = (
        float(np.max(displacements[~ionic_free])) if np.any(~ionic_free) else 0.0
    )

    nonionic_free = free_slab[~ionic[free_slab]]
    slab_rearranged = bool(
        nonionic_free.size
        and np.any(
            mic_displacements(
                atoms,
                current_positions[nonionic_free]
                - reference.start_positions[nonionic_free],
            )
            > SLAB_REARRANGED_DISPLACEMENT_A
        )
    )
    far_field_changed = False
    if free_slab.size and adsorbate.size:
        distances_to_adsorbate = mic_distance_matrix(atoms, free_slab, adsorbate)
        nearest = np.min(distances_to_adsorbate, axis=1)
        far = nearest > FAR_FIELD_DISTANCE_A
        if np.any(far):
            far_field_changed = bool(
                np.any(displacements[far] > FAR_FIELD_DISPLACEMENT_A)
            )

    ionic_coordination: list[tuple[int, int]] = []
    for index in adsorbate:
        if not ionic[index]:
            continue
        others = np.setdiff1d(np.arange(len(atoms)), np.array([index]))
        distances = mic_distance_matrix(atoms, np.array([index]), others)[0]
        threshold = IONIC_COORDINATION_SCALE * (radii[index] + radii[others])
        ionic_coordination.append(
            (int(index), int(np.count_nonzero(distances <= threshold)))
        )

    flags: list[str] = []
    if cross.bonds:
        flags.append("chemisorbed")
    else:
        flags.append("physisorbed")
    if slab_rearranged:
        flags.append("slab_rearranged")
    if far_field_changed:
        flags.append("far_field_slab_changed")

    reacted_bonds = {
        "adsorbate": adsorbate_diff.to_dict(),
        "slab": slab_diff.to_dict(),
    }

    if adsorbate_diff.changed or slab_diff.changed:
        status = cs_status.REACTED
    elif min_adsorbate_slab > options.cs_desorption_distance:
        status = cs_status.DESORBED
    elif (
        image_distance < options.cs_min_image_distance
        or vacuum < options.cs_min_vacuum_gap
    ):
        status = cs_status.IMAGE_CONTACT
    else:
        status = cs_status.VALID

    return MinimumClassification(
        status=status,
        flags=tuple(flags),
        adsorbate_slab_bonds=adsorbate_slab_bonds,
        reacted_bonds=reacted_bonds,
        min_adsorbate_slab_distance=min_adsorbate_slab,
        min_self_image_distance=image_distance,
        min_vacuum_gap=vacuum,
        max_slab_displacement_ionic=max_ionic,
        max_slab_displacement_nonionic=max_nonionic,
        ionic_coordination=tuple(ionic_coordination),
    )


__all__ = [
    "FAR_FIELD_DISPLACEMENT_A",
    "FAR_FIELD_DISTANCE_A",
    "IONIC_COORDINATION_SCALE",
    "SLAB_REARRANGED_DISPLACEMENT_A",
    "ClassificationReference",
    "CovalentGraph",
    "GraphDifference",
    "MinimumClassification",
    "broken_reference_bonds",
    "classify_minimum",
    "covalent_bonds",
    "ionic_mask",
]
