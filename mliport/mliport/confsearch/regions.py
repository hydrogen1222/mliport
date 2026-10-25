"""Region resolution and the shared fail-closed input validation.

Regions (spec section 4):

* ``A`` adsorbate atom indices
* ``S`` slab atom indices (everything not in ``A``)
* ``F`` fixed atoms (``FixAtoms`` from the input, optionally extended by
  ``cs_fix_below``)
* ``R`` relax set = everything not in ``F``
* ``M`` MD-mobile set (section 7.2); atoms outside ``M`` are held fixed during
  MD only, while every quench relaxes ``R``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ase.constraints import FixAtoms

from mliport.confsearch.errors import (
    ConfsearchError,
    ConfsearchInternalError,
    ConfsearchPreparationError,
)
from mliport.confsearch.geometry import (
    periodic_gap_and_contiguity,
    slab_interval,
    wrap_positions,
)

if TYPE_CHECKING:
    from ase import Atoms

    from mliport.confsearch.options import ConfsearchOptions

#: Surface-normal tolerance for the in-plane cell vectors (spec section 5.3).
SURFACE_NORMAL_TOLERANCE_A = 1e-8
#: Timestep limit when hydrogen is MD-mobile (spec section 5.3).
MAX_H_TIMESTEP_FS = 1.0


@dataclass(frozen=True, slots=True)
class Regions:
    """Resolved atom-index regions plus the warnings they produced."""

    adsorbate: np.ndarray
    slab: np.ndarray
    fixed: np.ndarray
    relax: np.ndarray
    warnings: tuple[str, ...]
    fix_below_added: tuple[int, ...] = ()

    @property
    def n_atoms(self) -> int:
        return int(self.adsorbate.size + self.slab.size)


@dataclass(frozen=True, slots=True)
class SurfaceGeometry:
    """Validated surface facts shared by placement and classification."""

    z_top: float
    z_min: float
    vacuum_gap: float
    adsorbate_centroid_z: float


def _fixed_indices(atoms: Atoms) -> np.ndarray:
    """Union of ``FixAtoms`` indices; any other constraint fails closed."""
    indices: set[int] = set()
    for constraint in atoms.constraints:
        if type(constraint) is not FixAtoms:  # noqa: E721 - FixScaled subclasses it
            raise ConfsearchPreparationError(
                "confsearch accepts only FixAtoms constraints; "
                f"{type(constraint).__name__} (partial selective dynamics, FixCom, "
                "FixSymmetry or any other constraint) is not supported because it "
                "would change the mobile set or the potential energy surface"
            )
        indices.update(int(index) for index in constraint.get_indices())
    return np.array(sorted(indices), dtype=int)


def resolve_regions(
    atoms: Atoms,
    options: ConfsearchOptions,
    *,
    adsorbate_indices: np.ndarray | None = None,
) -> Regions:
    """Resolve ``A/S/F/R`` for one structure and validate the constraints."""
    n_atoms = len(atoms)
    if n_atoms == 0:
        raise ConfsearchPreparationError("The structure contains no atoms")

    if adsorbate_indices is None:
        raise ConfsearchPreparationError(
            "confsearch needs the adsorbate atom indices (A); mode A takes them "
            "from cs_adsorbate_indices and mode B from the trailing molecule block"
        )
    adsorbate = np.asarray(adsorbate_indices, dtype=int)
    if adsorbate.size == 0:
        raise ConfsearchPreparationError("The adsorbate selection A is empty")
    if np.any(adsorbate < 0) or np.any(adsorbate >= n_atoms):
        raise ConfsearchPreparationError(
            f"The adsorbate selection A contains indices outside the {n_atoms}-atom "
            f"structure: {sorted(int(i) for i in adsorbate if i < 0 or i >= n_atoms)}"
        )
    if np.unique(adsorbate).size != adsorbate.size:
        raise ConfsearchPreparationError("The adsorbate selection A repeats indices")
    if adsorbate.size == n_atoms:
        raise ConfsearchPreparationError(
            "The adsorbate selection A is the whole structure; confsearch needs a "
            "slab plus one adsorbate"
        )

    fixed = _fixed_indices(atoms)
    fix_below_added: tuple[int, ...] = ()
    warnings: list[str] = []
    if options.cs_fix_below is not None:
        slab_candidates = np.setdiff1d(np.arange(n_atoms), adsorbate)
        wrapped = wrap_positions(atoms)
        z_min = float(np.min(wrapped[slab_candidates, 2]))
        below = slab_candidates[
            wrapped[slab_candidates, 2] < z_min + float(options.cs_fix_below)
        ]
        added = np.setdiff1d(below, fixed)
        if added.size:
            fixed = np.union1d(fixed, added).astype(int)
            fix_below_added = tuple(int(index) for index in added)

    if fixed.size and np.intersect1d(fixed, adsorbate).size:
        raise ConfsearchPreparationError(
            "Fixed atoms F must not intersect the adsorbate A (spec section 5.3); "
            f"the overlap is {sorted(int(i) for i in np.intersect1d(fixed, adsorbate))}"
        )
    if fixed.size == 0:
        warnings.append(
            "F is empty: a fully free slab can drift or bend during the search, so "
            "energy differences may reflect slab changes rather than adsorption"
        )

    slab = np.setdiff1d(np.arange(n_atoms), adsorbate)
    relax = np.setdiff1d(np.arange(n_atoms), fixed)
    return Regions(
        adsorbate=np.sort(adsorbate),
        slab=slab,
        fixed=np.sort(fixed),
        relax=relax,
        warnings=tuple(warnings),
        fix_below_added=fix_below_added,
    )


def validate_surface(
    atoms: Atoms, regions: Regions, options: ConfsearchOptions
) -> SurfaceGeometry:
    """Validate the surface convention and return the validated geometry.

    Fail-closed rules (spec section 5.3): 3D PBC, surface normal along +z, a
    contiguous slab whose vacuum is the periodic gap, and an adsorbate above the
    slab top.
    """
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if not bool(pbc.all()):
        raise ConfsearchPreparationError(
            f"confsearch requires pbc == (True, True, True), got {pbc.tolist()}; "
            "molecular (non-periodic) tasks are a v1 non-goal"
        )

    cell = np.asarray(atoms.cell, dtype=float)
    if (
        abs(float(cell[0, 2])) > SURFACE_NORMAL_TOLERANCE_A
        or abs(float(cell[1, 2])) > SURFACE_NORMAL_TOLERANCE_A
    ):
        raise ConfsearchPreparationError(
            "The surface normal must be +z: a_z and b_z must be 0 within "
            f"{SURFACE_NORMAL_TOLERANCE_A:g} Angstrom, got a_z={cell[0, 2]!r}, "
            f"b_z={cell[1, 2]!r}. confsearch does not rotate the cell for you."
        )
    cell_height = float(cell[2, 2])
    if cell_height <= 0:
        raise ConfsearchPreparationError(
            f"The cell height c_z must be > 0, got {cell_height!r}"
        )

    wrapped = wrap_positions(atoms)
    slab_z = wrapped[regions.slab, 2]
    periodic, internal = periodic_gap_and_contiguity(slab_z, cell_height)
    if periodic <= internal:
        raise ConfsearchPreparationError(
            "The slab must be contiguous in z with its vacuum across the c "
            "boundary (place the slab at the bottom of the cell, vacuum on +z). "
            f"Got periodic gap {periodic:.3f} A <= largest internal gap "
            f"{internal:.3f} A; some slab atoms look wrapped across the boundary."
        )
    z_min, z_max = slab_interval(slab_z, cell_height)
    centroid_z = float(np.mean(wrapped[regions.adsorbate, 2]))
    if centroid_z <= z_max:
        raise ConfsearchPreparationError(
            "One-sided adsorption requires the adsorbate centroid above the slab "
            f"top: centroid z={centroid_z:.3f} A, slab z_top={z_max:.3f} A. "
            "Put the molecule above the +z surface."
        )
    return SurfaceGeometry(
        z_top=z_max,
        z_min=z_min,
        vacuum_gap=periodic,
        adsorbate_centroid_z=centroid_z,
    )


def validate_adsorbate_connectivity(
    atoms: Atoms, regions: Regions, options: ConfsearchOptions
) -> None:
    """The non-ionic part of A must form one connected covalent graph."""
    from mliport.confsearch.graphs import (  # noqa: PLC0415
        covalent_bonds,
        ionic_mask,
    )

    adsorbate = regions.adsorbate
    ionic = ionic_mask(atoms, options)
    non_ionic = adsorbate[~ionic[adsorbate]]
    if non_ionic.size <= 1:
        return
    graph = covalent_bonds(
        atoms,
        non_ionic,
        bond_scale=options.cs_bond_scale,
        ionic_symbols=options.ionic_symbols,
    )
    adjacency: dict[int, set[int]] = {int(index): set() for index in non_ionic}
    for a, b in graph.bonds:
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    start = int(non_ionic[0])
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    if len(seen) != non_ionic.size:
        missing = sorted(int(index) for index in non_ionic if int(index) not in seen)
        raise ConfsearchPreparationError(
            "The non-ionic part of the adsorbate selection A is not one connected "
            f"covalent graph; atoms {missing} are disconnected from "
            f"atom {start} at cs_bond_scale={options.cs_bond_scale}. This is the "
            "most likely user error: check cs_adsorbate_indices (0-based, like ASE; "
            "CP2K/VMD count from 1)."
        )


def mobile_indices(
    atoms: Atoms,
    regions: Regions,
    options: ConfsearchOptions,
    *,
    z_top: float | None = None,
) -> np.ndarray:
    """MD-mobile set M for the current minimum (spec section 7.2)."""
    mode = options.cs_md_mobile
    if mode == "adsorbate":
        return np.sort(regions.adsorbate)
    if mode == "all_free":
        return np.sort(regions.relax)
    if mode == "adsorbate_and_top":
        if z_top is None:
            raise ConfsearchInternalError(
                "cs_md_mobile='adsorbate_and_top' needs z_top of the current minimum"
            )
        free_slab = np.setdiff1d(regions.slab, regions.fixed, assume_unique=False)
        wrapped = wrap_positions(atoms)
        top = free_slab[wrapped[free_slab, 2] >= z_top - float(options.cs_top_depth)]
        return np.sort(np.union1d(regions.adsorbate, top))
    raise ConfsearchError(f"Unknown cs_md_mobile {mode!r}")


def validate_h_timestep(
    atoms: Atoms, mobile: np.ndarray, options: ConfsearchOptions
) -> None:
    """Refuse a large timestep when hydrogen is MD-mobile (spec section 5.3)."""
    symbols = atoms.get_chemical_symbols()
    has_hydrogen = any(symbols[int(index)] == "H" for index in mobile)
    if has_hydrogen and options.timestep > MAX_H_TIMESTEP_FS:
        raise ConfsearchPreparationError(
            "timestep must be <= "
            f"{MAX_H_TIMESTEP_FS} fs when H is in the MD-mobile set M "
            f"(timestep={options.timestep} fs); hydrogen vibrations are not "
            "resolved by a larger step"
        )


def validate_all(
    atoms: Atoms,
    regions: Regions,
    options: ConfsearchOptions,
) -> SurfaceGeometry:
    """Run every structure-level fail-closed check and return the geometry."""
    geometry = validate_surface(atoms, regions, options)
    validate_adsorbate_connectivity(atoms, regions, options)
    mobile = mobile_indices(atoms, regions, options, z_top=geometry.z_top)
    validate_h_timestep(atoms, mobile, options)
    return geometry


__all__ = [
    "MAX_H_TIMESTEP_FS",
    "SURFACE_NORMAL_TOLERANCE_A",
    "Regions",
    "SurfaceGeometry",
    "mobile_indices",
    "resolve_regions",
    "validate_adsorbate_connectivity",
    "validate_all",
    "validate_h_timestep",
    "validate_surface",
]
