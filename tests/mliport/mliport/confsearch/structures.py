"""Synthetic structures and options shared by the confsearch tests.

Everything here is calculator-free: a small periodic "slab", a bonded
two-atom "molecule", and the option sets for the two start modes.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from ase import Atoms

from mliport.confsearch.graphs import (
    ClassificationReference,
    MinimumClassification,
    covalent_bonds,
)
from mliport.confsearch.options import ConfsearchOptions

#: A slab of sulfur-like atoms: 2x2 grid, S-S bonds at 2.0 A (1.2 * 2.05 = 2.46).
SLAB_SPACING_A = 2.0
SLAB_Z_A = 1.0
SLAB_HEIGHT_A = 20.0


def slab_atoms(
    *,
    spacing: float = SLAB_SPACING_A,
    z: float = SLAB_Z_A,
    height: float = SLAB_HEIGHT_A,
    element: str = "S",
    n: int = 2,
) -> Atoms:
    """Periodic single-layer slab at the bottom of the cell, vacuum on +z."""
    positions = [
        [0.5 + i * spacing, 0.5 + j * spacing, z] for i in range(n) for j in range(n)
    ]
    cell = [n * spacing, n * spacing, height]
    return Atoms(element * (n * n), positions=positions, cell=cell, pbc=True)


def oh_molecule(*, offset: float = 0.0) -> Atoms:
    """A bonded O-H molecule (d(O,H) = 0.96 A < 1.2 * (0.66 + 0.31))."""
    return Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.96 + offset, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )


def combine(slab: Atoms, molecule: Atoms, *, shift=(0.0, 0.0, 3.0)) -> Atoms:
    """Append the molecule to the slab after a rigid shift."""
    positions = np.asarray(molecule.get_positions(), dtype=float) + np.asarray(
        shift, dtype=float
    )
    return Atoms(
        numbers=np.concatenate([slab.numbers, molecule.numbers]),
        positions=np.concatenate([slab.get_positions(), positions]),
        cell=slab.cell,
        pbc=slab.pbc,
    )


def adsorbate_indices(slab: Atoms, molecule: Atoms) -> np.ndarray:
    return np.arange(len(slab), len(slab) + len(molecule))


def mode_a_options(**overrides) -> ConfsearchOptions:
    base = {
        "cs_starts": ("starts.xyz",),
        "cs_adsorbate_indices": "4-5",
        "cs_cycles": 0,
    }
    base.update(overrides)
    return ConfsearchOptions(**base)


def mode_b_options(**overrides) -> ConfsearchOptions:
    base = {
        "cs_slab": "slab.vasp",
        "cs_molecule": "molecule.xyz",
        "cs_n_starts": 1,
        "cs_cycles": 0,
    }
    base.update(overrides)
    return ConfsearchOptions(**base)


def reference_for(
    atoms: Atoms,
    regions,
    options: ConfsearchOptions,
    *,
    start_positions: np.ndarray | None = None,
) -> ClassificationReference:
    """Build the reference graphs a minimum is classified against."""
    return ClassificationReference(
        adsorbate_graph=covalent_bonds(
            atoms,
            regions.adsorbate,
            bond_scale=options.cs_bond_scale,
            ionic_symbols=options.ionic_symbols,
        ),
        slab_graph=covalent_bonds(
            atoms,
            regions.slab,
            bond_scale=options.cs_bond_scale,
            ionic_symbols=options.ionic_symbols,
        ),
        start_positions=(
            np.asarray(atoms.get_positions(), dtype=float).copy()
            if start_positions is None
            else np.asarray(start_positions, dtype=float).copy()
        ),
    )


def fake_classification(status: str = "valid", **overrides) -> MinimumClassification:
    base = {
        "status": status,
        "flags": ("physisorbed",),
        "adsorbate_slab_bonds": (),
        "reacted_bonds": {
            "adsorbate": {"lost": [], "gained": []},
            "slab": {"lost": [], "gained": []},
        },
        "min_adsorbate_slab_distance": 3.0,
        "min_self_image_distance": 6.0,
        "min_vacuum_gap": 12.0,
        "max_slab_displacement_ionic": 0.0,
        "max_slab_displacement_nonionic": 0.0,
        "ionic_coordination": (),
    }
    base.update(overrides)
    return MinimumClassification(**base)


def with_energy(atoms: Atoms, energy: float) -> Atoms:
    """Attach a SinglePointCalculator with a fixed energy (no forces needed)."""
    from ase.calculators.singlepoint import SinglePointCalculator

    result = atoms.copy()
    result.calc = SinglePointCalculator(result, energy=float(energy))
    return result


__all__ = [
    "SLAB_HEIGHT_A",
    "SLAB_SPACING_A",
    "SLAB_Z_A",
    "adsorbate_indices",
    "combine",
    "fake_classification",
    "mode_a_options",
    "mode_b_options",
    "oh_molecule",
    "reference_for",
    "replace",
    "slab_atoms",
    "with_energy",
]
