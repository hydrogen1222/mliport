"""Deterministic structure fixtures for the beta validation suite.

Small, fully periodic reference crystals spanning metallic (Cu), covalent
(Si) and ionic (MgO) bonding.  Lattice constants are standard room-temperature
experimental values; these fixtures are identity/runtime checks and static-
workflow starting points, not accuracy benchmarks on their own.
"""

from __future__ import annotations

from typing import Any

from ase.build import bulk

from common import structure_id

#: fixture name -> (symbol, crystalstructure, a, cubic, source note)
FIXTURES: dict[str, dict[str, Any]] = {
    "cu_fcc": {
        "symbol": "Cu",
        "structure": "fcc",
        "a": 3.615,
        "cubic": True,
        "source": "experimental RT lattice constant",
    },
    "si_diamond": {
        "symbol": "Si",
        "structure": "diamond",
        "a": 5.430,
        "cubic": True,
        "source": "experimental RT lattice constant",
    },
    "mgo_rocksalt": {
        "symbol": "MgO",
        "structure": "rocksalt",
        "a": 4.212,
        "cubic": True,
        "source": "experimental RT lattice constant",
    },
}


#: alpha-Na3PS4 (tetragonal, space group P-42_1c, No. 114) pinned from the
#: public Materials Project entry mp-28782 via the OPTIMADE API on 2026-09-11.
#: Geometry fixture only (taskbook section 15); never compared against MP
#: total energies.  Source/license/citation are recorded in data_manifest.json.
NA3PS4_CELL = [
    [6.92369091, 0.0, 0.0],
    [0.0, 6.92369091, 0.0],
    [0.0, 0.0, 7.07714131],
]
NA3PS4_SITES = [  # (symbol, x, y, z) in Angstrom, OPTIMADE cartesian_site_positions
    ("Na", 0.0, 3.46185, 2.9701),
    ("Na", 3.46185, 0.0, 4.10704),
    ("Na", 3.46185, 0.0, 0.56847),
    ("Na", 0.0, 3.46185, 6.50867),
    ("Na", 0.0, 0.0, 0.0),
    ("Na", 3.46185, 3.46185, 3.53857),
    ("P", 0.0, 0.0, 3.53857),
    ("P", 3.46185, 3.46185, 0.0),
    ("S", 2.14503, 2.41271, 1.17396),
    ("S", 2.41271, 4.77866, 5.90318),
    ("S", 4.51098, 2.14503, 5.90318),
    ("S", 5.60688, 1.04914, 2.36461),
    ("S", 1.31681, 5.87455, 2.36461),
    ("S", 4.77866, 4.51098, 1.17396),
    ("S", 1.04914, 1.31681, 4.71253),
    ("S", 5.87455, 5.60688, 4.71253),
]


def build_extra(name: str):
    """Build the non-bulk fixtures (triclinic, distorted) deterministically."""

    import numpy as np

    if name == "si_diamond_prim":
        # Diamond primitive cell: genuinely non-orthogonal (rhombohedral, 60 deg).
        atoms = bulk("Si", "diamond", a=5.430, cubic=False)
    elif name == "cu_distorted":
        # 2x2x2 Cu supercell with a seeded 0.05 A displacement pattern so
        # forces and shear stress are non-trivial.
        base = bulk("Cu", "fcc", a=3.615, cubic=True)
        atoms = base.repeat((2, 2, 2))
        rng = np.random.default_rng(20260911)
        disp = rng.normal(scale=0.05, size=(len(atoms), 3))
        disp -= disp.mean(axis=0)  # keep the cell centred
        atoms.positions += disp
    elif name == "na3ps4":
        # alpha-Na3PS4 from Materials Project mp-28782 (pinned geometry fixture;
        # see NA3PS4_* block above and data_manifest.json for provenance).
        from ase import Atoms

        atoms = Atoms(
            [s[0] for s in NA3PS4_SITES],
            positions=[s[1:] for s in NA3PS4_SITES],
            cell=NA3PS4_CELL,
            pbc=True,
        )
    else:
        msg = f"unknown extra fixture {name!r}"
        raise ValueError(msg)
    return atoms, {
        "name": name,
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "structure_id": structure_id(atoms),
    }


# Systems used by the invariance and finite-difference tiers.
INVARIANCE_SYSTEMS = (
    "cu_fcc",
    "si_diamond_prim",  # triclinic coverage (taskbook section 11.4)
    "cu_distorted",     # non-trivial forces/stress
    "mgo_rocksalt",
)


def build_fixture(name: str):
    """Build one deterministic fixture and return ``(atoms, descriptor)``."""
    spec = FIXTURES[name]
    atoms = bulk(
        spec["symbol"],
        spec["structure"],
        a=spec["a"],
        cubic=spec["cubic"],
    )
    desc = {
        "name": name,
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "a_A": spec["a"],
        "crystalstructure": spec["structure"],
        "source": spec["source"],
        "structure_id": structure_id(atoms),
    }
    return atoms, desc


def all_fixture_descriptors() -> dict[str, dict[str, Any]]:
    return {name: build_fixture(name)[1] for name in FIXTURES}
