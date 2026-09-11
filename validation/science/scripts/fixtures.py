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
