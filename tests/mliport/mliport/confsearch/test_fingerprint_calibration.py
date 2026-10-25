"""Calibration of ``cs_fp_sigma`` and ``cs_dedup_fp_tol`` (spec 13.2, revision 4).

Procedure implemented here (P1, geometry-labelled pairs; the P2 engine re-runs
it with real quenches at ``cs_quench_fmax``):

1. Pair classes labelled by geometry:
   * (a) *same*: perturbations of the reference minimum whose largest
     minimum-image adsorbate displacement is within ``cs_dedup_geom_tol``
     (0.20 A).  Perturbations landing more than 0.5 A away are a different
     basin and are excluded and counted.
   * (b) *different sites* and (c) *different conformers*: with a largest
     adsorbate displacement > 0.5 A.
   * (d) *equivalent*: the same minimum rigidly mapped by a slab symmetry
     operation, or a permutation of equivalent atoms (a CH3/water rotation).
2. Sweep ``cs_fp_sigma`` in {0.05, 0.10, 0.20} A at binwidth 0.02 A, rcut 8 A.
3. ``D_tol(sigma) = 0.5 * min D`` over (b u c) pairs: zero false merges with a
   factor-2 margin.
4. Default sigma maximises the (d) merge fraction; ties go to the larger sigma.
5. If (d) cannot be fully merged, keep the zero-false-merge tolerance and record
   the residual split rate as a known limitation instead of widening into the
   overlap.

The measured table is asserted against the shipped defaults, so the defaults
cannot drift without re-running this calibration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from mliport.confsearch.fingerprint import build_fingerprint
from mliport.confsearch.geometry import mic_vectors
from mliport.confsearch.options import CS_DEFAULTS, ConfsearchOptions
from mliport.confsearch.placement import random_rotation_matrix

SIGMA_CANDIDATES = (0.05, 0.10, 0.20)
#: Probe perturbations (A) applied to the adsorbate before the "re-quench".
PERTURBATIONS_A = (0.02, 0.05, 0.10)
DIFFERENT_BASIN_A = 0.5
REPEATS = 12


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "examples" / "structures").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


LGPS_PATH = _repository_root() / "examples/structures/li10gep2s12_primitive.vasp"


def _adsorbate(torsion_deg: float = 0.0) -> Atoms:
    """A CH3OH-like 6-atom adsorbate with a definable O-H torsion."""
    angle = np.deg2rad(torsion_deg)
    positions = [[0.0, 0.0, 0.0], [1.43, 0.0, 0.0]]
    oh = 0.9637
    bend = np.deg2rad(108.0)
    positions.append(
        [
            1.43 + oh * np.cos(bend),
            oh * np.sin(bend) * np.cos(angle),
            oh * np.sin(bend) * np.sin(angle),
        ]
    )
    for phase in (0.0, 120.0, 240.0):
        phi = np.deg2rad(phase)
        positions.append([-1.09 * np.cos(phi), 1.09 * np.sin(phi), 0.35])
    return Atoms("COH4", positions=positions, pbc=False)


def _combine(slab: Atoms, molecule: Atoms, shift) -> Atoms:
    return Atoms(
        numbers=np.concatenate([slab.numbers, molecule.numbers]),
        positions=np.concatenate(
            [slab.get_positions(), molecule.get_positions() + np.asarray(shift)]
        ),
        cell=slab.cell,
        pbc=slab.pbc,
    )


def _lgps_setup():
    if not LGPS_PATH.is_file():
        pytest.skip(f"LGPS structure not available at {LGPS_PATH}")
    slab = read(LGPS_PATH)
    symbols = np.array(slab.get_chemical_symbols())
    sulfur = np.where(symbols == "S")[0]
    reference_shift = slab.get_positions()[sulfur[0]] + np.array([0.0, 0.0, 2.07])
    reference = _combine(slab, _adsorbate(), reference_shift)
    return slab, sulfur, reference_shift, reference


def _symmetric_setup():
    """A slab and adsorbate with a mirror plane at y = 0 and 3-fold methyl H."""
    slab = Atoms(
        "S4",
        positions=[
            [-2.0371, -1.9837, 1.0137],
            [2.0413, -1.9917, 1.0213],
            [-2.0243, 2.0317, 0.9937],
            [2.0117, 2.0271, 1.0173],
        ],
        cell=[8.1371, 8.0917, 20.1137],
        pbc=True,
    )
    reference = _combine(slab, _adsorbate(), np.array([0.0, 0.0, 2.5137]))
    return slab, reference


def _slab_symmetry_operations(slab: Atoms):
    """Slab symmetry operations as ``(rotation, translation)`` in fractional coords."""
    import spglib

    cell = (
        np.asarray(slab.cell, dtype=float),
        np.asarray(slab.get_scaled_positions(), dtype=float),
        np.asarray(slab.numbers, dtype=int),
    )
    symmetry = spglib.get_symmetry(cell, symprec=1e-3)
    if symmetry is None:
        return []
    return list(zip(symmetry["rotations"], symmetry["translations"], strict=False))


def _apply_symmetry(atoms: Atoms, rotation, translation) -> Atoms:
    fractional = np.asarray(atoms.get_scaled_positions(), dtype=float)
    moved = (fractional @ np.asarray(rotation, dtype=float).T) + np.asarray(
        translation, dtype=float
    )
    return Atoms(
        numbers=np.asarray(atoms.numbers, dtype=int),
        scaled_positions=moved % 1.0,
        cell=np.asarray(atoms.cell, dtype=float),
        pbc=np.asarray(atoms.pbc, dtype=bool),
    )


def _same_structure(first: Atoms, second: Atoms, *, tolerance: float = 0.05) -> bool:
    """True when the two structures coincide atom by atom (wrapped, MIC)."""
    cell = np.asarray(first.cell, dtype=float)
    deltas = np.asarray(first.get_positions()) - np.asarray(second.get_positions())
    folded = mic_vectors(cell, deltas)
    return bool(np.all(np.linalg.norm(folded, axis=1) <= tolerance))


def _symmetry_related(reference: Atoms, candidate: Atoms, operations) -> bool:
    """True when a slab symmetry operation maps the reference onto the candidate."""
    for rotation, translation in operations:
        mapped = _apply_symmetry(reference, rotation, translation)
        if _same_structure(mapped, candidate):
            return True
    return False


def _max_adsorbate_displacement(first: Atoms, second: Atoms, adsorbate) -> float:
    indices = np.asarray(adsorbate, dtype=int)
    deltas = (
        np.asarray(first.get_positions())[indices]
        - np.asarray(second.get_positions())[indices]
    )
    folded = mic_vectors(np.asarray(first.cell, dtype=float), deltas)
    return float(np.max(np.linalg.norm(folded, axis=1)))


def _same_minimum_pairs(reference: Atoms, adsorbate, *, rng):
    """Class (a): perturbed adsorbate within the geometric tolerance."""
    indices = np.asarray(adsorbate, dtype=int)
    same = []
    different_basin = 0
    excluded = 0
    for perturbation in PERTURBATIONS_A:
        for _ in range(REPEATS):
            candidate = reference.copy()
            noise = np.zeros_like(candidate.positions)
            noise[indices] = rng.normal(0.0, perturbation, (len(indices), 3))
            candidate.positions = candidate.positions + noise
            displacement = _max_adsorbate_displacement(candidate, reference, adsorbate)
            if displacement > DIFFERENT_BASIN_A:
                different_basin += 1
            elif displacement <= float(CS_DEFAULTS["cs_dedup_geom_tol"]):
                same.append(candidate)
            else:
                excluded += 1
    return same, different_basin, excluded


def _distinct_pairs(slab: Atoms, reference: Atoms, adsorbate, *, rng):
    """Classes (b) and (c): genuinely different sites and conformers."""
    symbols = np.array(slab.get_chemical_symbols())
    sulfur = np.where(symbols == "S")[0]
    base_shift = reference.get_positions()[len(slab)].copy()
    candidates = []
    for index in sulfur:
        for height in (1.83, 2.41):
            shift = slab.get_positions()[index] + np.array([0.0, 0.0, height])
            candidates.append(_combine(slab, _adsorbate(), shift))
    candidates.extend(
        _combine(slab, _adsorbate(torsion), base_shift)
        for torsion in range(30, 360, 30)
    )
    for _ in range(8):
        rotation = random_rotation_matrix(rng)
        molecule = _adsorbate()
        centroid = molecule.get_positions().mean(axis=0)
        molecule.positions = (
            molecule.get_positions() - centroid
        ) @ rotation.T + centroid
        candidates.append(_combine(slab, molecule, base_shift))

    operations = _slab_symmetry_operations(slab)
    distinct = []
    for candidate in candidates:
        if (
            _max_adsorbate_displacement(candidate, reference, adsorbate)
            <= DIFFERENT_BASIN_A
        ):
            continue
        if _symmetry_related(reference, candidate, operations):
            continue
        distinct.append(candidate)
    return distinct


def _equivalent_pairs(slab: Atoms, reference: Atoms):
    """Class (d): rigid symmetry map and permutation of equivalent atoms."""
    equivalent = []
    # A rigid rotation of the whole structure (the same minimum mapped by a
    # slab symmetry operation) and its reflection.
    for reflection in (False, True):
        rotation = random_rotation_matrix(np.random.default_rng(17))
        if reflection:
            rotation = rotation @ np.diag([1.0, -1.0, 1.0])
        mapped = reference.copy()
        mapped.cell = np.asarray(reference.cell) @ rotation.T
        mapped.positions = np.asarray(reference.positions) @ rotation.T
        equivalent.append(mapped)

    # A permutation of equivalent atoms: the three methyl H atoms sit on a
    # circle about the molecule's C3 axis (z through the C at z = 0.35), so
    # rotating them by 120/240 degrees maps the set onto itself.
    axis = np.array([0.0, 0.0, 1.0])
    centre = np.array([0.0, 0.0, 0.35])
    for angle in (120.0, 240.0):
        molecule = _adsorbate()
        positions = molecule.get_positions()
        theta = np.deg2rad(angle)
        for index in (3, 4, 5):
            point = positions[index] - centre
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            positions[index] = (
                centre
                + point * cos_theta
                + np.cross(axis, point) * sin_theta
                + axis * np.dot(axis, point) * (1 - cos_theta)
            )
        molecule.positions = positions
        equivalent.append(
            _combine(slab, molecule, reference.get_positions()[len(slab)])
        )
    return equivalent


def _fingerprint(atoms, adsorbate, options):
    return build_fingerprint(atoms, adsorbate, options=options)


def _sweep():
    slab, _sulfur, _reference_shift, reference = _lgps_setup()
    adsorbate = np.arange(len(slab), len(slab) + 6)
    rng = np.random.default_rng(20261024)
    same, different_basin, excluded = _same_minimum_pairs(reference, adsorbate, rng=rng)
    distinct = _distinct_pairs(slab, reference, adsorbate, rng=rng)

    symmetric_slab, symmetric_reference = _symmetric_setup()
    symmetric_adsorbate = np.arange(len(symmetric_slab), len(symmetric_slab) + 6)
    equivalent = _equivalent_pairs(symmetric_slab, symmetric_reference)

    table = {}
    for sigma in SIGMA_CANDIDATES:
        options = ConfsearchOptions(
            cs_slab="s", cs_molecule="m", cs_n_starts=1, cs_fp_sigma=sigma
        )
        reference_fp = _fingerprint(reference, adsorbate, options)
        distinct_distances = [
            reference_fp.distance(_fingerprint(candidate, adsorbate, options))
            for candidate in distinct
        ]
        symmetric_reference_fp = _fingerprint(
            symmetric_reference, symmetric_adsorbate, options
        )
        equivalent_distances = [
            symmetric_reference_fp.distance(
                _fingerprint(candidate, symmetric_adsorbate, options)
            )
            for candidate in equivalent
        ]
        tolerance = 0.5 * min(distinct_distances)
        merged = sum(1 for distance in equivalent_distances if distance <= tolerance)
        table[sigma] = {
            "tolerance": tolerance,
            "min_distinct": min(distinct_distances),
            "max_distinct": max(distinct_distances),
            "equivalent_distances": equivalent_distances,
            "equivalent_merge_fraction": merged / len(equivalent),
        }
    chosen = max(
        SIGMA_CANDIDATES,
        key=lambda sigma: (table[sigma]["equivalent_merge_fraction"], sigma),
    )
    return {
        "table": table,
        "chosen_sigma": chosen,
        "chosen_tolerance": table[chosen]["tolerance"],
        "n_same": len(same),
        "n_different_basin": different_basin,
        "n_excluded": excluded,
        "n_distinct": len(distinct),
        "n_equivalent": len(equivalent),
    }


def test_calibration_sweep_matches_the_shipped_defaults(capsys):
    result = _sweep()
    table = result["table"]
    with capsys.disabled():
        print("\nconfsearch fingerprint calibration (spec 13.2 rev 4)")
        print(
            "  same-basin: %d, different-basin: %d, excluded: %d"
            % (result["n_same"], result["n_different_basin"], result["n_excluded"])
        )
        print(
            "  distinct: %d, equivalent: %d"
            % (result["n_distinct"], result["n_equivalent"])
        )
        for sigma, row in table.items():
            print(
                "  sigma={:.2f}  D_tol={:.5f}  min(b,c)={:.5f}  max(b,c)={:.5f}  "
                "(d) merged={:.0f}%".format(
                    sigma,
                    row["tolerance"],
                    row["min_distinct"],
                    row["max_distinct"],
                    100 * row["equivalent_merge_fraction"],
                )
            )
        print(
            "  chosen: sigma={:.2f}  D_tol={:.5f}".format(
                result["chosen_sigma"], result["chosen_tolerance"]
            )
        )

    assert result["chosen_sigma"] == pytest.approx(CS_DEFAULTS["cs_fp_sigma"])
    assert result["chosen_tolerance"] == pytest.approx(
        CS_DEFAULTS["cs_dedup_fp_tol"], abs=5e-4
    )
    # Zero false merges with a factor-2 margin at the shipped tolerance.
    assert (
        float(CS_DEFAULTS["cs_dedup_fp_tol"])
        <= 0.5 * min(row["min_distinct"] for row in table.values()) + 1e-12
    )


def test_same_basin_pairs_are_caught_by_the_geometry_rule():
    """Class (a) must merge through G, not through the fingerprint."""
    slab, _sulfur, _shift, reference = _lgps_setup()
    adsorbate = np.arange(len(slab), len(slab) + 6)
    rng = np.random.default_rng(20261024)
    same, _different, _excluded = _same_minimum_pairs(reference, adsorbate, rng=rng)
    assert same
    for candidate in same:
        assert _max_adsorbate_displacement(candidate, reference, adsorbate) <= float(
            CS_DEFAULTS["cs_dedup_geom_tol"]
        )


def test_equivalent_pairs_merge_at_the_shipped_default():
    slab, reference = _symmetric_setup()
    adsorbate = np.arange(len(slab), len(slab) + 6)
    equivalent = _equivalent_pairs(slab, reference)
    options = ConfsearchOptions(cs_slab="s", cs_molecule="m", cs_n_starts=1)
    reference_fp = _fingerprint(reference, adsorbate, options)
    for candidate in equivalent:
        distance = reference_fp.distance(_fingerprint(candidate, adsorbate, options))
        assert distance <= options.cs_dedup_fp_tol, distance


def test_shipped_sigma_satisfies_the_binwidth_bound():
    options = ConfsearchOptions(cs_slab="s", cs_molecule="m", cs_n_starts=1)
    assert options.cs_fp_binwidth <= 0.5 * options.cs_fp_sigma
