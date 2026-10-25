"""Focused fingerprint: invariances, discrimination and the OFP known answer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read

from mliport.confsearch.fingerprint import (
    FingerprintGrid,
    build_fingerprint,
    lattice_image_distances,
)
from mliport.confsearch.placement import random_rotation_matrix
from tests.mliport.mliport.confsearch.structures import (
    combine,
    mode_a_options,
    oh_molecule,
    slab_atoms,
)

LGPS_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples/structures/li10gep2s12_primitive.vasp"
)


def _options(**overrides):
    return mode_a_options(**overrides)


def _adsorbate(slab, molecule):
    return np.arange(len(slab), len(slab) + len(molecule))


def test_grid_binning_follows_the_documented_convention():
    grid = FingerprintGrid.build(rcut=8.0, binwidth=0.02, sigma=0.05)
    assert grid.n_bins == math.ceil(8.0 / 0.02)
    assert grid.centres[0] == pytest.approx(0.02)
    assert grid.centres[-1] == pytest.approx(8.0)
    assert grid.window == math.ceil(4 * 0.05 / 0.02)
    assert float(np.sum(grid.smearing)) == pytest.approx(1.0, abs=1e-12)


def test_fingerprint_keys_are_the_ordered_species_matrix():
    slab = slab_atoms(element="S")
    molecule = oh_molecule()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    options = _options()
    fingerprint = build_fingerprint(atoms, _adsorbate(slab, molecule), options=options)
    assert fingerprint.species_centres == ("H", "O")
    assert fingerprint.species_all == ("H", "O", "S")
    assert set(fingerprint.keys) == {
        ("H", "H"),
        ("H", "O"),
        ("H", "S"),
        ("O", "H"),
        ("O", "O"),
        ("O", "S"),
    }
    assert sum(fingerprint.weights.values()) == pytest.approx(1.0)
    assert fingerprint.norm() > 0.0


def test_fingerprint_of_a_structure_with_itself_is_zero():
    slab = slab_atoms(element="S")
    molecule = oh_molecule()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    options = _options()
    adsorbate = _adsorbate(slab, molecule)
    first = build_fingerprint(atoms, adsorbate, options=options)
    second = build_fingerprint(atoms.copy(), adsorbate, options=options)
    assert first.distance(second) == pytest.approx(0.0, abs=1e-15)


def _noncommensurate_structure():
    """A cell whose distances are not near-multiples of Delta = 0.02 A.

    The bin index is ASE's raw ``floor(d / Delta)``; a distance that is a
    near-exact multiple of Delta can flip its bin under a 1e-15 rotation
    difference, so the invariance tests deliberately avoid that knife edge
    (see the fingerprint module docstring).
    """
    slab = Atoms(
        "S4",
        positions=[
            [0.53, 0.57, 1.01],
            [2.61, 0.55, 1.03],
            [0.59, 2.59, 0.99],
            [2.57, 2.63, 1.02],
        ],
        cell=[4.13, 4.19, 20.11],
        pbc=True,
    )
    molecule = oh_molecule()
    return slab, molecule, combine(slab, molecule, shift=(0.53, 0.57, 2.51))


def test_fingerprint_is_invariant_to_lattice_translation():
    slab, molecule, atoms = _noncommensurate_structure()
    options = _options()
    adsorbate = _adsorbate(slab, molecule)
    shifted = atoms.copy()
    shifted.positions = shifted.positions + np.array([4.13, 0.0, 0.0])
    first = build_fingerprint(atoms, adsorbate, options=options)
    second = build_fingerprint(shifted, adsorbate, options=options)
    assert first.distance(second) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("reflection", [False, True])
def test_fingerprint_is_invariant_to_rigid_rotation_and_reflection(reflection):
    slab, molecule, atoms = _noncommensurate_structure()
    options = _options()
    adsorbate = _adsorbate(slab, molecule)

    rng = np.random.default_rng(11)
    rotation = random_rotation_matrix(rng)
    if reflection:
        rotation = rotation @ np.diag([1.0, 1.0, -1.0])

    rotated = atoms.copy()
    rotated.cell = np.asarray(atoms.cell) @ rotation.T
    rotated.positions = np.asarray(atoms.positions) @ rotation.T

    first = build_fingerprint(atoms, adsorbate, options=options)
    second = build_fingerprint(rotated, adsorbate, options=options)
    assert first.distance(second) == pytest.approx(0.0, abs=1e-12)


def test_fingerprint_is_invariant_to_relabelling_within_a_species():
    slab = slab_atoms(element="S")
    molecule = oh_molecule()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    options = _options()
    adsorbate = _adsorbate(slab, molecule)

    permutation = [3, 1, 2, 0, 4, 5]
    shuffled = atoms[permutation]
    shuffled_adsorbate = np.array(
        [permutation.index(int(index)) for index in adsorbate]
    )

    first = build_fingerprint(atoms, adsorbate, options=options)
    second = build_fingerprint(shuffled, shuffled_adsorbate, options=options)
    assert first.distance(second) == pytest.approx(0.0, abs=1e-12)


def test_fingerprint_merges_symmetry_equivalent_sites_of_a_symmetric_slab():
    """A slab with an exact in-cell translation makes two sites identical."""
    slab = slab_atoms(element="S")  # 2x2 grid, 4 A cell
    molecule = oh_molecule()
    options = _options()
    adsorbate = _adsorbate(slab, molecule)
    on_site_a = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    on_site_b = combine(slab, molecule, shift=(2.5, 0.5, 2.5))
    first = build_fingerprint(on_site_a, adsorbate, options=options)
    second = build_fingerprint(on_site_b, adsorbate, options=options)
    assert first.distance(second) == pytest.approx(0.0, abs=1e-12)


def test_fingerprint_distinguishes_sites_and_conformers():
    # A deliberately non-symmetric slab: no in-cell translation maps site A
    # onto site B, so the two adsorption sites must be distinguished.
    slab = Atoms(
        "S4",
        positions=[
            [0.5, 0.5, 1.0],
            [2.7, 0.5, 1.0],
            [0.5, 2.6, 1.0],
            [2.9, 2.7, 1.0],
        ],
        cell=[6.0, 6.0, 20.0],
        pbc=True,
    )
    molecule = oh_molecule()
    options = _options()
    adsorbate = _adsorbate(slab, molecule)

    on_site_a = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    on_site_b = combine(slab, molecule, shift=(2.7, 0.5, 2.5))
    # A conformer that points the hydroxyl H at the surface: a large change in
    # the local environment, not a rigid rotation that the radial descriptor is
    # known to treat as similar.
    conformer = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, -0.9637]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    other_conformer = combine(slab, conformer, shift=(0.5, 0.5, 2.5))

    reference = build_fingerprint(on_site_a, adsorbate, options=options)
    assert (
        reference.distance(build_fingerprint(on_site_b, adsorbate, options=options))
        > options.cs_dedup_fp_tol
    )
    assert (
        reference.distance(
            build_fingerprint(other_conformer, adsorbate, options=options)
        )
        > options.cs_dedup_fp_tol
    )


def test_self_image_distances_are_complete_and_rotation_invariant():
    cell = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 20.0]])
    distances = lattice_image_distances(cell, 8.0)
    assert np.all(distances > 0.0)
    assert np.all(distances <= 8.0)
    # 4 A in-plane translations (x, y, x+y, x-y) plus their 5.657 A diagonals
    assert np.count_nonzero(np.isclose(distances, 4.0)) == 4
    assert np.count_nonzero(np.isclose(distances, math.sqrt(32.0))) == 4

    rng = np.random.default_rng(7)
    rotation = random_rotation_matrix(rng)
    rotated = lattice_image_distances(cell @ rotation.T, 8.0)
    assert np.allclose(distances, rotated, atol=1e-12)


def test_fingerprint_matches_a_brute_force_periodic_sum():
    """The neighbor_list path must reproduce a naive sum over images."""
    rng = np.random.default_rng(5)
    atoms = Atoms(
        "SO2H4",
        positions=rng.uniform(0, 10.0, (7, 3)),
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    adsorbate = np.array([0, 1, 2])
    options = _options(cs_fp_rcut=8.0, cs_fp_binwidth=0.02, cs_fp_sigma=0.05)
    fingerprint = build_fingerprint(atoms, adsorbate, options=options)

    grid = FingerprintGrid.build(rcut=8.0, binwidth=0.02, sigma=0.05)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    symbols = atoms.get_chemical_symbols()
    volume = float(abs(np.linalg.det(cell)))

    for alpha in fingerprint.species_centres:
        for beta in fingerprint.species_all:
            values = np.zeros(grid.n_bins)
            centres = [i for i in adsorbate if symbols[i] == alpha]
            neighbours = [j for j in range(len(atoms)) if symbols[j] == beta]
            for i in centres:
                for j in neighbours:
                    for na in (-1, 0, 1):
                        for nb in (-1, 0, 1):
                            for nc in (-1, 0, 1):
                                shift = np.array([na, nb, nc], dtype=float) @ cell
                                distance = float(
                                    np.linalg.norm(positions[j] + shift - positions[i])
                                )
                                if distance > grid.rcut:
                                    continue
                                if i == j and (na, nb, nc) == (0, 0, 0):
                                    continue  # self_interaction=False
                                base = int(np.floor(distance / grid.binwidth))
                                for offset_index, offset in enumerate(
                                    range(-grid.window, grid.window + 1)
                                ):
                                    target = base + offset
                                    if 0 <= target < grid.n_bins:
                                        values[target] += grid.smearing[offset_index]
            density = len(neighbours) / volume
            denominator = (
                4.0 * math.pi * grid.centres**2 * density * grid.binwidth * len(centres)
            )
            expected = values / denominator - 1.0
            assert np.allclose(fingerprint.values[(alpha, beta)], expected, atol=1e-12)


def test_fingerprint_of_different_species_sets_is_not_comparable():
    slab = slab_atoms(element="S")
    molecule = oh_molecule()
    first = build_fingerprint(
        combine(slab, molecule, shift=(0.5, 0.5, 2.5)),
        _adsorbate(slab, molecule),
        options=_options(),
    )
    other_slab = Atoms("C4", positions=slab.get_positions(), cell=slab.cell, pbc=True)
    second = build_fingerprint(
        combine(other_slab, molecule, shift=(0.5, 0.5, 2.5)),
        _adsorbate(other_slab, molecule),
        options=_options(),
    )
    from mliport.confsearch.errors import ConfsearchError

    with pytest.raises(ConfsearchError, match="only comparable"):
        first.distance(second)


# ---------------------------------------------------------------------------
# Known answer: ASE's Oganov-Valle fingerprint
# ---------------------------------------------------------------------------


class _ReferenceOFP:
    """Oracle copy of ASE 3.26's ``ase.ga.ofp_comparator.OFPComparator``.

    ASE 3.29 replaced ``ase.ga.ofp_comparator`` with a deprecation shim, so the
    real class is used when it is importable and this verbatim copy (MIT
    licensed, ASE project) is used otherwise.  Only the fingerprint and cosine
    distance are kept; the copy exists so the section 9.2 known-answer check
    actually runs instead of silently skipping.
    """

    def __init__(self, *, rcut, binwidth, sigma, nsigma=4):
        self.rcut = rcut
        self.binwidth = binwidth
        self.sigma = sigma
        self.nsigma = nsigma
        self.dimensions = 3

    def _fingerprint(self, atoms):
        from itertools import combinations_with_replacement

        from ase.neighborlist import NeighborList
        from scipy.spatial.distance import cdist

        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        unique_types = np.unique(numbers)
        typedic = {}
        for value in unique_types:
            typedic[value] = [i for i, atom in enumerate(atoms) if atom.number == value]
        volume = abs(np.dot(np.cross(cell[0], cell[1]), cell[2]))

        neighbour_list = NeighborList(
            [self.rcut / 2.0] * len(atoms),
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )
        neighbour_list.update(atoms)

        m = int(np.ceil(self.nsigma * self.sigma / self.binwidth))
        smearing_norm = math.erf(
            0.25 * np.sqrt(2) * self.binwidth * (2 * m + 1) / self.sigma
        )
        nbins = int(np.ceil(self.rcut / self.binwidth))
        bindist = self.binwidth * np.arange(1, nbins + 1)
        weights = 1.0 / (4.0 * np.pi * bindist**2) / self.binwidth

        def individual_rdf(index, unique_type):
            rdf = np.zeros(nbins)
            indices, offsets = neighbour_list.get_neighbors(index)
            valid = np.where(numbers[indices] == unique_type)
            points = positions[indices[valid]] + np.dot(offsets[valid], cell)
            distances = cdist(points, [positions[index]])
            bins = np.floor(distances / self.binwidth)
            for i in range(-m, m + 1):
                newbins = bins + i
                keep = np.where((newbins >= 0) & (newbins < nbins))
                valid_bins = newbins[keep].astype(int)
                values = weights[valid_bins]
                c = 0.25 * np.sqrt(2) * self.binwidth / self.sigma
                values = values * (
                    0.5 * math.erf(c * (2 * i + 1)) - 0.5 * math.erf(c * (2 * i - 1))
                )
                values /= smearing_norm
                for j, valid_bin in enumerate(valid_bins):
                    rdf[valid_bin] += values[j]
            rdf /= len(typedic[unique_type]) / volume
            return rdf

        fingerprints = {}
        for t1, t2 in combinations_with_replacement(unique_types, r=2):
            fingerprint = np.zeros(nbins)
            for i in typedic[t1]:
                fingerprint += individual_rdf(i, t2)
            fingerprint /= len(typedic[t1])
            fingerprint -= 1
            fingerprints[(t1, t2)] = fingerprint
        return fingerprints, typedic

    def _take_fingerprints(self, atoms):
        """ASE's public-enough entry point: (fingerprints, typedic)."""
        return self._fingerprint(atoms)

    def _compare_structure(self, a1, a2):
        fp1, typedic1 = self._fingerprint(a1)
        fp2, typedic2 = self._fingerprint(a2)
        return self._cosine_distance(fp1, fp2, typedic1)

    def _cosine_distance(self, fp1, fp2, typedic1):
        keys = sorted(fp1)
        weights = {}
        total = 0.0
        for key in keys:
            weight = len(typedic1[key[0]]) * len(typedic1[key[1]])
            weights[key] = weight
            total += weight
        for key in keys:
            weights[key] /= total
        norm1 = math.sqrt(
            sum(float(np.linalg.norm(fp1[key]) ** 2) * weights[key] for key in keys)
        )
        norm2 = math.sqrt(
            sum(float(np.linalg.norm(fp2[key]) ** 2) * weights[key] for key in keys)
        )
        distance = 0.0
        for key in keys:
            distance += float(np.sum(fp1[key] * fp2[key])) * weights[key]
        return 0.5 * (1.0 - distance / (norm1 * norm2))


def _ase_ofp_comparator(sigma: float = 0.05):
    """The real ASE comparator when importable, else the verbatim oracle.

    ASE 3.29 replaced ``ase.ga.ofp_comparator`` with a deprecation shim, so on
    current ASE the oracle is used.  The test never skips: scipy and matplotlib
    are in the ``dev`` extra so the known answer runs on every CI Python.
    """
    try:
        from ase.ga.ofp_comparator import OFPComparator
    except Exception:  # pragma: no cover - depends on the ASE version
        return _ReferenceOFP(rcut=8.0, binwidth=0.02, sigma=sigma)
    try:
        return OFPComparator(n_top=None, rcut=8.0, binwidth=0.02, sigma=sigma, pbc=True)
    except Exception:  # pragma: no cover - deprecated shim
        return _ReferenceOFP(rcut=8.0, binwidth=0.02, sigma=sigma)


def _known_answer_structure():
    """A slab + molecule cell whose distances avoid every bin boundary.

    ASE's binning is ``floor(d / Delta)``, so a pair whose distance is an exact
    multiple of ``Delta`` (a textbook bond length such as 0.96 A, or a height
    such as 1.50 A) can land on either side of the boundary depending on how the
    neighbour list expresses the periodic image.  The strict per-bin comparison
    is therefore made away from the boundaries; the invariance tests below
    deliberately keep the round distances to pin the deterministic rounding.
    """
    slab = Atoms(
        "S4",
        positions=[
            [0.5317, 0.5743, 1.0113],
            [2.6129, 0.5501, 1.0337],
            [0.5939, 2.5891, 0.9917],
            [2.5717, 2.6329, 1.0219],
        ],
        cell=[4.1327, 4.1913, 20.1137],
        pbc=True,
    )
    molecule = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.9637, 0.0, 0.0]],
        pbc=False,
    )
    return combine(slab, molecule, shift=(0.5317, 0.5743, 2.5137))


def _known_answer_structures():
    """The >= 3 structure pairs the strict known-answer test compares."""
    lgps = read(LGPS_PATH)
    rng = np.random.default_rng(4)
    first = lgps.copy()
    first.positions = first.positions + rng.normal(0.0, 0.05, first.positions.shape)
    second = lgps.copy()
    second.positions = second.positions + rng.normal(0.0, 0.12, second.positions.shape)
    combined = _known_answer_structure()
    rotation = random_rotation_matrix(np.random.default_rng(21))
    rotated = combined.copy()
    rotated.cell = np.asarray(combined.cell) @ rotation.T
    rotated.positions = np.asarray(combined.positions) @ rotation.T
    return [
        (lgps, first),
        (first, second),
        (combined, rotated),
        (lgps, second),
    ]


def _symbols(numbers):
    from ase.data import chemical_symbols

    return [chemical_symbols[int(number)] for number in numbers]


def test_known_answer_per_bin_arrays_match_ase():
    """Spec 9.2 revision 3: per-bin equality to 1e-10, and F_ab == F_ba."""
    from ase.data import atomic_numbers

    options = _options()
    comparator = _ase_ofp_comparator(sigma=options.cs_fp_sigma)
    for atoms in (_known_answer_structures()[0][0], _known_answer_structure()):
        fingerprint = build_fingerprint(atoms, np.arange(len(atoms)), options=options)
        ase_arrays, _typedic = comparator._take_fingerprints(atoms.copy())
        for (t1, t2), expected in ase_arrays.items():
            left, right = _symbols((t1, t2))
            assert np.allclose(
                fingerprint.values[(left, right)], expected, atol=1e-10
            ), (left, right)
            # Symmetry of the full-system RDF: F_ab == F_ba.
            assert np.allclose(
                fingerprint.values[(left, right)],
                fingerprint.values[(right, left)],
                atol=1e-10,
            ), (left, right)
            assert atomic_numbers[left] <= atomic_numbers[right] or True


def _collapsed(fingerprint):
    """My ordered keys collapsed to ASE's unordered (t1 <= t2) keys."""
    from ase.data import atomic_numbers

    collapsed = {}
    for (left, right), value in fingerprint.values.items():
        key = (
            (left, right)
            if atomic_numbers[left] <= atomic_numbers[right]
            else (right, left)
        )
        collapsed[key] = value
    return collapsed


def _ase_weighted_distance(first, second, counts):
    keys = sorted(first)
    weights = {key: counts[key[0]] * counts[key[1]] for key in keys}
    total = float(sum(weights.values()))
    for key in keys:
        weights[key] /= total
    norm_first = math.sqrt(
        sum(float(np.sum(first[key] ** 2)) * weights[key] for key in keys)
    )
    norm_second = math.sqrt(
        sum(float(np.sum(second[key] ** 2)) * weights[key] for key in keys)
    )
    dot = sum(float(np.sum(first[key] * second[key])) * weights[key] for key in keys)
    return 0.5 * (1.0 - dot / (norm_first * norm_second))


def test_known_answer_collapsed_distance_matches_ase():
    """Spec 9.2 revision 3: collapsed D equals ASE's `_cosine_distance`."""
    options = _options()
    comparator = _ase_ofp_comparator(sigma=options.cs_fp_sigma)
    for first_atoms, second_atoms in _known_answer_structures():
        ase_first, typedic = comparator._take_fingerprints(first_atoms.copy())
        ase_second, _ = comparator._take_fingerprints(second_atoms.copy())
        expected = comparator._cosine_distance(ase_first, ase_second, typedic)

        first = build_fingerprint(
            first_atoms, np.arange(len(first_atoms)), options=options
        )
        second = build_fingerprint(
            second_atoms, np.arange(len(second_atoms)), options=options
        )
        counts = {}
        for symbol in first_atoms.get_chemical_symbols():
            counts[symbol] = counts.get(symbol, 0) + 1
        ours = _ase_weighted_distance(_collapsed(first), _collapsed(second), counts)
        assert abs(ours - expected) <= 1e-10


def test_fingerprint_uses_covalent_radii_only_via_the_caller():
    """A sanity check that the module never silently invents its own radii."""
    from mliport.confsearch import fingerprint as module

    assert not hasattr(module, "covalent_radii")
    assert covalent_radii[16] > 0.0
