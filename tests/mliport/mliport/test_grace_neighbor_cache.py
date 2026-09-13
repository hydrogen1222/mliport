"""Unit tests for the GRACE verlet-style neighbor cache (_NeighborListCache).

These tests run without TensorFlow/GPU: they drive the cache with a fake
``GeometricalDataBuilder``-like object whose ``extract_from_ase_atoms`` calls
``matscipy.neighbours.neighbour_list`` on small periodic cells.

Key invariants under test:
  * the cached periodic-image multiset equals a fresh exact-cutoff search;
  * rebuild triggers (species/cell/PBC change, displacement > skin/2);
  * triclinic MICs, diagonal displacements, and repeated images are retained;
  * non/partial-PBC structures use the unmodified upstream path;
  * zero-neighbor atoms keep fictitious neighbours.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

# These tests exercise the real tensorpotential geometry-data contract via
# matscipy; skip them (instead of erroring) in environments without those
# optional GRACE dependencies. They run in the dedicated GRACE venv.
pytest.importorskip("matscipy")
pytest.importorskip("tensorpotential")

from matscipy.neighbours import neighbour_list  # noqa: E402

from mliport.calculators.grace_calc import _NeighborListCache  # noqa: E402


class FakeBuilder:
    """Mimics tensorpotential GeometricalDataBuilder for cache tests."""

    def __init__(self, symbols, cutoff_dict=None):
        self.cutoff = 5.0
        self.max_cutoff = 5.0
        self.cutoff_dict = cutoff_dict
        self.float_dtype = np.float64
        self.elements_map = {s: i for i, s in enumerate(sorted(set(symbols)))}

    def extract_from_ase_atoms(self, ase_atoms, **kwarg):
        from tensorpotential import constants

        symbols = ase_atoms.get_chemical_symbols()
        if self.cutoff_dict is not None:
            from itertools import combinations_with_replacement

            cutoff = {}
            for e1, e2 in combinations_with_replacement(set(symbols), 2):
                cutoff[(e1, e2)] = self.cutoff_dict.get((e1, e2), self.cutoff)
        else:
            cutoff = self.cutoff
        ind_i, ind_j, bond_vector = neighbour_list("ijD", ase_atoms, cutoff=cutoff)
        # Fictitious neighbours for zero-neighbor atoms (same rule as the
        # real tensorpotential extractor).
        all_atom_ind = np.arange(len(ase_atoms))
        if np.unique(ind_i).shape[0] < all_atom_ind.shape[0]:
            missing_ind = all_atom_ind[~np.isin(all_atom_ind, np.unique(ind_i))]
            ind_j_to_add = np.zeros(len(missing_ind)).astype(int)
            dv_j_to_add = (
                np.dot(
                    ase_atoms.cell,
                    np.array([[1, 1, 1] for _ in missing_ind]).reshape(3, -1),
                ).reshape(-1, 3)
                + self.max_cutoff
            )
            ind_i = np.append(ind_i, missing_ind)
            ind_j = np.append(ind_j, ind_j_to_add)
            bond_vector = np.append(bond_vector, dv_j_to_add, axis=0)
            sort = np.argsort(ind_i)
            ind_i = ind_i[sort]
            ind_j = ind_j[sort]
            bond_vector = bond_vector[sort]
        atomic_mu_i = np.array([self.elements_map[s] for s in symbols], dtype=np.int32)
        mu_i = np.array([atomic_mu_i[i] for i in ind_i], dtype=np.int32)
        mu_j = np.array([atomic_mu_i[j] for j in ind_j], dtype=np.int32)
        return {
            constants.ATOMIC_MU_I: atomic_mu_i,
            constants.BOND_VECTOR: bond_vector.astype(self.float_dtype),
            constants.BOND_MU_I: mu_i,
            constants.BOND_MU_J: mu_j,
            constants.BOND_IND_I: ind_i.astype(np.int32),
            constants.BOND_IND_J: ind_j.astype(np.int32),
            constants.N_ATOMS_BATCH_REAL: np.array(len(ase_atoms), dtype=np.int32),
            constants.N_STRUCTURES_BATCH_REAL: np.array(1, dtype=np.int32),
            constants.N_NEIGHBORS_REAL: np.array(len(ind_i), dtype=np.int32),
        }


def _make_atoms():
    # 3x3x3 simple-cubic lattice (spacing 4 A) with a small rattle: every
    # neighbour pair stays far from periodic-boundary/bin boundaries, so
    # matscipy's shift choice is independent of the (extended) cutoff and the
    # cached neighbour SET must equal a fresh exact-cutoff search.
    rng = np.random.default_rng(0)
    cell = np.eye(3) * 20.0
    pos = np.array(
        [
            [ix * 4.0, iy * 4.0, iz * 4.0]
            for ix in range(3)
            for iy in range(3)
            for iz in range(3)
        ]
    )
    pos = pos + rng.normal(0, 0.05, pos.shape)
    return Atoms("Li9Ge9S9", positions=pos, cell=cell, pbc=True)


def _data(calc, atoms):
    return calc(atoms)


def _sorted_images(data):
    from tensorpotential import constants

    ind_i = np.asarray(data[constants.BOND_IND_I], dtype=np.int32)
    ind_j = np.asarray(data[constants.BOND_IND_J], dtype=np.int32)
    vectors = np.asarray(data[constants.BOND_VECTOR], dtype=float)
    # Rounded coordinates are used for ordering only.  The actual vectors are
    # compared below at tight float64 tolerance.
    order = np.lexsort(
        (
            np.round(vectors[:, 2], 10),
            np.round(vectors[:, 1], 10),
            np.round(vectors[:, 0], 10),
            ind_j,
            ind_i,
        )
    )
    return ind_i[order], ind_j[order], vectors[order]


def _assert_same_model_input(cached, exact, *, atol=2e-12):
    from tensorpotential import constants

    np.testing.assert_array_equal(
        cached[constants.ATOMIC_MU_I], exact[constants.ATOMIC_MU_I]
    )
    ci, cj, cv = _sorted_images(cached)
    ei, ej, ev = _sorted_images(exact)
    np.testing.assert_array_equal(ci, ei)
    np.testing.assert_array_equal(cj, ej)
    np.testing.assert_allclose(cv, ev, rtol=0.0, atol=atol)


def test_cache_matches_fresh_search_within_skin():
    builder = FakeBuilder(["Li", "Ge", "S"])
    cache = _NeighborListCache(builder, skin=1.0)
    atoms = _make_atoms()
    exact = _data(builder.extract_from_ase_atoms, atoms)
    cached = _data(cache, atoms)
    _assert_same_model_input(cached, exact)  # rebuild frame

    rng = np.random.default_rng(1)
    for _ in range(5):
        atoms.positions += rng.normal(0, 0.08, atoms.positions.shape)  # < skin/2
        exact = _data(builder.extract_from_ase_atoms, atoms)
        cached = _data(cache, atoms)
        _assert_same_model_input(cached, exact)


def test_rebuild_on_large_displacement():
    builder = FakeBuilder(["Li", "Ge", "S"])
    cache = _NeighborListCache(builder, skin=1.0)
    atoms = _make_atoms()
    cache(atoms)
    n_rebuild = [0]
    orig = cache._build_extended

    def counted(a):
        n_rebuild[0] += 1
        return orig(a)

    cache._build_extended = counted
    atoms.positions += 0.1  # 0.1 < skin/2 -> no rebuild
    cache(atoms)
    assert n_rebuild[0] == 0
    atoms.positions += 1.0  # cumulative 1.1 > skin/2 -> rebuild
    cache(atoms)
    assert n_rebuild[0] == 1


def test_rebuild_on_species_change():
    builder = FakeBuilder(["Li", "Ge", "S"])
    cache = _NeighborListCache(builder, skin=1.0)
    atoms = _make_atoms()
    cache(atoms)
    n_rebuild = [0]
    orig = cache._build_extended

    def counted(a):
        n_rebuild[0] += 1
        return orig(a)

    cache._build_extended = counted
    atoms.numbers[0] = 16  # Li -> S: species change
    cache(atoms)
    assert n_rebuild[0] == 1


def test_rebuild_on_atom_count_change():
    builder = FakeBuilder(["Li", "Ge", "S"])
    cache = _NeighborListCache(builder, skin=1.0)
    atoms = _make_atoms()
    cache(atoms)
    atoms = atoms[:-1]
    cached = cache(atoms)
    exact = builder.extract_from_ase_atoms(atoms)
    _assert_same_model_input(cached, exact)
    assert len(cache._ref_numbers) == len(atoms)


def test_rebuild_on_cell_change():
    builder = FakeBuilder(["Li", "Ge", "S"])
    cache = _NeighborListCache(builder, skin=1.0)
    atoms = _make_atoms()
    cache(atoms)
    n_rebuild = [0]
    orig = cache._build_extended

    def counted(a):
        n_rebuild[0] += 1
        return orig(a)

    cache._build_extended = counted
    atoms.set_cell(np.eye(3) * 8.5, scale_atoms=False)
    cache(atoms)
    assert n_rebuild[0] == 1


def test_diagonal_displacement_uses_euclidean_norm_and_rebuilds():
    """Regression: component-wise max misses a diagonal skin violation."""
    builder = FakeBuilder(["Li"])
    cache = _NeighborListCache(builder, skin=1.0)
    atoms = Atoms(
        "Li2",
        positions=[[0.0, 0.0, 0.0], [4.25, 4.25, 0.0]],
        cell=np.eye(3) * 30.0,
        pbc=True,
    )
    cache(atoms)
    n_rebuild = [0]
    orig = cache._build_extended

    def counted(a):
        n_rebuild[0] += 1
        return orig(a)

    cache._build_extended = counted
    atoms.positions[0] += [0.4, 0.4, 0.0]
    atoms.positions[1] -= [0.4, 0.4, 0.0]
    cached = cache(atoms)
    exact = builder.extract_from_ase_atoms(atoms)
    assert n_rebuild[0] == 1
    _assert_same_model_input(cached, exact)
    from tensorpotential import constants

    pairs = set(
        zip(
            cached[constants.BOND_IND_I].tolist(),
            cached[constants.BOND_IND_J].tolist(),
            strict=True,
        )
    )
    assert (0, 1) in pairs
    assert (1, 0) in pairs


def test_triclinic_cell_reuse_matches_fresh_search_without_rebuild():
    builder = FakeBuilder(["Li", "S"])
    cache = _NeighborListCache(builder, skin=1.2)
    atoms = Atoms(
        "Li2S2",
        scaled_positions=[
            [0.08, 0.12, 0.20],
            [0.42, 0.18, 0.27],
            [0.71, 0.73, 0.62],
            [0.94, 0.89, 0.84],
        ],
        cell=[[8.0, 0.0, 0.0], [2.1, 7.4, 0.0], [1.2, 0.7, 6.8]],
        pbc=True,
    )
    cache(atoms)
    n_rebuild = [0]
    orig = cache._build_extended

    def counted(a):
        n_rebuild[0] += 1
        return orig(a)

    cache._build_extended = counted
    atoms.positions += np.array(
        [
            [0.12, -0.07, 0.09],
            [-0.10, 0.05, 0.03],
            [0.08, 0.04, -0.11],
            [0.0, -0.09, 0.07],
        ]
    )
    cached = cache(atoms)
    exact = builder.extract_from_ase_atoms(atoms)
    assert n_rebuild[0] == 0
    _assert_same_model_input(cached, exact)


def test_pair_specific_cutoff_is_restored_after_extended_search():
    cutoff_dict = {("Li", "S"): 3.0, ("S", "Li"): 3.0}
    builder = FakeBuilder(["Li", "S"], cutoff_dict=cutoff_dict)
    cache = _NeighborListCache(builder, skin=2.0)
    atoms = Atoms(
        "LiS",
        positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    cached = cache(atoms)
    exact = builder.extract_from_ase_atoms(atoms)
    _assert_same_model_input(cached, exact)


def test_periodic_wrap_uses_mic_displacement_without_rebuild():
    builder = FakeBuilder(["Li"])
    cache = _NeighborListCache(builder, skin=1.0)
    atoms = Atoms(
        "Li2",
        positions=[[0.1, 1.0, 1.0], [4.0, 1.0, 1.0]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    cache(atoms)
    n_rebuild = [0]
    orig = cache._build_extended

    def counted(a):
        n_rebuild[0] += 1
        return orig(a)

    cache._build_extended = counted
    atoms.positions[0, 0] = 9.9  # MIC displacement is -0.2 A.
    cached = cache(atoms)
    exact = builder.extract_from_ase_atoms(atoms)
    assert n_rebuild[0] == 0
    _assert_same_model_input(cached, exact)


def test_repeated_periodic_images_are_not_collapsed():
    builder = FakeBuilder(["Li"])
    cache = _NeighborListCache(builder, skin=0.8)
    atoms = Atoms(
        "Li2",
        positions=[[0.2, 0.3, 0.4], [2.1, 1.8, 1.4]],
        cell=[[4.0, 0.0, 0.0], [0.8, 3.7, 0.0], [0.4, 0.5, 3.5]],
        pbc=True,
    )
    cache(atoms)
    atoms.positions += [[0.08, -0.04, 0.03], [-0.06, 0.05, -0.02]]
    cached = cache(atoms)
    exact = builder.extract_from_ase_atoms(atoms)
    _assert_same_model_input(cached, exact)
    from tensorpotential import constants

    pairs = list(
        zip(
            cached[constants.BOND_IND_I].tolist(),
            cached[constants.BOND_IND_J].tolist(),
            strict=True,
        )
    )
    assert len(pairs) > len(set(pairs))


def test_nonperiodic_fallback_preserves_callers_pbc():
    class MutatingBuilder(FakeBuilder):
        def extract_from_ase_atoms(self, ase_atoms, **kwarg):
            ase_atoms.set_pbc(True)
            return super().extract_from_ase_atoms(ase_atoms, **kwarg)

    builder = MutatingBuilder(["Li"])
    cache = _NeighborListCache(builder, skin=1.0)
    atoms = Atoms(
        "Li2", positions=[[0, 0, 0], [2, 0, 0]], cell=np.eye(3) * 20, pbc=False
    )
    cached = cache(atoms)
    assert not atoms.pbc.any()
    assert cache._ref_positions is None
    from tensorpotential import constants

    assert int(cached[constants.N_STRUCTURES_BATCH_REAL]) == 1


def test_zero_neighbor_keeps_fictitious():
    builder = FakeBuilder(["Li", "Ge", "S"])
    cache = _NeighborListCache(builder, skin=1.0)
    # Two atoms very far apart in a huge cell: no neighbors within cutoff.
    cell = np.eye(3) * 40.0
    atoms = Atoms("Li2", positions=[[0, 0, 0], [20, 20, 20]], cell=cell, pbc=True)
    data = cache(atoms)
    from tensorpotential import constants

    ind_i = data[constants.BOND_IND_I]
    # Every atom must appear as a neighbor source (fictitious kept).
    assert set(ind_i.tolist()) == {0, 1}
    assert len(ind_i) >= 2


def test_skin_validation():
    builder = FakeBuilder(["Li", "Ge", "S"])
    with pytest.raises(ValueError):
        _NeighborListCache(builder, skin=0.0)
    with pytest.raises(ValueError):
        _NeighborListCache(builder, skin=-1.0)
