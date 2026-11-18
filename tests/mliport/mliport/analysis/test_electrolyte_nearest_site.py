"""Known-answer tests for the GEMDAT nearest-site cross-check.

``_nearest_site_change_count`` decides whether an empty GEMDAT transition
collection may be reported as a physical zero.  It therefore has to compare
mobile atoms and sites in **one** coordinate frame, using the true minimum
image of the lattice (not a per-component rounding, which is only correct for
orthogonal cells).

These tests use real ``pymatgen`` structures: a pymatgen ``PeriodicSite.coords``
is Cartesian while fractional coordinates live in ``.frac_coords``, so a
site/frame frame mix-up stays invisible to a hand-rolled fake that stores one
array and returns it from every accessor.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mliport.analysis import electrolyte
from mliport.analysis.dataset import TrajectoryDataset

pymatgen_core = pytest.importorskip(
    "pymatgen.core", reason="pymatgen is provided by mliport[electrolyte]"
)
Structure = pymatgen_core.Structure

LI = "Li"
#: Tetragonal Na3PS4 cell parameters (a = b = 6.952 A, c = 7.076 A). Geometry
#: only: the sites below are synthetic, not a DFT-relaxed structure.
TETRAGONAL_NA3PS4_CELL = np.diag([6.952, 6.952, 7.076])
#: Strongly skewed but valid cell. A cell with 60 deg < gamma < 90 deg is where
#: component-wise wrapping stops being the minimum image.
SKEWED_CELL = np.array([[10.0, 0.0, 0.0], [9.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
SPECTATOR = [8.0, 8.0, 8.0]


def _frames(positions, cell, symbols=("Li", "S")):
    positions = np.asarray(positions, dtype=float).reshape(-1, len(symbols), 3)
    return [Atoms(symbols, positions=frame, cell=cell, pbc=True) for frame in positions]


def _sites(cell, coords, species=None):
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    return Structure(
        lattice=np.asarray(cell, dtype=float),
        species=species or [LI] * len(coords),
        coords=coords,
        coords_are_cartesian=True,
    )


def _count(positions, cell, sites, symbols=("Li", "S")):
    """Run the cross-check on unwrapped production frames with a spectator atom."""
    positions = np.asarray(positions, dtype=float).reshape(-1, len(symbols), 3)
    frames = _frames(positions, cell, symbols=symbols)
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(len(frames)) * 100.0,
        positions_convention="unwrapped",
        phases=["production"] * len(frames),
    )
    view = dataset.analysis_view(include_equilibration=False)
    return electrolyte._nearest_site_change_count(
        view=view,
        corrected_positions_A=positions,
        mobile=view.select(LI),
        sites=sites,
    )


def _li(*xs, y=1.0, z=1.0):
    """Two-frame Li path plus a fixed spectator, shaped (2, 2, 3)."""

    return np.array([[list(path), SPECTATOR] for path in xs])


def test_pymatgen_site_coords_are_cartesian():
    """Pin the documented pymatgen surface this cross-check depends on."""

    sites = _sites(np.diag([10.0, 10.0, 10.0]), [(2.0, 3.0, 4.0)])
    np.testing.assert_allclose(sites[0].coords, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(sites[0].frac_coords, [0.2, 0.3, 0.4])


def test_cubic_hop_between_two_sites_is_counted():
    cell = np.diag([10.0, 10.0, 10.0])
    sites = _sites(cell, [(1.0, 1.0, 1.0), (5.0, 1.0, 1.0)])
    positions = _li([1.0, 1.0, 1.0], [5.0, 1.0, 1.0])
    assert _count(positions, cell, sites) == 1


def test_no_transition_is_zero_changes():
    cell = np.diag([10.0, 10.0, 10.0])
    sites = _sites(cell, [(1.0, 1.0, 1.0), (5.0, 1.0, 1.0)])
    positions = _li([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert _count(positions, cell, sites) == 0


def test_boundary_crossing_hop_is_counted():
    """A hop through the cell boundary is still a site change."""

    cell = np.diag([10.0, 10.0, 10.0])
    sites = _sites(cell, [(9.5, 5.0, 5.0), (0.5, 5.0, 5.0)])
    # Unwrapped x: 9.5 -> 10.2 (= 0.2 after wrapping); 3.0 A from 9.5, 0.3 A
    # from 0.5, so the nearest site label must flip.
    positions = _li([9.5, 5.0, 5.0], [10.2, 5.0, 5.0])
    assert _count(positions, cell, sites) == 1


def test_lattice_image_near_the_same_site_is_not_a_transition():
    """A boundary crossing must not be mistaken for a change of site."""

    cell = np.diag([10.0, 10.0, 10.0])
    sites = _sites(cell, [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0)])
    # 9.9 A is 0.17 A from the origin site (through the boundary) but 8.5 A
    # from the second site: the atom never leaves its site.
    positions = _li([0.2, 0.2, 0.2], [9.9, 9.9, 9.9])
    assert _count(positions, cell, sites) == 0


def test_tetragonal_non_cubic_cell():
    """Na3PS4-like tetragonal cell: no hop, then one hop."""

    cell = TETRAGONAL_NA3PS4_CELL
    sites = _sites(cell, [(1.0, 1.0, 1.0), (5.0, 2.0, 5.5)])
    staying = np.array(
        [[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]]
    )
    hopping = np.array(
        [[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], [[5.0, 2.0, 5.5], [3.0, 3.0, 3.0]]]
    )
    assert _count(staying, cell, sites) == 0
    assert _count(hopping, cell, sites) == 1


def test_skewed_cell_uses_the_true_minimum_image():
    """Component-wise wrapping picks the wrong site in a skewed cell.

    With a = (10, 0, 0) and b = (9, 10, 0), the position (9, 5, 0) is at
    fractional (0.45, 0.5, 0).  Rounding each component keeps it 10.3 A from
    site 0 (and 6.4 A from site 1), but the true minimum image is 5.1 A from
    site 0 -- so the atom never leaves site 0.
    """

    cell = SKEWED_CELL
    sites = _sites(cell, [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    positions = np.array(
        [
            [[0.5, 0.5, 0.0], [3.0, 8.0, 3.0]],
            [[9.0, 5.0, 0.0], [3.0, 8.0, 3.0]],
        ]
    )
    assert _count(positions, cell, sites) == 0


def test_ambiguous_nearest_site_is_reported_as_unknown():
    """Equidistant sites must not be turned into a confident 'no change'."""

    cell = np.diag([10.0, 10.0, 10.0])
    sites = _sites(cell, [(0.0, 5.0, 5.0), (5.0, 5.0, 5.0)])
    positions = _li([2.5, 5.0, 5.0], [2.5, 5.0, 5.0])
    assert _count(positions, cell, sites) is None


def test_non_finite_positions_are_not_a_physical_zero():
    cell = np.diag([10.0, 10.0, 10.0])
    sites = _sites(cell, [(1.0, 1.0, 1.0)])
    positions = _li([np.nan, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert _count(positions, cell, sites) is None
