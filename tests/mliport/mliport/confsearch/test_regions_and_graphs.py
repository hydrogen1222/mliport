"""Regions, surface validation, covalent graphs and status rules (spec section 5, 8)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixScaled

from mliport.confsearch.errors import (
    ConfsearchInternalError,
    ConfsearchPreparationError,
)
from mliport.confsearch.geometry import periodic_gap_and_contiguity
from mliport.confsearch.graphs import (
    broken_reference_bonds,
    classify_minimum,
    covalent_bonds,
)
from mliport.confsearch.regions import (
    mobile_indices,
    resolve_regions,
    validate_all,
    validate_surface,
)
from tests.mliport.mliport.confsearch.structures import (
    adsorbate_indices,
    combine,
    mode_a_options,
    oh_molecule,
    reference_for,
    slab_atoms,
)

# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------


def test_regions_split_adsorbate_slab_fixed_and_relax():
    slab = slab_atoms()
    molecule = oh_molecule()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    options = mode_a_options()
    regions = resolve_regions(
        atoms, options, adsorbate_indices=adsorbate_indices(slab, molecule)
    )

    assert regions.adsorbate.tolist() == [4, 5]
    assert regions.slab.tolist() == [0, 1, 2, 3]
    assert regions.fixed.tolist() == [0, 1]
    assert regions.relax.tolist() == [2, 3, 4, 5]
    assert regions.warnings == ()


def test_regions_reject_partial_selective_dynamics():
    slab = slab_atoms()
    atoms = combine(slab, oh_molecule())
    atoms.set_constraint(FixScaled([0], mask=[True, False, False]))
    options = mode_a_options()
    with pytest.raises(ConfsearchPreparationError, match="only FixAtoms"):
        resolve_regions(atoms, options, adsorbate_indices=np.array([4, 5]))


def test_regions_reject_fixed_adsorbate_overlap():
    slab = slab_atoms()
    atoms = combine(slab, oh_molecule())
    atoms.set_constraint(FixAtoms(indices=[4]))
    with pytest.raises(ConfsearchPreparationError, match="must not intersect"):
        resolve_regions(atoms, mode_a_options(), adsorbate_indices=np.array([4, 5]))


def test_regions_warn_when_no_atoms_are_fixed():
    atoms = combine(slab_atoms(), oh_molecule())
    regions = resolve_regions(
        atoms, mode_a_options(), adsorbate_indices=np.array([4, 5])
    )
    assert regions.fixed.size == 0
    assert any("F is empty" in warning for warning in regions.warnings)


def test_fix_below_extends_the_fixed_set():
    slab = slab_atoms(height=20.0)
    slab.positions[:, 2] = [1.0, 1.0, 1.0, 1.0]
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 2.5))
    regions = resolve_regions(
        atoms,
        mode_a_options(cs_fix_below=0.5),
        adsorbate_indices=np.array([4, 5]),
    )
    assert regions.fixed.tolist() == [0, 1, 2, 3]
    assert set(regions.fix_below_added) == {0, 1, 2, 3}


def test_regions_validate_the_adsorbate_selection():
    atoms = combine(slab_atoms(), oh_molecule())
    options = mode_a_options()
    with pytest.raises(ConfsearchPreparationError, match="empty"):
        resolve_regions(atoms, options, adsorbate_indices=np.array([], dtype=int))
    with pytest.raises(ConfsearchPreparationError, match="whole structure"):
        resolve_regions(atoms, options, adsorbate_indices=np.arange(len(atoms)))
    with pytest.raises(ConfsearchPreparationError, match="outside the"):
        resolve_regions(atoms, options, adsorbate_indices=np.array([99]))
    with pytest.raises(ConfsearchPreparationError, match="repeats"):
        resolve_regions(atoms, options, adsorbate_indices=np.array([4, 4]))


# ---------------------------------------------------------------------------
# surface validation
# ---------------------------------------------------------------------------


def test_validate_surface_accepts_a_conventional_slab():
    slab = slab_atoms()
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 2.5))
    regions = resolve_regions(
        atoms, mode_a_options(), adsorbate_indices=np.array([4, 5])
    )
    geometry = validate_all(atoms, regions, mode_a_options())
    assert geometry.z_top == pytest.approx(1.0)
    assert geometry.vacuum_gap == pytest.approx(20.0)
    assert geometry.adsorbate_centroid_z == pytest.approx(2.5)


def test_validate_surface_requires_3d_pbc():
    slab = slab_atoms()
    slab.pbc = (True, True, False)
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 2.5))
    regions = resolve_regions(
        atoms, mode_a_options(), adsorbate_indices=np.array([4, 5])
    )
    with pytest.raises(ConfsearchPreparationError, match="pbc"):
        validate_surface(atoms, regions, mode_a_options())


def test_validate_surface_requires_the_normal_to_be_z():
    slab = slab_atoms()
    slab.cell[0] = [4.0, 0.0, 0.5]
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 2.5))
    regions = resolve_regions(
        atoms, mode_a_options(), adsorbate_indices=np.array([4, 5])
    )
    with pytest.raises(ConfsearchPreparationError, match="surface normal must be \\+z"):
        validate_surface(atoms, regions, mode_a_options())


def test_validate_surface_rejects_a_wrapped_slab():
    slab = slab_atoms(height=6.0)
    # Move one slab atom across the cell boundary: the vacuum is no longer the
    # periodic gap, so the slab is not contiguous inside the cell.
    slab.positions[3] = [2.5, 2.5, 5.0]
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 1.0))
    regions = resolve_regions(
        atoms, mode_a_options(), adsorbate_indices=np.array([4, 5])
    )
    with pytest.raises(ConfsearchPreparationError, match="contiguous in z"):
        validate_surface(atoms, regions, mode_a_options())


def test_validate_surface_requires_the_adsorbate_above_the_slab():
    atoms = combine(slab_atoms(), oh_molecule(), shift=(0.5, 0.5, 0.2))
    regions = resolve_regions(
        atoms, mode_a_options(), adsorbate_indices=np.array([4, 5])
    )
    with pytest.raises(ConfsearchPreparationError, match="centroid above the slab top"):
        validate_surface(atoms, regions, mode_a_options())


def test_periodic_gap_helper():
    periodic, internal = periodic_gap_and_contiguity(
        np.array([1.0, 1.0, 1.0, 1.0]), 20.0
    )
    assert periodic == pytest.approx(20.0)
    assert internal == pytest.approx(0.0)
    periodic, internal = periodic_gap_and_contiguity(
        np.array([1.0, 1.5, 5.0, 5.5]), 20.0
    )
    assert periodic == pytest.approx(15.5)
    assert internal == pytest.approx(3.5)


def test_adsorbate_connectivity_is_enforced():
    slab = slab_atoms()
    molecule = oh_molecule()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    options = mode_a_options()
    regions = resolve_regions(atoms, options, adsorbate_indices=np.array([4, 5]))
    validate_all(atoms, regions, options)

    # A molecule whose two non-ionic atoms are not bonded is not one graph.
    wider = slab_atoms(spacing=3.0)
    stretched = combine(wider, oh_molecule(offset=3.0), shift=(0.5, 0.5, 2.5))
    disconnected = resolve_regions(
        stretched, options, adsorbate_indices=np.array([4, 5])
    )
    with pytest.raises(ConfsearchPreparationError, match="not one connected"):
        validate_all(stretched, disconnected, options)


def test_ionic_only_adsorbate_is_allowed():
    slab = slab_atoms()
    atoms = combine(slab, Atoms("Li", positions=[[0.5, 0.5, 2.5]]))
    options = mode_a_options()
    regions = resolve_regions(atoms, options, adsorbate_indices=np.array([4]))
    validate_all(atoms, regions, options)


def test_mobile_sets_follow_the_option():
    slab = slab_atoms(height=20.0)
    slab.positions[:, 2] = [0.5, 1.5, 2.5, 3.5]
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 6.0))
    options = mode_a_options()
    regions = resolve_regions(atoms, options, adsorbate_indices=np.array([4, 5]))

    assert mobile_indices(atoms, regions, options).tolist() == [4, 5]
    top = mobile_indices(
        atoms,
        regions,
        replace(options, cs_md_mobile="adsorbate_and_top"),
        z_top=3.5,
    )
    assert top.tolist() == [1, 2, 3, 4, 5]
    all_free = replace(options, cs_md_mobile="all_free")
    assert mobile_indices(atoms, regions, all_free).tolist() == [0, 1, 2, 3, 4, 5]


def test_h_timestep_rule():
    slab = slab_atoms()
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 2.5))
    options = mode_a_options(timestep=2.0, cs_hold_ps=4.0, cs_cool_ps=4.0)
    regions = resolve_regions(atoms, options, adsorbate_indices=np.array([4, 5]))
    with pytest.raises(ConfsearchPreparationError, match="H is in the MD-mobile"):
        validate_all(atoms, regions, options)


# ---------------------------------------------------------------------------
# covalent graphs
# ---------------------------------------------------------------------------


def test_covalent_bonds_respect_the_scale_and_ionic_exclusion():
    slab = slab_atoms()  # S-S at 2.0 A, 1.2 * 2.05 = 2.46
    graph = covalent_bonds(
        slab, np.arange(len(slab)), bond_scale=1.2, ionic_symbols=("Li",)
    )
    assert graph.as_sorted_list() == [[0, 1], [0, 2], [1, 3], [2, 3]]

    tight = covalent_bonds(slab, np.arange(len(slab)), bond_scale=0.9, ionic_symbols=())
    assert len(tight) == 0

    with_lithium = Atoms(
        "LiS",
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    assert (
        len(
            covalent_bonds(
                with_lithium, np.arange(2), bond_scale=1.2, ionic_symbols=("Li",)
            )
        )
        == 0
    )


def test_broken_reference_bonds_use_the_break_scale():
    molecule = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    reference = covalent_bonds(molecule, np.arange(2), bond_scale=1.2, ionic_symbols=())
    assert reference.as_sorted_list() == [[0, 1]]
    assert (
        broken_reference_bonds(molecule, reference, break_scale=1.5, ionic_symbols=())
        == ()
    )

    stretched = molecule.copy()
    stretched.positions[1, 0] = 1.35  # broken at 1.2x, intact at 1.5x
    assert broken_reference_bonds(
        stretched, reference, break_scale=1.2, ionic_symbols=()
    ) == ((0, 1),)
    assert (
        broken_reference_bonds(stretched, reference, break_scale=1.5, ionic_symbols=())
        == ()
    )


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def _chemisorbed_case():
    slab = slab_atoms(spacing=3.0)  # 6 A cell: molecule images stay >= 4 A away
    molecule = oh_molecule()
    atoms = combine(slab, molecule, shift=(0.5, 0.5, 2.5))
    options = mode_a_options()
    regions = resolve_regions(atoms, options, adsorbate_indices=np.array([4, 5]))
    return atoms, regions, options


def test_adsorbate_slab_bond_is_chemisorbed_not_reacted():
    atoms, regions, options = _chemisorbed_case()
    reference = reference_for(atoms, regions, options)
    classification = classify_minimum(
        atoms, regions=regions, options=options, reference=reference
    )
    assert classification.status == "valid"
    assert classification.chemisorbed
    assert classification.reacted_bonds["adsorbate"]["lost"] == []
    assert len(classification.adsorbate_slab_bonds) == 1
    bond = classification.adsorbate_slab_bonds[0]
    assert bond[0] == 4 and bond[1] == 0
    assert bond[4] == pytest.approx(1.5)
    assert classification.min_adsorbate_slab_distance == pytest.approx(1.5)


def test_broken_skeleton_bond_is_reacted():
    atoms, regions, options = _chemisorbed_case()
    reference = reference_for(atoms, regions, options)
    broken = atoms.copy()
    broken.positions[5] += [2.0, 0.0, 0.0]
    classification = classify_minimum(
        broken, regions=regions, options=options, reference=reference
    )
    assert classification.status == "reacted"
    assert classification.reacted_bonds["adsorbate"]["lost"] == [[4, 5]]


def test_desorbed_minimum():
    slab = slab_atoms(spacing=3.0)
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 8.0))
    options = mode_a_options()
    regions = resolve_regions(atoms, options, adsorbate_indices=np.array([4, 5]))
    reference = reference_for(atoms, regions, options)
    classification = classify_minimum(
        atoms, regions=regions, options=options, reference=reference
    )
    assert classification.status == "desorbed"
    assert not classification.chemisorbed


def test_image_contact_via_self_image():
    slab = slab_atoms(spacing=1.5)
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 2.5))
    options = mode_a_options()
    regions = resolve_regions(atoms, options, adsorbate_indices=np.array([4, 5]))
    reference = reference_for(atoms, regions, options)
    classification = classify_minimum(
        atoms, regions=regions, options=options, reference=reference
    )
    assert classification.min_self_image_distance < 4.0
    assert classification.status == "image_contact"


def test_image_contact_via_vacuum_image():
    slab = slab_atoms(spacing=3.0, height=6.0)
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 4.0))
    options = mode_a_options()
    regions = resolve_regions(atoms, options, adsorbate_indices=np.array([4, 5]))
    reference = reference_for(atoms, regions, options)
    classification = classify_minimum(
        atoms, regions=regions, options=options, reference=reference
    )
    assert classification.min_vacuum_gap < 8.0
    assert classification.status == "image_contact"


def test_slab_flags_are_recorded_without_changing_status():
    slab = slab_atoms(spacing=5.0, n=3)
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 2.5))
    options = mode_a_options()
    regions = resolve_regions(atoms, options, adsorbate_indices=np.array([9, 10]))
    reference = reference_for(atoms, regions, options)

    rearranged = atoms.copy()
    rearranged.positions[8] += [0.0, 0.0, 1.2]  # far corner, > 1.0 A
    classification = classify_minimum(
        rearranged, regions=regions, options=options, reference=reference
    )
    assert classification.status == "valid"
    assert "slab_rearranged" in classification.flags
    assert "far_field_slab_changed" in classification.flags
    assert classification.max_slab_displacement_nonionic == pytest.approx(1.2)

    nudged = atoms.copy()
    nudged.positions[8] += [0.0, 0.0, 0.6]  # > 0.5 A but < 1.0 A
    classification = classify_minimum(
        nudged, regions=regions, options=options, reference=reference
    )
    assert "far_field_slab_changed" in classification.flags
    assert "slab_rearranged" not in classification.flags


def test_fixed_atom_movement_is_an_internal_error():
    slab = slab_atoms()
    atoms = combine(slab, oh_molecule(), shift=(0.5, 0.5, 2.5))
    atoms.set_constraint(FixAtoms(indices=[0]))
    options = mode_a_options()
    regions = resolve_regions(atoms, options, adsorbate_indices=np.array([4, 5]))
    reference = reference_for(atoms, regions, options)
    moved = atoms.copy()
    moved.positions[0] += [0.0, 0.0, 0.01]
    with pytest.raises(ConfsearchInternalError, match="fixed atom moved"):
        classify_minimum(moved, regions=regions, options=options, reference=reference)
