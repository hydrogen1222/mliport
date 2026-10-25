"""Automatic placement: determinism, distances, targets and rejections (spec 5.2)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.data import covalent_radii

from mliport.confsearch.errors import ConfsearchPreparationError
from mliport.confsearch.geometry import (
    mic_distance_matrix,
    self_image_distance,
    vacuum_gap,
)
from mliport.confsearch.placement import (
    draw_rotation,
    place_molecule,
    random_rotation_matrix,
)
from tests.mliport.mliport.confsearch.structures import (
    mode_b_options,
    oh_molecule,
    slab_atoms,
)


def _slab():
    return slab_atoms(spacing=3.0)  # 6 A cell, 20 A vacuum


def _options(**overrides):
    return mode_b_options(**overrides)


def _molecule_indices(slab, molecule):
    return np.arange(len(slab), len(slab) + len(molecule))


def test_placement_is_deterministic_for_a_seed():
    slab = _slab()
    molecule = oh_molecule()
    first = place_molecule(slab, [molecule], _options(), rng=np.random.default_rng(7))
    second = place_molecule(slab, [molecule], _options(), rng=np.random.default_rng(7))
    assert np.array_equal(first.atoms.positions, second.atoms.positions)
    assert first.conformer_index == second.conformer_index
    assert first.quaternion == second.quaternion
    assert (first.u, first.v, first.z_offset) == (second.u, second.v, second.z_offset)
    # A different seed gives a different placement.
    third = place_molecule(slab, [molecule], _options(), rng=np.random.default_rng(8))
    assert not np.array_equal(first.atoms.positions, third.atoms.positions)


def test_placed_structure_keeps_the_slab_and_appends_the_molecule():
    slab = _slab()
    molecule = oh_molecule()
    result = place_molecule(slab, [molecule], _options(), rng=np.random.default_rng(1))
    assert len(result.atoms) == len(slab) + len(molecule)
    assert np.array_equal(result.atoms.positions[: len(slab)], slab.get_positions())
    assert result.atoms.get_chemical_symbols()[len(slab) :] == ["O", "H"]


def test_random_rotation_matrix_is_orthogonal():
    rng = np.random.default_rng(2)
    for _ in range(20):
        rotation = random_rotation_matrix(rng)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_random_rotations_are_uniform_on_the_sphere():
    """Section 13.2: rotated unit vectors have mean ~ 0 and <vv> ~ I/3."""
    rng = np.random.default_rng(1234)
    vector = np.array([0.0, 0.0, 1.0])
    samples = np.array([random_rotation_matrix(rng) @ vector for _ in range(10_000)])
    mean = samples.mean(axis=0)
    second = samples[:, :, None] * samples[:, None, :]
    second = second.mean(axis=0)
    assert np.all(np.abs(mean) < 0.03)
    assert np.allclose(second, np.eye(3) / 3.0, atol=0.02)


def test_contact_placement_reaches_the_contact_scale():
    slab = _slab()
    molecule = oh_molecule()
    options = _options()
    result = place_molecule(slab, [molecule], options, rng=np.random.default_rng(4))
    indices = _molecule_indices(slab, molecule)
    assert result.kind == "contact"
    assert result.min_pair_q == pytest.approx(options.cs_place_contact_scale, abs=1e-3)

    distances = mic_distance_matrix(result.atoms, indices, np.arange(len(slab)))
    radii = covalent_radii[np.asarray(result.atoms.numbers, dtype=int)]
    q = distances / (radii[indices][:, None] + radii[np.arange(len(slab))][None, :])
    assert np.min(q) == pytest.approx(options.cs_place_contact_scale, abs=1e-3)
    assert result.min_pair_distance_A == pytest.approx(np.min(distances))
    assert result.min_self_image_distance_A >= options.cs_min_image_distance
    assert result.min_vacuum_gap_A >= options.cs_min_vacuum_gap


def test_anchored_placement_starts_already_bonded():
    slab = _slab()
    molecule = oh_molecule()
    options = _options(cs_place_anchor="0")
    result = place_molecule(slab, [molecule], options, rng=np.random.default_rng(5))
    indices = _molecule_indices(slab, molecule)
    assert result.kind == "anchored"
    assert result.anchor_index == 0
    assert result.anchor_symbol == "O"
    assert result.anchor_q == pytest.approx(options.cs_place_anchor_scale, abs=1e-3)
    assert result.anchor_slab_index is not None

    # The anchor is the lowest adsorbate atom and no other pair is bonded.
    molecule_z = result.atoms.positions[indices, 2]
    assert int(np.argmin(molecule_z)) == 0
    anchor_q = mic_distance_matrix(
        result.atoms, np.array([indices[0]]), np.arange(len(slab))
    )[0] / (
        covalent_radii[result.atoms.numbers[indices[0]]]
        + covalent_radii[np.asarray(result.atoms.numbers[: len(slab)], dtype=int)]
    )
    assert np.min(anchor_q) == pytest.approx(options.cs_place_anchor_scale, abs=1e-3)
    other_q = mic_distance_matrix(
        result.atoms, np.array([indices[1]]), np.arange(len(slab))
    )[0] / (
        covalent_radii[result.atoms.numbers[indices[1]]]
        + covalent_radii[np.asarray(result.atoms.numbers[: len(slab)], dtype=int)]
    )
    assert np.min(other_q) >= options.cs_place_other_min_scale


def test_anchored_placement_respects_the_target_elements():
    slab = _slab()
    molecule = oh_molecule()
    options = _options(cs_place_anchor="0", cs_place_anchor_targets="S")
    result = place_molecule(slab, [molecule], options, rng=np.random.default_rng(6))
    assert result.anchor_slab_index is not None
    assert result.atoms.get_chemical_symbols()[result.anchor_slab_index] == "S"

    with pytest.raises(ConfsearchPreparationError, match="not present in the slab"):
        place_molecule(
            slab,
            [molecule],
            _options(cs_place_anchor="0", cs_place_anchor_targets="P"),
            rng=np.random.default_rng(6),
        )


def test_anchor_indices_are_validated_against_the_molecule():
    slab = _slab()
    molecule = oh_molecule()
    with pytest.raises(ConfsearchPreparationError, match="outside the 2-atom molecule"):
        place_molecule(
            slab,
            [molecule],
            _options(cs_place_anchor="5"),
            rng=np.random.default_rng(6),
        )


def test_conformers_must_share_one_element_order():
    slab = _slab()
    first = oh_molecule()
    second = Atoms("HO", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])
    with pytest.raises(ConfsearchPreparationError, match="different element order"):
        place_molecule(slab, [first, second], _options(), rng=np.random.default_rng(6))


def test_conformer_choice_is_recorded():
    slab = _slab()
    first = oh_molecule()
    second = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.96, 0.0]])
    result = place_molecule(
        slab, [first, second], _options(), rng=np.random.default_rng(3)
    )
    assert result.conformer_index in (0, 1)


def test_self_image_rejection_is_counted():
    slab = slab_atoms(spacing=1.5)  # 3 A cell: molecule images are too close
    molecule = oh_molecule()
    with pytest.raises(ConfsearchPreparationError, match="self_image"):
        place_molecule(
            slab,
            [molecule],
            _options(cs_place_max_attempts=5),
            rng=np.random.default_rng(9),
        )


def test_vacuum_rejection_is_counted():
    slab = slab_atoms(spacing=3.0, height=6.0)  # only 6 A of cell height
    molecule = oh_molecule()
    with pytest.raises(ConfsearchPreparationError, match="vacuum"):
        place_molecule(
            slab,
            [molecule],
            _options(cs_place_max_attempts=5),
            rng=np.random.default_rng(9),
        )


def test_anchored_other_min_scale_failure_reports_counts():
    slab = _slab()
    molecule = oh_molecule()
    with pytest.raises(ConfsearchPreparationError, match="other_min_scale"):
        place_molecule(
            slab,
            [molecule],
            _options(
                cs_place_anchor="0",
                cs_place_other_min_scale=5.0,
                cs_place_max_attempts=7,
            ),
            rng=np.random.default_rng(10),
        )


def test_orientation_distribution_under_the_anchor_condition_is_uniform():
    """Rejection sampling must match brute-force rejection (section 13.2)."""
    molecule = oh_molecule()
    centred = np.asarray(molecule.get_positions()) - np.mean(
        molecule.get_positions(), axis=0
    )
    rng = np.random.default_rng(4321)
    samples = 2000

    sampled = []
    for _ in range(samples):
        for _ in range(1000):
            _, rotation = draw_rotation(rng)
            rotated = centred @ rotation.T
            if int(np.argmin(rotated[:, 2])) == 0:
                sampled.append(rotated[0] / np.linalg.norm(rotated[0]))
                break

    brute = []
    while len(brute) < samples:
        _, rotation = draw_rotation(rng)
        rotated = centred @ rotation.T
        if int(np.argmin(rotated[:, 2])) == 0:
            brute.append(rotated[0] / np.linalg.norm(rotated[0]))

    sampled = np.array(sampled)
    brute = np.array(brute)
    assert sampled.shape == (samples, 3)
    assert np.allclose(sampled.mean(axis=0), brute.mean(axis=0), atol=0.05)
    assert np.allclose(
        (sampled[:, :, None] * sampled[:, None, :]).mean(axis=0),
        (brute[:, :, None] * brute[:, None, :]).mean(axis=0),
        atol=0.03,
    )


def test_surface_validator_rejections_are_counted():
    slab = _slab()
    molecule = oh_molecule()

    def always_reject(_atoms):
        raise ConfsearchPreparationError("nope")

    with pytest.raises(ConfsearchPreparationError, match="surface_geometry"):
        place_molecule(
            slab,
            [molecule],
            _options(cs_place_max_attempts=4),
            rng=np.random.default_rng(11),
            surface_validator=always_reject,
        )


def test_self_image_distance_helper_matches_the_placement_record():
    slab = _slab()
    molecule = oh_molecule()
    result = place_molecule(slab, [molecule], _options(), rng=np.random.default_rng(12))
    indices = _molecule_indices(slab, molecule)
    assert result.min_self_image_distance_A == pytest.approx(
        self_image_distance(result.atoms, indices)
    )
    assert result.min_vacuum_gap_A == pytest.approx(
        vacuum_gap(result.atoms, indices, np.arange(len(slab)))
    )


def _li_terminated_slab():
    """A slab whose topmost atom is an ionic Li (spec 5.2 revision 5 case)."""
    slab = slab_atoms(spacing=3.0)  # 4 S at z = 1.0, cell 6 x 6 x 20
    slab += Atoms("Li", positions=[[0.5, 0.5, 2.0]], cell=slab.cell, pbc=True)
    return slab


def test_anchored_placement_bonds_to_the_non_ionic_partner_not_to_lithium():
    """The anchor height is solved against the partner set, not every slab atom.

    With the pre-revision-5 behaviour the O would be placed at 1.1x the O+Li
    radii sum from the Li (an ionic contact, classified physisorbed); the partner
    set must exclude the ionic Li and bond to the sulfur instead.
    """
    slab = _li_terminated_slab()
    molecule = oh_molecule()
    options = _options(cs_place_anchor="0")
    result = place_molecule(slab, [molecule], options, rng=np.random.default_rng(5))

    symbols = result.atoms.get_chemical_symbols()
    assert result.anchor_symbol == "O"
    assert result.anchor_slab_index is not None
    assert symbols[result.anchor_slab_index] == "S"
    assert result.anchor_q == pytest.approx(options.cs_place_anchor_scale, abs=1e-3)

    # Recompute against the non-ionic slab atoms: the same value.
    non_ionic = np.array(
        [index for index, symbol in enumerate(symbols[: len(slab)]) if symbol != "Li"]
    )
    radii = covalent_radii[np.asarray(result.atoms.numbers, dtype=int)]
    anchor_global = len(slab) + int(result.anchor_index or 0)
    q = mic_distance_matrix(result.atoms, np.array([anchor_global]), non_ionic)[0] / (
        radii[anchor_global] + radii[non_ionic]
    )
    assert float(np.min(q)) == pytest.approx(options.cs_place_anchor_scale, abs=1e-3)


def test_anchor_targets_can_explicitly_bond_to_an_ionic_partner():
    """Naming the ionic element in cs_place_anchor_targets is the opt-in."""
    slab = _li_terminated_slab()
    molecule = oh_molecule()
    options = _options(cs_place_anchor="0", cs_place_anchor_targets="Li")
    result = place_molecule(slab, [molecule], options, rng=np.random.default_rng(5))
    assert result.anchor_slab_index is not None
    assert result.atoms.get_chemical_symbols()[result.anchor_slab_index] == "Li"
    assert result.anchor_q == pytest.approx(options.cs_place_anchor_scale, abs=1e-3)


def test_anchored_placement_requires_a_partner_set():
    """A slab made only of ionic species cannot form the covalent bond."""
    slab = Atoms(
        "Li4",
        positions=[[0.5, 0.5, 1.0], [3.5, 0.5, 1.0], [0.5, 3.5, 1.0], [3.5, 3.5, 1.0]],
        cell=[6.0, 6.0, 20.0],
        pbc=True,
    )
    with pytest.raises(ConfsearchPreparationError, match="no non-ionic atoms"):
        place_molecule(
            slab,
            [oh_molecule()],
            _options(cs_place_anchor="0"),
            rng=np.random.default_rng(5),
        )
