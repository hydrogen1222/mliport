"""PR-D acceptance tests: full-band constraint consistency (N02 / R12).

An explicit user-provided band must be validated image by image: atom count,
species order, masses, per-atom mapping identity, cell/PBC, FixAtoms indices
and frozen-atom coordinates.  Comparing only the endpoints is not enough.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from mliport.neb.prepare import prepare_band, validate_band_images
from mliport.neb.schema import NEBOptions, NEBPreparationError


def _image(*, fixed_x=5.0, moving_y=7.0, symbols="Cu2", masses=None):
    atoms = Atoms(
        symbols,
        positions=[[fixed_x, 5.0, 5.0], [5.0, moving_y, 5.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.set_constraint(FixAtoms(indices=[0]))
    if masses is not None:
        atoms.set_masses(masses)
    return atoms


def _band(*, fixed_x=(5.0, 5.0, 5.0), moving_y=(7.0, 7.1, 7.2)):
    return [
        _image(fixed_x=fx, moving_y=my)
        for fx, my in zip(fixed_x, moving_y, strict=True)
    ]


def _options():
    return NEBOptions(n_intermediate_images=1, interpolation="linear")


def test_provided_band_rejects_moving_fixed_atom():
    images = _band(fixed_x=(5.0, 6.0, 5.0))
    with pytest.raises(NEBPreparationError, match="moves FixAtoms atom"):
        validate_band_images(images, _options())


def test_provided_band_accepts_constant_fixed_atoms():
    band = validate_band_images(_band(), _options())
    assert band.source == "provided_band"
    assert len(band.images) == 3
    for image in band.images:
        np.testing.assert_allclose(image.positions[0], [5.0, 5.0, 5.0], atol=1e-12)


def test_provided_band_rejects_equivalent_lattice_image_of_fixed_atom():
    """The raw-coordinate contract rejects a wrapped-away frozen atom."""
    images = [
        _image(fixed_x=5.0, moving_y=7.0),
        _image(fixed_x=15.0, moving_y=7.1),  # same image in a 10 A cell
        _image(fixed_x=5.0, moving_y=7.2),
    ]
    with pytest.raises(NEBPreparationError, match="equivalent lattice images"):
        validate_band_images(images, _options())


def test_provided_band_rejects_middle_species_reorder():
    images = [
        Atoms(
            "CuAg", positions=[[5, 5, 5], [5, 7, 5]], cell=[10.0, 10.0, 10.0], pbc=True
        ),
        Atoms(
            "AgCu",
            positions=[[5, 5, 5], [5, 7.1, 5]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        Atoms(
            "CuAg",
            positions=[[5, 5, 5], [5, 7.2, 5]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]
    for image in images:
        image.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(NEBPreparationError, match="changes atom identity/order"):
        validate_band_images(images, _options())


def test_provided_band_rejects_mass_mutation():
    images = _band()
    images[1].set_masses([63.546, 196.97])
    with pytest.raises(NEBPreparationError, match="changes atomic masses"):
        validate_band_images(images, _options())


def test_provided_band_rejects_constraint_set_mutation():
    images = _band()
    images[1].set_constraint(FixAtoms(indices=[1]))
    with pytest.raises(NEBPreparationError, match="changes FixAtoms constraints"):
        validate_band_images(images, _options())


def test_provided_band_rejects_atom_id_mutation():
    images = _band()
    for image in images:
        image.arrays["atom_id"] = np.asarray([0, 1])
    images[1].arrays["atom_id"] = np.asarray([1, 0])
    with pytest.raises(NEBPreparationError, match="mapping identity"):
        validate_band_images(images, _options())


def test_provided_band_rejects_atom_id_presence_mutation():
    images = _band()
    images[1].arrays["atom_id"] = np.asarray([0, 1])
    with pytest.raises(NEBPreparationError, match="atom_id metadata presence"):
        validate_band_images(images, _options())


def test_endpoint_interpolation_still_keeps_fixed_atoms_constant():
    initial = _image(fixed_x=5.0, moving_y=7.0)
    final = _image(fixed_x=5.0, moving_y=8.0)
    band = prepare_band(initial, final, _options())
    assert len(band.images) == 3
    for image in band.images:
        assert tuple(image.get_chemical_symbols()) == ("Cu", "Cu")
        np.testing.assert_allclose(image.positions[0], [5.0, 5.0, 5.0], atol=1e-12)
        np.testing.assert_allclose(image.get_masses(), initial.get_masses(), atol=1e-12)
