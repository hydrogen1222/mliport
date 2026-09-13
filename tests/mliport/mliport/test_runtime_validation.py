from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from mliport.validation import displacement_checks


def test_known_answer_energy_gradient_and_geometry_preservation():
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.2, 0.2, 0.1]])
    atoms.calc = LennardJones()
    original = atoms.positions.copy()
    result = displacement_checks(atoms, force_atol=1e-3, repeat_atol=1e-12)
    assert all(value["status"] == "passed" for value in result.values())
    np.testing.assert_array_equal(atoms.positions, original)
    assert len(result["energy_force_consistency"]["samples"]) == 3


def test_independent_force_error_cannot_be_certified():
    class Inconsistent(LennardJones):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results["forces"] += 1.0

    atoms = Atoms("Ar2", positions=[[0, 0, 0], [1.2, 0, 0]])
    atoms.calc = Inconsistent()
    result = displacement_checks(atoms, force_atol=1e-3, repeat_atol=1e-12)
    assert result["energy_force_consistency"]["status"] == "failed"


def test_large_validation_fixture_is_rejected():
    with pytest.raises(ValueError, match="2–32"):
        displacement_checks(Atoms("Ar33"), force_atol=1e-3, repeat_atol=1e-12)
