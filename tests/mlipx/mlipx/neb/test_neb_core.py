from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from mlipx.neb import (
    NEBEndpointNotConvergedError,
    NEBOptions,
    NEBPreparationError,
    prepare_band,
    validate_band_images,
)
from mlipx.runners.neb import NEBRunner


class DoubleWellCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=float)
        x = positions[:, 0]
        energy = np.sum((x**2 - 1.0) ** 2 + positions[:, 1] ** 2 + positions[:, 2] ** 2)
        forces = np.zeros_like(positions)
        forces[:, 0] = -4.0 * x * (x**2 - 1.0)
        forces[:, 1:] = -2.0 * positions[:, 1:]
        self.results = {"energy": float(energy), "forces": forces}


def _double_well_options(**overrides) -> NEBOptions:
    values = {
        "n_intermediate_images": 6,
        "climb": True,
        "pre_fmax_eV_A": 0.05,
        "pre_max_steps": 200,
        "fmax_eV_A": 1.0e-4,
        "max_steps": 400,
        "endpoint_fmax_eV_A": 1.0e-8,
        "min_distance_A": 0.1,
    }
    values.update(overrides)
    return NEBOptions(**values)


def _double_well_endpoints() -> tuple[Atoms, Atoms]:
    return (
        Atoms("H", positions=[[-1.0, 0.0, 0.0]], pbc=False),
        Atoms("H", positions=[[1.0, 0.0, 0.0]], pbc=False),
    )


def test_ci_neb_double_well_known_barrier_and_force() -> None:
    options = _double_well_options()
    initial, final = _double_well_endpoints()
    band = prepare_band(initial, final, options)

    result = NEBRunner(DoubleWellCalculator(), options, verbose=False).run(band)

    assert result.status == "completed"
    assert result.converged is True
    assert result.climbing_image_index == 3
    assert result.energies_eV[result.climbing_image_index] == pytest.approx(
        1.0, abs=2e-4
    )
    assert result.to_dict()["barrier_forward_sampled_eV"] == pytest.approx(
        1.0, abs=2e-4
    )
    assert result.to_dict()["barrier_reverse_sampled_eV"] == pytest.approx(
        1.0, abs=2e-4
    )
    assert result.max_neb_force_eV_A <= options.fmax_eV_A
    assert result.stages[0]["stage"] == "neb_pre"
    assert result.stages[1]["stage"] == "ci_neb"


def test_prepare_band_preserves_explicit_atom_mapping() -> None:
    initial = Atoms(
        ["Li", "S", "Li"],
        positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0]],
        pbc=False,
    )
    final = Atoms(
        ["Li", "Li", "S"],
        positions=[[0, 0, 0], [4, 0, 0], [2, 0, 0]],
        pbc=False,
    )
    options = NEBOptions(n_intermediate_images=1, min_distance_A=0.1)

    band = prepare_band(initial, final, options, atom_map=[0, 2, 1])

    assert band.atom_map.tolist() == [0, 2, 1]
    assert band.images[-1].get_chemical_symbols() == ["Li", "S", "Li"]


def test_prepare_band_rejects_invalid_geometry_before_calculator() -> None:
    initial, final = _double_well_endpoints()
    partial_pbc = final.copy()
    partial_pbc.pbc = [True, False, False]
    with pytest.raises(NEBPreparationError, match="PBC"):
        prepare_band(initial, partial_pbc, NEBOptions(n_intermediate_images=1))

    two_initial = Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]], pbc=False)
    two_final = Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]], pbc=False)
    with pytest.raises(NEBPreparationError, match="permutation"):
        prepare_band(
            two_initial,
            two_final,
            NEBOptions(n_intermediate_images=1),
            atom_map=[0, 0],
        )


def test_fixed_atoms_are_preserved_and_endpoint_mismatch_is_rejected() -> None:
    initial, final = _double_well_endpoints()
    initial = Atoms(
        "HHe",
        positions=[[-1.0, 0, 0], [3.0, 0, 0]],
        pbc=False,
    )
    final = Atoms(
        "HHe",
        positions=[[1.0, 0, 0], [3.0, 0, 0]],
        pbc=False,
    )
    initial.set_constraint(FixAtoms(indices=[1]))
    final.set_constraint(FixAtoms(indices=[1]))
    options = NEBOptions(n_intermediate_images=2, min_distance_A=0.1)
    band = prepare_band(initial, final, options)
    assert all(np.allclose(image.positions[1], [3.0, 0, 0]) for image in band.images)

    final.positions[1, 0] = 3.2
    with pytest.raises(NEBPreparationError, match="fixed atom"):
        prepare_band(initial, final, options)


def test_endpoint_validation_does_not_enter_neb_when_unconverged() -> None:
    initial, final = _double_well_endpoints()
    initial.positions[0, 0] = -0.8
    options = NEBOptions(n_intermediate_images=1, endpoint_fmax_eV_A=1.0e-6)
    band = prepare_band(
        initial,
        final,
        options,
    )

    with pytest.raises(NEBEndpointNotConvergedError, match="Endpoint 0"):
        NEBRunner(DoubleWellCalculator(), options, verbose=False).run(band)


def test_existing_band_count_is_not_silently_reinterpolated() -> None:
    initial, final = _double_well_endpoints()
    options = NEBOptions(n_intermediate_images=2, min_distance_A=0.1)
    band = prepare_band(initial, final, options)
    with pytest.raises(NEBPreparationError, match="conflicts"):
        validate_band_images(band.images, NEBOptions(n_intermediate_images=3))


def test_periodic_winding_is_preserved_and_underresolved_segments_fail() -> None:
    initial = Atoms("H", positions=[[0.2, 1.0, 1.0]], cell=[4, 4, 4], pbc=True)
    final = Atoms("H", positions=[[0.3, 1.0, 1.0]], cell=[4, 4, 4], pbc=True)
    options = NEBOptions(
        n_intermediate_images=2,
        path_convention="unwrapped",
        min_distance_A=0.1,
    )
    band = prepare_band(initial, final, options, image_shifts=[[1, 0, 0]])

    assert band.image_shifts.tolist() == [[1, 0, 0]]
    assert band.images[-1].positions[0, 0] == pytest.approx(4.3)

    with pytest.raises(NEBPreparationError, match="Adjacent images"):
        prepare_band(
            initial,
            final,
            NEBOptions(
                n_intermediate_images=1,
                path_convention="unwrapped",
                min_distance_A=0.1,
            ),
            image_shifts=[[1, 0, 0]],
        )


def test_endpoint_relaxation_rebuilds_the_band() -> None:
    initial, final = _double_well_endpoints()
    initial.positions[0, 0] = -0.8
    final.positions[0, 0] = 0.8
    options = NEBOptions(
        n_intermediate_images=3,
        endpoint_policy="relax",
        endpoint_fmax_eV_A=1.0e-3,
        endpoint_steps=100,
        fmax_eV_A=1.0e-3,
        max_steps=100,
        min_distance_A=0.1,
    )
    band = prepare_band(initial, final, options)

    result = NEBRunner(DoubleWellCalculator(), options, verbose=False).run(band)

    assert result.status == "completed"
    assert result.energies_eV[0] == pytest.approx(0.0, abs=1.0e-5)
    assert result.energies_eV[-1] == pytest.approx(0.0, abs=1.0e-5)
