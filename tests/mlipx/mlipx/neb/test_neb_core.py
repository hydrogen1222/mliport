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


class AsymmetricDoubleWellCalculator(Calculator):
    """Double well with stationary endpoints at unequal reference energies."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]
    tilt = 0.3

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=float)
        x = positions[:, 0]
        energy = np.sum(
            (x**2 - 1.0) ** 2
            + self.tilt * (x**3 / 3.0 - x)
            + positions[:, 1] ** 2
            + positions[:, 2] ** 2
        )
        forces = np.zeros_like(positions)
        forces[:, 0] = -(x**2 - 1.0) * (4.0 * x + self.tilt)
        forces[:, 1:] = -2.0 * positions[:, 1:]
        self.results = {"energy": float(energy), "forces": forces}


class CurvedMEPCalculator(Calculator):
    """Two-dimensional potential whose minimum-energy path arches in y."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]
    arch_height = 0.6
    transverse_stiffness = 5.0

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=float)
        x = positions[:, 0]
        y = positions[:, 1]
        residual = y - self.arch_height * (1.0 - x**2)
        energy = np.sum(
            (x**2 - 1.0) ** 2
            + self.transverse_stiffness * residual**2
            + positions[:, 2] ** 2
        )
        forces = np.zeros_like(positions)
        forces[:, 0] = -(
            4.0 * x * (x**2 - 1.0)
            + 4.0 * self.transverse_stiffness * self.arch_height * x * residual
        )
        forces[:, 1] = -2.0 * self.transverse_stiffness * residual
        forces[:, 2] = -2.0 * positions[:, 2]
        self.results = {"energy": float(energy), "forces": forces}


class ZeroCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros_like(atoms.positions, dtype=float),
        }


class PathDoubleWellCalculator(Calculator):
    """Rigid-rotation-invariant double well along one Cartesian path."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, origin: np.ndarray, displacement: np.ndarray) -> None:
        super().__init__()
        self.origin = np.asarray(origin, dtype=float)
        self.displacement = np.asarray(displacement, dtype=float)
        self.displacement_sq = float(self.displacement @ self.displacement)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        delta = np.asarray(atoms.positions, dtype=float) - self.origin
        progress = (delta @ self.displacement) / self.displacement_sq
        reaction_coordinate = 2.0 * progress - 1.0
        perpendicular = delta - progress[:, None] * self.displacement
        energy = np.sum(
            (reaction_coordinate**2 - 1.0) ** 2 + 2.0 * np.sum(perpendicular**2, axis=1)
        )
        gradient = (
            4.0
            * reaction_coordinate[:, None]
            * (reaction_coordinate**2 - 1.0)[:, None]
            * (2.0 * self.displacement / self.displacement_sq)
            + 4.0 * perpendicular
        )
        self.results = {"energy": float(energy), "forces": -gradient}


class SnapshotRunner(NEBRunner):
    """Test probe that retains the final copied band geometry."""

    snapshot_positions_A: np.ndarray | None = None

    def _snapshot(self, images, *, neb_forces=None):
        self.snapshot_positions_A = np.asarray(
            [image.positions for image in images], dtype=float
        ).copy()
        return super()._snapshot(images, neb_forces=neb_forces)


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


def test_asymmetric_double_well_forward_reverse_and_reaction_energy() -> None:
    options = _double_well_options()
    initial, final = _double_well_endpoints()
    result = NEBRunner(AsymmetricDoubleWellCalculator(), options, verbose=False).run(
        prepare_band(initial, final, options)
    )

    metrics = result.to_dict()
    assert result.status == "completed"
    assert metrics["reaction_energy_eV"] == pytest.approx(-0.4, abs=1.0e-10)
    assert (
        metrics["barrier_forward_sampled_eV"] - metrics["barrier_reverse_sampled_eV"]
    ) == pytest.approx(metrics["reaction_energy_eV"], abs=1.0e-10)
    assert metrics["barrier_forward_sampled_eV"] != pytest.approx(
        metrics["barrier_reverse_sampled_eV"]
    )


def test_curved_mep_relaxes_sideways_from_the_linear_path() -> None:
    options = _double_well_options(
        pre_fmax_eV_A=0.03,
        pre_max_steps=300,
        fmax_eV_A=2.0e-3,
        max_steps=500,
        maxstep_A=0.05,
    )
    initial, final = _double_well_endpoints()
    runner = SnapshotRunner(CurvedMEPCalculator(), options, verbose=False)

    result = runner.run(prepare_band(initial, final, options))

    assert result.status == "completed"
    assert result.to_dict()["barrier_forward_sampled_eV"] == pytest.approx(
        1.0, abs=2.0e-3
    )
    assert runner.snapshot_positions_A is not None
    assert np.max(runner.snapshot_positions_A[1:-1, 0, 1]) > 0.55


def test_triclinic_winding_barrier_and_displacement_are_rotation_invariant() -> None:
    cell = np.asarray([[4.0, 0.0, 0.0], [1.8, 3.5, 0.0], [-1.2, 0.9, 3.2]], dtype=float)
    initial_fractional = np.asarray([0.1, 0.25, 0.3])
    final_fractional = np.asarray([0.2, 0.25, 0.3])
    origin = initial_fractional @ cell
    wrapped_final = final_fractional @ cell
    displacement = 1.1 * cell[0]
    axis = np.asarray([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    options = _double_well_options(
        n_intermediate_images=8,
        path_convention="unwrapped",
        pre_max_steps=100,
        max_steps=200,
    )
    barriers = []
    displacements = []
    energies = []
    for transform in (np.eye(3), rotation):
        rotated_cell = cell @ transform.T
        initial = Atoms(
            "H", positions=[origin @ transform.T], cell=rotated_cell, pbc=True
        )
        final = Atoms(
            "H", positions=[wrapped_final @ transform.T], cell=rotated_cell, pbc=True
        )
        band = prepare_band(initial, final, options, image_shifts=[[1, 0, 0]])
        displacements.append(band.final.positions[0] - band.initial.positions[0])
        result = NEBRunner(
            PathDoubleWellCalculator(origin @ transform.T, displacement @ transform.T),
            options,
            verbose=False,
        ).run(band)
        assert result.status == "completed"
        barriers.append(result.to_dict()["barrier_forward_sampled_eV"])
        energies.append(result.energies_eV)

    assert displacements[0] == pytest.approx(displacement, abs=1.0e-10)
    assert displacements[1] == pytest.approx(displacement @ rotation.T, abs=1.0e-10)
    assert barriers == pytest.approx([1.0, 1.0], abs=2.0e-4)
    assert energies[0] == pytest.approx(energies[1], abs=1.0e-10)


def test_idpp_restores_calculators_and_does_not_leak_idpp_energy() -> None:
    initial = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], pbc=False)
    final = Atoms("H2", positions=[[0, 0, 0], [2, 1, 0]], pbc=False)
    options = NEBOptions(
        n_intermediate_images=2,
        interpolation="idpp",
        idpp_fmax_eV_A=0.1,
        idpp_steps=200,
        fmax_eV_A=0.01,
        max_steps=0,
        min_distance_A=0.1,
    )

    band = prepare_band(initial, final, options)
    assert all(image.calc is None for image in band.images)
    result = NEBRunner(ZeroCalculator(), options, verbose=False).run(band)

    assert result.status == "completed"
    assert result.energies_eV == pytest.approx(np.zeros(4))


def test_idpp_failure_is_fail_closed_without_linear_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], pbc=False)
    final = Atoms("H2", positions=[[0, 0, 0], [2, 1, 0]], pbc=False)
    options = NEBOptions(
        n_intermediate_images=2,
        interpolation="idpp",
        min_distance_A=0.1,
    )

    def fail_idpp(*args, **kwargs):
        raise RuntimeError("synthetic IDPP failure")

    monkeypatch.setattr("mlipx.neb.prepare.idpp_interpolate", fail_idpp)
    with pytest.raises(NEBPreparationError, match="refusing.*fallback"):
        prepare_band(initial, final, options)


def test_shared_calculator_recomputes_a_b_a_without_stale_results() -> None:
    calculator = DoubleWellCalculator()
    images = [
        Atoms("H", positions=[[x, 0.0, 0.0]], calculator=calculator)
        for x in (-0.8, 0.2, -0.8)
    ]
    runner = NEBRunner(calculator, NEBOptions(), verbose=False)

    energies, forces, _, _ = runner._snapshot(images)

    assert energies[0] == pytest.approx(energies[2])
    assert forces[0] == pytest.approx(forces[2])
    assert energies[0] != pytest.approx(energies[1])
    assert forces[0] != pytest.approx(forces[1])


def test_final_climbing_image_index_tracks_changed_highest_image() -> None:
    options = _double_well_options(pre_fmax_eV_A=0.01, maxstep_A=0.05)
    initial_x = [-1.0, -0.9, -0.1, 0.1, 0.2, 0.3, 0.6, 1.0]
    images = [Atoms("H", positions=[[x, 0.0, 0.0]]) for x in initial_x]
    initial_highest = int(np.argmax([(x**2 - 1.0) ** 2 for x in initial_x]))

    result = NEBRunner(DoubleWellCalculator(), options, verbose=False).run(
        validate_band_images(images, options)
    )

    assert result.status == "completed"
    assert initial_highest == 2
    assert result.climbing_image_index == 3
    assert result.climbing_image_index == int(np.argmax(result.energies_eV))


def test_ci_requires_climbing_force_even_if_ordinary_neb_residual_is_small() -> None:
    options = NEBOptions(
        n_intermediate_images=3,
        climb=True,
        pre_fmax_eV_A=0.01,
        pre_max_steps=10,
        fmax_eV_A=0.01,
        max_steps=0,
        endpoint_fmax_eV_A=1.0e-8,
        min_distance_A=0.1,
    )
    initial, final = _double_well_endpoints()

    result = NEBRunner(AsymmetricDoubleWellCalculator(), options, verbose=False).run(
        prepare_band(initial, final, options)
    )

    assert result.stages[0]["converged_by_full_band_force"] is True
    assert result.stages[0]["max_neb_force_eV_A"] == pytest.approx(0.0)
    assert result.stages[1]["max_neb_force_eV_A"] == pytest.approx(0.3)
    assert result.status == "not_converged"
    assert result.converged is False


@pytest.mark.parametrize("failure", ["energy_nan", "force_inf"])
def test_non_finite_intermediate_state_aborts_without_mutating_input_band(
    failure: str,
) -> None:
    class NonFiniteCalculator(DoubleWellCalculator):
        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ) -> None:
            super().calculate(atoms, properties, system_changes)
            if abs(float(atoms.positions[0, 0])) < 0.75:
                if failure == "energy_nan":
                    self.results["energy"] = float("nan")
                else:
                    self.results["forces"][0, 0] = float("inf")

    options = NEBOptions(
        n_intermediate_images=3,
        max_steps=2,
        endpoint_fmax_eV_A=1.0e-8,
        min_distance_A=0.1,
    )
    initial, final = _double_well_endpoints()
    band = prepare_band(initial, final, options)
    trusted_positions = np.asarray([image.positions for image in band.images]).copy()

    with pytest.raises(NEBPreparationError, match="NaN or Inf"):
        NEBRunner(NonFiniteCalculator(), options, verbose=False).run(band)

    assert np.asarray([image.positions for image in band.images]) == pytest.approx(
        trusted_positions
    )


def test_raw_force_abort_is_not_hidden_by_fix_atoms() -> None:
    class HiddenRawForceCalculator(Calculator):
        implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ) -> None:
            super().calculate(atoms, properties, system_changes)
            x = float(atoms.positions[0, 0])
            forces = np.zeros_like(atoms.positions, dtype=float)
            forces[1, 1] = 2.0 * (1.0 - x**2)
            self.results = {"energy": 0.0, "forces": forces}

    initial = Atoms("HHe", positions=[[-1, 0, 0], [0, 2, 0]], pbc=False)
    final = Atoms("HHe", positions=[[1, 0, 0], [0, 2, 0]], pbc=False)
    initial.set_constraint(FixAtoms(indices=[1]))
    final.set_constraint(FixAtoms(indices=[1]))
    options = NEBOptions(
        n_intermediate_images=3,
        max_steps=2,
        force_abort_eV_A=0.5,
        min_distance_A=0.1,
    )
    band = prepare_band(initial, final, options)

    with pytest.raises(NEBPreparationError, match="Raw physical.*force_abort"):
        NEBRunner(HiddenRawForceCalculator(), options, verbose=False).run(band)


def test_zero_steps_distinguishes_unconverged_and_already_converged() -> None:
    initial, final = _double_well_endpoints()
    options = NEBOptions(
        n_intermediate_images=3,
        fmax_eV_A=1.0e-6,
        max_steps=0,
        endpoint_fmax_eV_A=1.0e-8,
        min_distance_A=0.1,
    )
    band = prepare_band(initial, final, options)

    unconverged = NEBRunner(CurvedMEPCalculator(), options, verbose=False).run(band)
    converged = NEBRunner(ZeroCalculator(), options, verbose=False).run(band)

    assert unconverged.status == "not_converged"
    assert unconverged.converged is False
    assert unconverged.max_neb_force_eV_A > options.fmax_eV_A
    assert converged.status == "completed"
    assert converged.converged is True
    assert converged.max_neb_force_eV_A == pytest.approx(0.0)


def test_snapshot_force_arrays_do_not_alias_reused_calculator_results() -> None:
    class ReusedBufferCalculator(Calculator):
        implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

        def __init__(self) -> None:
            super().__init__()
            self.buffer = np.zeros((1, 3), dtype=float)

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ) -> None:
            super().calculate(atoms, properties, system_changes)
            self.buffer[:] = 2.0 * np.asarray(atoms.positions, dtype=float)
            self.results = {
                "energy": float(np.sum(atoms.positions**2)),
                "forces": self.buffer,
            }

    calculator = ReusedBufferCalculator()
    images = [
        Atoms("H", positions=[[x, 0.0, 0.0]], calculator=calculator)
        for x in (0.2, 0.7, -0.4)
    ]
    runner = NEBRunner(calculator, NEBOptions(), verbose=False)

    _, forces, _, _ = runner._snapshot(images)
    expected = np.asarray([[[0.4, 0, 0]], [[1.4, 0, 0]], [[-0.8, 0, 0]]])
    calculator.buffer[:] = 99.0

    assert forces == pytest.approx(expected)
    assert not np.shares_memory(forces[0], forces[1])


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
