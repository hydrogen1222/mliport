"""NEB runtime energy-force consistency pre-flight."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from mliport.neb.preflight import (
    NEBGradientConsistencyError,
    directional_check,
    energy_noise_floor_eV,
    run_gradient_preflight,
)

R0 = 1.2


class PairCalculator(Calculator):
    """Anharmonic pair potential with analytic forces (conservative)."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, force_scale: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.force_scale = force_scale
        self.evaluated_positions: list[np.ndarray] = []

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        pos = atoms.positions
        self.evaluated_positions.append(pos.copy())
        energy = 0.0
        forces = np.zeros_like(pos)
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                d = pos[i] - pos[j]
                r = float(np.linalg.norm(d))
                x = r - R0
                energy += 2.0 * x**2 + 3.0 * x**3 + 5.0 * x**4
                dedr = 4.0 * x + 9.0 * x**2 + 20.0 * x**3
                pair_force = -dedr * d / r
                forces[i] += pair_force
                forces[j] -= pair_force
        self.results = {"energy": energy, "forces": self.force_scale * forces}


class SwirlCalculator(PairCalculator):
    """Adds a non-conservative rotational force field to the true gradient."""

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        swirl = 0.3 * np.stack(
            [-atoms.positions[:, 1], atoms.positions[:, 0], np.zeros(len(atoms))],
            axis=1,
        )
        self.results["forces"] = self.results["forces"] + swirl


def _band(n_images: int = 5) -> list[Atoms]:
    start = Atoms(
        "H4",
        positions=[[0, 0, 0], [1.3, 0, 0], [0, 1.4, 0], [0.1, 0.2, 1.35]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    end = start.copy()
    end.positions[3] += [0.4, -0.3, 0.2]
    images = []
    for index in range(n_images):
        image = start.copy()
        image.positions = start.positions + (index / (n_images - 1)) * (
            end.positions - start.positions
        )
        images.append(image)
    return images


# --- directional_check ----------------------------------------------------


def _quartic(x: np.ndarray) -> float:
    return float(0.5 * np.sum(x**2) + 0.1 * np.sum(x**4))


def _quartic_forces(x: np.ndarray) -> np.ndarray:
    return -(x + 0.4 * x**3)


def test_consistent_forces_pass_with_tiny_error():
    x = np.random.default_rng(0).uniform(-1, 1, (6, 3))
    direction = np.random.default_rng(1).standard_normal((6, 3))
    direction /= np.linalg.norm(direction)
    result = directional_check(
        _quartic,
        x,
        _quartic_forces(x),
        direction,
        noise_floor_eV=energy_noise_floor_eV(_quartic(x), "float64"),
    )
    assert result["passed"] is True
    assert result["abs_error_eV_A"] < 1e-6


def test_forces_off_by_five_percent_fail():
    x = np.random.default_rng(0).uniform(-1, 1, (6, 3))
    forces = 1.05 * _quartic_forces(x)
    direction = forces / np.linalg.norm(forces)
    result = directional_check(
        _quartic,
        x,
        forces,
        direction,
        noise_floor_eV=energy_noise_floor_eV(_quartic(x), "float64"),
    )
    assert result["passed"] is False
    assert result["resolved"] is True


def test_non_finite_force_fails():
    x = np.zeros((2, 3))
    forces = np.zeros((2, 3))
    forces[0, 0] = np.nan
    direction = np.zeros((2, 3))
    direction[1, 1] = 1.0
    result = directional_check(_quartic, x, forces, direction, noise_floor_eV=0.0)
    assert result["finite"] is False
    assert result["passed"] is False


def test_float32_rounding_is_tolerated_but_unresolvable_energy_is_refused():
    x = np.random.default_rng(2).uniform(-1, 1, (6, 3))
    direction = np.random.default_rng(3).standard_normal((6, 3))
    direction /= np.linalg.norm(direction)

    def rounded(y: np.ndarray) -> float:
        return float(np.float32(_quartic(y)))

    ok = directional_check(
        rounded,
        x,
        _quartic_forces(x),
        direction,
        noise_floor_eV=energy_noise_floor_eV(_quartic(x), "float32"),
    )
    assert ok["passed"] is True

    offset = 1.0e4  # |E| so large that float32 cannot resolve the derivative

    def huge(y: np.ndarray) -> float:
        return float(np.float32(offset + _quartic(y)))

    result = directional_check(
        huge,
        x,
        _quartic_forces(x),
        direction,
        noise_floor_eV=energy_noise_floor_eV(offset, "float32"),
    )
    assert result["resolved"] is False
    assert result["passed"] is False


def test_unknown_precision_is_treated_as_float32():
    assert energy_noise_floor_eV(100.0, None) == energy_noise_floor_eV(100.0, "float32")
    assert energy_noise_floor_eV(100.0, "float64") < energy_noise_floor_eV(
        100.0, "float32"
    )


# --- run_gradient_preflight ------------------------------------------------


def test_conservative_calculator_passes_and_records_evidence():
    images = _band()
    summary = run_gradient_preflight(PairCalculator(), images, dtype="float64")
    assert summary["status"] == "passed"
    assert summary["n_failed"] == 0
    assert summary["sampled_images"] == [0, 1, 3, 4]
    assert summary["n_checks"] >= 2 * len(summary["sampled_images"])
    assert summary["max_abs_error_eV_A"] < 1e-5
    assert {check["direction"] for check in summary["checks"]} >= {"random"}


def test_input_images_are_not_modified():
    images = _band()
    before = [image.positions.copy() for image in images]
    run_gradient_preflight(PairCalculator(), images, dtype="float64")
    for image, positions in zip(images, before, strict=True):
        assert np.array_equal(image.positions, positions)
        assert image.calc is None


def test_scaled_forces_are_refused():
    with pytest.raises(
        NEBGradientConsistencyError, match="not verifiably the gradient"
    ):
        run_gradient_preflight(
            PairCalculator(force_scale=1.1), _band(), dtype="float64"
        )


def test_non_conservative_force_field_is_refused():
    with pytest.raises(NEBGradientConsistencyError) as excinfo:
        run_gradient_preflight(SwirlCalculator(), _band(), dtype="float64")
    assert excinfo.value.summary["status"] == "failed"
    assert excinfo.value.summary["n_failed"] >= 1


def test_fixed_atoms_are_never_displaced():
    images = _band()
    for image in images:
        image.set_constraint(FixAtoms(indices=[0]))
    calc = PairCalculator()
    summary = run_gradient_preflight(calc, images, dtype="float64")
    assert summary["status"] == "passed"
    assert calc.evaluated_positions
    # atom 0 is fixed at the origin in every image and is never displaced
    for positions in calc.evaluated_positions:
        assert np.allclose(positions[0], 0.0)


def test_all_atoms_fixed_is_refused():
    images = _band()
    for image in images:
        image.set_constraint(FixAtoms(indices=list(range(len(image)))))
    with pytest.raises(NEBGradientConsistencyError, match="no free atoms"):
        run_gradient_preflight(PairCalculator(), images, dtype="float64")


def test_preflight_is_deterministic():
    first = run_gradient_preflight(PairCalculator(), _band(), dtype="float64")
    second = run_gradient_preflight(PairCalculator(), _band(), dtype="float64")
    assert first["checks"] == second["checks"]
