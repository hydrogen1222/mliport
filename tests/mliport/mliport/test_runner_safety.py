"""Pinning tests for the shared runner safety helpers (P0 refactor).

The helpers were extracted from :class:`mliport.runners.md.MDRunner` and
:class:`mliport.runners.base.BaseRunner` without changing behaviour.  These
tests pin that contract directly, so a later edit cannot silently relax the
fail-closed gates that MD (and, from P2 on, confsearch) depend on.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from mliport.runners import safety
from mliport.runners.base import BaseRunner
from mliport.runners.md import MDRunner


class _ForcesStub(Calculator):
    """Calculator stub returning a fixed energy/force pair."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, energy: float = 0.0, forces=None) -> None:
        super().__init__()
        self._energy = float(energy)
        self._forces = forces

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if self._forces is None:
            forces = np.zeros((len(atoms), 3), dtype=float)
        else:
            forces = np.asarray(self._forces, dtype=float)
        self.results = {"energy": self._energy, "forces": forces}


class _WrapperStub:
    """Minimal BaseMLIPCalculator-like wrapper for runner construction."""

    task = "bulk"

    def __init__(self, calculator: Calculator) -> None:
        self._calculator = calculator

    def get_calculator(self) -> Calculator:
        return self._calculator


def _two_atom_cell() -> Atoms:
    return Atoms(
        "HHe",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )


def _runner(tmp_path) -> MDRunner:
    return MDRunner(
        _WrapperStub(_ForcesStub()),
        temperature=300.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=1,
    )


# --- scalar coercion -------------------------------------------------------


def test_finite_float_rejects_bool_and_non_finite():
    assert safety.finite_float(2, name="x") == 2.0
    assert safety.finite_float(np.float32(0.5), name="x") == pytest.approx(0.5)
    with pytest.raises(TypeError, match="finite real number"):
        safety.finite_float(True, name="x")
    with pytest.raises(TypeError, match="finite real number"):
        safety.finite_float("1.0", name="x")
    with pytest.raises(ValueError, match="must be finite"):
        safety.finite_float(float("inf"), name="x")
    with pytest.raises(ValueError, match="must be finite"):
        safety.finite_float(float("nan"), name="x")


def test_strict_integer_accepts_integral_floats_but_not_fractions_or_bool():
    assert safety.strict_integer(3, name="n") == 3
    assert safety.strict_integer(3.0, name="n") == 3
    assert safety.strict_integer(np.int64(4), name="n") == 4
    with pytest.raises(TypeError, match="must be an integer"):
        safety.strict_integer(3.5, name="n")
    with pytest.raises(TypeError, match="must be an integer"):
        safety.strict_integer(True, name="n")
    with pytest.raises(TypeError, match="must be an integer"):
        safety.strict_integer("3", name="n")


# --- atomic state ----------------------------------------------------------


def test_validate_atomic_state_accepts_a_well_formed_structure():
    safety.validate_atomic_state(_two_atom_cell(), context="unit test")


def test_validate_atomic_state_rejects_empty_and_malformed_inputs():
    with pytest.raises(ValueError, match="at least one atom"):
        safety.validate_atomic_state(Atoms(), context="unit test")

    bad_positions = _two_atom_cell()
    bad_positions.positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="positions must have shape"):
        safety.validate_atomic_state(bad_positions, context="unit test")

    bad_cell = _two_atom_cell()
    bad_cell.cell[0] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="linearly independent"):
        safety.validate_atomic_state(bad_cell, context="unit test")

    bad_momenta = _two_atom_cell()
    bad_momenta.set_momenta(np.array([[np.inf, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="momenta must have shape"):
        safety.validate_atomic_state(bad_momenta, context="unit test")


def test_validate_atomic_state_rejects_non_positive_masses():
    atoms = _two_atom_cell()
    atoms.set_masses([1.0, -1.0])
    with pytest.raises(ValueError, match="every atomic mass must be > 0"):
        safety.validate_atomic_state(atoms, context="unit test")


# --- finite checks and force views ----------------------------------------


def test_check_finite_rejects_non_finite_energy_and_forces():
    safety.check_finite(0.0, np.zeros((2, 3)), context="unit test")
    with pytest.raises(RuntimeError, match="Non-finite energy"):
        safety.check_finite(float("nan"), np.zeros((2, 3)), context="unit test")
    with pytest.raises(RuntimeError, match="Non-finite forces"):
        safety.check_finite(
            0.0, np.array([[np.inf, 0.0, 0.0], [0.0, 0.0, 0.0]]), context="unit test"
        )


def test_force_views_keep_raw_and_constraint_applied_forces_distinct():
    atoms = _two_atom_cell()
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.calc = _ForcesStub(energy=-1.5, forces=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    raw, applied = safety.force_views(atoms, -1.5, context="unit test")

    assert raw[0, 0] == pytest.approx(1.0)
    assert raw[1, 0] == pytest.approx(2.0)
    assert applied[0, 0] == pytest.approx(0.0)
    assert applied[1, 0] == pytest.approx(2.0)


def test_force_views_reject_malformed_shapes():
    atoms = _two_atom_cell()
    atoms.calc = _ForcesStub(forces=[[1.0, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match="Malformed raw forces"):
        safety.force_views(atoms, 0.0, context="unit test")


def test_evaluate_force_abort_uses_raw_forces_and_names_the_worst_atom():
    forces = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    gate = safety.evaluate_force_abort(forces, threshold=4.9)
    assert gate.atom_index == 1
    assert gate.max_force_raw == pytest.approx(5.0)
    assert gate.threshold == pytest.approx(4.9)
    assert gate.abort is True
    assert safety.evaluate_force_abort(forces, threshold=5.0).abort is False


def test_force_magnitudes_requires_a_three_column_array():
    magnitudes = safety.force_magnitudes(np.array([[3.0, 4.0, 0.0]]))
    assert magnitudes.tolist() == [5.0]
    with pytest.raises(RuntimeError, match=r"shape \(N, 3\)"):
        safety.force_magnitudes(np.zeros((2, 2)))


# --- safe point ------------------------------------------------------------


def test_validate_safe_point_returns_kinetic_temperature_and_total_energy():
    atoms = _two_atom_cell()
    atoms.calc = _ForcesStub()
    atoms.set_momenta(np.zeros((2, 3)))

    kinetic, temperature, total = safety.validate_safe_point(
        atoms,
        energy=-2.0,
        forces_raw=np.zeros((2, 3)),
        forces_applied=np.zeros((2, 3)),
        temperature_fn=lambda _atoms: 0.0,
        context="unit test",
    )

    assert kinetic == pytest.approx(0.0)
    assert temperature == pytest.approx(0.0)
    assert total == pytest.approx(-2.0)


def test_validate_safe_point_fails_closed_on_non_finite_temperature():
    atoms = _two_atom_cell()
    atoms.calc = _ForcesStub()
    atoms.set_momenta(np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="Non-finite temperature"):
        safety.validate_safe_point(
            atoms,
            energy=0.0,
            forces_raw=np.zeros((2, 3)),
            forces_applied=np.zeros((2, 3)),
            temperature_fn=lambda _atoms: float("nan"),
            context="unit test",
        )


# --- delegation pins -------------------------------------------------------


def test_md_runner_force_views_delegates_to_the_shared_helper(tmp_path):
    runner = _runner(tmp_path)
    atoms = _two_atom_cell()
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.calc = _ForcesStub(energy=0.5, forces=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    raw, applied = runner._force_views(atoms, 0.5, context="delegation pin")
    expected_raw, expected_applied = safety.force_views(
        atoms, 0.5, context="delegation pin"
    )

    assert np.array_equal(raw, expected_raw)
    assert np.array_equal(applied, expected_applied)


def test_force_safety_abort_is_the_shared_class():
    from mliport.runners import md as md_module

    assert md_module.ForceSafetyAbort is safety.ForceSafetyAbort
    error = safety.ForceSafetyAbort(
        step=7, max_force=25.0, atom_index=2, threshold=20.0
    )
    assert error.step == 7
    assert "raw model force" in str(error)


def test_base_runner_check_finite_keeps_the_historical_messages(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(RuntimeError, match="Non-finite energy"):
        runner._check_finite(_two_atom_cell(), float("nan"), context="unit test")
    with pytest.raises(RuntimeError, match="Non-finite forces during unit test"):
        runner._check_finite(
            _two_atom_cell(),
            0.0,
            np.array([[np.nan, 0.0, 0.0]]),
            context="unit test",
        )


def test_base_runner_is_abstract_but_check_finite_needs_no_state():
    assert issubclass(MDRunner, BaseRunner)
