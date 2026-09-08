"""Known-answer tests for the MD constraint and finite-state contract."""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLength, FixCartesian, FixCom
from ase.io import Trajectory

from mlipx.runners.md import ForceSafetyAbort, MDRunner
from mlipx.writers.json_writer import JsonWriter


class _Wrapper:
    """Minimal calculator wrapper that records premature model loads."""

    def __init__(self, calculator: Calculator, *, has_stress: bool = False):
        self.calculator = calculator
        self.has_stress = has_stress
        self.load_calls = 0
        self.task = "bulk"

    def get_calculator(self) -> Calculator:
        self.load_calls += 1
        return self.calculator

    def info(self) -> dict[str, str]:
        return {"model_type": "known-answer", "task": self.task}


class _ZeroCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(
        self,
        *,
        energy: float = 0.0,
        forces: np.ndarray | None = None,
        stress: np.ndarray | None = None,
    ):
        super().__init__()
        self._energy = energy
        self._forces = None if forces is None else np.asarray(forces)
        self._stress = np.zeros(6) if stress is None else np.asarray(stress)

    def calculate(self, atoms, properties, system_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": self._energy,
            "forces": (
                np.zeros((len(atoms), 3))
                if self._forces is None
                else self._forces.copy()
            ),
            "stress": self._stress.copy(),
        }


class _FixedAtomForceCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, force: float):
        super().__init__()
        self.force = force

    def calculate(self, atoms, properties, system_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        forces = np.zeros((len(atoms), 3))
        forces[0, 0] = self.force
        self.results = {
            "energy": -self.force * float(atoms.positions[0, 0]),
            "forces": forces,
        }


class _HarmonicCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, center: np.ndarray, spring_constant: float):
        super().__init__()
        self.center = np.asarray(center, dtype=float)
        self.spring_constant = spring_constant

    def calculate(self, atoms, properties, system_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        displacement = atoms.positions - self.center
        self.results = {
            "energy": 0.5 * self.spring_constant * float(np.sum(displacement**2)),
            "forces": -self.spring_constant * displacement,
        }


def _atoms(n: int = 4) -> Atoms:
    return Atoms(
        f"Ar{n}",
        positions=np.arange(n * 3, dtype=float).reshape(n, 3) * 0.25 + 1.0,
        cell=np.eye(3) * 12.0,
        pbc=True,
    )


def _runner(wrapper: _Wrapper, tmp_path, **kwargs) -> MDRunner:
    options = {
        "ensemble": "NVE",
        "temperature": 300.0,
        "timestep": 0.25,
        "steps": 2,
        "save_interval": 1,
        "output_dir": tmp_path,
        "pre_relax": False,
        "verbose": False,
        "seed": 17,
        "write_outcar": False,
        "write_xdatcar": False,
        "write_json": False,
        "write_stress": False,
    }
    options.update(kwargs)
    return MDRunner(wrapper, **options)


def test_fixatoms_never_moves_and_force_views_remain_distinct(tmp_path) -> None:
    atoms = _atoms(3)
    frozen_position = atoms.positions[0].copy()
    atoms.set_constraint(FixAtoms(indices=[0]))
    wrapper = _Wrapper(_FixedAtomForceCalculator(force=2.0))
    runner = _runner(wrapper, tmp_path, com_policy="auto", steps=3, write_json=True)

    results = runner.run(atoms)

    with Trajectory(results["trajectory_path"]) as trajectory:
        frames = list(trajectory)
    assert all(frame.positions[0] == pytest.approx(frozen_position) for frame in frames)
    assert all(
        frame.get_velocities()[0] == pytest.approx(np.zeros(3)) for frame in frames
    )
    assert results["forces_raw_eV_A"][0] == pytest.approx([2.0, 0.0, 0.0])
    assert results["forces_applied_eV_A"][0] == pytest.approx(np.zeros(3))
    assert results["forces"] == pytest.approx(results["forces_applied_eV_A"])
    assert frames[-1].info["forces_raw_eV_A"][0] == pytest.approx([2.0, 0.0, 0.0])
    assert frames[-1].info["forces_applied_eV_A"][0] == pytest.approx(np.zeros(3))
    payload = json.loads((tmp_path / "raw" / "mlipx_results.json").read_text())
    output = payload["calculation"]["results"]
    assert output["forces_raw_eV_A"][0] == pytest.approx([2.0, 0.0, 0.0])
    assert output["forces_applied_eV_A"][0] == pytest.approx(np.zeros(3))
    assert payload["calculation"]["md"]["force_contract"]["forces"] == (
        "constraint-applied compatibility alias"
    )


def test_force_abort_uses_raw_force_even_when_constraint_projects_it_out(
    tmp_path,
) -> None:
    atoms = _atoms(3)
    atoms.set_constraint(FixAtoms(indices=[0]))
    runner = _runner(
        _Wrapper(_FixedAtomForceCalculator(force=6.0)),
        tmp_path,
        com_policy="auto",
        steps=10,
        save_interval=10,
        fmax_abort=5.0,
    )

    with pytest.raises(ForceSafetyAbort, match="raw model force"):
        runner.run(atoms)


@pytest.mark.parametrize(
    ("constraint", "com_policy"),
    [
        (FixCom(), "constraint"),
        (FixAtoms(indices=[0]), "none"),
        (FixCartesian(0, mask=(True, False, False)), "none"),
    ],
    ids=("fixcom", "fixatoms", "fixcartesian"),
)
def test_nhc_rejects_every_constraint_combination(
    tmp_path, constraint, com_policy
) -> None:
    atoms = _atoms(3)
    atoms.set_constraint(constraint)
    wrapper = _Wrapper(_ZeroCalculator())
    runner = _runner(
        wrapper,
        tmp_path,
        ensemble="NVT",
        thermostat="NHC",
        com_policy=com_policy,
    )

    with pytest.raises(ValueError, match="NHC.*constraint"):
        runner.run(atoms)
    assert wrapper.load_calls == 0


def test_nhc_requires_explicit_none_instead_of_auto_com_policy(tmp_path) -> None:
    wrapper = _Wrapper(_ZeroCalculator())
    runner = _runner(
        wrapper,
        tmp_path,
        ensemble="NVT",
        thermostat="NHC",
        com_policy="auto",
    )

    with pytest.raises(ValueError, match="com_policy='none'"):
        runner.run(_atoms(3))
    assert wrapper.load_calls == 0


def test_unconstrained_nhc_with_explicit_no_com_is_supported(tmp_path) -> None:
    runner = _runner(
        _Wrapper(_ZeroCalculator()),
        tmp_path,
        ensemble="NVT",
        thermostat="NHC",
        com_policy="none",
    )

    results = runner.run(_atoms(3))

    assert np.isfinite(results["temperature"])
    assert results["md_provenance"]["com_policy"] == "none"
    assert results["md_provenance"]["com_policy_effective"] == "none"
    assert results["md_provenance"]["degrees_of_freedom"] == 3 * 3


def test_unconstrained_nhc_thermostat_target_uses_the_same_3n_dof(tmp_path) -> None:
    atoms = _atoms(3)
    runner = _runner(
        _Wrapper(_ZeroCalculator()),
        tmp_path,
        ensemble="NVT",
        thermostat="NHC",
        com_policy="none",
    )
    runner._configure_com_policy(atoms)
    runner._initialize_velocities(atoms)
    atoms.calc = _ZeroCalculator()

    dynamics = runner._build_dynamics(atoms)
    thermostat = dynamics._thermostat
    target_dof = thermostat._Q[0] / (thermostat._kT * thermostat._tdamp**2)

    assert atoms.get_number_of_degrees_of_freedom() == 3 * len(atoms)
    assert target_dof == pytest.approx(3 * len(atoms))


@pytest.mark.parametrize(
    ("policy", "ensemble", "thermostat", "expect_constraint", "expect_zero_com"),
    [
        ("none", "NVE", "LANGEVIN", False, False),
        ("initialize_only", "NVT", "LANGEVIN", False, True),
        ("constraint", "NVE", "LANGEVIN", True, True),
        ("auto", "NVE", "LANGEVIN", True, True),
    ],
)
def test_each_com_policy_has_explicit_initialization_semantics(
    tmp_path, policy, ensemble, thermostat, expect_constraint, expect_zero_com
) -> None:
    atoms = _atoms(4)
    runner = _runner(
        _Wrapper(_ZeroCalculator()),
        tmp_path,
        com_policy=policy,
        ensemble=ensemble,
        thermostat=thermostat,
    )

    runner._configure_com_policy(atoms)
    runner._initialize_velocities(atoms)

    assert any(isinstance(item, FixCom) for item in atoms.constraints) is (
        expect_constraint
    )
    total_momentum = np.sum(atoms.get_momenta(), axis=0)
    if expect_zero_com:
        assert total_momentum == pytest.approx(np.zeros(3), abs=1.0e-12)
    else:
        assert not np.allclose(total_momentum, np.zeros(3), atol=1.0e-12)
    assert runner._calculate_temperature(atoms) == pytest.approx(300.0)


def test_initialize_only_rejects_nonzero_com_momentum_when_preserving(
    tmp_path,
) -> None:
    atoms = _atoms(4)
    atoms.set_momenta(np.ones((len(atoms), 3)))
    wrapper = _Wrapper(_ZeroCalculator())
    runner = _runner(
        wrapper,
        tmp_path,
        ensemble="NVT",
        thermostat="LANGEVIN",
        com_policy="initialize_only",
        velocity_policy="preserve",
    )

    with pytest.raises(ValueError, match="would alter the supplied momenta"):
        runner.run(atoms)

    assert wrapper.load_calls == 0


def test_auto_preserves_existing_fixatoms_without_appending_fixcom(tmp_path) -> None:
    atoms = _atoms(3)
    atoms.set_constraint(FixAtoms(indices=[0]))
    runner = _runner(_Wrapper(_ZeroCalculator()), tmp_path, com_policy="auto")

    runner._configure_com_policy(atoms)

    assert len(atoms.constraints) == 1
    assert isinstance(atoms.constraints[0], FixAtoms)
    assert runner.com_policy_effective == "none"


def test_fixcom_initialization_handles_mixed_masses(tmp_path) -> None:
    atoms = _atoms(4)
    atoms.set_masses([1.0, 2.0, 3.0, 4.0])
    runner = _runner(_Wrapper(_ZeroCalculator()), tmp_path, com_policy="constraint")

    runner._configure_com_policy(atoms)
    runner._initialize_velocities(atoms)

    assert np.sum(atoms.get_momenta(), axis=0) == pytest.approx(
        np.zeros(3), abs=1.0e-12
    )
    assert runner._calculate_temperature(atoms) == pytest.approx(300.0)


def test_fixcartesian_coordinate_and_velocity_remain_frozen(tmp_path) -> None:
    atoms = _atoms(3)
    frozen_x = float(atoms.positions[0, 0])
    atoms.set_constraint(FixCartesian(0, mask=(True, False, False)))
    runner = _runner(_Wrapper(_ZeroCalculator()), tmp_path, com_policy="auto", steps=3)

    results = runner.run(atoms)

    with Trajectory(results["trajectory_path"]) as trajectory:
        for frame in trajectory:
            assert frame.positions[0, 0] == pytest.approx(frozen_x)
            assert frame.get_velocities()[0, 0] == pytest.approx(0.0, abs=1.0e-14)


def test_unvalidated_and_repeated_constraints_fail_closed_before_model_load(
    tmp_path,
) -> None:
    cases = []
    bond = _atoms(3)
    bond.set_constraint(FixBondLength(0, 1))
    cases.append((bond, "has not been independently validated"))
    repeated_com = _atoms(3)
    repeated_com.set_constraint([FixCom(), FixCom()])
    cases.append((repeated_com, "Repeated FixCom"))

    for index, (atoms, message) in enumerate(cases):
        wrapper = _Wrapper(_ZeroCalculator())
        runner = _runner(wrapper, tmp_path / str(index), com_policy="auto")
        with pytest.raises(ValueError, match=message):
            runner.run(atoms)
        assert wrapper.load_calls == 0


@pytest.mark.parametrize(
    ("ensemble", "thermostat", "label"),
    [("NVE", "LANGEVIN", "NVE"), ("NVT", "BUSSI", "BUSSI")],
)
def test_deterministic_integrators_reject_initialize_only_com_policy(
    tmp_path, ensemble, thermostat, label
) -> None:
    wrapper = _Wrapper(_ZeroCalculator())
    runner = _runner(
        wrapper,
        tmp_path,
        ensemble=ensemble,
        thermostat=thermostat,
        com_policy="initialize_only",
    )

    with pytest.raises(ValueError, match=rf"{label}.*initialize_only"):
        runner.run(_atoms(4))
    assert wrapper.load_calls == 0


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": float("inf")}, "temperature"),
        ({"timestep": float("nan")}, "timestep"),
        ({"timestep": float("inf")}, "timestep"),
        ({"friction": float("nan")}, "friction"),
        ({"thermostat": "BUSSI", "bussi_tau": float("inf")}, "bussi_tau"),
        (
            {
                "thermostat": "NHC",
                "com_policy": "none",
                "nhc_tdamp": float("nan"),
            },
            "nhc_tdamp",
        ),
        ({"fmax_abort": float("nan")}, "fmax_abort"),
        ({"pre_relax_fmax": float("inf")}, "pre_relax_fmax"),
        ({"steps": True}, "steps"),
        ({"steps": 3.7}, "steps"),
        ({"equilibration_steps": False}, "equilibration_steps"),
        ({"save_interval": 2.5}, "save_interval"),
        ({"pre_relax_steps": True}, "pre_relax_steps"),
        ({"nhc_tchain": 2.2}, "nhc_tchain"),
        ({"nhc_tloop": True}, "nhc_tloop"),
        ({"seed": 1.5}, "seed"),
    ],
)
def test_md_rejects_nonfinite_scalars_and_noninteger_counts(
    tmp_path, kwargs, name
) -> None:
    with pytest.raises((TypeError, ValueError), match=name):
        _runner(_Wrapper(_ZeroCalculator()), tmp_path, **kwargs)


def _invalid_atoms(case: str) -> Atoms:
    if case == "empty":
        return Atoms(cell=np.eye(3), pbc=True)
    atoms = _atoms(2)
    if case == "positions":
        atoms.positions[0, 0] = np.nan
    elif case == "cell":
        atoms.cell[0, 0] = np.inf
    elif case == "mass-zero":
        atoms.set_masses([0.0, 1.0])
    elif case == "mass-negative":
        atoms.set_masses([-1.0, 1.0])
    elif case == "mass-nan":
        atoms.set_masses([np.nan, 1.0])
    elif case == "momenta":
        momenta = np.zeros((2, 3))
        momenta[0, 0] = np.inf
        atoms.set_momenta(momenta)
    else:  # pragma: no cover - protects the test helper itself
        raise AssertionError(case)
    return atoms


@pytest.mark.parametrize(
    "case",
    ["empty", "positions", "cell", "mass-zero", "mass-negative", "mass-nan", "momenta"],
)
def test_invalid_atomic_state_fails_before_model_load_or_raw_output(
    tmp_path, case
) -> None:
    wrapper = _Wrapper(_ZeroCalculator())
    runner = _runner(wrapper, tmp_path)

    with pytest.raises(ValueError, match="atoms|positions|cell|mass|momenta"):
        runner.run(_invalid_atoms(case))

    assert wrapper.load_calls == 0
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "artifacts.json").exists()


def test_zero_dof_fails_before_model_load(tmp_path) -> None:
    atoms = _atoms(2)
    atoms.set_constraint(FixAtoms(indices=range(len(atoms))))
    wrapper = _Wrapper(_ZeroCalculator())
    runner = _runner(wrapper, tmp_path, com_policy="auto")

    with pytest.raises(ValueError, match="zero.*degrees of freedom|0 DOF"):
        runner.run(atoms)
    assert wrapper.load_calls == 0
    assert not (tmp_path / "raw").exists()


def test_single_atom_auto_fails_but_explicit_no_com_is_well_defined(tmp_path) -> None:
    atom = _atoms(1)
    automatic = _runner(_Wrapper(_ZeroCalculator()), tmp_path / "auto")
    with pytest.raises(ValueError, match="single atom.*com_policy='none'"):
        automatic.run(atom)

    explicit = _runner(
        _Wrapper(_ZeroCalculator()),
        tmp_path / "none",
        com_policy="none",
        steps=1,
    )
    results = explicit.run(atom)
    assert np.isfinite(results["temperature"])


def test_nonperiodic_molecule_without_cell_has_well_defined_md(tmp_path) -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]])
    wrapper = _Wrapper(_ZeroCalculator())
    wrapper.task = "molecule"
    runner = _runner(
        wrapper,
        tmp_path,
        com_policy="none",
        steps=1,
    )

    results = runner.run(atoms)

    assert results["trajectory"][0]["volume"] > 0.0
    assert results["configurational_stress"] is None
    assert results["total_pressure_gpa"] is None


def test_periodic_system_requires_independent_cell_vectors_before_model_load(
    tmp_path,
) -> None:
    atoms = _atoms(2)
    atoms.cell[2] = atoms.cell[1]
    wrapper = _Wrapper(_ZeroCalculator())
    runner = _runner(wrapper, tmp_path)

    with pytest.raises(ValueError, match="periodic cell vectors"):
        runner.run(atoms)

    assert wrapper.load_calls == 0
    assert not (tmp_path / "raw").exists()


def test_nonfinite_stress_aborts_before_success_is_published(tmp_path) -> None:
    wrapper = _Wrapper(_ZeroCalculator(stress=np.full(6, np.nan)), has_stress=True)
    runner = _runner(wrapper, tmp_path, steps=0, write_stress=True)

    with pytest.raises(RuntimeError, match="Non-finite.*stress"):
        runner.run(_atoms(3))

    assert not (tmp_path / "raw" / "mlipx_results.json").exists()
    assert not (tmp_path / "artifacts.json").exists()


@pytest.mark.parametrize(
    "calculator",
    [
        _ZeroCalculator(energy=float("nan")),
        _ZeroCalculator(forces=np.full((3, 3), float("inf"))),
    ],
    ids=("energy", "forces"),
)
def test_nonfinite_model_results_abort_before_success_is_published(
    tmp_path, calculator
) -> None:
    runner = _runner(
        _Wrapper(calculator),
        tmp_path,
        steps=0,
        write_json=True,
    )

    with pytest.raises(RuntimeError, match="Non-finite"):
        runner.run(_atoms(3))

    assert not (tmp_path / "raw" / "mlipx_results.json").exists()
    assert not (tmp_path / "artifacts.json").exists()


def test_json_writer_never_serializes_nonstandard_nan_tokens(tmp_path) -> None:
    path = tmp_path / "nonfinite.json"
    atoms = _atoms(1)
    with pytest.raises(ValueError, match="JSON compliant"):
        JsonWriter().write(
            atoms,
            {"energy": float("nan"), "forces": np.zeros((1, 3))},
            path,
        )
    assert "NaN" not in path.read_text()


def _run_harmonic(tmp_path, *, timestep_fs: float, steps: int) -> tuple[float, float]:
    center = np.array([[6.0, 6.0, 6.0]])
    displacement = 0.2
    atoms = Atoms(
        "H",
        positions=center + np.array([[displacement, 0.0, 0.0]]),
        cell=np.eye(3) * 12.0,
        pbc=True,
    )
    atoms.set_momenta(np.zeros((1, 3)))
    spring_constant = 1.0
    runner = _runner(
        _Wrapper(_HarmonicCalculator(center, spring_constant)),
        tmp_path,
        com_policy="none",
        temperature=0.0,
        timestep=timestep_fs,
        steps=steps,
        velocity_policy="preserve",
    )

    results = runner.run(atoms)
    final_x = Trajectory(results["trajectory_path"])[-1].positions[0, 0]
    elapsed_ase_time = steps * timestep_fs * units.fs
    omega = np.sqrt(spring_constant / atoms.get_masses()[0])
    exact_x = center[0, 0] + displacement * np.cos(omega * elapsed_ase_time)
    energies = np.asarray([frame["total_energy"] for frame in results["trajectory"]])
    relative_energy_span = float(np.ptp(energies) / energies[0])
    return abs(final_x - exact_x), relative_energy_span


def test_nve_harmonic_oscillator_has_second_order_known_answer(tmp_path) -> None:
    coarse_error, coarse_energy_span = _run_harmonic(
        tmp_path / "coarse", timestep_fs=1.0, steps=20
    )
    fine_error, fine_energy_span = _run_harmonic(
        tmp_path / "fine", timestep_fs=0.5, steps=40
    )

    assert coarse_error > 0.0
    assert fine_error < coarse_error / 3.5
    assert coarse_energy_span < 0.01
    assert fine_energy_span < coarse_energy_span / 3.5


@pytest.mark.parametrize("thermostat", ["LANGEVIN", "BUSSI"])
def test_supported_constrained_thermostats_have_short_finite_sanity(
    tmp_path, thermostat
) -> None:
    atoms = _atoms(8)
    frozen_position = atoms.positions[0].copy()
    atoms.set_constraint(FixAtoms(indices=[0]))
    runner = _runner(
        _Wrapper(_ZeroCalculator()),
        tmp_path / thermostat,
        ensemble="NVT",
        thermostat=thermostat,
        com_policy="auto",
        temperature=300.0,
        friction=0.01,
        bussi_tau=50.0,
        timestep=0.25,
        steps=12,
    )

    results = runner.run(atoms)
    temperatures = np.asarray([frame["temperature"] for frame in results["trajectory"]])

    assert np.all(np.isfinite(temperatures))
    assert 50.0 < float(np.mean(temperatures)) < 1000.0
    with Trajectory(results["trajectory_path"]) as trajectory:
        assert all(
            frame.positions[0] == pytest.approx(frozen_position) for frame in trajectory
        )
