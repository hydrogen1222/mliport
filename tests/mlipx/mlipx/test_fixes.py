"""Regression tests for the UMA/MACE audit fixes.

Covers:
* A1 - MACE ``--head`` is forwarded as ``head=`` (singular string), not ``heads=``.
* A3 - MD temperature DOF uses the real constraint removed-DOF count (FixAtoms).
* B1 - MACE ``info()`` reads the correct MACE attributes.
* B2 - runners abort on NaN/inf energy/forces instead of writing "successful" NaN.
* B3 - NVE defaults to no pre-relaxation; NVT keeps it; explicit override wins.
* B4 - MACE dtype default is float64 for all calc types (accuracy first).
* B5 - stress is skipped for non-periodic systems even when ``has_stress`` is True.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms


# ---------------------------------------------------------------------------
# A1 + B1: MACE wrapper head kwarg + info() attributes
# ---------------------------------------------------------------------------
def _install_fake_mace(monkeypatch, mace_calculator_mock):
    """Inject a fake ``mace.calculators.MACECalculator`` into sys.modules."""
    mace_pkg = types.ModuleType("mace")
    calculators = types.ModuleType("mace.calculators")
    calculators.MACECalculator = mace_calculator_mock
    mace_pkg.calculators = calculators
    monkeypatch.setitem(sys.modules, "mace", mace_pkg)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)


def test_mace_head_forwarded_as_singular_string(tmp_path, monkeypatch):
    """A1: ``head`` must reach MACECalculator as ``head=<str>``, not ``heads=[...]``."""
    from mlipx.calculators.mace_calc import MACECalculatorWrapper

    captured = {}

    def fake_mace_calculator(**kwargs):
        captured.update(kwargs)
        m = Mock()
        m.implemented_properties = ["energy", "forces", "stress"]
        return m

    _install_fake_mace(monkeypatch, fake_mace_calculator)
    monkeypatch.setattr(
        "mlipx.doctor._installed_dependency_conflicts", lambda names: []
    )

    model = tmp_path / "mace.model"
    model.write_text("x")
    w = MACECalculatorWrapper(model, device="cpu", default_dtype="float64", head="omol")
    w.get_calculator()

    assert captured.get("head") == "omol"  # singular string
    assert "heads" not in captured  # plural list must NOT be passed


def test_mace_info_reads_correct_attributes(tmp_path):
    """B1: info() enrichment must use MACE's real attribute names."""
    from mlipx.calculators.mace_calc import MACECalculatorWrapper

    model = tmp_path / "mace.model"
    model.write_text("x")
    w = MACECalculatorWrapper(model, default_dtype="float64")

    fake_model = Mock()
    fake_model.r_max = 5.0
    fake_model.atomic_numbers = [1, 8]
    fake_calc = Mock()
    fake_calc.models = [fake_model]
    fake_calc.available_heads = ["default", "omol"]
    fake_calc.head = "omol"
    fake_calc.z_table = type("Z", (), {"zs": [1, 8]})()
    fake_calc.implemented_properties = ["energy", "forces", "stress"]
    w._calculator = fake_calc

    info = w.info()
    assert info["cutoff"] == 5.0
    assert info["num_elements"] == 2
    assert info["available_heads"] == ["default", "omol"]
    assert info["active_head"] == "omol"
    assert info["elements"] == ["H", "O"]
    assert info["num_models"] == 1


# ---------------------------------------------------------------------------
# B4: MACE dtype default is float64 for all calc types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "calc_type, expected_dtype",
    [("sp", "float64"), ("opt", "float64"), ("md", "float64")],
)
def test_mace_dtype_default_is_float64_all_calc_types(calc_type, expected_dtype):
    from mlipx.engine import CalculationEngine, EngineConfig

    config = EngineConfig(
        calc_type=calc_type,
        model_path=Path("mace.model"),
        model_type="mace",
        task="bulk",
        device="cpu",
        output_dir=Path("./results"),
    )
    with patch.object(Path, "exists", return_value=True):
        engine = CalculationEngine.from_config(config)

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return Mock()

    with patch(
        "mlipx.calculators.factory.CalculatorFactory.create", side_effect=fake_create
    ):
        engine._create_calculator()

    assert captured["default_dtype"] == expected_dtype


def test_mace_dtype_explicit_override_wins():
    """An explicit default_dtype from the config layer must not be overridden."""
    from mlipx.engine import CalculationEngine, EngineConfig

    config = EngineConfig(
        calc_type="md",
        model_path=Path("mace.model"),
        model_type="mace",
        task="bulk",
        device="cpu",
        output_dir=Path("./results"),
        calculator_options={"default_dtype": "float64"},
    )
    with patch.object(Path, "exists", return_value=True):
        engine = CalculationEngine.from_config(config)

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return Mock()

    with patch(
        "mlipx.calculators.factory.CalculatorFactory.create", side_effect=fake_create
    ):
        engine._create_calculator()

    assert captured["default_dtype"] == "float64"


# ---------------------------------------------------------------------------
# B3: NVE defaults to no pre-relax; NVT keeps it; explicit override wins
# ---------------------------------------------------------------------------
class _CalcStub:
    inference_mode = "turbo"


def test_engine_nve_defaults_pre_relax_off():
    from mlipx.engine import CalculationEngine, EngineConfig

    config = EngineConfig(
        calc_type="md",
        model_path=Path("uma.pt"),
        model_type="uma",
        task="omat",
        device="cpu",
        output_dir=Path("./results"),
        run_options={"ensemble": "NVE"},
    )
    with patch.object(Path, "exists", return_value=True):
        engine = CalculationEngine.from_config(config)
    runner = engine._create_runner(_CalcStub())
    assert runner.pre_relax is False


def test_engine_nvt_defaults_pre_relax_on():
    from mlipx.engine import CalculationEngine, EngineConfig

    config = EngineConfig(
        calc_type="md",
        model_path=Path("uma.pt"),
        model_type="uma",
        task="omat",
        device="cpu",
        output_dir=Path("./results"),
        run_options={"ensemble": "NVT"},
    )
    with patch.object(Path, "exists", return_value=True):
        engine = CalculationEngine.from_config(config)
    runner = engine._create_runner(_CalcStub())
    assert runner.pre_relax is True


def test_engine_explicit_pre_relax_wins_for_nve():
    from mlipx.engine import CalculationEngine, EngineConfig

    config = EngineConfig(
        calc_type="md",
        model_path=Path("uma.pt"),
        model_type="uma",
        task="omat",
        device="cpu",
        output_dir=Path("./results"),
        run_options={"ensemble": "NVE", "pre_relax": True},
    )
    with patch.object(Path, "exists", return_value=True):
        engine = CalculationEngine.from_config(config)
    runner = engine._create_runner(_CalcStub())
    assert runner.pre_relax is True


def test_resolver_path_nve_pre_relax_off():
    """B3 regression: the config resolver must NOT inject a blanket
    pre_relax=True that overrides the engine's NVE-aware default. The
    earlier B3 tests bypassed the resolver, so this guards the production
    (CLI / INCAR / API) path."""
    from mlipx.config.resolver import resolve_config
    from mlipx.engine import CalculationEngine, EngineConfig

    for ens, expect in [("NVE", False), ("NVT", True)]:
        resolved = resolve_config(
            calc_type="md",
            cli={"model_path": "uma-s-1.pt", "ensemble": ens},
        )
        # The resolver must not force pre_relax into run_options.
        assert "pre_relax" not in resolved.run_options
        ec = EngineConfig.from_resolved(resolved)
        engine = CalculationEngine.from_config(ec)
        runner = engine._create_runner(_CalcStub())
        assert runner.pre_relax is expect, f"{ens}: {runner.pre_relax}"


# ---------------------------------------------------------------------------
# A3: MD temperature DOF accounts for FixAtoms via get_removed_dof
# ---------------------------------------------------------------------------
def test_temperature_dof_accounts_for_fixatoms():
    from mlipx.runners.md import MDRunner

    atoms = Atoms(
        "Ar4",
        positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
        cell=[10, 10, 10],
        pbc=True,
    )
    # Delegates to ASE: ndof = 3N - sum(removed_dof). FixAtoms([0]) removes
    # 3 DOF, so ndof = 12 - 3 = 9 (ASE convention, matching force_temperature).
    atoms.set_constraint(FixAtoms(indices=[0]))
    # Give a known kinetic energy.
    atoms.set_velocities(np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    ke = atoms.get_kinetic_energy()

    runner = MDRunner(
        _CalcStub(), ensemble="NVT", output_dir=Path("./_tmp_md_test"), pre_relax=False
    )
    temp = runner._calculate_temperature(atoms)

    expected_dof = 3 * len(atoms) - 3 * 1  # 3N(12) - FixAtoms(3) = 9
    assert temp == pytest.approx(2 * ke / (expected_dof * units.kB))
    # Must match ASE's own temperature (same convention).
    assert temp == pytest.approx(atoms.get_temperature())


def test_temperature_fails_closed_for_constraint_without_dof_contract():
    """FixSymmetry cannot report DOF, so MD must not invent an approximation."""
    from ase.constraints import FixSymmetry
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
    )

    from mlipx.runners.md import MDRunner

    # A symmetric periodic cell so FixSymmetry can attach.
    atoms = Atoms(
        "NaCl", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, force_temp=True)
    Stationary(atoms, preserve_temperature=True)
    atoms.set_constraint(FixSymmetry(atoms))
    runner = MDRunner(
        _CalcStub(), ensemble="NVT", output_dir=Path("./_tmp_md_test"), pre_relax=False
    )
    with pytest.raises(NotImplementedError):
        atoms.get_temperature()
    with pytest.raises(ValueError, match="does not report removed degrees of freedom"):
        runner._calculate_temperature(atoms)


# ---------------------------------------------------------------------------
# B2: runners abort on NaN/inf
# ---------------------------------------------------------------------------
class _FiniteCalc(Calculator):
    implemented_properties = ["energy", "forces", "stress"]  # noqa: RUF012

    def __init__(self, energy, forces, stress=None):
        super().__init__()
        self._e = energy
        self._f = np.asarray(forces, dtype=float)
        self._s = stress

    def calculate(self, atoms, properties, system_changes):
        self.results = {
            "energy": self._e,
            "forces": self._f,
            "stress": self._s if self._s is not None else np.zeros(6),
        }


class _CalcWrapperStub:
    """Minimal BaseMLIPCalculator stub for the runners."""

    def __init__(self, calc, task="bulk", has_stress=True):
        self._calc = calc
        self._task = task
        self._has_stress = has_stress

    def get_calculator(self):
        return self._calc

    @property
    def task(self):
        return self._task

    @property
    def has_stress(self):
        return self._has_stress

    @property
    def implemented_properties(self):
        return list(self._calc.implemented_properties)

    def info(self):
        return {"model_type": "stub", "task": self._task}


def test_singlepoint_aborts_on_nan_energy(tmp_path):
    from mlipx.runners.singlepoint import SinglePointRunner

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True)
    calc = _FiniteCalc(float("nan"), np.zeros((2, 3)))
    wrapper = _CalcWrapperStub(calc, task="bulk", has_stress=True)
    runner = SinglePointRunner(wrapper, output_dir=tmp_path, verbose=False)
    with pytest.raises(RuntimeError, match="Non-finite energy"):
        runner.run(atoms)


def test_singlepoint_aborts_on_nan_forces(tmp_path):
    from mlipx.runners.singlepoint import SinglePointRunner

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True)
    forces = np.array([[float("nan"), 0, 0], [0, 0, 0]])
    calc = _FiniteCalc(-1.0, forces)
    wrapper = _CalcWrapperStub(calc, task="bulk", has_stress=True)
    runner = SinglePointRunner(wrapper, output_dir=tmp_path, verbose=False)
    with pytest.raises(RuntimeError, match="Non-finite forces"):
        runner.run(atoms)


def test_optimization_aborts_on_nan(tmp_path):
    """B2: optimization must abort on NaN energy/forces, not write NaN results."""
    from mlipx.runners.optimization import OptimizationRunner

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=True)
    calc = _FiniteCalc(float("nan"), np.full((2, 3), float("nan")))
    wrapper = _CalcWrapperStub(calc, task="bulk", has_stress=True)
    runner = OptimizationRunner(
        wrapper, fmax=0.05, max_steps=10, output_dir=tmp_path, verbose=False
    )
    with pytest.raises(RuntimeError, match="Non-finite"):
        runner.run(atoms)
    # No NaN results written to disk.
    assert not (tmp_path / "mlipx_results.json").exists()


# ---------------------------------------------------------------------------
# B5: stress skipped for non-periodic systems even when has_stress is True
# ---------------------------------------------------------------------------
def test_singlepoint_skips_stress_for_nonperiodic(tmp_path):
    from mlipx.runners.singlepoint import SinglePointRunner

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10], pbc=False)
    # MACE-style: advertises stress even for a molecule.
    calc = _FiniteCalc(-1.0, np.zeros((2, 3)), stress=np.full(6, 0.123))
    # Track whether stress was ever requested.
    orig_get_stress = atoms.get_stress
    called = {"stress": False}

    def tracking_get_stress(*a, **k):
        called["stress"] = True
        return orig_get_stress(*a, **k)

    atoms.get_stress = tracking_get_stress  # type: ignore[method-assign]
    wrapper = _CalcWrapperStub(calc, task="molecule", has_stress=True)
    runner = SinglePointRunner(wrapper, output_dir=tmp_path, verbose=False)
    results = runner.run(atoms)
    assert results["stress"] is None
    assert called["stress"] is False


def test_singlepoint_accepts_molecule_without_cell(tmp_path):
    """A gas molecule loaded from .xyz has no cell (volume 0); the runner must
    not reject it -- this is the path the isolated gas takes in an
    adsorption-energy calculation."""
    from mlipx.runners.singlepoint import SinglePointRunner

    # No cell at all, like ase.read('co2.xyz').
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    assert atoms.cell.volume == 0
    calc = _FiniteCalc(-1.0, np.zeros((2, 3)))
    wrapper = _CalcWrapperStub(calc, task="molecule", has_stress=True)
    runner = SinglePointRunner(wrapper, output_dir=tmp_path, verbose=False)
    results = runner.run(atoms)
    assert results["energy"] == -1.0
    assert results["stress"] is None


def test_singlepoint_rejects_periodic_without_cell(tmp_path):
    """A periodic system still requires a real cell."""
    from mlipx.runners.singlepoint import SinglePointRunner

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], pbc=True)
    calc = _FiniteCalc(-1.0, np.zeros((2, 3)))
    wrapper = _CalcWrapperStub(calc, task="bulk", has_stress=True)
    runner = SinglePointRunner(wrapper, output_dir=tmp_path, verbose=False)
    with pytest.raises(ValueError, match="Invalid cell"):
        runner.run(atoms)
