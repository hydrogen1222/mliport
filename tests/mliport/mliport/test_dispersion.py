"""DFT-D3 dispersion on top of an MLIP backend (``--d3`` / ``D3``).

The correction is ``E_model + E_D3`` (and the same for forces and stress), so
the known-answer tests below use fake model and D3 calculators with exact
values and check that the sum, the D3 settings and the provenance all reach
the calculation.  Guard tests check that D3 is refused where it would count
dispersion twice or would not execute.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mliport.backend_selection import CALCULATOR_OPTION_KEYS
from mliport.base_calculator import BaseMLIPCalculator
from mliport.calculators.dispersion import (
    D3CorrectedCalculator,
    d3_device,
    normalize_d3_options,
)
from mliport.calculators.factory import CalculatorFactory
from mliport.calculators.mace_calc import MACECalculatorWrapper
from mliport.cli import _build_cli_opts, create_parser
from mliport.config import get_schema, resolve_config
from mliport.config.resolver import ResolvedConfig
from mliport.queue import build_mliport_command
from mliport.writers.outcar import _dispersion_label

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _ConstantCalculator(Calculator):
    """Energy, forces and stress that do not depend on the geometry."""

    implemented_properties = ("energy", "forces", "stress")

    def __init__(self, energy: float, force: float, stress: float, **kwargs):
        super().__init__()
        self._energy = energy
        self._force = force
        self._stress = stress
        self.init_kwargs = kwargs

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        self.results = {
            "energy": self._energy,
            "free_energy": self._energy,
            "forces": np.full((n, 3), self._force),
            "stress": np.full(6, self._stress),
        }


class _FakeBackend(BaseMLIPCalculator):
    def __init__(self):
        self.calc = _ConstantCalculator(energy=-10.0, force=0.1, stress=0.01)

    def get_calculator(self):
        return self.calc

    @property
    def task(self):
        return "bulk"

    @property
    def has_stress(self):
        return True

    def info(self):
        return {"model_type": "mace", "model_path": "m.model", "task": "bulk"}


@pytest.fixture()
def fake_torch_dftd(monkeypatch):
    """Install a fake ``torch_dftd`` whose D3 term is a known constant."""
    created: list[_ConstantCalculator] = []

    def factory(**kwargs):
        calc = _ConstantCalculator(energy=-0.5, force=0.02, stress=0.003, **kwargs)
        created.append(calc)
        return calc

    package = ModuleType("torch_dftd")
    module = ModuleType("torch_dftd.torch_dftd3_calculator")
    module.TorchDFTD3Calculator = factory
    monkeypatch.setitem(sys.modules, "torch_dftd", package)
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float64="float64"))
    return created


def _atoms() -> Atoms:
    return Atoms(
        "Li2",
        positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
        cell=[3.0, 3.0, 3.0],
        pbc=True,
    )


# ---------------------------------------------------------------------------
# known answers: the correction is added, with the requested parameters
# ---------------------------------------------------------------------------


def test_energy_forces_and_stress_are_model_plus_d3(fake_torch_dftd):
    wrapper = D3CorrectedCalculator(
        _FakeBackend(), device="cpu", xc="pbe", damping="bj"
    )
    atoms = _atoms()
    atoms.calc = wrapper.get_calculator()

    assert atoms.get_potential_energy() == pytest.approx(-10.5)
    np.testing.assert_allclose(atoms.get_forces(), np.full((2, 3), 0.12))
    np.testing.assert_allclose(atoms.get_stress(), np.full(6, 0.013))


def test_d3_term_receives_xc_damping_device_and_float64(fake_torch_dftd):
    wrapper = D3CorrectedCalculator(
        _FakeBackend(), device="cuda:1", xc="rpbe", damping="zero"
    )
    wrapper.get_calculator()

    assert len(fake_torch_dftd) == 1
    assert fake_torch_dftd[0].init_kwargs == {
        "device": "cuda:1",
        "xc": "rpbe",
        "damping": "zero",
        "dtype": "float64",
    }


def test_provenance_records_the_dispersion_term(fake_torch_dftd):
    wrapper = D3CorrectedCalculator(_FakeBackend(), device="cpu")
    wrapper.get_calculator()
    info = wrapper.info()

    assert info["model_type"] == "mace"
    assert info["dispersion"]["method"] == "DFT-D3"
    assert info["dispersion"]["xc"] == "pbe"
    assert info["dispersion"]["damping"] == "bj"
    assert info["dispersion"]["implementation"] == "torch-dftd"
    assert wrapper.task == "bulk"
    assert wrapper.has_stress is True
    assert _dispersion_label(info) == "DFT-D3(BJ), xc=pbe (torch-dftd)"


def test_missing_torch_dftd_is_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", None)
    wrapper = D3CorrectedCalculator(_FakeBackend(), device="cpu")

    with pytest.raises(ImportError, match="torch-dftd"):
        wrapper.get_calculator()


# ---------------------------------------------------------------------------
# guards: never add dispersion twice, never record an option that does not run
# ---------------------------------------------------------------------------


def test_d3_is_off_by_default():
    assert normalize_d3_options("mace", "bulk") is None
    assert normalize_d3_options("mace", "bulk", d3="false") is None


def test_d3_defaults_to_pbe_with_bj_damping():
    assert normalize_d3_options("mace", "bulk", d3=True) == {
        "xc": "pbe",
        "damping": "bj",
    }
    assert normalize_d3_options("dpa", "bulk", d3=".TRUE.", d3_xc="PBEsol") == {
        "xc": "pbesol",
        "damping": "bj",
    }


@pytest.mark.parametrize("option", ["d3_xc", "d3_damping"])
def test_d3_settings_without_d3_are_rejected(option):
    with pytest.raises(ValueError, match="not enabled"):
        normalize_d3_options("mace", "bulk", **{option: "bj"})


def test_unknown_damping_is_rejected():
    with pytest.raises(ValueError, match="d3_damping"):
        normalize_d3_options("mace", "bulk", d3=True, d3_damping="becke")


@pytest.mark.parametrize("task", ["omol", "omc", "odac", "oc25"])
def test_uma_tasks_with_dispersion_reject_d3(task):
    with pytest.raises(ValueError, match="not allowed for UMA task"):
        normalize_d3_options("uma", task, d3=True)


def test_uma_task_must_use_its_reference_functional():
    assert normalize_d3_options("uma", "omat", d3=True)["xc"] == "pbe"
    assert normalize_d3_options("uma", "oc20", d3=True, d3_xc="rpbe")["xc"] == "rpbe"
    with pytest.raises(ValueError, match="trained on PBE"):
        normalize_d3_options("uma", "omat", d3=True, d3_xc="rpbe")
    with pytest.raises(ValueError, match="trained on RPBE"):
        normalize_d3_options("uma", "oc20", d3=True)


def test_d3_device_follows_the_backend_device():
    assert d3_device("cpu", process_isolated=False) == "cpu"
    assert d3_device("gpu", process_isolated=False) == "cuda"
    assert d3_device("cuda:2", process_isolated=False) == "cuda:2"
    # DPA/GRACE run with only the selected GPU visible, which torch sees as 0.
    assert d3_device("cuda:2", process_isolated=True) == "cuda"


# ---------------------------------------------------------------------------
# plumbing: factory, schema/INCAR, resolver, CLI, queue, NEB
# ---------------------------------------------------------------------------


def test_factory_wraps_the_backend_only_when_requested(tmp_path):
    model = tmp_path / "m.model"
    model.write_bytes(b"never loaded")

    plain = CalculatorFactory.create("mace", model, task="bulk", strict=True)
    corrected = CalculatorFactory.create(
        "mace", model, task="bulk", strict=True, d3=True, d3_damping="zero"
    )

    assert isinstance(plain, MACECalculatorWrapper)
    assert isinstance(corrected, D3CorrectedCalculator)
    assert isinstance(corrected.base, MACECalculatorWrapper)
    assert corrected.dispersion_info()["damping"] == "zero"
    assert corrected.dispersion_info()["xc"] == "pbe"


def test_factory_rejects_d3_for_uma_omol_before_loading(tmp_path):
    model = tmp_path / "uma.pt"
    model.write_bytes(b"never loaded")

    with pytest.raises(ValueError, match="not allowed for UMA task 'omol'"):
        CalculatorFactory.create("uma", model, task="omol", strict=True, d3=True)


def test_every_backend_accepts_the_d3_options():
    for keys in CALCULATOR_OPTION_KEYS.values():
        assert {"d3", "d3_xc", "d3_damping"} <= keys


def test_incar_keys_are_known_to_the_schema():
    errors = get_schema().validate_dict(
        {"D3": ".TRUE.", "D3_XC": "pbe", "D3_DAMPING": "bj"},
        strict=True,
        context="INCAR",
    )
    assert errors == []
    assert get_schema().validate_dict(
        {"D3_DAMPING": "becke"}, strict=True, context="INCAR"
    )


def test_resolver_routes_d3_to_calculator_options():
    resolved = resolve_config(
        calc_type="opt",
        cli={
            "model_type": "mace",
            "model_path": "m.model",
            "d3": True,
            "d3_damping": "bj",
        },
    )
    assert resolved.calculator_options["d3"] is True
    assert resolved.calculator_options["d3_damping"] == "bj"
    assert "d3" not in resolved.run_options


def test_cli_flags_reach_the_resolver_options():
    args = create_parser().parse_args(
        [
            "opt",
            "slab.xyz",
            "--model",
            "m.model",
            "--model-type",
            "mace",
            "--d3",
            "--d3-xc",
            "pbe",
            "--d3-damping",
            "bj",
        ]
    )
    opts = _build_cli_opts(args, "opt")
    assert opts["d3"] is True
    assert opts["d3_xc"] == "pbe"
    assert opts["d3_damping"] == "bj"

    plain = create_parser().parse_args(
        ["opt", "slab.xyz", "--model", "m.model", "--model-type", "mace"]
    )
    assert "d3" not in _build_cli_opts(plain, "opt")


def test_queue_command_forwards_d3():
    cmd = build_mliport_command(
        "opt",
        "/s/slab.xyz",
        "/m/m.model",
        model_type="mace",
        options={"d3": True, "d3_damping": "bj"},
    )
    assert "--d3" in cmd
    assert cmd[cmd.index("--d3-damping") + 1] == "bj"


def test_neb_refuses_d3():
    from mliport.neb.schema import NEBPreparationError
    from mliport.neb.workflow import _run_neb_workflow_locked

    resolved = ResolvedConfig(
        calc_type="neb",
        model_type="mace",
        model_path="m.model",
        task="bulk",
        device="cpu",
        inference_mode="default",
        calculator_options={"d3": True},
    )
    with pytest.raises(NEBPreparationError, match="not supported for NEB"):
        _run_neb_workflow_locked(object(), resolved, output_dir="unused")
