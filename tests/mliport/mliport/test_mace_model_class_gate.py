"""MACE checkpoints that cannot serve as a potential-energy model fail closed.

Spec #61 (model-support §6.3). The decision is made from the loaded model
object: class name and the attributes mace-torch 0.3.16 sets, never the file
name. These tests use stub models, so no mace-torch, weights or GPU are needed.
"""

from __future__ import annotations

import sys
import types

import pytest

from mliport.calculators.mace_calc import (
    MACECalculatorWrapper,
    unsupported_mace_model_reason,
)


def _stub(class_name: str, **attributes):
    return type(class_name, (), attributes)()


def _install_fake_mace(monkeypatch, model):
    """Put a fake ``mace.calculators.MACECalculator`` in ``sys.modules``."""
    calculators = types.ModuleType("mace.calculators")

    class FakeMACECalculator:
        def __init__(self, **kwargs):
            self.models = [model]
            self.available_heads = list(getattr(model, "heads", ["Default"]))
            requested = kwargs.get("head")
            self.head = requested or self.available_heads[0]
            self.r_max = 5.0
            self.implemented_properties = ["energy", "forces", "stress"]

    calculators.MACECalculator = FakeMACECalculator
    package = types.ModuleType("mace")
    package.calculators = calculators
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)


# ---------------------------------------------------------- the pure gate
def test_ordinary_energy_models_pass_the_gate():
    assert (
        unsupported_mace_model_reason(_stub("ScaleShiftMACE", heads=["Default"]))
        is None
    )
    assert unsupported_mace_model_reason(_stub("MACE", heads=["Default"])) is None


def test_mace_mh1_multihead_model_is_not_gated():
    """MACE-MH-1 loads as ScaleShiftMACE with six heads and no embeddings."""
    mh1 = _stub(
        "ScaleShiftMACE",
        heads=[
            "matpes_r2scan",
            "mp_pbe_refit_add",
            "spice_wB97M",
            "oc20_usemppbe",
            "omol",
            "omat_pbe",
        ],
    )
    assert unsupported_mace_model_reason(mh1) is None


@pytest.mark.parametrize("class_name", ["AtomicDipolesMACE", "AtomicDielectricMACE"])
def test_dipole_and_polarizability_only_models_are_rejected(class_name):
    reason = unsupported_mace_model_reason(_stub(class_name))
    assert reason is not None
    assert "dipole/polarizability" in reason
    assert "model_type='DipoleMACE'" in reason


def test_polar_model_is_rejected_by_class_name():
    reason = unsupported_mace_model_reason(_stub("PolarMACE"))
    assert reason is not None
    assert "MACE-POLAR" in reason
    assert "mace_polar" in reason
    assert "PolarMACE" in reason


def test_polar_model_is_rejected_by_its_attributes_too():
    """A renamed class is still caught by the attributes PolarMACE sets."""
    polar = _stub(
        "ScaleShiftMACE",
        atomic_multipoles_max_l=2,
        charges_irreps="2x0e+2x1o",
        heads=["Default"],
    )
    reason = unsupported_mace_model_reason(polar)
    assert reason is not None
    assert "mace_polar" in reason


def test_charge_spin_embedding_model_is_rejected():
    omol = _stub(
        "MACE",
        heads=["omat", "omol"],
        embedding_specs={"charge": {"per": "graph"}, "spin": {"per": "graph"}},
        joint_embedding=object(),
    )
    reason = unsupported_mace_model_reason(omol)
    assert reason is not None
    assert "charge/spin" in reason
    assert "embedding_specs" in reason
    assert "atom" in reason  # atoms.info is where upstream reads charge and spin


def test_joint_embedding_without_specs_is_still_rejected():
    omol = _stub("MACE", joint_embedding=object())
    assert unsupported_mace_model_reason(omol) is not None


# ------------------------------------------------------- wired into the wrapper
def test_wrapper_raises_for_a_dipole_only_checkpoint(tmp_path, monkeypatch):
    model = tmp_path / "mace-mdp.model"
    model.write_bytes(b"stub")
    _install_fake_mace(monkeypatch, _stub("AtomicDipolesMACE"))
    wrapper = MACECalculatorWrapper(model, device="cpu")
    with pytest.raises(ValueError, match="Unsupported MACE model: AtomicDipolesMACE"):
        wrapper.get_calculator()
    # The failed load must not leave a half-initialised calculator behind.
    assert wrapper._calculator is None


def test_wrapper_raises_for_charge_spin_embedding_checkpoint(tmp_path, monkeypatch):
    model = tmp_path / "mace-omol.model"
    model.write_bytes(b"stub")
    _install_fake_mace(
        monkeypatch,
        _stub(
            "MACE",
            embedding_specs={"charge": {"per": "graph"}},
            joint_embedding=object(),
        ),
    )
    wrapper = MACECalculatorWrapper(model, device="cpu")
    with pytest.raises(ValueError, match="charge/spin"):
        wrapper.get_calculator()


def test_wrapper_still_loads_a_multihead_energy_model(tmp_path, monkeypatch):
    model = tmp_path / "mace-mh-1.model"
    model.write_bytes(b"stub")
    _install_fake_mace(
        monkeypatch,
        _stub("ScaleShiftMACE", heads=["omat_pbe", "matpes_r2scan"]),
    )
    wrapper = MACECalculatorWrapper(model, device="cpu", head="matpes_r2scan")
    calculator = wrapper.get_calculator()
    assert calculator.head == "matpes_r2scan"

    # An implicit head on a multihead model is still refused (unchanged rule).
    with pytest.raises(ValueError, match="poses multiple energy heads"):
        MACECalculatorWrapper(model, device="cpu").get_calculator()
