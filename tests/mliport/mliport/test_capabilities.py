from __future__ import annotations

import pytest

from mliport.capabilities import (
    IDENTITY_KEYS,
    CalculatorCapabilities,
    model_identity,
    require_neb_capability,
    resolve_capabilities,
)


def test_autograd_style_model_is_unknown_and_passes_the_structural_gate():
    capability = resolve_capabilities({}, ["energy", "forces"])
    assert capability == CalculatorCapabilities(True, True, False, "unknown")
    require_neb_capability(capability)  # measured by the runtime pre-flight


def test_direct_force_model_is_always_rejected():
    capability = resolve_capabilities({"direct_forces": True}, ["energy", "forces"])
    assert capability.energy_gradient_consistency == "not_guaranteed"
    with pytest.raises(ValueError, match="non-guaranteed"):
        require_neb_capability(capability)


@pytest.mark.parametrize("properties", [["energy"], ["forces"], []])
def test_neb_requires_energy_and_forces(properties):
    capability = resolve_capabilities({}, properties)
    with pytest.raises(ValueError, match="energy and forces"):
        require_neb_capability(capability)


def test_stress_is_reported():
    assert resolve_capabilities({}, ["energy", "forces", "stress"]).stress is True
    assert resolve_capabilities({}, ["energy", "forces"]).stress is False


def test_model_identity_projects_exactly_the_identity_keys():
    model = {key: f"value-{key}" for key in IDENTITY_KEYS}
    model["extra"] = "ignored"
    assert model_identity(model) == {key: f"value-{key}" for key in IDENTITY_KEYS}
    assert model_identity({}) == dict.fromkeys(IDENTITY_KEYS)
