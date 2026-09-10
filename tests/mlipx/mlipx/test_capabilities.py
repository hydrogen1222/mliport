from __future__ import annotations

import json

import pytest

from mlipx.capabilities import (
    IDENTITY_KEYS,
    CalculatorCapabilities,
    model_identity,
    require_neb_capability,
    resolve_capabilities,
)


def test_unknown_requires_explicit_experimental_override():
    capability = CalculatorCapabilities(True, True, False)
    with pytest.raises(ValueError, match="allow-unvalidated-neb"):
        require_neb_capability(capability, experimental=False)
    require_neb_capability(capability, experimental=True)


def test_direct_force_model_is_always_rejected():
    capability = resolve_capabilities({"direct_forces": True}, ["energy", "forces"])
    with pytest.raises(ValueError, match="non-guaranteed"):
        require_neb_capability(capability, experimental=True)


@pytest.mark.parametrize("changed", IDENTITY_KEYS)
def test_evidence_is_exact_not_a_backend_wide_claim(tmp_path, changed):
    model = {key: f"value-{key}" for key in IDENTITY_KEYS}
    record = {
        "model_identity": model_identity(model),
        "evidence_id": "known-answer",
        "workloads": {
            key: {"status": "passed"}
            for key in ["single_point", "shared_calculator", "energy_force_consistency"]
        },
    }
    (tmp_path / "evidence.json").write_text(json.dumps(record))
    assert (
        resolve_capabilities(
            model, ["energy", "forces"], evidence_dir=tmp_path
        ).energy_gradient_consistency
        == "validated"
    )
    model[changed] = "different"
    assert (
        resolve_capabilities(
            model, ["energy", "forces"], evidence_dir=tmp_path
        ).energy_gradient_consistency
        == "unknown"
    )


def test_validated_without_evidence_id_is_invalid():
    with pytest.raises(ValueError, match="evidence ID"):
        CalculatorCapabilities(True, True, False, "validated")
