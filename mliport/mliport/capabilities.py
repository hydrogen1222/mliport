"""What a loaded calculator provides, for NEB's structural gate.

NEB needs ``forces = -grad(energy)``.  Whether a model satisfies that is not
looked up in recorded evidence: models that obtain forces by automatic
differentiation of the energy are consistent by construction, direct-force
models are not, and NEB measures the property on the actual band before the
first step (:mod:`mliport.neb.preflight`).  This module only reports what the
calculator advertises and rejects what can never work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Fields that identify the exact model/runtime in run provenance records.
IDENTITY_KEYS = (
    "model_type",
    "model_sha256",
    "backend_version",
    "framework_versions",
    "task",
    "head_effective",
    "dtype_effective",
    "inference_mode",
    "compile_enabled",
    "actual_device_type",
)


@dataclass(frozen=True)
class CalculatorCapabilities:
    energy: bool
    forces: bool
    stress: bool
    #: ``not_guaranteed`` for direct-force models; ``unknown`` otherwise (the
    #: runtime pre-flight measures it before NEB starts).
    energy_gradient_consistency: Literal["unknown", "not_guaranteed"] = "unknown"


def model_identity(model: dict) -> dict:
    return {key: model.get(key) for key in IDENTITY_KEYS}


def resolve_capabilities(model: dict, properties) -> CalculatorCapabilities:
    consistency = "not_guaranteed" if model.get("direct_forces") is True else "unknown"
    return CalculatorCapabilities(
        "energy" in properties,
        "forces" in properties,
        "stress" in properties,
        consistency,
    )


def require_neb_capability(capability: CalculatorCapabilities) -> None:
    if not capability.energy or not capability.forces:
        raise ValueError("NEB requires energy and forces")
    if capability.energy_gradient_consistency == "not_guaranteed":
        raise ValueError(
            "NEB rejects a model with non-guaranteed energy gradients "
            "(its forces are predicted directly, not derived from the energy)"
        )
