"""Workload capability claims bound to exact, reviewed validation evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    energy_gradient_consistency: Literal["validated", "unknown", "not_guaranteed"] = (
        "unknown"
    )
    validation_evidence_id: str | None = None

    def __post_init__(self):
        if (
            self.energy_gradient_consistency == "validated"
            and not self.validation_evidence_id
        ):
            raise ValueError("Validated capabilities require an evidence ID")


def model_identity(model: dict) -> dict:
    return {key: model.get(key) for key in IDENTITY_KEYS}


def resolve_capabilities(model: dict, properties, *, evidence_dir: Path | None = None):
    state = "not_guaranteed" if model.get("direct_forces") is True else "unknown"
    evidence_id = None
    directory = evidence_dir or Path(__file__).parent / "data" / "validation"
    if state != "not_guaranteed":
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("model_identity") != model_identity(model):
                continue
            workloads = record.get("workloads", {})
            if all(
                workloads.get(key, {}).get("status") == "passed"
                for key in (
                    "single_point",
                    "shared_calculator",
                    "energy_force_consistency",
                )
            ) and record.get("evidence_id"):
                state, evidence_id = "validated", record["evidence_id"]
                break
    return CalculatorCapabilities(
        "energy" in properties,
        "forces" in properties,
        "stress" in properties,
        state,
        evidence_id,
    )


def require_neb_capability(capability: CalculatorCapabilities, *, experimental: bool):
    if not capability.energy or not capability.forces:
        raise ValueError("NEB requires energy and forces")
    if capability.energy_gradient_consistency == "not_guaranteed":
        raise ValueError("NEB rejects a model with non-guaranteed energy gradients")
    if capability.energy_gradient_consistency == "unknown" and not experimental:
        raise ValueError(
            "NEB energy-gradient consistency is unvalidated for this exact model/runtime. "
            "Explicitly opt in with --allow-unvalidated-neb for an experimental run."
        )
