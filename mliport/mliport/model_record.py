"""Shared model-identity record for NEB and confsearch provenance.

The record was extracted from :mod:`mliport.neb.workflow` without changing its
behaviour.  It answers the question "which exact model produced this number?"
with the resolved configuration cross-checked against what the backend actually
reports, so a run can never silently use a different model, task, head, dtype,
inference mode or device than the one recorded.

Callers pass their own ``error_class`` so the failure is reported in their own
vocabulary (NEB raises :class:`~mliport.neb.schema.NEBPreparationError`,
confsearch raises its own preparation error); every message text is unchanged.

The ``dispersion`` field is part of the identity: a search with a DFT-D3
correction must never be merged with, or compared to, one without it
(confsearch spec section 12).
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from mliport.base_calculator import BaseMLIPCalculator
    from mliport.config.resolver import ResolvedConfig

#: Backend package whose version identifies the runtime for each model type.
BACKEND_DISTRIBUTIONS = {
    "uma": "fairchem-core",
    "fairchem": "fairchem-core",
    "mace": "mace-torch",
    "dpa": "deepmd-kit",
    "grace": "tensorpotential",
}


def framework_distributions(model_type: str, model_path: str | Path) -> tuple[str, ...]:
    """Framework distributions that can supply the model's numerics."""
    if model_type == "dpa":
        return (
            ("tensorflow",) if Path(model_path).suffix.lower() == ".pb" else ("torch",)
        )
    if model_type == "grace":
        return ("tensorflow",)
    return ("torch",)


def distribution_version(name: str) -> str | None:
    """Installed version of ``name``, or None when it is not installed."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def hash_model_path(path: str | Path, *, error_class: type[BaseException]) -> str:
    """Content hash of a model file or of every file in a model directory."""
    model = Path(path).expanduser().resolve()
    if not model.exists():
        raise FileNotFoundError(f"Model path not found: {model}")
    digest = hashlib.sha256()
    if model.is_file():
        with model.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not model.is_dir():
        raise error_class(f"Unsupported model path type: {model}")
    files = sorted(path for path in model.rglob("*") if path.is_file())
    if not files:
        raise error_class(f"Model directory contains no files: {model}")
    for child in files:
        relative = child.relative_to(model).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def build_model_record(
    calculator: BaseMLIPCalculator,
    resolved: ResolvedConfig,
    *,
    error_class: type[BaseException] = ValueError,
) -> dict[str, Any]:
    """Build the provenance/identity record for the loaded model.

    Raises ``error_class`` when the backend reports an identity that conflicts
    with the resolved configuration, or when an exact version cannot be
    determined.
    """
    info = dict(calculator.info())
    model_type = resolved.model_type.lower()
    reported_model_type = info.get("model_type")
    if (
        reported_model_type is not None
        and str(reported_model_type).lower() != model_type
    ):
        raise error_class(
            f"Calculator reports model_type={reported_model_type!r}, but resolved "
            f"configuration requires {model_type!r}"
        )
    reported_task = info.get("task")
    if (
        reported_task is not None
        and str(reported_task).lower() != resolved.task.lower()
    ):
        raise error_class(
            f"Calculator reports task={reported_task!r}, but resolved configuration "
            f"requires {resolved.task!r}"
        )
    reported_path = info.get("model_path")
    if (
        reported_path is not None
        and Path(reported_path).expanduser().resolve()
        != Path(resolved.model_path).expanduser().resolve()
    ):
        raise error_class("Calculator model path differs from the resolved model path")
    backend_distribution = BACKEND_DISTRIBUTIONS[model_type]
    backend_version = info.get("backend_version") or distribution_version(
        backend_distribution
    )
    if not backend_version:
        raise error_class(
            f"Cannot determine exact backend version for {backend_distribution!r}"
        )
    framework_versions = dict(info.get("framework_versions") or {})
    if not framework_versions:
        for distribution in framework_distributions(model_type, resolved.model_path):
            installed = distribution_version(distribution)
            if installed is not None:
                framework_versions[distribution] = installed
    if not framework_versions:
        raise error_class(
            f"Cannot determine an exact framework version for {model_type!r}"
        )
    calculator_options = dict(resolved.calculator_options)
    dtype_requested = calculator_options.get("default_dtype")
    dtype_effective = info.get("default_dtype")
    if dtype_effective is None and info.get("model_precision"):
        dtype_effective = info["model_precision"]
    head_requested = calculator_options.get("head")
    head_effective = info.get("active_head", info.get("head", head_requested))
    source = resolved.sources.get("model_path")
    model_alias = None
    if source is not None and source.source.startswith("model alias "):
        model_alias = source.source.removeprefix("model alias ").strip("'")
    return {
        "model_type": model_type,
        "actual_device_type": info.get("actual_device_type", "unknown"),
        "actual_device_logical_index": info.get("actual_device_logical_index"),
        "actual_device_uuid": info.get("actual_device_uuid"),
        "inference_mode": info.get(
            "inference_mode",
            calculator.inference_mode
            if hasattr(calculator, "inference_mode")
            else "default",
        ),
        "compile_enabled": info.get("compile_enabled"),
        "direct_forces": info.get("direct_forces"),
        "model_path": str(Path(resolved.model_path).expanduser().resolve()),
        "model_sha256": hash_model_path(resolved.model_path, error_class=error_class),
        "model_alias": model_alias,
        "model_source": source.source if source is not None else None,
        "backend_distribution": backend_distribution,
        "backend_version": str(backend_version),
        "framework_versions": framework_versions,
        "task": resolved.task,
        "head_requested": head_requested,
        "head_effective": head_effective,
        "dtype_requested": dtype_requested,
        "dtype_effective": dtype_effective,
        # DFT-D3 (or any other dispersion correction) is part of the model
        # identity; None means "no correction applied".
        "dispersion": info.get("dispersion"),
        "backend_reported_device": info.get(
            "actual_device", info.get("device", resolved.device)
        ),
    }


__all__ = [
    "BACKEND_DISTRIBUTIONS",
    "build_model_record",
    "distribution_version",
    "framework_distributions",
    "hash_model_path",
]
