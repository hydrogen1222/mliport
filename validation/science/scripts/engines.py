"""Engine adapters: build real backend calculators through mlipx itself.

All four engines are loaded through the same ``CalculatorFactory`` layer the
public CLI/API uses, so the validation exercises the product's own loading
path.  There is deliberately no fallback: if the requested engine cannot be
loaded under its profile, the case fails and nothing else may substitute for
it (taskbook section 47).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402

_ENGINES = ("uma", "mace", "dpa", "grace")


@dataclass
class EngineContext:
    engine: str
    profile_id: str
    profile: dict[str, Any]
    wrapper: Any
    calculator: Any
    device: str
    dtype: str
    task: str | None
    head: str | None
    backend_version: str
    framework_version: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def model_sha256(self) -> str:
        return self.profile["model_sha256"]

    @property
    def model_identity(self) -> str:
        return self.profile.get("identity", self.profile["upstream_model_id"])


def _framework_version(engine: str) -> str:
    if engine == "grace":
        return common.package_version("tensorflow")
    return common.package_version("torch")


def _backend_dist(engine: str) -> str:
    return {
        "uma": "fairchem-core",
        "mace": "mace-torch",
        "dpa": "deepmd-kit",
        "grace": "tensorpotential",
    }[engine]


def build_engine(
    engine: str,
    profile_id: str,
    profile: dict[str, Any],
    model_path: str | Path,
    device: str = "cuda:0",
    dtype: str | None = None,
    head: str | None = None,
    neighbor_cache: bool = True,
) -> EngineContext:
    """Load one backend calculator through mlipx's CalculatorFactory.

    ``dtype`` is only interpreted by MACE (``default_dtype``).  ``head`` is
    the DPA branch / MACE multi-head selector.  ``neighbor_cache`` is the
    GRACE-specific toggle.
    """
    if engine not in _ENGINES:
        msg = f"unknown engine {engine!r}; expected one of {_ENGINES}"
        raise ValueError(msg)

    task = (profile.get("task") or "omat") if engine == "uma" else "bulk"
    kwargs: dict[str, Any] = {}
    if engine == "mace":
        kwargs["default_dtype"] = dtype or "float64"
    if head is not None:
        kwargs["head"] = head
    if engine == "grace":
        kwargs["neighbor_cache"] = neighbor_cache

    from mlipx.calculators.factory import CalculatorFactory  # noqa: PLC0415

    wrapper = CalculatorFactory.create(
        model_type=engine,
        model_path=str(Path(model_path).resolve()),
        device=device,
        task=task,
        **kwargs,
    )
    calculator = wrapper.get_calculator()

    dtype_recorded = dtype or "upstream/model-defined"
    return EngineContext(
        engine=engine,
        profile_id=profile_id,
        profile=profile,
        wrapper=wrapper,
        calculator=calculator,
        device=device,
        dtype=dtype_recorded,
        task=task,
        head=head,
        backend_version=common.package_version(_backend_dist(engine)),
        framework_version=_framework_version(engine),
        diagnostics=common.calculator_identity(wrapper, calculator),
    )
