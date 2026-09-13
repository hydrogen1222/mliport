# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mliport project: MACE engine wrapper.

"""
MACE engine wrapper.

Wraps ``mace.calculators.MACECalculator`` behind the ``BaseMLIPCalculator``
contract. The ``mace-torch`` package is imported lazily so mliport works without
it installed; users only need it when selecting ``--model-type mace``.

Phase 1 (plan section 11.1 / 12 / 17.3) propagates the full set of
backend-specific options that actually reach the underlying ASE calculator:

* ``default_dtype`` (float32 / float64) -- the historical single most-requested
  option that previously did not flow through from CLI/INCAR/API;
* ``head`` -- the model head for MACE foundation multi-head models.

The remaining acceleration options (``compile_mode``, ``fullgraph``,
``enable_cueq`` ...) are intentionally deferred to Phase 2 per the plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mliport.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from typing import Any


# Dtype values accepted by MACE's MACECalculator.
_VALID_DTYPES = {"float32", "float64"}


class MACECalculatorWrapper(BaseMLIPCalculator):
    """Wrapper for MACE ASE calculators."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        default_dtype: str = "float64",
        task: str = "bulk",
        head: str | None = None,
    ):
        """
        Initialize MACE calculator wrapper.

        Args:
            model_path: Path to a MACE model file (``.model`` / ``.pt``).
            device: Device for calculation (``cpu``, ``cuda`` or ``cuda:N``).
            default_dtype: Model dtype (``float32`` or ``float64``). Defaults to
            ``float64`` for accuracy-first calculations. Users may explicitly
            request ``float32`` for performance.
            task: PBC hint (``bulk`` or ``molecule``); not consumed by MACE.
            head: Optional MACE foundation-model head name. Forwarded to
                ``MACECalculator`` as ``head=`` (a single string) when not
                ``None``; MACE ignores ``None`` and picks its default head.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            ValueError: If ``default_dtype`` is not a recognised dtype.
        """
        if str(default_dtype).lower() not in _VALID_DTYPES:
            raise ValueError(
                f"Invalid default_dtype {default_dtype!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_DTYPES))}"
            )

        self.model_path = Path(model_path)
        dev = str(device).lower()
        if dev not in {"cpu", "cuda", "gpu"} and not (
            dev.startswith("cuda:") and dev[5:].isdigit()
        ):
            raise ValueError(
                f"Invalid MACE device {device!r}. Use cpu, cuda, gpu, or cuda:N."
            )
        # torch/mace use "cuda"; accept mliport's legacy "gpu" synonym.
        self._device = "cuda" if dev == "gpu" else dev
        self._default_dtype = str(default_dtype).lower()
        self._task = str(task).strip().lower()
        self._head = str(head).strip() if head is not None else None
        self._calculator: Calculator | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if self._task not in {"bulk", "molecule"}:
            raise ValueError("MACE task must be 'bulk' or 'molecule'.")
        if head is not None and not self._head:
            raise ValueError("MACE head must be a non-empty head name.")

    def get_calculator(self) -> Calculator:
        """Return the cached MACE ASE calculator (lazy import)."""
        if self._calculator is None:
            from mliport.doctor import (  # noqa: PLC0415
                _installed_dependency_conflicts,
            )

            conflicts = _installed_dependency_conflicts(("mace-torch",))
            if conflicts:
                details = "\n".join(f"  - {item}" for item in conflicts)
                raise RuntimeError(
                    "MACE environment is incompatible:\n"
                    f"{details}\n"
                    "MACE and UMA must not share the same virtual environment.\n"
                    "Create .venv-mace and launch the TUI with:\n"
                    "  .venv-mace/bin/mliport tui\n"
                    "See the repository root README.md or README_CN.md."
                )
            try:
                from mace.calculators import MACECalculator  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "MACE support requires the 'mace-torch' package.\n"
                    "Install it in the dedicated .venv-mace environment; "
                    "see the repository root README.md or README_CN.md."
                ) from e

            kwargs: dict[str, Any] = {
                "model_paths": str(self.model_path),
                "device": self._device,
                "default_dtype": self._default_dtype,
            }
            # MACE's MACECalculator selects the active head via the ``head``
            # keyword (singular, a string) -- NOT ``heads`` (plural). The
            # plural ``heads`` attribute lives on the trained *model* object;
            # passing ``heads=[...]`` to the calculator is silently swallowed
            # into ASE's parameter dict and never selects a head, so a user's
            # ``--head`` choice would be ignored (or fall back / error).
            if self._head is not None:
                kwargs["head"] = self._head
            calculator = MACECalculator(**kwargs)
            raw_heads = getattr(calculator, "available_heads", None)
            if isinstance(raw_heads, str):
                available_heads = [raw_heads]
            elif isinstance(raw_heads, (list, tuple, set)):
                available_heads = list(raw_heads)
            else:
                # Older/custom calculators may not expose concrete head
                # metadata. Do not treat an arbitrary proxy object as an
                # iterable; the backend still receives the requested head.
                available_heads = []
            if self._head is None and len(available_heads) > 1:
                self._calculator = None
                raise ValueError(
                    "This MACE model exposes multiple energy heads, so "
                    "--head/HEAD is required. Available heads: "
                    f"{available_heads}"
                )
            if (
                self._head is not None
                and available_heads
                and self._head not in available_heads
            ):
                # mace-torch currently warns and silently falls back to the
                # last head.  Silent head substitution changes the potential
                # energy surface, so mliport fails closed instead.
                raise ValueError(
                    f"MACE head {self._head!r} is not available in this model. "
                    f"Available heads: {available_heads}"
                )
            self._calculator = calculator
        return self._calculator

    @property
    def task(self) -> str:
        """PBC hint (``bulk``/``molecule``); MACE itself is task-agnostic."""
        return self._task

    @property
    def has_stress(self) -> bool:
        """Whether stress is supported."""
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Return model metadata.

        Enriched per plan section 17.3 with the elements/cutoff/head/dtype/
        device/model-count fields when the underlying calculator exposes them,
        so OUTCAR/JSON output records what actually ran.
        """
        base = {
            "model_type": "mace",
            "model_path": str(self.model_path),
            "device": self._device,
            "default_dtype": self._default_dtype,
            "head": self._head,
            "task": self._task,
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
        # Best-effort enrichment from the real MACE model. These attributes
        # are not part of the ASE Calculator contract, so guard every access.
        # NOTE: MACE exposes ``r_max``/``atomic_numbers`` on each *model*, and
        # ``available_heads``/``head``/``z_table``/``num_models`` on the
        # calculator -- there is no ``elements``/``heads``/``num_elements``
        # attribute, so the previous lookups silently never resolved.
        calc = self._calculator
        if calc is not None:
            model = getattr(calc, "models", None)
            if isinstance(model, list) and model:
                first = model[0]
                r_max = getattr(first, "r_max", None)
                if r_max is not None:
                    base["cutoff"] = float(r_max)
                atomic_numbers = getattr(first, "atomic_numbers", None)
                if atomic_numbers is not None:
                    base["num_elements"] = len(atomic_numbers)
            # Cutoff may also be readable from the calculator.
            if "cutoff" not in base:
                r_max = getattr(calc, "r_max", None)
                if r_max is not None:
                    base["cutoff"] = float(r_max)
            available_heads = getattr(calc, "available_heads", None)
            if available_heads is not None:
                base["available_heads"] = list(available_heads)
            active_head = getattr(calc, "head", None)
            if active_head is not None:
                # Record the head MACE actually selected (may differ from the
                # requested ``self._head`` when it was None / not found).
                base["active_head"] = active_head
            z_table = getattr(calc, "z_table", None)
            if z_table is not None:
                try:
                    from ase.data import chemical_symbols  # noqa: PLC0415

                    zs = list(getattr(z_table, "zs", z_table))
                    base["elements"] = [chemical_symbols[int(z)] for z in zs]
                except Exception:
                    pass
            base["num_models"] = len(model) if isinstance(model, list) else 1
            from mliport.devices import torch_model_identity

            base.update(
                torch_model_identity(
                    model[0] if isinstance(model, list) and model else None,
                    self._device,
                )
            )
        return base
