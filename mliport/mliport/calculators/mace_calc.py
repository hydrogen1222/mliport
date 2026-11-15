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
    from typing import Any

    from ase.calculators.calculator import Calculator


# Dtype values accepted by MACE's MACECalculator.
_VALID_DTYPES = {"float32", "float64"}

#: mace-torch 0.3.16 model classes that have no potential-energy head.
#: ``AtomicDipolesMACE`` and ``AtomicDielectricMACE`` (mace/modules/models.py)
#: predict dipoles and polarizabilities; the MACE-MDP family is built from them.
#: A single-point energy, force or stress from one of these would be fabricated.
_DIPOLE_ONLY_MODEL_CLASSES = frozenset(
    {"AtomicDipolesMACE", "AtomicDielectricMACE", "DipolePolarizabilityMACE"}
)

#: ``PolarMACE`` (mace/modules/extensions.py) is the polarizable-electrostatics
#: model MACE-POLAR-1 ships; upstream loads it through ``mace_polar`` with
#: ``MACECalculator(model_type="PolarMACE")`` and extra charge/density inputs.
_POLAR_MODEL_CLASSES = frozenset({"PolarMACE"})


def _is_neural_network_module(model: Any) -> bool:
    """True for a real ``torch.nn.Module`` (never for a mock or a mapping).

    The attribute-shaped discriminators below (``atomic_multipoles_max_l``,
    ``joint_embedding``) only mean something on a constructed MACE model.
    Testing attributes on an arbitrary object — a ``MagicMock`` in a test, a
    dict-like proxy in a plugin — reports every attribute as present and would
    reject a legitimate energy model.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - mace-torch always ships torch
        return False
    return isinstance(model, torch.nn.Module)


def unsupported_mace_model_reason(model: Any) -> str | None:
    """Why ``model`` cannot serve as an mliport potential-energy model.

    The decision is made from the *loaded model object* -- its class name and
    the attributes mace-torch 0.3.16 sets on it -- never from the file name.
    Returns ``None`` for an ordinary energy model (``MACE`` /
    ``ScaleShiftMACE``, including the multihead MACE-MH-1).

    Discriminators, from the pinned mace-torch 0.3.16 source:

    * dipole/polarizability only: ``AtomicDipolesMACE`` / ``AtomicDielectricMACE``
      (``mace/modules/models.py``); they forward ``dipole``/
      ``polarizability``, not ``energy``.
    * polarizable electrostatics: ``PolarMACE``
      (``mace/modules/extensions.py``), which also carries
      ``atomic_multipoles_max_l`` and ``charges_irreps``.
    * charge/spin embeddings: any model built with ``embedding_specs`` gets a
      ``joint_embedding`` block (``mace/modules/models.py``); the MACE-OMOL
      family reads the total charge and spin from ``atoms.info``.
    """
    name = type(model).__name__
    if name in _DIPOLE_ONLY_MODEL_CLASSES:
        return (
            f"{name} is a dipole/polarizability model without a potential-energy "
            "head (the MACE-MDP family). An energy, force or stress from it would "
            "be fabricated. Upstream loads it with MACECalculator("
            "model_type='DipoleMACE' or 'DipolePolarizabilityMACE'); mliport only "
            "runs potential-energy models."
        )
    polar_attributes = _is_neural_network_module(model) and (
        getattr(model, "atomic_multipoles_max_l", None) is not None
        and getattr(model, "charges_irreps", None) is not None
    )
    if name in _POLAR_MODEL_CLASSES or polar_attributes:
        return (
            f"{name} is a polarizable-electrostatics model (MACE-POLAR). It needs "
            "the upstream mace_polar loader with MACECalculator("
            "model_type='PolarMACE') and external charge/density inputs that "
            "mliport does not supply, so the stress and energy would be wrong."
        )
    embedding_specs = getattr(model, "embedding_specs", None)
    has_joint_embedding = _is_neural_network_module(model) and hasattr(
        model, "joint_embedding"
    )
    is_embedding_model = isinstance(embedding_specs, dict) and bool(embedding_specs)
    if is_embedding_model or has_joint_embedding:
        keys = sorted(str(key) for key in embedding_specs) if is_embedding_model else []
        return (
            f"{name} carries charge/spin joint embeddings (embedding_specs="
            f"{keys}). The MACE-OMOL family reads the total charge and spin from "
            "atoms.info; mliport does not propagate them, so the potential would "
            "silently describe a different charge state. Load it with the "
            "upstream MACE-OMOL tooling instead of a plain MACE energy model."
        )
    return None


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
            # Fail closed on checkpoints that cannot serve as a potential-energy
            # model: a dipole/polarizability-only or polarizable-electrostatics
            # model, or one whose charge/spin embeddings mliport cannot feed.
            # The gate is only meaningful if the loaded calculator exposes the
            # models it built: a missing or non-list ``models`` attribute means
            # mliport cannot inspect the checkpoint, and skipping the check would
            # silently accept whatever class mace-torch constructed.
            loaded_models = getattr(calculator, "models", None)
            if not isinstance(loaded_models, list) or not loaded_models:
                self._calculator = None
                raise ValueError(
                    "Unsupported MACE calculator: the loaded calculator does not "
                    "expose a non-empty 'models' list (got "
                    f"{type(loaded_models).__name__}), so mliport cannot verify "
                    "that the checkpoint is a potential-energy model. Refusing to "
                    "run an uninspected model class."
                )
            for model in loaded_models:
                reason = unsupported_mace_model_reason(model)
                if reason is not None:
                    self._calculator = None
                    raise ValueError(f"Unsupported MACE model: {reason}")
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
            if isinstance(available_heads, str):
                # A single head name must not be split into characters.
                base["available_heads"] = [available_heads]
            elif available_heads is not None:
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
