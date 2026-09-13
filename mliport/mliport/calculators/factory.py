# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mliport project: multi-engine CalculatorFactory.

"""
Calculator factory.

Selects the right MLIP engine wrapper from a ``model_type`` string. Each
wrapper is imported lazily so only the chosen engine's backend is loaded.
Adding a new engine only requires one new wrapper module + one ``elif``
branch here; Runners/Writers/CLI/TUI stay untouched.

Phase 1 changes (plan section 11.1 / 17.2):

* MACE receives its full backend-specific options (``default_dtype`` with a
  factory-level float64 fallback, and ``head``) so the dtype chosen on the
  CLI/INCAR/API actually reaches ``MACECalculator``.
* ``cuda:0`` style device strings are accepted and forwarded unchanged.
* Unknown / misspelled engine-specific options are rejected (or warned about)
  instead of being silently dropped -- a typo such as ``default_dtpe`` now
  reports ``Did you mean 'default_dtype'?`` rather than silently using float64.
"""

from __future__ import annotations

import difflib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from mliport.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from typing import Any


# Engine type aliases. ``fairchem`` is accepted as a synonym for ``uma``.
UMA_ALIASES = {"uma", "fairchem"}
SUPPORTED_TYPES = {"uma", "fairchem", "mace", "dpa", "grace"}

# Calculator-option keys each engine actually consumes. Keys that belong to a
# *different* engine are dropped with a warning (benign cross-engine leftover);
# keys that match no engine are treated as potential typos.
_CALC_KEYS: dict[str, set[str]] = {
    "uma": {"inference_mode", "torch_num_threads", "activation_checkpointing"},
    "fairchem": {"inference_mode", "torch_num_threads", "activation_checkpointing"},
    "mace": {"default_dtype", "head"},
    "dpa": {"head"},
    "grace": {
        "cpu_threads",
        "gpu_memory_growth",
        "gpu_memory_limit_mb",
        "neighbor_cache",
        "neighbor_skin",
    },
}
_ALL_CALC_KEYS: set[str] = set().union(*_CALC_KEYS.values())


def _check_unknown_kwargs(
    model_type: str, kwargs: dict[str, Any], *, strict: bool
) -> dict[str, Any]:
    """Filter ``kwargs`` to the keys ``model_type`` consumes.

    Args:
        model_type: Lowercased engine name.
        kwargs: Raw engine-specific options.
        strict: When True, unknown keys raise ``ValueError`` (with a typo
            suggestion); otherwise they emit a ``UserWarning``.

    Returns:
        The subset of ``kwargs`` that ``model_type`` actually consumes.
    """
    accepted = _CALC_KEYS.get(model_type, set())
    kept: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in accepted:
            kept[key] = value
            continue
        # A key valid for another engine -> benign leftover, warn and drop.
        if key in _ALL_CALC_KEYS:
            warnings.warn(
                f"Option {key!r} is not applicable to engine {model_type!r}; "
                f"ignoring it.",
                stacklevel=3,
            )
            continue
        # Truly unknown key -> likely a typo.
        suggestion = difflib.get_close_matches(
            key, sorted(_ALL_CALC_KEYS | {"device", "task"}), n=1, cutoff=0.6
        )
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        message = f"Unknown calculator option {key!r} for engine {model_type!r}.{hint}"
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)
    return kept


class CalculatorFactory:
    """Create the appropriate MLIP Calculator wrapper from a model type."""

    @staticmethod
    def create(
        model_type: str,
        model_path: str | Path,
        device: str = "cpu",
        task: str = "bulk",
        *,
        strict: bool = False,
        **kwargs: Any,
    ) -> BaseMLIPCalculator:
        """
        Create a calculator wrapper for the requested engine.

        Args:
            model_type: Engine name (``uma``/``fairchem``, ``mace``,
                ``dpa``, ``grace``).
            model_path: Path to the model checkpoint/file.
            device: Device for calculation (``cpu``, ``cuda``, ``gpu`` or
                ``cuda:N``). Forwarded unchanged to the wrapper.
            task: Task type. UMA uses ``omat``/``omol``/...; other engines use
                ``bulk``/``molecule`` (PBC hint only).
            strict: When True, unknown engine-specific options raise instead of
                warning (plan section 10 / 17.2).
            **kwargs: Engine-specific options. UMA accepts ``inference_mode``,
                ``torch_num_threads`` and ``activation_checkpointing``; MACE
                accepts ``default_dtype`` (factory fallback ``float64``) and
                ``head``; DPA accepts ``head`` for multi-task branches; GRACE
                accepts ``cpu_threads`` for TensorFlow intra-op parallelism and
                GPU memory-growth / hard-limit controls.

        Returns:
            A ``BaseMLIPCalculator`` subclass instance.

        Raises:
            ValueError: If ``model_type`` is not supported, or (in strict mode)
                an unknown engine-specific option is supplied.
        """
        m_type = (model_type or "uma").lower()
        kwargs = _check_unknown_kwargs(m_type, kwargs, strict=strict)
        normalized_task = str(task).strip().lower()
        if m_type not in UMA_ALIASES and normalized_task not in {
            "bulk",
            "molecule",
        }:
            raise ValueError(
                f"Invalid task {task!r} for engine {m_type!r}. Non-UMA "
                "engines require the explicit PBC semantic 'bulk' or "
                "'molecule'."
            )

        if m_type in UMA_ALIASES:
            from mliport.calculator import UMACalculator  # noqa: PLC0415

            return UMACalculator(
                model_path=model_path,
                task=normalized_task,
                device=device,
                inference_mode=kwargs.get("inference_mode", "default"),
                torch_num_threads=kwargs.get("torch_num_threads"),
                activation_checkpointing=kwargs.get("activation_checkpointing"),
            )
        elif m_type == "mace":
            from mliport.calculators.mace_calc import MACECalculatorWrapper  # noqa: PLC0415

            # Accuracy-first fallback for direct construction. Users must opt
            # in explicitly to float32 when throughput matters more.
            return MACECalculatorWrapper(
                model_path=model_path,
                device=device,
                task=normalized_task,
                default_dtype=kwargs.get("default_dtype", "float64"),
                head=kwargs.get("head"),
            )
        elif m_type == "dpa":
            from mliport.calculators.dpa_calc import DPACalculatorWrapper  # noqa: PLC0415

            return DPACalculatorWrapper(
                model_path=model_path,
                device=device,
                task=normalized_task,
                head=kwargs.get("head"),
            )
        elif m_type == "grace":
            from mliport.calculators.grace_calc import GRACECalculatorWrapper  # noqa: PLC0415

            return GRACECalculatorWrapper(
                model_path=model_path,
                device=device,
                task=normalized_task,
                cpu_threads=kwargs.get("cpu_threads"),
                gpu_memory_growth=kwargs.get("gpu_memory_growth", True),
                gpu_memory_limit_mb=kwargs.get("gpu_memory_limit_mb"),
                neighbor_cache=kwargs.get("neighbor_cache", True),
                neighbor_skin=kwargs.get("neighbor_skin", 1.5),
            )
        else:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_TYPES - {'fairchem'}))}"
            )
