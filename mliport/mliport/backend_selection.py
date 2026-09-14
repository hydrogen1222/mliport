"""Backend selection contract (task book ARCH-01).

mliport is a backend-neutral workflow layer: ``uma``, ``mace``, ``dpa`` and
``grace`` are equal runtimes and there is no default scientific model.  A
missing backend is a configuration error, never an implicit UMA run.

``fairchem`` remains a supported alias for ``uma`` (the upstream package
name); it is the only alias.
"""

from __future__ import annotations

#: Canonical backend names, in the order shown to users.
SUPPORTED_BACKENDS: tuple[str, ...] = ("mace", "dpa", "grace", "uma")

#: ``fairchem`` is an accepted synonym for the UMA runtime.
UMA_ALIASES: frozenset[str] = frozenset({"uma", "fairchem"})

#: Everything accepted by ``MODEL_TYPE`` / ``--model-type`` / API calls.
SUPPORTED_TYPES: frozenset[str] = frozenset(set(SUPPORTED_BACKENDS) | UMA_ALIASES)

NO_BACKEND_SELECTED_MESSAGE = """No MLIP backend was selected.

Choose one of:
  --model-type mace
  --model-type dpa
  --model-type grace
  --model-type uma

or use a configured model alias/profile."""


def normalize_model_type(model_type: str | None) -> str:
    """Lower-case and validate a backend name (``fairchem`` stays an alias)."""
    value = "" if model_type is None else str(model_type).strip().lower()
    if value not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported model type {value!r}. Choose one of: "
            + ", ".join(sorted(SUPPORTED_TYPES))
        )
    return value


def require_model_type(model_type: str | None) -> str:
    """Return a normalized backend or raise the fail-closed selection error."""
    value = "" if model_type is None else str(model_type).strip().lower()
    if value in {"", "none", "null"}:
        raise ValueError(NO_BACKEND_SELECTED_MESSAGE)
    return normalize_model_type(value)


def canonical_model_type(model_type: str | None) -> str:
    """Normalize a backend and fold the ``fairchem`` alias into ``uma``."""
    value = require_model_type(model_type)
    return "uma" if value in UMA_ALIASES else value


def is_uma(model_type: str) -> bool:
    """True for the UMA runtime (including its ``fairchem`` alias)."""
    return str(model_type).strip().lower() in UMA_ALIASES


def default_task_for(model_type: str) -> str:
    """Task default that follows the backend, never a backend default.

    UMA task families (``omat``/``omol``/...) are UMA-specific; every other
    backend uses the explicit PBC semantics ``bulk``/``molecule``.
    """
    return "omat" if is_uma(model_type) else "bulk"


__all__ = [
    "NO_BACKEND_SELECTED_MESSAGE",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_TYPES",
    "UMA_ALIASES",
    "canonical_model_type",
    "default_task_for",
    "is_uma",
    "normalize_model_type",
    "require_model_type",
]
