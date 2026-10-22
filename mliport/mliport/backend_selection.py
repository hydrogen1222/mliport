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

#: DFT-D3 dispersion options. The correction is added on top of the model by
#: the calculator factory, so every backend consumes them.
DISPERSION_OPTION_KEYS: frozenset[str] = frozenset({"d3", "d3_xc", "d3_damping"})

#: Calculator (backend) options each runtime consumes. This is the single
#: table shared by the resolver, the calculator factory, the engine and the
#: queue command builder; adding a backend option means editing only here.
CALCULATOR_OPTION_KEYS: dict[str, frozenset[str]] = {
    "uma": frozenset(
        {"inference_mode", "torch_num_threads", "activation_checkpointing"}
    )
    | DISPERSION_OPTION_KEYS,
    "mace": frozenset({"default_dtype", "head"}) | DISPERSION_OPTION_KEYS,
    "dpa": frozenset({"head"}) | DISPERSION_OPTION_KEYS,
    "grace": frozenset(
        {
            "cpu_threads",
            "gpu_memory_growth",
            "gpu_memory_limit_mb",
            "neighbor_cache",
            "neighbor_skin",
        }
    )
    | DISPERSION_OPTION_KEYS,
}

#: Every backend-specific calculator option.
ALL_CALCULATOR_OPTION_KEYS: frozenset[str] = frozenset().union(
    *CALCULATOR_OPTION_KEYS.values()
)

#: Options that apply to every backend even though one backend also receives
#: them as a calculator option (the CPU thread count).
SHARED_OPTION_KEYS: frozenset[str] = frozenset({"torch_num_threads"})

#: Backends whose CPU intra-op threads are PyTorch threads.
TORCH_BACKENDS: frozenset[str] = frozenset({"uma", "mace", "dpa"})

#: Backends whose ASE adapters cannot choose a GPU inside an existing process;
#: they need CUDA_VISIBLE_DEVICES isolation before the framework is imported.
PROCESS_ISOLATED_BACKENDS: frozenset[str] = frozenset({"dpa", "grace"})

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


def calculator_option_keys(model_type: str) -> frozenset[str]:
    """Calculator options consumed by ``model_type`` (``fairchem`` = ``uma``)."""
    return CALCULATOR_OPTION_KEYS.get(canonical_model_type(model_type), frozenset())


def option_applies_to_backend(key: str, model_type: str) -> bool:
    """False when ``key`` is a calculator option of other backends only."""
    if key in SHARED_OPTION_KEYS or key not in ALL_CALCULATOR_OPTION_KEYS:
        return True
    return key in calculator_option_keys(model_type)


def default_task_for(model_type: str) -> str:
    """Task default that follows the backend, never a backend default.

    UMA task families (``omat``/``omol``/...) are UMA-specific; every other
    backend uses the explicit PBC semantics ``bulk``/``molecule``.
    """
    return "omat" if is_uma(model_type) else "bulk"


__all__ = [
    "ALL_CALCULATOR_OPTION_KEYS",
    "CALCULATOR_OPTION_KEYS",
    "DISPERSION_OPTION_KEYS",
    "NO_BACKEND_SELECTED_MESSAGE",
    "PROCESS_ISOLATED_BACKENDS",
    "SHARED_OPTION_KEYS",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_TYPES",
    "TORCH_BACKENDS",
    "UMA_ALIASES",
    "calculator_option_keys",
    "canonical_model_type",
    "default_task_for",
    "is_uma",
    "normalize_model_type",
    "option_applies_to_backend",
    "require_model_type",
]
