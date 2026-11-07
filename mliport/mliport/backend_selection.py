"""Backend selection contract (task book ARCH-01).

mliport is a backend-neutral workflow layer: ``uma``, ``mace``, ``dpa``,
``grace`` and ``nep`` are equal runtimes and there is no default scientific
model.  A missing backend is a configuration error, never an implicit UMA run.

``fairchem`` remains a supported alias for ``uma`` (the upstream package
name); it is the only alias.

NEP is the only backend with **two execution runtimes** (PR #61 §C).  The
runtime is an explicit user decision (``NEP_RUNTIME``), never inferred from the
device, the task or the availability of Calorine: silently switching between
the ASE calculator path and the native GPUMD process path would change units of
truth (one ASE evaluation per step versus one native integration) behind the
user's back.
"""

from __future__ import annotations

#: Canonical backend names, in the order shown to users.
SUPPORTED_BACKENDS: tuple[str, ...] = ("mace", "dpa", "grace", "uma", "nep")

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
    # NEP runtime + native executor selection (PR #61 §C).  ``d3`` is
    # deliberately NOT accepted: the only way to prove a NEP model plus a D3
    # correction does not double-count dispersion is a dedicated study, so the
    # factory rejects it instead of silently applying a correction.
    "nep": frozenset({"nep_runtime", "nep_gpumd_command"}),
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
  --model-type nep   (also needs --nep-runtime ase|gpumd)

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


# ---------------------------------------------------------------------------
# NEP dual-runtime contract (PR #61 §C / §B)
# ---------------------------------------------------------------------------

#: The two NEP execution runtimes.  There is deliberately no ``auto``.
NEP_RUNTIMES: tuple[str, ...] = ("ase", "gpumd")

#: Calculation types each NEP runtime can actually perform.
#:
#: ``ase`` reuses the existing ASE runner stack, so every workflow that already
#: works for the other four backends works here too.  ``gpumd`` is a native
#: external process: it can do SP, fixed/box optimisation and continuous MD,
#: but a NEB band, a batch sweep or the confsearch cycle engine are driver-side
#: algorithms that would need their own native scheduler.  Claiming them by
#: silently routing to the ASE runtime would mix two potentials' energy
#: identities inside one ranking, so they fail closed instead.
NEP_CALC_TYPES_BY_RUNTIME: dict[str, frozenset[str]] = {
    "ase": frozenset({"sp", "opt", "md", "neb", "batch", "confsearch"}),
    "gpumd": frozenset({"sp", "opt", "md"}),
}

#: Human-readable description of each runtime, used by CLI/TUI/doctor/docs.
NEP_RUNTIME_DESCRIPTIONS: dict[str, str] = {
    "ase": (
        "Calorine ASE calculator: CPUNEP on cpu (float64, in-process) or "
        "GPUNEP on cuda (float32, one GPUMD subprocess per ASE evaluation)"
    ),
    "gpumd": (
        "native gpumd process: one GPU process per run, float32 inference, "
        "continuous NVE/NVT integration (SP, fixed-cell/box optimisation, MD)"
    ),
}

NEP_RUNTIME_REQUIRED_MESSAGE = """NEP requires an explicit execution runtime.

Choose one:
  --nep-runtime ase     (Calorine CPUNEP on cpu / GPUNEP on cuda)
  --nep-runtime gpumd   (native gpumd process on cuda)

or set NEP_RUNTIME in the INCAR/settings/model alias. mliport will not infer
the runtime from the device or pick one for you."""


def normalize_nep_runtime(value: str | None) -> str:
    """Lower-case and validate ``NEP_RUNTIME``; ``None`` is a hard error."""
    text = "" if value is None else str(value).strip().lower()
    if not text:
        raise ValueError(NEP_RUNTIME_REQUIRED_MESSAGE)
    if text not in NEP_RUNTIMES:
        raise ValueError(
            f"Invalid NEP_RUNTIME {value!r}. Choose one of: " + ", ".join(NEP_RUNTIMES)
        )
    return text


def normalize_device_for_nep(runtime: str, device: str | None) -> str:
    """Return the runtime's canonical device string, or raise.

    ``runtime="gpumd"`` has no CPU implementation here: a CPU request must be
    rejected rather than quietly downgraded to ``CPUNEP``, because the two
    runtimes report different precisions and different per-step costs.
    """
    runtime = normalize_nep_runtime(runtime)
    text = "cpu" if device is None else str(device).strip().lower()
    if text == "gpu":
        text = "cuda"
    if not (text == "cpu" or text == "cuda" or text.startswith("cuda:")):
        raise ValueError(
            f"Invalid device {device!r} for NEP. Use cpu, cuda, cuda:N or gpu."
        )
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        if not suffix.isdigit():
            raise ValueError(
                f"Invalid device {device!r} for NEP: the CUDA index must be an "
                "integer (cuda:0, cuda:1, ...)."
            )
    if runtime == "gpumd" and text == "cpu":
        raise ValueError(
            "NEP_RUNTIME=gpumd has no CPU implementation. Use "
            "NEP_RUNTIME=gpumd with DEVICE=cuda[:N], or switch explicitly to "
            "NEP_RUNTIME=ase with DEVICE=cpu (a different execution path)."
        )
    return text


def nep_calc_types(runtime: str) -> frozenset[str]:
    """Calculation types the given NEP runtime supports."""
    return NEP_CALC_TYPES_BY_RUNTIME[normalize_nep_runtime(runtime)]


def require_nep_support(runtime: str, calc_type: str) -> None:
    """Fail closed when ``calc_type`` cannot run on ``runtime``."""
    runtime = normalize_nep_runtime(runtime)
    kind = str(calc_type).strip().lower()
    supported = NEP_CALC_TYPES_BY_RUNTIME[runtime]
    if kind in supported:
        return
    raise ValueError(
        f"NEP_RUNTIME={runtime} does not support calc_type={kind!r}. "
        f"Supported for this runtime: {', '.join(sorted(supported))}. "
        "Use NEP_RUNTIME=ase explicitly (a different execution path with its "
        "own performance characteristics) if you need it; mliport never "
        "reroutes a runtime silently."
    )


def model_type_choices() -> tuple[str, ...]:
    """Every accepted ``MODEL_TYPE`` / ``--model-type`` value, in user order.

    Single source shared by the CLI, the INCAR writer, the schema and the
    queue command builder so a backend is never added to one entry point only.
    """
    return SUPPORTED_BACKENDS + ("fairchem",)


__all__ = [
    "ALL_CALCULATOR_OPTION_KEYS",
    "CALCULATOR_OPTION_KEYS",
    "DISPERSION_OPTION_KEYS",
    "NEP_CALC_TYPES_BY_RUNTIME",
    "NEP_RUNTIMES",
    "NEP_RUNTIME_DESCRIPTIONS",
    "NEP_RUNTIME_REQUIRED_MESSAGE",
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
    "model_type_choices",
    "nep_calc_types",
    "normalize_device_for_nep",
    "normalize_model_type",
    "normalize_nep_runtime",
    "option_applies_to_backend",
    "require_model_type",
    "require_nep_support",
]
