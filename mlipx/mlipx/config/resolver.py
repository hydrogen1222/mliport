"""Layered configuration resolver (plan section 4.3 / 17.6 / 17.8).

The resolver merges, from lowest to highest priority::

    built-in defaults  <  settings.ini  <  model alias  <  profile
                        <  INCAR / job   <  CLI / kwargs

and produces:

* the model-level fields (``model_type`` / ``model_path`` / ``task`` / ``device``
  / ``inference_mode``),
* ``calculator_options`` (engine-specific, e.g. MACE ``default_dtype``/``head``),
* ``run_options`` (calc-type-specific, e.g. ``fmax`` / ``temperature``),
* a per-key ``source`` trace so ``mlipx config explain TEMPERATURE`` can say
  *why* a parameter ended up with a given value.

The output is engine-agnostic: callers (CLI / API / engine) consume
``calculator_options`` and ``run_options`` directly. Path resolution follows the
plan (section 18.1): model paths are resolved relative to the loaded
settings.ini, structure paths relative to the BATCH file, output paths relative
to the current working directory.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from mlipx.config.aliases import (
    resolve_model_alias,
    resolve_profile,
)
from mlipx.config.defaults import (
    BUILTIN_DEFAULTS,
    DEFAULT_DEVICE_BY_CALC_TYPE,
)
from mlipx.config.provenance import SourceLocation
from mlipx.config.schema import Schema, get_schema

if TYPE_CHECKING:
    from typing import Any

    from mlipx.config.aliases import ModelAlias, Profile
    from mlipx.config.settings import MlipxSettings


@dataclass(frozen=True)
class ResolvedValue:
    """A single resolved parameter with provenance."""

    value: Any
    source: str
    base_dir: str | None = field(default=None, compare=False)
    location: str | None = field(default=None, compare=False)

    def __repr__(self) -> str:
        return f"ResolvedValue({self.value!r}, source={self.source!r})"


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved configuration produced by :func:`resolve_config`."""

    calc_type: str
    model_type: str
    model_path: str
    task: str
    device: str
    inference_mode: str
    calculator_options: Mapping[str, Any] = field(default_factory=dict)
    run_options: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, ResolvedValue] = field(default_factory=dict)
    unknown_options: Mapping[str, Any] = field(default_factory=dict)
    strict: bool = False
    settings_path: str | None = None

    def __post_init__(self) -> None:
        """Freeze mapping fields so execution cannot mutate resolved values."""
        for name in (
            "calculator_options",
            "run_options",
            "settings",
            "sources",
            "unknown_options",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def as_dict(self) -> dict[str, Any]:
        """Flat dict suitable for ``resolved_config.json`` output."""
        return {
            "calc_type": self.calc_type,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "task": self.task,
            "device": self.device,
            "inference_mode": self.inference_mode,
            "calculator_options": dict(self.calculator_options),
            "run_options": dict(self.run_options),
            "settings": dict(self.settings),
            "sources": {
                k: {
                    "value": v.value,
                    "source": v.source,
                    "base_dir": v.base_dir,
                    "location": v.location,
                }
                for k, v in self.sources.items()
            },
            "unknown_options": dict(self.unknown_options),
            "strict": self.strict,
            "settings_path": self.settings_path,
        }

    def explain(self, key: str) -> str:
        """Human-readable explanation of why ``key`` has its resolved value."""
        canon = get_schema().canonical_name(key)
        lookup = canon or key
        rv = self.sources.get(lookup)
        if rv is None:
            return f"{key}: not set"
        return f"{key} = {rv.value!r}  (source: {rv.source})"


# Keys that are model-level (not calculator/run options) and are consumed by
# the engine directly rather than passed through **calculator_options.
_MODEL_KEYS = {"model_type", "model_path", "task", "device", "inference_mode"}

# Calculator-option keys, per engine. Anything not here and not a model key is
# treated as a run option (scoped to the calc_type by the schema).
_CALCULATOR_KEYS_BY_ENGINE: dict[str, set[str]] = {
    "uma": {"inference_mode", "torch_num_threads", "activation_checkpointing"},
    "fairchem": {"inference_mode", "torch_num_threads", "activation_checkpointing"},
    "mace": {"default_dtype", "head"},
    "dpa": {"head"},
    "grace": {
        "gpu_memory_growth",
        "gpu_memory_limit_mb",
        "neighbor_cache",
        "neighbor_skin",
    },
}


def _is_calculator_key(key: str, model_type: str) -> bool:
    engine = (model_type or "uma").lower()
    return key in _CALCULATOR_KEYS_BY_ENGINE.get(engine, set())


def _settings_layer(
    settings: MlipxSettings | None,
    schema: Schema,
    sections: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, SourceLocation]]:
    """Flatten settings sections while retaining each value's declaration."""
    values: dict[str, Any] = {}
    origins: dict[str, SourceLocation] = {}
    if settings is None:
        return values, origins
    for section_name in sections:
        for key, value in settings.section(section_name).items():
            name = schema.canonical_name(key) or key.lower()
            values[name] = value
            origin = settings.origin(section_name, key)
            if origin is not None:
                origins[name] = origin
    return values, origins


def _mapping_origins(raw: Mapping[str, Any]) -> dict[str, SourceLocation]:
    """Extract per-key origins from an IncarConfig-like mapping."""
    get_origin = getattr(raw, "origin", None)
    if not callable(get_origin):
        return {}
    origins: dict[str, SourceLocation] = {}
    for key in raw:
        origin = get_origin(str(key))
        if origin is not None:
            origins[str(key)] = origin
    return origins


def _canonicalize_layer(
    raw: Mapping[str, Any],
    schema: Schema,
    *,
    source: str,
    base_dir: str | Path | None = None,
    origins: Mapping[str, SourceLocation] | None = None,
) -> dict[str, ResolvedValue]:
    """Canonicalize and schema-coerce one raw layer with provenance."""
    out: dict[str, ResolvedValue] = {}
    default_location = SourceLocation.synthetic(source, base_dir)
    origins = origins or {}
    for raw_key, raw_value in raw.items():
        if raw_value is None or raw_value == "":
            continue
        key = str(raw_key)
        spec = schema.resolve(key)
        name = spec.name if spec is not None else key.lower()
        location = origins.get(key) or origins.get(name) or origins.get(key.lower())
        location = location or default_location
        if spec is None:
            value = raw_value
        else:
            try:
                value = spec.coerce(raw_value)
            except ValueError as exc:
                raise ValueError(f"{location.display}: {exc}") from exc
            errors = spec.validate_value(value)
            if errors:
                raise ValueError(f"{location.display}: " + "; ".join(errors))
        resolved = ResolvedValue(
            value=value,
            source=source,
            base_dir=str(location.base_dir),
            location=location.display,
        )
        previous = out.get(name)
        if previous is not None and previous.value != resolved.value:
            raise ValueError(
                f"Conflicting values for {name!r} in {source}: "
                f"{previous.value!r} vs {resolved.value!r}"
            )
        out[name] = resolved
    return out


def _merge_layer(
    base: dict[str, ResolvedValue], layer: Mapping[str, ResolvedValue]
) -> None:
    """Merge a typed layer into the accumulating resolved source map."""
    base.update(layer)


def _resolve_declared_paths(sources: dict[str, ResolvedValue], schema: Schema) -> None:
    """Resolve every schema-declared path against its exact source base."""
    for key, resolved in list(sources.items()):
        spec = schema.resolve(key)
        if spec is None or not spec.is_path or not resolved.value:
            continue
        path = Path(str(resolved.value)).expanduser()
        if not path.is_absolute():
            if resolved.base_dir is None:
                raise ValueError(f"Path option {key!r} has no source base directory")
            path = Path(resolved.base_dir) / path
        sources[key] = ResolvedValue(
            str(path.resolve()),
            resolved.source,
            base_dir=resolved.base_dir,
            location=resolved.location,
        )


def resolve_config(
    *,
    calc_type: str,
    settings: MlipxSettings | None = None,
    model_aliases: dict[str, ModelAlias] | None = None,
    profiles: dict[str, Profile] | None = None,
    model_alias_name: str | None = None,
    profile_name: str | None = None,
    incar: Mapping[str, Any] | None = None,
    cli: Mapping[str, Any] | None = None,
    incar_base_dir: str | Path | None = None,
    cli_base_dir: str | Path | None = None,
    schema: Schema | None = None,
) -> ResolvedConfig:
    """Resolve a full configuration from all layers.

    Args:
        calc_type: ``sp``/``opt``/``md``/``neb``/``batch``.
        settings: Loaded :class:`MlipxSettings` (may be None).
        model_aliases/profiles: Parsed alias/profile maps (usually extracted
            from ``settings``). If None they are derived from ``settings``.
        model_alias_name: Name of a ``[model:...]`` alias to apply.
        profile_name: Name of a ``[profile:...]`` profile to apply.
        incar: INCAR/job-level overrides (already parsed, keys may be aliases).
        cli: Highest-priority overrides (CLI args or API kwargs).
        incar_base_dir: Declaration directory for a plain INCAR/job mapping.
            An :class:`IncarConfig` supplies its own per-key locations.
        cli_base_dir: Declaration directory for relative CLI/API paths.
            Defaults to the current working directory captured here.
        schema: Optional schema override (defaults to the global one).

    Returns:
        A :class:`ResolvedConfig` with calculator/run options split out and a
        per-key ``sources`` trace.
    """
    schema = schema or get_schema()
    calc_spec = schema.resolve("calc_type")
    assert calc_spec is not None
    calc_type = calc_spec.coerce(calc_type)
    calc_errors = calc_spec.validate_value(calc_type)
    if calc_errors:
        raise ValueError("; ".join(calc_errors))
    model_aliases = (
        model_aliases if model_aliases is not None else _aliases_from_settings(settings)
    )
    profiles = profiles if profiles is not None else _profiles_from_settings(settings)

    sources: dict[str, ResolvedValue] = {}
    cli_base = Path.cwd() if cli_base_dir is None else Path(cli_base_dir)
    cli_base = cli_base.expanduser().resolve()
    inferred_incar_base = getattr(incar, "base_dir", None)
    if inferred_incar_base is None:
        inferred_incar_base = incar_base_dir or Path.cwd()
    inferred_incar_base = Path(inferred_incar_base).expanduser().resolve()
    settings_base = settings.path.parent if settings and settings.path else Path.cwd()

    # Layer 1: built-in defaults.
    defaults_layer: dict[str, Any] = {}
    defaults_layer.update(BUILTIN_DEFAULTS.get("general", {}))
    defaults_layer.update(BUILTIN_DEFAULTS.get(calc_type, {}))
    defaults_layer.update(BUILTIN_DEFAULTS.get("calculator", {}))
    if calc_type in {"md", "neb"}:
        # Implemented force guards are resolved like user [safety] overrides,
        # so execution cannot silently fall back to a different threshold.
        defaults_layer.update(BUILTIN_DEFAULTS.get("safety", {}))
    defaults_layer["device"] = DEFAULT_DEVICE_BY_CALC_TYPE.get(calc_type, "cpu")
    _merge_layer(
        sources,
        _canonicalize_layer(
            defaults_layer,
            schema,
            source="built-in defaults",
            base_dir=Path.cwd(),
        ),
    )

    settings_raw, settings_origins = _settings_layer(
        settings,
        schema,
        ("general", "resources", "output", "safety", calc_type),
    )
    settings_layer = _canonicalize_layer(
        settings_raw,
        schema,
        source="settings.ini",
        base_dir=settings_base,
        origins=settings_origins,
    )

    selected_alias = model_aliases.get(model_alias_name) if model_alias_name else None
    alias_layer = _canonicalize_layer(
        resolve_model_alias(model_alias_name, model_aliases),
        schema,
        source=f"model alias {model_alias_name!r}",
        base_dir=settings_base,
        origins=selected_alias.origins if selected_alias is not None else None,
    )
    selected_profile = profiles.get(profile_name) if profile_name else None
    profile_layer = _canonicalize_layer(
        resolve_profile(profile_name, profiles),
        schema,
        source=f"profile {profile_name!r}",
        base_dir=settings_base,
        origins=selected_profile.origins if selected_profile is not None else None,
    )
    profile_layer.pop("calc_type", None)
    incar_layer = _canonicalize_layer(
        incar or {},
        schema,
        source="INCAR/job",
        base_dir=inferred_incar_base,
        origins=_mapping_origins(incar) if incar is not None else None,
    )
    incar_layer.pop("calc_type", None)
    cli_layer = _canonicalize_layer(
        cli or {},
        schema,
        source="CLI",
        base_dir=cli_base,
    )
    cli_layer.pop("calc_type", None)

    # Select the final engine before applying any settings so its *built-in*
    # defaults participate at the correct lowest precedence.  Previously only
    # [engine:*] settings were merged; factory fallbacks such as MACE dtype and
    # GRACE cache policy therefore affected execution but were absent from
    # resolved_config.json.
    engine_name = "uma"
    for candidate in (
        settings_layer,
        alias_layer,
        profile_layer,
        incar_layer,
        cli_layer,
    ):
        if "model_type" in candidate:
            engine_name = str(candidate["model_type"].value).lower()
    engine_builtin_scope = f"calculator.{engine_name}"
    _merge_layer(
        sources,
        _canonicalize_layer(
            BUILTIN_DEFAULTS.get(engine_builtin_scope, {}),
            schema,
            source=f"built-in defaults ({engine_builtin_scope})",
            base_dir=Path.cwd(),
        ),
    )

    # Layer 2: settings.ini (calculation section + selected engine defaults).
    if settings is not None:
        engine_raw, engine_origins = _settings_layer(
            settings, schema, (f"engine:{engine_name}",)
        )
        settings_layer.update(
            _canonicalize_layer(
                engine_raw,
                schema,
                source="settings.ini",
                base_dir=settings_base,
                origins=engine_origins,
            )
        )
    _merge_layer(sources, settings_layer)

    # Layer 3: model alias.
    _merge_layer(sources, alias_layer)

    # Layer 4: profile.
    # ``calc_type`` is authoritative from the caller (CLI subcommand / API);
    # a profile's calc_type is declarative only and is not allowed to override
    # it. (Plan section 4.3: CLI > profile.)
    _merge_layer(sources, profile_layer)

    # Layer 5: INCAR / job.
    _merge_layer(sources, incar_layer)

    # Layer 6: CLI / kwargs (highest priority).
    _merge_layer(sources, cli_layer)

    _resolve_declared_paths(sources, schema)

    # ---- Finalise model-level fields. ----
    model_type = str(
        sources.get("model_type", ResolvedValue("uma", "built-in defaults")).value
    ).lower()
    model_path_value = sources.get("model_path")
    model_path = str(model_path_value.value) if model_path_value is not None else ""
    task = str(
        sources.get(
            "task",
            ResolvedValue(
                "omat" if model_type in {"uma", "fairchem"} else "bulk",
                "built-in defaults",
            ),
        ).value
    ).lower()
    device = str(
        sources.get(
            "device",
            ResolvedValue(
                DEFAULT_DEVICE_BY_CALC_TYPE.get(calc_type, "cpu"), "built-in defaults"
            ),
        ).value
    )
    # inference_mode defaults: 'turbo' for MD (historical behaviour),
    # 'default' for everything else.
    _default_inference = "turbo" if calc_type == "md" else "default"
    inference_mode = str(
        sources.get(
            "inference_mode", ResolvedValue(_default_inference, "built-in defaults")
        ).value
    )
    if model_type not in {"uma", "fairchem"}:
        inference_source = sources.get("inference_mode")
        if (
            inference_source is not None
            and not inference_source.source.startswith("built-in defaults")
            and str(inference_source.value).lower() != "default"
        ):
            raise ValueError(
                f"inference_mode={inference_source.value!r} is UMA-only; engine "
                f"{model_type!r} would ignore it. Remove the option instead of "
                "recording a mode that does not execute."
            )
        inference_mode = "default"
        sources["inference_mode"] = ResolvedValue(
            "default", f"built-in defaults ({model_type}: not applicable)"
        )
    settings_path = str(settings.path) if settings and settings.path else None

    # ---- Split calculator vs run options. ----
    calculator_options: dict[str, Any] = {}
    run_options: dict[str, Any] = {}
    settings_bag: dict[str, Any] = {}
    unknown_options: dict[str, Any] = {}
    scope_errors: list[str] = []
    calc_scopes = {"sp", "opt", "md", "neb", "batch"}

    for key, rv in sources.items():
        if key in _MODEL_KEYS:
            continue
        spec = schema.resolve(key)
        if spec is None:
            unknown_options[key] = rv.value
            continue
        if _is_calculator_key(key, model_type):
            calculator_options[key] = rv.value
            continue
        option_calc_scopes = set(spec.scopes) & calc_scopes
        non_calc_scopes = set(spec.scopes) - calc_scopes
        if calc_type in option_calc_scopes:
            run_options[key] = rv.value
            continue
        if option_calc_scopes and not non_calc_scopes:
            scope_errors.append(
                f"Option {key!r} from {rv.location or rv.source} is not valid "
                f"for calc_type={calc_type!r}"
            )
            continue
        calculator_scopes = {
            scope for scope in spec.scopes if scope.startswith("calculator.")
        }
        if calculator_scopes:
            scope_errors.append(
                f"Option {key!r} from {rv.location or rv.source} is not "
                f"applicable to engine {model_type!r}"
            )
            continue
        if key == "torch_num_threads" or non_calc_scopes:
            settings_bag[key] = rv.value
            continue
        scope_errors.append(f"Option {key!r} has no executable configuration scope")

    if scope_errors:
        raise ValueError(
            "Configuration scope validation failed:\n  - " + "\n  - ".join(scope_errors)
        )

    strict_value = sources.get("strict_config")
    strict = bool(strict_value.value) if strict_value is not None else False
    unknown_errors: list[str] = []
    for key in unknown_options:
        rv = sources[key]
        suggestion = schema.suggest(key)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        message = f"Unknown key {key!r} from {rv.location or rv.source}.{hint}"
        if strict:
            unknown_errors.append(message)
        else:
            warnings.warn(message, stacklevel=2)
    if unknown_errors:
        raise ValueError(
            "Strict config validation failed:\n  - " + "\n  - ".join(unknown_errors)
        )

    # Seed handling: auto-generate when not set, and record it.
    if "seed" not in sources and calc_type == "md":
        seed = random.randint(0, 2**31 - 1)
        sources["seed"] = ResolvedValue(seed, "auto-generated")
        run_options["seed"] = seed

    resolved = ResolvedConfig(
        calc_type=calc_type,
        model_type=model_type,
        model_path=model_path,
        task=task,
        device=device,
        inference_mode=inference_mode,
        calculator_options=calculator_options,
        run_options=run_options,
        settings=settings_bag,
        sources=sources,
        unknown_options=unknown_options,
        strict=strict,
        settings_path=settings_path,
    )

    _validate_resolved(resolved, schema)
    return resolved


def _validate_resolved(resolved: ResolvedConfig, schema: Schema) -> None:
    """Run schema validation; warn or raise depending on ``strict``."""
    output_format = resolved.settings.get("output_format")
    if output_format is not None and str(output_format).lower() != "vasp":
        raise ValueError(
            f"Unsupported OUTPUT_FORMAT {output_format!r}; only VASP is "
            "currently implemented."
        )

    if resolved.calc_type == "neb":
        for required in ("write_forces", "write_json"):
            if resolved.settings.get(required) is False:
                raise ValueError(
                    f"{required}=false is incompatible with the mandatory NEB "
                    "result/provenance schema"
                )
        for unsupported in (
            "write_outcar",
            "write_xdatcar",
            "write_trajectory",
            "write_stress",
        ):
            if resolved.settings.get(unsupported) is True:
                raise ValueError(
                    f"{unsupported}=true is not a supported NEB output; use the "
                    "complete-band checkpoints and VASP path export"
                )

    if resolved.calc_type == "md":
        options = resolved.run_options
        ensemble = str(options.get("ensemble", "NVT")).lower()
        thermostat = str(options.get("thermostat", "LANGEVIN")).lower()
        com_policy = str(options.get("com_policy", "auto")).lower()
        temperature = float(options.get("temperature", 300.0))
        always_positive = {
            "timestep": float(options.get("timestep", 1.0)),
            "pre_relax_fmax": float(options.get("pre_relax_fmax", 0.1)),
            "fmax_abort": float(
                options.get(
                    "fmax_abort",
                    resolved.settings.get(
                        "fmax_abort", BUILTIN_DEFAULTS["safety"]["fmax_abort"]
                    ),
                )
            ),
        }
        for name, value in always_positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0 for MD")
        if com_policy == "initialize_only" and (
            ensemble == "nve" or thermostat in {"bussi", "nhc"}
        ):
            integrator = "NVE" if ensemble == "nve" else thermostat.upper()
            raise ValueError(
                f"{integrator} does not support com_policy='initialize_only'"
            )
        if ensemble == "nvt" and thermostat == "nhc" and com_policy != "none":
            raise ValueError(
                "NHC requires explicit com_policy='none'; constrained NHC and "
                "automatic COM removal are unsupported"
            )
        if ensemble == "nvt" and thermostat in {"bussi", "nhc"} and temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0 K for NVT {thermostat.upper()} dynamics"
            )
        active_coupling = {
            "langevin": ("friction", float(options.get("friction", 0.001))),
            "bussi": ("bussi_tau", float(options.get("bussi_tau", 1000.0))),
            "nhc": ("nhc_tdamp", float(options.get("nhc_tdamp", 100.0))),
        }
        if ensemble == "nvt":
            coupling_name, coupling_value = active_coupling[thermostat]
            if coupling_value <= 0.0:
                raise ValueError(
                    f"{coupling_name} must be > 0 for NVT {thermostat.upper()} dynamics"
                )

    all_opts: dict[str, Any] = {}
    all_opts.update(resolved.calculator_options)
    all_opts.update(resolved.run_options)

    errors = schema.validate_dict(
        all_opts,
        strict=resolved.strict,
        context=f"calc_type={resolved.calc_type}",
        known_extra=set(resolved.settings.keys()),
    )
    if resolved.strict and errors:
        raise ValueError(
            "Strict config validation failed:\n  - " + "\n  - ".join(errors)
        )
    if errors and not resolved.strict:
        for err in errors:
            warnings.warn(err, stacklevel=2)


def _aliases_from_settings(settings: MlipxSettings | None) -> dict[str, ModelAlias]:
    if settings is None:
        return {}
    from mlipx.config.aliases import parse_model_aliases  # noqa: PLC0415

    return parse_model_aliases(settings.parser, settings.origins)


def _profiles_from_settings(settings: MlipxSettings | None) -> dict[str, Profile]:
    if settings is None:
        return {}
    from mlipx.config.aliases import parse_profiles  # noqa: PLC0415

    return parse_profiles(settings.parser, settings.origins)
