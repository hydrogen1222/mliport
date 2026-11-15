"""Single source of truth for mliport built-in defaults.

Per the improvement plan (section 4.5 / 17.7) there must be exactly *one* place
that defines built-in default values. Every other interface (CLI argparse
defaults, the INCAR ``template`` generator, the Python API, the engine fallbacks)
must read from here rather than re-hard-coding its own copy.

Only settings which are currently consumed are exposed here.  Future queue,
restart and output-policy ideas must not masquerade as working configuration:
advertising an ignored safety or restart switch is worse than rejecting it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from mliport.config.incar import IncarConfig


# ---------------------------------------------------------------------------
# Built-in defaults, grouped by scope.
#
# Scopes mirror the settings.ini sections from the plan (section 4.4):
#   general / resources / batch / output / safety / sp / opt / md / neb
# plus the per-engine calculator scopes ``calculator.uma`` / ``calculator.mace``.
#
# Keys are lowercase internal names. The schema (see schema.py) records the
# allowed type, scope and aliases for each key.
# ---------------------------------------------------------------------------
BUILTIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "general": {
        # Scientific config is strict by default: a misspelled or
        # cross-backend option is a fatal error, not a warning. Legacy
        # behaviour is an explicit opt-in via ``strict_config = false`` /
        # ``--lenient-config``.
        "strict_config": True,
        "write_resolved_config": True,
        # None = generate a seed and record it (see resolver).
        "default_seed": None,
    },
    # Keep the scopes present for stable schema/introspection, but do not put
    # ignored queue/output switches in resolved_config.json.
    "resources": {},
    "batch": {},
    "output": {},
    "safety": {
        # MD and NEB check non-finite values unconditionally and apply this
        # finite raw-force safety threshold.
        "fmax_abort": 20.0,
    },
    "sp": {
        "device": "cpu",
        "inference_mode": "default",
    },
    "opt": {
        "inference_mode": "default",
        "optimizer": "FIRE",
        "fmax": 0.05,
        "max_steps": 500,
        "cell_opt": False,
        "fix_symmetry": False,
    },
    "md": {
        # Accuracy-first, like MACE's float64 default: UMA "turbo" enables
        # TF32 matmuls and torch.compile, i.e. reduced precision. It stays an
        # explicit opt-in (INFERENCE_MODE = turbo / --inference-mode turbo).
        "inference_mode": "default",
        "ensemble": "NVT",
        "temperature": 300.0,
        "pressure_gpa": 0.0,
        "npt_tdamp": 100.0,
        "npt_pdamp": 1000.0,
        "npt_tchain": 3,
        "npt_pchain": 3,
        "npt_tloop": 1,
        "npt_ploop": 1,
        "timestep": 1.0,
        "steps": 1000,
        "equilibration_steps": 0,
        "thermostat": "LANGEVIN",
        "friction": 0.001,
        "bussi_tau": 1000.0,
        "nhc_tdamp": 100.0,
        "nhc_tchain": 3,
        "nhc_tloop": 1,
        "save_interval": 10,
        "com_policy": "auto",
        # ``pre_relax`` has NO built-in default here: its default is
        # ensemble-aware (off for NVE, on for NVT) and is applied by the
        # engine (CalculationEngine._create_runner). Putting a blanket
        # True here would override that logic for every config-layer path.
        "pre_relax_steps": 50,
        "pre_relax_fmax": 0.1,
    },
    # confsearch mirrors mliport.confsearch.options.CS_DEFAULTS (single source
    # of truth for the scientific defaults of spec section 7.6 / 12.3).
    "confsearch": {
        "cs_mode": "adsorbate",
        "cs_md_mobile": "adsorbate",
        "cs_top_depth": 2.0,
        "cs_temperature_high": 500.0,
        "cs_temperature_low": 300.0,
        "cs_hold_ps": 2.0,
        "cs_cool_ps": 2.0,
        "cs_snapshot_interval_ps": 0.5,
        "cs_ramp_interval": 10,
        "cs_cycles": 5,
        "cs_save_interval": 100,
        "cs_accept_temperature": 500.0,
        "timestep": 0.5,
        "friction": 0.01,
        "cs_quench_fmax": 0.05,
        "cs_quench_steps": 2000,
        "cs_bond_scale": 1.2,
        "cs_bond_break_scale": 1.5,
        "cs_ionic_species": "Li,Na,K,Rb,Cs,Mg,Ca",
        "cs_reaction_policy": "exclude",
        "cs_desorption_distance": 5.0,
        "cs_min_image_distance": 4.0,
        "cs_min_vacuum_gap": 8.0,
        "cs_place_contact_scale": 1.5,
        "cs_place_anchor_scale": 1.1,
        "cs_place_other_min_scale": 1.3,
        "cs_place_max_attempts": 1000,
        "cs_dedup_energy_tol": 0.010,
        "cs_dedup_geom_tol": 0.20,
        "cs_dedup_fp_tol": 0.0004,
        "cs_fp_rcut": 8.0,
        "cs_fp_binwidth": 0.02,
        "cs_fp_sigma": 0.20,
        "cs_top_k": 10,
        "cs_export_window": 0.5,
        "cs_refine": True,
        "cs_refine_fmax": 0.02,
        "cs_refine_steps": 3000,
        "cs_allow_direct_forces": False,
    },
    "neb": {
        "inference_mode": "default",
        "n_intermediate_images": 7,
        "climb": False,
        "neb_method": "improvedtangent",
        "neb_interpolation": "linear",
        "path_convention": "mic",
        "neb_spring": 0.1,
        "fmax": 0.03,
        "max_steps": 1000,
        "neb_pre_fmax": 0.10,
        "neb_pre_max_steps": 300,
        "neb_maxstep": 0.10,
        "endpoint_policy": "validate",
        "endpoint_fmax": 0.02,
        "endpoint_steps": 500,
        "idpp_fmax": 0.10,
        "idpp_steps": 200,
        "idpp_mic": True,
        "neb_min_distance": 0.5,
        "checkpoint_interval": 10,
    },
    "calculator": {
        # Historical canonical name; the public CLI/TUI calls this CPU Threads.
        # PyTorch consumes it for UMA/MACE/DPA and TensorFlow for GRACE.
        "torch_num_threads": None,
        "activation_checkpointing": None,
    },
    # Engine-specific calculator options. These flow all the way to the
    # underlying ASE calculator (plan section 11.1 / 12).
    "calculator.uma": {
        "torch_num_threads": None,
        "activation_checkpointing": None,
    },
    "calculator.mace": {
        # Accuracy is the scientific default. Users may explicitly opt into
        # float32 with --dtype float32 / DEFAULT_DTYPE=float32 for throughput.
        "default_dtype": "float64",
        "head": None,
    },
    "calculator.dpa": {},
    "calculator.grace": {
        # TensorFlow otherwise commonly reserves nearly all visible VRAM.
        "gpu_memory_growth": True,
        # Verlet-style neighbor-list cache.  The periodic-image multiset is
        # equivalent to a fresh search; pair ordering is not a public contract.
        "neighbor_cache": True,
        "neighbor_skin": 1.5,
    },
}


# Map calc_type -> default device, used by the CLI resolver when nothing else
# specifies a device. Every generic template is CPU-safe: a device default must
# not assume that the user's machine has a GPU. GPU users pass --device cuda
# (or cuda:N) explicitly; the installer and queue record the chosen device.
DEFAULT_DEVICE_BY_CALC_TYPE: dict[str, str] = {
    "sp": "cpu",
    "opt": "cpu",
    "md": "cpu",
    "neb": "cpu",
    "batch": "cpu",
    "confsearch": "cpu",
}


def get_default(calc_type: str | None, key: str, default: Any = None) -> Any:
    """Return a built-in default value for ``key`` scoped to ``calc_type``.

    Looks up ``<calc_type>`` scope first, then falls back to ``general`` /
    ``calculator``. ``calc_type`` may be ``None`` to only consult the
    non-calc-type scopes.
    """
    scopes: list[str] = []
    if calc_type:
        scopes.append(calc_type)
    scopes.extend(["general", "calculator"])
    for scope in scopes:
        block = BUILTIN_DEFAULTS.get(scope)
        if block and key in block:
            return block[key]
    return default


def _neutral_model_section() -> list[str]:
    """Backend-neutral model section: no implicit UMA (task book 5.2/5.3)."""
    return [
        "# No MLIP backend is selected by default. Choose one explicitly:",
        f"MODEL_TYPE = REQUIRED{_model_comment('model_type')}",
        f"MODEL_PATH = REQUIRED{_model_comment('model_path')}",
        "#",
        "# Backend examples (fairchem is an alias for uma):",
        "#   MODEL_TYPE = UMA    TASK = omat    # UMA task family: omat/omol/...",
        "#   MODEL_TYPE = MACE   TASK = bulk    # optional: HEAD, DEFAULT_DTYPE",
        "#   MODEL_TYPE = DPA    TASK = bulk    HEAD = Omat24",
        "#   MODEL_TYPE = GRACE  TASK = bulk    # optional: NEIGHBOR_CACHE",
        "#   MODEL_TYPE = NEP    TASK = bulk    NEP_RUNTIME = ase|gpumd",
        "#                       (gpumd also needs NEP_GPUMD_COMMAND)",
    ]


def _model_comment(key: str) -> str:
    """The one-line meaning of a model key, from the shared option table."""
    from mliport.config.schema import get_schema  # noqa: PLC0415

    spec = get_schema().resolve(key)
    if spec is None or not spec.description:  # pragma: no cover - table invariant
        return ""
    return f"  # {spec.description}"


def _engine_model_section(engine: str, calc_type: str) -> list[str]:
    """Engine-specific model fields (only what that backend really needs).

    The keys are the ones the option table declares for the model section; the
    backend-specific *values* (UMA's task family, DPA's branch, NEP's runtime)
    are the part that genuinely depends on the engine. Explanations come from
    the table, so they cannot drift from `docs/incar.md`.
    """
    from mliport.backend_selection import is_uma  # noqa: PLC0415

    normalized = str(engine).strip().lower()
    lines = [f"MODEL_TYPE = {'UMA' if is_uma(normalized) else normalized.upper()}"]
    lines.append(f"MODEL_PATH = REQUIRED{_model_comment('model_path')}")
    if is_uma(normalized):
        lines.append("TASK = omat")
        lines.append("# Molecular tasks only: CHARGE = 0")
        lines.append("# UMA omol spin multiplicity: SPIN = 1")
        lines.append("INFERENCE_MODE = default")
        if calc_type == "md":
            lines.append("# INFERENCE_MODE = turbo  # faster: TF32 + torch.compile")
    elif normalized == "mace":
        lines.append("TASK = bulk")
        lines.append("# Optional MACE fields:")
        lines.append("# HEAD = default")
        lines.append("# DEFAULT_DTYPE = float64")
    elif normalized == "dpa":
        lines.append("TASK = bulk")
        lines.append("HEAD = Omat24  # branch of the multi-task checkpoint")
    elif normalized == "grace":
        lines.append("TASK = bulk")
        lines.append("# Optional GRACE fields:")
        lines.append("# NEIGHBOR_CACHE = true")
        lines.append("# NEIGHBOR_SKIN = 1.5")
    elif normalized == "nep":
        # NEP has two execution runtimes; the runtime is a required user
        # decision and is never inferred (PR #61 section C).
        lines.append("TASK = bulk")
        lines.append("NEP_RUNTIME = ase  # ase | gpumd (required)")
        lines.append("# NEP_GPUMD_COMMAND = /path/to/gpumd")
    else:  # pragma: no cover - require_model_type validates before this point
        lines.append("TASK = bulk")
    return lines


def _template_value(calc_type: str, spec) -> Any:
    """The value a template writes for one option.

    Placeholders win (the user must supply the value), then a display-only
    ``template_value``, then the built-in default for this calculation type,
    then the schema default.
    """
    if spec.placeholder:
        return spec.placeholder
    if spec.template_value is not None:
        return spec.template_value
    value = get_default(calc_type, spec.name)
    if value is not None:
        return value
    if spec.default is not None:
        return spec.default
    return ""


def _format_template_value(value: Any) -> str:
    if isinstance(value, bool):
        return _bool(value)
    return str(value)


def _template_line(spec, value: Any, *, commented: bool = False) -> str:
    """One ``KEY = value  # one-line meaning`` template line."""
    line = f"{spec.incar_key} = {_format_template_value(value)}"
    note = spec.description.strip()
    if note:
        line = f"{line}  # {note}"
    return f"# {line}" if commented else line


def build_incar_default(calc_type: str, engine: str | None = None) -> str:
    """Build the INCAR template text for ``calc_type`` from the single source.

    ``engine=None`` renders a backend-neutral template: no implicit UMA, the
    user must select one of the four runtimes.  A specific ``engine`` renders
    only the fields that backend actually needs (UMA task family vs the
    explicit ``bulk``/``molecule`` PBC semantics of MACE/DPA/GRACE).

    The body of the template is generated from the option table in
    :mod:`mliport.config.schema` (#61 R3.1/R3.4): every section appears exactly
    once, in ``TEMPLATE_GROUP_ORDER``; each key is listed once with its one-line
    meaning; a key whose ``applies_when`` condition does not hold for the
    template's own values is either left out (thermostat-specific keys) or kept
    as a commented example (the NPT block of the MD template).
    """
    from mliport.backend_selection import normalize_model_type  # noqa: PLC0415
    from mliport.config.schema import (  # noqa: PLC0415
        TEMPLATE_GROUP_ORDER,
        get_schema,
    )

    calc_type = calc_type.lower()
    if calc_type not in {"sp", "opt", "md", "neb", "confsearch"}:
        raise ValueError(f"Unknown calculation type: {calc_type}")
    if engine is not None:
        engine = normalize_model_type(engine)

    all_specs = get_schema().specs
    specs = [
        spec
        for spec in all_specs
        if spec.template
        and (not spec.template_scopes or calc_type in spec.template_scopes)
    ]
    values = {spec.name: _template_value(calc_type, spec) for spec in specs}
    all_specs_by_name = {spec.name: spec for spec in all_specs}
    # Conditional keys need the value of the key their condition names, even
    # when that key is not itself part of the template.
    for spec in all_specs:
        for key, _expected in spec.applies_when:
            if key not in values:
                values[key] = _template_value(calc_type, all_specs_by_name[key])

    lines: list[str] = []
    lines.append("# mliport Calculation Settings")
    lines.append(
        f"# Generated from the single source of defaults (calc_type={calc_type})."
    )
    lines.append("")

    for group in TEMPLATE_GROUP_ORDER:
        group_specs = [spec for spec in specs if spec.group == group]
        if not group_specs and group != "Model":
            continue
        body: list[str] = []
        # Conditional keys whose condition does not hold: keep them as a
        # commented example only when the table asks for one.
        examples: dict[tuple[tuple[str, str], ...], list] = {}
        for spec in sorted(group_specs, key=lambda item: item.order):
            if group == "Calculation":
                # ``CALC_TYPE``/``CALCULATION`` is the header of every template;
                # NEB keeps its historical CALCULATION spelling.
                key = "CALCULATION" if calc_type == "neb" else spec.incar_key
                note = f"  # {spec.description}" if spec.description else ""
                body.append(f"{key} = {calc_type.upper()}{note}")
                continue
            if group == "Model":
                continue  # rendered backend-specifically below
            if spec.applies_to(values, calc_type):
                body.append(_template_line(spec, values[spec.name]))
            elif spec.example_when_inactive:
                examples.setdefault(spec.applies_when, []).append(spec)

        if group == "Model":
            body = (
                _neutral_model_section()
                if engine is None
                else _engine_model_section(engine, calc_type)
            )
        if not body and not examples:
            continue
        lines.append(f"# {group} Settings")
        lines.extend(body)
        for condition, example_specs in examples.items():
            note = next(
                (spec.example_note for spec in example_specs if spec.example_note), ""
            )
            if note:
                lines.append(f"# {note}")
            if example_specs[0].example_condition_line:
                for key, value in condition:
                    spec = get_schema().resolve(key)
                    spelling = spec.incar_key if spec is not None else key.upper()
                    lines.append(
                        f"# {spelling} = {value}  # switch that enables this block"
                    )
            for spec in example_specs:
                lines.append(_template_line(spec, values[spec.name], commented=True))
        lines.append("")

    return "".join(line + "\n" for line in lines)


def _bool(value: bool) -> str:
    return ".TRUE." if value else ".FALSE."


def get_default_config(calc_type: str, engine: str | None = None) -> IncarConfig:
    """Return an :class:`IncarConfig` built from the single source of defaults.

    ``engine=None`` yields the backend-neutral template (no implicit UMA).
    """
    from mliport.config.incar import IncarConfig  # noqa: PLC0415

    return IncarConfig.from_string(build_incar_default(calc_type, engine=engine))
