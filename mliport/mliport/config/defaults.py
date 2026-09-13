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
        # Backward compatible: unknown keys warn (existing behaviour). Users opt
        # into hard errors via settings.ini ``strict_config = true``.
        "strict_config": False,
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
        "inference_mode": "turbo",
        "ensemble": "NVT",
        "temperature": 300.0,
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
        "allow_unvalidated_neb": False,
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
# specifies a device. NEB is CPU-safe by default; users must explicitly select GPU.
DEFAULT_DEVICE_BY_CALC_TYPE: dict[str, str] = {
    "sp": "cpu",
    "opt": "cpu",
    "md": "cuda",
    "neb": "cpu",
    "batch": "cpu",
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


def build_incar_default(calc_type: str) -> str:
    """Build the INCAR template text for ``calc_type`` from the single source.

    Replaces the previously hand-written ``DEFAULT_SP_CONFIG`` /
    ``DEFAULT_OPT_CONFIG`` / ``DEFAULT_MD_CONFIG`` strings so there is only one
    place that owns the values (plan section 17.7).
    """
    calc_type = calc_type.lower()
    if calc_type not in {"sp", "opt", "md", "neb"}:
        raise ValueError(f"Unknown calculation type: {calc_type}")

    lines: list[str] = []
    lines.append("# mliport Calculation Settings")
    lines.append(
        f"# Generated from the single source of defaults (calc_type={calc_type})."
    )
    lines.append("")
    calculation_key = "CALCULATION" if calc_type == "neb" else "CALC_TYPE"
    lines.append(f"{calculation_key} = {calc_type.upper()}")
    lines.append("TASK = omat")
    lines.append("")
    lines.append("# Model Settings")
    lines.append("MODEL_TYPE = UMA")
    lines.append("MODEL_PATH = uma-s-1p2p1.pt")
    device = DEFAULT_DEVICE_BY_CALC_TYPE.get(calc_type, "cpu")
    lines.append(f"DEVICE = {device}")
    lines.append("# Molecular tasks only: CHARGE = 0")
    lines.append("# UMA omol spin multiplicity: SPIN = 1")
    if calc_type == "md":
        lines.append("INFERENCE_MODE = turbo")
    else:
        lines.append("INFERENCE_MODE = default")
    lines.append("")

    if calc_type == "opt":
        opt = BUILTIN_DEFAULTS["opt"]
        lines.append("# Optimization Settings")
        lines.append(f"OPT_ALGO = {opt['optimizer']}")
        lines.append(f"FMAX = {opt['fmax']}")
        lines.append(f"MAX_STEPS = {opt['max_steps']}")
        lines.append(f"CELL_OPT = {_bool(opt['cell_opt'])}")
        lines.append(f"FIX_SYMMETRY = {_bool(opt['fix_symmetry'])}")
        lines.append("")
    elif calc_type == "md":
        md = BUILTIN_DEFAULTS["md"]
        lines.append("# MD Settings")
        lines.append(f"MD_ENSEMBLE = {md['ensemble']}")
        lines.append(f"TEMPERATURE = {md['temperature']}")
        lines.append(f"TIMESTEP = {md['timestep']}")
        lines.append(f"EQUILIBRATION_STEPS = {md['equilibration_steps']}")
        lines.append(f"STEPS = {md['steps']}")
        lines.append(f"THERMOSTAT = {md['thermostat']}")
        lines.append(f"FRICTION = {md['friction']}")
        lines.append(f"BUSSI_TAU = {md['bussi_tau']}")
        lines.append(f"NHC_TDAMP = {md['nhc_tdamp']}")
        lines.append(f"NHC_TCHAIN = {md['nhc_tchain']}")
        lines.append(f"NHC_TLOOP = {md['nhc_tloop']}")
        lines.append(f"SAVE_INTERVAL = {md['save_interval']}")
        lines.append(f"COM_POLICY = {md['com_policy']}")
        lines.append("")
    elif calc_type == "neb":
        neb = BUILTIN_DEFAULTS["neb"]
        lines.append("# Fixed-cell NEB Settings")
        lines.append("NEB_INITIAL = initial.vasp")
        lines.append("NEB_FINAL = final.vasp")
        lines.append(f"NEB_IMAGES = {neb['n_intermediate_images']}")
        lines.append(f"NEB_CLIMB = {_bool(neb['climb'])}")
        lines.append(f"NEB_METHOD = {neb['neb_method']}")
        lines.append(f"NEB_INTERPOLATION = {neb['neb_interpolation']}")
        lines.append(f"NEB_PATH_CONVENTION = {neb['path_convention']}")
        lines.append(f"NEB_SPRING = {neb['neb_spring']}")
        lines.append(f"FMAX = {neb['fmax']}")
        lines.append(f"MAX_STEPS = {neb['max_steps']}")
        lines.append(f"NEB_PRE_FMAX = {neb['neb_pre_fmax']}")
        lines.append(f"NEB_PRE_MAX_STEPS = {neb['neb_pre_max_steps']}")
        lines.append(f"NEB_MAXSTEP = {neb['neb_maxstep']}")
        lines.append(f"NEB_ENDPOINT_POLICY = {neb['endpoint_policy']}")
        lines.append(f"NEB_ENDPOINT_FMAX = {neb['endpoint_fmax']}")
        lines.append(f"NEB_ENDPOINT_STEPS = {neb['endpoint_steps']}")
        lines.append(f"IDPP_FMAX = {neb['idpp_fmax']}")
        lines.append(f"IDPP_STEPS = {neb['idpp_steps']}")
        lines.append(f"IDPP_MIC = {_bool(neb['idpp_mic'])}")
        lines.append(f"NEB_MIN_DISTANCE = {neb['neb_min_distance']}")
        lines.append(f"NEB_CHECKPOINT_INTERVAL = {neb['checkpoint_interval']}")
        lines.append("")

    if calc_type != "neb":
        lines.append("# Output Control")
        lines.append("WRITE_FORCES = .TRUE.")
        lines.append("WRITE_STRESS = .TRUE.")
        lines.append("OUTPUT_FORMAT = VASP")
        lines.append("")
    return "".join(line + "\n" for line in lines)


def _bool(value: bool) -> str:
    return ".TRUE." if value else ".FALSE."


def get_default_config(calc_type: str) -> IncarConfig:
    """Return an :class:`IncarConfig` built from the single source of defaults."""
    from mliport.config.incar import IncarConfig  # noqa: PLC0415

    return IncarConfig.from_string(build_incar_default(calc_type))
