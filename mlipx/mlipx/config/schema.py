"""Configuration schema and strict validation (plan section 10).

The schema records, for every recognised option:

* its canonical (internal) name,
* the Python type,
* the scopes it belongs to (``sp``/``opt``/``md``/``neb``/``batch``/``calculator``/
  ``general`` ...),
* any aliases (e.g. ``TEMPERATURE`` / ``--temp`` / ``temperature``),
* allowed ``choices`` and numeric bounds.

In strict mode (``strict_config = true``) an unknown key is a hard error and the
schema suggests the closest legal key via :func:`difflib.get_close_matches`, so a
typo such as ``DEFAULT_DTPE`` is reported as::

    Unknown key DEFAULT_DTPE in [job:01_sp]. Did you mean DEFAULT_DTYPE?

instead of being silently ignored.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass(frozen=True)
class OptionSpec:
    """Specification of a single recognised option."""

    name: str
    type: type = str
    scopes: frozenset[str] = field(default_factory=frozenset)
    aliases: frozenset[str] = field(default_factory=frozenset)
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    default: Any | None = None
    description: str = ""
    is_path: bool = False

    def matches(self, key: str) -> bool:
        """True if ``key`` (case-insensitive) is this option's name or an alias."""
        k = key.lower()
        return k == self.name.lower() or k in {a.lower() for a in self.aliases}

    def coerce(self, value: Any) -> Any:
        """Coerce ``value`` to this spec's type, raising on failure."""
        if self.type is bool:
            return _coerce_bool(value)
        if self.type is int:
            return _coerce_int(self.name, value)
        if self.type is float:
            return _coerce_float(self.name, value)
        if self.type is str:
            return _coerce_string(self.name, value, self.choices)
        try:
            return self.type(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Option '{self.name}' expects {self.type.__name__}, got {value!r}"
            ) from exc

    def validate_value(self, value: Any) -> list[str]:
        """Return a list of human-readable validation errors for ``value``."""
        errors: list[str] = []
        if self.choices is not None and value not in self.choices:
            errors.append(
                f"Option '{self.name}' must be one of "
                f"{', '.join(repr(c) for c in self.choices)}, got {value!r}"
            )
        if self.minimum is not None and isinstance(value, (int, float)):
            if value < self.minimum:
                errors.append(
                    f"Option '{self.name}'={value} is below minimum {self.minimum}"
                )
        if self.maximum is not None and isinstance(value, (int, float)):
            if value > self.maximum:
                errors.append(
                    f"Option '{self.name}'={value} is above maximum {self.maximum}"
                )
        return errors


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "t", "yes", "y", "1", ".true.", ".t."}:
            return True
        if v in {"false", "f", "no", "n", "0", ".false.", ".f."}:
            return False
    raise ValueError(f"Cannot convert {value!r} to boolean")


def _coerce_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Option '{name}' expects int, got bool {value!r}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Option '{name}' expects a finite int, got {value!r}")
        if numeric.is_integer():
            return int(numeric)
        raise ValueError(f"Option '{name}' expects int, got {value!r}")
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"Option '{name}' expects int, got {value!r}")


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Option '{name}' expects float, got bool {value!r}")
    token = value
    if isinstance(value, str):
        token = value.strip().replace("D", "E").replace("d", "e")
    try:
        numeric = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Option '{name}' expects float, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Option '{name}' expects a finite float, got {value!r}")
    return numeric


def _coerce_string(name: str, value: Any, choices: tuple[Any, ...] | None) -> str:
    if value is None:
        raise ValueError(f"Option '{name}' expects str, got None")
    text = value if isinstance(value, str) else str(value)
    if choices is not None and all(isinstance(choice, str) for choice in choices):
        stripped = text.strip()
        for choice in choices:
            if stripped.casefold() == choice.casefold():
                return choice
    return text


# ---------------------------------------------------------------------------
# Registry of recognised options.
#
# Names are lowercase internal names. Aliases include the INCAR (UPPER) form and
# common CLI short forms so the same schema validates every interface.
# ---------------------------------------------------------------------------
_SPECS: list[OptionSpec] = [
    # --- global configuration ---
    OptionSpec(
        "strict_config",
        bool,
        frozenset({"general"}),
        aliases={"STRICT_CONFIG"},
        description="Treat unknown configuration keys as errors.",
    ),
    OptionSpec(
        "write_resolved_config",
        bool,
        frozenset({"general"}),
        aliases={"WRITE_RESOLVED_CONFIG"},
        description="Write resolved_config.json before a calculation.",
    ),
    OptionSpec(
        "fmax_abort",
        float,
        frozenset({"safety"}),
        aliases={"FMAX_ABORT"},
        minimum=0.0,
        description="MD force-safety abort threshold in eV/Angstrom.",
    ),
    # --- model / device (calculator scope) ---
    OptionSpec(
        "model_type",
        str,
        frozenset({"calculator"}),
        aliases={"MODEL_TYPE", "model"},
        choices=("uma", "fairchem", "mace", "dpa", "grace"),
        description="MLIP engine.",
    ),
    OptionSpec(
        "model_path",
        str,
        frozenset({"calculator"}),
        aliases={"MODEL_PATH", "model"},
        description="Path to the model checkpoint/file.",
        is_path=True,
    ),
    OptionSpec(
        "task",
        str,
        frozenset({"calculator"}),
        aliases={"TASK"},
        description="UMA task (omat/omol/...) or bulk/molecule for other engines.",
    ),
    OptionSpec(
        "device",
        str,
        frozenset({"calculator", "general"}),
        aliases={"DEVICE"},
        description="Compute device: cpu, cuda, gpu or cuda:N.",
    ),
    OptionSpec(
        "inference_mode",
        str,
        frozenset({"calculator"}),
        aliases={"INFERENCE_MODE"},
        choices=("default", "turbo"),
        description=(
            "UMA inference mode. Non-default modes are rejected for other engines."
        ),
    ),
    # --- MACE calculator options (plan section 11.1 / 12) ---
    OptionSpec(
        "default_dtype",
        str,
        frozenset({"calculator.mace"}),
        aliases={"DEFAULT_DTYPE", "dtype"},
        choices=("float32", "float64"),
        description="MACE model dtype (accuracy-first default: float64).",
    ),
    OptionSpec(
        "head",
        str,
        frozenset({"calculator.mace", "calculator.dpa"}),
        aliases={"HEAD"},
        description="MACE or DeepMD model head/branch (multi-task models).",
    ),
    OptionSpec(
        "torch_num_threads",
        int,
        frozenset({"calculator"}),
        aliases={"TORCH_NUM_THREADS", "CPU_THREADS", "cpu_threads"},
        minimum=1,
        description=(
            "Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; "
            "TensorFlow for GRACE; legacy canonical field name)."
        ),
    ),
    OptionSpec(
        "activation_checkpointing",
        bool,
        frozenset({"calculator.uma", "calculator.fairchem"}),
        aliases={"ACTIVATION_CHECKPOINTING"},
        description="GPU memory saving (UMA, overrides inference_mode preset).",
    ),
    OptionSpec(
        "gpu_memory_growth",
        bool,
        frozenset({"calculator.grace"}),
        aliases={"GPU_MEMORY_GROWTH"},
        description=(
            "GRACE TensorFlow allocator grows on demand instead of claiming "
            "the whole visible GPU."
        ),
    ),
    OptionSpec(
        "gpu_memory_limit_mb",
        int,
        frozenset({"calculator.grace"}),
        aliases={"GPU_MEMORY_LIMIT_MB"},
        minimum=1,
        description="Hard GRACE TensorFlow logical-device memory limit in MiB.",
    ),
    OptionSpec(
        "neighbor_cache",
        bool,
        frozenset({"calculator.grace"}),
        aliases={"NEIGHBOR_CACHE"},
        description=(
            "GRACE verlet-style neighbor-list cache (extended cutoff + exact "
            "re-filter). The complete periodic-image multiset matches a fresh "
            "search; pair ordering may differ. Default: True."
        ),
    ),
    OptionSpec(
        "neighbor_skin",
        float,
        frozenset({"calculator.grace"}),
        aliases={"NEIGHBOR_SKIN"},
        minimum=1.0e-12,
        description=(
            "GRACE neighbor-cache skin in Å: the cached neighbor list is "
            "rebuilt when any atom moves more than skin/2. Larger values "
            "rebuild less often but filter longer lists."
        ),
    ),
    # --- molecular electronic state (applied to atoms.info by every runner) ---
    OptionSpec(
        "charge",
        int,
        frozenset({"sp", "opt", "md", "neb", "batch"}),
        aliases={"CHARGE"},
        minimum=-100,
        maximum=100,
        description="Total molecular charge stored in atoms.info.",
    ),
    OptionSpec(
        "spin",
        int,
        frozenset({"sp", "opt", "md", "neb", "batch"}),
        aliases={"SPIN", "SPIN_MULTIPLICITY"},
        minimum=0,
        maximum=100,
        description=(
            "Molecular spin metadata; UMA omol uses spin multiplicity (2S+1)."
        ),
    ),
    # --- OPT run options ---
    OptionSpec(
        "optimizer",
        str,
        frozenset({"opt"}),
        aliases={"OPT_ALGO", "opt_algo"},
        choices=("FIRE", "BFGS", "LBFGS"),
        description="Geometry optimization algorithm.",
    ),
    OptionSpec(
        "fmax",
        float,
        frozenset({"opt", "neb"}),
        aliases={"FMAX"},
        minimum=0.0,
        description="Force convergence threshold in eV/Angstrom.",
    ),
    OptionSpec(
        "max_steps",
        int,
        frozenset({"opt", "neb"}),
        aliases={"MAX_STEPS"},
        minimum=0,
        description="Maximum optimization steps.",
    ),
    OptionSpec(
        "cell_opt",
        bool,
        frozenset({"opt"}),
        aliases={"CELL_OPT"},
        description="Optimize cell parameters (requires stress support).",
    ),
    OptionSpec(
        "fix_symmetry",
        bool,
        frozenset({"opt"}),
        aliases={"FIX_SYMMETRY"},
        description="Preserve crystal symmetry during optimization.",
    ),
    # --- fixed-cell NEB run options ---
    OptionSpec(
        "neb_initial",
        str,
        frozenset({"neb"}),
        aliases={"NEB_INITIAL"},
        is_path=True,
        description="Initial endpoint structure for NEB.",
    ),
    OptionSpec(
        "neb_final",
        str,
        frozenset({"neb"}),
        aliases={"NEB_FINAL"},
        is_path=True,
        description="Final endpoint structure for NEB.",
    ),
    OptionSpec(
        "n_intermediate_images",
        int,
        frozenset({"neb"}),
        aliases={"NEB_IMAGES", "IMAGES"},
        minimum=1,
        description="Number of intermediate images; endpoints are additional.",
    ),
    OptionSpec(
        "climb",
        bool,
        frozenset({"neb"}),
        aliases={"NEB_CLIMB", "CLIMB"},
        description="Run a second climbing-image NEB stage.",
    ),
    OptionSpec(
        "neb_method",
        str,
        frozenset({"neb"}),
        aliases={"NEB_METHOD"},
        choices=("improvedtangent",),
        description="ASE NEB tangent method.",
    ),
    OptionSpec(
        "neb_interpolation",
        str,
        frozenset({"neb"}),
        aliases={"NEB_INTERPOLATION"},
        choices=("linear", "idpp"),
        description="Initial band interpolation method.",
    ),
    OptionSpec(
        "path_convention",
        str,
        frozenset({"neb"}),
        aliases={"NEB_PATH_CONVENTION"},
        choices=("mic", "unwrapped"),
        description="Periodic path convention.",
    ),
    OptionSpec(
        "neb_spring",
        float,
        frozenset({"neb"}),
        aliases={"NEB_SPRING"},
        minimum=1.0e-12,
        description="NEB spring constant in eV/Angstrom^2.",
    ),
    OptionSpec(
        "neb_pre_fmax",
        float,
        frozenset({"neb"}),
        aliases={"NEB_PRE_FMAX"},
        minimum=1.0e-12,
        description="Ordinary pre-stage force criterion in eV/Angstrom.",
    ),
    OptionSpec(
        "neb_pre_max_steps",
        int,
        frozenset({"neb"}),
        aliases={"NEB_PRE_MAX_STEPS"},
        minimum=0,
        description="Maximum ordinary pre-stage optimizer steps.",
    ),
    OptionSpec(
        "neb_maxstep",
        float,
        frozenset({"neb"}),
        aliases={"NEB_MAXSTEP"},
        minimum=1.0e-12,
        description="Maximum FIRE displacement per step in Angstrom.",
    ),
    OptionSpec(
        "endpoint_policy",
        str,
        frozenset({"neb"}),
        aliases={"NEB_ENDPOINT_POLICY"},
        choices=("validate", "relax"),
        description="Validate or relax endpoint geometries before NEB.",
    ),
    OptionSpec(
        "endpoint_fmax",
        float,
        frozenset({"neb"}),
        aliases={"NEB_ENDPOINT_FMAX"},
        minimum=1.0e-12,
        description="Endpoint convergence threshold in eV/Angstrom.",
    ),
    OptionSpec(
        "endpoint_steps",
        int,
        frozenset({"neb"}),
        aliases={"NEB_ENDPOINT_STEPS"},
        minimum=1,
        description="Maximum endpoint relaxation steps.",
    ),
    OptionSpec(
        "idpp_fmax",
        float,
        frozenset({"neb"}),
        aliases={"IDPP_FMAX"},
        minimum=1.0e-12,
        description="IDPP interpolation force criterion.",
    ),
    OptionSpec(
        "idpp_steps",
        int,
        frozenset({"neb"}),
        aliases={"IDPP_STEPS"},
        minimum=1,
        description="Maximum IDPP interpolation steps.",
    ),
    OptionSpec(
        "idpp_mic",
        bool,
        frozenset({"neb"}),
        aliases={"IDPP_MIC"},
        description="Use minimum-image distances in IDPP.",
    ),
    OptionSpec(
        "neb_min_distance",
        float,
        frozenset({"neb"}),
        aliases={"NEB_MIN_DISTANCE"},
        minimum=1.0e-12,
        description="Minimum allowed interatomic distance in Angstrom.",
    ),
    OptionSpec(
        "checkpoint_interval",
        int,
        frozenset({"neb"}),
        aliases={"NEB_CHECKPOINT_INTERVAL"},
        minimum=1,
        description="Trusted optimizer-state interval between full-band checkpoints.",
    ),
    # --- MD run options ---
    OptionSpec(
        "allow_unvalidated_neb",
        bool,
        frozenset({"neb"}),
        aliases={"NEB_ALLOW_UNVALIDATED"},
        description="Explicit experimental override for unknown energy-gradient consistency.",
    ),
    OptionSpec(
        "ensemble",
        str,
        frozenset({"md"}),
        aliases={"MD_ENSEMBLE"},
        choices=("NVT", "NVE"),
        description="MD ensemble.",
    ),
    OptionSpec(
        "temperature",
        float,
        frozenset({"md"}),
        aliases={"TEMPERATURE", "temp"},
        minimum=0.0,
        description="Temperature in Kelvin.",
    ),
    OptionSpec(
        "timestep",
        float,
        frozenset({"md"}),
        aliases={"TIMESTEP", "timestep_fs"},
        minimum=0.0,
        description="Time step in femtoseconds.",
    ),
    OptionSpec(
        "steps",
        int,
        frozenset({"md"}),
        aliases={"STEPS", "production_steps", "PRODUCTION_STEPS"},
        minimum=0,
        description="Number of production MD steps.",
    ),
    OptionSpec(
        "equilibration_steps",
        int,
        frozenset({"md"}),
        aliases={"EQUIL_STEPS", "EQUILIBRATION_STEPS"},
        minimum=0,
        default=0,
        description=("Number of same-ensemble equilibration steps before production."),
    ),
    OptionSpec(
        "thermostat",
        str,
        frozenset({"md"}),
        aliases={"THERMOSTAT"},
        choices=("LANGEVIN", "BUSSI", "NHC"),
        description="NVT thermostat (Langevin, Bussi/CSVR, or Nose-Hoover chain).",
    ),
    OptionSpec(
        "friction",
        float,
        frozenset({"md"}),
        aliases={"FRICTION", "friction_per_fs"},
        minimum=0.0,
        description="Langevin friction coefficient (1/fs).",
    ),
    OptionSpec(
        "bussi_tau",
        float,
        frozenset({"md"}),
        aliases={"BUSSI_TAU"},
        minimum=0.0,
        description="Bussi/CSVR coupling time in femtoseconds.",
    ),
    OptionSpec(
        "nhc_tdamp",
        float,
        frozenset({"md"}),
        aliases={"NHC_TDAMP"},
        minimum=0.0,
        description="Nose-Hoover-chain damping time in femtoseconds.",
    ),
    OptionSpec(
        "nhc_tchain",
        int,
        frozenset({"md"}),
        aliases={"NHC_TCHAIN"},
        minimum=1,
        description="Number of Nose-Hoover thermostat variables.",
    ),
    OptionSpec(
        "nhc_tloop",
        int,
        frozenset({"md"}),
        aliases={"NHC_TLOOP"},
        minimum=1,
        description="Nose-Hoover thermostat integration substeps.",
    ),
    OptionSpec(
        "save_interval",
        int,
        frozenset({"md"}),
        aliases={"SAVE_INTERVAL", "trajectory_interval", "TRAJECTORY_INTERVAL"},
        minimum=1,
        description="Interval for saving trajectory frames.",
    ),
    OptionSpec(
        "pre_relax",
        bool,
        frozenset({"md"}),
        aliases={"PRE_RELAX"},
        description="Pre-relax the structure before MD (legacy).",
    ),
    OptionSpec(
        "pre_relax_steps",
        int,
        frozenset({"md"}),
        aliases={"PRE_RELAX_STEPS"},
        minimum=0,
        description="Maximum pre-relaxation steps (legacy).",
    ),
    OptionSpec(
        "pre_relax_fmax",
        float,
        frozenset({"md"}),
        aliases={"PRE_RELAX_FMAX"},
        minimum=0.0,
        description="Pre-relaxation force threshold (legacy).",
    ),
    # --- plan vocabulary recognised as aliases / future keys (Phase 3+) ---
    OptionSpec(
        "pre_relax_mode",
        str,
        frozenset({"md"}),
        aliases={"PRE_RELAX_MODE"},
        choices=("none", "positions", "cell"),
        default="none",
        description="Pre-relaxation mode (plan section 13.2; Phase 3).",
    ),
    OptionSpec(
        "velocity_policy",
        str,
        frozenset({"md"}),
        aliases={"VELOCITY_POLICY"},
        choices=("auto", "initialize", "preserve"),
        default="auto",
        description="Velocity initialisation policy (plan section 13.5; Phase 3).",
    ),
    OptionSpec(
        "com_policy",
        str,
        frozenset({"md"}),
        aliases={"COM_POLICY"},
        choices=("auto", "none", "initialize_only", "constraint"),
        default="auto",
        description="Explicit center-of-mass policy for MD.",
    ),
    OptionSpec(
        "seed",
        int,
        frozenset({"md", "general"}),
        aliases={"SEED", "default_seed", "DEFAULT_SEED"},
        minimum=0,
        description="Random seed for reproducible MD.",
    ),
    # --- batch run options ---
    OptionSpec(
        "sub_calc_type",
        str,
        frozenset({"batch"}),
        aliases={"sub-calc-type"},
        choices=("sp", "opt"),
        description="Sub-calculation type for a sweep batch (sp/opt).",
    ),
    # ``calc_type`` is a meta key: it selects the runner (sp/opt/md/neb/batch).
    # The resolver pops it from every layer; it never becomes a run option.
    OptionSpec(
        "calc_type",
        str,
        frozenset({"meta"}),
        aliases={"CALC_TYPE", "CALCULATION"},
        # Keep in sync with CalculationEngine.VALID_CALC_TYPES and
        # IncarConfig.validate(): "analyze" was previously accepted here but
        # rejected by the engine with a late, confusing error.
        choices=("sp", "opt", "md", "neb", "batch"),
        description="Top-level calculation type (selects the runner).",
    ),
    OptionSpec(
        "pattern",
        str,
        frozenset({"batch"}),
        aliases={"PATTERN"},
        description="Glob pattern for batch structure discovery.",
    ),
    # --- INCAR output / meta keys (recognised so INCAR flows don't warn).
    # These are consumed by writers, not runners; they land in the settings bag.
    OptionSpec(
        "write_forces",
        bool,
        frozenset({"output"}),
        aliases={"WRITE_FORCES"},
        description="Write forces to OUTCAR.",
    ),
    OptionSpec(
        "write_outcar",
        bool,
        frozenset({"output"}),
        aliases={"WRITE_OUTCAR"},
        description="Write the VASP-like OUTCAR interoperability view.",
    ),
    OptionSpec(
        "write_xdatcar",
        bool,
        frozenset({"output"}),
        aliases={"WRITE_XDATCAR"},
        description="Write the VASP XDATCAR interoperability trajectory.",
    ),
    OptionSpec(
        "write_stress",
        bool,
        frozenset({"output"}),
        aliases={"WRITE_STRESS"},
        description="Write stress to OUTCAR.",
    ),
    OptionSpec(
        "write_trajectory",
        bool,
        frozenset({"output"}),
        aliases={"WRITE_TRAJECTORY"},
        description="Write ASE trajectory.",
    ),
    OptionSpec(
        "write_json",
        bool,
        frozenset({"output"}),
        aliases={"WRITE_JSON"},
        description="Write JSON results.",
    ),
    OptionSpec(
        "output_format",
        str,
        frozenset({"output"}),
        aliases={"OUTPUT_FORMAT"},
        description="Output format (VASP).",
    ),
    OptionSpec(
        "job_name",
        str,
        frozenset({"meta"}),
        aliases={"JOB_NAME"},
        description="Optional job name.",
    ),
]


def get_schema() -> Schema:
    """Return the singleton :class:`Schema` instance."""
    global _SCHEMA  # noqa: PLW0603
    if _SCHEMA is None:
        _SCHEMA = Schema(_SPECS)
    return _SCHEMA


_SCHEMA: Schema | None = None


class Schema:
    """Registry of recognised :class:`OptionSpec` objects."""

    def __init__(self, specs: list[OptionSpec]):
        self._specs: list[OptionSpec] = list(specs)
        self._by_name: dict[str, OptionSpec] = {}
        for spec in self._specs:
            self._by_name[spec.name.lower()] = spec
            for alias in spec.aliases:
                self._by_name[alias.lower()] = spec

    @property
    def specs(self) -> list[OptionSpec]:
        return list(self._specs)

    def known_names(self) -> set[str]:
        """All recognised keys (canonical names + aliases), lowercased."""
        return set(self._by_name.keys())

    def canonical_names(self) -> set[str]:
        """Only canonical option names (lowercased)."""
        return {s.name.lower() for s in self._specs}

    def resolve(self, key: str) -> OptionSpec | None:
        """Return the spec matching ``key`` (case-insensitive) or ``None``."""
        return self._by_name.get(key.lower())

    def canonical_name(self, key: str) -> str | None:
        """Return the canonical name for ``key`` or ``None`` if unknown."""
        spec = self.resolve(key)
        return spec.name if spec is not None else None

    def suggest(self, key: str, n: int = 3) -> list[str]:
        """Return the closest recognised keys to ``key`` (for typo hints)."""
        candidates = sorted(self._by_name.keys())
        return difflib.get_close_matches(key.lower(), candidates, n=n, cutoff=0.6)

    def validate_dict(
        self,
        values: dict[str, Any],
        *,
        strict: bool,
        context: str = "",
        known_extra: set[str] | None = None,
    ) -> list[str]:
        """Validate a flat ``{key: value}`` mapping.

        Args:
            values: Options to validate (keys may be canonical or alias forms).
            strict: When True, unknown keys are returned as errors (with a typo
                suggestion); when False, unknown keys are ignored by this method
                and the caller may choose to warn.
            context: Human-readable location included in error messages, e.g.
                ``"[job:01_sp]"`` or ``"CLI --opt"``.
            known_extra: Additional keys to treat as known (e.g. keys consumed
                by the resolver itself rather than a runner/calculator).

        Returns:
            A list of error messages. Empty when everything is valid. When
            ``strict`` is False, unknown-key errors are *not* included so the
            caller can emit warnings instead.
        """
        errors: list[str] = []
        known_extra = known_extra or set()
        loc = f" in {context}" if context else ""
        for key, value in values.items():
            spec = self.resolve(key)
            if spec is None:
                if key.lower() in {k.lower() for k in known_extra}:
                    continue
                if strict:
                    suggestion = self.suggest(key)
                    hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
                    errors.append(f"Unknown key {key!r}{loc}.{hint}")
                continue
            # Coerce + validate the value.
            try:
                coerced = spec.coerce(value)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            errors.extend(spec.validate_value(coerced))
        return errors
