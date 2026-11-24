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
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


@dataclass(frozen=True)
class OptionSpec:
    """Specification of a single recognised option.

    Besides validation, this record is the single source of the *presentation*
    facts about a key:

    * ``description`` is the one-line English explanation used as the INCAR
      template comment and as the "meaning" column of the generated key
      reference (``docs/incar.md``);
    * ``group`` is the section a key belongs to; ``order`` is its position
      inside that group (assigned from the declaration order in ``_SPECS``, so
      one table fixes both the template and the documentation order);
    * ``template`` marks the keys a generated INCAR template shows. Keys
      without it are INCAR-writable but not part of the template (for example
      ``TORCH_NUM_THREADS``);
    * ``applies_when`` is the "applies only when" condition of a key, as
      ``(key, value)`` pairs that must all hold, e.g. the MTK barostat keys
      only apply when ``ENSEMBLE = NPT``;
    * ``example_when_inactive`` keeps a conditional key in the template as a
      commented example instead of dropping it (the NPT block of the MD
      template);
    * ``placeholder`` is the value a template writes when the user must supply
      one (``MODEL_PATH = REQUIRED``).

    Tracking: #61 R3.1 / R3.4. Templates, ``mliport config schema`` and the
    generated key reference all read these fields; there is no second list.
    """

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
    group: str = ""
    order: int = 0
    template: bool = False
    template_scopes: tuple[str, ...] = ()
    applies_when: tuple[tuple[str, str], ...] = ()
    example_when_inactive: bool = False
    placeholder: str = ""
    template_value: Any = None
    example_note: str = ""
    example_condition_line: bool = True
    condition_calc_types: tuple[str, ...] = ()

    @property
    def incar_key(self) -> str:
        """The UPPER_CASE spelling a template writes for this option.

        The canonical spelling is the option name in upper case when it is a
        declared alias; otherwise the longest declared upper-case alias (which
        is the most descriptive one, e.g. ``NEB_IMAGES`` over ``IMAGES``).
        """
        upper = sorted(alias for alias in self.aliases if alias.isupper())
        if self.name.upper() in upper:
            return self.name.upper()
        if upper:
            return max(upper, key=len)
        return self.name.upper()

    def applies_to(
        self, values: Mapping[str, Any], calc_type: str | None = None
    ) -> bool:
        """True when ``applies_when`` holds for the given option values.

        Conditions are compared case-insensitively against the *string form* of
        the value, so ``ENSEMBLE = NPT`` matches ``"npt"``. ``condition_calc_types``
        scopes a condition to the calculation types it belongs to: the Langevin
        friction key is thermostat-conditional for MD, where a thermostat is
        chosen, and unconditional for the configuration search, which always
        runs Langevin.
        """
        if (
            calc_type is not None
            and self.condition_calc_types
            and calc_type not in self.condition_calc_types
        ):
            return True
        for key, expected in self.applies_when:
            actual = values.get(key)
            if actual is None:
                return False
            if str(actual).strip().casefold() != str(expected).strip().casefold():
                return False
        return True

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
        if (
            self.minimum is not None
            and isinstance(value, (int, float))
            and value < self.minimum
        ):
            errors.append(
                f"Option '{self.name}'={value} is below minimum {self.minimum}"
            )
        if (
            self.maximum is not None
            and isinstance(value, (int, float))
            and value > self.maximum
        ):
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
        frozenset({"safety", "confsearch"}),
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
        choices=("uma", "fairchem", "mace", "dpa", "grace", "nep"),
        description="MLIP engine.",
    ),
    OptionSpec(
        "model_path",
        str,
        frozenset({"calculator"}),
        aliases={"MODEL_PATH", "model"},
        description=(
            "Model checkpoint file or directory. A value without any path "
            "separator is a file name: it is resolved next to the declaring "
            "file first, then in <MLIPORT_ROOT>/models (one level of "
            "subdirectories allowed)."
        ),
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
    # --- DFT-D3 dispersion correction (every backend) ---
    OptionSpec(
        "d3",
        bool,
        frozenset({"calculator"}),
        aliases={"D3"},
        description=(
            "Add Grimme DFT-D3 dispersion (torch-dftd) on top of the model. "
            "Only for models trained on dispersion-free reference data."
        ),
    ),
    OptionSpec(
        "d3_xc",
        str,
        frozenset({"calculator"}),
        aliases={"D3_XC"},
        description=(
            "Reference functional of the model's training data, selecting the "
            "D3 parameters (default: pbe)."
        ),
    ),
    OptionSpec(
        "d3_damping",
        str,
        frozenset({"calculator"}),
        aliases={"D3_DAMPING"},
        choices=("zero", "bj", "zerom", "bjm"),
        description="DFT-D3 damping function (default: bj, i.e. D3(BJ)).",
    ),
    OptionSpec(
        "d3_cutoff",
        float,
        frozenset({"calculator"}),
        aliases={"D3_CUTOFF"},
        minimum=1.0e-12,
        description=(
            "DFT-D3 pair cutoff in Angstrom (default: torch-dftd's 95 bohr, "
            "about 50.3 A). Smaller is faster but changes energies slightly."
        ),
    ),
    # --- NEP dual-runtime calculator options (PR #61 section C) ---
    # ``nep_runtime`` is scoped to ``calculator.nep`` so the resolver rejects it
    # for every other engine instead of recording a switch that never executes.
    # D3 is deliberately absent from the NEP option set: the factory rejects a
    # NEP + D3 combination rather than silently double-counting dispersion.
    # DPA has two *environments*: the pinned DPA-4-capable runtime and a
    # pre-release DPA-2 runtime for legacy checkpoints such as DPA-SSE
    # (#61 DPA-SSE spec §5). The option names the environment the user is
    # running in; mliport never switches between them by itself.
    OptionSpec(
        "dpa_runtime",
        str,
        frozenset({"calculator.dpa"}),
        aliases={"DPA_RUNTIME"},
        choices=("current", "legacy"),
        description=(
            "DPA runtime environment: 'current' (pinned DPA-4-capable "
            "deepmd-kit) or 'legacy' (.venv-dpa-legacy, deepmd-kit 3.0.0b3, "
            "for DPA-2 checkpoints such as DPA-SSE). No automatic fallback."
        ),
    ),
    OptionSpec(
        "nep_runtime",
        str,
        frozenset({"calculator.nep"}),
        aliases={"NEP_RUNTIME"},
        choices=("ase", "gpumd"),
        description=(
            "NEP execution runtime: 'ase' (Calorine CPUNEP/GPUNEP calculator) "
            "or 'gpumd' (native gpumd process). Required for MODEL_TYPE=nep; "
            "mliport never infers it or falls back between the two."
        ),
    ),
    OptionSpec(
        "nep_gpumd_command",
        str,
        frozenset({"calculator.nep"}),
        aliases={"NEP_GPUMD_COMMAND", "gpumd_executable"},
        description=(
            "Path to the gpumd executable for NEP_RUNTIME=gpumd (and for the "
            "GPUNEP ASE adapter). Launched as an argv list, never through a "
            "shell; its real path, version and SHA256 are recorded."
        ),
        is_path=True,
    ),
    # --- molecular electronic state (applied to atoms.info by every runner) ---
    OptionSpec(
        "charge",
        int,
        frozenset({"sp", "opt", "md", "neb", "batch", "confsearch"}),
        aliases={"CHARGE"},
        minimum=-100,
        maximum=100,
        description="Total molecular charge stored in atoms.info.",
    ),
    OptionSpec(
        "spin",
        int,
        frozenset({"sp", "opt", "md", "neb", "batch", "confsearch"}),
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
        frozenset({"opt", "batch"}),
        aliases={"OPT_ALGO", "opt_algo"},
        choices=("FIRE", "BFGS", "LBFGS"),
        description="Geometry optimization algorithm.",
    ),
    OptionSpec(
        "fmax",
        float,
        frozenset({"opt", "neb", "batch"}),
        aliases={"FMAX"},
        minimum=0.0,
        description="Force convergence threshold in eV/Angstrom.",
    ),
    OptionSpec(
        "max_steps",
        int,
        frozenset({"opt", "neb", "batch"}),
        aliases={"MAX_STEPS"},
        minimum=0,
        description="Maximum optimization steps.",
    ),
    OptionSpec(
        "cell_opt",
        bool,
        frozenset({"opt", "batch"}),
        aliases={"CELL_OPT"},
        description="Optimize cell parameters (requires stress support).",
    ),
    OptionSpec(
        "fix_symmetry",
        bool,
        frozenset({"opt", "batch"}),
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
        "ensemble",
        str,
        frozenset({"md"}),
        aliases={"MD_ENSEMBLE"},
        choices=("NVT", "NVE", "NPT"),
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
        "pressure_gpa",
        float,
        frozenset({"md"}),
        aliases={"PRESSURE_GPA", "PRESSURE"},
        description="NPT hydrostatic target pressure in GPa (positive is compressive).",
    ),
    OptionSpec(
        "npt_tdamp",
        float,
        frozenset({"md"}),
        aliases={"NPT_TDAMP"},
        minimum=0.0,
        description="MTK thermostat damping time in fs (strictly positive).",
    ),
    OptionSpec(
        "npt_pdamp",
        float,
        frozenset({"md"}),
        aliases={"NPT_PDAMP"},
        minimum=0.0,
        description="MTK barostat damping time in fs (strictly positive).",
    ),
    OptionSpec(
        "npt_tchain",
        int,
        frozenset({"md"}),
        aliases={"NPT_TCHAIN"},
        minimum=1,
        description=(
            "MTK NPT only: number of Nose-Hoover chain variables in the "
            "thermostat; >= 1, default 3"
        ),
    ),
    OptionSpec(
        "npt_pchain",
        int,
        frozenset({"md"}),
        aliases={"NPT_PCHAIN"},
        minimum=1,
        description=(
            "MTK NPT only: number of chain variables in the barostat; >= 1, "
            "default 3"
        ),
    ),
    OptionSpec(
        "npt_tloop",
        int,
        frozenset({"md"}),
        aliases={"NPT_TLOOP"},
        minimum=1,
        description=(
            "MTK NPT only: number of Suzuki-Yoshida sub-steps per thermostat "
            "integration; >= 1, default 1"
        ),
    ),
    OptionSpec(
        "npt_ploop",
        int,
        frozenset({"md"}),
        aliases={"NPT_PLOOP"},
        minimum=1,
        description=(
            "MTK NPT only: number of Suzuki-Yoshida sub-steps per barostat "
            "integration; >= 1, default 1"
        ),
    ),
    OptionSpec(
        "timestep",
        float,
        frozenset({"md", "confsearch"}),
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
        frozenset({"md", "confsearch"}),
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
        "raw_forces_storage",
        str,
        frozenset({"md"}),
        aliases={"RAW_FORCES_STORAGE"},
        choices=("json", "binary"),
        default="json",
        description=(
            "How raw (unprojected) MD forces are stored in raw/trajectory.traj: "
            "'json' (info field, default) or 'binary' (exact float64 ULM field, "
            "about 35% smaller; read with mliport.trajectory_forces)."
        ),
    ),
    OptionSpec(
        "seed",
        int,
        frozenset({"md", "general", "confsearch"}),
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
        choices=("sp", "opt", "md", "neb", "batch", "confsearch"),
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
        "output_dir",
        str,
        frozenset({"output"}),
        aliases={"OUTPUT_DIR"},
        description=(
            "Output directory for the calculation. Used when the command line "
            "does not pass --output; the resolved config is written into the "
            "final job directory. Relative values resolve against the file that "
            "declares them (the INCAR folder for an INCAR), and `mliport run` "
            "defaults to <INCAR folder>/results when neither --output nor "
            "OUTPUT_DIR is given."
        ),
        is_path=True,
    ),
    OptionSpec(
        "job_name",
        str,
        frozenset({"meta"}),
        aliases={"JOB_NAME"},
        description="Optional job name.",
    ),
]


# ---------------------------------------------------------------------------
# confsearch run options (spec section 12.3).  The scientific defaults live in
# ``mliport.confsearch.options.CS_DEFAULTS`` and are mirrored into
# BUILTIN_DEFAULTS["confsearch"]; the bounds here are the fail-closed schema
# bounds, while the cross-key rules live in ConfsearchOptions.
# ---------------------------------------------------------------------------


#: One-line meanings of the confsearch keys, used as the template comment
#: and as the "meaning" column of the generated key reference (#61 R4).
_CONFSEARCH_DESCRIPTIONS: dict[str, str] = {
    "cs_mode": (
        "Search mode; v1 accepts only 'adsorbate' (bulk_mobile, melt_quench, interface, multi_adsorbate are rejected)"
    ),
    "cs_starts": (
        "Mode A: start structures (files or directories), one independent replica each; needs cs_adsorbate_indices"
    ),
    "cs_adsorbate_indices": (
        "Mode A: 0-based adsorbate atom indices such as 300-341 or 0,5,7-9; must be one connected covalent molecule"
    ),
    "cs_slab": (
        "Mode B: slab structure file, exactly one frame; conformers from cs_molecule are placed above it"
    ),
    "cs_molecule": (
        "Mode B: molecule conformer file; one conformer is drawn at random for each placed start"
    ),
    "cs_n_starts": (
        "Mode B: number of placed start structures to build, each becoming one independent replica; minimum 1"
    ),
    "cs_fix_below": (
        "Extend the input FixAtoms to slab atoms within this height in Angstrom of the lowest slab atom; unset = off"
    ),
    "cs_md_mobile": (
        "Which atoms are mobile in MD: adsorbate (default), adsorbate_and_top or all_free (every atom not fixed)"
    ),
    "cs_top_depth": (
        "Depth in Angstrom below the slab top of the extra mobile atoms in cs_md_mobile=adsorbate_and_top; default 2.0"
    ),
    "cs_temperature_high": (
        "Hold temperature in K of every heat-hold-cool cycle and of the initial velocities; default 500"
    ),
    "cs_temperature_low": (
        "End temperature in K of the linear cooling ramp in each cycle, must be <= cs_temperature_high; default 300"
    ),
    "cs_hold_ps": (
        "Duration in ps of the hold phase at cs_temperature_high in each cycle; must be a whole number of timesteps"
    ),
    "cs_cool_ps": (
        "Duration in ps of the linear cooling ramp from cs_temperature_high to cs_temperature_low in each cycle"
    ),
    "cs_snapshot_interval_ps": (
        "Spacing in ps between hold-phase snapshots that get quenched; must be < cs_hold_ps, 0 disables snapshots"
    ),
    "cs_ramp_interval": (
        "Number of MD steps between thermostat set-point updates during the linear cooling ramp; default 10"
    ),
    "cs_cycles": (
        "Number of heat-hold-cool cycles run per replica after its start quench; 0 leaves only the start quenches"
    ),
    "cs_save_interval": (
        "Write one raw MD frame every this many MD steps (default 100); quench snapshots and cycle ends are always kept"
    ),
    "cs_accept_temperature": (
        "Temperature in K of the Metropolis test that accepts or rejects a replica's end-of-cycle minimum; default 500"
    ),
    "cs_quench_fmax": (
        "FIRE force convergence threshold in eV/A for every snapshot and cycle-end quench; default 0.05"
    ),
    "cs_quench_steps": (
        "Maximum FIRE steps per quench; exceeding it marks the minimum quench_not_converged and unranked; default 2000"
    ),
    "cs_bond_scale": (
        "Bond test in the covalent graphs: bonded when closer than this multiple of the covalent-radii sum; default 1.2"
    ),
    "cs_bond_break_scale": (
        "MD aborts if an adsorbate bond exceeds this multiple of the covalent-radii sum; must be > cs_bond_scale (1.5)"
    ),
    "cs_ionic_species": (
        "Element symbols excluded from every covalent graph (ionic hopping is no reaction); default Li,Na,K,Rb,Cs,Mg,Ca"
    ),
    "cs_reaction_policy": (
        "Whether minima whose covalent bonds changed are ranked and exported: exclude (default) or include"
    ),
    "cs_desorption_distance": (
        "Adsorbate-slab distance in Angstrom above which a minimum is labelled desorbed and dropped; default 5.0"
    ),
    "cs_min_image_distance": (
        "Adsorbate in-plane self-image distance in Angstrom below which a minimum is image_contact; default 4.0"
    ),
    "cs_min_vacuum_gap": (
        "Gap in Angstrom from the adsorbate to the slab image one cell along c; smaller marks image_contact (8.0)"
    ),
    "cs_place_contact_scale": (
        "Mode B: target ratio of the closest adsorbate-slab pair to the covalent-radii sum (unbonded); default 1.5"
    ),
    "cs_place_anchor": (
        "Mode B: 0-based anchor atom indices in the molecule file; the lowest atom is drawn from them and placed bonded"
    ),
    "cs_place_anchor_scale": (
        "Mode B with cs_place_anchor: anchored pair ratio to the covalent-radii sum, i.e. already bonded; default 1.1"
    ),
    "cs_place_other_min_scale": (
        "Mode B with cs_place_anchor: lower bound for every other adsorbate-slab pair in radii-sum units; default 1.3"
    ),
    "cs_place_anchor_targets": (
        "Mode B with cs_place_anchor: slab elements the closest slab atom must belong to (e.g. an ionic species)"
    ),
    "cs_place_max_attempts": (
        "Mode B: random placement attempts tried per start before the run fails; default 1000"
    ),
    "cs_dedup_energy_tol": (
        "Energy window in eV within which two minima may be merged as the same minimum; default 0.010"
    ),
    "cs_dedup_geom_tol": (
        "Per-atom displacement in Angstrom below which two minima count as the same geometry (rule G); default 0.20"
    ),
    "cs_dedup_fp_tol": (
        "Fingerprint distance below which two minima merge (rule D, catches symmetry equivalents); default 0.0004"
    ),
    "cs_fp_rcut": (
        "Cutoff radius in Angstrom of the adsorbate Oganov-Valle fingerprint used for dedup; default 8.0"
    ),
    "cs_fp_binwidth": (
        "Fingerprint histogram bin width in Angstrom, at most half of cs_fp_sigma; default 0.02"
    ),
    "cs_fp_sigma": (
        "Gaussian smearing width in Angstrom applied to the fingerprint histogram bins; default 0.20"
    ),
    "cs_top_k": (
        "Maximum number of ranked minima exported as candidates for DFT re-ranking; default 10"
    ),
    "cs_export_window": (
        "Only minima within this energy window in eV above the best one are refined and exported; default 0.5"
    ),
    "cs_refine": (
        "Re-refine the exported candidates with a tighter force threshold before ranking and export; default true"
    ),
    "cs_refine_fmax": (
        "Force threshold in eV/A for the final refinement, must be <= cs_quench_fmax; default 0.02"
    ),
    "cs_refine_steps": (
        "Maximum number of FIRE steps in the final refinement of each exported minimum; default 3000"
    ),
    "cs_allow_direct_forces": (
        "Allow a model that reports direct_forces=true, in which case the ranking is not meaningful; default false"
    ),
}


def _confsearch_specs() -> list[OptionSpec]:
    scope = frozenset({"confsearch"})
    specifications: list[tuple] = [
        ("cs_mode", str, {"CS_MODE"}, ("adsorbate",), None, None, "adsorbate"),
        ("cs_starts", str, {"CS_STARTS"}, None, None, None, None),
        ("cs_adsorbate_indices", str, {"CS_ADSORBATE_INDICES"}, None, None, None, None),
        ("cs_slab", str, {"CS_SLAB"}, None, None, None, None),
        ("cs_molecule", str, {"CS_MOLECULE"}, None, None, None, None),
        ("cs_n_starts", int, {"CS_N_STARTS"}, None, 1, None, None),
        ("cs_fix_below", float, {"CS_FIX_BELOW"}, None, None, None, None),
        (
            "cs_md_mobile",
            str,
            {"CS_MD_MOBILE"},
            ("adsorbate", "adsorbate_and_top", "all_free"),
            None,
            None,
            "adsorbate",
        ),
        ("cs_top_depth", float, {"CS_TOP_DEPTH"}, None, None, None, 2.0),
        (
            "cs_temperature_high",
            float,
            {"CS_TEMPERATURE_HIGH"},
            None,
            None,
            None,
            500.0,
        ),
        ("cs_temperature_low", float, {"CS_TEMPERATURE_LOW"}, None, None, None, 300.0),
        ("cs_hold_ps", float, {"CS_HOLD_PS"}, None, None, None, 2.0),
        ("cs_cool_ps", float, {"CS_COOL_PS"}, None, None, None, 2.0),
        (
            "cs_snapshot_interval_ps",
            float,
            {"CS_SNAPSHOT_INTERVAL_PS"},
            None,
            None,
            None,
            0.5,
        ),
        ("cs_ramp_interval", int, {"CS_RAMP_INTERVAL"}, None, 1, None, 10),
        ("cs_cycles", int, {"CS_CYCLES"}, None, 0, None, 5),
        ("cs_save_interval", int, {"CS_SAVE_INTERVAL"}, None, 1, None, 100),
        (
            "cs_accept_temperature",
            float,
            {"CS_ACCEPT_TEMPERATURE"},
            None,
            None,
            None,
            500.0,
        ),
        ("cs_quench_fmax", float, {"CS_QUENCH_FMAX"}, None, None, None, 0.05),
        ("cs_quench_steps", int, {"CS_QUENCH_STEPS"}, None, 1, None, 2000),
        ("cs_bond_scale", float, {"CS_BOND_SCALE"}, None, None, None, 1.2),
        ("cs_bond_break_scale", float, {"CS_BOND_BREAK_SCALE"}, None, None, None, 1.5),
        (
            "cs_ionic_species",
            str,
            {"CS_IONIC_SPECIES"},
            None,
            None,
            None,
            "Li,Na,K,Rb,Cs,Mg,Ca",
        ),
        (
            "cs_reaction_policy",
            str,
            {"CS_REACTION_POLICY"},
            ("exclude", "include"),
            None,
            None,
            "exclude",
        ),
        (
            "cs_desorption_distance",
            float,
            {"CS_DESORPTION_DISTANCE"},
            None,
            None,
            None,
            5.0,
        ),
        (
            "cs_min_image_distance",
            float,
            {"CS_MIN_IMAGE_DISTANCE"},
            None,
            None,
            None,
            4.0,
        ),
        ("cs_min_vacuum_gap", float, {"CS_MIN_VACUUM_GAP"}, None, None, None, 8.0),
        (
            "cs_place_contact_scale",
            float,
            {"CS_PLACE_CONTACT_SCALE"},
            None,
            None,
            None,
            1.5,
        ),
        ("cs_place_anchor", str, {"CS_PLACE_ANCHOR"}, None, None, None, None),
        (
            "cs_place_anchor_scale",
            float,
            {"CS_PLACE_ANCHOR_SCALE"},
            None,
            None,
            None,
            1.1,
        ),
        (
            "cs_place_other_min_scale",
            float,
            {"CS_PLACE_OTHER_MIN_SCALE"},
            None,
            None,
            None,
            1.3,
        ),
        (
            "cs_place_anchor_targets",
            str,
            {"CS_PLACE_ANCHOR_TARGETS"},
            None,
            None,
            None,
            None,
        ),
        ("cs_place_max_attempts", int, {"CS_PLACE_MAX_ATTEMPTS"}, None, 1, None, 1000),
        (
            "cs_dedup_energy_tol",
            float,
            {"CS_DEDUP_ENERGY_TOL"},
            None,
            None,
            None,
            0.010,
        ),
        ("cs_dedup_geom_tol", float, {"CS_DEDUP_GEOM_TOL"}, None, None, None, 0.20),
        ("cs_dedup_fp_tol", float, {"CS_DEDUP_FP_TOL"}, None, None, None, 0.0004),
        ("cs_fp_rcut", float, {"CS_FP_RCUT"}, None, None, None, 8.0),
        ("cs_fp_binwidth", float, {"CS_FP_BINWIDTH"}, None, None, None, 0.02),
        ("cs_fp_sigma", float, {"CS_FP_SIGMA"}, None, None, None, 0.20),
        ("cs_top_k", int, {"CS_TOP_K"}, None, 1, None, 10),
        ("cs_export_window", float, {"CS_EXPORT_WINDOW"}, None, None, None, 0.5),
        ("cs_refine", bool, {"CS_REFINE"}, None, None, None, True),
        ("cs_refine_fmax", float, {"CS_REFINE_FMAX"}, None, None, None, 0.02),
        ("cs_refine_steps", int, {"CS_REFINE_STEPS"}, None, 1, None, 3000),
        (
            "cs_allow_direct_forces",
            bool,
            {"CS_ALLOW_DIRECT_FORCES"},
            None,
            None,
            None,
            False,
        ),
    ]
    specs: list[OptionSpec] = []
    for name, kind, aliases, choices, minimum, maximum, default in specifications:
        specs.append(
            OptionSpec(
                name,
                kind,
                scope,
                aliases=aliases,
                choices=choices,
                minimum=minimum,
                maximum=maximum,
                default=default,
                description=_CONFSEARCH_DESCRIPTIONS[name],
            )
        )
    return specs


_SPECS.extend(_confsearch_specs())


# ---------------------------------------------------------------------------
# Presentation metadata (#61 R3.1 / R3.4).
#
# One table, applied to the specs above, fixes three things at once:
#
# * which section a key belongs to and in which order it is listed, both in a
#   generated INCAR template and in the generated key reference
#   (``docs/incar.md``),
# * which keys a generated template actually shows (``template``),
# * the "applies only when" conditions (NPT keys, thermostat-specific keys) and
#   the placeholder a template writes for a value the user must supply.
#
# ``_SPECS`` above stays the single list of recognised keys; this table only
# annotates those keys and is checked against them (see
# ``tests/mliport/mliport/config/test_schema.py``).
# ---------------------------------------------------------------------------

#: Section order of every generated template and of the key reference.
TEMPLATE_GROUP_ORDER: tuple[str, ...] = (
    "Calculation",
    "Model",
    "Device and resources",
    "Optimization",
    "Molecular dynamics",
    "NEB",
    "Configuration search",
    "Output",
    "Batch",
    "General",
)

#: canonical key -> section. Every key that is user-facing has exactly one.
_TEMPLATE_GROUPS: dict[str, tuple[str, ...]] = {
    "Calculation": ("calc_type",),
    "Model": (
        "model_type",
        "model_path",
        "task",
        "head",
        "inference_mode",
        "default_dtype",
        "activation_checkpointing",
        "dpa_runtime",
        "nep_runtime",
        "nep_gpumd_command",
        "charge",
        "spin",
        "d3",
        "d3_xc",
        "d3_damping",
        "d3_cutoff",
        "gpu_memory_growth",
        "gpu_memory_limit_mb",
        "neighbor_cache",
        "neighbor_skin",
    ),
    "Device and resources": ("device", "torch_num_threads"),
    "Optimization": ("optimizer", "fmax", "max_steps", "cell_opt", "fix_symmetry"),
    "Molecular dynamics": (
        "ensemble",
        "temperature",
        "timestep",
        "steps",
        "equilibration_steps",
        "thermostat",
        "friction",
        "bussi_tau",
        "nhc_tdamp",
        "nhc_tchain",
        "nhc_tloop",
        "pressure_gpa",
        "npt_tdamp",
        "npt_pdamp",
        "npt_tchain",
        "npt_pchain",
        "npt_tloop",
        "npt_ploop",
        "save_interval",
        "com_policy",
        "raw_forces_storage",
        "velocity_policy",
        "pre_relax",
        "pre_relax_steps",
        "pre_relax_fmax",
        "pre_relax_mode",
        "seed",
        "fmax_abort",
    ),
    "NEB": (
        "neb_initial",
        "neb_final",
        "n_intermediate_images",
        "climb",
        "neb_method",
        "neb_interpolation",
        "path_convention",
        "neb_spring",
        "neb_pre_fmax",
        "neb_pre_max_steps",
        "neb_maxstep",
        "endpoint_policy",
        "endpoint_fmax",
        "endpoint_steps",
        "idpp_fmax",
        "idpp_steps",
        "idpp_mic",
        "neb_min_distance",
        "checkpoint_interval",
    ),
    "Configuration search": tuple(
        sorted(
            (
                "cs_mode",
                "cs_starts",
                "cs_adsorbate_indices",
                "cs_slab",
                "cs_molecule",
                "cs_n_starts",
                "cs_fix_below",
                "cs_md_mobile",
                "cs_top_depth",
                "cs_temperature_high",
                "cs_temperature_low",
                "cs_hold_ps",
                "cs_cool_ps",
                "cs_snapshot_interval_ps",
                "cs_ramp_interval",
                "cs_cycles",
                "cs_save_interval",
                "cs_accept_temperature",
                "cs_quench_fmax",
                "cs_quench_steps",
                "cs_bond_scale",
                "cs_bond_break_scale",
                "cs_ionic_species",
                "cs_reaction_policy",
                "cs_desorption_distance",
                "cs_min_image_distance",
                "cs_min_vacuum_gap",
                "cs_place_contact_scale",
                "cs_place_anchor",
                "cs_place_anchor_scale",
                "cs_place_other_min_scale",
                "cs_place_anchor_targets",
                "cs_place_max_attempts",
                "cs_dedup_energy_tol",
                "cs_dedup_geom_tol",
                "cs_dedup_fp_tol",
                "cs_fp_rcut",
                "cs_fp_binwidth",
                "cs_fp_sigma",
                "cs_top_k",
                "cs_export_window",
                "cs_refine",
                "cs_refine_fmax",
                "cs_refine_steps",
                "cs_allow_direct_forces",
            )
        )
    ),
    "Output": (
        "write_outcar",
        "write_forces",
        "write_stress",
        "write_xdatcar",
        "write_trajectory",
        "write_json",
        "output_format",
        "output_dir",
        "job_name",
        "write_resolved_config",
    ),
    "Batch": ("sub_calc_type", "pattern"),
    "General": ("strict_config",),
}

#: Keys a generated INCAR template shows, grouped by the calc type they are
#: rendered for. ``None`` means "every calc type".
_TEMPLATE_KEYS: dict[str, tuple[str, ...] | None] = {
    "calc_type": None,
    "device": None,
    "optimizer": ("opt",),
    "fmax": ("opt", "neb"),
    "max_steps": ("opt", "neb"),
    "cell_opt": ("opt",),
    "fix_symmetry": ("opt",),
    "ensemble": ("md",),
    "temperature": ("md",),
    "timestep": ("md", "confsearch"),
    "equilibration_steps": ("md",),
    "steps": ("md",),
    "thermostat": ("md",),
    "friction": ("md", "confsearch"),
    "bussi_tau": ("md",),
    "nhc_tdamp": ("md",),
    "nhc_tchain": ("md",),
    "nhc_tloop": ("md",),
    "pressure_gpa": ("md",),
    "npt_tdamp": ("md",),
    "npt_pdamp": ("md",),
    "npt_tchain": ("md",),
    "npt_pchain": ("md",),
    "npt_tloop": ("md",),
    "npt_ploop": ("md",),
    "save_interval": ("md",),
    "com_policy": ("md",),
    "raw_forces_storage": ("md",),
    "neb_initial": ("neb",),
    "neb_final": ("neb",),
    "n_intermediate_images": ("neb",),
    "climb": ("neb",),
    "neb_method": ("neb",),
    "neb_interpolation": ("neb",),
    "path_convention": ("neb",),
    "neb_spring": ("neb",),
    "neb_pre_fmax": ("neb",),
    "neb_pre_max_steps": ("neb",),
    "neb_maxstep": ("neb",),
    "endpoint_policy": ("neb",),
    "endpoint_fmax": ("neb",),
    "endpoint_steps": ("neb",),
    "idpp_fmax": ("neb",),
    "idpp_steps": ("neb",),
    "idpp_mic": ("neb",),
    "neb_min_distance": ("neb",),
    "checkpoint_interval": ("neb",),
    "cs_starts": ("confsearch",),
    "cs_adsorbate_indices": ("confsearch",),
    "cs_slab": ("confsearch",),
    "cs_molecule": ("confsearch",),
    "cs_n_starts": ("confsearch",),
    "cs_mode": ("confsearch",),
    "cs_md_mobile": ("confsearch",),
    "cs_cycles": ("confsearch",),
    "cs_temperature_high": ("confsearch",),
    "cs_temperature_low": ("confsearch",),
    "cs_hold_ps": ("confsearch",),
    "cs_cool_ps": ("confsearch",),
    "cs_snapshot_interval_ps": ("confsearch",),
    "cs_ramp_interval": ("confsearch",),
    "cs_save_interval": ("confsearch",),
    "cs_accept_temperature": ("confsearch",),
    "cs_quench_fmax": ("confsearch",),
    "cs_quench_steps": ("confsearch",),
    "cs_bond_scale": ("confsearch",),
    "cs_bond_break_scale": ("confsearch",),
    "cs_ionic_species": ("confsearch",),
    "cs_reaction_policy": ("confsearch",),
    "cs_desorption_distance": ("confsearch",),
    "cs_min_image_distance": ("confsearch",),
    "cs_min_vacuum_gap": ("confsearch",),
    "cs_place_contact_scale": ("confsearch",),
    "cs_place_anchor_scale": ("confsearch",),
    "cs_place_other_min_scale": ("confsearch",),
    "cs_place_max_attempts": ("confsearch",),
    "cs_dedup_energy_tol": ("confsearch",),
    "cs_dedup_geom_tol": ("confsearch",),
    "cs_dedup_fp_tol": ("confsearch",),
    "cs_fp_rcut": ("confsearch",),
    "cs_fp_binwidth": ("confsearch",),
    "cs_fp_sigma": ("confsearch",),
    "cs_top_k": ("confsearch",),
    "cs_export_window": ("confsearch",),
    "cs_refine": ("confsearch",),
    "cs_refine_fmax": ("confsearch",),
    "cs_refine_steps": ("confsearch",),
    "cs_allow_direct_forces": ("confsearch",),
    # NEB refuses these outputs, so its template does not offer them.
    "write_forces": ("sp", "opt", "md", "confsearch"),
    "write_stress": ("sp", "opt", "md", "confsearch"),
    "output_format": ("sp", "opt", "md", "confsearch"),
}

#: "Applies only when" conditions and the placeholders users must fill in.
_TEMPLATE_CONDITIONS: dict[str, dict[str, Any]] = {
    # MTK barostat: the NPT example block of the MD template.
    "pressure_gpa": {
        "applies_when": (("ensemble", "NPT"),),
        "example_when_inactive": True,
        "example_note": "To run NPT instead, uncomment this block:",
    },
    "npt_tdamp": {
        "applies_when": (("ensemble", "NPT"),),
        "example_when_inactive": True,
    },
    "npt_pdamp": {
        "applies_when": (("ensemble", "NPT"),),
        "example_when_inactive": True,
    },
    "npt_tchain": {
        "applies_when": (("ensemble", "NPT"),),
        "example_when_inactive": True,
    },
    "npt_pchain": {
        "applies_when": (("ensemble", "NPT"),),
        "example_when_inactive": True,
    },
    "npt_tloop": {
        "applies_when": (("ensemble", "NPT"),),
        "example_when_inactive": True,
    },
    "npt_ploop": {
        "applies_when": (("ensemble", "NPT"),),
        "example_when_inactive": True,
    },
    # Displayed template values for keys whose runtime default lives in the
    # writer/engine rather than in BUILTIN_DEFAULTS.
    "device": {"template_value": "cpu"},
    "write_forces": {"template_value": True},
    "write_stress": {"template_value": True},
    "output_format": {"template_value": "VASP"},
    "neb_initial": {"placeholder": "REQUIRED"},
    "neb_final": {"placeholder": "REQUIRED"},
    # Thermostat-specific keys: only the selected thermostat is written.
    "friction": {
        "applies_when": (("thermostat", "LANGEVIN"),),
        "condition_calc_types": ("md",),
    },
    "bussi_tau": {"applies_when": (("thermostat", "BUSSI"),)},
    "nhc_tdamp": {"applies_when": (("thermostat", "NHC"),)},
    "nhc_tchain": {"applies_when": (("thermostat", "NHC"),)},
    "nhc_tloop": {"applies_when": (("thermostat", "NHC"),)},
    # Values the user must supply: the template cannot invent a model or the
    # structures to search.
    "model_path": {"placeholder": "REQUIRED"},
    "cs_starts": {
        "placeholder": "REQUIRED",
        "example_note": "Mode A: your own start structures and the adsorbate indices",
    },
    "cs_adsorbate_indices": {"placeholder": "REQUIRED"},
    # Mode B is the alternative start source: a commented example only.
    "cs_slab": {
        "applies_when": (("cs_mode", "MODE_B"),),
        "example_when_inactive": True,
        "example_condition_line": False,
        "example_note": "Mode B alternative: let mliport place the molecule on the slab",
        "placeholder": "slab.vasp",
    },
    "cs_molecule": {
        "applies_when": (("cs_mode", "MODE_B"),),
        "example_when_inactive": True,
        "example_condition_line": False,
        "placeholder": "molecule.xyz",
    },
    "cs_n_starts": {
        "applies_when": (("cs_mode", "MODE_B"),),
        "example_when_inactive": True,
        "example_condition_line": False,
        "template_value": 10,
    },
}

#: Template values that are not the built-in default (the user must supply a
#: real value, and a template must still be syntactically valid).
_TEMPLATE_VALUES: dict[str, Any] = {}


def _apply_presentation_metadata(specs: list[OptionSpec]) -> list[OptionSpec]:
    """Attach group/order/template/condition metadata to the declared specs.

    The order inside a group follows the declaration order in ``_SPECS`` (a
    stable, already-curated order), and every key of ``_TEMPLATE_GROUPS`` must
    resolve to a declared option - a typo is an error, not a silently missing
    documentation row.
    """
    known = {spec.name for spec in specs}
    unknown = sorted(
        key for keys in _TEMPLATE_GROUPS.values() for key in keys if key not in known
    )
    if unknown:
        raise RuntimeError(f"Unknown options in the template table: {unknown}")
    duplicates = sorted(
        key
        for key in {key for keys in _TEMPLATE_GROUPS.values() for key in keys}
        if sum(key in keys for keys in _TEMPLATE_GROUPS.values()) > 1
    )
    if duplicates:
        raise RuntimeError(f"Options in more than one template group: {duplicates}")

    group_of = {key: group for group, keys in _TEMPLATE_GROUPS.items() for key in keys}
    template_scopes = dict(_TEMPLATE_KEYS)
    unknown_template = sorted(key for key in template_scopes if key not in known)
    if unknown_template:
        raise RuntimeError(f"Unknown template keys: {unknown_template}")
    unknown_conditions = sorted(key for key in _TEMPLATE_CONDITIONS if key not in known)
    if unknown_conditions:
        raise RuntimeError(f"Unknown template conditions: {unknown_conditions}")

    annotated: list[OptionSpec] = []
    for index, spec in enumerate(specs):
        conditions = _TEMPLATE_CONDITIONS.get(spec.name, {})
        annotated.append(
            replace(
                spec,
                group=group_of.get(spec.name, ""),
                order=index,
                template=spec.name in template_scopes,
                template_scopes=template_scopes.get(spec.name, ()),
                applies_when=conditions.get("applies_when", ()),
                example_when_inactive=bool(
                    conditions.get("example_when_inactive", False)
                ),
                placeholder=str(conditions.get("placeholder", "")),
                template_value=conditions.get("template_value"),
                example_note=str(conditions.get("example_note", "")),
                example_condition_line=bool(
                    conditions.get("example_condition_line", True)
                ),
                condition_calc_types=tuple(conditions.get("condition_calc_types", ())),
            )
        )
    return annotated


def get_schema() -> Schema:
    """Return the singleton :class:`Schema` instance."""
    global _SCHEMA  # noqa: PLW0603
    if _SCHEMA is None:
        _SCHEMA = Schema(_apply_presentation_metadata(_SPECS))
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
