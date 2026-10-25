"""Typed, calculator-independent options for the confsearch workflow.

The dataclass mirrors :class:`mliport.neb.schema.NEBOptions`: it is frozen,
validates every cross-key rule at construction time and fails closed with a
message that names the nearest valid value.  Nothing here touches a calculator,
a file or an ASE ``Atoms`` object, so the whole core library is testable on
synthetic structures.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

from ase.data import atomic_numbers

from mliport.confsearch.errors import (
    ConfsearchError,
    reject_non_goal,
)
from mliport.confsearch.index_spec import parse_index_spec
from mliport.runners.safety import finite_float, strict_integer

if TYPE_CHECKING:
    pass

#: Default ionic species excluded from covalent graphs (spec section 8.1).
DEFAULT_IONIC_SPECIES = "Li,Na,K,Rb,Cs,Mg,Ca"

#: Mobile-set choices (spec section 7.2).
MD_MOBILE_CHOICES = ("adsorbate", "adsorbate_and_top", "all_free")

#: Reaction policies (spec section 8.2).
REACTION_POLICIES = ("exclude", "include")

#: Modes named in the spec as future work; asking for them is a hard error.
_FUTURE_MODES = ("bulk_mobile", "melt_quench", "interface", "multi_adsorbate")

#: confsearch-scoped defaults (spec section 7.6 / 12.3).
CS_DEFAULTS: dict[str, object] = {
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
    "cs_ionic_species": DEFAULT_IONIC_SPECIES,
    "cs_reaction_policy": "exclude",
    "cs_desorption_distance": 5.0,
    "cs_min_image_distance": 4.0,
    "cs_min_vacuum_gap": 8.0,
    "cs_place_contact_scale": 1.5,
    "cs_place_anchor_scale": 1.1,
    "cs_place_other_min_scale": 1.3,
    "cs_place_max_attempts": 1000,
    "cs_dedup_energy_tol": 0.010,
    # Direct geometric identity (spec 9.3 revision 4): with a fixed slab and a
    # fixed atom order, the same minimum re-quenched moves its adsorbate atoms
    # by less than this.  This is the primary "same basin" test; the fingerprint
    # tolerance below is only needed for what G cannot see (symmetry-equivalent
    # sites and permutation-equivalent conformers).
    "cs_dedup_geom_tol": 0.20,
    # P1 calibration (spec 13.2 revision 4), geometry-labelled pairs on the
    # repository's LGPS geometry; see
    # tests/mliport/mliport/confsearch/test_fingerprint_calibration.py:
    #   (a) same basin: 25 pairs within cs_dedup_geom_tol (handled by G);
    #       11 further pairs (0.20-0.5 A) are documented false splits;
    #   (b/c) 63 distinct sites/conformers: min D = 8.5e-4 at sigma=0.20;
    #   (d) 4 equivalent pairs (rigid slab symmetry + CH3 permutation):
    #       D ~ 0 and all merge.
    # Zero false merges with a factor-2 margin: 0.0004 <= 0.5 * 8.5e-4.
    #
    # Known limitation (PR #63 review, spec rev 5): the calibration geometry is
    # CH3OH inside the LGPS bulk cell, class (a) comes from random perturbations
    # rather than quenches, and class (d) uses exact rigid images (D ~ 0).  The
    # shipped tolerance is therefore proven to give zero false merges but is
    # untested for *noisy* equivalent minima: in practice D merges little beyond
    # exact images and the geometric rule G does most of the work.  That is the
    # safe direction (duplicates instead of deleted candidates), and a real-slab
    # / real-model recalibration is a P5 item in the validation campaign.
    # Changing this default later bumps CS_SCIENTIFIC_REVISION.
    "cs_dedup_fp_tol": 0.0004,
    "cs_fp_rcut": 8.0,
    "cs_fp_binwidth": 0.02,
    # P1 calibration (spec 13.2 revision 4): the sigma sweep {0.05, 0.10, 0.20}
    # merged all equivalent pairs at every sigma, so the tie-break (larger
    # sigma) selects 0.20.
    "cs_fp_sigma": 0.20,
    "cs_top_k": 10,
    "cs_export_window": 0.5,
    "cs_refine": True,
    "cs_refine_fmax": 0.02,
    "cs_refine_steps": 3000,
    "cs_allow_direct_forces": False,
}


def _parse_elements(spec: str, *, label: str) -> tuple[str, ...]:
    if not isinstance(spec, str):
        raise ConfsearchError(f"{label} must be a string of element symbols")
    symbols: list[str] = []
    for token in spec.split(","):
        item = token.strip()
        if not item:
            raise ConfsearchError(f"{label}={spec!r} has an empty element entry")
        if item in atomic_numbers:
            symbols.append(item)
            continue
        capitalized = item.capitalize()
        if capitalized in atomic_numbers:
            symbols.append(capitalized)
            continue
        raise ConfsearchError(
            f"{label}={spec!r}: {item!r} is not an element symbol "
            "(use ASE symbols such as Li, C, H)"
        )
    unique = tuple(dict.fromkeys(symbols))
    if not unique:
        raise ConfsearchError(f"{label} selects no elements")
    return unique


def _positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    number = finite_float(value, name=name)
    if number < 0 or (number == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        raise ConfsearchError(f"{name} must be {comparator}, got {number!r}")
    return number


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    number = strict_integer(value, name=name)
    if number < minimum:
        raise ConfsearchError(f"{name} must be >= {minimum}, got {number}")
    return number


def steps_from_ps(
    value_ps: float, timestep_fs: float, *, name: str, allow_zero: bool = False
) -> int:
    """Convert a duration in ps to whole timesteps, or fail closed.

    The specification requires the conversion to be an integer within 1e-9;
    otherwise the run is refused with the nearest valid durations.
    """
    ps = _positive(value_ps, name, allow_zero=allow_zero)
    steps_exact = ps * 1000.0 / timestep_fs
    steps = int(round(steps_exact))
    if abs(steps_exact - steps) > 1e-9:
        lower = max(0, math.floor(steps_exact))
        upper = math.ceil(steps_exact)
        lower_ps = lower * timestep_fs / 1000.0
        upper_ps = upper * timestep_fs / 1000.0
        raise ConfsearchError(
            f"{name}={ps} ps is not an integer number of {timestep_fs} fs steps "
            f"({steps_exact:.9g}); nearest valid values are {lower_ps:g} ps "
            f"({lower} steps) and {upper_ps:g} ps ({upper} steps)"
        )
    if not allow_zero and steps < 1:
        raise ConfsearchError(f"{name}={ps} ps is shorter than one timestep")
    return steps


@dataclass(frozen=True, slots=True)
class ConfsearchOptions:
    """Validated options for one confsearch run (v1, ``adsorbate`` mode)."""

    cs_mode: str = "adsorbate"

    # --- starts: mode A (user-prepared) ---
    cs_starts: tuple[str, ...] | None = None
    cs_adsorbate_indices: str | None = None

    # --- starts: mode B (automatic placement) ---
    cs_slab: str | None = None
    cs_molecule: str | None = None
    cs_n_starts: int | None = None

    cs_fix_below: float | None = None

    # --- MD ---
    cs_md_mobile: str = str(CS_DEFAULTS["cs_md_mobile"])
    cs_top_depth: float = float(CS_DEFAULTS["cs_top_depth"])  # type: ignore[arg-type]
    cs_temperature_high: float = float(CS_DEFAULTS["cs_temperature_high"])  # type: ignore[arg-type]
    cs_temperature_low: float = float(CS_DEFAULTS["cs_temperature_low"])  # type: ignore[arg-type]
    cs_hold_ps: float = float(CS_DEFAULTS["cs_hold_ps"])  # type: ignore[arg-type]
    cs_cool_ps: float = float(CS_DEFAULTS["cs_cool_ps"])  # type: ignore[arg-type]
    cs_snapshot_interval_ps: float = float(CS_DEFAULTS["cs_snapshot_interval_ps"])  # type: ignore[arg-type]
    cs_ramp_interval: int = int(CS_DEFAULTS["cs_ramp_interval"])  # type: ignore[arg-type]
    cs_cycles: int = int(CS_DEFAULTS["cs_cycles"])  # type: ignore[arg-type]
    cs_save_interval: int = int(CS_DEFAULTS["cs_save_interval"])  # type: ignore[arg-type]
    cs_accept_temperature: float = float(CS_DEFAULTS["cs_accept_temperature"])  # type: ignore[arg-type]
    timestep: float = float(CS_DEFAULTS["timestep"])  # type: ignore[arg-type]
    friction: float = float(CS_DEFAULTS["friction"])  # type: ignore[arg-type]
    seed: int | None = None
    fmax_abort: float = 20.0

    # --- quench ---
    cs_quench_fmax: float = float(CS_DEFAULTS["cs_quench_fmax"])  # type: ignore[arg-type]
    cs_quench_steps: int = int(CS_DEFAULTS["cs_quench_steps"])  # type: ignore[arg-type]

    # --- chemistry ---
    cs_bond_scale: float = float(CS_DEFAULTS["cs_bond_scale"])  # type: ignore[arg-type]
    cs_bond_break_scale: float = float(CS_DEFAULTS["cs_bond_break_scale"])  # type: ignore[arg-type]
    cs_ionic_species: str = DEFAULT_IONIC_SPECIES
    cs_reaction_policy: str = str(CS_DEFAULTS["cs_reaction_policy"])
    cs_desorption_distance: float = float(CS_DEFAULTS["cs_desorption_distance"])  # type: ignore[arg-type]
    cs_min_image_distance: float = float(CS_DEFAULTS["cs_min_image_distance"])  # type: ignore[arg-type]
    cs_min_vacuum_gap: float = float(CS_DEFAULTS["cs_min_vacuum_gap"])  # type: ignore[arg-type]

    # --- placement ---
    cs_place_contact_scale: float = float(CS_DEFAULTS["cs_place_contact_scale"])  # type: ignore[arg-type]
    cs_place_anchor: str | None = None
    cs_place_anchor_scale: float = float(CS_DEFAULTS["cs_place_anchor_scale"])  # type: ignore[arg-type]
    cs_place_other_min_scale: float = float(CS_DEFAULTS["cs_place_other_min_scale"])  # type: ignore[arg-type]
    cs_place_anchor_targets: str | None = None
    cs_place_max_attempts: int = int(CS_DEFAULTS["cs_place_max_attempts"])  # type: ignore[arg-type]

    # --- dedup / fingerprint ---
    cs_dedup_energy_tol: float = float(CS_DEFAULTS["cs_dedup_energy_tol"])  # type: ignore[arg-type]
    cs_dedup_geom_tol: float = float(CS_DEFAULTS["cs_dedup_geom_tol"])  # type: ignore[arg-type]
    cs_dedup_fp_tol: float = float(CS_DEFAULTS["cs_dedup_fp_tol"])  # type: ignore[arg-type]
    cs_fp_rcut: float = float(CS_DEFAULTS["cs_fp_rcut"])  # type: ignore[arg-type]
    cs_fp_binwidth: float = float(CS_DEFAULTS["cs_fp_binwidth"])  # type: ignore[arg-type]
    cs_fp_sigma: float = float(CS_DEFAULTS["cs_fp_sigma"])  # type: ignore[arg-type]

    # --- refinement / export ---
    cs_top_k: int = int(CS_DEFAULTS["cs_top_k"])  # type: ignore[arg-type]
    cs_export_window: float = float(CS_DEFAULTS["cs_export_window"])  # type: ignore[arg-type]
    cs_refine: bool = bool(CS_DEFAULTS["cs_refine"])
    cs_refine_fmax: float = float(CS_DEFAULTS["cs_refine_fmax"])  # type: ignore[arg-type]
    cs_refine_steps: int = int(CS_DEFAULTS["cs_refine_steps"])  # type: ignore[arg-type]

    # --- gates ---
    cs_allow_direct_forces: bool = bool(CS_DEFAULTS["cs_allow_direct_forces"])

    #: Parsed ``cs_place_anchor`` indices (0-based within the molecule file).
    _anchor_indices: tuple[int, ...] | None = field(
        init=False, default=None, repr=False, compare=False
    )
    #: Parsed ``cs_ionic_species`` element symbols.
    _ionic_symbols: tuple[str, ...] = field(
        init=False, default=(), repr=False, compare=False
    )
    #: Parsed ``cs_place_anchor_targets`` element symbols.
    _anchor_target_symbols: tuple[str, ...] | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.cs_mode, str):
            raise ConfsearchError("cs_mode must be a string")
        mode = self.cs_mode.strip().lower()
        if mode != "adsorbate":
            if mode in _FUTURE_MODES:
                raise reject_non_goal(mode)
            raise ConfsearchError(
                f"cs_mode must be 'adsorbate' in v1, got {self.cs_mode!r}"
            )
        object.__setattr__(self, "cs_mode", mode)

        self._validate_regions()
        self._validate_md()
        self._validate_chemistry()
        self._validate_placement()
        self._validate_dedup()
        self._validate_refinement()
        self._validate_gates()

    # -- derived, validated quantities ------------------------------------

    @property
    def ionic_symbols(self) -> tuple[str, ...]:
        """Element symbols excluded from covalent graphs."""
        return self._ionic_symbols

    @property
    def anchor_indices(self) -> tuple[int, ...] | None:
        """Parsed ``cs_place_anchor`` (0-based within the molecule file)."""
        return self._anchor_indices

    @property
    def anchor_target_symbols(self) -> tuple[str, ...] | None:
        """Parsed ``cs_place_anchor_targets`` (element symbols)."""
        return self._anchor_target_symbols

    @property
    def mode(self) -> str:
        """``"A"`` (user-prepared starts) or ``"B"`` (automatic placement)."""
        return "B" if self.cs_slab is not None else "A"

    @property
    def n_hold(self) -> int:
        return steps_from_ps(self.cs_hold_ps, self.timestep, name="cs_hold_ps")

    @property
    def n_cool(self) -> int:
        return steps_from_ps(self.cs_cool_ps, self.timestep, name="cs_cool_ps")

    @property
    def n_snapshot_interval(self) -> int:
        return steps_from_ps(
            self.cs_snapshot_interval_ps,
            self.timestep,
            name="cs_snapshot_interval_ps",
            allow_zero=True,
        )

    @property
    def snapshot_steps(self) -> tuple[int, ...]:
        """Hold-phase steps at which frames are quenched (spec 7.3 step 4)."""
        interval = self.n_snapshot_interval
        if interval <= 0:
            return ()
        hold = self.n_hold
        return tuple(range(interval, hold, interval))

    @property
    def total_md_steps_per_cycle(self) -> int:
        return self.n_hold + self.n_cool

    @property
    def total_quench_upper_bound(self) -> int:
        """One quench per snapshot plus one end-of-cycle frame, per cycle."""
        return (len(self.snapshot_steps) + 1) * self.cs_cycles

    # -- validation -------------------------------------------------------

    def _validate_regions(self) -> None:
        starts = self._normalize_paths(self.cs_starts, label="cs_starts")
        object.__setattr__(self, "cs_starts", starts)
        object.__setattr__(
            self, "cs_slab", self._normalize_path(self.cs_slab, label="cs_slab")
        )
        object.__setattr__(
            self,
            "cs_molecule",
            self._normalize_path(self.cs_molecule, label="cs_molecule"),
        )

        if self.cs_adsorbate_indices is not None:
            # Syntax only; the atom-count bound is checked against the structure.
            parse_index_spec(self.cs_adsorbate_indices, label="cs_adsorbate_indices")

        has_starts = bool(starts)
        has_indices = self.cs_adsorbate_indices is not None
        has_slab = self.cs_slab is not None
        has_molecule = self.cs_molecule is not None
        has_n_starts = self.cs_n_starts is not None

        if self.cs_n_starts is not None:
            object.__setattr__(
                self,
                "cs_n_starts",
                _positive_int(self.cs_n_starts, "cs_n_starts", minimum=1),
            )

        if has_slab or has_molecule or has_n_starts:
            # Mode B: placement.  Every mode-B key is required, and the
            # mode-A keys must be absent (spec 12.3 cross-key validation).
            if not (has_slab and has_molecule and has_n_starts):
                missing = [
                    name
                    for name, present in (
                        ("cs_slab", has_slab),
                        ("cs_molecule", has_molecule),
                        ("cs_n_starts", has_n_starts),
                    )
                    if not present
                ]
                raise ConfsearchError(
                    "automatic placement (mode B) requires "
                    f"{', '.join(missing)} as well; alternatively give "
                    "cs_starts + cs_adsorbate_indices (mode A)"
                )
            if has_starts or has_indices:
                raise ConfsearchError(
                    "cs_starts/cs_adsorbate_indices (mode A) cannot be combined "
                    "with cs_slab/cs_molecule/cs_n_starts (mode B); the two start "
                    "sources are separate runs in v1"
                )
        else:
            if not (has_starts and has_indices):
                raise ConfsearchError(
                    "confsearch needs either cs_starts + cs_adsorbate_indices "
                    "(mode A) or cs_slab + cs_molecule + cs_n_starts (mode B)"
                )

        if self.cs_fix_below is not None:
            object.__setattr__(
                self,
                "cs_fix_below",
                _positive(self.cs_fix_below, "cs_fix_below"),
            )

    def _validate_md(self) -> None:
        object.__setattr__(self, "timestep", _positive(self.timestep, "timestep"))
        object.__setattr__(self, "friction", _positive(self.friction, "friction"))
        object.__setattr__(
            self,
            "fmax_abort",
            _positive(self.fmax_abort, "fmax_abort"),
        )
        mobile = str(self.cs_md_mobile).strip().lower()
        if mobile not in MD_MOBILE_CHOICES:
            raise ConfsearchError(
                f"cs_md_mobile must be one of {', '.join(MD_MOBILE_CHOICES)}, "
                f"got {self.cs_md_mobile!r}"
            )
        object.__setattr__(self, "cs_md_mobile", mobile)
        object.__setattr__(
            self, "cs_top_depth", _positive(self.cs_top_depth, "cs_top_depth")
        )

        high = _positive(self.cs_temperature_high, "cs_temperature_high")
        low = _positive(self.cs_temperature_low, "cs_temperature_low")
        if low > high:
            raise ConfsearchError(
                f"cs_temperature_low ({low}) must be <= cs_temperature_high ({high})"
            )
        object.__setattr__(self, "cs_temperature_high", high)
        object.__setattr__(self, "cs_temperature_low", low)

        hold = _positive(self.cs_hold_ps, "cs_hold_ps")
        cool = _positive(self.cs_cool_ps, "cs_cool_ps")
        snapshot = _positive(
            self.cs_snapshot_interval_ps, "cs_snapshot_interval_ps", allow_zero=True
        )
        if snapshot >= hold:
            raise ConfsearchError(
                f"cs_snapshot_interval_ps ({snapshot}) must be < cs_hold_ps ({hold})"
            )
        object.__setattr__(self, "cs_hold_ps", hold)
        object.__setattr__(self, "cs_cool_ps", cool)
        object.__setattr__(self, "cs_snapshot_interval_ps", snapshot)

        object.__setattr__(
            self,
            "cs_ramp_interval",
            _positive_int(self.cs_ramp_interval, "cs_ramp_interval", minimum=1),
        )
        object.__setattr__(
            self,
            "cs_cycles",
            _positive_int(self.cs_cycles, "cs_cycles", minimum=0),
        )
        object.__setattr__(
            self,
            "cs_save_interval",
            _positive_int(self.cs_save_interval, "cs_save_interval", minimum=1),
        )
        object.__setattr__(
            self,
            "cs_accept_temperature",
            _positive(self.cs_accept_temperature, "cs_accept_temperature"),
        )
        if self.seed is not None:
            object.__setattr__(
                self, "seed", _positive_int(self.seed, "seed", minimum=0)
            )

        # Step conversions must be exact integers (spec 7.3); calling the
        # properties here makes a bad combination fail at construction.
        if self.n_hold < 1 or self.n_cool < 1:
            raise ConfsearchError(
                "cs_hold_ps and cs_cool_ps must each cover at least one timestep"
            )
        self.n_snapshot_interval

    def _validate_chemistry(self) -> None:
        bond = finite_float(self.cs_bond_scale, name="cs_bond_scale")
        if bond <= 1.0:
            raise ConfsearchError(f"cs_bond_scale must be > 1, got {bond}")
        broken = finite_float(self.cs_bond_break_scale, name="cs_bond_break_scale")
        if broken <= bond:
            raise ConfsearchError(
                f"cs_bond_break_scale ({broken}) must be > cs_bond_scale ({bond})"
            )
        object.__setattr__(self, "cs_bond_scale", bond)
        object.__setattr__(self, "cs_bond_break_scale", broken)

        object.__setattr__(
            self,
            "_ionic_symbols",
            _parse_elements(self.cs_ionic_species, label="cs_ionic_species"),
        )

        policy = str(self.cs_reaction_policy).strip().lower()
        if policy not in REACTION_POLICIES:
            raise ConfsearchError(
                f"cs_reaction_policy must be one of "
                f"{', '.join(REACTION_POLICIES)}, got {self.cs_reaction_policy!r}"
            )
        object.__setattr__(self, "cs_reaction_policy", policy)

        object.__setattr__(
            self,
            "cs_desorption_distance",
            _positive(self.cs_desorption_distance, "cs_desorption_distance"),
        )
        object.__setattr__(
            self,
            "cs_min_image_distance",
            _positive(self.cs_min_image_distance, "cs_min_image_distance"),
        )
        object.__setattr__(
            self,
            "cs_min_vacuum_gap",
            _positive(self.cs_min_vacuum_gap, "cs_min_vacuum_gap"),
        )

    def _validate_placement(self) -> None:
        contact = finite_float(
            self.cs_place_contact_scale, name="cs_place_contact_scale"
        )
        if contact <= self.cs_bond_scale:
            raise ConfsearchError(
                f"cs_place_contact_scale ({contact}) must be > cs_bond_scale "
                f"({self.cs_bond_scale})"
            )
        anchor = finite_float(self.cs_place_anchor_scale, name="cs_place_anchor_scale")
        if anchor <= 0 or anchor > self.cs_bond_scale:
            raise ConfsearchError(
                f"cs_place_anchor_scale must be > 0 and <= cs_bond_scale "
                f"({self.cs_bond_scale}), got {anchor}"
            )
        other = finite_float(
            self.cs_place_other_min_scale, name="cs_place_other_min_scale"
        )
        if other <= self.cs_bond_scale:
            raise ConfsearchError(
                f"cs_place_other_min_scale ({other}) must be > cs_bond_scale "
                f"({self.cs_bond_scale})"
            )
        object.__setattr__(self, "cs_place_contact_scale", contact)
        object.__setattr__(self, "cs_place_anchor_scale", anchor)
        object.__setattr__(self, "cs_place_other_min_scale", other)
        object.__setattr__(
            self,
            "cs_place_max_attempts",
            _positive_int(
                self.cs_place_max_attempts, "cs_place_max_attempts", minimum=1
            ),
        )

        if self.cs_place_anchor is not None:
            if self.mode != "B":
                raise ConfsearchError(
                    "cs_place_anchor is only valid in mode B (automatic placement)"
                )
            object.__setattr__(
                self,
                "_anchor_indices",
                parse_index_spec(self.cs_place_anchor, label="cs_place_anchor"),
            )
        if self.cs_place_anchor_targets is not None:
            if self.cs_place_anchor is None:
                raise ConfsearchError(
                    "cs_place_anchor_targets requires cs_place_anchor (anchored "
                    "placement); otherwise the closest slab element is not defined"
                )
            object.__setattr__(
                self,
                "_anchor_target_symbols",
                _parse_elements(
                    self.cs_place_anchor_targets, label="cs_place_anchor_targets"
                ),
            )

    def _validate_dedup(self) -> None:
        object.__setattr__(
            self,
            "cs_dedup_energy_tol",
            _positive(self.cs_dedup_energy_tol, "cs_dedup_energy_tol"),
        )
        object.__setattr__(
            self,
            "cs_dedup_geom_tol",
            _positive(self.cs_dedup_geom_tol, "cs_dedup_geom_tol"),
        )
        fp_tol = finite_float(self.cs_dedup_fp_tol, name="cs_dedup_fp_tol")
        if not 0.0 < fp_tol < 1.0:
            raise ConfsearchError(f"cs_dedup_fp_tol must be in (0, 1), got {fp_tol}")
        object.__setattr__(self, "cs_dedup_fp_tol", fp_tol)
        object.__setattr__(self, "cs_fp_rcut", _positive(self.cs_fp_rcut, "cs_fp_rcut"))
        object.__setattr__(
            self, "cs_fp_binwidth", _positive(self.cs_fp_binwidth, "cs_fp_binwidth")
        )
        object.__setattr__(
            self, "cs_fp_sigma", _positive(self.cs_fp_sigma, "cs_fp_sigma")
        )
        if self.cs_fp_sigma >= self.cs_fp_rcut:
            raise ConfsearchError(
                f"cs_fp_sigma ({self.cs_fp_sigma}) must be < cs_fp_rcut "
                f"({self.cs_fp_rcut})"
            )
        if self.cs_fp_binwidth > self.cs_fp_rcut:
            raise ConfsearchError(
                f"cs_fp_binwidth ({self.cs_fp_binwidth}) must be <= cs_fp_rcut "
                f"({self.cs_fp_rcut})"
            )
        if self.cs_fp_binwidth > 0.5 * self.cs_fp_sigma:
            raise ConfsearchError(
                f"cs_fp_binwidth ({self.cs_fp_binwidth}) must be <= "
                f"cs_fp_sigma/2 ({0.5 * self.cs_fp_sigma}); the Gaussian smearing "
                "must span at least two bins"
            )

    def _validate_refinement(self) -> None:
        object.__setattr__(self, "cs_top_k", _positive_int(self.cs_top_k, "cs_top_k"))
        object.__setattr__(
            self,
            "cs_export_window",
            _positive(self.cs_export_window, "cs_export_window"),
        )
        if not isinstance(self.cs_refine, bool):
            raise ConfsearchError("cs_refine must be a boolean")
        fmax = _positive(self.cs_refine_fmax, "cs_refine_fmax")
        if fmax > self.cs_quench_fmax:
            raise ConfsearchError(
                f"cs_refine_fmax ({fmax}) must be <= cs_quench_fmax "
                f"({self.cs_quench_fmax})"
            )
        object.__setattr__(self, "cs_refine_fmax", fmax)
        object.__setattr__(
            self,
            "cs_refine_steps",
            _positive_int(self.cs_refine_steps, "cs_refine_steps", minimum=1),
        )

    def _validate_gates(self) -> None:
        object.__setattr__(
            self,
            "cs_quench_fmax",
            _positive(self.cs_quench_fmax, "cs_quench_fmax"),
        )
        object.__setattr__(
            self,
            "cs_quench_steps",
            _positive_int(self.cs_quench_steps, "cs_quench_steps", minimum=1),
        )
        if not isinstance(self.cs_allow_direct_forces, bool):
            raise ConfsearchError("cs_allow_direct_forces must be a boolean")

    @staticmethod
    def _normalize_path(value: object, *, label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, (str, Path)):
            raise ConfsearchError(f"{label} must be a path, got {value!r}")
        text = str(value).strip()
        if not text:
            raise ConfsearchError(f"{label} must not be empty")
        return text

    @classmethod
    def _normalize_paths(cls, value: object, *, label: str) -> tuple[str, ...] | None:
        """Accept one path, a sequence of paths, or an ``os.pathsep``-joined list.

        The INCAR/CLI/API config layers carry this key as a scalar, so the CLI
        and the API join several positional start paths with the platform path
        separator (``:`` on POSIX, ``;`` on Windows) before resolution.
        """
        if value is None:
            return None
        if isinstance(value, str) and os.pathsep in value:
            parts = [part for part in value.split(os.pathsep) if part.strip()]
            if not parts:
                raise ConfsearchError(f"{label} must contain at least one path")
            return tuple(cls._normalize_path(part, label=label) or "" for part in parts)
        if isinstance(value, (str, Path)):
            return (cls._normalize_path(value, label=label) or "",)
        if isinstance(value, Sequence):
            normalized = tuple(
                item for item in (cls._normalize_path(v, label=label) for v in value)
            )
            if not normalized:
                raise ConfsearchError(f"{label} must contain at least one path")
            return normalized  # type: ignore[return-value]
        raise ConfsearchError(f"{label} must be a path or a list of paths")


def options_from_run_options(
    run_options: dict, *, extra: dict | None = None
) -> ConfsearchOptions:
    """Build :class:`ConfsearchOptions` from a resolved run-options mapping.

    Unknown keys are rejected so a configuration typo cannot silently change the
    scientific defaults; only keys that the dataclass declares are accepted.
    """
    known = {item.name for item in fields(ConfsearchOptions)}
    values: dict = {}
    for key, value in run_options.items():
        if key in known:
            values[key] = value
    for key, value in (extra or {}).items():
        if key not in known:
            raise ConfsearchError(f"Unknown confsearch option {key!r}")
        values[key] = value
    return ConfsearchOptions(**values)


__all__ = [
    "CS_DEFAULTS",
    "DEFAULT_IONIC_SPECIES",
    "MD_MOBILE_CHOICES",
    "REACTION_POLICIES",
    "ConfsearchOptions",
    "options_from_run_options",
    "steps_from_ps",
]
