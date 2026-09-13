"""settings.ini loading and search order (plan section 4.2 / 4.4).

The search order is::

    1. --settings /path/to/settings.ini
    2. environment variable MLIPORT_SETTINGS
    3. current working directory ./settings.ini
    4. user config ~/.config/mliport/settings.ini
       (Windows: %APPDATA%/mliport/settings.ini)
    5. mliport built-in defaults

mliport never ships a settings.ini inside the installed package on purpose: a
file living inside the site-packages of one virtualenv would be invisible to
other backends' environments. The defaults live in code
(:mod:`mliport.config.defaults`); settings.ini only overrides them.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mliport.config.provenance import SourceLocation

if TYPE_CHECKING:
    import configparser
    from typing import Any


# A conservative built-in settings.ini template (plan section 4.4). It is used
# by ``mliport config init`` and as the documentation baseline. It deliberately
# keeps scientific defaults (temperature, steps, ...) out of settings.ini so
# that every run records the *resolved* value in its output directory instead.
DEFAULT_SETTINGS_INI = """\
; ============================================================
; mliport settings
; ============================================================
; Edit this file to point at your model backends / Python
; environments. Scientific defaults (temperature, MD steps, ...)
; are owned by mliport built-in defaults and recorded per-run, so
; they are intentionally NOT set here.
; ============================================================

[general]
; strict by default: unknown/cross-backend keys are fatal.
; set strict_config = false (or pass --lenient-config) to opt into warnings.
strict_config = true
write_resolved_config = true
default_seed =

[safety]
fmax_abort = 20.0

[md]
ensemble = NVT
temperature = 300
timestep_fs = 1.0
production_steps = 1000
thermostat = LANGEVIN
friction_per_fs = 0.001
bussi_tau = 1000.0
nhc_tdamp = 100.0
nhc_tchain = 3
nhc_tloop = 1
velocity_policy = auto

[opt]
optimizer = FIRE
fmax = 0.05
max_steps = 500
cell_opt = false
fix_symmetry = false

; ------------------------------------------------------------
; Per-backend calculation defaults. Environment executables and
; queue/restart/checkpoint policies are not implemented configuration
; options yet and are therefore intentionally absent.
; ------------------------------------------------------------

; [engine:mace]
; task = bulk
; device = cuda:0
; default_dtype = float64

; [engine:uma]
; task = omat
; device = cuda:0
; inference_mode = turbo

; ------------------------------------------------------------
; Model aliases
; ------------------------------------------------------------

; [model:mace_mpa0]
; engine = mace
; path = models/mace/mace-mpa-0-medium.model
; task = bulk
; dtype = float64

; ------------------------------------------------------------
; Reusable profiles
; ------------------------------------------------------------

; [profile:bulk_sp]
; calc_type = sp
; task = bulk

; [profile:md_smoke_300K]
; calc_type = md
; ensemble = NVT
; temperature = 300
; timestep_fs = 1.0
; production_steps = 2000
"""


def user_config_dir() -> Path:
    """Return the per-user config directory for mliport settings.ini."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "mliport"
    return Path.home() / ".config" / "mliport"


def settings_search_paths(
    *,
    explicit: str | Path | None = None,
    cwd: str | Path | None = None,
) -> list[Path]:
    """Return the ordered list of candidate settings.ini locations.

    Args:
        explicit: Value of ``--settings`` (highest priority if given).
        cwd: Working directory used for the ``./settings.ini`` candidate.
            Defaults to the current working directory.
    """
    base = Path(cwd).expanduser() if cwd is not None else Path.cwd()
    base = base.resolve()

    def declared_path(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (base / path).resolve()

    candidates: list[Path] = []
    if explicit:
        candidates.append(declared_path(explicit))
    env_path = os.environ.get("MLIPORT_SETTINGS")
    if env_path:
        candidates.append(declared_path(env_path))
    candidates.append(base / "settings.ini")
    candidates.append((user_config_dir() / "settings.ini").resolve())
    return candidates


def _empty_parser() -> configparser.ConfigParser:
    import configparser  # noqa: PLC0415

    return configparser.ConfigParser(interpolation=None)


@dataclass
class MliportSettings:
    """Resolved settings.ini contents plus provenance.

    Attributes:
        parser: The raw ``configparser.ConfigParser`` (may be empty when no
            settings.ini was found).
        path: The file that was actually loaded, or ``None`` when only the
            built-in defaults apply.
        searched: Every candidate path that was considered (for
            ``mliport config paths``).
        sections: A flat ``{section: {key: value}}`` dict view of the parser.
    """

    parser: configparser.ConfigParser = field(default_factory=_empty_parser)
    path: Path | None = None
    """Highest-priority settings.ini actually loaded (None = built-in only)."""
    loaded_paths: list[Path] = field(default_factory=list)
    """Every settings.ini that contributed, lowest-priority first."""
    searched: list[Path] = field(default_factory=list)
    sections: dict[str, dict[str, str]] = field(default_factory=dict)
    origins: dict[tuple[str, str], SourceLocation] = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        """True when at least one settings.ini file was loaded."""
        return bool(self.loaded_paths)

    def get(self, section: str, key: str, default: Any | None = None) -> Any | None:
        """Return a raw value from ``[section] key``."""
        if not self.parser.has_option(section, key):
            return default
        return self.parser.get(section, key).strip()

    def section(self, name: str) -> dict[str, str]:
        """Return raw tokens for one section (empty if absent)."""
        if not self.parser.has_section(name):
            return {}
        return {k: v.strip() for k, v in self.parser.items(name)}

    def engine_section(self, engine: str) -> dict[str, str]:
        """Return the ``[engine:<name>]`` section for ``engine``."""
        return self.section(f"engine:{engine}")

    def origin(self, section: str, key: str) -> SourceLocation | None:
        """Return the file/line that declared one merged setting."""
        return self.origins.get((section.lower(), key.lower()))


def _option_line_numbers(path: Path) -> dict[tuple[str, str], int]:
    """Find first-line locations for configparser options in one file."""
    locations: dict[tuple[str, str], int] = {}
    section: str | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().lower()
            continue
        if section is None or line[:1].isspace():
            continue
        delimiters = [index for token in ("=", ":") if (index := line.find(token)) >= 0]
        if not delimiters:
            continue
        key = line[: min(delimiters)].strip().lower()
        if key:
            locations[(section, key)] = line_number
    return locations


def load_settings(
    *,
    explicit: str | Path | None = None,
    cwd: str | Path | None = None,
) -> MliportSettings:
    """Load and merge settings.ini files following the search order.

    Per the plan (section 4.3) the merge priority is::

        user settings  <  project settings  <  MLIPORT_SETTINGS  <  --settings

    i.e. higher-priority files override lower-priority ones. All existing
    candidates are merged (not just the first one found) so a project-level
    ``settings.ini`` can override a user-level one.
    """
    import configparser  # noqa: PLC0415

    # `searched` is high-priority-first (for `config paths` display); merging
    # is applied low-priority-first so higher priority wins.
    searched = settings_search_paths(explicit=explicit, cwd=cwd)
    merge_order = list(reversed(searched))
    parser = configparser.ConfigParser(interpolation=None)
    loaded_paths: list[Path] = []
    origins: dict[tuple[str, str], SourceLocation] = {}
    for candidate in merge_order:
        if not candidate.is_file():
            continue
        try:
            candidate_parser = configparser.ConfigParser(interpolation=None)
            candidate_parser.read(candidate, encoding="utf-8")
            parser.read(candidate, encoding="utf-8")
        except (configparser.Error, OSError) as exc:
            raise ValueError(
                f"Failed to parse settings file {candidate}: {exc}"
            ) from exc
        loaded_paths.append(candidate)
        line_numbers = _option_line_numbers(candidate)
        for section in candidate_parser.sections():
            for key, _value in candidate_parser.items(section):
                line = line_numbers.get((section.lower(), key.lower()))
                origins[(section.lower(), key.lower())] = SourceLocation.from_file(
                    candidate, line
                )

    # `path` is the highest-priority file loaded (last in merge_order).
    top = loaded_paths[-1] if loaded_paths else None

    sections: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        sections[section] = {k: v for k, v in parser.items(section)}

    return MliportSettings(
        parser=parser,
        path=top,
        loaded_paths=loaded_paths,
        searched=searched,
        sections=sections,
        origins=origins,
    )


def init_settings_file(target: str | Path, *, force: bool = False) -> Path:
    """Write the template settings.ini to ``target`` (plan section 4.2).

    Args:
        target: Destination path, or the strings ``"project"``/``"user"`` to
            write to ``./settings.ini`` or the user config dir respectively.
        force: Overwrite an existing file when True.

    Returns:
        The path that was written.
    """
    if target == "project":
        path = Path.cwd() / "settings.ini"
    elif target == "user":
        path = user_config_dir() / "settings.ini"
    else:
        path = Path(target).expanduser()

    if path.exists() and not force:
        raise FileExistsError(f"settings.ini already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_SETTINGS_INI, encoding="utf-8")
    return path
