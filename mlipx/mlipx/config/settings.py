"""settings.ini loading and search order (plan section 4.2 / 4.4).

The search order is::

    1. --settings /path/to/settings.ini
    2. environment variable MLIPX_SETTINGS
    3. current working directory ./settings.ini
    4. user config ~/.config/mlipx/settings.ini
       (Windows: %APPDATA%/mlipx/settings.ini)
    5. mlipx built-in defaults

mlipx never ships a settings.ini inside the installed package on purpose: a
file living inside the site-packages of one virtualenv would be invisible to
other backends' environments. The defaults live in code
(:mod:`mlipx.config.defaults`); settings.ini only overrides them.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import configparser
    from typing import Any


# A conservative built-in settings.ini template (plan section 4.4). It is used
# by ``mlipx config init`` and as the documentation baseline. It deliberately
# keeps scientific defaults (temperature, steps, ...) out of settings.ini so
# that every run records the *resolved* value in its output directory instead.
DEFAULT_SETTINGS_INI = """\
; ============================================================
; mlipx settings
; ============================================================
; Edit this file to point at your model backends / Python
; environments. Scientific defaults (temperature, MD steps, ...)
; are owned by mlipx built-in defaults and recorded per-run, so
; they are intentionally NOT set here.
; ============================================================

[general]
strict_config = false
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
    """Return the per-user config directory for mlipx settings.ini."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "mlipx"
    return Path.home() / ".config" / "mlipx"


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
    base = Path(cwd) if cwd is not None else Path.cwd()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("MLIPX_SETTINGS")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(base / "settings.ini")
    candidates.append(user_config_dir() / "settings.ini")
    return candidates


def _empty_parser() -> configparser.ConfigParser:
    import configparser  # noqa: PLC0415

    return configparser.ConfigParser(interpolation=None)


@dataclass
class MlipxSettings:
    """Resolved settings.ini contents plus provenance.

    Attributes:
        parser: The raw ``configparser.ConfigParser`` (may be empty when no
            settings.ini was found).
        path: The file that was actually loaded, or ``None`` when only the
            built-in defaults apply.
        searched: Every candidate path that was considered (for
            ``mlipx config paths``).
        sections: A flat ``{section: {key: value}}`` dict view of the parser.
    """

    parser: configparser.ConfigParser = field(default_factory=_empty_parser)
    path: Path | None = None
    """Highest-priority settings.ini actually loaded (None = built-in only)."""
    loaded_paths: list[Path] = field(default_factory=list)
    """Every settings.ini that contributed, lowest-priority first."""
    searched: list[Path] = field(default_factory=list)
    sections: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        """True when at least one settings.ini file was loaded."""
        return bool(self.loaded_paths)

    def get(self, section: str, key: str, default: Any | None = None) -> Any | None:
        """Return a typed value from ``[section] key``."""
        if not self.parser.has_option(section, key):
            return default
        return self._coerce(self.parser.get(section, key))

    def section(self, name: str) -> dict[str, Any]:
        """Return a typed view of one section (empty if absent)."""
        if not self.parser.has_section(name):
            return {}
        return {k: self._coerce(v) for k, v in self.parser.items(name)}

    def engine_section(self, engine: str) -> dict[str, Any]:
        """Return the ``[engine:<name>]`` section for ``engine``."""
        return self.section(f"engine:{engine}")

    @staticmethod
    def _coerce(raw: str) -> Any:
        value = raw.strip()
        lowered = value.lower()
        if lowered in {"true", ".true.", "yes", "y", "t", "1"}:
            return True
        if lowered in {"false", ".false.", "no", "n", "f", "0"}:
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value


def load_settings(
    *,
    explicit: str | Path | None = None,
    cwd: str | Path | None = None,
) -> MlipxSettings:
    """Load and merge settings.ini files following the search order.

    Per the plan (section 4.3) the merge priority is::

        user settings  <  project settings  <  MLIPX_SETTINGS  <  --settings

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
    for candidate in merge_order:
        if not candidate.is_file():
            continue
        try:
            parser.read(candidate, encoding="utf-8")
        except (configparser.Error, OSError) as exc:
            raise ValueError(
                f"Failed to parse settings file {candidate}: {exc}"
            ) from exc
        loaded_paths.append(candidate)

    # `path` is the highest-priority file loaded (last in merge_order).
    top = loaded_paths[-1] if loaded_paths else None

    sections: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        sections[section] = {k: v for k, v in parser.items(section)}

    return MlipxSettings(
        parser=parser,
        path=top,
        loaded_paths=loaded_paths,
        searched=searched,
        sections=sections,
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
