"""Model aliases and reusable profiles (plan section 4.3 / 4.4 / 6.2).

Both are declared in ``settings.ini`` using section prefixes::

    [model:mace_mpa0]
    engine = mace
    path = models/mace/mace-mpa-0-medium.model
    task = bulk
    dtype = float32

    [profile:md_smoke_300K]
    calc_type = md
    ensemble = NVT
    temperature = 300

This module parses those sections into small dataclasses and provides lookup
helpers used by the resolver. ``dtype`` is accepted as a friendly alias for the
canonical ``default_dtype`` calculator option.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mliport.config.provenance import SourceLocation

if TYPE_CHECKING:
    import configparser
    from typing import Any


_MODEL_PREFIX = "model:"
_PROFILE_PREFIX = "profile:"


@dataclass
class ModelAlias:
    """A named model + engine + task bundle."""

    name: str
    engine: str
    path: str
    task: str = "bulk"
    options: dict[str, Any] = field(default_factory=dict)
    origins: dict[str, SourceLocation] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict usable as a resolver layer."""
        result: dict[str, Any] = {
            "model_type": self.engine,
            "model_path": self.path,
            "task": self.task,
        }
        result.update(self.options)
        return result


@dataclass
class Profile:
    """A named, reusable bundle of calc_type + run/calculator options."""

    name: str
    options: dict[str, Any] = field(default_factory=dict)
    origins: dict[str, SourceLocation] = field(default_factory=dict)

    @property
    def calc_type(self) -> str | None:
        value = self.options.get("calc_type")
        return str(value).lower() if value is not None else None


def _parse_section_items(section: dict[str, str]) -> dict[str, Any]:
    """Strip whitespace while preserving raw tokens for schema coercion."""
    return {key: raw.strip() for key, raw in section.items()}


def parse_model_aliases(
    parser: configparser.ConfigParser,
    origins: dict[tuple[str, str], SourceLocation] | None = None,
) -> dict[str, ModelAlias]:
    """Extract all ``[model:...]`` sections from a parsed settings.ini."""
    aliases: dict[str, ModelAlias] = {}
    for section in parser.sections():
        if not section.lower().startswith(_MODEL_PREFIX):
            continue
        name = section[len(_MODEL_PREFIX) :].strip()
        if not name:
            continue
        items = _parse_section_items(dict(parser.items(section)))
        engine = str(items.pop("engine", "uma")).lower()
        path = str(items.pop("path", ""))
        task = str(items.pop("task", "bulk")).lower()
        value_origins: dict[str, SourceLocation] = {}
        origin_map = origins or {}
        for raw_key, canonical in (
            ("engine", "model_type"),
            ("path", "model_path"),
            ("task", "task"),
        ):
            origin = origin_map.get((section.lower(), raw_key))
            if origin is not None:
                value_origins[canonical] = origin
        # ``dtype`` is a friendly alias for the MACE default_dtype option.
        if "dtype" in items:
            items["default_dtype"] = str(items.pop("dtype")).lower()
            origin = origin_map.get((section.lower(), "dtype"))
            if origin is not None:
                value_origins["default_dtype"] = origin
        for key in items:
            origin = origin_map.get((section.lower(), key.lower()))
            if origin is not None:
                value_origins[key] = origin
        aliases[name] = ModelAlias(
            name=name,
            engine=engine,
            path=path,
            task=task,
            options=items,
            origins=value_origins,
        )
    return aliases


def parse_profiles(
    parser: configparser.ConfigParser,
    origins: dict[tuple[str, str], SourceLocation] | None = None,
) -> dict[str, Profile]:
    """Extract all ``[profile:...]`` sections from a parsed settings.ini."""
    profiles: dict[str, Profile] = {}
    for section in parser.sections():
        if not section.lower().startswith(_PROFILE_PREFIX):
            continue
        name = section[len(_PROFILE_PREFIX) :].strip()
        if not name:
            continue
        options = _parse_section_items(dict(parser.items(section)))
        value_origins = {
            key: origin
            for key in options
            if (origin := (origins or {}).get((section.lower(), key.lower())))
            is not None
        }
        profiles[name] = Profile(name=name, options=options, origins=value_origins)
    return profiles


def resolve_model_alias(
    name: str | None,
    aliases: dict[str, ModelAlias],
) -> dict[str, Any]:
    """Return the flat option dict for model alias ``name`` (empty if None)."""
    if not name:
        return {}
    alias = aliases.get(name)
    if alias is None:
        raise KeyError(
            f"Unknown model alias {name!r}. "
            f"Known aliases: {', '.join(sorted(aliases)) or '(none)'}"
        )
    return alias.to_dict()


def resolve_profile(
    name: str | None,
    profiles: dict[str, Profile],
) -> dict[str, Any]:
    """Return the flat option dict for profile ``name`` (empty if None)."""
    if not name:
        return {}
    profile = profiles.get(name)
    if profile is None:
        raise KeyError(
            f"Unknown profile {name!r}. "
            f"Known profiles: {', '.join(sorted(profiles)) or '(none)'}"
        )
    return dict(profile.options)
