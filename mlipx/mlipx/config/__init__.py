"""mlipx configuration system (Phase 1).

This package is the single source of truth for built-in defaults, settings.ini
loading, model/profile aliases, strict schema validation and layered config
resolution. It is designed so that the CLI, the Python API and the INCAR flow
all read the *same* defaults instead of each hard-coding their own.

Public surface (re-exported here for convenience)::

    from mlipx.config import (
        BUILTIN_DEFAULTS,
        ConfigResolver,
        IncarConfig,
        MlipxSettings,
        OptionSpec,
        Schema,
        get_default_config,
        settings_search_paths,
    )

The legacy ``mlipx.config`` module (``IncarConfig`` + ``DEFAULT_*_CONFIG``) is
kept as a thin backward-compatibility shim that re-exports from here.
"""

from __future__ import annotations

from mlipx.config.aliases import ModelAlias, Profile, resolve_model_alias
from mlipx.config.defaults import (
    BUILTIN_DEFAULTS,
    DEFAULT_DEVICE_BY_CALC_TYPE,
    build_incar_default,
    get_default_config,
)
from mlipx.config.incar import IncarConfig
from mlipx.config.provenance import SourceLocation
from mlipx.config.resolver import ResolvedConfig, ResolvedValue, resolve_config
from mlipx.config.schema import OptionSpec, Schema, get_schema
from mlipx.config.settings import (
    DEFAULT_SETTINGS_INI,
    MlipxSettings,
    init_settings_file,
    load_settings,
    settings_search_paths,
)

# Backward-compatible template strings, regenerated from the single source of
# defaults (plan section 17.7) so there is only one owner of the values.
DEFAULT_SP_CONFIG = build_incar_default("sp")
DEFAULT_OPT_CONFIG = build_incar_default("opt")
DEFAULT_MD_CONFIG = build_incar_default("md")

__all__ = [
    "BUILTIN_DEFAULTS",
    "DEFAULT_DEVICE_BY_CALC_TYPE",
    "DEFAULT_MD_CONFIG",
    "DEFAULT_OPT_CONFIG",
    "DEFAULT_SETTINGS_INI",
    "DEFAULT_SP_CONFIG",
    "IncarConfig",
    "ModelAlias",
    "MlipxSettings",
    "OptionSpec",
    "Profile",
    "ResolvedConfig",
    "ResolvedValue",
    "Schema",
    "SourceLocation",
    "build_incar_default",
    "get_default_config",
    "get_schema",
    "init_settings_file",
    "load_settings",
    "resolve_config",
    "resolve_model_alias",
    "settings_search_paths",
]
