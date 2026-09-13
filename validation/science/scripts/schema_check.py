"""Backward-compatible entry point for the canonical schema validator.

The implementation lives in :mod:`evidence.schema`; this shim keeps existing
imports (``import schema_check``) working while guaranteeing there is only
one validator implementation in the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCIENCE_ROOT = Path(__file__).resolve().parent.parent
if str(_SCIENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCIENCE_ROOT))

from evidence.schema import (  # noqa: E402
    SCHEMA_DIR,
    SUPPORTED_KEYWORDS,
    SchemaDefinitionError,
    SchemaError,
    load_schema,
    validate,
    validate_file,
)

__all__ = [
    "SCHEMA_DIR",
    "SUPPORTED_KEYWORDS",
    "SchemaDefinitionError",
    "SchemaError",
    "load_schema",
    "validate",
    "validate_file",
]
