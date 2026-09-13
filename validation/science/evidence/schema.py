"""Dependency-free JSON Schema interpreter for the evidence layer.

The validation scripts run inside backend virtual environments that carry
only the model stack (torch / tensorflow / deepmd), so ``jsonschema`` is
deliberately not a requirement.  The committed schema documents must not be
decorative: this module executes the exact keyword subset they use and fails
loudly on any keyword it does not understand, so silently weakening a schema
is impossible.

Supported keywords: ``$schema``, ``$id``, ``title``, ``description``,
``type``, ``required``, ``properties``, ``additionalProperties``, ``enum``,
``const``, ``pattern``, ``minimum``, ``maximum``, ``items``, ``default``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .constants import RESULT_SCHEMA, RESULT_SCHEMA_V1

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
RESULT_SCHEMA_FILES = {
    RESULT_SCHEMA_V1: SCHEMA_DIR / "result-v1.schema.json",
    RESULT_SCHEMA: SCHEMA_DIR / "result.schema.json",
}
SUMMARY_SCHEMA_FILES = {
    "mlipx.beta-validation-summary/1": SCHEMA_DIR / "summary-v1.schema.json",
    "mlipx.beta-validation-summary/2": SCHEMA_DIR / "summary.schema.json",
}

SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "title",
        "description",
        "type",
        "required",
        "properties",
        "additionalProperties",
        "enum",
        "const",
        "pattern",
        "minimum",
        "maximum",
        "items",
        "default",
    }
)

_TYPE_CHECKS: dict[str, Any] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: (
        isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    ),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


class SchemaError(RuntimeError):
    """The schema document itself is invalid or uses unsupported keywords."""


def _check_keywords(schema: dict[str, Any], where: str) -> None:
    unknown = sorted(set(schema) - SUPPORTED_KEYWORDS)
    if unknown:
        msg = (
            f"{where}: unsupported JSON Schema keyword(s) {unknown}; extend "
            "evidence/schema.py deliberately instead of ignoring them"
        )
        raise SchemaError(msg)


def _validate(node: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict):
        msg = f"{path}: schema node must be an object"
        raise SchemaError(msg)
    _check_keywords(schema, path)

    if "const" in schema and node != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {node!r}")
    if "enum" in schema and node not in schema["enum"]:
        errors.append(f"{path}: {node!r} is not one of {schema['enum']}")

    declared = schema.get("type")
    if declared is not None:
        types = declared if isinstance(declared, list) else [declared]
        for t in types:
            if t not in _TYPE_CHECKS:
                msg = f"{path}: unknown JSON Schema type {t!r}"
                raise SchemaError(msg)
        if not any(_TYPE_CHECKS[t](node) for t in types):
            errors.append(
                f"{path}: expected type {declared}, got {type(node).__name__}"
            )
            return

    if (
        isinstance(node, str)
        and "pattern" in schema
        and re.search(schema["pattern"], node) is None
    ):
        errors.append(f"{path}: {node!r} does not match {schema['pattern']!r}")
    if (
        isinstance(node, int | float)
        and not isinstance(node, bool)
        and "minimum" in schema
        and node < schema["minimum"]
    ):
        errors.append(f"{path}: {node!r} < minimum {schema['minimum']!r}")
    if (
        isinstance(node, int | float)
        and not isinstance(node, bool)
        and "maximum" in schema
        and node > schema["maximum"]
    ):
        errors.append(f"{path}: {node!r} > maximum {schema['maximum']!r}")

    if isinstance(node, dict):
        errors.extend(
            f"{path}: missing required property {key!r}"
            for key in schema.get("required", [])
            if key not in node
        )
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, value in node.items():
            child = f"{path}.{key}"
            if key in props:
                _validate(value, props[key], child, errors)
            elif extra is False:
                errors.append(f"{child}: additional property not allowed")
            elif isinstance(extra, dict):
                _validate(value, extra, child, errors)
    if isinstance(node, list) and "items" in schema:
        for i, value in enumerate(node):
            _validate(value, schema["items"], f"{path}[{i}]", errors)


def validate(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of human-readable violations (empty means valid)."""
    errors: list[str] = []
    _validate(instance, schema, "$", errors)
    return errors


def load_schema(path: str | Path) -> dict[str, Any]:
    """Load and structurally check a committed schema document."""
    path = Path(path)
    schema = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        msg = f"{path}: schema document must be a JSON object"
        raise SchemaError(msg)
    _check_keywords(schema, str(path))
    return schema


def validate_file(instance: Any, path: str | Path) -> list[str]:
    return validate(instance, load_schema(path))


def result_schema_path(schema_id: str) -> Path | None:
    return RESULT_SCHEMA_FILES.get(schema_id)
