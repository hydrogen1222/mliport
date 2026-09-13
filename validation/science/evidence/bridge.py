"""Canonical readers for release-bridge artifacts (recovery plan section 12-14).

Bridge records are *not* raw scientific evidence: they are a small,
release-scoped smoke campaign (four backends, model load + single point,
resolver/no-backend/strict-config checks).  They therefore live in
``validation/science/bridge/`` with their own schemas, and the report
generator reads them only through this module -- the same "no independent
record parsing in the report" rule that applies to the beta evidence loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BRIDGE_SUMMARY_SCHEMA = "mliport.release-bridge-campaign/1"
BRIDGE_RECORD_SCHEMA = "mliport.release-bridge/1"
BRIDGE_ARCHIVE_SCHEMA = "mliport.release-bridge-archive-manifest/1"


class BridgeArtifactError(ValueError):
    """A bridge summary/record/manifest is missing or malformed."""


def _load_json(path: Path, schema: str | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeArtifactError(f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BridgeArtifactError(f"{path}: expected a JSON object")
    if schema is not None and payload.get("schema") != schema:
        raise BridgeArtifactError(
            f"{path}: expected schema {schema!r}, got {payload.get('schema')!r}"
        )
    return payload


def load_bridge_summary(path: str | Path) -> dict[str, Any]:
    """Load and validate a ``mliport.release-bridge-campaign/1`` summary."""
    summary = _load_json(Path(path), BRIDGE_SUMMARY_SCHEMA)
    if not summary.get("release_candidate_commit"):
        raise BridgeArtifactError(f"{path}: missing release_candidate_commit")
    return summary


def load_bridge_archive_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a bridge archive manifest."""
    return _load_json(Path(path), BRIDGE_ARCHIVE_SCHEMA)


def latest_bridge_summary(root: str | Path) -> Path | None:
    """Newest ``*/summary.json`` under the bridge root (path only)."""
    candidates = sorted(Path(root).glob("*/summary.json"))
    return candidates[-1] if candidates else None


def bridge_engine_checks(campaign_dir: str | Path, engine: str) -> dict[str, Any]:
    """Checks of one engine record, or an empty mapping when absent."""
    record_path = Path(campaign_dir) / engine / f"{engine}.json"
    if not record_path.is_file():
        return {}
    try:
        payload = _load_json(record_path, BRIDGE_RECORD_SCHEMA)
    except BridgeArtifactError:
        return {}
    checks = payload.get("checks")
    return checks if isinstance(checks, dict) else {}


__all__ = [
    "BRIDGE_ARCHIVE_SCHEMA",
    "BRIDGE_RECORD_SCHEMA",
    "BRIDGE_SUMMARY_SCHEMA",
    "BridgeArtifactError",
    "bridge_engine_checks",
    "latest_bridge_summary",
    "load_bridge_archive_manifest",
    "load_bridge_summary",
]
