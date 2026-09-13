"""Identity derivation for validation evidence.

Profile identity answers "which model produced this?"; record identity answers
"which computation was attempted?".  Both are deterministic content hashes so
two records can only be aggregated together when every identity component
matches (model artifact, dtype, task/head, inference mode, device class,
software commit, seed).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .constants import (
    LEGACY_PROJECT_NAME,
    RESULT_SCHEMA,
    RESULT_SCHEMA_V1,
    STATUS_SEMANTICS,
    VOLATILE_RECORD_FIELDS,
)


def canonical_json(payload: Any) -> str:
    """Stable JSON encoding used for every identity hash."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def device_class(device: Any) -> str:
    """Coarse device class used in profile identity (never the raw UUID)."""
    if not isinstance(device, dict):
        return "unknown"
    actual = device.get("actual")
    if not actual:
        return "unknown"
    actual = str(actual)
    if actual == "cpu":
        return "cpu"
    if actual.startswith("cuda"):
        return "cuda"
    return actual


def seed_from(record: dict[str, Any]) -> Any:
    """Best-effort seed identity: top-level field, else common parameter keys."""
    if record.get("seed") is not None:
        return record["seed"]
    params = record.get("parameters")
    if isinstance(params, dict):
        for key in ("seed", "random_seed", "rng_seed", "nvt_seed", "md_seed"):
            if params.get(key) is not None:
                return params[key]
    return None


def profile_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Full identity of the model/profile that produced a record."""
    return {
        "campaign_id": record.get("campaign_id"),
        "engine": record.get("engine"),
        "model_identity": record.get("model_identity"),
        "model_sha256": record.get("model_sha256"),
        "dtype": record.get("dtype"),
        "task": record.get("task"),
        "head": record.get("head"),
        "inference_mode": record.get("inference_mode"),
        "device_class": device_class(record.get("device")),
        "git_commit": record.get("git_commit"),
        "seed": seed_from(record),
    }


def profile_id_for(record: dict[str, Any]) -> str:
    """Human-readable, stable profile id: ``<engine>-<dtype>-<digest>``."""
    identity = profile_identity(record)
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    engine = str(record.get("engine", "unknown"))
    dtype = str(record.get("dtype", "unknown"))
    dtype = dtype.replace("/", "-").replace(" ", "-").lower()
    return f"{engine}-{dtype}-{digest[:10]}"


def record_id_for(record: dict[str, Any]) -> str:
    """Deterministic identity of the *intended computation*.

    Deliberately excludes measured values and timing: the same case on the
    same profile with the same parameters has the same record identity even
    when it is executed twice (each execution gets a fresh ``run_id``).
    """
    payload = {
        "profile_id": record.get("profile_id") or profile_id_for(record),
        "case_id": record.get("case_id"),
        "test_id": record.get("test_id"),
        "input_structure_id": record.get("input_structure_id"),
        "parameters": record.get("parameters"),
        "seed": seed_from(record),
        "schema": RESULT_SCHEMA,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:32]


def new_run_id() -> str:
    """A fresh execution-attempt id (32 hex chars)."""
    import uuid

    return uuid.uuid4().hex


def migrate_record(record: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Explicitly migrate an older-schema record to the current schema.

    Returns ``(record, note)``; old files are never rewritten on disk, so the
    original evidence stays byte-identical and every migration is visible in
    the loader diagnostics.
    """
    if not isinstance(record, dict):
        msg = "migrate_record expects a JSON object"
        raise TypeError(msg)
    schema = record.get("schema")
    if schema == RESULT_SCHEMA:
        return record, None
    legacy_v1 = f"{LEGACY_PROJECT_NAME}.beta-validation-result/1"
    if schema not in (RESULT_SCHEMA_V1, legacy_v1):
        msg = f"cannot migrate unknown result schema {schema!r}"
        raise ValueError(msg)
    migrated = dict(record)
    migrated["schema"] = RESULT_SCHEMA
    if str(schema).startswith(f"{LEGACY_PROJECT_NAME}."):
        migrated["legacy_schema_namespace"] = str(schema)
    migrated["inference_mode"] = record.get("inference_mode")
    migrated["seed"] = seed_from(record)
    migrated["profile_id"] = profile_id_for(migrated)
    migrated["record_id"] = record_id_for(migrated)
    migrated["run_id"] = hashlib.sha256(
        canonical_json(
            {
                "legacy_schema": RESULT_SCHEMA_V1,
                "profile_id": migrated["profile_id"],
                "record_id": migrated["record_id"],
                "wall_seconds": record.get("wall_seconds"),
            }
        ).encode("utf-8")
    ).hexdigest()[:32]
    migrated["migrated_from"] = RESULT_SCHEMA_V1
    if (
        "mliport_version" not in migrated
        and migrated.get(f"{LEGACY_PROJECT_NAME}_version") is not None
    ):
        migrated["mliport_version"] = migrated[f"{LEGACY_PROJECT_NAME}_version"]
    return migrated, f"migrated {schema} -> {RESULT_SCHEMA}"


def record_fingerprint(record: dict[str, Any]) -> str:
    """Content fingerprint of one record, excluding volatile metadata."""
    payload = {
        k: v
        for k, v in record.items()
        if k not in VOLATILE_RECORD_FIELDS and not k.startswith("_")
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def status_semantics(status: Any) -> str:
    return STATUS_SEMANTICS.get(str(status), "unknown status semantics")
