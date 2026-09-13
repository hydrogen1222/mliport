"""Strict, single-entry-point loader for validation evidence.

``load_evidence`` is the only place that turns files on disk into canonical
records.  Every consumer (``aggregate.py``, the report generator, the README
block, the figure generator) must go through it; none of them may re-implement
glob selection, JSON parsing or validation (task book PR-A).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import schema as schema_module
from .constants import (
    BETA_VALIDATION_SUITE_REVISION,
    RESULT_SCHEMA,
    RESULT_SCHEMA_PREFIX,
    RESULT_SCHEMA_V1,
    VALIDATION_LOGIC_VERSION,
)
from .identity import (
    canonical_json,
    migrate_record,
    record_fingerprint,
    seed_from,
    status_semantics,
)

#: Directory names that select one validation tier.  A record's tier is the
#: first path component matching one of these names, which works both for the
#: historical ``t4/t4/...`` layout and for per-campaign roots such as
#: ``<campaign>/t1/...``.
TIER_NAMES = ("t1", "t2", "t2fd", "t3", "t4", "t5", "t6", "t7", "t8")

#: Canonical identity fields every normalized record must carry (task book
#: section 1.3).  Some are derived from the stored record when the schema did
#: not name them explicitly (see :func:`normalize_identity`).
IDENTITY_FIELDS = (
    "schema_version",
    "campaign_id",
    "record_id",
    "software_commit",
    "validation_logic_version",
    "engine",
    "upstream_model_id",
    "model_sha256",
    "dtype",
    "task",
    "head",
    "device_requested",
    "device_actual",
    "gpu_uuid_hash",
    "seed",
    "input_hash",
    "config_hash",
    "test_id",
    "case_id",
    "status",
    "status_semantics",
    "created_at",
)


def strict_json_loads(text: str) -> Any:
    """Parse JSON, rejecting NaN/Infinity constants instead of accepting them."""

    def _reject(token: str) -> None:
        msg = f"non-finite JSON constant {token!r} is not a valid result value"
        raise ValueError(msg)

    return json.loads(text, parse_constant=_reject)


def _nonfinite_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        found.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_nonfinite_paths(item, f"{path}.{key}"))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            found.extend(_nonfinite_paths(item, f"{path}[{index}]"))
    return found


def tier_of_path(path: Path) -> str | None:
    """First path component that names a validation tier, or ``None``."""
    for part in path.parts:
        if part in TIER_NAMES:
            return part
    return None


def _validate_result_schema(record: dict[str, Any], rel: str) -> list[str]:
    schema_id = record.get("schema")
    path = schema_module.result_schema_path(str(schema_id))
    if path is None:
        return [f"{rel}: unknown result schema {schema_id!r}"]
    try:
        violations = schema_module.validate_file(record, path)
    except schema_module.SchemaError as exc:
        return [f"{rel}: harness schema {path.name} is unusable ({exc})"]
    return [f"{rel}: {violation}" for violation in violations]


def _semantic_problems(record: dict[str, Any], rel: str) -> list[str]:
    problems: list[str] = []
    for section in ("parameters", "metrics", "diagnostics"):
        value = record.get(section)
        if not isinstance(value, dict):
            continue
        problems.extend(
            f"{rel}: non-finite number at {path}"
            for path in _nonfinite_paths(value, f"$.{section}")
        )
    if record.get("suite_revision", 0) > BETA_VALIDATION_SUITE_REVISION:
        problems.append(
            f"{rel}: suite_revision {record['suite_revision']} is newer than this "
            f"loader ({BETA_VALIDATION_SUITE_REVISION}); refusing to interpret "
            "future evidence"
        )
    if record.get("status") == "fail" and record.get("exception") is None:
        metrics = record.get("metrics") or {}
        diagnostics = record.get("diagnostics") or {}
        if not metrics and not diagnostics:
            problems.append(
                f"{rel}: status 'fail' carries neither exception, metrics nor "
                "diagnostics; a failure must carry a reason"
            )
    return problems


def normalize_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Canonical identity block for one record (never raises)."""
    device = record.get("device") if isinstance(record.get("device"), dict) else {}
    parameters = record.get("parameters") if isinstance(record.get("parameters"), dict) else {}
    config_digest = canonical_json(parameters)
    import hashlib

    return {
        "schema_version": record.get("schema"),
        "campaign_id": record.get("campaign_id"),
        "record_id": record.get("record_id"),
        "run_id": record.get("run_id"),
        "profile_id": record.get("profile_id"),
        "software_commit": record.get("git_commit"),
        "validation_logic_version": VALIDATION_LOGIC_VERSION,
        "engine": record.get("engine"),
        "upstream_model_id": record.get("model_identity"),
        "model_sha256": record.get("model_sha256"),
        "dtype": record.get("dtype"),
        "task": record.get("task"),
        "head": record.get("head"),
        "device_requested": device.get("requested"),
        "device_actual": device.get("actual"),
        "gpu_uuid_hash": device.get("gpu_uuid_hash"),
        "seed": seed_from(record),
        "input_hash": record.get("input_structure_id"),
        "config_hash": hashlib.sha256(config_digest.encode("utf-8")).hexdigest()[:16],
        "test_id": record.get("test_id"),
        "case_id": record.get("case_id"),
        "status": record.get("status"),
        "status_semantics": status_semantics(record.get("status")),
        "created_at": record.get("generated_at") or record.get("finished_at"),
    }


def _identity_problems(record: dict[str, Any], rel: str) -> list[str]:
    problems: list[str] = []
    for field in ("record_id", "profile_id", "case_id", "test_id", "engine"):
        if not record.get(field):
            problems.append(f"{rel}: identity field {field!r} is missing or empty")
    commit = record.get("git_commit")
    if not commit or commit == "unknown":
        problems.append(
            f"{rel}: software commit is missing/unknown; evidence without a "
            "software commit cannot be attributed"
        )
    if not record.get("model_sha256"):
        problems.append(f"{rel}: model_sha256 is missing")
    if record.get("schema") == RESULT_SCHEMA and not record.get("profile_id"):
        problems.append(f"{rel}: current-schema record has no profile_id")
    return problems


@dataclass
class LoadedEvidence:
    """Canonical loading result; every consumer reads this object."""

    root: Path
    records: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    ancillary_files: int = 0
    migrated_records: int = 0
    scanned_files: int = 0
    out_of_scope_records: int = 0
    untagged_records: int = 0
    campaign: str | None = None
    tier_counts: dict[str, int] = field(default_factory=dict)

    def by_tier(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {tier: [] for tier in TIER_NAMES}
        for record in self.records:
            tier = record.get("_tier")
            if tier is not None:
                grouped.setdefault(str(tier), []).append(record)
        return grouped

    def tier(self, name: str) -> list[dict[str, Any]]:
        return self.by_tier().get(name, [])

    @property
    def consistent(self) -> bool:
        """True only when no import/validation problem was found."""
        return not self.problems

    @property
    def counts(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for record in self.records:
            status = str(record.get("status"))
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "total": len(self.records),
            "by_status": dict(sorted(by_status.items())),
            "by_tier": dict(sorted(self.tier_counts.items())),
            "migrated": self.migrated_records,
            "ancillary": self.ancillary_files,
            "out_of_scope": self.out_of_scope_records,
            "untagged": self.untagged_records,
            "problems": len(self.problems),
        }


def load_evidence(
    root: str | Path,
    *,
    campaign: str | None = None,
    tiers: tuple[str, ...] | None = None,
) -> LoadedEvidence:
    """Load every declared result record under ``root``.

    Parameters
    ----------
    root:
        Evidence root (scanned recursively).  A missing root is a problem,
        never an empty success.
    campaign:
        When given, only records whose ``campaign_id`` equals this value are
        in scope.  Records with a different or absent campaign are counted as
        out-of-scope instead of being silently mixed.
    tiers:
        Optional tier allow-list (``("t1", "t3")``).

    Returns
    -------
    LoadedEvidence
        Records carry ``_path`` (relative), ``_tier`` and ``_identity`` in
        addition to their stored fields; the stored document itself is never
        modified on disk.
    """
    root = Path(root)
    bundle = LoadedEvidence(root=root, campaign=campaign)
    if not root.exists():
        bundle.problems.append(f"{root}: evidence root does not exist")
        return bundle
    if not root.is_dir():
        bundle.problems.append(f"{root}: evidence root is not a directory")
        return bundle

    seen_record_ids: dict[str, str] = {}
    for path in sorted(root.rglob("*.json")):
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        # Archived/superseded records never enter the canonical summary.
        if "attic" in rel_path.parts:
            continue
        tier = tier_of_path(rel_path)
        if tiers is not None and tier not in tiers:
            continue
        bundle.scanned_files += 1
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            bundle.problems.append(f"{rel}: unreadable file ({exc})")
            continue
        try:
            record = strict_json_loads(text)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            bundle.problems.append(f"{rel}: unreadable JSON ({exc})")
            continue
        if not isinstance(record, dict):
            bundle.problems.append(
                f"{rel}: not a JSON object (result records must be objects); "
                "ancillary data must be an object with its own schema id"
            )
            continue
        schema_id = record.get("schema")
        if not (isinstance(schema_id, str) and schema_id.startswith(RESULT_SCHEMA_PREFIX)):
            # Explicitly not a result record (summary/manifest/artefact).
            bundle.ancillary_files += 1
            continue
        violations = _validate_result_schema(record, rel)
        violations += _semantic_problems(record, rel)
        if violations:
            bundle.problems.extend(violations)
            continue
        if record["schema"] == RESULT_SCHEMA_V1:
            try:
                record, note = migrate_record(record)
            except (TypeError, ValueError) as exc:
                bundle.problems.append(
                    f"{rel}: legacy record could not be migrated ({exc})"
                )
                continue
            if note:
                bundle.migrated_records += 1
        if campaign is not None and record.get("campaign_id") != campaign:
            bundle.out_of_scope_records += 1
            if not record.get("campaign_id"):
                bundle.untagged_records += 1
            continue
        if campaign is None and not record.get("campaign_id"):
            bundle.untagged_records += 1
        identity_violations = _identity_problems(record, rel)
        if identity_violations:
            bundle.problems.extend(identity_violations)
            continue

        record_id = str(record["record_id"])
        if record_id in seen_record_ids:
            other = seen_record_ids[record_id]
            same_content = False
            if bundle.records:
                first = next(
                    (
                        candidate
                        for candidate in bundle.records
                        if candidate.get("record_id") == record_id
                    ),
                    None,
                )
                if first is not None:
                    same_content = record_fingerprint(first) == record_fingerprint(
                        record
                    )
            kind = "duplicate" if same_content else "conflict"
            bundle.problems.append(
                f"{rel}: {kind} record_id {record_id!r} already seen at {other}"
                + (
                    " with identical content"
                    if same_content
                    else " with different content (immutable identity collision)"
                )
            )
            continue
        seen_record_ids[record_id] = rel

        record["_path"] = rel
        record["_tier"] = tier
        record["_identity"] = normalize_identity(record)
        bundle.records.append(record)
        if tier is not None:
            bundle.tier_counts[tier] = bundle.tier_counts.get(tier, 0) + 1
    return bundle


def load_records(results_dir: Path) -> LoadedEvidence:
    """Backwards-compatible alias for :func:`load_evidence`."""
    return load_evidence(results_dir)


__all__ = [
    "IDENTITY_FIELDS",
    "LoadedEvidence",
    "TIER_NAMES",
    "load_evidence",
    "load_records",
    "normalize_identity",
    "strict_json_loads",
    "tier_of_path",
]
