"""Canonical evidence layer for the mlipx beta validation harness.

Single source of truth for reading, validating, normalizing and aggregating
validation result records.  Report generators, README blocks, figures and the
``aggregate.py`` CLI must consume this package instead of re-implementing
record selection (task book PR-A: scientific truth has exactly one
interpretation entry point).

Public API::

    from evidence import load_evidence, aggregate_records

    bundle = load_evidence(Path(".validation-work"))
    summary = aggregate_records(bundle.records, expected_engines=("mace",))
"""

from __future__ import annotations

from .aggregate import (
    ENGINE_PROPERTIES,
    EXIT_HARNESS_VIOLATION,
    EXIT_IMPORT_PROBLEM,
    EXIT_INCOMPLETE,
    EXIT_NO_RECORDS,
    EXIT_OK,
    STATUS_RANK,
    aggregate_records,
    exit_code,
    expected_engines_from_manifest,
    harness_violations,
    support_matrix,
)
from .constants import (
    ALLOWED_WRAPPER_MODULES,
    BETA_VALIDATION_SUITE_REVISION,
    RESULT_SCHEMA,
    RESULT_SCHEMA_V1,
    RESULT_SCHEMAS,
    STATUS_SEMANTICS,
    STATUSES,
    SUMMARY_SCHEMA,
    VALIDATION_LOGIC_VERSION,
    VOLATILE_RECORD_FIELDS,
)
from .identity import (
    canonical_json,
    device_class,
    migrate_record,
    new_run_id,
    profile_id_for,
    profile_identity,
    record_fingerprint,
    record_id_for,
    seed_from,
)
from .loader import (
    LoadedEvidence,
    TIER_NAMES,
    load_evidence,
    strict_json_loads,
    tier_of_path,
)
from .versioning import (
    CAMPAIGN_MANIFEST_SCHEMA,
    REVALIDATION_STATUSES,
    CampaignManifestError,
    VersionBlock,
    build_version_block,
    load_campaign_manifest,
)

__all__ = [
    "ALLOWED_WRAPPER_MODULES",
    "BETA_VALIDATION_SUITE_REVISION",
    "CAMPAIGN_MANIFEST_SCHEMA",
    "ENGINE_PROPERTIES",
    "EXIT_HARNESS_VIOLATION",
    "EXIT_IMPORT_PROBLEM",
    "EXIT_INCOMPLETE",
    "EXIT_NO_RECORDS",
    "EXIT_OK",
    "LoadedEvidence",
    "RESULT_SCHEMA",
    "RESULT_SCHEMAS",
    "RESULT_SCHEMA_V1",
    "REVALIDATION_STATUSES",
    "STATUSES",
    "STATUS_RANK",
    "STATUS_SEMANTICS",
    "SUMMARY_SCHEMA",
    "TIER_NAMES",
    "VALIDATION_LOGIC_VERSION",
    "VOLATILE_RECORD_FIELDS",
    "CampaignManifestError",
    "VersionBlock",
    "aggregate_records",
    "build_version_block",
    "canonical_json",
    "expected_engines_from_manifest",
    "device_class",
    "exit_code",
    "harness_violations",
    "load_campaign_manifest",
    "load_evidence",
    "migrate_record",
    "new_run_id",
    "profile_id_for",
    "profile_identity",
    "record_fingerprint",
    "record_id_for",
    "seed_from",
    "strict_json_loads",
    "support_matrix",
    "tier_of_path",
]
