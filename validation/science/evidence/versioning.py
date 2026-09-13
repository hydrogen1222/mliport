"""Version semantics for validation reports (task book PR-B).

A report must never imply that historical evidence validates the current
HEAD.  Four commits are tracked separately:

* ``software_commit``           -- the software revision being claimed validated;
* ``validation_code_commit``    -- the loader/aggregator that interpreted verdicts;
* ``report_generator_commit``   -- the renderer;
* ``evidence_source_commits``   -- the software revisions of the model runs.

``scientific_revalidation_status`` is one of:

* ``no_evidence``              -- nothing to interpret;
* ``partial_reaggregation``    -- historical evidence re-aggregated under the
  current validation semantics; NOT current-HEAD validation;
* ``current_head_revalidated`` -- every record was produced at
  ``software_commit`` and a campaign manifest declares completion.

The current-head status can only come from an explicit campaign manifest
(``mliport.beta-campaign/1``), never from the mere presence of records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import VALIDATION_LOGIC_VERSION

CAMPAIGN_MANIFEST_SCHEMA = "mliport.beta-campaign/1"
ARCHIVE_MANIFEST_SCHEMA = "mliport.beta-archive-manifest/1"

REVALIDATION_STATUSES = (
    "no_evidence",
    "partial_reaggregation",
    "current_head_revalidated",
)

CAMPAIGN_MANIFEST_FIELDS = (
    "schema",
    "campaign_id",
    "software_commit",
    "validation_commit",
    "status",
    "evidence_source_commits",
)


@dataclass(frozen=True)
class VersionBlock:
    """Report version identity."""

    software_commit: str | None
    validation_code_commit: str | None
    report_generator_commit: str | None
    evidence_campaign: str | None
    evidence_source_commits: tuple[str, ...]
    scientific_revalidation_status: str
    status_reason: str
    validation_logic_version: int = VALIDATION_LOGIC_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "software_commit": self.software_commit,
            "validation_code_commit": self.validation_code_commit,
            "report_generator_commit": self.report_generator_commit,
            "evidence_campaign": self.evidence_campaign,
            "evidence_source_commits": list(self.evidence_source_commits),
            "scientific_revalidation_status": self.scientific_revalidation_status,
            "status_reason": self.status_reason,
            "validation_logic_version": self.validation_logic_version,
        }

    @property
    def is_current_head(self) -> bool:
        return self.scientific_revalidation_status == "current_head_revalidated"


class CampaignManifestError(RuntimeError):
    """The campaign manifest is missing, malformed or inconsistent."""


def load_campaign_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a ``mliport.beta-campaign/1`` manifest."""
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"campaign manifest {manifest_path} is unreadable ({exc})"
        raise CampaignManifestError(msg) from exc
    if not isinstance(manifest, dict):
        msg = f"campaign manifest {manifest_path} must be a JSON object"
        raise CampaignManifestError(msg)
    if manifest.get("schema") != CAMPAIGN_MANIFEST_SCHEMA:
        msg = (
            f"campaign manifest {manifest_path} has schema "
            f"{manifest.get('schema')!r}; expected {CAMPAIGN_MANIFEST_SCHEMA}"
        )
        raise CampaignManifestError(msg)
    missing = [key for key in CAMPAIGN_MANIFEST_FIELDS if key not in manifest]
    if missing:
        msg = f"campaign manifest {manifest_path} misses fields {missing}"
        raise CampaignManifestError(msg)
    if manifest.get("status") not in {
        "in_progress",
        "complete",
        "partial_reaggregation",
        "superseded",
    }:
        msg = (
            f"campaign manifest {manifest_path} has unknown status "
            f"{manifest.get('status')!r}"
        )
        raise CampaignManifestError(msg)
    return manifest


def load_archive_manifest(path: str | Path) -> dict[str, Any]:
    """Load the committed evidence-archive identity (sha256 / url / records)."""
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"archive manifest {manifest_path} is unreadable ({exc})"
        raise CampaignManifestError(msg) from exc
    if not isinstance(manifest, dict):
        msg = f"archive manifest {manifest_path} must be a JSON object"
        raise CampaignManifestError(msg)
    if manifest.get("schema") != ARCHIVE_MANIFEST_SCHEMA:
        msg = (
            f"archive manifest {manifest_path} has schema "
            f"{manifest.get('schema')!r}; expected {ARCHIVE_MANIFEST_SCHEMA}"
        )
        raise CampaignManifestError(msg)
    return manifest


def build_version_block(
    records: list[dict[str, Any]],
    *,
    software_commit: str | None = None,
    validation_code_commit: str | None = None,
    report_generator_commit: str | None = None,
    evidence_campaign: str | None = None,
    campaign_manifest: dict[str, Any] | None = None,
    campaign_manifest_path: str | Path | None = None,
) -> VersionBlock:
    """Classify how far the evidence actually revalidates the current HEAD."""
    source_commits = tuple(
        sorted(
            {
                str(record.get("git_commit"))
                for record in records
                if record.get("git_commit")
            }
        )
    )
    manifest = campaign_manifest
    if manifest is None and campaign_manifest_path is not None:
        manifest = load_campaign_manifest(campaign_manifest_path)

    if manifest is not None:
        manifest_campaign = str(manifest.get("campaign_id"))
        if evidence_campaign is not None and evidence_campaign != manifest_campaign:
            msg = (
                f"campaign mismatch: loader filtered on {evidence_campaign!r} "
                f"but manifest declares {manifest_campaign!r}"
            )
            raise CampaignManifestError(msg)
        manifest_commit = str(manifest.get("software_commit"))
        if software_commit is None:
            software_commit = manifest_commit
        elif software_commit != manifest_commit:
            msg = (
                f"software_commit {software_commit!r} differs from the campaign "
                f"manifest target {manifest_commit!r}"
            )
            raise CampaignManifestError(msg)
        if evidence_campaign is None:
            evidence_campaign = manifest_campaign

    if not records:
        return VersionBlock(
            software_commit=software_commit,
            validation_code_commit=validation_code_commit,
            report_generator_commit=report_generator_commit,
            evidence_campaign=evidence_campaign,
            evidence_source_commits=(),
            scientific_revalidation_status="no_evidence",
            status_reason="no result records were loaded",
        )

    if manifest is None or software_commit is None:
        return VersionBlock(
            software_commit=software_commit,
            validation_code_commit=validation_code_commit,
            report_generator_commit=report_generator_commit,
            evidence_campaign=evidence_campaign,
            evidence_source_commits=source_commits,
            scientific_revalidation_status="partial_reaggregation",
            status_reason=(
                "historical evidence re-aggregated under the current validation "
                "semantics; no completed current-head campaign manifest was "
                "provided, so this is NOT current-HEAD validation"
            ),
        )

    manifest_status = str(manifest.get("status"))
    if manifest_status != "complete":
        return VersionBlock(
            software_commit=software_commit,
            validation_code_commit=validation_code_commit,
            report_generator_commit=report_generator_commit,
            evidence_campaign=evidence_campaign,
            evidence_source_commits=source_commits,
            scientific_revalidation_status="partial_reaggregation",
            status_reason=(
                f"campaign manifest status is {manifest_status!r}, not 'complete'"
            ),
        )

    if set(source_commits) != {software_commit}:
        msg = (
            "campaign manifest declares completion at "
            f"{software_commit!r}, but the loaded evidence was produced at "
            f"{list(source_commits)}; a current-head claim requires every "
            "record to come from the target commit"
        )
        raise CampaignManifestError(msg)

    declared = manifest.get("evidence_source_commits") or []
    undeclared = sorted(set(source_commits) - {str(commit) for commit in declared})
    if undeclared:
        msg = (
            f"evidence commits {undeclared} are not declared by the campaign "
            "manifest; unreviewed evidence cannot be part of a current-head "
            "claim"
        )
        raise CampaignManifestError(msg)

    return VersionBlock(
        software_commit=software_commit,
        validation_code_commit=validation_code_commit,
        report_generator_commit=report_generator_commit,
        evidence_campaign=evidence_campaign,
        evidence_source_commits=source_commits,
        scientific_revalidation_status="current_head_revalidated",
        status_reason=(
            "campaign manifest declares completion and every record was "
            "produced at the target software commit"
        ),
    )


__all__ = [
    "ARCHIVE_MANIFEST_SCHEMA",
    "CAMPAIGN_MANIFEST_FIELDS",
    "CAMPAIGN_MANIFEST_SCHEMA",
    "REVALIDATION_STATUSES",
    "CampaignManifestError",
    "VersionBlock",
    "build_version_block",
    "load_archive_manifest",
    "load_campaign_manifest",
]
