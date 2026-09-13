"""Profile-aware aggregation of canonical evidence records.

Aggregation is deliberately conservative: an engine-level cell is only a
single status when every profile agrees.  Mixed profiles are reported as
``mixed`` and the per-profile verdicts stay visible -- a failing float32
profile can never be compressed into a passing engine (task book PR-A).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import (
    ALLOWED_WRAPPER_MODULES,
    BETA_VALIDATION_SUITE_REVISION,
    SUMMARY_SCHEMA,
    VALIDATION_LOGIC_VERSION,
)
from .loader import LoadedEvidence
from .identity import profile_identity

ENGINE_PROPERTIES = ("energy", "forces", "stress")

#: Worst-wins ranking, used for the per-profile/run roll-up and reported as
#: ``worst_status`` next to the ``mixed`` engine verdict.
STATUS_RANK = {
    "pass": 0,
    "characterized": 1,
    "unsupported": 2,
    "insufficient_sampling": 3,
    "blocked": 4,
    "fail": 5,
}

EXIT_OK = 0
EXIT_HARNESS_VIOLATION = 1
EXIT_IMPORT_PROBLEM = 2
EXIT_NO_RECORDS = 3
EXIT_INCOMPLETE = 4


def _worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "not_run"
    return max(statuses, key=lambda status: STATUS_RANK.get(status, -1))


def _rolled_status(statuses: list[str]) -> tuple[str, str]:
    """Return ``(verdict, worst)`` for a set of profile statuses.

    A single distinct status is reported directly; anything else is ``mixed``
    so no profile can hide behind another (task book section 3.4).
    """
    distinct = sorted(set(statuses))
    if not distinct:
        return "not_run", "not_run"
    if len(distinct) == 1:
        return distinct[0], distinct[0]
    return "mixed", _worst_status(distinct)


def harness_violations(records: list[dict[str, Any]]) -> list[str]:
    """Calculator-identity checks that must never trip."""
    violations = []
    for record in records:
        diagnostics = record.get("diagnostics", {})
        wrapper_module = diagnostics.get("wrapper_module", "")
        if wrapper_module and wrapper_module not in ALLOWED_WRAPPER_MODULES:
            violations.append(
                f"{record.get('_path')}: wrapper {wrapper_module} is not an "
                "allowed mlipx backend wrapper"
            )
    return violations


def support_matrix(
    records: list[dict[str, Any]], expected_engines: tuple[str, ...] = ()
) -> dict[str, dict[str, Any]]:
    """Engine x property capability, with every profile kept visible."""
    matrix: dict[str, dict[str, Any]] = {}
    for engine in expected_engines:
        matrix[engine] = {
            prop: {
                "status": "not_run",
                "worst_status": "not_run",
                "by_profile": {},
                "profiles": [],
            }
            for prop in ENGINE_PROPERTIES
        }
    for record in records:
        if record.get("test_id") != "t1_real_inference":
            continue
        engine = record.get("engine")
        slot = matrix.setdefault(
            engine,
            {
                prop: {
                    "status": "not_run",
                    "worst_status": "not_run",
                    "by_profile": {},
                    "profiles": [],
                }
                for prop in ENGINE_PROPERTIES
            },
        )
        metrics = record.get("metrics", {})
        has_energy = "energy_eV" in metrics and metrics["energy_eV"] is not None
        has_forces = "forces_max_abs_eV_A" in metrics
        has_stress = bool(metrics.get("stress_supported"))
        profile_label = record.get("profile_id") or record.get(
            "model_identity", record.get("model_sha256", "?")
        )
        statuses = {
            "energy": record.get("status") if has_energy else None,
            "forces": record.get("status") if has_forces else None,
            "stress": record.get("status") if has_stress else None,
        }
        if has_energy and not has_stress:
            statuses["stress"] = "unsupported"
        for prop in ENGINE_PROPERTIES:
            status = statuses[prop]
            if status is None:
                continue
            entry = slot[prop]
            entry["by_profile"][profile_label] = status
            if profile_label not in entry["profiles"]:
                entry["profiles"].append(profile_label)
            verdict, worst = _rolled_status(list(entry["by_profile"].values()))
            if len(entry["by_profile"]) > 1:
                # Multiple profiles (precision/model/commit/...): a single
                # engine-level status would misrepresent the evidence.
                verdict = "mixed"
            entry["status"] = verdict
            entry["worst_status"] = worst
    for slot in matrix.values():
        for prop in ENGINE_PROPERTIES:
            slot[prop]["profiles"].sort()
    return dict(sorted(matrix.items()))


def _profiles_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for record in records:
        profile_id = record["profile_id"]
        identity = profile_identity(record)
        identity["campaign_id"] = record.get("campaign_id")
        identity["inference_mode"] = record.get("inference_mode")
        entry = profiles.setdefault(
            profile_id,
            {**identity, "profile_id": profile_id, "runs": 0, "records": []},
        )
        entry["runs"] += 1
        label = f"{record['case_id']}/{record['test_id']}"
        if label not in entry["records"]:
            entry["records"].append(label)
    for entry in profiles.values():
        entry["records"].sort()
    return dict(sorted(profiles.items()))


def _cases(records: list[dict[str, Any]]) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for record in records:
        tests = cases.setdefault(record["case_id"], {})
        buckets = tests.setdefault(record["test_id"], {})
        bucket = buckets.setdefault(
            record["profile_id"], {"status": "not_run", "worst_status": "not_run", "runs": []}
        )
        bucket["runs"].append(
            {
                "run_id": record.get("run_id"),
                "record_id": record.get("record_id"),
                "status": record.get("status"),
                "path": record.get("_path"),
                "exception": record.get("exception"),
            }
        )
    for tests in cases.values():
        for buckets in tests.values():
            for bucket in buckets.values():
                bucket["runs"].sort(
                    key=lambda run: (run["path"] or "", run["run_id"] or "")
                )
                statuses = [str(run["status"]) for run in bucket["runs"]]
                bucket["status"], bucket["worst_status"] = _rolled_status(statuses)
    return {
        case: dict(sorted(tests.items())) for case, tests in sorted(cases.items())
    }


def expected_engines_from_manifest(manifest: dict[str, Any] | None) -> tuple[str, ...]:
    if not manifest:
        return ()
    return tuple(
        sorted(
            {
                profile["engine"]
                for profile in manifest.get("profiles", {}).values()
                if profile.get("engine")
            }
        )
    )


def aggregate_records(
    records: list[dict[str, Any]],
    *,
    expected_engines: tuple[str, ...] = (),
    manifest: dict[str, Any] | None = None,
    code_commit: str | None = None,
    loader: LoadedEvidence | None = None,
) -> dict[str, Any]:
    """Build the canonical summary document for normalized records."""
    problems = list(loader.problems) if loader is not None else []
    violations = harness_violations(records)

    statuses: dict[str, int] = {}
    by_profile: dict[str, int] = {}
    for record in records:
        status = str(record.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
        by_profile[record["profile_id"]] = by_profile.get(record["profile_id"], 0) + 1

    environment: dict[str, Any] = {
        "python_versions": sorted({record.get("python") for record in records if record.get("python")}),
        "mlipx_versions": sorted(
            {record.get("mlipx_version") for record in records if record.get("mlipx_version")}
        ),
        "backend_versions": sorted(
            {
                str(value)
                for record in records
                for value in [record.get("diagnostics", {}).get("backend_version")]
                if value
            }
        ),
        "gpus": sorted(
            {
                json.dumps(record["device"], sort_keys=True)
                for record in records
                if record.get("device")
            }
        ),
        "git_commits": sorted(
            {record.get("git_commit") for record in records if record.get("git_commit")}
        ),
    }

    observed_engines = sorted({record["engine"] for record in records})
    missing_engines = sorted(set(expected_engines) - set(observed_engines))
    campaign_ids = sorted(
        {str(record.get("campaign_id")) for record in records if record.get("campaign_id")}
    )

    loader_meta: dict[str, Any] = {
        "root": str(loader.root) if loader is not None else None,
        "campaign": loader.campaign if loader is not None else None,
        "scanned_files": loader.scanned_files if loader is not None else len(records),
        "ancillary_files": loader.ancillary_files if loader is not None else 0,
        "migrated_records": loader.migrated_records if loader is not None else 0,
        "out_of_scope_records": loader.out_of_scope_records if loader is not None else 0,
        "untagged_records": loader.untagged_records if loader is not None else 0,
    }

    return {
        "schema": SUMMARY_SCHEMA,
        "suite_revision": BETA_VALIDATION_SUITE_REVISION,
        "validation_logic_version": VALIDATION_LOGIC_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": code_commit,
        "campaign_ids": campaign_ids,
        "model_manifest_suite_revision": (
            manifest.get("suite_revision") if manifest else None
        ),
        "model_profiles": (
            {
                profile_id: {
                    "identity": profile.get("identity"),
                    "model_sha256": profile.get("model_sha256"),
                    "dtype": profile.get("dtype"),
                    "task": profile.get("task"),
                    "head": profile.get("head"),
                }
                for profile_id, profile in (manifest or {}).get("profiles", {}).items()
            }
            if manifest
            else None
        ),
        "profiles": _profiles_from_records(records),
        "record_counts": {
            "total": len(records),
            "by_status": dict(sorted(statuses.items())),
            "by_profile": dict(sorted(by_profile.items())),
            "expected_engines": list(expected_engines),
        },
        "support_matrix": support_matrix(records, expected_engines),
        "cases": _cases(records),
        "environment": environment,
        "problems": problems,
        "harness_violations": violations,
        "import_summary": {
            "records": len(records),
            "migrated_records": loader_meta["migrated_records"],
            "ancillary_files": loader_meta["ancillary_files"],
            "problems": len(problems),
        },
        "loader": loader_meta,
        "expected_matrix": {
            "engines": list(expected_engines),
            "observed_engines": observed_engines,
            "missing_engines": missing_engines,
            "complete": not missing_engines,
        },
    }


def exit_code(summary: dict[str, Any], *, expect_complete: bool) -> int:
    """Documented aggregator exit contract."""
    if summary.get("problems"):
        return EXIT_IMPORT_PROBLEM
    if summary.get("harness_violations"):
        return EXIT_HARNESS_VIOLATION
    if summary.get("record_counts", {}).get("total", 0) == 0:
        return EXIT_NO_RECORDS
    if expect_complete and not summary.get("expected_matrix", {}).get("complete", True):
        return EXIT_INCOMPLETE
    return EXIT_OK


def write_summary(
    summary: dict[str, Any], out: str | Path
) -> Path:
    """Atomically write a canonical summary document."""
    import os
    import uuid

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return path


__all__ = [
    "ENGINE_PROPERTIES",
    "EXIT_HARNESS_VIOLATION",
    "EXIT_IMPORT_PROBLEM",
    "EXIT_INCOMPLETE",
    "EXIT_NO_RECORDS",
    "EXIT_OK",
    "STATUS_RANK",
    "aggregate_records",
    "exit_code",
    "expected_engines_from_manifest",
    "harness_violations",
    "support_matrix",
    "write_summary",
]
