"""Aggregate validation result records into beta-summary.json.

Scans a results directory tree for ``*.json`` records produced by the
suite scripts, validates each against the result schema, and emits a
single summary document: per-case status tables, the engine x property
support matrix, environment/version fingerprints and any harness
integrity violations (calculator identity outside the allowed wrapper
modules).  The aggregator never rewrites statuses and never drops
failures -- bad results stay visible (taskbook section 54).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402

REQUIRED_FIELDS = (
    "schema",
    "suite_revision",
    "git_commit",
    "case_id",
    "test_id",
    "status",
    "engine",
    "model_identity",
    "model_sha256",
    "task",
    "head",
    "dtype",
    "python",
    "mlipx_version",
    "device",
    "parameters",
    "metrics",
    "diagnostics",
    "exception",
)

ENGINE_PROPERTIES = ("energy", "forces", "stress")


def load_records(results_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load every result record, collecting validation problems."""
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{path}: unreadable JSON ({exc})")
            continue
        if rec.get("schema") != common.RESULT_SCHEMA:
            continue  # not a suite record (e.g. summary/manifest files)
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            problems.append(f"{path}: missing fields {missing}")
            continue
        if rec["status"] not in common.STATUSES:
            problems.append(f"{path}: invalid status {rec['status']!r}")
            continue
        rec["_path"] = str(path.relative_to(results_dir))
        records.append(rec)
    return records, problems


def harness_violations(records: list[dict[str, Any]]) -> list[str]:
    """Identity checks that must never trip (taskbook section 47)."""
    violations = []
    for rec in records:
        diag = rec.get("diagnostics", {})
        wrapper_mod = diag.get("wrapper_module", "")
        if wrapper_mod and wrapper_mod not in common.ALLOWED_WRAPPER_MODULES:
            violations.append(
                f"{rec.get('_path')}: wrapper {wrapper_mod} is not an allowed "
                "mlipx backend wrapper"
            )
    return violations


def support_matrix(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Engine x property capability as observed (never assumed)."""
    matrix: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec["test_id"] != "t1_real_inference":
            continue
        eng = rec["engine"]
        slot = matrix.setdefault(
            eng,
            {
                prop: {
                    "status": "not_run",
                    "profiles": [],
                }
                for prop in ENGINE_PROPERTIES
            },
        )
        metrics = rec.get("metrics", {})
        has_e = "energy_eV" in metrics and metrics["energy_eV"] is not None
        has_f = "forces_max_abs_eV_A" in metrics
        has_s = bool(metrics.get("stress_supported"))
        profile_label = rec.get("model_identity", rec.get("model_sha256", "?"))
        if has_e:
            slot["energy"]["profiles"].append(profile_label)
            slot["energy"]["status"] = rec["status"]
        if has_f:
            slot["forces"]["profiles"].append(profile_label)
            slot["forces"]["status"] = rec["status"]
        if has_s:
            slot["stress"]["profiles"].append(profile_label)
            slot["stress"]["status"] = rec["status"]
        elif has_e:
            if slot["stress"]["status"] in ("not_run", "pass"):
                slot["stress"]["status"] = "unsupported"
    return matrix


def build_summary(results_dir: Path, manifest_path: Path) -> dict[str, Any]:
    records, problems = load_records(results_dir)
    violations = harness_violations(records)
    statuses: dict[str, int] = {}
    for rec in records:
        statuses[rec["status"]] = statuses.get(rec["status"], 0) + 1

    cases: dict[str, dict[str, Any]] = {}
    for rec in records:
        case = cases.setdefault(rec["case_id"], {})
        case.setdefault(rec["test_id"], {})[f"{rec['engine']}"] = {
            "status": rec["status"],
            "dtype": rec["dtype"],
            "model_identity": rec["model_identity"],
            "path": rec.get("_path"),
        }

    env: dict[str, Any] = {
        "python_versions": sorted({rec["python"] for rec in records}),
        "mlipx_versions": sorted({rec["mlipx_version"] for rec in records}),
        "backend_versions": sorted(
            {
                str(v)
                for rec in records
                for v in [rec.get("diagnostics", {}).get("backend_version")]
                if v
            }
        ),
        "gpus": sorted(
            {
                json.dumps(rec["device"], sort_keys=True)
                for rec in records
                if rec.get("device")
            }
        ),
    }

    manifest = None
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    return {
        "schema": common.SUMMARY_SCHEMA,
        "suite_revision": common.BETA_VALIDATION_SUITE_REVISION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_commit": common._git_commit(),
        "model_manifest_suite_revision": (
            manifest.get("suite_revision") if manifest else None
        ),
        "model_profiles": (
            {
                pid: {
                    "identity": p.get("identity"),
                    "model_sha256": p.get("model_sha256"),
                    "dtype": p.get("dtype"),
                    "task": p.get("task"),
                    "head": p.get("head"),
                }
                for pid, p in (manifest or {}).get("profiles", {}).items()
            }
            if manifest
            else None
        ),
        "record_counts": {"total": len(records), "by_status": statuses},
        "support_matrix": support_matrix(records),
        "cases": cases,
        "environment": env,
        "problems": problems,
        "harness_violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results)
    summary = build_summary(
        results_dir,
        Path(args.manifest) if args.manifest else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[aggregate] {summary['record_counts']} -> {out}")
    for v in summary["harness_violations"]:
        print(f"[aggregate] VIOLATION: {v}", file=sys.stderr)
    for p in summary["problems"]:
        print(f"[aggregate] PROBLEM: {p}", file=sys.stderr)
    return 1 if summary["harness_violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
