"""Aggregate validation result records into beta-summary.json.

Scans a results directory tree for ``*.json`` records produced by the
suite scripts, validates each against the versioned JSON Schema plus
semantic identity checks, and emits a single summary document: per-profile
status tables, the engine x property support matrix, environment/version
fingerprints and any harness integrity violations (calculator identity
outside the allowed wrapper modules).

Identity rules (review R03/R04):

* Records are aggregated per **profile** (engine x model artifact hash x
  dtype x task/head x inference mode x device class x commit x seed), never
  per engine alone, so float32 failures and float64 passes stay separately
  visible.
* File order never decides a verdict: every run is kept in ``cases`` and
  rolled-up statuses take the *worst* status, deterministically.
* Legacy ``mlipx.beta-validation-result/1`` records are migrated in memory
  (never rewritten on disk) so existing raw evidence can be re-aggregated.
* Ancillary JSON (summaries, manifests, NEB artefacts) is skipped by its
  declared schema; a file that *declares itself* a result record but fails
  validation is a problem, never a silent omission.

Exit codes (``main``):

* ``0`` -- records imported, no problems, no harness violations
* ``1`` -- harness violation (calculator identity outside the wrappers)
* ``2`` -- import problem (at least one declared result record is invalid)
* ``3`` -- no records at all (an empty set is never reported as all-pass)
* ``4`` -- ``--expect-complete`` was requested and declared evidence is missing
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402
import schema_check  # noqa: E402

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
RESULT_SCHEMA_FILES = {
    common.RESULT_SCHEMA_V1: SCHEMA_DIR / "result-v1.schema.json",
    common.RESULT_SCHEMA: SCHEMA_DIR / "result.schema.json",
}

ENGINE_PROPERTIES = ("energy", "forces", "stress")

#: Worst-wins ranking for rolled-up statuses.  A run that failed can never
#: be hidden by a later pass -- ``fail`` is always the aggregate status.
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


class LoadResult(NamedTuple):
    records: list[dict[str, Any]]
    problems: list[str]
    ancillary_files: int
    migrated_records: int


def _reject_nonfinite_token(token: str) -> None:
    msg = f"non-finite JSON constant {token!r} is not a valid result value"
    raise ValueError(msg)


def strict_json_loads(text: str) -> Any:
    """Parse JSON, rejecting NaN/Infinity constants instead of accepting them."""
    return json.loads(text, parse_constant=_reject_nonfinite_token)


def _nonfinite_paths(value: Any, path: str = "$") -> list[str]:
    """Paths of non-finite numbers anywhere in a decoded record."""
    import math

    found: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        found.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_nonfinite_paths(item, f"{path}.{key}"))
    elif isinstance(value, list | tuple):
        for i, item in enumerate(value):
            found.extend(_nonfinite_paths(item, f"{path}[{i}]"))
    return found


def _validate_record_schema(record: dict[str, Any], rel: str) -> list[str]:
    """Run the committed versioned schema for this record's schema id."""
    schema_id = record.get("schema")
    path = RESULT_SCHEMA_FILES.get(schema_id)
    if path is None:
        return [f"{rel}: unknown result schema {schema_id!r}"]
    try:
        violations = schema_check.validate_file(record, path)
    except schema_check.SchemaError as exc:
        return [f"{rel}: harness schema {path.name} is unusable ({exc})"]
    return [f"{rel}: {v}" for v in violations]


def _semantic_problems(record: dict[str, Any], rel: str) -> list[str]:
    """Checks the schema cannot express (free-form metrics/parameters trees)."""
    problems: list[str] = []
    for section in ("parameters", "metrics", "diagnostics"):
        value = record.get(section)
        if not isinstance(value, dict):
            continue  # schema already reported the type violation
        problems.extend(
            f"{rel}: non-finite number at {path}"
            for path in _nonfinite_paths(value, f"$.{section}")
        )
    if record.get("suite_revision", 0) > common.BETA_VALIDATION_SUITE_REVISION:
        problems.append(
            f"{rel}: suite_revision {record['suite_revision']} is newer than this "
            f"aggregator ({common.BETA_VALIDATION_SUITE_REVISION}); refusing to "
            "interpret future evidence"
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


def load_records(results_dir: Path) -> LoadResult:
    """Load every result record, collecting validation problems.

    A file is treated as a result record only when it declares a
    ``mlipx.beta-validation-result/*`` schema.  Ancillary JSON objects
    (summaries, manifests, NEB artefacts) are counted and skipped; every
    other file is an import problem.
    """
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    ancillary = 0
    migrated = 0
    for path in sorted(results_dir.rglob("*.json")):
        if "attic" in path.parts:
            continue  # archived/superseded records are not live evidence
        rel = str(path.relative_to(results_dir))
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{rel}: unreadable file ({exc})")
            continue
        try:
            rec = strict_json_loads(text)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            problems.append(f"{rel}: unreadable JSON ({exc})")
            continue
        if not isinstance(rec, dict):
            problems.append(
                f"{rel}: not a JSON object (result records must be objects); "
                "ancillary data must be an object with its own schema id"
            )
            continue
        schema_id = rec.get("schema")
        if not (
            isinstance(schema_id, str)
            and schema_id.startswith(common.RESULT_SCHEMA_V1.split("/")[0])
        ):
            ancillary += 1  # explicitly not a result record
            continue
        violations = _validate_record_schema(rec, rel)
        violations += _semantic_problems(rec, rel)
        if violations:
            problems.extend(violations)
            continue
        if rec["schema"] == common.RESULT_SCHEMA_V1:
            try:
                rec, note = common.migrate_record(rec)
            except (TypeError, ValueError) as exc:
                problems.append(f"{rel}: legacy record could not be migrated ({exc})")
                continue
            if note:
                migrated += 1
        rec["_path"] = rel
        records.append(rec)
    return LoadResult(records, problems, ancillary, migrated)


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


def _worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "not_run"
    return max(statuses, key=lambda s: STATUS_RANK.get(s, -1))


def support_matrix(
    records: list[dict[str, Any]], expected_engines: tuple[str, ...] = ()
) -> dict[str, dict[str, Any]]:
    """Engine x property capability as observed (never assumed).

    Every observed run of a case is kept in ``by_profile``; the rolled-up
    ``status`` is the worst status over all profiles that produced evidence
    for that property, so a float32 failure cannot be masked by a float64
    pass (review R03).
    """
    matrix: dict[str, dict[str, Any]] = {}
    for engine in expected_engines:
        matrix[engine] = {
            prop: {"status": "not_run", "by_profile": {}, "profiles": []}
            for prop in ENGINE_PROPERTIES
        }
    for rec in records:
        if rec["test_id"] != "t1_real_inference":
            continue
        eng = rec["engine"]
        slot = matrix.setdefault(
            eng,
            {
                prop: {"status": "not_run", "by_profile": {}, "profiles": []}
                for prop in ENGINE_PROPERTIES
            },
        )
        metrics = rec.get("metrics", {})
        has_e = "energy_eV" in metrics and metrics["energy_eV"] is not None
        has_f = "forces_max_abs_eV_A" in metrics
        has_s = bool(metrics.get("stress_supported"))
        profile_label = rec.get("profile_id") or rec.get(
            "model_identity", rec.get("model_sha256", "?")
        )
        statuses = {
            "energy": rec["status"] if has_e else None,
            "forces": rec["status"] if has_f else None,
            "stress": rec["status"] if has_s else None,
        }
        if has_e and not has_s:
            # observed absence: the engine ran and did not provide stress
            statuses["stress"] = "unsupported"
        for prop in ENGINE_PROPERTIES:
            status = statuses[prop]
            if status is None:
                continue
            entry = slot[prop]
            entry["by_profile"][profile_label] = status
            if profile_label not in entry["profiles"]:
                entry["profiles"].append(profile_label)
            entry["status"] = _worst_status(
                [entry["status"], status] if entry["status"] != "not_run" else [status]
            )
    for slot in matrix.values():
        for prop in ENGINE_PROPERTIES:
            slot[prop]["profiles"].sort()
    return dict(sorted(matrix.items()))


def _profiles_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for rec in records:
        pid = rec["profile_id"]
        identity = common.profile_identity(rec)
        entry = profiles.setdefault(
            pid,
            {
                **identity,
                "profile_id": pid,
                "runs": 0,
                "records": [],
            },
        )
        entry["runs"] += 1
        label = f"{rec['case_id']}/{rec['test_id']}"
        if label not in entry["records"]:
            entry["records"].append(label)
    for entry in profiles.values():
        entry["records"].sort()
    return dict(sorted(profiles.items()))


def _cases(records: list[dict[str, Any]]) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for rec in records:
        test = cases.setdefault(rec["case_id"], {}).setdefault(rec["test_id"], {})
        bucket = test.setdefault(
            rec["profile_id"],
            {"status": "not_run", "runs": []},
        )
        bucket["runs"].append(
            {
                "run_id": rec["run_id"],
                "record_id": rec["record_id"],
                "status": rec["status"],
                "path": rec.get("_path"),
                "exception": rec.get("exception"),
            }
        )
    for tests in cases.values():
        for buckets in tests.values():
            for bucket in buckets.values():
                bucket["runs"].sort(key=lambda r: (r["path"] or "", r["run_id"] or ""))
                bucket["status"] = _worst_status([r["status"] for r in bucket["runs"]])
    return {case: dict(sorted(tests.items())) for case, tests in sorted(cases.items())}


def expected_engines_from_manifest(manifest: dict[str, Any] | None) -> tuple[str, ...]:
    if not manifest:
        return ()
    return tuple(
        sorted(
            {
                p["engine"]
                for p in manifest.get("profiles", {}).values()
                if p.get("engine")
            }
        )
    )


def build_summary(
    results_dir: Path,
    manifest_path: Path | None,
    expect_engines: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    loaded = load_records(results_dir)
    records = loaded.records
    problems = list(loaded.problems)
    violations = harness_violations(records)

    manifest = None
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    declared_engines = expected_engines_from_manifest(manifest)
    expected = tuple(expect_engines) if expect_engines is not None else declared_engines

    statuses: dict[str, int] = {}
    by_profile: dict[str, int] = {}
    for rec in records:
        statuses[rec["status"]] = statuses.get(rec["status"], 0) + 1
        by_profile[rec["profile_id"]] = by_profile.get(rec["profile_id"], 0) + 1

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
        "git_commits": sorted({rec["git_commit"] for rec in records}),
    }

    observed_engines = sorted({rec["engine"] for rec in records})
    missing_engines = sorted(set(expected) - set(observed_engines))
    expected_matrix = {
        "engines": list(expected),
        "observed_engines": observed_engines,
        "missing_engines": missing_engines,
        "complete": not missing_engines,
    }

    return {
        "schema": common.SUMMARY_SCHEMA,
        "suite_revision": common.BETA_VALIDATION_SUITE_REVISION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
        "profiles": _profiles_from_records(records),
        "record_counts": {
            "total": len(records),
            "by_status": dict(sorted(statuses.items())),
            "by_profile": dict(sorted(by_profile.items())),
            "expected_engines": list(expected),
        },
        "support_matrix": support_matrix(records, expected),
        "cases": _cases(records),
        "environment": env,
        "problems": problems,
        "harness_violations": violations,
        "import_summary": {
            "records": len(records),
            "migrated_records": loaded.migrated_records,
            "ancillary_files": loaded.ancillary_files,
            "problems": len(problems),
        },
        "expected_matrix": expected_matrix,
    }


def exit_code(summary: dict[str, Any], *, expect_complete: bool) -> int:
    """Map the summary to the documented aggregator exit contract."""
    if summary.get("problems"):
        return EXIT_IMPORT_PROBLEM
    if summary.get("harness_violations"):
        return EXIT_HARNESS_VIOLATION
    if summary.get("record_counts", {}).get("total", 0) == 0:
        return EXIT_NO_RECORDS
    if expect_complete and not summary.get("expected_matrix", {}).get("complete", True):
        return EXIT_INCOMPLETE
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--expect-complete",
        action="store_true",
        help=(
            "exit 4 unless every engine declared by the manifest produced at "
            "least one record (missing evidence is never a pass)"
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    summary = build_summary(
        results_dir,
        Path(args.manifest) if args.manifest else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[aggregate] {summary['record_counts']} -> {out}")
    for v in summary["harness_violations"]:
        print(f"[aggregate] VIOLATION: {v}", file=sys.stderr)
    for p in summary["problems"]:
        print(f"[aggregate] PROBLEM: {p}", file=sys.stderr)
    if args.expect_complete and summary["expected_matrix"]["missing_engines"]:
        print(
            "[aggregate] INCOMPLETE: no records for declared engine(s) "
            f"{summary['expected_matrix']['missing_engines']}",
            file=sys.stderr,
        )
    return exit_code(summary, expect_complete=args.expect_complete)


if __name__ == "__main__":
    sys.exit(main())
