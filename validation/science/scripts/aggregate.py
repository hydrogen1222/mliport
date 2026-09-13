"""Aggregate validation result records into the canonical summary JSON.

This CLI is a thin wrapper over the canonical evidence layer
(:mod:`evidence`): ``evidence.load_evidence`` is the only component that
reads raw files, validates them and turns them into normalized records, and
``evidence.aggregate_records`` is the only component that turns records into
verdicts.  Report generators, README blocks and figures consume the same
layer (task book PR-A).

Exit codes (``main``):

* ``0`` -- records imported, no problems, no harness violations
* ``1`` -- harness violation (calculator identity outside the wrappers)
* ``2`` -- import/validation problem (any malformed declared result)
* ``3`` -- no records at all (an empty set is never reported as all-pass)
* ``4`` -- ``--expect-complete`` was requested and declared evidence is missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCIENCE_ROOT = Path(__file__).resolve().parent.parent
if str(_SCIENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCIENCE_ROOT))

from evidence import (  # noqa: E402
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
    load_evidence,
    support_matrix,
)
from evidence.aggregate import write_summary  # noqa: E402
from evidence.loader import LoadedEvidence  # noqa: E402

#: Backwards-compatible name for the loader result.
LoadResult = LoadedEvidence


def strict_json_loads(text: str):
    """Backwards-compatible re-export of the strict JSON parser."""
    from evidence.loader import strict_json_loads as _strict

    return _strict(text)


def load_records(results_dir: Path) -> LoadedEvidence:
    """Load every declared result record under ``results_dir``."""
    return load_evidence(results_dir)


def build_summary(
    results_dir: Path,
    manifest_path: Path | None,
    expect_engines: tuple[str, ...] | None = None,
    *,
    campaign: str | None = None,
) -> dict:
    """Canonical summary for a results directory (compatibility wrapper)."""
    bundle = load_evidence(results_dir, campaign=campaign)
    manifest = None
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    declared = expected_engines_from_manifest(manifest)
    expected = tuple(expect_engines) if expect_engines is not None else declared
    return aggregate_records(
        bundle.records,
        expected_engines=expected,
        manifest=manifest,
        code_commit=_git_commit(),
        loader=bundle,
    )


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--campaign",
        default=None,
        help=(
            "restrict the summary to records tagged with this campaign id; "
            "records for other campaigns are counted as out-of-scope"
        ),
    )
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
        campaign=args.campaign,
    )
    out = Path(args.out)
    write_summary(summary, out)
    print(f"[aggregate] {summary['record_counts']} -> {out}")
    for violation in summary["harness_violations"]:
        print(f"[aggregate] VIOLATION: {violation}", file=sys.stderr)
    for problem in summary["problems"]:
        print(f"[aggregate] PROBLEM: {problem}", file=sys.stderr)
    if args.expect_complete and summary["expected_matrix"]["missing_engines"]:
        print(
            "[aggregate] INCOMPLETE: no records for declared engine(s) "
            f"{summary['expected_matrix']['missing_engines']}",
            file=sys.stderr,
        )
    return exit_code(summary, expect_complete=args.expect_complete)


if __name__ == "__main__":
    sys.exit(main())
