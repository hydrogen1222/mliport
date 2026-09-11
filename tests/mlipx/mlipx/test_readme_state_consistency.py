"""README verification-state consistency (release-candidate audit RC-03).

The README runtime-validation table may only claim ``passed`` for runtimes
whose sanitized records exist in the repository *and* whose primary evidence
is promoted in the curated capability registry. The compatibility table may
only use the two honest install-route states from
``mlipx/install/compatibility.py`` (``experimental`` / ``needs runtime smoke
test``). This guards against documentation drift when a runtime contract
changes without revalidation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

RUNTIME_TABLE_HEADER = "| Backend / model | SP | Short MD | E–F gradient | NEB smoke | GRACE cache | Records |"
STATUS_WORKLOADS = (
    "single_point",
    "shared_calculator",
    "energy_force_consistency",
    "short_md",
    "neb",
)
_ALLOWED_COMPAT_STATES = {"experimental", "needs runtime smoke test"}


def _runtime_table_rows() -> list[list[str]]:
    """Rows of the README runtime-validation table (cells, links intact)."""
    lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    start = lines.index(RUNTIME_TABLE_HEADER)
    rows = []
    for line in lines[start + 2 :]:  # skip the |---| separator row
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _registry_evidence_ids() -> set[str]:
    ids: set[str] = set()
    for path in sorted((REPO_ROOT / "mlipx/mlipx/data/validation").glob("*.json")):
        record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        ids.add(record["evidence_id"])
    return ids


def test_runtime_table_exists() -> None:
    assert _runtime_table_rows(), "README runtime-validation table missing"


def test_runtime_table_rows_are_backed_by_records() -> None:
    registry_ids = _registry_evidence_ids()
    assert registry_ids, "curated registry is empty"

    for row in _runtime_table_rows():
        label = row[0]
        statuses = row[1:6]
        record_links = re.findall(
            r"\]\((validation/runtime/v100/[^)]+)\)", " | ".join(row)
        )
        has_passed = "passed" in statuses

        if has_passed:
            assert record_links, f"{label} claims passed without records"
            # The primary record must exist, be passed on every workload, and
            # be promoted in the curated registry (RC-03: no doc-only truth).
            primary = REPO_ROOT / record_links[0]
            assert primary.exists(), f"{label}: missing record {record_links[0]}"
            record = json.loads(primary.read_text(encoding="utf-8"))
            assert record["status"] == "passed", f"{label}: record not passed"
            for workload in STATUS_WORKLOADS:
                workload_status = record["workloads"][workload]["status"]
                assert (
                    workload_status == "passed"
                ), f"{label}: workload {workload} is {workload_status!r}"
            assert record["evidence_id"] in registry_ids, (
                f"{label}: evidence_id {record['evidence_id']} is not in the "
                f"curated capability registry"
            )
        else:
            assert "not run" in statuses, f"{label}: unexpected status row {statuses}"

        # Every linked record must exist and be consistent with the claim.
        for link in record_links:
            record_path = REPO_ROOT / link
            assert record_path.exists(), f"{label}: missing record {link}"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if has_passed:
                assert record["status"] == "passed", f"{label}: {link} not passed"
            else:
                assert record["status"] != "passed", f"{label}: {link} passed?!"


def test_uma_stays_fail_closed_in_readme() -> None:
    uma_row = next(row for row in _runtime_table_rows() if row[0].startswith("UMA"))
    assert all(status == "not run" for status in uma_row[1:5])
    assert uma_row[5] == "n/a"  # cache column not applicable


def test_compatibility_table_does_not_claim_verified() -> None:
    lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    start = lines.index(
        "| Engine | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | Other Turing+ | Hopper / Blackwell |"
    )
    compat_rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        compat_rows.append(line)
    assert compat_rows, "compatibility table missing"
    for line in compat_rows:
        states = {cell.strip().strip("*") for cell in line.strip("|").split("|")[1:]}
        unexpected = states - _ALLOWED_COMPAT_STATES
        assert not unexpected, (
            f"compatibility row claims non-install-route states {unexpected}: "
            f"{line}"
        )
