"""VALID-01 narrative contract (task book sections 17-20).

A tier with no records in the current campaign may only render
"Not run in this campaign." — historical records are listed separately with
their campaign/commit/reason and never leak into current tier narratives.
Every GO checklist item carries a machine-readable evidence query.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "validation" / "science" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402
import generate_beta_report as report  # noqa: E402


def _device():
    return {"requested": "cpu", "actual": "cpu", "gpu_name": None, "gpu_uuid_hash": None}


def _record(*, case_id, test_id, git_commit, campaign_id):
    record = common.result_record(
        case_id=case_id,
        test_id=test_id,
        status="pass",
        engine="mace",
        model_identity="mace-toy",
        model_sha256="a" * 64,
        task="bulk",
        head=None,
        dtype="float64",
        device=_device(),
        input_structure_id=None,
        parameters={},
        metrics={"energy_eV": -1.0, "forces_max_abs_eV_A": 0.1},
        exception=None,
    )
    record["git_commit"] = git_commit
    record["campaign_id"] = campaign_id
    record["profile_id"] = common.profile_id_for(record)
    record["record_id"] = common.record_id_for(record)
    return record


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def generated_report(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    _write(
        root / "t1" / "current.json",
        _record(
            case_id="t1_case",
            test_id="t1_real_inference",
            git_commit="a" * 40,
            campaign_id="c1",
        ),
    )
    # historical, different campaign and commit: must not enter the T2 section
    _write(
        root / "t2" / "old.json",
        _record(
            case_id="t2_case",
            test_id="t2_invariance",
            git_commit="b" * 40,
            campaign_id="old-campaign",
        ),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "mliport.beta-campaign/1",
                "campaign_id": "c1",
                "software_commit": "a" * 40,
                "validation_commit": "a" * 40,
                "status": "complete",
                "evidence_source_commits": ["a" * 40],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_beta_report",
            "--evidence-root",
            str(root),
            "--out",
            str(out),
            "--campaign",
            "c1",
            "--software-commit",
            "a" * 40,
            "--campaign-manifest",
            str(manifest),
        ],
    )
    assert report.main() == 0
    return out


def _section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    stop = text.index(end, begin)
    return text[begin:stop]


def test_zero_record_tier_renders_only_not_run(generated_report):
    english = (generated_report / "BETA_VALIDATION.md").read_text(encoding="utf-8")
    t2 = _section(english, "## T2:", "## T2fd:")
    assert "Not run in this campaign." in t2
    # the historical T2 narrative must not leak into the current section
    assert "arithmetic noise" not in t2
    assert "bitwise" not in t2
    assert "| profile |" not in t2
    chinese = (generated_report / "BETA_VALIDATION_CN.md").read_text(encoding="utf-8")
    assert "Not run in this campaign." in _section(chinese, "## T2:", "## T2fd:")


def test_historical_evidence_is_listed_separately(generated_report):
    english = (generated_report / "BETA_VALIDATION.md").read_text(encoding="utf-8")
    historical = _section(
        english,
        "## Historical evidence not in this campaign",
        "## Support matrix",
    )
    assert "| t2 | 1 | old-campaign |" in historical
    assert "`bbbbbbbbbbbb...`" in historical
    assert "excluded from current tables" in historical


def test_summary_tiers_carry_explicit_scope_metadata(generated_report):
    summary = json.loads(
        (generated_report / "beta-summary.json").read_text(encoding="utf-8")
    )
    t1 = summary["tiers"]["t1"]
    assert t1["scope"] == "current_campaign"
    assert t1["records"] == 1
    assert t1["campaign_id"] == "c1"
    assert t1["software_commit"] == "a" * 40

    t2 = summary["tiers"]["t2"]
    assert t2["scope"] == "historical_reuse"
    assert t2["records"] == 0
    assert t2["historical_records"] == 1
    assert t2["historical_sources"][0]["campaign_id"] == "old-campaign"
    assert t2["historical_sources"][0]["software_commit"] == "b" * 40
    assert "not re-run" in t2["historical_sources"][0]["reason"]


def test_version_metadata_uses_target_commit_naming(generated_report):
    summary = json.loads(
        (generated_report / "beta-summary.json").read_text(encoding="utf-8")
    )
    versions = summary["versions"]
    assert versions["scientific_revalidation_status"] == "target_commit_revalidated"
    assert versions["target_software_commit"] == "a" * 40
    assert "repository_head_at_render_time" in versions
    for name in (
        "target_software_commit",
        "evidence_campaign",
        "validation_code_commit",
        "report_generator_commit",
    ):
        assert name in versions, name
    text = (generated_report / "BETA_VALIDATION.md").read_text(encoding="utf-8")
    assert "current_head_revalidated" not in text


_GO_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|(.*)\|\s*$")


def _go_rows(english: str) -> list[tuple[str, dict[str, str]]]:
    section = _section(
        english, "## Beta GO / NO-GO checklist", "## T1: inference"
    )
    rows: list[tuple[str, dict[str, str]]] = []
    for line in section.splitlines():
        match = _GO_ROW_RE.match(line)
        if not match or match.group(1) in {"#"}:
            continue
        cells = [cell.strip() for cell in match.group(2).split("|")]
        if len(cells) < 4:
            continue
        status, _evidence, query = cells[1], cells[2], cells[3].strip("`")
        title = cells[0]
        rows.append((title, {"status": status, "query": query}))
    return rows


def test_every_go_item_has_a_resolvable_evidence_query(generated_report):
    english = (generated_report / "BETA_VALIDATION.md").read_text(encoding="utf-8")
    summary = json.loads(
        (generated_report / "beta-summary.json").read_text(encoding="utf-8")
    )
    tiers = summary["tiers"]
    rows = _go_rows(english)
    assert rows, "GO checklist table not found"
    assert all(row["query"] for _title, row in rows), "GO item without query"

    current = {name for name, data in tiers.items() if data["scope"] == "current_campaign"}
    historical = {
        name for name, data in tiers.items() if data["scope"] == "historical_reuse"
    }
    for title, row in rows:
        query = row["query"]
        for match in re.finditer(r"tier=([a-z0-9]+)", query):
            tier = match.group(1)
            assert tier in tiers, f"{title}: unknown tier {tier}"
            if row["status"] == "✅":
                assert tier in current, f"{title}: tier {tier} has no current records"
        multi = re.search(r"tier_records=([a-z0-9,]+)", query)
        if multi:
            for tier in multi.group(1).split(","):
                assert tier in current, f"{title}: {tier} is not a current tier"
        hist = re.search(r"tiers=historical_reuse:([a-z0-9,]*)", query)
        if hist and hist.group(1):
            for tier in hist.group(1).split(","):
                assert tier in historical, f"{title}: {tier} is not historical"

    historical_note = _section(
        english, "## Beta GO / NO-GO checklist", "## T1: inference"
    )
    for tier in historical:
        assert tier in historical_note, tier
