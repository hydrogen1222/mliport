"""PR-A acceptance tests: one canonical evidence pipeline.

These tests pin the contract from the e579937 revalidation task book
(section 3.5, V01-T1..T10): one strict loader, profile-aware aggregation,
fail-closed reports, and no report-side record selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "validation" / "science" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate  # noqa: E402
import common  # noqa: E402
import generate_beta_report as report  # noqa: E402
from evidence import aggregate_records, load_evidence  # noqa: E402


def _device(actual="cpu"):
    return {
        "requested": actual,
        "actual": actual,
        "gpu_name": None,
        "gpu_uuid_hash": None,
    }


def _record(
    *,
    case_id="t1_case",
    test_id="t1_real_inference",
    engine="mace",
    dtype="float64",
    status="pass",
    model_sha256="a" * 64,
    git_commit=None,
    exception=None,
    metrics=None,
):
    record = common.result_record(
        case_id=case_id,
        test_id=test_id,
        status=status,
        engine=engine,
        model_identity=f"{engine}-toy",
        model_sha256=model_sha256,
        task="bulk",
        head=None,
        dtype=dtype,
        device=_device(),
        input_structure_id=None,
        parameters={},
        metrics=metrics
        or {
            "energy_eV": -1.0,
            "forces_max_abs_eV_A": 0.1,
            "stress_supported": False,
        },
        exception=exception,
    )
    if git_commit is not None:
        record["git_commit"] = git_commit
        record["profile_id"] = common.profile_id_for(record)
        record["record_id"] = common.record_id_for(record)
    return record


def _write(directory: Path, name: str, payload) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _strip_run_paths(cases):
    """Drop the per-run file path, which legitimately differs between roots."""
    import copy

    stripped = copy.deepcopy(cases)
    for tests in stripped.values():
        for buckets in tests.values():
            for bucket in buckets.values():
                for run in bucket["runs"]:
                    run.pop("path", None)
    return stripped


def _summary_without_timestamp(summary):
    return {key: value for key, value in summary.items() if key != "generated_at"}


# --------------------------------------------------------------- V01-T1
def test_v01_t1_input_order_does_not_change_summary(tmp_path):
    records = [
        _record(dtype="float32", status="fail", exception="overflow"),
        _record(dtype="float64", status="pass"),
    ]
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    for index, record in enumerate(records):
        _write(dir_a, f"{index}.json", record)
    for index, record in enumerate(reversed(records)):
        _write(dir_b, f"zz_{index}.json", record)
    first = _summary_without_timestamp(aggregate.build_summary(dir_a, None))
    second = _summary_without_timestamp(aggregate.build_summary(dir_b, None))
    for key in ("record_counts", "support_matrix", "profiles"):
        assert first[key] == second[key], key
    assert _strip_run_paths(first["cases"]) == _strip_run_paths(second["cases"])
    assert first["problems"] == second["problems"] == []


# --------------------------------------------------------------- V01-T2
def test_v01_t2_float32_fail_is_not_hidden_by_float64_pass(tmp_path):
    _write(
        tmp_path,
        "fail32.json",
        _record(dtype="float32", status="fail", exception="overflow"),
    )
    _write(tmp_path, "pass64.json", _record(dtype="float64", status="pass"))
    summary = aggregate.build_summary(tmp_path, None)
    matrix = summary["support_matrix"]["mace"]["energy"]
    assert matrix["status"] == "mixed"
    assert matrix["worst_status"] == "fail"
    assert sorted(matrix["by_profile"].values()) == ["fail", "pass"]
    buckets = summary["cases"]["t1_case"]["t1_real_inference"]
    assert {bucket["status"] for bucket in buckets.values()} == {"fail", "pass"}


# --------------------------------------------------------------- V01-T3
def test_v01_t3_different_model_hash_is_a_different_profile(tmp_path):
    _write(tmp_path, "a.json", _record(model_sha256="a" * 64))
    _write(tmp_path, "b.json", _record(model_sha256="b" * 64))
    summary = aggregate.build_summary(tmp_path, None)
    matrix = summary["support_matrix"]["mace"]["energy"]
    assert matrix["status"] == "mixed"
    assert len(matrix["by_profile"]) == 2
    assert len(summary["profiles"]) == 2
    for entry in summary["profiles"].values():
        assert entry["model_sha256"] in {"a" * 64, "b" * 64}


# --------------------------------------------------------------- V01-T4
def test_v01_t4_different_software_commit_is_a_different_profile(tmp_path):
    _write(tmp_path, "old.json", _record(git_commit="1" * 40))
    _write(tmp_path, "new.json", _record(git_commit="2" * 40))
    summary = aggregate.build_summary(tmp_path, None)
    profiles = summary["profiles"]
    assert len(profiles) == 2
    assert sorted(entry["git_commit"] for entry in profiles.values()) == [
        "1" * 40,
        "2" * 40,
    ]
    # both profiles stay visible in the environment fingerprint
    assert summary["environment"]["git_commits"] == ["1" * 40, "2" * 40]


# --------------------------------------------------------------- V01-T5
def test_v01_t5_malformed_declared_result_makes_report_fail(tmp_path, monkeypatch):
    _write(tmp_path, "good.json", _record())
    _write(tmp_path, "bad.json", "{not valid json")
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_beta_report",
            "--evidence-root",
            str(tmp_path),
            "--out",
            str(out),
        ],
    )
    assert report.main() == 2
    assert not (out / "BETA_VALIDATION.md").exists()


# --------------------------------------------------------------- V01-T6
def test_v01_t6_sidecars_are_recognised_explicitly(tmp_path):
    _write(tmp_path, "good.json", _record())
    _write(
        tmp_path,
        "summary.json",
        {"schema": common.SUMMARY_SCHEMA, "record_counts": {"total": 1}},
    )
    loaded = load_evidence(tmp_path)
    assert loaded.problems == []
    assert len(loaded.records) == 1
    assert loaded.ancillary_files == 1

    # a JSON file that fails to parse is a problem, never assumed to be a sidecar
    _write(tmp_path, "broken.json", "{oops")
    loaded = load_evidence(tmp_path)
    assert any("unreadable JSON" in problem for problem in loaded.problems)


# --------------------------------------------------------------- V01-T7
def test_v01_t7_empty_evidence_root_is_not_success(tmp_path, monkeypatch):
    loaded = load_evidence(tmp_path)
    assert loaded.records == []
    assert loaded.problems == []
    summary = aggregate_records([], loader=loaded)
    assert aggregate.exit_code(summary, expect_complete=False) == 3

    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_beta_report",
            "--evidence-root",
            str(tmp_path),
            "--out",
            str(out),
        ],
    )
    assert report.main() == 3
    assert not (out / "BETA_VALIDATION.md").exists()


# --------------------------------------------------------------- V01-T8/T9
def test_v01_t8_duplicate_record_id_is_a_problem(tmp_path):
    record = _record()
    _write(tmp_path, "a.json", record)
    _write(tmp_path, "b.json", record)  # same computation identity and content
    loaded = load_evidence(tmp_path)
    assert len(loaded.records) == 1
    assert any("duplicate record_id" in problem for problem in loaded.problems)


def test_v01_t9_immutable_identity_collision_is_a_problem(tmp_path):
    first = _record()
    second = _record()
    second["metrics"] = dict(second["metrics"], energy_eV=-2.0)
    # same record_id (parameters/identity unchanged) but different content
    assert first["record_id"] == second["record_id"]
    _write(tmp_path, "a.json", first)
    _write(tmp_path, "b.json", second)
    loaded = load_evidence(tmp_path)
    assert any("immutable identity collision" in problem for problem in loaded.problems)


# --------------------------------------------------------------- V01-T10
def test_v01_t10_attic_never_enters_the_summary(tmp_path):
    _write(tmp_path, "t1/live.json", _record())
    _write(tmp_path, "t1/attic/old.json", _record())
    loaded = load_evidence(tmp_path)
    assert len(loaded.records) == 1
    assert loaded.problems == []
    assert loaded.records[0]["_path"] == "t1/live.json"


# ------------------------------------------------------- single pipeline
def test_report_generator_has_no_independent_record_selection():
    source = (SCRIPTS / "generate_beta_report.py").read_text(encoding="utf-8")
    assert '*.json"' not in source, "report generator must not glob raw evidence"
    assert "json.loads(" not in source, "report generator must not parse records"
    assert "_passing(" not in source, "prefer-pass selection is forbidden"
    assert "load_evidence(" in source

    figures = (SCRIPTS / "generate_figures.py").read_text(encoding="utf-8")
    assert '*.json"' not in figures
    assert "json.loads(" not in figures
    assert "load_evidence(" in figures


def test_canonical_counts_match_between_loader_summary_and_report(tmp_path):
    _write(tmp_path, "t1/a.json", _record())
    _write(tmp_path, "t1/b.json", _record(dtype="float32", status="fail"))
    bundle = load_evidence(tmp_path)
    canonical = aggregate_records(bundle.records, loader=bundle)
    tiers = {tier: bundle.tier(tier) for tier in ("t1",)}
    assert canonical["record_counts"]["total"] == len(bundle.records) == 2
    assert sum(len(records) for records in tiers.values()) == 2
    # the same rule is applied by the report-side support matrix
    matrix = report.build_support_matrix({"t1": bundle.tier("t1")})
    entry = matrix["single_point"]["mace"]
    assert entry["records"] == 2
    assert entry["by_status"]["fail"] == 1


# ------------------------------------------------- campaign scope (V03)
def test_campaign_scope_excludes_untagged_and_other_campaigns(tmp_path):
    tagged = _record()
    tagged["campaign_id"] = "campaign-a"
    other = _record(dtype="float32", status="pass")
    other["campaign_id"] = "campaign-b"
    untagged = _record(dtype="float64", status="pass")
    untagged["campaign_id"] = None
    for name, record in (("a.json", tagged), ("b.json", other), ("c.json", untagged)):
        # distinct computation identities so the scope, not dedup, is tested
        record["parameters"] = {"tag": name}
        record["record_id"] = common.record_id_for(record)
        _write(tmp_path, name, record)

    all_records = load_evidence(tmp_path)
    assert len(all_records.records) == 3
    assert all_records.untagged_records == 1

    scoped = load_evidence(tmp_path, campaign="campaign-a")
    assert [record["campaign_id"] for record in scoped.records] == ["campaign-a"]
    assert scoped.out_of_scope_records == 2
    assert scoped.untagged_records == 1
    assert scoped.problems == []


def test_report_embeds_the_canonical_aggregate_verbatim(tmp_path, monkeypatch):
    """beta-summary.json must not re-derive counts/verdicts differently."""
    _write(tmp_path / "t1", "pass.json", _record())
    _write(
        tmp_path / "t1",
        "fail.json",
        _record(dtype="float32", status="fail", exception="overflow"),
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_beta_report",
            "--evidence-root",
            str(tmp_path),
            "--out",
            str(out),
        ],
    )
    assert report.main() == 0
    beta = json.loads((out / "beta-summary.json").read_text(encoding="utf-8"))
    canonical = beta["canonical_summary"]
    bundle = load_evidence(tmp_path)
    direct = aggregate_records(bundle.records, loader=bundle)
    assert canonical["record_counts"] == direct["record_counts"]
    assert canonical["profiles"] == direct["profiles"]
    assert canonical["support_matrix"] == direct["support_matrix"]
    assert canonical["expected_matrix"] == direct["expected_matrix"]


def test_campaign_id_is_part_of_the_profile_identity(tmp_path):
    first = _record()
    first["campaign_id"] = "campaign-a"
    first["profile_id"] = common.profile_id_for(first)
    first["record_id"] = common.record_id_for(first)
    second = _record()
    second["campaign_id"] = "campaign-b"
    second["profile_id"] = common.profile_id_for(second)
    second["record_id"] = common.record_id_for(second)
    assert first["profile_id"] != second["profile_id"]
    _write(tmp_path, "a.json", first)
    _write(tmp_path, "b.json", second)
    summary = aggregate.build_summary(tmp_path, None)
    assert summary["support_matrix"]["mace"]["energy"]["status"] == "mixed"
    assert len(summary["profiles"]) == 2


# ------------------------------------------------- PR-B version semantics
from evidence import (  # noqa: E402
    CampaignManifestError,
    build_version_block,
)


def _campaign_manifest(tmp_path, *, campaign_id, status, commit, sources):
    path = tmp_path / "campaign.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "mliport.beta-campaign/1",
                "campaign_id": campaign_id,
                "software_commit": commit,
                "validation_commit": commit,
                "status": status,
                "evidence_source_commits": sources,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prb_historical_evidence_is_partial_reaggregation(tmp_path):
    record = _record(git_commit="a" * 40)
    versions = build_version_block(
        [record],
        software_commit=None,
        validation_code_commit="v" * 40,
        report_generator_commit="v" * 40,
    )
    assert versions.scientific_revalidation_status == "partial_reaggregation"
    assert versions.is_current_head is False
    snippet = report.readme_validation_snippet(
        report.build_support_matrix({}), [], versions.as_dict()
    )
    assert "post-fix beta candidate" in snippet
    assert "Current-HEAD scientific revalidation is pending" in snippet
    assert "beta validation completed" not in snippet


def test_prb_complete_manifest_rejects_mismatched_evidence(tmp_path):
    record = _record(git_commit="a" * 40)
    manifest = _campaign_manifest(
        tmp_path,
        campaign_id="c1",
        status="complete",
        commit="b" * 40,
        sources=["b" * 40],
    )
    with pytest.raises(CampaignManifestError, match="every record"):
        build_version_block(
            [record],
            validation_code_commit="v" * 40,
            evidence_campaign="c1",
            campaign_manifest_path=manifest,
        )


def test_prb_current_head_status_only_with_complete_campaign(tmp_path):
    record = _record(git_commit="a" * 40)
    complete = _campaign_manifest(
        tmp_path,
        campaign_id="c1",
        status="complete",
        commit="a" * 40,
        sources=["a" * 40],
    )
    versions = build_version_block(
        [record],
        validation_code_commit="v" * 40,
        evidence_campaign="c1",
        campaign_manifest_path=complete,
    )
    assert versions.scientific_revalidation_status == "current_head_revalidated"
    snippet = report.readme_validation_snippet(
        report.build_support_matrix({}), [], versions.as_dict()
    )
    assert "beta validation completed at software commit" in snippet

    in_progress = _campaign_manifest(
        tmp_path / "in_progress",
        campaign_id="c1",
        status="in_progress",
        commit="a" * 40,
        sources=["a" * 40],
    )
    partial = build_version_block(
        [record],
        validation_code_commit="v" * 40,
        evidence_campaign="c1",
        campaign_manifest_path=in_progress,
    )
    assert partial.scientific_revalidation_status == "partial_reaggregation"


def test_prb_generator_writes_version_block_and_pending_status(tmp_path, monkeypatch):
    tagged = _record()
    tagged["campaign_id"] = "c1"
    tagged["profile_id"] = common.profile_id_for(tagged)
    tagged["record_id"] = common.record_id_for(tagged)
    _write(tmp_path / "t1", "a.json", tagged)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_beta_report",
            "--evidence-root",
            str(tmp_path),
            "--out",
            str(out),
        ],
    )
    assert report.main() == 0
    beta = json.loads((out / "beta-summary.json").read_text(encoding="utf-8"))
    assert beta["versions"]["scientific_revalidation_status"] == (
        "partial_reaggregation"
    )
    snippet = (out / "README_VALIDATION.md").read_text(encoding="utf-8")
    assert "Current-HEAD scientific revalidation is pending" in snippet
    assert "beta validation completed" not in snippet

    # a complete manifest for the wrong commit must make the render fail closed
    manifest = _campaign_manifest(
        tmp_path / "manifest",
        campaign_id="c1",
        status="complete",
        commit="b" * 40,
        sources=["b" * 40],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_beta_report",
            "--evidence-root",
            str(tmp_path),
            "--out",
            str(out),
            "--campaign",
            "c1",
            "--campaign-manifest",
            str(manifest),
        ],
    )
    assert report.main() == 2


def test_prb_generator_allows_go_wording_only_for_complete_campaign(
    tmp_path, monkeypatch
):
    record = _record(git_commit="a" * 40)
    record["campaign_id"] = "c1"
    record["profile_id"] = common.profile_id_for(record)
    record["record_id"] = common.record_id_for(record)
    _write(tmp_path / "t1", "a.json", record)
    manifest = _campaign_manifest(
        tmp_path / "manifest",
        campaign_id="c1",
        status="complete",
        commit="a" * 40,
        sources=["a" * 40],
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_beta_report",
            "--evidence-root",
            str(tmp_path),
            "--out",
            str(out),
            "--campaign",
            "c1",
            "--campaign-manifest",
            str(manifest),
        ],
    )
    assert report.main() == 0
    beta = json.loads((out / "beta-summary.json").read_text(encoding="utf-8"))
    versions = beta["versions"]
    assert versions["scientific_revalidation_status"] == "current_head_revalidated"
    assert versions["software_commit"] == "a" * 40
    assert versions["evidence_campaign"] == "c1"
    snippet = (out / "README_VALIDATION.md").read_text(encoding="utf-8")
    assert "beta validation completed at software commit" in snippet


# ------------------------------------------------- PR-I schema hardening
def test_pr_i_nested_unknown_keyword_fails_at_schema_load(tmp_path):
    """The task book's misspelled `minLenght` inside an optional property."""
    schema_path = tmp_path / "broken.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "optional_never_present": {
                        "type": "string",
                        "minLenght": 5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    from evidence.schema import SchemaDefinitionError, load_schema

    with pytest.raises(SchemaDefinitionError, match="minLenght"):
        load_schema(schema_path)


def test_pr_i_unknown_keyword_in_items_and_additional_properties(tmp_path):
    from evidence.schema import SchemaDefinitionError, load_schema

    items_path = tmp_path / "items.schema.json"
    items_path.write_text(
        json.dumps({"type": "array", "items": {"type": "string", "pattrn": "x"}}),
        encoding="utf-8",
    )
    with pytest.raises(SchemaDefinitionError, match="pattrn"):
        load_schema(items_path)

    additional_path = tmp_path / "additional.schema.json"
    additional_path.write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": {"type": "number", "miximum": 1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SchemaDefinitionError, match="miximum"):
        load_schema(additional_path)


def test_pr_i_committed_schemas_audit_clean_and_loader_fails_closed(
    tmp_path, monkeypatch
):
    import evidence.schema as schema_module

    for schema_file in sorted(schema_module.SCHEMA_DIR.glob("*.json")):
        schema_module.load_schema(schema_file)
    assert issubclass(schema_module.SchemaDefinitionError, schema_module.SchemaError)

    # A broken schema file must make the loader report a problem, not pass.
    good = _record()
    _write(tmp_path, "t1/a.json", good)
    broken_schema = tmp_path / "broken.json"
    broken_schema.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"unused": {"type": "string", "minLenght": 5}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        schema_module,
        "result_schema_path",
        lambda _schema_id: broken_schema,
    )
    from evidence import load_evidence

    loaded = load_evidence(tmp_path)
    assert loaded.records == []
    assert any("unusable" in problem for problem in loaded.problems)


# --------------------------------------- single consolidated report (PR6)
def test_archive_manifest_loader_is_fail_closed(tmp_path):
    from evidence import load_archive_manifest
    from evidence.versioning import CampaignManifestError

    def _write(payload):
        path = tmp_path / "archive_manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    valid = _write(
        {
            "schema": "mliport.beta-archive-manifest/1",
            "campaign_id": "c1",
            "archive_sha256": "a" * 64,
            "archive_url": "https://example.invalid/archive.tar.zst",
            "records": 1,
        }
    )
    assert load_archive_manifest(valid)["campaign_id"] == "c1"
    with pytest.raises(CampaignManifestError, match="schema"):
        load_archive_manifest(_write({"schema": "wrong"}))


def test_report_embeds_go_checklist_in_one_document(tmp_path, monkeypatch):
    software_commit = "a" * 40
    monkeypatch.setenv(common.SOFTWARE_COMMIT_ENV, software_commit)
    record = _record()
    record["campaign_id"] = "camp-consolidated"
    record["profile_id"] = common.profile_id_for(record)
    record["record_id"] = common.record_id_for(record)
    _write(tmp_path / "evidence" / "t1", "a.json", record)

    campaign_manifest = tmp_path / "campaign.json"
    campaign_manifest.write_text(
        json.dumps(
            {
                "schema": "mliport.beta-campaign/1",
                "campaign_id": "camp-consolidated",
                "software_commit": software_commit,
                "validation_commit": "b" * 40,
                "status": "complete",
                "evidence_source_commits": [software_commit],
            }
        ),
        encoding="utf-8",
    )
    archive_manifest = tmp_path / "archive_manifest.json"
    archive_manifest.write_text(
        json.dumps(
            {
                "schema": "mliport.beta-archive-manifest/1",
                "campaign_id": "camp-consolidated",
                "archive_sha256": "c" * 64,
                "archive_url": "https://example.invalid/archive.tar.zst",
                "archive_bytes": 123,
                "records": 1,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_beta_report",
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--campaign",
            "camp-consolidated",
            "--campaign-manifest",
            str(campaign_manifest),
            "--archive-manifest",
            str(archive_manifest),
            "--out",
            str(out),
        ],
    )
    assert report.main() == 0
    document = (out / "BETA_VALIDATION.md").read_text(encoding="utf-8")
    chinese = (out / "BETA_VALIDATION_CN.md").read_text(encoding="utf-8")
    assert "## Beta GO / NO-GO checklist" in document
    assert "## Beta GO / NO-GO 清单" in chinese
    assert "Verdict: **" in document
    assert "GO for beta" in document or "NO-GO" in document
    assert "https://example.invalid/archive.tar.zst" in document
    assert "cccccccccccccccc" in document  # archive sha prefix from live manifest
    # there is exactly one report document per language; no scattered checklist
    assert not (out / "BETA_GO_CHECKLIST.md").exists()
