"""PR-A acceptance tests: one canonical evidence pipeline.

These tests pin the contract from the e579937 revalidation task book
(section 3.5, V01-T1..T10): one strict loader, profile-aware aggregation,
fail-closed reports, and no report-side record selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
