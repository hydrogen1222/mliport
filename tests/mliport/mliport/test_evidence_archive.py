"""Tests for the evidence archive builder (task book section 16)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "validation" / "science" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_evidence_archive as archive_builder  # noqa: E402
import common  # noqa: E402


def _record(case_id: str, *, campaign: str):
    return common.result_record(
        case_id=case_id,
        test_id="t1_real_inference",
        status="pass",
        engine="mace",
        model_identity="mace-omat",
        model_sha256="a" * 64,
        task="bulk",
        head=None,
        dtype="float64",
        device={
            "requested": "cuda",
            "actual": "cuda:0",
            "gpu_name": "Tesla V100-SXM2-16GB",
            "gpu_uuid_hash": "b" * 16,
        },
        input_structure_id=None,
        parameters={},
        metrics={
            "energy_eV": -1.0,
            "forces_max_abs_eV_A": 0.1,
            "stress_supported": False,
        },
        campaign_id=campaign,
    )


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_archive_builder_packages_campaign_records(tmp_path, monkeypatch):
    if shutil.which("zstd") is None:
        pytest.skip("zstd CLI is unavailable")
    software_commit = "c" * 40
    validation_commit = "d" * 40
    monkeypatch.setenv(common.SOFTWARE_COMMIT_ENV, software_commit)
    root = tmp_path / "evidence"
    _write(root / "t1" / "a.json", _record("case_a", campaign="camp-1"))
    _write(root / "t1" / "b.json", _record("case_b", campaign="camp-1"))
    # other campaign and attic records must stay out of the archive
    _write(root / "t1" / "other.json", _record("case_c", campaign="camp-2"))
    _write(root / "attic" / "old.json", _record("case_d", campaign="camp-1"))

    campaign_manifest = tmp_path / "campaign.json"
    campaign_manifest.write_text(
        json.dumps(
            {
                "schema": "mliport.beta-campaign/1",
                "campaign_id": "camp-1",
                "software_commit": software_commit,
                "validation_commit": validation_commit,
                "status": "complete",
                "evidence_source_commits": [software_commit],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "evidence-archive.tar.zst"
    manifest_out = tmp_path / "archive_manifest.json"
    manifest = archive_builder.build(
        root=root,
        campaign="camp-1",
        out=out,
        software_commit=software_commit,
        validation_commit=validation_commit,
        campaign_manifest=campaign_manifest,
        manifest_out=manifest_out,
    )
    assert out.is_file()
    assert manifest["schema"] == "mliport.beta-archive-manifest/1"
    assert manifest["records"] == 2
    assert manifest["by_status"] == {"pass": 2}
    assert manifest["archive_sha256"] == archive_builder.sha256_file(out)
    assert manifest_out.is_file()

    listing = subprocess.run(
        ["tar", "-tf", str(out)], capture_output=True, text=True, check=True
    ).stdout
    assert "./records/t1/a.json" in listing
    assert "./records/t1/b.json" in listing
    assert "./manifests/campaign_manifest.json" in listing
    assert "./README.md" in listing
    assert "other.json" not in listing
    assert "attic" not in listing


def test_archive_builder_refuses_empty_or_broken_campaign(tmp_path, monkeypatch):
    monkeypatch.setenv(common.SOFTWARE_COMMIT_ENV, "e" * 40)
    root = tmp_path / "evidence"
    root.mkdir()
    with pytest.raises(SystemExit, match="no records"):
        archive_builder.build(
            root=root,
            campaign="empty",
            out=tmp_path / "x.tar.zst",
            software_commit="e" * 40,
            validation_commit="f" * 40,
            manifest_out=tmp_path / "m.json",
        )
    _write(root / "t1" / "good.json", _record("case_a", campaign="camp-1"))
    root.joinpath("t1", "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="evidence problem"):
        archive_builder.build(
            root=root,
            campaign="camp-1",
            out=tmp_path / "y.tar.zst",
            software_commit="e" * 40,
            validation_commit="f" * 40,
            manifest_out=tmp_path / "m.json",
        )
