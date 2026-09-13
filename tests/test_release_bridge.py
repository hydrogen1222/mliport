"""Release-bridge acceptance tests (recovery plan sections 11-15, 46-49).

The bridge runner is a validation-side orchestrator; these tests pin its
classification logic so a future regression cannot make a negative test
"pass" by succeeding, cannot let a non-finite SP through, and cannot let the
fairchem alias resolve to a non-UMA runtime.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "validation" / "science" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import release_bridge  # noqa: E402

TARGET = "5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c"


_AUTO = object()


def _result(
    *,
    engine: str = "mace",
    energy: float = -10.0,
    fmax: float = 0.1,
    force_value: float = 0.1,
    has_stress: bool = True,
    stress: object = _AUTO,
    actual_device_type: str = "cuda",
) -> dict:
    if stress is _AUTO:
        stress = [-0.001] * 6 if has_stress else None
    return {
        "mliport_version": "2.0.0b3",
        "metadata": {
            "model_type": engine,
            "has_stress": has_stress,
            "actual_device_type": actual_device_type,
            "actual_device_uuid": "GPU-abc",
            "implemented_properties": ["energy", "forces", "stress"]
            if has_stress
            else ["energy", "forces"],
        },
        "calculation": {
            "system": {"natoms": 4},
            "results": {
                "energy": energy,
                "forces": [[force_value, 0.0, 0.0]] * 4,
                "stress": stress,
                "force_statistics": {"fmax": fmax},
            },
        },
    }


def _resolved(engine: str = "mace", path: str = "/models/model.bin") -> dict:
    return {
        "model_type": engine,
        "model_path": path,
        "task": "bulk",
        "device": "cuda:0",
    }


# ------------------------------------------------------------------ status
def test_bridge_status_requires_every_check_to_pass():
    assert release_bridge.bridge_status({"a": {"status": "pass"}}) == "pass"
    assert (
        release_bridge.bridge_status({"a": {"status": "pass"}, "b": {"status": "fail"}})
        == "fail"
    )
    assert release_bridge.bridge_status({}) == "fail"
    assert release_bridge.bridge_status({"a": {}}) == "fail"


# ------------------------------------------------- negative-test semantics
def test_negative_evaluation_requires_expected_failure():
    # a negative test that unexpectedly succeeds must fail the bridge
    unexpected_success = release_bridge.evaluate_negative(
        exit_code=0,
        output="everything worked",
        expected_phrases=("No MLIP backend was selected",),
        label="x",
    )
    assert unexpected_success["status"] == "fail"

    wrong_message = release_bridge.evaluate_negative(
        exit_code=1,
        output="some other error",
        expected_phrases=("No MLIP backend was selected",),
        label="x",
    )
    assert wrong_message["status"] == "fail"

    expected = release_bridge.evaluate_negative(
        exit_code=1,
        output="Error: No MLIP backend was selected.",
        expected_phrases=("No MLIP backend was selected",),
        label="x",
    )
    assert expected["status"] == "pass"
    assert expected["observed_exit_code"] == 1


# ------------------------------------------------------------- single point
def test_single_point_check_accepts_finite_explicit_run(tmp_path):
    model = tmp_path / "model.bin"
    model.write_text("x", encoding="utf-8")
    check = release_bridge.check_single_point(
        _result(),
        _resolved(path=str(model)),
        engine="mace",
        model_path=str(model),
    )
    assert check["status"] == "pass"
    assert check["energy_eV"] == -10.0
    assert check["stress_supported"] is True


@pytest.mark.parametrize(
    "result,expected",
    [
        (_result(energy=float("nan")), "fail"),
        (_result(force_value=float("inf")), "fail"),
        (_result(has_stress=True, stress=None), "fail"),
        (_result(has_stress=False, stress=None), "pass"),
    ],
)
def test_single_point_check_rejects_nonfinite_observables(tmp_path, result, expected):
    model = tmp_path / "model.bin"
    model.write_text("x", encoding="utf-8")
    check = release_bridge.check_single_point(
        result, _resolved(path=str(model)), engine="mace", model_path=str(model)
    )
    assert check["status"] == expected


def test_single_point_check_requires_explicit_backend_attribution(tmp_path):
    model = tmp_path / "model.bin"
    model.write_text("x", encoding="utf-8")
    mismatched = release_bridge.check_single_point(
        _result(engine="uma"),
        _resolved(engine="mace", path=str(model)),
        engine="mace",
        model_path=str(model),
    )
    assert mismatched["status"] == "fail"


def test_device_attribution_flags_silent_cpu_fallback():
    assert (
        release_bridge.check_device_attribution(
            {"actual_device_type": "cpu"}, "cuda:0"
        )["status"]
        == "fail"
    )
    assert (
        release_bridge.check_device_attribution(
            {"actual_device_type": "cuda"}, "cuda:0"
        )["status"]
        == "pass"
    )
    assert (
        release_bridge.check_device_attribution({"actual_device_type": "cpu"}, "cpu")[
            "status"
        ]
        == "pass"
    )


# ------------------------------------------------------------ fairchem alias
def test_alias_check_requires_fairchem_to_resolve_to_uma():
    assert (
        release_bridge.check_alias_resolution(
            {"model_type": "fairchem"}, {"metadata": {"model_type": "uma"}}
        )["status"]
        == "pass"
    )
    wrong = release_bridge.check_alias_resolution(
        {"model_type": "fairchem"}, {"metadata": {"model_type": "mace"}}
    )
    assert wrong["status"] == "fail"


# --------------------------------------------------------------- CLI guard
def test_bridge_cli_rejects_a_bad_release_commit():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "release_bridge.py"),
            "--release-commit",
            "not-a-sha",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "40-character hex" in (result.stdout + result.stderr)


# ------------------------------------------------- committed bridge summary
def test_committed_bridge_summary_is_consistent():
    summaries = sorted(
        (REPO / "validation" / "science" / "bridge").glob("*/summary.json")
    )
    if not summaries:
        return  # the bridge runs on the GPU host after the code is frozen
    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    assert summary["schema"] == release_bridge.BRIDGE_SUMMARY_SCHEMA
    assert summary["status"] == "pass"
    assert summary["scientific_campaign_target"] == TARGET
    assert len(summary["release_candidate_commit"]) == 40
    assert set(summary["engines"]) == {"mace", "dpa", "grace", "uma"}
    assert all(status == "pass" for status in summary["engines"].values())
    assert summary["fairchem_alias"]["status"] == "pass"
    for name, status in summary["negative_checks"].items():
        assert status == "pass", name


# ------------------------------------------------------- bridge archive
def test_bridge_archive_builder_refuses_a_non_pass_summary(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import build_bridge_archive
    from evidence import BridgeArtifactError

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "summary.json").write_text(
        json.dumps(
            {
                "schema": release_bridge.BRIDGE_SUMMARY_SCHEMA,
                "campaign_id": "c",
                "release_candidate_commit": "a" * 40,
                "scientific_campaign_target": TARGET,
                "status": "fail",
                "engines": {"mace": "pass"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BridgeArtifactError, match="refusing"):
        build_bridge_archive.load_validated_summary(campaign)


def test_committed_bridge_archive_manifest_is_consistent():
    manifests = sorted(
        (REPO / "validation" / "science" / "bridge").glob("*/archive_manifest.json")
    )
    if not manifests:
        return  # the bridge archive is built after the GPU bridge run
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    assert manifest["schema"] == "mliport.release-bridge-archive-manifest/1"
    assert len(manifest["release_candidate_commit"]) == 40
    assert manifest["scientific_campaign_target"] == TARGET
    assert len(manifest["archive_sha256"]) == 64
    assert int(manifest["archive_bytes"]) > 0
    assert manifest["records"] == 4
    assert set(manifest["engines"]) == {"mace", "dpa", "grace", "uma"}
    assert str(manifest["archive_url"]).startswith("https://")
