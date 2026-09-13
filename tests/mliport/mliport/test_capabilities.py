from __future__ import annotations

import json
from pathlib import Path

import pytest

from mliport.capabilities import (
    IDENTITY_KEYS,
    CalculatorCapabilities,
    model_identity,
    require_neb_capability,
    resolve_capabilities,
)

# Reference identities matching the packaged curated registry
# (mliport/mliport/data/validation/*.json). Extracted dynamically so the tests
# fail loudly if the registry drifts from the documented identity fields.
DPA_EVIDENCE_ID = "v100-dpa-876354744aea-native-cacheTrue"
GRACE_EVIDENCE_ID = "v100-grace-87e72b50a2eb-native-cacheTrue"


def _packaged_identity(evidence_id: str) -> dict:
    from mliport.capabilities import _package_evidence_dir

    for entry in _package_evidence_dir().iterdir():
        if not entry.name.endswith(".json"):
            continue
        record = json.loads(entry.read_text(encoding="utf-8"))
        if record["evidence_id"] == evidence_id:
            return record["model_identity"]
    pytest.fail(f"packaged registry is missing evidence {evidence_id!r}")


def test_unknown_requires_explicit_experimental_override():
    capability = CalculatorCapabilities(True, True, False)
    with pytest.raises(ValueError, match="allow-unvalidated-neb"):
        require_neb_capability(capability, experimental=False)
    require_neb_capability(capability, experimental=True)


def test_direct_force_model_is_always_rejected():
    capability = resolve_capabilities({"direct_forces": True}, ["energy", "forces"])
    with pytest.raises(ValueError, match="non-guaranteed"):
        require_neb_capability(capability, experimental=True)


def _evidence_record(model, *, evidence_id="known-answer", workloads=None, **overrides):
    record = {
        "schema": "mliport.capability-evidence/1",
        "evidence_id": evidence_id,
        "source_record": "test",
        "model_identity": model_identity(model),
        "workloads": workloads
        or {
            key: {"status": "passed"}
            for key in ["single_point", "shared_calculator", "energy_force_consistency"]
        },
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize("changed", IDENTITY_KEYS)
def test_evidence_is_exact_not_a_backend_wide_claim(tmp_path, changed):
    model = {key: f"value-{key}" for key in IDENTITY_KEYS}
    (tmp_path / "evidence.json").write_text(json.dumps(_evidence_record(model)))
    assert (
        resolve_capabilities(
            model, ["energy", "forces"], evidence_dir=tmp_path
        ).energy_gradient_consistency
        == "validated"
    )
    model[changed] = "different"
    assert (
        resolve_capabilities(
            model, ["energy", "forces"], evidence_dir=tmp_path
        ).energy_gradient_consistency
        == "unknown"
    )


def test_validated_without_evidence_id_is_invalid():
    with pytest.raises(ValueError, match="evidence ID"):
        CalculatorCapabilities(True, True, False, "validated")


# ---------------------------------------------------------------------------
# Packaged curated registry (RC-01)
# ---------------------------------------------------------------------------


def test_packaged_registry_resolves_dpa_and_grace_without_evidence_dir():
    for evidence_id in (DPA_EVIDENCE_ID, GRACE_EVIDENCE_ID):
        capability = resolve_capabilities(
            _packaged_identity(evidence_id), ["energy", "forces"]
        )
        assert capability.energy_gradient_consistency == "validated"
        assert capability.validation_evidence_id == evidence_id


def test_packaged_registry_rejects_mace_and_uma_identities():
    # MACE's committed V100 evidence predates the current installer contract
    # (torch 2.6.0+cu124 vs torch 2.8.0+cu126) and is deliberately NOT in the
    # curated registry; UMA has no runtime validation at all.
    for model_type in ("mace", "uma"):
        identity = _packaged_identity(DPA_EVIDENCE_ID) | {"model_type": model_type}
        capability = resolve_capabilities(identity, ["energy", "forces"])
        assert capability.energy_gradient_consistency == "unknown"
        assert capability.validation_evidence_id is None


@pytest.mark.parametrize(
    "field",
    ["model_sha256", "backend_version", "dtype_effective", "head_effective"],
)
def test_single_field_mismatch_stays_unknown(field):
    identity = dict(_packaged_identity(DPA_EVIDENCE_ID))
    identity[field] = (
        "0" * 64 if field == "model_sha256" else "definitely-not-the-validated-value"
    )
    capability = resolve_capabilities(identity, ["energy", "forces"])
    assert capability.energy_gradient_consistency == "unknown"


def test_torch_version_mismatch_stays_unknown():
    identity = dict(_packaged_identity(DPA_EVIDENCE_ID))
    identity["framework_versions"] = {"torch": "2.6.0+cu124"}
    capability = resolve_capabilities(identity, ["energy", "forces"])
    assert capability.energy_gradient_consistency == "unknown"


def test_missing_energy_force_workload_stays_unknown(tmp_path):
    model = {key: f"value-{key}" for key in IDENTITY_KEYS}
    record = _evidence_record(
        model,
        workloads={
            "single_point": {"status": "passed"},
            "shared_calculator": {"status": "passed"},
        },
    )
    (tmp_path / "evidence.json").write_text(json.dumps(record))
    capability = resolve_capabilities(
        model, ["energy", "forces"], evidence_dir=tmp_path
    )
    assert capability.energy_gradient_consistency == "unknown"


def test_unrelated_malformed_file_warns_but_does_not_crash(tmp_path, recwarn):
    model = {key: f"value-{key}" for key in IDENTITY_KEYS}
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "evidence.json").write_text(json.dumps(_evidence_record(model)))
    capability = resolve_capabilities(
        model, ["energy", "forces"], evidence_dir=tmp_path
    )
    assert capability.energy_gradient_consistency == "validated"
    assert any("broken.json" in str(w.message) for w in recwarn.list)


def test_candidate_identity_with_wrong_schema_fails_closed(tmp_path):
    model = {key: f"value-{key}" for key in IDENTITY_KEYS}
    record = _evidence_record(model, schema="mliport.capability-evidence/0")
    (tmp_path / "evidence.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="wrong schema|schema.*fail closed"):
        resolve_capabilities(model, ["energy", "forces"], evidence_dir=tmp_path)


def test_candidate_identity_missing_required_fields_fails_closed(tmp_path):
    model = {key: f"value-{key}" for key in IDENTITY_KEYS}
    record = _evidence_record(model)
    del record["source_record"]
    (tmp_path / "evidence.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="missing required fields"):
        resolve_capabilities(model, ["energy", "forces"], evidence_dir=tmp_path)


def test_candidate_identity_malformed_workloads_fails_closed(tmp_path):
    model = {key: f"value-{key}" for key in IDENTITY_KEYS}
    record = _evidence_record(model, workloads="passed")
    (tmp_path / "evidence.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="malformed workloads"):
        resolve_capabilities(model, ["energy", "forces"], evidence_dir=tmp_path)


def test_duplicate_conflicting_evidence_fails_closed(tmp_path):
    model = {key: f"value-{key}" for key in IDENTITY_KEYS}
    (tmp_path / "a.json").write_text(json.dumps(_evidence_record(model)))
    partial = {
        "single_point": {"status": "passed"},
        "shared_calculator": {"status": "passed"},
    }
    (tmp_path / "b.json").write_text(
        json.dumps(_evidence_record(model, workloads=partial))
    )
    with pytest.raises(ValueError, match="Conflicting validation evidence"):
        resolve_capabilities(model, ["energy", "forces"], evidence_dir=tmp_path)


def test_duplicate_agreeing_evidence_resolves_deterministically(tmp_path):
    model = {key: f"value-{key}" for key in IDENTITY_KEYS}
    (tmp_path / "b.json").write_text(
        json.dumps(_evidence_record(model, evidence_id="second"))
    )
    (tmp_path / "a.json").write_text(
        json.dumps(_evidence_record(model, evidence_id="first"))
    )
    capability = resolve_capabilities(
        model, ["energy", "forces"], evidence_dir=tmp_path
    )
    assert capability.energy_gradient_consistency == "validated"
    assert capability.validation_evidence_id == "first"


def test_registry_matches_full_records_in_repo(tmp_path):
    """Every packaged evidence id must exist in the committed full records."""
    repo_record_dir = (
        Path(__file__).resolve().parents[3] / "validation" / "runtime" / "v100"
    )
    if not repo_record_dir.is_dir():
        pytest.skip("repository validation records not present")
    committed = {
        record["evidence_id"]
        for p in repo_record_dir.glob("*.json")
        if "evidence_id" in (record := json.loads(p.read_text(encoding="utf-8")))
    }
    from mliport.capabilities import _package_evidence_dir

    packaged = {
        json.loads(entry.read_text(encoding="utf-8"))["evidence_id"]
        for entry in _package_evidence_dir().iterdir()
        if entry.name.endswith(".json")
    }
    assert packaged <= committed
