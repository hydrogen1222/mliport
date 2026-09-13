"""CPU known-answer regression tests for the beta science harness.

Everything here runs without any ML backend: transformations are checked
against hand-computed tensors, fixture identities are pinned against the
committed manifest, and the finite-difference machinery is exercised on
ASE's EMT potential (a real, smooth, CPU-only potential).  EMT is used
ONLY as the reference system for these arithmetic/plumbing tests -- it is
never a substitute for a production model in any recorded suite evidence
(taskbook section 47).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.calculators.fd import calculate_numerical_forces, calculate_numerical_stress

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "validation" / "science" / "scripts"
sys.path.insert(0, str(SCRIPTS))  # noqa: E402

import aggregate  # noqa: E402
import common  # noqa: E402
import finite_difference  # noqa: E402
import fixtures  # noqa: E402
import invariance  # noqa: E402
import schema_check  # noqa: E402
import transforms  # noqa: E402

MANIFEST = REPO / "validation" / "science" / "model_manifest.json"
FIXTURE_MANIFEST = (
    REPO / "validation" / "science" / "cases" / "manifests" / "fixtures.json"
)


# --------------------------------------------------------------------------
# unit conversion (taskbook section 13)
# --------------------------------------------------------------------------


def test_ev_per_a3_to_gpa_exact():
    assert pytest.approx(160.21766208, abs=1e-9) == common.EV_A3_TO_GPA


# --------------------------------------------------------------------------
# transforms: synthetic-tensor known answers (taskbook section 11.6)
# --------------------------------------------------------------------------


def test_proper_rotation_is_proper():
    rot = transforms.proper_rotation((1.0, 1.0, 1.0), 0.37)
    assert rot.shape == (3, 3)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    # 90 deg about z: x -> y
    rot_z = transforms.proper_rotation((0.0, 0.0, 1.0), np.pi / 2)
    assert np.allclose(rot_z @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_rotate_vectors_known_answer():
    rot_z = transforms.proper_rotation((0.0, 0.0, 1.0), np.pi / 2)
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    expected = np.array([[0.0, 1.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    assert np.allclose(transforms.rotate_vectors(vectors, rot_z), expected, atol=1e-12)


def test_voigt_roundtrip():
    mat = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]])
    voigt = transforms.matrix_to_voigt(mat)
    assert list(voigt) == [1.0, 5.0, 9.0, 6.0, 3.0, 2.0]  # xx,yy,zz,yz,xz,xy
    assert np.allclose(transforms.voigt_to_matrix(voigt), mat, atol=1e-15)


def test_rotate_stress_voigt_known_answer():
    """Hand-computed rotation of a full stress tensor (sigma_zz shear intact)."""
    sigma = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]])
    rot_z90 = transforms.proper_rotation((0.0, 0.0, 1.0), np.pi / 2)
    expected = rot_z90 @ sigma @ rot_z90.T
    assert np.allclose(
        expected,
        [[5.0, -2.0, -6.0], [-2.0, 1.0, 3.0], [-6.0, 3.0, 9.0]],
        atol=1e-10,
    )
    got_voigt = transforms.rotate_stress_voigt(
        transforms.matrix_to_voigt(sigma), rot_z90
    )
    assert np.allclose(transforms.voigt_to_matrix(got_voigt), expected, atol=1e-10)


def test_isotropic_stress_is_rotation_invariant():
    rot = transforms.proper_rotation((2.0, -1.0, 0.5), 0.83)
    sigma = 4.2 * np.eye(3)
    rotated = transforms.voigt_to_matrix(
        transforms.rotate_stress_voigt(transforms.matrix_to_voigt(sigma), rot)
    )
    assert np.allclose(rotated, sigma, atol=1e-10)


# --------------------------------------------------------------------------
# structure identity
# --------------------------------------------------------------------------


def test_structure_id_determinism_and_sensitivity():
    atoms, _ = fixtures.build_fixture("cu_fcc")
    sid1 = common.structure_id(atoms)
    assert sid1 == common.structure_id(atoms.copy())

    moved = atoms.copy()
    moved.positions[0, 0] += 1e-3
    assert common.structure_id(moved) != sid1

    wrapped = atoms.copy()
    wrapped.pbc = [True, True, False]
    assert common.structure_id(wrapped) != sid1


def test_committed_fixture_manifest_matches_live_fixtures():
    from build_cases import build_fixtures_manifest

    committed = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    assert committed == build_fixtures_manifest()


def test_invariance_systems_include_triclinic_and_distorted():
    # taskbook sections 11.4 / 12: triclinic + non-zero forces coverage
    assert "si_diamond_prim" in fixtures.INVARIANCE_SYSTEMS
    assert "cu_distorted" in fixtures.INVARIANCE_SYSTEMS
    prim, _ = fixtures.build_extra("si_diamond_prim")
    cellpar = prim.cell.cellpar()
    assert not np.allclose(cellpar[3:], 90.0, atol=1e-6)  # non-orthogonal angles


# --------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------


def test_committed_model_manifest_loads():
    manifest = common.load_model_manifest(MANIFEST)
    assert manifest["schema"] == common.MODEL_MANIFEST_SCHEMA
    engines = {p["engine"] for p in manifest["profiles"].values()}
    assert engines == {"uma", "mace", "dpa", "grace"}
    # 6 profiles: UMA, MACE f64/f32, DPA, GRACE default/fp64
    assert len(manifest["profiles"]) == 6


def test_resolve_profile_hash_enforcement(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"not-a-model")
    real_hash = common.sha256_file(artifact)
    manifest = {
        "schema": common.MODEL_MANIFEST_SCHEMA,
        "profiles": {
            "toy": {
                "engine": "mace",
                "upstream_model_id": "toy",
                "model_sha256": real_hash,
                "dtype": "float64",
            }
        },
    }
    resolved = common.resolve_profile(manifest, "toy", artifact)
    assert resolved["model_sha256"] == real_hash

    tampered = json.loads(json.dumps(manifest))
    tampered["profiles"]["toy"]["model_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="hash mismatch"):
        common.resolve_profile(tampered, "toy", artifact)


def test_schema_ids_match_scripts():
    result_schema = json.loads(
        (REPO / "validation" / "science" / "schemas" / "result.schema.json").read_text()
    )
    summary_schema = json.loads(
        (
            REPO / "validation" / "science" / "schemas" / "summary.schema.json"
        ).read_text()
    )
    assert result_schema["$id"] == common.RESULT_SCHEMA
    assert summary_schema["$id"] == common.SUMMARY_SCHEMA


# --------------------------------------------------------------------------
# result records
# --------------------------------------------------------------------------


def _record(status="pass", **overrides):
    base = {
        "case_id": "t_case",
        "test_id": "t_test",
        "status": status,
        "engine": "mace",
        "model_identity": "toy",
        "model_sha256": "a" * 64,
        "task": "bulk",
        "head": None,
        "dtype": "float64",
        "device": {
            "requested": "cpu",
            "actual": "cpu",
            "gpu_name": None,
            "gpu_uuid_hash": None,
        },
        "input_structure_id": None,
        "parameters": {},
        "metrics": {},
    }
    base.update(overrides)
    return common.result_record(**base)


def test_result_record_rejects_unknown_status():
    with pytest.raises(ValueError, match="invalid status"):
        _record(status="probably-fine")


def test_write_result_filename_composition(tmp_path):
    path = common.write_result(_record(), tmp_path)
    assert path.name == "t_case__t_test__mace-float64.json"
    tagged = common.write_result(_record(), tmp_path, tag="neighoff")
    assert tagged.name == "t_case__t_test__mace-float64-neighoff.json"
    loaded = json.loads(tagged.read_text(encoding="utf-8"))
    assert loaded["schema"] == common.RESULT_SCHEMA


# --------------------------------------------------------------------------
# aggregation logic (incl. harness-integrity violations, section 47)
# --------------------------------------------------------------------------


def _raw_record(wrapper="mliport.calculators", **kw):
    rec = _record(**kw)
    rec["diagnostics"] = {"wrapper_module": wrapper}
    return rec


def test_aggregate_flags_foreign_wrapper(tmp_path):
    foreign = _raw_record(wrapper="some.silent.emt.fallback")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(foreign))
    loaded = aggregate.load_records(tmp_path)
    assert not loaded.problems
    violations = aggregate.harness_violations(loaded.records)
    assert violations and "some.silent.emt.fallback" in violations[0]

    # every mliport wrapper module must be whitelisted -- keeps the test honest.
    # The pre-rename "mlipx.calculators.*" paths are allowed as legacy history
    # only; no other namespace may appear.
    assert "mliport.calculators.mace_calc" in common.ALLOWED_WRAPPER_MODULES
    assert "mliport.calculators.dpa_calc" in common.ALLOWED_WRAPPER_MODULES
    assert "mliport.calculators.grace_calc" in common.ALLOWED_WRAPPER_MODULES
    assert all(
        m.startswith(("mliport.", "mlipx.")) for m in common.ALLOWED_WRAPPER_MODULES
    )
    assert {m for m in common.ALLOWED_WRAPPER_MODULES if m.startswith("mlipx.")} == {
        "mlipx.calculator",
        "mlipx.calculators.mace_calc",
        "mlipx.calculators.dpa_calc",
        "mlipx.calculators.grace_calc",
    }


def test_support_matrix_distinguishes_unsupported_and_not_run():
    rec_full = _raw_record()
    rec_full["test_id"] = "t1_real_inference"
    rec_full["metrics"] = {
        "energy_eV": -1.0,
        "forces_max_abs_eV_A": 0.1,
        "stress_supported": False,
    }
    matrix = aggregate.support_matrix([rec_full])
    assert matrix["mace"]["energy"]["status"] == "pass"
    assert matrix["mace"]["forces"]["status"] == "pass"
    assert matrix["mace"]["stress"]["status"] == "unsupported"  # observed absence
    # an engine with no t1 record must be absent from the matrix (= not run)
    assert "dpa" not in matrix
    # declared engines are seeded as not_run, never as pass
    seeded = aggregate.support_matrix([rec_full], expected_engines=("dpa", "mace"))
    assert seeded["dpa"]["energy"]["status"] == "not_run"
    assert seeded["mace"]["energy"]["status"] == "pass"


# --------------------------------------------------------------------------
# R03/R04: import identity, strict schema validation, order independence
# --------------------------------------------------------------------------


def _write_json(tmp_path, name, payload_text):
    path = tmp_path / name
    path.write_text(payload_text, encoding="utf-8")
    return path


def test_load_records_rejects_non_object_json(tmp_path):
    _write_json(
        tmp_path, "list.json", '[{"schema": "mliport.beta-validation-result/2"}]'
    )
    _write_json(tmp_path, "scalar.json", "42")
    loaded = aggregate.load_records(tmp_path)
    assert loaded.records == []
    assert len(loaded.problems) == 2
    assert all("not a JSON object" in p for p in loaded.problems)


def test_load_records_rejects_missing_required_fields(tmp_path):
    rec = _raw_record()
    del rec["metrics"]
    _write_json(tmp_path, "missing.json", json.dumps(rec))
    loaded = aggregate.load_records(tmp_path)
    assert loaded.records == []
    assert any("metrics" in p for p in loaded.problems)


def test_load_records_rejects_invalid_enums(tmp_path):
    bad_engine = _raw_record()
    bad_engine["engine"] = "fake-engine"
    bad_status = _raw_record()
    bad_status["status"] = "probably-fine"
    _write_json(tmp_path, "engine.json", json.dumps(bad_engine))
    _write_json(tmp_path, "status.json", json.dumps(bad_status))
    loaded = aggregate.load_records(tmp_path)
    assert loaded.records == []
    assert any("fake-engine" in p for p in loaded.problems)
    assert any("probably-fine" in p for p in loaded.problems)


def test_load_records_rejects_bad_model_hash_and_device(tmp_path):
    bad_hash = _raw_record()
    bad_hash["model_sha256"] = "not-a-hash"
    bad_device = _raw_record()
    bad_device["device"] = {}
    _write_json(tmp_path, "hash.json", json.dumps(bad_hash))
    _write_json(tmp_path, "device.json", json.dumps(bad_device))
    loaded = aggregate.load_records(tmp_path)
    assert loaded.records == []
    assert any("model_sha256" in p for p in loaded.problems)
    assert any("requested" in p for p in loaded.problems)


def test_load_records_rejects_nan_and_infinity(tmp_path):
    # NaN/Infinity literals are not valid JSON numbers ...
    rec = _record()
    nan_text = json.dumps(rec).replace('"wall_seconds": 0.0', '"wall_seconds": NaN')
    assert "NaN" in nan_text
    _write_json(tmp_path, "nan.json", nan_text)
    # ... and a huge exponent decodes to inf, caught by the semantic walk.
    rec_inf = _record()
    inf_text = json.dumps(rec_inf).replace(
        '"metrics": {}', '"metrics": {"energy_eV": 1e999}'
    )
    assert "1e999" in inf_text
    _write_json(tmp_path, "inf.json", inf_text)
    loaded = aggregate.load_records(tmp_path)
    assert loaded.records == []
    assert any("non-finite" in p for p in loaded.problems)
    assert any("$.metrics.energy_eV" in p for p in loaded.problems)


def test_load_records_rejects_corrupt_json_and_unknown_schema(tmp_path):
    _write_json(tmp_path, "corrupt.json", "{not json at all")
    unknown = _raw_record()
    unknown["schema"] = "mliport.beta-validation-result/99"
    _write_json(tmp_path, "unknown.json", json.dumps(unknown))
    loaded = aggregate.load_records(tmp_path)
    assert loaded.records == []
    assert any("unreadable JSON" in p for p in loaded.problems)
    assert any("unknown result schema" in p for p in loaded.problems)


def test_load_records_skips_ancillary_json_by_schema(tmp_path):
    summary_like = {
        "schema": common.SUMMARY_SCHEMA,
        "record_counts": {"total": 0, "by_status": {}},
    }
    manifest_like = {"schema": common.MODEL_MANIFEST_SCHEMA, "profiles": {}}
    _write_json(tmp_path, "summary.json", json.dumps(summary_like))
    _write_json(tmp_path, "manifest.json", json.dumps(manifest_like))
    good = _raw_record()
    _write_json(tmp_path, "record.json", json.dumps(good))
    loaded = aggregate.load_records(tmp_path)
    assert len(loaded.records) == 1
    assert loaded.problems == []
    assert loaded.ancillary_files == 2


def test_load_records_migrates_v1_and_marks_it(tmp_path):
    legacy = _record()
    for key in (
        "profile_id",
        "record_id",
        "run_id",
        "inference_mode",
        "seed",
    ):
        legacy.pop(key)
    legacy["schema"] = common.RESULT_SCHEMA_V1
    legacy["parameters"] = {"seed": 7}
    _write_json(tmp_path, "legacy.json", json.dumps(legacy))
    loaded = aggregate.load_records(tmp_path)
    assert loaded.problems == []
    assert len(loaded.records) == 1
    assert loaded.migrated_records == 1
    rec = loaded.records[0]
    assert rec["schema"] == common.RESULT_SCHEMA
    assert rec["migrated_from"] == common.RESULT_SCHEMA_V1
    assert rec["seed"] == 7
    assert len(rec["profile_id"]) > 0 and len(rec["record_id"]) == 32
    # migration is idempotent when the same legacy file is re-aggregated
    loaded_again = aggregate.load_records(tmp_path)
    assert loaded_again.records[0]["record_id"] == rec["record_id"]
    assert loaded_again.records[0]["run_id"] == rec["run_id"]


def _t1_record(*, dtype, status, wrapper="mliport.calculators.mace_calc"):
    rec = _raw_record(wrapper=wrapper)
    rec["test_id"] = "t1_real_inference"
    rec["case_id"] = "t1_case"
    rec["dtype"] = dtype
    rec["status"] = status
    rec["metrics"] = {
        "energy_eV": -1.0,
        "forces_max_abs_eV_A": 0.1,
        "stress_supported": False,
    }
    rec["profile_id"] = common.profile_id_for(rec)
    rec["record_id"] = common.record_id_for(rec)
    return rec


def test_aggregate_keeps_float32_fail_visible_next_to_float64_pass(tmp_path):
    fail32 = _t1_record(dtype="float32", status="fail")
    fail32["exception"] = "float32 overflow"
    pass64 = _t1_record(dtype="float64", status="pass")
    pass64["metrics"] = dict(pass64["metrics"], energy_eV=-1.0000001)
    _write_json(tmp_path, "a_fail32.json", json.dumps(fail32))
    _write_json(tmp_path, "b_pass64.json", json.dumps(pass64))
    summary = aggregate.build_summary(tmp_path, None)

    buckets = summary["cases"]["t1_case"]["t1_real_inference"]
    assert len(buckets) == 2, "one bucket per profile identity"
    by_status = {b["status"] for b in buckets.values()}
    assert by_status == {"fail", "pass"}
    matrix = summary["support_matrix"]["mace"]["energy"]
    # Engine-level cells are only a single status when every profile agrees;
    # mixed profiles are reported as "mixed" with the worst kept alongside so
    # a float32 failure can never be compressed into a passing engine
    # (task book section 3.4).
    assert matrix["status"] == "mixed"
    assert matrix["worst_status"] == "fail"
    assert set(matrix["by_profile"].values()) == {"fail", "pass"}
    # dtype is part of the profile identity, so the two never merge
    assert len({pid.split("-")[1] for pid in buckets}) == 2


def _summary_without_paths(summary):
    """Deterministic projection: drop the file path each run was read from."""
    import copy

    stripped = copy.deepcopy(summary)
    for tests in stripped["cases"].values():
        for buckets in tests.values():
            for bucket in buckets.values():
                for run in bucket["runs"]:
                    run.pop("path", None)
                bucket["runs"].sort(key=lambda r: r["run_id"])
    return stripped["cases"]


def test_aggregate_is_order_independent(tmp_path):
    records = [
        _t1_record(dtype="float32", status="fail"),
        _t1_record(dtype="float64", status="pass"),
    ]
    records[0]["exception"] = "float32 overflow"
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    for i, rec in enumerate(records):
        _write_json(dir_a, f"{i}_record.json", json.dumps(rec))
    for i, rec in enumerate(reversed(records)):
        _write_json(dir_b, f"{i}_other_name.json", json.dumps(rec))
    summary_a = aggregate.build_summary(dir_a, None)
    summary_b = aggregate.build_summary(dir_b, None)
    for key in ("cases", "support_matrix", "record_counts", "profiles"):
        a = _summary_without_paths(summary_a) if key == "cases" else summary_a[key]
        b = _summary_without_paths(summary_b) if key == "cases" else summary_b[key]
        assert a == b, f"{key} depends on file order"


def test_aggregate_expected_matrix_marks_missing_engines(tmp_path):
    rec = _t1_record(dtype="float64", status="pass")
    _write_json(tmp_path, "mace.json", json.dumps(rec))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": common.MODEL_MANIFEST_SCHEMA,
                "suite_revision": 1,
                "profiles": {
                    "mace": {"engine": "mace", "model_sha256": "a" * 64},
                    "uma": {"engine": "uma", "model_sha256": "b" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    summary = aggregate.build_summary(tmp_path, manifest)
    assert summary["expected_matrix"]["missing_engines"] == ["uma"]
    assert summary["support_matrix"]["uma"]["energy"]["status"] == "not_run"
    assert aggregate.exit_code(summary, expect_complete=False) == aggregate.EXIT_OK
    assert (
        aggregate.exit_code(summary, expect_complete=True) == aggregate.EXIT_INCOMPLETE
    )


def test_aggregate_exit_contract(tmp_path):
    clean = aggregate.build_summary(tmp_path, None)
    assert clean["record_counts"]["total"] == 0
    assert (
        aggregate.exit_code(clean, expect_complete=False) == aggregate.EXIT_NO_RECORDS
    )

    violating = _raw_record(wrapper="some.silent.emt.fallback")
    _write_json(tmp_path, "foreign.json", json.dumps(violating))
    summary = aggregate.build_summary(tmp_path, None)
    assert summary["harness_violations"]
    assert (
        aggregate.exit_code(summary, expect_complete=False)
        == aggregate.EXIT_HARNESS_VIOLATION
    )


def test_aggregate_main_exit_code_for_bad_import(tmp_path, monkeypatch):
    bad = _raw_record()
    bad["engine"] = "fake-engine"
    _write_json(tmp_path, "bad.json", json.dumps(bad))
    out = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate",
            "--results",
            str(tmp_path),
            "--out",
            str(out),
        ],
    )
    assert aggregate.main() == aggregate.EXIT_IMPORT_PROBLEM
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["problems"], "bad imports must stay visible in the summary"


def test_write_result_refuses_to_overwrite_different_record(tmp_path):
    first = _record()
    path = common.write_result(first, tmp_path)
    assert path.name == "t_case__t_test__mace-float64.json"

    second = _record()
    second["parameters"] = {"fixture": "different"}
    second["profile_id"] = common.profile_id_for(second)
    second["record_id"] = common.record_id_for(second)
    with pytest.raises(common.ResultCollisionError, match="collision"):
        common.write_result(second, tmp_path, version_on_collision=False)

    versioned = common.write_result(second, tmp_path)
    assert versioned != path
    assert versioned.name.startswith("t_case__t_test__mace-float64__run-")
    original = json.loads(path.read_text(encoding="utf-8"))
    assert original["parameters"] == {}, "the old record must stay untouched"
    assert json.loads(versioned.read_text(encoding="utf-8"))["parameters"] == {
        "fixture": "different"
    }


def test_write_result_is_idempotent_for_identical_content(tmp_path):
    rec = _record()
    first = common.write_result(rec, tmp_path)
    second = common.write_result(rec, tmp_path)
    assert first == second
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_profile_identity_separates_model_identity():
    base = _record()
    other_dtype = _record(dtype="float32")
    other_head = _record(head="Omat24")
    other_commit = _record()
    other_commit["git_commit"] = "deadbeef" * 5
    assert common.profile_id_for(base) != common.profile_id_for(other_dtype)
    assert common.profile_id_for(base) != common.profile_id_for(other_head)
    assert common.profile_id_for(base) != common.profile_id_for(other_commit)


def test_blocked_provisioning_record_is_valid_but_not_a_model_result():
    """run_suite's blocked path has no model at all: 'n/a' is explicit, not a hash."""
    rec = common.result_record(
        case_id="orchestrator",
        test_id="engine_provisioning",
        status="blocked",
        engine="uma",
        model_identity="uma",
        model_sha256="n/a",
        task=None,
        head=None,
        dtype="n/a",
        device={
            "requested": "cuda:0",
            "actual": None,
            "gpu_name": None,
            "gpu_uuid_hash": None,
        },
        input_structure_id=None,
        parameters={"reason": "venv missing"},
        metrics={},
    )
    schema = schema_check.load_schema(
        REPO / "validation" / "science" / "schemas" / "result.schema.json"
    )
    assert schema_check.validate(rec, schema) == []
    # but an arbitrary bogus hash is still rejected
    rec["model_sha256"] = "definitely-a-model"
    assert schema_check.validate(rec, schema)


def test_committed_schemas_are_executable_and_strict():
    for name in (
        "result.schema.json",
        "result-v1.schema.json",
        "summary.schema.json",
        "summary-v1.schema.json",
    ):
        schema_check.load_schema(REPO / "validation" / "science" / "schemas" / name)
    schema = schema_check.load_schema(
        REPO / "validation" / "science" / "schemas" / "result.schema.json"
    )
    assert schema["$id"] == common.RESULT_SCHEMA
    assert schema_check.validate(_record(), schema) == []
    bad = _record()
    bad["model_sha256"] = "nope"
    assert schema_check.validate(bad, schema)

    # unsupported keywords must fail loudly, not be ignored
    with pytest.raises(schema_check.SchemaError, match="unsupported"):
        schema_check.validate({}, {"type": "object", "oneOf": [{"type": "object"}]})


# --------------------------------------------------------------------------
# finite difference machinery on EMT (CPU known answer)
# --------------------------------------------------------------------------


def _cu_distorted_emt():
    atoms, _ = fixtures.build_extra("cu_distorted")
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert np.isfinite(energy)
    assert np.abs(forces).max() > 1e-4  # fixture must be strained
    return atoms, energy, forces, stress


def test_emt_analytical_matches_fd_forces():
    atoms, _, forces, _ = _cu_distorted_emt()
    for eps in (5e-4, 2e-3):
        fd = calculate_numerical_forces(atoms, eps=eps)
        assert np.abs(fd - forces).max() < 1e-5


def test_emt_analytical_matches_fd_stress():
    atoms, _, _, stress = _cu_distorted_emt()
    for eps in (5e-4, 2e-3):
        fd = calculate_numerical_stress(atoms, eps=eps)
        assert np.abs(fd - stress).max() < 1e-5


def _fd_ctx(calculator, dtype="float64"):
    import engines as engines_mod

    return engines_mod.EngineContext(
        engine="mace",
        profile_id="fd-toy",
        profile={
            "model_sha256": "c" * 64,
            "upstream_model_id": "fd-toy",
            "identity": "fd toy",
        },
        wrapper=None,
        calculator=calculator,
        device="cpu",
        dtype=dtype,
        task="bulk",
        head=None,
        backend_version="toy",
        framework_version="toy",
    )


def _patch_numerical_forces(monkeypatch, errors_by_h):
    """Return analytic + prescribed error per displacement (no backend)."""

    def fake(atoms, eps=1e-3, iatoms=None, icarts=None):
        analytic = np.asarray(atoms.get_forces(), dtype=float)
        i, c = iatoms[0], icarts[0]
        return np.array([[float(analytic[i, c]) + errors_by_h[float(eps)]]])

    monkeypatch.setattr(finite_difference, "calculate_numerical_forces", fake)


def test_fd_isolated_good_point_is_not_a_stable_region(monkeypatch):
    """R11 red case: errors [0.5, 0.0005, 0.5, 0.5] must not be 'characterized'."""
    atoms, _ = fixtures.build_extra("cu_distorted")
    atoms.calc = EMT()
    ctx = _fd_ctx(atoms.calc)
    disps = (5e-4, 1e-3, 2e-3, 5e-3)
    _patch_numerical_forces(
        monkeypatch, dict(zip(disps, (0.5, 5e-4, 0.5, 0.5), strict=False))
    )
    case = finite_difference.force_fd_case(ctx, "cu", atoms, disps, "cpu")
    assert case["status"] == "fail"
    assert case["region"]["stable_region_found"] is False
    assert case["region"]["n_qualifying_scales"] == 1
    assert case["region"]["adjacent_qualifying_pairs"] == []
    # the full sweep is preserved -- no single "best" displacement is selected
    assert len(case["per_displacement"]) == 4
    assert [
        entry["max_abs_error_eV_A"] for entry in case["per_displacement"]
    ] == pytest.approx([0.5, 5e-4, 0.5, 0.5], rel=1e-9, abs=1e-15)
    assert case["region"]["biased_scales"] == [5e-4, 2e-3, 5e-3]


def test_fd_adjacent_good_points_form_a_region(monkeypatch):
    atoms, _ = fixtures.build_extra("cu_distorted")
    atoms.calc = EMT()
    ctx = _fd_ctx(atoms.calc)
    disps = (5e-4, 1e-3, 2e-3, 5e-3)
    _patch_numerical_forces(
        monkeypatch, dict(zip(disps, (0.5, 5e-4, 4e-4, 0.5), strict=False))
    )
    case = finite_difference.force_fd_case(ctx, "cu", atoms, disps, "cpu")
    assert case["status"] == "characterized"
    assert case["region"]["stable_region_found"] is True
    assert case["region"]["adjacent_qualifying_pairs"] == [[1e-3, 2e-3]]
    assert case["region"]["stable_region_kind"] in {"plateau", "convergence"}


def test_fd_float32_cancellation_is_classified_not_hidden(monkeypatch):
    """A float32 total energy cannot resolve 1e-3 eV/A at tiny displacements.

    The scale ``eps32*|E|/(2h)`` is recorded and the sweep becomes
    ``insufficient_sampling`` instead of a false pass or a false backend
    failure.
    """

    class _LargeEnergyCalculator(EMT):
        def get_potential_energy(self, atoms=None, **_kwargs):
            return 1.0e4

    atoms, _ = fixtures.build_extra("cu_distorted")
    calc = _LargeEnergyCalculator()
    atoms.calc = calc
    ctx = _fd_ctx(calc, dtype="float32")
    disps = (5e-4, 1e-3)
    errs = (1.0, 1.1)
    _patch_numerical_forces(monkeypatch, dict(zip(disps, errs, strict=False)))
    case = finite_difference.force_fd_case(ctx, "cu", atoms, disps, "cpu")
    expected_floor = finite_difference.FLOAT32_EPS * 1.0e4
    assert case["acceptance"]["energy_noise_floor_eV"] == pytest.approx(
        expected_floor, rel=1e-12
    )
    first = case["per_displacement"][0]
    assert first["noise_force_scale_eV_A"] == pytest.approx(expected_floor / (2 * 5e-4))
    assert first["noise_limited"] is True
    assert first["qualifies"] is False
    assert case["status"] == "insufficient_sampling"
    assert case["region"]["stable_region_kind"] == "noise_limited_plateau"


def test_fd_known_wrong_forces_fail_closed(monkeypatch):
    """A constant 0.1 eV/A force error is not explained by any noise scale."""
    atoms, _ = fixtures.build_extra("cu_distorted")
    atoms.calc = EMT()
    ctx = _fd_ctx(atoms.calc)
    disps = (5e-4, 1e-3, 2e-3, 5e-3)
    _patch_numerical_forces(monkeypatch, dict.fromkeys(disps, 0.1))
    case = finite_difference.force_fd_case(ctx, "cu", atoms, disps, "cpu")
    assert case["status"] == "fail"
    assert case["region"]["biased_scales"] == list(disps)


def test_fd_stress_distinguishes_unsupported_from_backend_failure(monkeypatch):
    from ase.calculators.calculator import PropertyNotImplementedError

    class _NoStress(EMT):
        def get_stress(self, *args, **kwargs):
            raise PropertyNotImplementedError("no stress")

    atoms, _ = fixtures.build_extra("cu_distorted")
    atoms.calc = _NoStress()
    case = finite_difference.stress_fd_case(
        _fd_ctx(atoms.calc), "cu", atoms, (1e-3, 2e-3), "cpu"
    )
    assert case["status"] == "unsupported"

    class _BrokenStress(EMT):
        def get_stress(self, *args, **kwargs):
            raise RuntimeError("CUDA error: device-side assert triggered")

    broken_atoms, _ = fixtures.build_extra("cu_distorted")
    broken_atoms.calc = _BrokenStress()
    with pytest.raises(RuntimeError, match="stress evaluation failed"):
        finite_difference.stress_fd_case(
            _fd_ctx(broken_atoms.calc), "cu", broken_atoms, (1e-3, 2e-3), "cpu"
        )


def test_fd_dof_selection_deterministic_and_capped():
    atoms, _, forces, _ = _cu_distorted_emt()
    a = finite_difference.select_force_dofs(
        atoms, forces, finite_difference.MAX_FORCE_DOFS
    )
    b = finite_difference.select_force_dofs(
        atoms, forces, finite_difference.MAX_FORCE_DOFS
    )
    assert a == b
    assert 1 <= len(a) <= finite_difference.MAX_FORCE_DOFS
    # multi-species coverage when species exist (MgO) and Cu2x2x2 here: Cu only,
    # so just assert DOFs are distinct
    assert len({tuple(d) for d in a}) == len(a)
    # selected DOFs' FD values must agree with the analytic gradient
    fd = calculate_numerical_forces(atoms, eps=2e-3)
    for i, comp in a:
        assert fd[i, comp] == pytest.approx(forces[i, comp], abs=1e-5)


# --------------------------------------------------------------------------
# invariance check machinery end-to-end on EMT (CPU)
# --------------------------------------------------------------------------


def test_invariance_checks_pass_on_emt():
    atoms, _ = fixtures.build_extra("cu_distorted")
    atoms.calc = EMT()
    floor = invariance.measure_floor(atoms, device="cpu")
    assert floor["energy_spread_eV"] < 1e-10

    status, metrics = invariance.check_repeatability(floor)
    assert status == "characterized"

    for name, check in invariance.CHECKS:
        if check is None:
            continue
        status, metrics = check(atoms, floor)
        assert status == "pass", f"{name}: {metrics}"
        # every check must report the tolerance policy it used
        assert any(k.startswith("tolerance_") for k in metrics)


def test_floor_policy_values_documented():
    assert invariance.FLOOR_POLICY["tolerance_multiplier"] == 10.0
    assert invariance.FLOOR_POLICY["rule"].startswith(
        "tolerance = max(multiplier * measured floor, absolute floor)"
    )


# --------------------------------------------------------------------------
# R02: repeated-inference floors must be real backend calls
# --------------------------------------------------------------------------


class _CountingNoiseCalculator(Calculator):
    """ASE calculator with a real call counter and optional known noise.

    ``calculate`` is the genuine inference entry point: the counter only
    advances when the backend is actually executed, so the test can prove
    that a repeatability floor came from real calls.
    """

    implemented_properties = ["energy", "forces", "stress"]  # noqa: RUF012

    def __init__(self, noise: float = 0.0, seed: int = 20260912):
        super().__init__()
        self.calls = 0
        self.noise = float(noise)
        self._rng = np.random.default_rng(seed)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        pos = np.asarray(atoms.positions, dtype=float)
        self.results = {
            "energy": float(0.5 * np.sum(pos**2)) + self.noise * self._rng.normal(),
            "forces": -pos + self.noise * self._rng.normal(size=pos.shape),
            "stress": np.zeros(6) + self.noise * self._rng.normal(size=6),
        }


def _counting_atoms(noise):
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    calc = _CountingNoiseCalculator(noise=noise)
    atoms.calc = calc
    return atoms, calc


def test_ase_result_cache_would_hide_repeat_inference():
    """Red case: without cache invalidation the loop measures nothing."""
    atoms, calc = _counting_atoms(noise=1e-3)
    atoms.get_potential_energy()
    atoms.get_forces()
    after_first = calc.calls
    assert after_first == 1
    for _ in range(5):
        atoms.get_potential_energy()
        atoms.get_forces()
    assert calc.calls == after_first, "ASE cache hit means no new inference"


def test_inference_probe_forces_and_counts_real_calls():
    atoms, calc = _counting_atoms(noise=1e-3)
    with common.InferenceProbe(atoms) as probe:
        probe.run()
        warmup = probe.calls
        for _ in range(4):
            probe.run()
        assert probe.calls == 5
        assert calc.calls == 5, "calculator counter agrees with the probe"
        assert probe.wrapped_calculate is True
    assert warmup == 1
    # the wrapper is removed again
    assert not hasattr(calc, "__dict__") or "calculate" not in calc.__dict__


def test_measure_floor_counts_calls_and_detects_noise():
    atoms, calc = _counting_atoms(noise=0.0)
    deterministic = invariance.measure_floor(atoms, "cpu")
    expected_calls = invariance.REPEAT_CALLS + 1  # warm-up + repetitions
    assert calc.calls == expected_calls
    assert deterministic["real_calculate_calls"] == expected_calls
    assert deterministic["warmup_calculate_calls"] == 1
    assert deterministic["calculate_call_count_verified"] is True
    assert deterministic["energy_spread_eV"] == pytest.approx(0.0, abs=0.0)
    assert deterministic["force_component_spread_eV_A"] == pytest.approx(0.0, abs=0.0)

    noisy_atoms, noisy_calc = _counting_atoms(noise=1e-3)
    noisy = invariance.measure_floor(noisy_atoms, "cpu")
    assert noisy_calc.calls == expected_calls
    assert noisy["energy_spread_eV"] > 0.0
    assert noisy["force_component_spread_eV_A"] > 0.0


def test_tolerance_uses_measured_floor():
    # measured floor above the absolute floor must win (x multiplier)
    assert invariance._tol(5e-9, 1e-9) == pytest.approx(5e-8)
    # degenerate floor falls back to the absolute floor
    assert invariance._tol(0.0, 1e-9) == pytest.approx(1e-9)
    assert invariance._tol(float("nan"), 1e-9) == pytest.approx(1e-9)


# --------------------------------------------------------------------------
# static suite (T4) machinery: CPU known answers only
# --------------------------------------------------------------------------


import static_suite  # noqa: E402


class _CubicHookeCalculator:
    """Exact cubic Hooke's-law calculator for convention tests.

    sigma = C @ eps(voigt, engineering shear), E = V/2 eps.sigma about the
    reference cell.  NOT a real backend: only used for CPU known-answer
    arithmetic tests (taskbook section 47 allows this class of synthetic
    reference for pure-math plumbing checks).
    """

    def __init__(self, reference_cell, c11, c12, c44):
        from ase.calculators.calculator import Calculator, all_changes

        ref = np.array(reference_cell, dtype=float)
        c = np.array(
            [
                [c11, c12, c12, 0.0, 0.0, 0.0],
                [c12, c11, c12, 0.0, 0.0, 0.0],
                [c12, c12, c11, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, c44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, c44, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, c44],
            ]
        )

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces", "stress"]  # noqa: RUF012  # noqa: RUF012

            def calculate(
                self, atoms, properties=("energy",), system_changes=all_changes
            ):
                super().calculate(atoms, properties, system_changes)
                f = atoms.cell.array @ np.linalg.inv(ref).T
                eps = (f + f.T) / 2.0 - np.eye(3)
                # voigt strain with ENGINEERING shear: [xx, yy, zz, yz, xz, xy]
                eps_v = np.array(
                    [
                        eps[0, 0],
                        eps[1, 1],
                        eps[2, 2],
                        2 * eps[1, 2],
                        2 * eps[0, 2],
                        2 * eps[0, 1],
                    ]
                )
                stress_v = c @ eps_v
                volume = atoms.get_volume()
                energy = 0.5 * volume * float(eps_v @ stress_v)
                self.results = {
                    "energy": energy,
                    "forces": np.zeros((len(atoms), 3)),
                    "stress": stress_v,
                }

        self._calc_cls = _Calc

    def get_calculator(self):
        return self._calc_cls()


def _math_ctx(calculator):
    from types import SimpleNamespace

    return SimpleNamespace(calculator=calculator, device="cpu", engine="cpu-math-test")


def test_strain_eps_shear_uses_engineering_convention():
    eps = static_suite.strain_eps("xy", 0.01)
    assert eps[0, 1] == pytest.approx(0.005)
    assert eps[1, 0] == pytest.approx(0.005)
    assert eps[2, 2] == 0.0
    normal = static_suite.strain_eps("xx", 0.0025)
    assert normal[0, 0] == pytest.approx(0.0025)
    assert abs(normal).sum() == pytest.approx(0.0025)


def test_apply_strain_volume_scaling_exact():
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.615, cubic=True)
    for ratio in (0.94, 1.06):
        s = ratio ** (1.0 / 3.0)
        work = atoms.copy()
        work.set_cell(atoms.cell.array * s, scale_atoms=True)
        assert work.get_volume() / atoms.get_volume() == pytest.approx(ratio, rel=1e-12)
    sheared = static_suite.apply_strain(atoms, "xy", 0.01)
    # symmetric pure shear det(I+eps) = 1 - eps_xy^2 -> O(gamma^2) volume change
    assert sheared.get_volume() / atoms.get_volume() == pytest.approx(
        1.0 - 0.005**2, abs=1e-12
    )


def test_cubic_elastic_conventions_recovered_exactly():
    from ase.build import bulk

    c11_ref, c12_ref, c44_ref = 100.0, 60.0, 30.0  # eV/A^3 scaled toy values
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True)
    calc = _CubicHookeCalculator(atoms.cell.array, c11_ref, c12_ref, c44_ref)
    result = static_suite._cubic_elastic(_math_ctx(calc.get_calculator()), atoms, False)
    assert result["C11_eV_A3"] == pytest.approx(c11_ref, rel=1e-8)
    assert result["C12_eV_A3"] == pytest.approx(c12_ref, rel=1e-8)
    assert result["C44_eV_A3"] == pytest.approx(c44_ref, rel=1e-8)
    assert result["max_fit_residual_eV_A3"] == pytest.approx(0.0, abs=1e-10)
    # energy-curvature cross-check must reproduce the same constants
    assert result["energy_curvature"]["C11_energy"] == pytest.approx(c11_ref, rel=1e-8)
    # shear energy carries an O(gamma^4) finite-strain correction
    # (E ~ 0.5 V C44 gamma^2 (1 - gamma^2/4)), so the quadratic-fit
    # recovery is only approximate at the largest sampled strain
    assert result["energy_curvature"]["C44_energy"] == pytest.approx(c44_ref, rel=1e-3)
    assert all(result["born_stability"].values())


def test_cubic_elastic_emt_known_answer():
    # EMT Cu near a=3.615: mechanically stable, B=(C11+2C12)/3 ~ 130 GPa
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.615, cubic=True)
    result = static_suite._cubic_elastic(_math_ctx(EMT()), atoms, False)
    assert result["born_stability"]["C11_minus_C12_positive"]
    assert result["born_stability"]["C11_plus_2C12_positive"]
    assert result["born_stability"]["C44_positive"]
    assert result["B_GPa"] == pytest.approx(134.0, abs=25.0)
    # cross-path consistency between the two C12 estimates (cubic symmetry)
    c12s = result["fits"]
    assert c12s["C12_from_yy"]["slope"] == pytest.approx(
        c12s["C12_from_xx"]["slope"], rel=0.05
    )


def test_thermo_einstein_crystal_limits():
    n_modes = 12
    omega0 = 0.03  # eV
    freqs = np.full(n_modes, omega0)
    out = static_suite._thermo_from_modes(freqs, [10.0, 5000.0])
    assert out["zero_point_energy_eV"] == pytest.approx(0.5 * n_modes * omega0)
    assert out["n_negative_modes"] == 0
    high = next(t for t in out["per_temperature"] if t["T_K"] == 5000.0)
    low = next(t for t in out["per_temperature"] if t["T_K"] == 10.0)
    # Dulong-Petit high-T limit: Cv -> n_modes * kB
    assert high["cv_eV_per_K"] == pytest.approx(
        n_modes * static_suite.KB_EV_PER_K, rel=0.01
    )
    assert low["cv_eV_per_K"] < 0.05 * high["cv_eV_per_K"]
    assert high["entropy_eV_per_K"] > 0.0


def test_thermo_q_grid_normalization_invariant():
    """Thermal sums must be per unit cell, independent of q-grid density.

    Regression: t4f once summed the (8,8,8) MP grid without the 1/N_q
    factor, inflating ZPE/Cv/S/F by 512x.
    """
    n_modes = 6
    omega = np.linspace(0.01, 0.06, n_modes)
    single = static_suite._thermo_from_modes(omega, [300.0])
    for n_q in (8, 64, 512):
        gridded = static_suite._thermo_from_modes(
            np.tile(omega, n_q), [300.0], n_qpoints=n_q
        )
        assert gridded["zero_point_energy_eV"] == pytest.approx(
            single["zero_point_energy_eV"]
        )
        ref = single["per_temperature"][0]
        got = gridded["per_temperature"][0]
        assert got["cv_eV_per_K"] == pytest.approx(ref["cv_eV_per_K"])
        assert got["entropy_eV_per_K"] == pytest.approx(ref["entropy_eV_per_K"])
        assert got["free_energy_eV"] == pytest.approx(ref["free_energy_eV"])
        # raw mode/q counts stay unnormalized
        assert gridded["n_modes"] == n_modes * n_q
    with pytest.raises(ValueError):
        static_suite._thermo_from_modes(omega, [300.0], n_qpoints=5)


def test_thermo_flags_negative_modes():
    freqs = np.array([0.03, 0.02, 0.01, -0.002])
    out = static_suite._thermo_from_modes(freqs, [300.0])
    assert out["n_negative_modes"] == 1
    assert out["n_robust_imaginary_modes"] == 1  # -2 meV is below -0.1 meV
    # ZPE uses |omega| over all modes (documented policy)
    assert out["zero_point_energy_eV"] == pytest.approx(
        0.5 * (0.03 + 0.02 + 0.01 + 0.002)
    )


def test_seeded_displacements_deterministic_and_exact():
    d1 = static_suite.seeded_displacements(16, 0.05)
    d2 = static_suite.seeded_displacements(16, 0.05)
    assert np.array_equal(d1, d2)
    assert np.allclose(np.linalg.norm(d1, axis=1), 0.05)
    d3 = static_suite.seeded_displacements(16, 0.01)
    assert np.allclose(np.linalg.norm(d3, axis=1), 0.01)
    assert not np.allclose(d1, d3)


def test_na3ps4_fixture_pinned_to_mp28782():
    atoms, desc = fixtures.build_extra("na3ps4")
    assert atoms.get_chemical_formula() == "Na6P2S8"
    assert desc["structure_id"] == (
        "7f4e778f75b21f59b9e40efdd1c6ff03cd4cda01a66ff53d40121f3aff433ff4"
    )
    spglib = pytest.importorskip("spglib")
    cell = (atoms.cell.array, atoms.get_scaled_positions(wrap=True), atoms.numbers)
    assert spglib.get_spacegroup(cell, symprec=1e-3) == "P-42_1c (114)"


def test_data_manifest_records_na3ps4_source():
    manifest = json.loads((REPO / "validation/science/data_manifest.json").read_text())
    sources = {s["name"]: s for s in manifest["sources"]}
    assert "alpha-Na3PS4 tetragonal (geometry fixture)" in sources
    entry = sources["alpha-Na3PS4 tetragonal (geometry fixture)"]
    assert entry["upstream_id"] == "mp-28782"
    assert entry["space_group"] == "P-42_1c (114)"
    assert entry["fixture_structure_id"].startswith("7f4e778f")


def test_vacancy_formula_emt_known_answer():
    from ase.build import bulk

    base = bulk("Cu", "fcc", a=3.615, cubic=True)
    supercell = base.repeat((2, 2, 2))
    n_sites = len(supercell)
    bulk_cell = supercell.copy()
    bulk_cell.calc = EMT()
    e_bulk = bulk_cell.get_potential_energy()
    defect = supercell.copy()
    del defect[0]
    defect.calc = EMT()
    e_def = defect.get_potential_energy()
    e_vac = e_def - (n_sites - 1) / n_sites * e_bulk
    # EMT Cu unrelaxed vacancy is a stable reference value for this fixture
    assert e_vac == pytest.approx(1.2534, abs=0.15)
    assert e_vac > 0


def test_surface_gamma_formula_emt_known_answer():
    from ase.build import bulk, fcc111

    prim = bulk("Cu", "fcc", a=3.615, cubic=False)
    prim.calc = EMT()
    e_per_atom = prim.get_potential_energy()
    slab = fcc111("Cu", size=(2, 2, 4), a=3.615, vacuum=10.0)
    area = abs(np.linalg.norm(np.cross(slab.cell[0], slab.cell[1])))
    slab.calc = EMT()
    e_slab = slab.get_potential_energy()
    gamma = (e_slab - len(slab) * e_per_atom) / (2.0 * area) * 16.021766208
    assert gamma == pytest.approx(1.035, abs=0.25)
    assert gamma > 0.0


def test_eos_fit_recovers_known_parameters():
    import ase.eos

    v = np.linspace(40.0, 52.0, 9)
    e_true = ase.eos.birchmurnaghan(v, -4.0, 0.75, 4.8, 46.0)
    out = static_suite._fit_eos(list(v), list(e_true), "birchmurnaghan")
    assert out["fit_converged"]
    assert out["v0_A3"] == pytest.approx(46.0, abs=0.01)
    assert out["b0_GPa"] == pytest.approx(0.75 * 160.21766208, abs=0.5)
    assert out["b0_prime"] == pytest.approx(4.8, abs=0.05)
    assert out["max_abs_residual_eV"] < 1e-8
    assert out["minimum_inside_sampled_range"]
    # cross-form fit must stay close (independent functional form)
    out2 = static_suite._fit_eos(list(v), list(e_true), "pouriertarantola")
    assert out2["b0_GPa"] == pytest.approx(0.75 * 160.21766208, abs=2.0)
    assert out2["b0_prime"] > 0.0


# --------------------------------------------------------------------------
# t8 performance suite plumbing (taskbook section 33)
# --------------------------------------------------------------------------


def test_t8_supercell_plan_builds_requested_sizes():
    """SP scaling sizes are exact Na3PS4 supercells of the pinned unit cell."""
    import performance_suite

    base, _meta = fixtures.build_extra("na3ps4")
    assert len(base) == 16
    for natoms, repeat in performance_suite.SP_SIZES.items():
        atoms, built = performance_suite._build_supercell(natoms)
        assert built == repeat
        assert len(atoms) == natoms
        assert len(atoms) == 16 * repeat[0] * repeat[1] * repeat[2]
        # composition stays a multiple of the pinned unit cell
        na_count = sum(1 for atom in atoms if atom.symbol == "Na")
        assert na_count == 6 * (repeat[0] * repeat[1] * repeat[2])


def test_t8_oom_classifier_matches_only_memory_failures():
    """OOM is 'characterized', everything else must stay 'fail'."""
    import performance_suite

    assert performance_suite._is_oom(Exception("CUDA out of memory. Tried to allocate"))
    assert performance_suite._is_oom(Exception("OOM when allocating tensor"))
    assert performance_suite._is_oom(Exception("OOM when allocating tensor"))
    assert not performance_suite._is_oom(
        Exception("ResourceExhausted: too many open files")
    )
    assert not performance_suite._is_oom(Exception("shape mismatch"))
    assert not performance_suite._is_oom(Exception("nan detected in forces"))
    assert not performance_suite._is_oom(Exception("boom"))
    assert not performance_suite._is_oom(Exception("doom"))


class _HarmonicCountingCalculator(Calculator):
    """E = 0.5 * sum(r^2), F = -r; records every energy it really computed."""

    implemented_properties = ["energy", "forces"]  # noqa: RUF012

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.energies: list[float] = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        pos = np.asarray(atoms.positions, dtype=float)
        energy = float(0.5 * np.sum(pos**2))
        self.energies.append(energy)
        self.results = {"energy": energy, "forces": -pos}


def test_t8_observables_are_measured_on_restored_pristine_structure():
    """R10 red case: the timed sample used a perturbed structure, forces did not.

    The reported energy must belong to the same (pristine, hash-recorded)
    structure as the forces, while the timing loop still uses perturbations
    to defeat the ASE result cache.
    """
    from types import SimpleNamespace

    import engines as engines_mod
    import performance_suite

    ref, _repeat = performance_suite._build_supercell(32)
    pristine_energy = float(0.5 * np.sum(np.asarray(ref.positions, dtype=float) ** 2))
    calc = _HarmonicCountingCalculator()
    ctx = engines_mod.EngineContext(
        engine="mace",
        profile_id="t8-toy",
        profile={
            "model_sha256": "b" * 64,
            "upstream_model_id": "t8-toy",
            "identity": "t8 toy",
        },
        wrapper=None,
        calculator=calc,
        device="cpu",
        dtype="float64",
        task="bulk",
        head=None,
        backend_version="toy",
        framework_version="toy",
    )
    rec = performance_suite._sp_record(ctx, 32, "cpu", SimpleNamespace())
    metrics = rec["metrics"]
    assert metrics["energy_eV"] == pytest.approx(pristine_energy, abs=1e-9)
    assert (
        metrics["observable_structure_id"] == rec["input_structure_id"]
    ), "energy/forces and the recorded structure hash must share one identity"
    assert metrics["observable_inference_calls"] == 1
    assert metrics["timing_samples_perturb_geometry"] is True
    assert metrics["observable_structure_restored"] is True
    # every timing sample plus the observable performed a real backend call,
    # so timing can never be a cache lookup
    assert calc.calls == 1 + performance_suite.WARM_REPS + 1
    # the reported energy is the pristine sample; the timed ones were perturbed
    assert calc.energies[-1] == pytest.approx(pristine_energy, abs=1e-9)
    assert any(
        abs(energy - pristine_energy) > 1e-12 for energy in calc.energies[:-1]
    ), "timing samples must actually perturb the geometry"


def test_t8_record_plumbing_writes_schema_valid_results(tmp_path):
    """A t8 record round-trips through common.result_record/write_result."""
    import performance_suite

    rec = common.result_record(
        case_id="t8_sp_scaling_32",
        test_id="t8_performance",
        status="pass",
        engine="test",
        model_identity="test",
        model_sha256="0" * 64,
        task="bulk",
        head=None,
        dtype="float64",
        device=common.device_record("cpu"),
        input_structure_id="abc",
        parameters={"natoms": 32, "repeat": [2, 1, 1]},
        metrics={
            "natoms": 32,
            "first_call_seconds": 1.0,
            "warm_call_median_seconds": 0.5,
            "atoms_per_second_warm": 64.0,
            "energy_finite": True,
        },
        diagnostics={"reason": None},
    )
    path = common.write_result(rec, tmp_path, tag="t8beta")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["test_id"] == "t8_performance"
    assert payload["status"] == "pass"
    assert payload["metrics"]["natoms"] == 32
    # the real suite must keep the same case naming
    assert performance_suite.SP_SIZES == {32: (2, 1, 1), 128: (2, 2, 2), 512: (4, 4, 2)}


def test_elastic_relaxed_ions_differ_from_clamped():
    """Relaxed-ion elastic constants must actually relax internal forces.

    Regression: workflow_elastic once discarded the relaxed copy returned
    by relax_positions, so the "relaxed-ion" t4d records silently
    duplicated the clamped-ion values (defect found 2026-09-11 while
    rendering the beta report: clamped == relaxed to 1e-13 GPa for all
    engines on the diamond fixture, where the internal-strain correction
    is physically nonzero).
    """
    from types import SimpleNamespace

    from ase.calculators.lj import LennardJones

    atoms, _desc = fixtures.build_fixture("si_diamond")
    ctx = SimpleNamespace(
        calculator=LennardJones(sigma=3.4, epsilon=0.017),
        engine="lj",
        device="cpu",
    )
    # precondition: the clamped xy-sheared geometry has nonzero internal
    # forces, so clamped and relaxed paths genuinely differ here
    work = static_suite.apply_strain(atoms, "xy", 0.01)
    work.calc = ctx.calculator
    assert float(np.abs(work.get_forces()).max()) > 0.01

    clamped = static_suite._cubic_elastic(ctx, atoms, relaxed_ions=False)
    relaxed = static_suite._cubic_elastic(ctx, atoms, relaxed_ions=True)
    assert clamped["fit_mode"] == "clamped-ion"
    assert relaxed["fit_mode"] == "relaxed-ion"
    # diamond symmetry: only the shear constant couples to the internal
    # (optical-mode) relaxation; C11/C12 legitimately coincide
    assert abs(relaxed["C44_GPa"] - clamped["C44_GPa"]) > 1.0


# ---------------------------------------------------------------------------
# T3 report field mapping (review section 0: missing stress must not blank E/F)
# ---------------------------------------------------------------------------


def test_t3_markdown_reads_real_metric_field_names():
    import generate_beta_report as report

    record = {
        "engine": "mace",
        "status": "pass",
        "metrics": {
            "energy_per_atom_eV": {
                "mae": 0.018388498,
                "rmse": 0.043544182,
                "median": 0.009547227,
                "p95": 0.054390283,
            },
            "forces": {
                "component_mae_eV_A": 0.090624042,
                "cosine": {"min": -0.89924578},
            },
            "stress": {"reference_stress_available": False},
            "n_structures": 256,
            "n_failed": 0,
        },
    }
    text = report.md_t3([record], REPO / "validation" / "science" / "t3")
    assert "None" not in text, text
    row = next(line for line in text.splitlines() if line.startswith("| mace "))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[1] != "None" and cells[2] != "None"
    assert cells[5] != "None" and cells[6] != "None"
    assert cells[7] == "256"
    assert "reference stress absent" in text


def test_t3_summary_json_keeps_energy_and_force_metrics_without_stress():
    import generate_beta_report as report

    t3_record = {
        "engine": "uma",
        "status": "pass",
        "metrics": {
            "energy_per_atom_eV": {"mae": 0.02, "rmse": 0.05},
            "forces": {"component_mae_eV_A": 0.1, "cosine": {"min": -0.5}},
            "stress": {"reference_stress_available": False},
            "n_structures": 256,
        },
    }
    tiers = {tier: [] for tier in report.TIER_NAMES}
    tiers["t3"] = [t3_record]
    summary = report._summary_json(Path("."), Path("."), tiers, {}, [])
    uma = summary["omat24"]["uma"]
    assert uma["energy_per_atom"]["mae"] == pytest.approx(0.02)
    assert uma["forces"]["component_mae_eV_A"] == pytest.approx(0.1)
    assert uma["stress"]["reference_stress_available"] is False


def test_software_commit_env_override_is_explicit_and_validated(monkeypatch):
    """A harness run may analyse trajectories produced at an earlier commit."""
    override = "a" * 40

    def _record():
        return common.result_record(
            case_id="c",
            test_id="t1_real_inference",
            status="pass",
            engine="mace",
            model_identity="mace-omat",
            model_sha256="b" * 64,
            task="bulk",
            head=None,
            dtype="float64",
            device={
                "requested": "cpu",
                "actual": "cpu",
                "gpu_name": None,
                "gpu_uuid_hash": None,
            },
            input_structure_id=None,
            parameters={},
            metrics={
                "energy_eV": -1.0,
                "forces_max_abs_eV_A": 0.1,
                "stress_supported": False,
            },
        )

    monkeypatch.delenv(common.SOFTWARE_COMMIT_ENV, raising=False)
    head_record = _record()
    assert head_record["git_commit"] == common._git_commit()
    assert head_record["software_commit_source"] in {"git_head", "unknown"}

    monkeypatch.setenv(common.SOFTWARE_COMMIT_ENV, override)
    overridden = _record()
    assert overridden["git_commit"] == override
    assert overridden["software_commit_source"] == "env_override"
    # the identity hash must follow the recorded software commit
    assert overridden["profile_id"] != head_record["profile_id"]

    monkeypatch.setenv(common.SOFTWARE_COMMIT_ENV, "not-a-sha")
    with pytest.raises(ValueError, match="40-character hex"):
        _record()


def test_analysis_setup_backend_version_does_not_require_engine_import():
    """Regression for the T7 stage-E AttributeError on engines.package_version."""
    import sys
    import types
    from pathlib import Path as _Path

    scripts = (
        _Path(__file__).resolve().parents[3] / "validation" / "science" / "scripts"
    )
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import analysis_suite

    explicit = analysis_suite._analysis_backend_version(
        types.SimpleNamespace(backend_version="9.9.9", engine="mace")
    )
    assert explicit == "9.9.9"
    resolved = analysis_suite._analysis_backend_version(
        types.SimpleNamespace(backend_version=None, engine="mace")
    )
    assert isinstance(resolved, str) and resolved
    source = (scripts / "analysis_suite.py").read_text(encoding="utf-8")
    assert "engines.package_version(" not in source
    assert "common.package_version(engines._backend_dist(" in source
