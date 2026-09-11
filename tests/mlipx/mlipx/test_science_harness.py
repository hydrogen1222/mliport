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
import transforms  # noqa: E402

MANIFEST = REPO / "validation" / "science" / "model_manifest.json"
FIXTURE_MANIFEST = REPO / "validation" / "science" / "cases" / "manifests" / "fixtures.json"


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
    got_voigt = transforms.rotate_stress_voigt(transforms.matrix_to_voigt(sigma), rot_z90)
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
        (REPO / "validation" / "science" / "schemas" / "summary.schema.json").read_text()
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
        "device": {"requested": "cpu", "actual": "cpu", "gpu_name": None, "gpu_uuid_hash": None},
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

def _raw_record(wrapper="mlipx.calculators", **kw):
    rec = _record(**kw)
    rec["diagnostics"] = {"wrapper_module": wrapper}
    return rec


def test_aggregate_flags_foreign_wrapper(tmp_path):

    foreign = _raw_record(wrapper="some.silent.emt.fallback")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(foreign))
    records, problems = aggregate.load_records(tmp_path)
    assert not problems
    violations = aggregate.harness_violations(records)
    assert violations and "some.silent.emt.fallback" in violations[0]

    # every mlipx wrapper module must be whitelisted -- keeps the test honest
    # the whitelist must cover real mlipx wrapper modules and nothing else
    assert "mlipx.calculators.mace_calc" in common.ALLOWED_WRAPPER_MODULES
    assert "mlipx.calculators.dpa_calc" in common.ALLOWED_WRAPPER_MODULES
    assert "mlipx.calculators.grace_calc" in common.ALLOWED_WRAPPER_MODULES
    assert all(m.startswith("mlipx.") for m in common.ALLOWED_WRAPPER_MODULES)

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


def test_fd_dof_selection_deterministic_and_capped():
    atoms, _, forces, _ = _cu_distorted_emt()
    a = finite_difference.select_force_dofs(atoms, forces, finite_difference.MAX_FORCE_DOFS)
    b = finite_difference.select_force_dofs(atoms, forces, finite_difference.MAX_FORCE_DOFS)
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


def test_tolerance_uses_measured_floor():
    # measured floor above the absolute floor must win (x multiplier)
    assert invariance._tol(5e-9, 1e-9) == pytest.approx(5e-8)
    # degenerate floor falls back to the absolute floor
    assert invariance._tol(0.0, 1e-9) == pytest.approx(1e-9)
    assert invariance._tol(float("nan"), 1e-9) == pytest.approx(1e-9)
