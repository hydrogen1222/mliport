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
            implemented_properties = ["energy", "forces", "stress"]  # noqa: RUF012

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
    assert not performance_suite._is_oom(Exception("ResourceExhausted: too many open files"))
    assert not performance_suite._is_oom(Exception("shape mismatch"))
    assert not performance_suite._is_oom(Exception("nan detected in forces"))
    assert not performance_suite._is_oom(Exception("boom"))
    assert not performance_suite._is_oom(Exception("doom"))


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
