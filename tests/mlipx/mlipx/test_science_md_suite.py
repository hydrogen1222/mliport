"""Known-answer and end-to-end tests for the MD validation tier (t6).

Everything here runs on CPU with the ASE EMT calculator driving the product
``mlipx.runners.md.MDRunner`` exactly as the GPU sweep does (the runner only
needs a wrapper exposing ``get_calculator``/``task``/``info``).

Scientific pins:
- The suite's 32-atom Cu fixture is deterministic; the seeded
  MaxwellBoltzmann velocity draw (velocity_policy="initialize", fixed seed)
  gives bitwise-identical initial kinetic energy across runs and timesteps.
- NVE drift slope must improve as the timestep decreases (no global drift
  cutoff is invented; the gate is the broad trend between extreme steps).
- Same-seed Langevin reruns must agree bitwise on EMT (delta == 0.0); the
  suite allows <= 1e-6 eV for runtimes that cannot reproduce exactly.
- The NHC rejection matrix must refuse every wrong-DOF combination
  (NHC+FixAtoms, NHC+auto COM, NHC+COM constraint, compound constraint)
  with an exception instead of silently integrating the wrong DOF set.
- FixAtoms positions stay fixed exactly (bitwise 0.0 A deviation) and the
  COM projector keeps the centre of mass fixed to float64 roundoff.
- The force-safety abort configures its threshold at half the measured
  pre-flight force of a controlled displacement, aborts cleanly (status
  "aborted", error type "force_safety_abort", unsafe frame + CONTCAR
  written) and never continues into NaN/Inf territory.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixCom

_SCRIPTS = Path(__file__).resolve().parents[3] / "validation" / "science" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

md_suite = importlib.import_module("md_suite")

EMT_MAX_FORCE_PRISTINE_EV_A = 1.0  # pristine Cu(32) forces are ~0.1 eV/A


class _Wrapper:
    """Minimal product calculator wrapper for MDRunner (bulk/periodic)."""

    task = "bulk"
    has_stress = False

    def __init__(self, calc):
        self._calc = calc

    def get_calculator(self):
        return self._calc

    def info(self):
        return {"engine": "emt-test", "task": "bulk"}


@pytest.fixture()
def emt():
    return EMT()


@pytest.fixture()
def ctx(emt):
    from engines import EngineContext

    return EngineContext(
        engine="emt-test",
        profile_id="emt_cpu",
        profile={
            "model_sha256": "0" * 64,
            "identity": "EMT-CPU-test",
            "upstream_model_id": "emt",
        },
        wrapper=_Wrapper(emt),
        calculator=emt,
        device="cpu",
        dtype="float64",
        task="bulk",
        head=None,
        backend_version="emt",
        framework_version="ase",
    )


@pytest.fixture()
def args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep any stray relative writes inside tmp
    import types

    return types.SimpleNamespace(out=str(tmp_path / "results"), tag=None)


@pytest.fixture(autouse=True)
def _short_budgets(monkeypatch):
    """Shrink pinned budgets; the workflow logic and gates are unchanged."""
    monkeypatch.setattr(md_suite, "_SAVE_INTERVAL", 10)
    monkeypatch.setattr(md_suite, "NVE_TIMESTEPS_FS", (0.25, 1.0, 2.0))
    monkeypatch.setattr(md_suite, "NVE_RUN_PS", 0.05)
    monkeypatch.setattr(md_suite, "LANGEVIN_STEPS", 100)
    monkeypatch.setattr(md_suite, "BUSSI_STEPS", 100)
    monkeypatch.setattr(md_suite, "NHC_STEPS", 100)
    monkeypatch.setattr(md_suite, "CONSTRAINT_STEPS", 100)


def _load_record(out: Path, case_id: str) -> dict:
    matches = sorted(out.glob(f"{case_id}*.json"))
    assert matches, f"no record for {case_id} in {out}"
    return json.loads(matches[-1].read_text())


# ---------------------------------------------------------------------------
# unit-level known answers
# ---------------------------------------------------------------------------


def test_drift_slope_units_and_linear_fit():
    """E rises linearly 0->1 eV over 100 fs for 2 atoms: 5 eV/atom/ps."""
    t = np.linspace(0.0, 100.0, 11)
    e = t / 100.0
    slope = md_suite._drift_slope_eV_per_atom_ps(t, e, natoms=2)
    assert slope == pytest.approx(5.0, rel=1e-9)
    # fewer than two samples cannot define a slope
    assert math.isnan(md_suite._drift_slope_eV_per_atom_ps(t[:1], e[:1], 2))


def test_cu32_fixture_is_deterministic_32_cu():
    a = md_suite.cu32()
    b = md_suite.cu32()
    assert len(a) == 32
    assert sorted(a.get_chemical_symbols()) == ["Cu"] * 32
    assert np.array_equal(a.positions, b.positions)
    assert a.pbc.all()


def test_fixed_layer_indices_are_deterministic_bottom_layer():
    atoms = md_suite.cu32()
    fixed = md_suite._fixed_layer_indices(atoms)
    a = 3.615
    assert len(fixed) == 8
    assert fixed == sorted(fixed)
    assert all(atoms.positions[i, 2] < 0.4 * a for i in fixed)
    assert md_suite._fixed_layer_indices(atoms) == fixed


def test_parse_md_csv_reads_product_header():
    import csv as _csv

    from mlipx.runners.md_output import MD_CSV_HEADER

    rows = [
        {"step": "0", "time_fs": "0.0", "potential_energy_eV": "-1.0",
         "kinetic_energy_eV": "1.2", "total_energy_eV": "0.2",
         "temperature_K": "300", "max_force_raw_eV_A": "0.1",
         "max_force_applied_eV_A": "0.05"},
        {"step": "10", "time_fs": "10.0", "potential_energy_eV": "-1.1",
         "kinetic_energy_eV": "1.3", "total_energy_eV": "0.2",
         "temperature_K": "310", "max_force_raw_eV_A": "0.2",
         "max_force_applied_eV_A": "0.06"},
    ]
    # write with the product header contract, then parse
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=MD_CSV_HEADER)
        writer.writeheader()
        for row in rows:
            full = {k: "" for k in MD_CSV_HEADER}
            full.update(row)
            writer.writerow(full)
        tmp_path_csv = Path(fh.name)
    parsed = md_suite._parse_md_csv(tmp_path_csv)
    assert parsed["step"].tolist() == [0.0, 10.0]
    assert parsed["total_energy_eV"].tolist() == [0.2, 0.2]
    assert parsed["temperature_K"].tolist() == [300.0, 310.0]
    assert parsed["max_force_raw_eV_A"].tolist() == [0.1, 0.2]


def test_run_suite_registers_t6():
    import run_suite

    assert run_suite.TIER_COMMANDS["t6"] == ["md_suite.py"]
    assert set(md_suite.DEFAULT_ORDER) == set(md_suite.WORKFLOW_FUNCS)
    assert len(md_suite.DEFAULT_ORDER) == 6


# ---------------------------------------------------------------------------
# product-level rejection matrix (no integration attempted)
# ---------------------------------------------------------------------------


def test_product_rejects_nhc_with_constraints_and_com_removal(emt, tmp_path):
    from mlipx.runners.md import MDRunner

    atoms = md_suite.cu32()
    fixed = md_suite._fixed_layer_indices(atoms)
    cases = [
        ("nhc_fixatoms", "nhc", "none", md_suite._atoms_with(
            md_suite.cu32(), FixAtoms(indices=fixed))),
        ("nhc_auto_com", "nhc", "auto", md_suite.cu32()),
        ("nhc_com_constraint", "nhc", "constraint", md_suite.cu32()),
    ]
    for label, thermostat, com_policy, work in cases:
        with pytest.raises(ValueError):
            MDRunner(
                _Wrapper(emt),
                ensemble="nvt",
                thermostat=thermostat,
                timestep=1.0,
                steps=1,
                seed=md_suite.MD_SEED,
                com_policy=com_policy,
                velocity_policy="initialize",
                pre_relax=False,
                output_dir=tmp_path / "_probe",
                verbose=False,
            ).run(work)


def test_product_rejects_compound_constraint_with_com_constraint(emt, tmp_path):
    from mlipx.runners.md import MDRunner

    compound = md_suite.cu32()
    compound.set_constraint(
        [FixAtoms(indices=md_suite._fixed_layer_indices(compound)), FixCom()]
    )
    with pytest.raises(ValueError):
        MDRunner(
            _Wrapper(emt),
            ensemble="nvt",
            thermostat="langevin",
            timestep=1.0,
            steps=1,
            seed=md_suite.MD_SEED,
            com_policy="constraint",
            velocity_policy="initialize",
            pre_relax=False,
            output_dir=tmp_path / "_probe",
            verbose=False,
        ).run(compound)


# ---------------------------------------------------------------------------
# end-to-end workflow tests through record() on EMT/CPU
# ---------------------------------------------------------------------------


def test_t6a_nve_dt_sweep_trend_and_seed_identity(ctx, args):
    out = Path(args.out)
    rc = md_suite.workflow_nve_dt_sweep(ctx, args, out)
    assert rc == 0
    rec = _load_record(out, "t6_nve")
    assert rec["status"] == "pass"
    assert rec["engine"] == "emt-test"
    assert rec["model_identity"] == "EMT-CPU-test"
    m = rec["metrics"]
    assert m["all_finite"] is True
    assert m["abort_state"] == "none"
    assert m["drift_improves_with_timestep"] is True
    slopes = m["drift_slopes_eV_atom_ps"]
    assert abs(slopes["0.25"]) < abs(slopes["2"])  # dt keys are :g-formatted
    # identical initial conditions across timesteps: same seed -> same KE(0)
    ke0 = set(m["initial_kinetic_energy_eV_by_dt"].values())
    assert len(ke0) == 1
    assert next(iter(ke0)) > 0.0
    for v in m["per_timestep"].values():
        assert v["all_finite"] is True
        assert 0.0 < v["temperature_min_K"] <= v["temperature_max_K"]


def test_t6b_langevin_same_seed_reproducibility(ctx, args):
    out = Path(args.out)
    rc = md_suite.workflow_langevin(ctx, args, out)
    assert rc == 0
    rec = _load_record(out, "t6_langevin")
    # EMT on CPU is deterministic: the two seeded runs must agree bitwise
    assert rec["status"] == "pass"
    m = rec["metrics"]
    assert m["same_seed_max_energy_delta_eV"] == pytest.approx(0.0, abs=md_suite.REPRO_MAX_DELTA_EV)
    assert m["all_finite"] is True
    assert m["seed_recorded"] == md_suite.MD_SEED
    prov = m["thermostat_metadata"]
    assert prov["thermostat"] == "LANGEVIN"
    assert prov["seed"] == md_suite.MD_SEED
    assert prov["velocity_policy"] == "initialize"
    assert prov["com_policy_effective"] == "constraint"  # auto -> FixCom
    assert prov["degrees_of_freedom"] == 3 * 32 - 3
    assert m["initial_kinetic_energy_eV"] > 0.0


def test_t6c_bussi_target_and_fixatoms_compatibility(ctx, args):
    out = Path(args.out)
    rc = md_suite.workflow_bussi(ctx, args, out)
    assert rc == 0
    rec = _load_record(out, "t6_bussi")
    assert rec["status"] == "pass"
    m = rec["metrics"]
    assert m["all_finite"] is True
    prov = m["thermostat_metadata"]
    assert prov["thermostat"] == "BUSSI"
    # unconstrained run: auto COM removal -> 3N - 3
    assert m["degrees_of_freedom"] == 3 * 32 - 3
    assert m["com_policy_effective"] == "constraint"
    # FixAtoms run: product keeps COM policy at "initialize_only" when
    # positional constraints exist; DOF = 3N - 3*8 (8 fixed atoms)
    assert m["fixatoms_run_dof"] == 3 * 32 - 3 * 8
    assert "FixAtoms" in m["fixatoms_run_constraints"]
    assert m["initial_kinetic_energy_eV"] > 0.0  # Bussi needs non-zero KE


def test_t6d_nhc_allowed_run_and_rejection_matrix(ctx, args):
    out = Path(args.out)
    rc = md_suite.workflow_nhc(ctx, args, out)
    assert rc == 0
    rec = _load_record(out, "t6_nhc")
    assert rec["status"] == "pass"
    m = rec["metrics"]
    assert m["all_finite"] is True
    # positive run: NHC only validated unconstrained, com_policy='none'
    assert m["degrees_of_freedom"] == 3 * 32
    prov = m["thermostat_metadata"]
    assert prov["thermostat"] == "NHC"
    assert prov["com_policy"] == "none"
    assert m["all_unsupported_combinations_rejected"] is True
    cases = {r["case"]: r for r in m["rejection_matrix"]}
    assert set(cases) == {
        "nhc_fixatoms",
        "nhc_auto_com",
        "nhc_com_constraint",
        "langevin_compound_constraint",
    }
    for label, r in cases.items():
        assert r["rejected"] is True, label
        assert r["exception"] == "ValueError", label
        assert r["message"], label


def test_t6e_constraint_exactness_and_dof(ctx, args):
    out = Path(args.out)
    rc = md_suite.workflow_constraints(ctx, args, out)
    assert rc == 0
    rec = _load_record(out, "t6_constraints")
    assert rec["status"] == "pass"
    m = rec["metrics"]
    # fixed atoms stay fixed exactly (bitwise) across every saved frame
    assert m["fixatoms_max_position_deviation_A"] == pytest.approx(0.0, abs=md_suite.EXACTNESS_TOL_A)
    # COM projector holds to float64 roundoff
    assert m["com_max_drift_A"] <= md_suite.COM_DRIFT_TOL_A
    assert m["fixatoms_dof"] == m["fixatoms_dof_expected"] == 3 * 32 - 3 * 8
    assert m["com_dof"] == m["com_dof_expected"] == 3 * 32 - 3
    assert m["free_dof"] == m["free_dof_expected"] == 3 * 32
    assert m["all_finite"] is True


def test_t6f_force_safety_abort_is_clean(ctx, args):
    out = Path(args.out)
    rc = md_suite.workflow_force_safety(ctx, args, out)
    assert rc == 0
    rec = _load_record(out, "t6_force_safety")
    assert rec["status"] == "pass"
    m = rec["metrics"]
    # the displacement creates a genuinely high-force state
    measured = m["measured_preflight_force_eV_A"]
    assert measured > EMT_MAX_FORCE_PRISTINE_EV_A
    # threshold = documented fraction of the measured force
    assert m["threshold_configured_eV_A"] == pytest.approx(
        md_suite.FORCE_SAFETY_FRACTION * measured, rel=1e-9
    )
    assert m["abort_raised"] is True
    assert m["artifacts_manifest_status"] == "aborted"
    assert m["artifacts_error"]["type"] == "force_safety_abort"
    assert m["abort_max_force_raw_eV_A"] > m["threshold_configured_eV_A"]
    assert 0 <= m["abort_atom_index"] < 32
    assert m["unsafe_frame_checkpointed"] is True
    assert m["contcar_written"] is True
    # the abort step is the first evaluated state of the displaced cell
    assert m["abort_step"] <= md_suite.FORCE_SAFETY_STEPS
    # displaced partner recorded for provenance
    assert 0 <= m["partner_atom_index"] < 32 and m["partner_atom_index"] != 0
