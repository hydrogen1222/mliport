"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import csv
import gc
import json
import subprocess
import sys
import textwrap
import tracemalloc
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixCom
from ase.io import Trajectory, read

from mliport.config.defaults import BUILTIN_DEFAULTS
from mliport.runners.md import ForceSafetyAbort, MDRunner
from mliport.runners.md_output import (
    AsyncMDOutputWriter,
    MDOutputError,
    MDTrajectorySummary,
)


class _CalculatorStub:
    inference_mode = "turbo"


@pytest.mark.parametrize("ensemble", ["NVT", "NVE"])
def test_md_initializes_velocities_at_target_temperature(tmp_path, ensemble):
    atoms = Atoms(
        "Ar4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
    )
    runner = MDRunner(
        _CalculatorStub(),
        ensemble=ensemble,
        temperature=900.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
    )

    runner._initialize_velocities(atoms)

    assert atoms.get_temperature() == pytest.approx(900.0)
    assert np.sum(atoms.get_momenta(), axis=0) == pytest.approx(np.zeros(3), abs=1e-12)


def _ar4():
    return Atoms(
        "Ar4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
    )


def test_md_seed_makes_velocities_reproducible(tmp_path):
    """Plan section 5.3: a recorded seed must actually drive velocity init."""
    a1, a2, a3 = _ar4(), _ar4(), _ar4()
    r42a = MDRunner(
        _CalculatorStub(),
        temperature=300.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=42,
    )
    r42b = MDRunner(
        _CalculatorStub(),
        temperature=300.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=42,
    )
    r7 = MDRunner(
        _CalculatorStub(),
        temperature=300.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=7,
    )
    r42a._initialize_velocities(a1)
    r42b._initialize_velocities(a2)
    r7._initialize_velocities(a3)
    # Same seed -> identical momenta; different seed -> different.
    assert np.allclose(a1.get_momenta(), a2.get_momenta())
    assert not np.allclose(a1.get_momenta(), a3.get_momenta())


def test_md_velocity_policy_preserve_requires_velocities(tmp_path):
    """preserve must fail loudly when there are no velocities (not silently init)."""
    runner = MDRunner(
        _CalculatorStub(),
        temperature=300.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        velocity_policy="preserve",
    )
    with pytest.raises(ValueError, match="preserve"):
        runner._initialize_velocities(_ar4())


def test_md_velocity_policy_auto_preserves_existing(tmp_path):
    """auto keeps existing velocities instead of re-initializing them."""
    atoms = _ar4()
    atoms.set_momenta(np.full((4, 3), 0.5))
    runner = MDRunner(
        _CalculatorStub(),
        temperature=300.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        velocity_policy="auto",
        com_policy="none",
    )
    runner._initialize_velocities(atoms)
    assert np.allclose(atoms.get_momenta(), np.full((4, 3), 0.5))


def test_md_velocity_policy_preserves_explicit_zero_momenta(tmp_path):
    """An all-zero momenta array is a valid, explicitly supplied restart state."""
    atoms = _ar4()
    atoms.set_momenta(np.zeros((4, 3)))
    runner = MDRunner(
        _CalculatorStub(),
        temperature=300.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        velocity_policy="preserve",
    )
    runner._initialize_velocities(atoms)
    assert np.array_equal(atoms.get_momenta(), np.zeros((4, 3)))


def test_md_velocity_policy_initialize_forces_reinit(tmp_path):
    """initialize re-initializes even when velocities already exist."""
    atoms = _ar4()
    atoms.set_momenta(np.full((4, 3), 0.5))
    runner = MDRunner(
        _CalculatorStub(),
        temperature=300.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        velocity_policy="initialize",
        seed=11,
    )
    runner._initialize_velocities(atoms)
    assert not np.allclose(atoms.get_momenta(), np.full((4, 3), 0.5))
    assert atoms.get_temperature() == pytest.approx(300.0)


@pytest.mark.parametrize("policy", ["auto", "preserve"])
def test_md_restart_momenta_reject_position_changing_pre_relax(tmp_path, policy):
    """A phase-space restart cannot silently alter positions before MD."""
    atoms = _bulk_atoms()
    atoms.set_momenta(np.ones((len(atoms), 3)))
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        ensemble="NVT",
        steps=1,
        output_dir=tmp_path,
        pre_relax=True,
        velocity_policy=policy,
        verbose=False,
    )
    with pytest.raises(ValueError, match="not a strict trajectory restart"):
        runner.run(atoms)


def test_md_rejects_pre_relax_mode(tmp_path):
    """Plan section 5.2: pre_relax_mode must not be silently ignored."""
    with pytest.raises(NotImplementedError, match="pre_relax_mode"):
        MDRunner(
            _CalculatorStub(),
            output_dir=tmp_path,
            pre_relax=False,
            verbose=False,
            pre_relax_mode="positions",
        )


def test_md_rejects_unknown_velocity_policy(tmp_path):
    with pytest.raises(ValueError, match="velocity_policy"):
        MDRunner(
            _CalculatorStub(),
            output_dir=tmp_path,
            pre_relax=False,
            verbose=False,
            velocity_policy="bogus",
        )


# ---------------------------------------------------------------------------
# P1-1: streaming trajectory -- RAM must not grow with frame count (plan 5.5)
# ---------------------------------------------------------------------------
class _FiniteCalc(Calculator):
    """Trivial ASE calculator with optional balanced internal forces."""

    implemented_properties = ["energy", "forces", "stress"]  # noqa: RUF012

    def __init__(self, force_scale: float = 0.0):
        super().__init__()
        self._force_scale = force_scale

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        forces = np.zeros((len(atoms), 3))
        # Equal/opposite forces survive the explicit FixCom constraint. A
        # spatially uniform force is pure COM acceleration and is correctly
        # projected out, so it is not an internal "explosion" test.
        forces[::2, 0] = self._force_scale
        forces[1::2, 0] = -self._force_scale
        self.results = {
            "energy": -1.0,
            "forces": forces,
            "stress": np.zeros(6),
        }


class _RunWrapper:
    """Minimal BaseMLIPCalculator stub that returns a real ASE calculator."""

    def __init__(self, calc, task="bulk", has_stress=True, model_type="stub"):
        self._calc = calc
        self._task = task
        self._has_stress = has_stress
        self._model_type = model_type

    def get_calculator(self):
        return self._calc

    @property
    def task(self):
        return self._task

    @property
    def has_stress(self):
        return self._has_stress

    @property
    def implemented_properties(self):
        return list(self._calc.implemented_properties)

    def info(self):
        return {"model_type": self._model_type, "task": self._task}


@pytest.mark.parametrize("model_type", ["uma", "mace", "dpa", "grace"])
@pytest.mark.parametrize(
    ("ensemble", "thermostat"),
    [
        ("NVE", "LANGEVIN"),
        ("NVT", "LANGEVIN"),
        ("NVT", "BUSSI"),
        ("NVT", "NHC"),
    ],
)
def test_all_backends_share_every_md_integrator_path(
    tmp_path, model_type, ensemble, thermostat
):
    """Level-1 matrix: each backend wrapper can drive each ASE MD method."""
    out = tmp_path / f"{model_type}-{ensemble}-{thermostat}"
    runner = MDRunner(
        _RunWrapper(_FiniteCalc(), model_type=model_type),
        ensemble=ensemble,
        thermostat=thermostat,
        temperature=300.0,
        timestep=0.25,
        steps=2,
        save_interval=1,
        output_dir=out,
        pre_relax=False,
        verbose=False,
        seed=42,
        com_policy="none" if thermostat == "NHC" else "auto",
    )

    results = runner.run(_bulk_atoms(n=4))

    assert np.isfinite(results["energy"])
    assert np.isfinite(results["temperature"])
    assert len(Trajectory(out / "raw" / "trajectory.traj")) == 3
    assert np.isfinite(results["temperature"])
    assert (out / "vasp" / "XDATCAR").is_file()
    assert results["thermostat"] == (thermostat if ensemble == "NVT" else None)


def test_short_nve_has_finite_stable_total_energy(tmp_path):
    """Catch catastrophic timestep/unit mistakes with a conservative toy system."""
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        ensemble="NVE",
        temperature=300.0,
        timestep=0.5,
        steps=10,
        save_interval=1,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=42,
    )

    results = runner.run(_bulk_atoms(n=4))
    total_energies = np.asarray(
        [frame["total_energy"] for frame in results["trajectory"]]
    )

    assert np.all(np.isfinite(total_energies))
    assert np.ptp(total_energies) < 1.0e-10


@pytest.mark.parametrize("model_type", ["uma", "mace", "dpa", "grace"])
def test_all_backends_support_basic_molecule_md_path(tmp_path, model_type):
    atoms = _bulk_atoms(n=4)
    atoms.pbc = False
    runner = MDRunner(
        _RunWrapper(
            _FiniteCalc(), task="molecule", has_stress=False, model_type=model_type
        ),
        ensemble="NVE",
        temperature=300.0,
        steps=1,
        save_interval=1,
        output_dir=tmp_path / model_type,
        pre_relax=False,
        verbose=False,
        seed=7,
    )

    results = runner.run(atoms)

    assert np.isfinite(results["energy"])
    assert results["configurational_stress"] is None
    assert results["total_stress"] is None
    assert results["configurational_pressure_gpa"] is None
    assert results["total_pressure_gpa"] is None


def test_bulk_total_stress_includes_kinetic_contribution(tmp_path):
    atoms = _bulk_atoms(n=4)
    atoms.calc = _FiniteCalc()
    runner = MDRunner(
        _RunWrapper(atoms.calc),
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
    )

    atoms.set_momenta(np.zeros((len(atoms), 3)))
    zero = runner._stress_observables(atoms)
    assert zero["total_stress"] == pytest.approx(zero["configurational_stress"])

    atoms.set_momenta(np.arange(1, 13, dtype=float).reshape(4, 3))
    moving = runner._stress_observables(atoms)
    assert not np.allclose(moving["total_stress"], moving["configurational_stress"])
    assert moving["total_pressure_gpa"] != pytest.approx(
        moving["configurational_pressure_gpa"]
    )


@pytest.mark.parametrize("pbc", [False, [True, True, False]])
def test_non_3d_periodic_md_never_queries_or_reports_bulk_stress(tmp_path, pbc):
    atoms = _bulk_atoms(n=4)
    atoms.pbc = pbc
    atoms.calc = _FiniteCalc()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("stress must not be queried for non-3D PBC")

    atoms.get_stress = fail_if_called
    runner = MDRunner(
        _RunWrapper(atoms.calc),
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
    )
    observables = runner._stress_observables(atoms)
    assert all(value is None for value in observables.values())


def test_timestep_and_saved_time_metadata_are_consistent(tmp_path):
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        ensemble="NVE",
        timestep=0.5,
        steps=4,
        save_interval=2,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=4,
    )
    results = runner.run(_bulk_atoms(n=4))
    assert [frame["time_fs"] for frame in results["trajectory"]] == [0.0, 1.0, 2.0]
    manifest = json.loads((tmp_path / "artifacts.json").read_text())
    assert manifest["trajectory"]["timestep_fs"] == 0.5
    assert manifest["trajectory"]["saved_interval_fs"] == 1.0


def test_pre_relax_nonconvergence_stops_before_md(tmp_path, monkeypatch):
    class NeverConvergedFIRE:
        def __init__(self, atoms, logfile=None):
            self.atoms = atoms
            self.nsteps = 0

        def attach(self, callback, interval=1):
            self.callback = callback

        def run(self, fmax, steps):
            self.nsteps = steps

        def converged(self):
            return False

    monkeypatch.setattr("mliport.runners.md.FIRE", NeverConvergedFIRE)
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        output_dir=tmp_path,
        pre_relax=True,
        pre_relax_steps=2,
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        runner.run(_bulk_atoms(n=4))
    assert not (tmp_path / "raw" / "trajectory.traj").exists()


@pytest.mark.parametrize(
    ("thermostat", "expected_key", "inactive_keys"),
    [
        (
            "LANGEVIN",
            "friction_fs^-1",
            {"bussi_tau_fs", "nhc_tdamp_fs", "nhc_tchain", "nhc_tloop"},
        ),
        (
            "BUSSI",
            "bussi_tau_fs",
            {
                "friction_fs^-1",
                "approx_velocity_damping_time_ps",
                "nhc_tdamp_fs",
                "nhc_tchain",
                "nhc_tloop",
            },
        ),
        (
            "NHC",
            "nhc_tdamp_fs",
            {"friction_fs^-1", "approx_velocity_damping_time_ps", "bussi_tau_fs"},
        ),
    ],
)
def test_md_provenance_records_only_active_thermostat_parameters(
    tmp_path, thermostat, expected_key, inactive_keys
):
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        thermostat=thermostat,
        steps=0,
        output_dir=tmp_path / thermostat,
        pre_relax=False,
        verbose=False,
        seed=12,
        velocity_policy="initialize",
        com_policy="none" if thermostat == "NHC" else "auto",
    )
    results = runner.run(_bulk_atoms(n=4))
    provenance = results["md_provenance"]

    assert provenance["thermostat"] == thermostat
    assert provenance["seed"] == 12
    assert provenance["velocity_policy"] == "initialize"
    assert expected_key in provenance
    assert inactive_keys.isdisjoint(provenance)

    json_data = json.loads(
        (tmp_path / thermostat / "raw" / "mliport_results.json").read_text()
    )
    md_data = json_data["calculation"]["md"]
    assert md_data["thermostat"] == thermostat
    assert expected_key in md_data
    assert inactive_keys.isdisjoint(md_data)


def test_nve_provenance_has_null_thermostat_and_no_coupling_parameters(tmp_path):
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        ensemble="NVE",
        thermostat="not-used",
        friction=-1.0,
        bussi_tau=-1.0,
        nhc_tdamp=-1.0,
        nhc_tchain=0,
        nhc_tloop=0,
        steps=0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=9,
    )
    results = runner.run(_bulk_atoms(n=4))

    assert results["md_provenance"]["thermostat"] is None
    assert set(results["md_provenance"]) == {
        "ensemble",
        "thermostat",
        "temperature",
        "timestep",
        "steps",
        "equilibration_steps",
        "production_steps",
        "total_steps",
        "seed",
        "velocity_policy",
        "com_policy",
        "com_policy_effective",
        "constraints",
        "degrees_of_freedom",
        "force_contract",
    }


def test_bussi_zero_kinetic_energy_has_mliport_error(tmp_path):
    atoms = _bulk_atoms(n=4)
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    runner = MDRunner(
        _CalculatorStub(),
        thermostat="BUSSI",
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
    )
    with pytest.raises(ValueError, match="Bussi/CSVR requires non-zero"):
        runner._build_dynamics(atoms)


def _bulk_atoms(n=8):
    """A simple periodic noble-gas cell for MD."""
    return Atoms(
        f"Ar{n}",
        positions=np.linspace(0, 5, n * 3).reshape(n, 3),
        cell=[10, 10, 10],
        pbc=True,
    )


def test_md_streams_trajectory_to_disk(tmp_path):
    """Plan 5.5: full frames go to disk (trajectory.traj + XDATCAR), scalars
    to md.csv, and only scalar summaries stay in RAM."""
    wrapper = _RunWrapper(_FiniteCalc())
    runner = MDRunner(
        wrapper,
        ensemble="NVE",
        temperature=300.0,
        steps=10,
        save_interval=2,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=42,
    )
    results = runner.run(_bulk_atoms())

    frames = results["trajectory"]
    nframes = len(frames)
    assert nframes == 10 // 2 + 1  # step 0 + every 2nd step (ASE fires at step 0)
    # In-memory trajectory is scalars-only: NO per-frame atoms.copy().
    for f in frames:
        assert "atoms" not in f
        assert set(f) == {
            "step",
            "time_fs",
            "phase",
            "energy",
            "kinetic_energy",
            "total_energy",
            "temperature",
            "volume",
            "max_force_raw_eV_A",
            "max_force_applied_eV_A",
            "configurational_stress",
            "total_stress",
            "configurational_pressure_gpa",
            "total_pressure_gpa",
        }
    # Full frames are on disk in trajectory.traj.
    assert results["trajectory_path"] == str(tmp_path / "raw" / "trajectory.traj")
    assert len(list(Trajectory(results["trajectory_path"]))) == nframes
    # XDATCAR was streamed using the standard VASP configuration marker and
    # is readable by ASE as a multi-frame trajectory.
    xdatcar = (tmp_path / "vasp" / "XDATCAR").read_text()
    assert xdatcar.count("Direct configuration=") == nframes
    assert "# Step:" not in xdatcar
    assert (
        len(read(tmp_path / "vasp" / "XDATCAR", index=":", format="vasp-xdatcar"))
        == nframes
    )
    # md.csv was streamed: header + nframes rows.
    with open(tmp_path / "raw" / "md.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "step",
        "time_fs",
        "phase",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "total_energy_eV",
        "temperature_K",
        "volume_A3",
        "max_force_raw_eV_A",
        "max_force_applied_eV_A",
        "configurational_stress_xx_eV_A3",
        "configurational_stress_yy_eV_A3",
        "configurational_stress_zz_eV_A3",
        "configurational_stress_yz_eV_A3",
        "configurational_stress_xz_eV_A3",
        "configurational_stress_xy_eV_A3",
        "total_stress_xx_eV_A3",
        "total_stress_yy_eV_A3",
        "total_stress_zz_eV_A3",
        "total_stress_yz_eV_A3",
        "total_stress_xz_eV_A3",
        "total_stress_xy_eV_A3",
        "configurational_pressure_GPa",
        "total_pressure_GPa",
    ]
    assert len(rows) == nframes + 1
    assert results["md_csv_path"] == str(tmp_path / "raw" / "md.csv")

    # VASP-like OUTCAR contains every saved frame and the requested physical
    # observables; the final CONTCAR is a readable VASP structure.
    outcar = (tmp_path / "vasp" / "OUTCAR").read_text()
    assert "NOT A NATIVE VASP OUTCAR" in outcar
    assert outcar.count(" POSITION ") == nframes
    assert outcar.count("MLIPORT-TOTEN") == nframes
    assert "stress tensor" in outcar
    assert len(read(tmp_path / "vasp" / "CONTCAR", format="vasp")) == 8
    assert not (tmp_path / "analysis").exists()
    manifest = json.loads((tmp_path / "artifacts.json").read_text())
    assert manifest["schema"] == "mliport.md-artifacts/3"
    assert manifest["status"] == "completed"
    assert manifest["trajectory"]["frames"] == nframes
    assert manifest["trajectory"]["positions_convention"] == "unwrapped"
    assert manifest["artifacts"]["xdatcar"]["path"] == "vasp/XDATCAR"


def test_md_records_every_step_independently_of_trajectory_interval(tmp_path):
    logs: list[str] = []
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        ensemble="NVE",
        temperature=300.0,
        steps=3,
        save_interval=2,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=42,
        log_fn=lambda message, _level: logs.append(message),
    )

    results = runner.run(_bulk_atoms())

    step_logs = [message for message in logs if "/3: E =" in message]
    assert [
        int(message.split("Step", 1)[1].split("/", 1)[0]) for message in step_logs
    ] == [0, 1, 2, 3]
    assert any(
        "Save interval:    2 steps (trajectory frames)" in message for message in logs
    )
    assert any(
        "Thermodynamic log interval: 1 step (buffered disk flush)" in message
        for message in logs
    )
    assert [frame["step"] for frame in results["trajectory"]] == [0, 2]


def test_400_atom_every_step_output_preserves_vacf_velocities(tmp_path):
    """A VACF-style save cadence retains all CPU frames and velocities."""
    from mliport.logger import LiveRunLogger

    steps = 16
    with LiveRunLogger(tmp_path / "run.log") as run_logger:
        runner = MDRunner(
            _RunWrapper(_FiniteCalc()),
            ensemble="NVE",
            temperature=300.0,
            timestep=1.0,
            steps=steps,
            save_interval=1,
            output_dir=tmp_path,
            pre_relax=False,
            verbose=False,
            seed=42,
            log_fn=run_logger,
        )
        results = runner.run(_bulk_atoms(n=400))

    expected_frames = steps + 1
    with Trajectory(results["trajectory_path"], mode="r") as trajectory:
        assert len(trajectory) == expected_frames
        assert all(frame.has("momenta") for frame in trajectory)

    run_log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert (
        len([line for line in run_log.splitlines() if f"/{steps}: E =" in line])
        == expected_frames
    )
    assert (tmp_path / "vasp" / "XDATCAR").read_text().count(
        "Direct configuration="
    ) == expected_frames
    assert (tmp_path / "vasp" / "OUTCAR").read_text().count(
        " POSITION "
    ) == expected_frames
    assert len(results["trajectory"]) == expected_frames


def test_md_equilibration_and_production_phase_metadata(tmp_path):
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        ensemble="NVE",
        temperature=300.0,
        equilibration_steps=2,
        steps=3,
        save_interval=1,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=42,
    )

    results = runner.run(_bulk_atoms())

    assert results["equilibration_steps"] == 2
    assert results["production_steps"] == 3
    assert results["md_steps"] == 5
    assert [frame["phase"] for frame in results["trajectory"]] == [
        "equilibration",
        "equilibration",
        "production",
        "production",
        "production",
        "production",
    ]
    with (tmp_path / "raw" / "md.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["phase"] for row in rows] == [
        "equilibration",
        "equilibration",
        "production",
        "production",
        "production",
        "production",
    ]
    manifest = json.loads((tmp_path / "artifacts.json").read_text())
    assert manifest["trajectory"]["production_start_step"] == 2
    assert manifest["trajectory"]["production_start_frame"] == 2


def test_md_long_trajectory_has_bounded_resident_summary_memory(tmp_path):
    """Frame summaries are disk-backed and the live queue is strictly bounded."""

    def _resident_for(steps, out_dir):
        gc.collect()
        tracemalloc.start()
        # pytest captures warnings with an "always" filter and retains every
        # WarningMessage for the item. With NumPy >= 2.5, ASE's Atoms.copy()
        # (via SinglePointCalculator) emits a deprecation warning per frame,
        # which the harness would then book-keep as per-frame retention.
        # Suppress warnings inside the measurement window so this test
        # measures mliport's own retention -- the actual disk-backed-summary
        # contract -- not the harness's warning records.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runner = MDRunner(
                _RunWrapper(_FiniteCalc()),
                ensemble="NVE",
                temperature=300.0,
                steps=steps,
                save_interval=1,
                output_dir=out_dir,
                pre_relax=False,
                verbose=False,
                seed=1,
            )
            res = runner.run(_bulk_atoms(256))
        gc.collect()
        resident, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return resident, res

    resident_short, res_short = _resident_for(20, tmp_path / "short")
    resident_long, res_long = _resident_for(200, tmp_path / "long")

    assert isinstance(res_short["trajectory"], MDTrajectorySummary)
    assert isinstance(res_long["trajectory"], MDTrajectorySummary)
    assert AsyncMDOutputWriter.QUEUE_MAX_FRAMES == 8
    assert all("atoms" not in f for f in res_short["trajectory"])
    assert all("atoms" not in f for f in res_long["trajectory"])
    assert len(res_long["trajectory"]) == 200 + 1
    assert len(res_short["trajectory"]) == 20 + 1
    # Current retained Python memory does not grow per frame. Peak allocator
    # memory is unsuitable here because ASE's ULM writer expands its on-disk
    # offset table geometrically at frame-count thresholds.
    #
    # Budget calibration (warning-free measurement window, 20 vs 200
    # frames): CPython 3.10/3.11 sit at ~0 net growth and 3.12 shows a
    # systematic ~270 B/frame of interpreter-level per-object overhead.
    # A real leak in this path is KB-scale per frame (a copied positions
    # array alone is ~6 KB), so 512 B/frame still fails hard on any
    # regression while tolerating the 3.12 allocator baseline.
    marginal_per_frame = (resident_long - resident_short) / (200 - 20)
    assert marginal_per_frame < 512, (
        f"resident marginal memory {marginal_per_frame:.0f} B/frame exceeds "
        "the disk-backed-summary budget"
    )


def test_md_output_failure_aborts_instead_of_dropping_frames(tmp_path, monkeypatch):
    """A writer-thread error must stop MD and leave failed provenance."""

    class FailingTrajectoryWriter:
        def __init__(self, *_args, **_kwargs):
            pass

        def write(self, _atoms):
            raise OSError("simulated full filesystem")

        def close(self):
            pass

    monkeypatch.setattr(
        "mliport.runners.md_output.AseTrajectoryWriter",
        FailingTrajectoryWriter,
    )
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        ensemble="NVE",
        steps=10,
        save_interval=1,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=2,
        write_outcar=False,
        write_xdatcar=False,
    )

    with pytest.raises(MDOutputError, match="simulated full filesystem"):
        runner.run(_bulk_atoms())

    manifest = json.loads((tmp_path / "artifacts.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "output_error"


# ---------------------------------------------------------------------------
# P1-2: finite-large-force abort threshold sourced from safety
# ---------------------------------------------------------------------------
def test_md_fmax_abort_defaults_to_safety():
    """The explosion-guard threshold defaults to safety.fmax_abort, not a
    magic literal (plan section 5.7)."""
    runner = MDRunner(_CalculatorStub(), output_dir=".", pre_relax=False, verbose=False)
    assert runner.fmax_abort == BUILTIN_DEFAULTS["safety"]["fmax_abort"]
    assert runner.fmax_abort == 20.0


def test_md_fmax_abort_checkpoints_and_marks_manifest_aborted(tmp_path):
    """The named abort threshold must stop, checkpoint, and report the cause."""
    wrapper = _RunWrapper(_FiniteCalc(force_scale=6.0))  # 6 eV/Å > threshold
    runner = MDRunner(
        wrapper,
        ensemble="NVE",
        temperature=300.0,
        steps=100,
        save_interval=1000,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=1,
        fmax_abort=5.0,
    )
    with pytest.raises(ForceSafetyAbort, match="Force safety abort"):
        runner.run(_bulk_atoms())

    manifest = json.loads((tmp_path / "artifacts.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["error"]["type"] == "force_safety_abort"
    assert manifest["error"]["step"] == 0
    assert manifest["error"]["max_force_eV_A"] == pytest.approx(6.0)
    assert manifest["trajectory"]["last_step"] == 0
    assert (tmp_path / "vasp" / "CONTCAR").is_file()
    assert "Status:              aborted" in (tmp_path / "vasp" / "OUTCAR").read_text()


def test_nvt_seed_reproduces_entire_stochastic_trajectory(tmp_path):
    """The seed must drive Langevin kicks, not only initial velocities."""

    def run(seed, name):
        out = tmp_path / name
        runner = MDRunner(
            _RunWrapper(_FiniteCalc()),
            ensemble="NVT",
            temperature=300.0,
            timestep=0.5,
            friction=0.01,
            steps=5,
            save_interval=1,
            output_dir=out,
            pre_relax=False,
            verbose=False,
            seed=seed,
        )
        runner.run(_bulk_atoms())
        return np.asarray(
            [a.positions for a in Trajectory(out / "raw" / "trajectory.traj")]
        )

    same_a = run(42, "same-a")
    same_b = run(42, "same-b")
    different = run(7, "different")
    assert np.array_equal(same_a, same_b)
    assert not np.array_equal(same_a, different)


def test_md_uses_explicit_com_constraint_for_consistent_temperature_dof(tmp_path):
    atoms = _bulk_atoms()
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        ensemble="NVT",
        temperature=300,
        steps=0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=3,
    )
    runner._ensure_com_constraint(atoms)
    runner._initialize_velocities(atoms)
    assert any(isinstance(c, FixCom) for c in atoms.constraints)
    assert atoms.get_number_of_degrees_of_freedom() == 3 * len(atoms) - 3
    assert atoms.get_temperature() == pytest.approx(300.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timestep": 0.0}, "timestep"),
        ({"steps": -1}, "steps"),
        ({"save_interval": 0}, "save_interval"),
        ({"fmax_abort": 0.0}, "fmax_abort"),
        ({"ensemble": "NVT", "friction": 0.0}, "friction"),
        ({"ensemble": "NVT", "thermostat": "BUSSI", "bussi_tau": 0.0}, "bussi_tau"),
        ({"ensemble": "NVT", "thermostat": "NHC", "nhc_tdamp": 0.0}, "nhc_tdamp"),
        ({"ensemble": "NVT", "thermostat": "NHC", "nhc_tchain": 0}, "nhc_tchain"),
        ({"ensemble": "NVT", "thermostat": "NHC", "nhc_tloop": 0}, "nhc_tloop"),
        ({"ensemble": "NVT", "thermostat": "unknown"}, "thermostat"),
    ],
)
def test_md_rejects_nonphysical_parameters(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        MDRunner(
            _CalculatorStub(),
            output_dir=tmp_path,
            pre_relax=False,
            verbose=False,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# R08: KeyboardInterrupt must drain the writer thread and mark the run cancelled
# ---------------------------------------------------------------------------

_R08_SCRIPT = """
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mliport.runners.md import MDRunner


class InterruptOnSecondCall(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.calls += 1
        if self.calls >= 2:
            raise KeyboardInterrupt
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": -1.0,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }


class Wrapper:
    task = "bulk"
    has_stress = True
    model_type = "stub"

    def __init__(self, calc):
        self._calc = calc

    def get_calculator(self):
        return self._calc

    def info(self):
        return {"model_type": "stub", "task": "bulk"}


out = Path(sys.argv[1])
atoms = Atoms(
    "Ar4",
    positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
    cell=[20.0, 20.0, 20.0],
    pbc=True,
)
runner = MDRunner(
    Wrapper(InterruptOnSecondCall()),
    ensemble="NVE",
    temperature=300.0,
    timestep=0.25,
    steps=20,
    save_interval=1,
    output_dir=out,
    pre_relax=False,
    verbose=False,
    seed=7,
    com_policy="none",
)
try:
    runner.execute(atoms)
except KeyboardInterrupt:
    print("INTERRUPTED")
else:
    print("NOT_INTERRUPTED")
"""


def test_md_keyboard_interrupt_exits_quickly_and_marks_cancelled(tmp_path):
    """Red case: the non-daemon writer thread kept the process alive >4 s.

    The interrupted run must drain its output stream, publish status
    ``cancelled`` and let the interpreter exit without external killing.
    """
    out = tmp_path / "md_interrupt"
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_R08_SCRIPT), str(out)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "INTERRUPTED" in completed.stdout
    artifacts = json.loads((out / "artifacts.json").read_text(encoding="utf-8"))
    assert artifacts["status"] == "cancelled"
    # No half-written frame files: the trajectory is readable or absent.
    trajectory = out / "raw" / "trajectory.traj"
    if trajectory.exists():
        assert len(Trajectory(trajectory)) >= 0
    # OUTCAR, when requested, carries the cancelled trailer rather than a
    # completed one.
    outcar = out / "vasp" / "OUTCAR"
    if outcar.exists():
        text = outcar.read_text(encoding="utf-8", errors="ignore")
        assert "cancelled" in text.lower()
