"""ASE-only isotropic MTK NPT regression contracts (no MLIP weights needed)."""

from __future__ import annotations

import csv
import json
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixCom
from ase.io import Trajectory

from mliport.analysis.dataset import TrajectoryDataset
from mliport.config import resolve_config
from mliport.config.defaults import BUILTIN_DEFAULTS
from mliport.config.schema import get_schema
from mliport.engine import CalculationEngine, EngineConfig
from mliport.runners.md import MDRunner


class VolumeCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        volume = atoms.get_volume()
        # U = B (V-V0)^2/(2V0), sigma = dU/dV, p = -trace(sigma)/3.
        derivative = 0.02 * (volume - 512.0) / 512.0
        self.results = {
            "energy": 0.01 * (volume - 512.0) ** 2 / 512.0,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.array([derivative, derivative, derivative, 0, 0, 0]),
        }


class Wrapper:
    def __init__(self, stress=True, calculator=None):
        self.has_stress = stress
        self.calculator = calculator or VolumeCalculator()
        self.task = "bulk"
        self.loads = 0

    def get_calculator(self):
        self.loads += 1
        return self.calculator

    def info(self):
        return {"model_type": "mace", "task": "bulk"}


def structure():
    return Atoms(
        "Ar4",
        positions=[[1, 1, 1], [3, 1, 1], [1, 3, 1], [1, 1, 3]],
        cell=[8, 8, 8],
        pbc=True,
    )


def runner(wrapper, output_dir, **kwargs):
    args = {
        "ensemble": "NPT",
        "temperature": 300,
        "pressure_gpa": 1.5,
        "npt_tdamp": 20.0,
        "npt_pdamp": 40.0,
        "timestep": 0.5,
        "steps": 24,
        "equilibration_steps": 4,
        "save_interval": 2,
        "seed": 42,
        "pre_relax": False,
        "output_dir": output_dir,
        "verbose": False,
        "write_outcar": False,
        "write_json": False,
    }
    args.update(kwargs)
    return MDRunner(wrapper, **args)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("pressure_gpa", 5.0),
        ("npt_tdamp", 100.0),
        ("npt_pdamp", 1000.0),
        ("npt_tchain", 7),
        ("npt_pchain", 4),
        ("npt_tloop", 2),
        ("npt_ploop", 2),
    ],
)
def test_npt_only_options_are_rejected_outside_npt(tmp_path, key, value):
    with pytest.raises(ValueError, match="NPT-only options require ensemble='NPT'"):
        resolve_config(
            calc_type="md",
            cli={
                "model_type": "mace",
                "model_path": str(tmp_path / "unused.model"),
                key: value,
            },
        )


def test_incar_file_path_accepts_npt_and_its_keys(tmp_path):
    """`mliport run` validates the INCAR before resolving it; NPT must pass both."""
    from mliport.config.incar import IncarConfig

    incar = IncarConfig.from_string(
        "CALC_TYPE = MD\nMODEL_TYPE = mace\n"
        f"MODEL_PATH = {tmp_path / 'unused.model'}\n"
        "MD_ENSEMBLE = NPT\nPRESSURE_GPA = 0.5\nNPT_TDAMP = 50\nNPT_PDAMP = 500\n"
    )
    assert incar.validate() == []
    resolved = resolve_config(calc_type="md", incar=dict(incar.items()))
    assert resolved.run_options["ensemble"] == "NPT"
    assert resolved.run_options["pressure_gpa"] == 0.5
    assert resolved.run_options["npt_tdamp"] == 50.0
    assert resolved.run_options["npt_pdamp"] == 500.0


def test_fairchem_alias_is_accepted_for_npt(tmp_path):
    resolved = resolve_config(
        calc_type="md",
        cli={
            "model_type": "fairchem",
            "model_path": str(tmp_path / "unused.model"),
            "ensemble": "NPT",
            "pressure_gpa": 0.0,
        },
    )
    engine = CalculationEngine(EngineConfig.from_resolved(resolved))
    assert engine.config.model_type == "fairchem"


def test_schema_npt_parameters():
    assert get_schema().resolve("ENSEMBLE").coerce("NPT") == "NPT"
    for name in (
        "pressure_gpa",
        "npt_tdamp",
        "npt_pdamp",
        "npt_tchain",
        "npt_pchain",
        "npt_tloop",
        "npt_ploop",
    ):
        assert get_schema().resolve(name) is not None
        assert name in BUILTIN_DEFAULTS["md"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("temperature", 0),
        ("npt_tdamp", 0),
        ("npt_pdamp", -1),
        ("pressure_gpa", float("nan")),
        ("pressure_gpa", float("inf")),
        ("npt_tchain", 0),
        ("npt_pchain", 0),
        ("npt_tloop", 0),
        ("npt_ploop", 0),
        ("write_trajectory", False),
        ("thermostat", "NHC"),
    ],
)
def test_bad_npt_parameter_fails_before_load(tmp_path, key, value):
    wrapper = Wrapper()
    with pytest.raises(ValueError):
        runner(wrapper, tmp_path, **{key: value})
    assert wrapper.loads == 0


@pytest.mark.parametrize("pbc", [(True, True, False), (False, False, False)])
def test_npt_requires_bulk_pbc(tmp_path, pbc):
    atoms = structure()
    atoms.pbc = pbc
    wrapper = Wrapper()
    with pytest.raises(ValueError, match="full 3D PBC"):
        runner(wrapper, tmp_path).run(atoms)
    assert wrapper.loads == 0


@pytest.mark.parametrize("constraint", [FixCom(), FixAtoms(indices=[0])])
def test_npt_rejects_constraints_before_load(tmp_path, constraint):
    atoms = structure()
    atoms.set_constraint(constraint)
    wrapper = Wrapper()
    with pytest.raises(ValueError, match="does not support ASE constraints"):
        runner(wrapper, tmp_path).run(atoms)
    assert wrapper.loads == 0


def test_npt_stress_capability_required_even_if_not_exported(tmp_path):
    with pytest.raises(ValueError, match="must implement stress"):
        runner(Wrapper(stress=False), tmp_path, write_stress=False).run(structure())


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_npt_rejects_nonfinite_stress(tmp_path, invalid):
    class BadStress(VolumeCalculator):
        def calculate(self, atoms, properties, system_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["stress"] = np.full(6, invalid)

    with pytest.raises(ValueError, match="finite six-component stress"):
        runner(Wrapper(calculator=BadStress()), tmp_path).run(structure())


def test_npt_rejects_left_handed_cell(tmp_path):
    atoms = structure()
    atoms.set_cell(np.diag([-8, 8, 8]))
    wrapper = Wrapper()
    with pytest.raises(ValueError, match="positive nonsingular cell"):
        runner(wrapper, tmp_path).run(atoms)
    assert wrapper.loads == 0


def test_npt_pressure_units_com_and_mtk_setpoint(tmp_path):
    atoms = structure()
    md = runner(Wrapper(), tmp_path, pressure_gpa=2.5)
    md._configure_com_policy(atoms)
    assert atoms.constraints == []
    assert md.com_policy_effective == "initialize_only"
    assert md.md_degrees_of_freedom == 9
    md._initialize_velocities(atoms)
    np.testing.assert_allclose(atoms.get_momenta().sum(axis=0), 0, atol=1e-10)
    atoms.calc = VolumeCalculator()
    dyn = md._build_dynamics(atoms)
    assert dyn._pressure_au == pytest.approx(2.5 * units.GPa)
    assert dyn._temperature_K == pytest.approx(225.0)


def test_npt_variable_cell_io_and_transport_gate(tmp_path):
    result = runner(Wrapper(), tmp_path, write_json=True).run(structure())
    assert result["ensemble"] == "NPT"
    assert result["thermostat"] == "MTK"
    assert result["target_pressure_GPa"] == 1.5
    assert result["trajectory_frame_count"] > 2
    assert not (tmp_path / "vasp" / "XDATCAR").exists()
    assert (tmp_path / "vasp" / "CONTCAR").exists()
    with Trajectory(tmp_path / "raw" / "trajectory.traj") as traj:
        frames = list(traj)
    assert len(frames) == result["trajectory_frame_count"]
    assert all(frame.has("momenta") for frame in frames)
    assert np.ptp([frame.get_volume() for frame in frames]) > 1e-9
    with (tmp_path / "raw" / "md.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and "total_pressure_GPa" in rows[0]
    assert np.isfinite(float(rows[-1]["total_pressure_GPa"]))
    manifest = json.loads((tmp_path / "artifacts.json").read_text())
    assert manifest["trajectory"]["variable_cell"] is True
    assert manifest["trajectory"]["transport_eligible"] is False
    assert manifest["trajectory"]["positions_convention"] == "unknown"
    dataset = TrajectoryDataset.load(tmp_path)
    assert dataset.positions_convention == "unknown"
    with pytest.raises(ValueError, match="NPT variable-cell trajectories"):
        TrajectoryDataset.load(tmp_path, positions_convention="unwrapped")
    assert "xdatcar" not in manifest["artifacts"]
    assert result["md_provenance"]["target_pressure_eV_A3"] == pytest.approx(
        1.5 * units.GPa
    )
    md_json = json.loads((tmp_path / "raw" / "mliport_results.json").read_text())[
        "calculation"
    ]["md"]
    assert md_json["target_pressure_GPa"] == pytest.approx(1.5)
    assert md_json["npt"]["npt_algorithm"] == "isotropic_mtk"
    assert md_json["npt"]["target_pressure_eV_A3"] == pytest.approx(1.5 * units.GPa)
    assert md_json["npt"]["npt_tdamp_fs"] == pytest.approx(20.0)
    assert md_json["npt"]["npt_pdamp_fs"] == pytest.approx(40.0)
    assert md_json["npt"]["variable_cell"] is True
    assert md_json["npt"]["transport_eligible"] is False


@pytest.mark.parametrize("ensemble", ["NVE", "NVT"])
def test_legacy_fixed_cell_exports_unchanged(tmp_path, ensemble):
    result = runner(
        Wrapper(),
        tmp_path,
        ensemble=ensemble,
        steps=2,
        equilibration_steps=0,
        write_xdatcar=True,
    ).run(structure())
    assert result["ensemble"] == ensemble
    assert (tmp_path / "vasp" / "XDATCAR").exists()
    meta = json.loads((tmp_path / "artifacts.json").read_text())["trajectory"]
    assert meta["variable_cell"] is False
    assert meta["transport_eligible"] is True
    assert meta["positions_convention"] == "unwrapped"
