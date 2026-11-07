"""Unit tests for GPUMD input generation (no GPU, no gpumd binary)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms, units
from ase.io import read

from mliport.gpumd.input import (
    ASE_TIME_UNIT_FS,
    build_model_xyz,
    build_run_in,
    momenta_to_velocities_ang_per_fs,
    parse_nep_model,
    run_in_provenance,
    validate_elements,
)


@pytest.fixture()
def nep_file(tmp_path):
    model = tmp_path / "nep.txt"
    model.write_text("nep4 2 Ba O\ncutoff 8 4 232 26\n", encoding="utf-8")
    return model


def _periodic_atoms(symbols=("H", "O"), positions=None):
    if positions is None:
        positions = [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]
    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=np.eye(3) * 10.0,
        pbc=True,
    )


def test_single_point_run_in_exact_lines(nep_file):
    text = build_run_in(calc_type="sp", nep_path=nep_file)
    assert text == (
        f"potential {nep_file.resolve()}\n"
        "velocity 1\n"
        "ensemble nve\n"
        "time_step 0\n"
        "dump_xyz 1 sp.xyz force virial precision double\n"
        "run 1\n"
    )


def test_optimization_run_in_maps_optimizer_case_insensitively(nep_file):
    text = build_run_in(
        calc_type="opt",
        nep_path=nep_file,
        optimizer="FiRe",
        fmax=0.05,
        max_steps=100,
    )
    assert "minimize fire 0.05 100\n" in text
    assert "dump_xyz 1 opt.xyz force virial precision double\n" in text
    assert text.endswith("run 1\n")


def test_optimization_rejects_unsupported_optimizer(nep_file):
    with pytest.raises(ValueError, match="native GPUMD implementation"):
        build_run_in(
            calc_type="opt",
            nep_path=nep_file,
            optimizer="BFGS",
            fmax=0.05,
            max_steps=100,
        )
    with pytest.raises(ValueError, match="requires an optimizer"):
        build_run_in(
            calc_type="opt",
            nep_path=nep_file,
            fmax=0.05,
            max_steps=100,
        )


def test_md_nve_run_in_exact_lines(nep_file):
    text = build_run_in(
        calc_type="md",
        nep_path=nep_file,
        ensemble="NVE",
        temperature=300.0,
        timestep_fs=1.0,
        steps=20,
        seed=7,
        dump_interval=10,
        restart_interval=5,
    )
    assert text == (
        f"potential {nep_file.resolve()}\n"
        "velocity 300 seed 7\n"
        "time_step 1\n"
        "ensemble nve\n"
        "dump_thermo 10\n"
        "dump_xyz 10 traj.xyz precision double force virial mass "
        "unwrapped_position\n"
        "dump_restart 5\n"
        "run 20\n"
    )


@pytest.mark.parametrize(
    ("thermostat", "kwargs", "expected_ensemble", "expected_coup"),
    [
        ("LANGEVIN", {"friction_per_fs": 0.01}, "nvt_lan", 100),
        ("BUSSI", {"bussi_tau_fs": 50.0}, "nvt_bdp", 50),
        ("NHC", {"nhc_tdamp_fs": 25.0}, "nvt_nhc", 25),
    ],
)
def test_nvt_thermostat_mapping(
    nep_file, thermostat, kwargs, expected_ensemble, expected_coup
):
    common = {
        "calc_type": "md",
        "nep_path": nep_file,
        "ensemble": "NVT",
        "temperature": 300.0,
        "timestep_fs": 1.0,
        "steps": 10,
        "thermostat": thermostat,
        "dump_interval": 5,
    }
    text = build_run_in(**common, **kwargs)
    assert f"ensemble {expected_ensemble} 300 300 {expected_coup}\n" in text
    provenance = run_in_provenance(**common, **kwargs)
    assert provenance["T_coup"] == expected_coup
    assert provenance["thermostat_mapping"]["ensemble"] == expected_ensemble


def test_t_coup_rounding_uses_native_time_step(nep_file):
    # tau_T = 1 / 0.3 = 3.333... fs, dt = 0.5 fs -> 6.666... -> 7
    text = build_run_in(
        calc_type="md",
        nep_path=nep_file,
        ensemble="NVT",
        temperature=300.0,
        timestep_fs=0.5,
        steps=10,
        thermostat="LANGEVIN",
        friction_per_fs=0.3,
        dump_interval=5,
    )
    assert "ensemble nvt_lan 300 300 7\n" in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thermostat": "LANGEVIN", "friction_per_fs": 10.0},
        {"thermostat": "BUSSI", "bussi_tau_fs": 0.4},
        {"thermostat": "NHC", "nhc_tdamp_fs": 0.1},
    ],
)
def test_t_coup_below_one_fails_closed(nep_file, kwargs):
    with pytest.raises(ValueError, match="T_coup >= 1"):
        build_run_in(
            calc_type="md",
            nep_path=nep_file,
            ensemble="NVT",
            temperature=300.0,
            timestep_fs=1.0,
            steps=10,
            dump_interval=5,
            **kwargs,
        )


def test_nvt_without_thermostat_fails_closed(nep_file):
    with pytest.raises(ValueError, match="Unsupported NVT thermostat"):
        build_run_in(
            calc_type="md",
            nep_path=nep_file,
            ensemble="NVT",
            temperature=300.0,
            timestep_fs=1.0,
            steps=10,
            dump_interval=5,
        )
    with pytest.raises(ValueError, match="Unsupported NVT thermostat"):
        build_run_in(
            calc_type="md",
            nep_path=nep_file,
            ensemble="NVT",
            temperature=300.0,
            timestep_fs=1.0,
            steps=10,
            thermostat="ANDERSEN",
            dump_interval=5,
        )


def test_unsupported_ensemble_fails_closed(nep_file):
    with pytest.raises(ValueError, match="Unsupported ensemble"):
        build_run_in(
            calc_type="md",
            nep_path=nep_file,
            ensemble="NPT",
            temperature=300.0,
            timestep_fs=1.0,
            steps=10,
        )


def test_run_in_provenance_records_exact_arguments(nep_file):
    provenance = run_in_provenance(
        calc_type="md",
        nep_path=nep_file,
        ensemble="NVT",
        temperature=300.0,
        timestep_fs=0.5,
        steps=20,
        thermostat="LANGEVIN",
        friction_per_fs=0.01,
        seed=42,
        dump_interval=10,
        restart_interval=20,
        unwrapped=True,
    )
    assert provenance["calc_type"] == "md"
    assert provenance["nep_path"] == str(nep_file)
    assert provenance["nep_path_resolved"] == str(nep_file.resolve())
    assert provenance["T_coup"] == 200  # tau_T=100 fs / dt=0.5 fs
    assert provenance["seed"] == 42
    assert provenance["dump_interval"] == 10
    assert provenance["restart_interval"] == 20
    assert provenance["thermostat_mapping"] == {
        "thermostat": "LANGEVIN",
        "ensemble": "nvt_lan",
        "tau_param": "friction_per_fs",
        "tau_T_fs": 100.0,
        "T_coup": 200,
    }


def test_model_xyz_round_trip_preserves_order_cell_and_pbc(tmp_path):
    atoms = _periodic_atoms(
        symbols=("O", "H", "H"),
        positions=[[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
    )
    destination = build_model_xyz(atoms, tmp_path / "model.xyz")
    parsed = read(destination)
    assert parsed.get_chemical_symbols() == ["O", "H", "H"]
    assert np.allclose(parsed.get_cell(), atoms.get_cell())
    assert tuple(parsed.pbc) == (True, True, True)
    assert np.allclose(parsed.get_positions(), atoms.get_positions())
    header = destination.read_text(encoding="utf-8").splitlines()[1]
    assert "Lattice=" in header
    assert 'pbc="T T T"' in header
    assert "vel:R:3" not in header


def test_model_xyz_writes_physical_velocity_known_answer(tmp_path):
    atoms = _periodic_atoms(symbols=("H",), positions=[[0.0, 0.0, 0.0]])
    # 1.0 amu*Angstrom/fs of *physical* momentum.  ASE momenta are expressed
    # in its internal time unit, so the stored value is 1 / ase.units.fs.
    atoms.set_momenta(np.array([[1.0 / units.fs, 0.0, 0.0]]))
    destination = build_model_xyz(atoms, tmp_path / "model.xyz")
    parsed = read(destination)
    assert parsed.arrays["vel"][0, 0] == pytest.approx(1.0 / 1.008, rel=1e-12)
    header = destination.read_text(encoding="utf-8").splitlines()[1]
    assert "vel:R:3" in header


def test_ase_momenta_unit_factor_is_applied():
    momenta = np.array([[1.0, 0.0, 0.0]])
    masses = np.array([1.008])
    velocities = momenta_to_velocities_ang_per_fs(momenta, masses)
    assert velocities[0, 0] == pytest.approx(
        (1.0 / 1.008) * ASE_TIME_UNIT_FS, rel=1e-12
    )
    # Round-trip through the physical kinetic energy: 0.5 m v^2 with v in
    # Angstrom/fs must equal ASE's 0.5 p^2/m in eV.
    amu_ang2_per_fs2_to_ev = units._amu * 1e-20 / 1e-30 / units._e
    kinetic_from_velocity = (
        0.5 * masses[0] * velocities[0, 0] ** 2 * amu_ang2_per_fs2_to_ev
    )
    kinetic_from_momentum = 0.5 * momenta[0, 0] ** 2 / masses[0]
    assert kinetic_from_velocity == pytest.approx(kinetic_from_momentum, rel=1e-12)


def test_model_xyz_rejects_partial_pbc(tmp_path):
    atoms = _periodic_atoms()
    atoms.pbc = (True, True, False)
    with pytest.raises(ValueError, match="pbc|periodic"):
        build_model_xyz(atoms, tmp_path / "model.xyz")


def test_model_xyz_rejects_nan_positions(tmp_path):
    atoms = _periodic_atoms()
    positions = atoms.get_positions()
    positions[0, 0] = float("nan")
    atoms.set_positions(positions)
    with pytest.raises(ValueError, match="NaN"):
        build_model_xyz(atoms, tmp_path / "model.xyz")


def test_model_xyz_rejects_singular_cell(tmp_path):
    atoms = _periodic_atoms()
    atoms.set_cell([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="singular"):
        build_model_xyz(atoms, tmp_path / "model.xyz")


def test_model_xyz_rejects_zero_volume(tmp_path):
    atoms = _periodic_atoms()
    atoms.set_cell(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="singular|volume"):
        build_model_xyz(atoms, tmp_path / "model.xyz")


def test_validate_elements_lists_both_sets():
    atoms = Atoms("FeO", positions=[[0, 0, 0], [1, 1, 1]], pbc=True)
    with pytest.raises(ValueError) as excinfo:
        validate_elements(atoms, ["Ba", "Zr", "O"])
    message = str(excinfo.value)
    assert "Fe" in message
    assert "Ba" in message and "Zr" in message and "O" in message
    validate_elements(atoms, ["Fe", "O"])


def test_parse_nep_model_header(tmp_path):
    model = tmp_path / "nep.txt"
    model.write_text(
        "nep4 3 Ba Zr O\ncutoff 8 4 232 26\nn_max 8 6\n",
        encoding="utf-8",
    )
    parsed = parse_nep_model(model)
    assert parsed["model_format"] == "nep4"
    assert parsed["model_version"] == 4
    assert parsed["element_types"] == ["Ba", "Zr", "O"]
    assert parsed["cutoffs"] == {"radial_A": 8.0, "angular_A": 4.0}
    assert parsed["cutoff_max_A"] == 8.0
