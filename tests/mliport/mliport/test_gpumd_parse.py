"""Unit tests for the fail-closed GPUMD output parsers."""

from __future__ import annotations

import numpy as np
import pytest
from ase.io import read

from mliport.gpumd.parse import (
    EV_A3_TO_GPA,
    GPUMDParseError,
    iter_extxyz_frames,
    md_frames_summary,
    optimization_result,
    parse_extxyz_frames,
    parse_thermo_out,
    single_point_from_dump,
    voigt_stress,
    write_normalized_trajectory,
)

_THERMO_HEADER = """\
# dump_thermo 10
# format_version 1
# num_atoms 2
# dt_output 10.0000000000 fs
# columns T KE PE sxx syy szz syz sxz sxy ax ay az bx by bz cx cy cz
"""


def _write_thermo(path, rows):
    text = _THERMO_HEADER + "\n".join(rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _frame(
    *,
    energy=-1.5,
    time=12.5,
    step=7,
    unwrapped=True,
    velocities=True,
    forces=True,
    positions=None,
    unwrapped_positions=None,
    cells="10 0 0 0 10 0 0 0 10",
    virial="100 0 0 0 200 0 0 0 300",
    stress="0.1 0 0 0 0.2 0 0 0 0.3",
    force_values=None,
    natoms_override=None,
):
    if positions is None:
        positions = [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]
    if unwrapped_positions is None:
        unwrapped_positions = [[10.1, 0.2, 0.3], [11.1, 1.2, 1.3]]
    properties = "species:S:1:pos:R:3"
    if forces:
        properties += ":forces:R:3"
    if unwrapped:
        properties += ":unwrapped_position:R:3"
    if velocities:
        properties += ":vel:R:3"
    header_parts = [
        f"Time={time}",
        'pbc="T T T"',
        f'Lattice="{cells}"',
        f"energy={energy}",
        f'virial="{virial}"' if virial is not None else "",
        f'stress="{stress}"' if stress is not None else "",
        f"Properties={properties}",
        f"step={step}" if step is not None else "",
    ]
    header = " ".join(part for part in header_parts if part)
    if natoms_override is not None:
        header += f" natoms={natoms_override}"
    lines = ["2", header]
    if force_values is None:
        force_values = [[0.10, -0.20, 0.30], [-0.40, 0.50, -0.60]]
    velocity_values = [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]
    for index, symbol in enumerate(["Ba", "O"]):
        columns = [f"{value:.8f}" for value in positions[index]]
        if forces:
            columns.extend(f"{value:.8f}" for value in force_values[index])
        if unwrapped:
            columns.extend(f"{value:.8f}" for value in unwrapped_positions[index])
        if velocities:
            columns.extend(f"{value:.8f}" for value in velocity_values[index])
        lines.append(symbol + " " + " ".join(columns))
    return "\n".join(lines) + "\n"


def test_parse_thermo_out_named_columns_and_step(tmp_path):
    path = _write_thermo(
        tmp_path / "thermo.out",
        [
            "  2.3599016907e+02 1.2201649389e+00 -2.5791221213e+02 "
            "3.1006944846e+01 3.0971007649e+01 3.1185575039e+01 "
            "-4.4245917911e-02 -1.1825265383e-01 1.4552110994e-01 "
            "8.0181236910e+00 0.0000000000e+00 0.0000000000e+00 "
            "0.0000000000e+00 8.0181236910e+00 0.0000000000e+00 "
            "0.0000000000e+00 0.0000000000e+00 8.0181236910e+00",
            "  1.8955965704e+02 9.8010035020e-01 -2.5751824200e+02 "
            "3.1246865049e+01 3.0933118380e+01 3.1294332382e+01 "
            "-1.7166079755e-01 -1.2380803003e-01 1.0396211428e-01 "
            "8.0181236910e+00 0.0000000000e+00 0.0000000000e+00 "
            "0.0000000000e+00 8.0181236910e+00 0.0000000000e+00 "
            "0.0000000000e+00 0.0000000000e+00 8.0181236910e+00",
        ],
    )
    parsed = parse_thermo_out(path)
    assert parsed["T"].tolist() == pytest.approx([235.99016907, 189.55965704])
    assert parsed["KE"].tolist() == pytest.approx([1.2201649389, 0.9801003502])
    assert parsed["PE"].tolist() == pytest.approx([-257.91221213, -257.51824200])
    assert parsed["sxx"].tolist() == pytest.approx([31.006944846, 31.246865049])
    assert parsed["cz"].tolist() == pytest.approx([8.0181236910, 8.0181236910])
    assert parsed["step"].tolist() == [1, 2]
    assert len(parsed) == 19  # 18 documented columns + step


def test_parse_thermo_out_rejects_wrong_field_count(tmp_path):
    path = _write_thermo(tmp_path / "thermo.out", ["1 2 3"])
    with pytest.raises(GPUMDParseError, match="expected 18 fields"):
        parse_thermo_out(path)


def test_parse_thermo_out_rejects_non_finite(tmp_path):
    row = " ".join(["1.0"] * 17 + ["nan"])
    path = _write_thermo(tmp_path / "thermo.out", [row])
    with pytest.raises(GPUMDParseError, match="non-finite"):
        parse_thermo_out(path)


def test_parse_thermo_out_rejects_empty_and_missing_header(tmp_path):
    empty = tmp_path / "empty.out"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(GPUMDParseError, match="empty"):
        parse_thermo_out(empty)
    no_header = tmp_path / "no_header.out"
    no_header.write_text(" ".join(["1.0"] * 18) + "\n", encoding="utf-8")
    with pytest.raises(GPUMDParseError, match="columns"):
        parse_thermo_out(no_header)


def test_parse_thermo_out_rejects_unknown_columns(tmp_path):
    text = _THERMO_HEADER.replace("# columns T KE PE", "# columns T KE PE extra")
    path = tmp_path / "thermo.out"
    path.write_text(text + " ".join(["1.0"] * 19) + "\n", encoding="utf-8")
    with pytest.raises(GPUMDParseError, match="unsupported thermo columns"):
        parse_thermo_out(path)


def test_extxyz_frame_header_and_conversion(tmp_path):
    path = tmp_path / "traj.xyz"
    path.write_text(_frame(), encoding="utf-8")
    frames = parse_extxyz_frames(path)
    assert len(frames) == 1
    frame = frames[0]
    assert frame.energy_eV == pytest.approx(-1.5)
    assert frame.time_fs == pytest.approx(12.5)
    assert frame.step == 7
    assert tuple(frame.atoms.pbc) == (True, True, True)
    assert np.allclose(frame.atoms.get_cell(), np.eye(3) * 10.0)
    assert frame.atoms.get_chemical_symbols() == ["Ba", "O"]
    assert np.allclose(frame.atoms.get_positions(), [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
    assert np.allclose(frame.unwrapped_positions, [[10.1, 0.2, 0.3], [11.1, 1.2, 1.3]])
    assert np.allclose(frame.forces, [[0.10, -0.20, 0.30], [-0.40, 0.50, -0.60]])
    assert np.allclose(frame.velocities, [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
    assert np.allclose(frame.virial_eV, np.diag([100.0, 200.0, 300.0]))
    # GPUMD stress is the negative of ASE stress.
    assert np.allclose(frame.stress_eV_A3, -np.diag([0.1, 0.2, 0.3]))
    assert np.allclose(frame.stress_GPa, -np.diag([0.1, 0.2, 0.3]) * EV_A3_TO_GPA)
    assert np.allclose(frame.extra["native_stress_eV_A3"], np.diag([0.1, 0.2, 0.3]))
    assert frame.extra["raw_header"].startswith("Time=12.5")


def test_single_point_from_dump_known_answer(tmp_path):
    path = tmp_path / "sp.xyz"
    path.write_text(_frame(), encoding="utf-8")
    result = single_point_from_dump(path)
    assert result["energy"] == pytest.approx(-1.5)
    assert result["energy_per_atom"] == pytest.approx(-0.75)
    assert result["natoms"] == 2
    assert result["stress"] == pytest.approx([-0.1, -0.2, -0.3, 0.0, 0.0, 0.0])
    assert result["stress_source"] == "header"
    assert np.allclose(result["positions"], [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])


def test_single_point_requires_exactly_one_frame(tmp_path):
    path = tmp_path / "sp.xyz"
    path.write_text(_frame() + _frame(energy=-2.0), encoding="utf-8")
    with pytest.raises(GPUMDParseError, match="exactly one SP frame"):
        single_point_from_dump(path)


def test_voigt_order_is_xx_yy_zz_yz_xz_xy():
    matrix = np.array(
        [
            [1.0, 6.0, 5.0],
            [6.0, 2.0, 4.0],
            [5.0, 4.0, 3.0],
        ]
    )
    assert voigt_stress(matrix).tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_multi_frame_append_and_split_style_file(tmp_path):
    appended = tmp_path / "traj.xyz"
    appended.write_text(
        _frame(time=1.0, step=1, energy=-1.0) + _frame(time=2.0, step=2, energy=-1.1),
        encoding="utf-8",
    )
    frames = parse_extxyz_frames(appended)
    assert [frame.time_fs for frame in frames] == [1.0, 2.0]
    assert [frame.energy_eV for frame in frames] == [-1.0, -1.1]
    split = tmp_path / "traj_0001.xyz"
    split.write_text(_frame(time=5.0, step=5), encoding="utf-8")
    assert len(parse_extxyz_frames(split)) == 1


def test_streaming_generator_matches_list(tmp_path):
    path = tmp_path / "traj.xyz"
    path.write_text(_frame() + _frame(time=13.5, step=8), encoding="utf-8")
    streamed = [frame.energy_eV for frame in iter_extxyz_frames(path)]
    listed = [frame.energy_eV for frame in parse_extxyz_frames(path)]
    assert streamed == listed


@pytest.mark.parametrize(
    "text",
    [
        "",
        "2\n",
        "2\nTime=0\n1 2 3\n",
        _frame(natoms_override=3),
        _frame().replace("species:S:1:pos:R:3", "species:S:1"),
    ],
)
def test_truncated_or_inconsistent_frames_raise(tmp_path, text):
    path = tmp_path / "bad.xyz"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(GPUMDParseError):
        parse_extxyz_frames(path)


def test_non_finite_frame_values_raise(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text(
        _frame().replace("0.10000000 0.20000000 0.30000000", "nan 0.2 0.3"),
        encoding="utf-8",
    )
    with pytest.raises(GPUMDParseError, match="non-finite"):
        parse_extxyz_frames(path)


def _write_opt_case(tmp_path, *, forces, tolerance, stdout_fmax, max_steps=100):
    path = tmp_path / "opt.xyz"
    values = [float(value) for value in forces.split()]
    frame = _frame(
        energy=-2.0,
        step=5,
        force_values=[values[0:3], values[3:6]],
    )
    path.write_text(frame, encoding="utf-8")
    (tmp_path / "run.in").write_text(
        f"potential nep.txt\nminimize fire {tolerance} {max_steps}\n"
        "velocity 1\nensemble nve\ntime_step 0\n"
        "dump_xyz 1 opt.xyz force virial precision double\nrun 1\n",
        encoding="utf-8",
    )
    (tmp_path / "stdout.log").write_text(
        "Energy minimization started.\n"
        f"    step 0: total_potential = -1.9 eV, f_max = 1.0 eV/A.\n"
        f"    step 3: total_potential = -2.0 eV, f_max = {stdout_fmax} eV/A.\n"
        "Energy minimization finished.\n",
        encoding="utf-8",
    )
    return path


def test_optimization_result_uses_native_component_criterion(tmp_path):
    path = _write_opt_case(
        tmp_path,
        forces="0.04000000 -0.03000000 0.0 -0.02000000 0.01000000 0.0",
        tolerance=0.05,
        stdout_fmax=0.04,
    )
    result = optimization_result(path, force_tolerance=0.05, max_steps=100)
    assert result["converged"] is True
    assert result["nsteps"] == 3
    assert result["fmax_component"] == pytest.approx(0.04)
    assert result["fmax_norm"] == pytest.approx(0.05)
    assert result["energy"] == pytest.approx(-2.0)
    assert result["max_steps_reached"] is False


def test_optimization_result_reports_max_steps_as_not_converged(tmp_path):
    path = _write_opt_case(
        tmp_path,
        forces="0.20000000 0.0 0.0 0.0 0.0 0.0",
        tolerance=0.05,
        stdout_fmax=0.2,
        max_steps=2,
    )
    result = optimization_result(path, force_tolerance=0.05, max_steps=2)
    assert result["converged"] is False
    assert result["max_steps_reached"] is True


def test_optimization_result_reads_criterion_from_run_in(tmp_path):
    path = _write_opt_case(
        tmp_path,
        forces="0.04000000 0.0 0.0 0.0 0.0 0.0",
        tolerance=0.05,
        stdout_fmax=0.04,
    )
    result = optimization_result(path)
    assert result["force_tolerance_eV_A"] == pytest.approx(0.05)
    assert result["optimizer"] == "fire"
    assert result["converged"] is True


def test_md_frames_summary_and_unwrapped_flag(tmp_path):
    with_unwrapped = tmp_path / "traj.xyz"
    with_unwrapped.write_text(_frame(time=10.0) + _frame(time=20.0), encoding="utf-8")
    summary = md_frames_summary(with_unwrapped)
    assert summary["nframes"] == 2
    assert summary["natoms"] == 2
    assert summary["times_fs"] == [10.0, 20.0]
    assert summary["unwrapped_available"] is True

    without_unwrapped = tmp_path / "wrapped.xyz"
    without_unwrapped.write_text(
        _frame(unwrapped=False) + _frame(unwrapped=False), encoding="utf-8"
    )
    summary = md_frames_summary(without_unwrapped)
    assert summary["unwrapped_available"] is False


def test_write_normalized_trajectory_keeps_unwrapped_positions(tmp_path):
    source = tmp_path / "traj.xyz"
    source.write_text(_frame(time=10.0, step=None), encoding="utf-8")
    frames = parse_extxyz_frames(source)
    destination = write_normalized_trajectory(
        frames, tmp_path / "normalized.xyz", dt_fs=1.0, source_path=source
    )
    parsed = read(destination, index=":")
    assert len(parsed) == 1
    assert parsed[0].info["step"] == 10  # derived from time / dt
    assert parsed[0].info["time"] == pytest.approx(10.0)
    assert tuple(parsed[0].pbc) == (True, True, True)
    assert np.allclose(parsed[0].get_positions(), [[10.1, 0.2, 0.3], [11.1, 1.2, 1.3]])
    assert "wrapped_position" in parsed[0].arrays


# ---------------------------------------------------------------------------
# S-02: stress units, signs and the kinetic-vs-potential distinction
# ---------------------------------------------------------------------------


def test_ev_per_a3_to_gpa_conversion_is_the_exact_constant():
    """1 eV/Angstrom^3 = 160.2176634 GPa (CODATA-consistent), used verbatim."""
    from mliport.gpumd.parse import EV_A3_TO_GPA

    assert EV_A3_TO_GPA == 160.2176634
    # A stress of 0.1914653489 eV/A^3 is the ~30.68 GPa of the BaZrO3 fixture.
    assert pytest.approx(30.676131, abs=1e-6) == 0.1914653489 * EV_A3_TO_GPA


def test_thermo_out_total_stress_is_not_the_potential_only_stress():
    """A measured GPUMD fact: thermo.out stress includes the kinetic term.

    Verified on GPUMD 5.9.1 (V100): for a 40-atom NVE frame at T = 182.844 K the
    ``dump_xyz`` virial predicts ``sxx = 31.1959 GPa`` while ``thermo.out``
    reports ``sxx = 31.3355 GPa``; the difference (0.1396 GPa) is the anisotropic
    kinetic contribution (the isotropic ``N kT / V`` estimate is 0.1959 GPa).

    A potential-only ASE stress must therefore never be replaced by the
    ``thermo.out`` column, and a test that compares them at finite temperature
    without adding the kinetic term is wrong.
    """
    thermo_rows = _thermo_fixture_with_kinetic_stress()
    virial_ev = _VIRIAL_EV_FIXTURE
    volume_a3 = _VOLUME_A3_FIXTURE
    potential_only_gpa = (virial_ev / volume_a3) * 160.2176634
    total_gpa = thermo_rows["sxx"]
    kinetic_gpa = total_gpa - potential_only_gpa

    # The gap is real, positive and of the order of N kT / V.
    isotropic_gpa = (40 * 8.617333262e-5 * thermo_rows["T"]) / volume_a3 * 160.2176634
    assert kinetic_gpa > 0.05
    assert kinetic_gpa == pytest.approx(isotropic_gpa, rel=0.6)

    # The signs still agree (both compressive-positive here), so the gap is a
    # physical term, not a sign error.
    assert total_gpa > 0 and potential_only_gpa > 0
    # Converting the ASE (potential-only) stress back with the documented flip
    # reproduces the potential term, never the total one.
    ase_stress_xx = -virial_ev / volume_a3
    assert -ase_stress_xx * 160.2176634 == pytest.approx(potential_only_gpa)
    assert -ase_stress_xx * 160.2176634 != pytest.approx(total_gpa, rel=1e-3)


#: Frame 1 of the fixture NVE run: GPUMD ``dump_xyz`` virial, xx component (eV).
_VIRIAL_EV_FIXTURE = 100.37033188501752
#: Orthorhombic 8.0181236909897908 A cube.
_VOLUME_A3_FIXTURE = 8.0181236909897908**3


def _thermo_fixture_with_kinetic_stress() -> dict[str, float]:
    """Row 1 of a real GPUMD 5.9.1 ``thermo.out`` (BaZrO3, 40 atoms, NVE)."""
    return {
        "T": 182.84416933,
        "KE": 0.94537855362,
        "PE": -257.56727934,
        "sxx": 31.335537171,
    }
