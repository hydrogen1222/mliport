"""Native GPUMD run directory -> ``TrajectoryDataset`` -> analysis.

A stub ``gpumd`` executable copies pre-generated, GPUMD-formatted output into the
run directory, so the *real* runner (parse, audit, normalise, manifest) and the
*real* loader are exercised end to end without a GPU.  The synthetic dumps have
analytically known motion (ballistic atoms that cross the periodic boundary), so
every number the analysis layer returns has a closed-form expected value:

* atom 0: ``x(t) = 8.6 + 0.05 t`` A  (crosses the +x face at t = 28 fs)
* atom 1: ``x(t) = 1.25 - 0.03 t`` A (crosses the 0 face at t = 42 fs)

Frame ``k`` (1-based) is written at step ``k * interval`` and ``Time = step * dt``.
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

import mliport.gpumd.runner as runner_module
from mliport.analysis.dataset import TrajectoryDataset
from mliport.analysis.msd import calculate_msd
from mliport.analysis.validation import InvalidTrajectoryError
from mliport.gpumd.runner import NativeGPUMDRunner

CELL_A = 10.0
MASS_O18 = 18.0  # deliberately not ASE's table value for O (15.999)
MOVERS = (
    # symbol, x0, y0, z0, vx (A/fs)
    ("O", 8.6, 2.0, 3.0, 0.05),
    ("O", 1.25, 5.0, 6.0, -0.03),
)

_THERMO_COLUMNS = "# columns T KE PE sxx syy szz syz sxz sxy ax ay az bx by bz cx cy cz"


def _script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _traj_text(
    nframes,
    *,
    interval=10,
    dt=1.0,
    unwrapped=True,
    movers=MOVERS,
    pbc="T T T",
    cell=CELL_A,
    first_step_offset=0,
    reorder=False,
):
    """GPUMD ``dump_xyz`` text for ballistic atoms (positions wrapped into the cell)."""
    properties = "species:S:1:pos:R:3:forces:R:3:mass:R:1"
    if unwrapped:
        properties += ":unwrapped_position:R:3"
    lattice = f"{cell} 0 0 0 {cell} 0 0 0 {cell}"
    lines = []
    for k in range(nframes):
        step = (k + 1) * interval + first_step_offset
        time_fs = step * dt
        lines.append(str(len(movers)))
        lines.append(
            f"Time={time_fs!r} "
            f'pbc="{pbc}" '
            f'Lattice="{lattice}" '
            "energy=-1.5 "
            'virial="1 0 0 0 1 0 0 0 1" '
            f"Properties={properties}"
        )
        rows = []
        for index, (symbol, x0, y0, z0, vx) in enumerate(movers):
            x = x0 + vx * time_fs
            row = [x % cell, y0, z0, 0.0, 0.0, 0.1, MASS_O18]
            if unwrapped:
                row += [x, y0, z0]
            # an atom that is not the input species, from the second frame on
            label = "H" if reorder and k == 1 and index == 0 else symbol
            rows.append(f"{label} " + " ".join(f"{value:.10f}" for value in row))
        lines.extend(rows)
    return "\n".join(lines) + "\n"


def _thermo_text(nrows, *, interval=10, dt=1.0, natoms=2):
    header = (
        f"# dump_thermo {interval}\n"
        "# format_version 1\n"
        f"# num_atoms {natoms}\n"
        f"# dt_output {interval * dt:.10f} fs\n"
        f"{_THERMO_COLUMNS}\n"
    )
    row = "300.0 0.01 -3.0 0 0 0 0 0 0 10 0 0 0 10 0 0 0 10"
    return header + "\n".join([row] * nrows) + "\n"


def _atoms():
    return Atoms(
        "O2",
        positions=[[8.6, 2.0, 3.0], [1.25, 5.0, 6.0]],
        cell=[CELL_A] * 3,
        pbc=True,
    )


def _runner(tmp_path, files, *, out="run", calc_type="md", **kwargs):
    """A real ``NativeGPUMDRunner`` whose ``gpumd`` is a stub that emits ``files``."""
    fixtures = tmp_path / f"fixtures-{out}"
    fixtures.mkdir(exist_ok=True)
    copies = []
    for name, text in files.items():
        (fixtures / name).write_text(text, encoding="utf-8")
        copies.append(f'  cp "{fixtures / name}" "{name}"')
    # Only a real run (run.in in the cwd, no arguments) produces output.
    body = (
        'if [ "$#" -eq 0 ] && [ -f run.in ]; then\n'
        + "\n".join(copies)
        + "\nfi\nexit 0"
    )
    script = _script(tmp_path, f"stub_gpumd_{out}.sh", body)
    nep = tmp_path / "nep.txt"
    nep.write_text("nep4 1 O\ncutoff 8 4 232 26\n", encoding="utf-8")
    engine = SimpleNamespace(gpumd_command=str(script), nep_path=str(nep), task="bulk")
    options = {
        "ensemble": "NVE",
        "temperature": 300.0,
        "timestep_fs": 1.0,
        "steps": 100,
        "dump_interval": 10,
        "seed": 1,
    }
    if calc_type != "md":
        options = {}
    options.update(kwargs)
    return NativeGPUMDRunner(
        engine,
        calc_type=calc_type,
        output_dir=tmp_path / out,
        verbose=False,
        **options,
    )


def _md_run(tmp_path, nframes=10, *, out="run", traj=None, thermo=None, **kwargs):
    interval = kwargs.get("dump_interval", 10)
    dt = kwargs.get("timestep_fs", 1.0)
    files = {
        "traj.xyz": traj
        if traj is not None
        else _traj_text(
            nframes,
            interval=interval,
            dt=dt,
            unwrapped=kwargs.get("unwrapped", True),
        ),
        "thermo.out": thermo
        if thermo is not None
        else _thermo_text(nframes, interval=interval, dt=dt),
    }
    runner = _runner(tmp_path, files, out=out, **kwargs)
    result = runner.execute(_atoms())
    return runner, result


def _manifest(runner):
    return json.loads(runner.manifest_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# P0: a native GPUMD run directory loads through TrajectoryDataset.load()
# ---------------------------------------------------------------------------


def test_native_run_directory_loads_with_real_time_steps_pbc_and_unwrapped_positions(
    tmp_path,
):
    runner, result = _md_run(tmp_path)
    assert result["status"] == "completed"
    dataset = TrajectoryDataset.load(runner.output_dir)

    expected_times = 10.0 * np.arange(1, 11)
    assert dataset.nframes == 10
    assert dataset.times_fs.tolist() == pytest.approx(expected_times.tolist())
    assert dataset.steps.tolist() == [10 * k for k in range(1, 11)]
    assert dataset.md_timestep_fs == pytest.approx(1.0)
    assert dataset.frame_stride_steps == 10
    assert dataset.frame_interval_fs == pytest.approx(10.0)
    assert dataset.pbc.tolist() == [True, True, True]
    assert dataset.positions_convention == "unwrapped"
    assert dataset.symbols == ("O", "O")
    assert dataset.metadata["run_status"] == "completed"
    assert dataset.metadata["source_format"] == "gpumd-extxyz"
    assert dataset.metadata["target_temperature_K"] == pytest.approx(300.0)
    assert dataset.metadata["pbc_verified"] is True
    assert dataset.target_temperature_K == pytest.approx(300.0)
    assert dataset.cells.shape == (10, 3, 3)
    assert np.allclose(dataset.cells, np.eye(3) * CELL_A)

    # Positions are the *unwrapped* ones: atom 0 leaves the box (x > 10) instead
    # of being folded back, and equals the analytic trajectory.
    x0 = 8.6 + 0.05 * expected_times
    x1 = 1.25 - 0.03 * expected_times
    assert dataset.positions[:, 0, 0] == pytest.approx(x0, abs=1e-7)
    assert dataset.positions[:, 1, 0] == pytest.approx(x1, abs=1e-7)
    assert dataset.positions[:, 0, 0].max() > CELL_A
    assert dataset.positions[:, 1, 0].min() < 0.0


def test_loading_the_trajectory_file_directly_gives_the_same_dataset(tmp_path):
    runner, _ = _md_run(tmp_path)
    by_dir = TrajectoryDataset.load(runner.output_dir)
    by_file = TrajectoryDataset.load(runner.output_dir / "trajectory.extxyz")
    assert np.array_equal(by_dir.positions, by_file.positions)
    assert np.array_equal(by_dir.times_fs, by_file.times_fs)
    assert by_file.metadata["source_format"] == "gpumd-extxyz"


def test_dump_masses_are_carried_not_replaced_by_the_ase_table(tmp_path):
    runner, _ = _md_run(tmp_path)
    dataset = TrajectoryDataset.load(runner.output_dir)
    assert dataset.masses.tolist() == pytest.approx([MASS_O18, MASS_O18])


def test_wrapped_dump_loads_as_wrapped_and_is_never_msd_eligible(tmp_path):
    runner, result = _md_run(tmp_path, unwrapped=False)
    assert result["status"] == "completed"
    assert result["msd_eligible"] is False
    assert "unwrapped_position" in result["msd_ineligible_reason"]
    assert _manifest(runner)["msd_eligible"] is False
    dataset = TrajectoryDataset.load(runner.output_dir)
    assert dataset.positions_convention == "wrapped"
    assert dataset.positions.min() >= 0.0
    assert dataset.positions.max() < CELL_A
    with pytest.raises(ValueError, match="conflicts with the mliport trajectory"):
        TrajectoryDataset.load(runner.output_dir, positions_convention="unwrapped")


def test_explicit_arguments_contradicting_the_manifest_are_refused(tmp_path):
    runner, _ = _md_run(tmp_path)
    with pytest.raises(ValueError, match="md_timestep_fs"):
        TrajectoryDataset.load(runner.output_dir, md_timestep_fs=0.5)
    with pytest.raises(ValueError, match="frame_stride_steps"):
        TrajectoryDataset.load(runner.output_dir, frame_stride_steps=5)
    with pytest.raises(ValueError, match="conflicts with declared frame interval"):
        TrajectoryDataset.load(runner.output_dir, frame_interval_fs=25.0)


def test_non_divisible_request_loads_with_the_frames_gpumd_wrote(tmp_path):
    """25 steps at interval 10: frames at steps 10 and 20; step 25 writes no frame."""
    runner, result = _md_run(tmp_path, nframes=2, steps=25)
    assert result["status"] == "completed"
    dataset = TrajectoryDataset.load(runner.output_dir)
    assert dataset.steps.tolist() == [10, 20]
    assert dataset.times_fs.tolist() == pytest.approx([10.0, 20.0])


def test_manifest_records_the_verified_schedule(tmp_path):
    runner, result = _md_run(tmp_path)
    manifest = _manifest(runner)
    block = manifest["trajectory"]
    assert manifest["status"] == "completed"
    assert manifest["trajectory_path"] == "trajectory.extxyz"
    assert block["nframes"] == 10
    assert block["natoms"] == 2
    assert block["positions_convention"] == "unwrapped"
    assert (block["first_step"], block["last_step"]) == (10, 100)
    assert block["schedule"]["schedule_verified"] is True
    assert block["schedule"]["expected_nframes"] == 10
    assert len(block["sha256"]) == 64
    assert block["bytes"] == (runner.output_dir / "trajectory.extxyz").stat().st_size
    assert result["md_schedule"]["last_step"] == 100
    assert manifest["artifacts"]["trajectory"]["path"] == "trajectory.extxyz"


# ---------------------------------------------------------------------------
# Known-answer transport observables on the loaded dataset
# ---------------------------------------------------------------------------


def test_tracer_msd_known_answer_across_the_periodic_boundary(tmp_path):
    """Ballistic MSD(tau) = (v tau)^2, exactly, although the atom wraps."""
    runner, _ = _md_run(tmp_path)
    dataset = TrajectoryDataset.load(runner.output_dir)
    for index, velocity in ((0, 0.05), (1, -0.03)):
        result = calculate_msd(
            dataset, mobile_indices=[index], method="direct", axes="xyz"
        )
        lag_fs = np.asarray(result["lag_time_ps"], dtype=float) * 1000.0
        msd = np.asarray(result["msd_by_axes_A2"]["xyz"], dtype=float)
        assert msd == pytest.approx((velocity * lag_fs) ** 2, rel=1e-6, abs=1e-9)


def test_collective_charge_displacement_known_answer(tmp_path):
    """Q(t) = sum_i q_i (r_i(t) - r_i(0)) with q = (+1, -1) is (v0 - v1) t x-hat."""
    runner, _ = _md_run(tmp_path)
    dataset = TrajectoryDataset.load(runner.output_dir)
    charges = np.array([1.0, -1.0])
    displacement = dataset.positions - dataset.positions[0]
    collective = np.einsum("i,kia->ka", charges, displacement)
    elapsed = dataset.times_fs - dataset.times_fs[0]
    assert collective[:, 0] == pytest.approx((0.05 + 0.03) * elapsed, abs=1e-7)
    assert np.allclose(collective[:, 1:], 0.0)


# ---------------------------------------------------------------------------
# P1: exit status 0 is not completion
# ---------------------------------------------------------------------------


def test_exit_zero_with_partial_dump_is_not_completed(tmp_path):
    """10000 requested steps but one well-formed frame on disk, exit status 0."""
    runner, result = _md_run(tmp_path, nframes=1, steps=10000)
    assert result["status"] == "failed"
    assert "GPUMDIncompleteRunError" in result["error"]
    assert "holds 1 frame" in result["error"]
    assert result["trajectory_path"] is None
    manifest = _manifest(runner)
    assert manifest["status"] == "failed"
    assert manifest["trajectory"] is None
    assert manifest["trajectory_path"] is None
    assert "trajectory" not in manifest["artifacts"]
    # Nothing that looks like a result is left behind; the raw evidence is.
    assert not (runner.output_dir / "trajectory.extxyz").exists()
    assert not (runner.output_dir / "trajectory.extxyz.partial").exists()
    assert (runner.native_dir / "traj.xyz").is_file()
    assert (runner.native_dir / "thermo.out").is_file()
    with pytest.raises(InvalidTrajectoryError, match="status is 'failed'"):
        TrajectoryDataset.load(runner.output_dir)


@pytest.mark.parametrize(
    ("label", "kwargs", "message"),
    [
        ("species reordered", {"traj": "reorder"}, "species in the input order"),
        ("partial pbc", {"traj": "pbc"}, "full 3D periodicity"),
        ("cell changed", {"traj": "cell"}, "changed the cell"),
        ("thermo short", {"thermo": "short"}, "thermo.out has 9 row"),
        ("thermo wrong interval", {"thermo": "interval"}, "dump_thermo 5"),
        ("thermo wrong atoms", {"thermo": "atoms"}, "states 3 atoms"),
        ("frames of two runs", {"traj": "double"}, "holds 20 frame"),
        ("shifted time axis", {"traj": "shifted"}, "outside the final dump"),
    ],
)
def test_inconsistent_output_fails_closed(tmp_path, label, kwargs, message):
    traj = thermo = None
    if kwargs.get("traj") == "reorder":
        traj = _traj_text(10, reorder=True)
    elif kwargs.get("traj") == "pbc":
        traj = _traj_text(10, pbc="T T F")
    elif kwargs.get("traj") == "cell":
        traj = _traj_text(10, cell=10.5)
    elif kwargs.get("traj") == "double":
        traj = _traj_text(10) + _traj_text(10)
    elif kwargs.get("traj") == "shifted":
        traj = _traj_text(10, first_step_offset=40)
    if kwargs.get("thermo") == "short":
        thermo = _thermo_text(9)
    elif kwargs.get("thermo") == "interval":
        thermo = _thermo_text(10, interval=5)
    elif kwargs.get("thermo") == "atoms":
        thermo = _thermo_text(10, natoms=3)
    runner, result = _md_run(tmp_path, traj=traj, thermo=thermo)
    assert result["status"] == "failed", label
    assert message in result["error"], (label, result["error"])
    assert result["trajectory_path"] is None
    assert not (runner.output_dir / "trajectory.extxyz").exists()
    assert _manifest(runner)["trajectory"] is None


def test_stale_trajectory_from_an_earlier_run_is_not_advertised(tmp_path):
    first, result = _md_run(tmp_path, out="shared")
    assert result["status"] == "completed"
    assert (first.output_dir / "trajectory.extxyz").is_file()

    second = _runner(
        tmp_path,
        {
            "traj.xyz": _traj_text(1),
            "thermo.out": _thermo_text(1),
        },
        out="shared",
        steps=10000,
        allow_existing_native=True,
    )
    result = second.execute(_atoms())
    assert result["status"] == "failed"
    assert result["trajectory_path"] is None
    manifest = _manifest(second)
    assert manifest["status"] == "failed"
    assert manifest["trajectory"] is None
    assert manifest["trajectory_path"] is None
    assert "trajectory" not in manifest["artifacts"]
    with pytest.raises(InvalidTrajectoryError, match="status is 'failed'"):
        TrajectoryDataset.load(second.output_dir)


# ---------------------------------------------------------------------------
# The loader trusts nothing the manifest does not vouch for
# ---------------------------------------------------------------------------


def _rewrite_manifest(runner, mutate):
    manifest = _manifest(runner)
    mutate(manifest)
    runner.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_modified_trajectory_is_refused(tmp_path):
    runner, _ = _md_run(tmp_path)
    path = runner.output_dir / "trajectory.extxyz"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(InvalidTrajectoryError, match="SHA-256"):
        TrajectoryDataset.load(runner.output_dir)


def test_truncated_trajectory_is_refused(tmp_path):
    runner, _ = _md_run(tmp_path)
    path = runner.output_dir / "trajectory.extxyz"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[: len(lines) // 2]), encoding="utf-8")
    with pytest.raises(InvalidTrajectoryError, match="SHA-256"):
        TrajectoryDataset.load(runner.output_dir)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.update(status="failed"), "status is 'failed'"),
        (lambda m: m.update(status="cancelled"), "status is 'cancelled'"),
        (lambda m: m.update(calc_type="opt"), "only an MD run"),
        (lambda m: m.update(calc_type="sp"), "only an MD run"),
        (lambda m: m.update(trajectory=None), "no trajectory block"),
        (lambda m: m["trajectory"].pop("sha256"), "no SHA-256"),
        (
            lambda m: m["trajectory"].update(positions_convention="unknown"),
            "not wrapped/unwrapped",
        ),
        (lambda m: m["trajectory"].update(nframes=11), "frame counts disagree"),
        (
            lambda m: m["trajectory"]["schedule"].update(schedule_verified=False),
            "verified MD schedule",
        ),
        (
            lambda m: m["trajectory"].update(path="../trajectory.extxyz"),
            "is not 'trajectory.extxyz'",
        ),
    ],
)
def test_manifest_that_does_not_vouch_for_the_trajectory_is_refused(
    tmp_path, mutate, message
):
    runner, _ = _md_run(tmp_path)
    _rewrite_manifest(runner, mutate)
    with pytest.raises(InvalidTrajectoryError, match=message):
        TrajectoryDataset.load(runner.output_dir)


def test_manifest_declaring_the_wrong_convention_is_refused(tmp_path):
    """The frames say 'unwrapped'; a manifest claiming 'wrapped' is a conflict."""
    runner, _ = _md_run(tmp_path)
    _rewrite_manifest(
        runner, lambda m: m["trajectory"].update(positions_convention="wrapped")
    )
    with pytest.raises(InvalidTrajectoryError, match="positions_convention"):
        TrajectoryDataset.load(runner.output_dir)


def test_ordinary_mliport_directories_are_still_discovered(tmp_path):
    """A manifest.json with another schema must not hijack ordinary discovery."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema": "something-else/1"}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="No supported trajectory"):
        TrajectoryDataset.load(tmp_path)


# ---------------------------------------------------------------------------
# P1: long MD must be streamed, not materialised
# ---------------------------------------------------------------------------


def _many_atoms(natoms):
    return tuple(
        ("O", (0.37 * i) % CELL_A, (0.61 * i) % CELL_A, (0.83 * i) % CELL_A, 0.0)
        for i in range(natoms)
    )


def test_md_post_processing_never_materialises_the_whole_trajectory(
    tmp_path, monkeypatch
):
    """Peak memory must not grow with the number of frames.

    The earlier implementation built every frame as a Python ``Atoms`` (three
    times: parse, summary, write).  Streaming keeps one frame alive, so the
    traced peak of a 10x longer run stays within a small factor of the short one
    instead of growing ~10x.
    """
    natoms = 64
    movers = _many_atoms(natoms)

    def boom(*_args, **_kwargs):
        raise AssertionError("MD post-processing must stream, not parse a list")

    monkeypatch.setattr(runner_module, "parse_extxyz_frames", boom)

    def run(nframes, out):
        files = {
            "traj.xyz": _traj_text(nframes, movers=movers),
            "thermo.out": _thermo_text(nframes, natoms=natoms),
        }
        runner = _runner(tmp_path, files, out=out, steps=nframes * 10)
        atoms = Atoms(
            ["O"] * natoms,
            positions=[(x, y, z) for _, x, y, z, _ in movers],
            cell=[CELL_A] * 3,
            pbc=True,
        )
        tracemalloc.start()
        try:
            result = runner.execute(atoms)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result["status"] == "completed", result.get("error")
        assert result["nframes"] == nframes
        return peak

    run(5, "warmup")  # one-time imports must not be charged to either run
    short_peak = run(50, "short")
    long_peak = run(500, "long")
    assert long_peak < 2.5 * short_peak, (short_peak, long_peak)


# ---------------------------------------------------------------------------
# Minor: a zero-time-step single point must reproduce its input structure
# ---------------------------------------------------------------------------


def _sp_text(x1):
    return (
        "2\n"
        'Time=0.0 pbc="T T T" Lattice="6 0 0 0 6 0 0 0 6" energy=-1.5 '
        'virial="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3:forces:R:3\n'
        "O 0.0 0.0 0.0 0.0 0.0 0.1\n"
        f"O {x1} 0.0 0.0 0.0 0.0 -0.1\n"
    )


def _sp_atoms():
    return Atoms("O2", positions=[[0, 0, 0], [1.2, 0, 0]], cell=[6, 6, 6], pbc=True)


def test_single_point_with_unchanged_geometry_completes(tmp_path):
    runner = _runner(tmp_path, {"sp.xyz": _sp_text(1.2)}, calc_type="sp")
    result = runner.execute(_sp_atoms())
    assert result["status"] == "completed"
    assert result["positions_unchanged"] is True
    assert result["positions_max_delta_A"] == pytest.approx(0.0, abs=1e-9)


def test_single_point_whose_coordinates_moved_fails_closed(tmp_path):
    """``time_step 0`` cannot move atoms; a moved dump belongs to another structure."""
    runner = _runner(tmp_path, {"sp.xyz": _sp_text(1.5)}, calc_type="sp")
    result = runner.execute(_sp_atoms())
    assert result["status"] == "failed"
    assert "does not reproduce the input structure" in result["error"]
    assert result["energy"] is None
    assert result["forces"] is None
    assert result["trajectory_path"] is None
    manifest = _manifest(runner)
    assert manifest["status"] == "failed"
    assert manifest["trajectory_path"] is None
    assert not (runner.output_dir / "trajectory.extxyz").exists()
