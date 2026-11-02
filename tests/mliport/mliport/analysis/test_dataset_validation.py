from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from ase import Atoms, units
from ase.io.trajectory import Trajectory

from mliport.analysis import TrajectoryDataset, require_analysis, validate_trajectory
from mliport.analysis.thermo import thermodynamic_diagnostics
from mliport.analysis.validation import InvalidTrajectoryError
from mliport.writers.xdatcar import XdatcarWriter


def _frames(n: int = 4) -> list[Atoms]:
    return [
        Atoms(
            "LiS",
            positions=[[0.1 + 0.1 * i, 0, 0], [2, 2, 2]],
            cell=[5, 5, 5],
            pbc=True,
        )
        for i in range(n)
    ]


def test_uniform_time_axis_is_explicit_and_eligible() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(),
        times_fs=[0, 10, 20, 30],
        positions_convention="unwrapped",
    )
    report = validate_trajectory(dataset)
    assert report.time.uniform is True
    assert report.time.frame_interval_fs == 10
    assert report.eligible_for_msd is True


def test_neb_band_is_rejected_as_a_time_trajectory() -> None:
    frames = _frames()
    for index, atoms in enumerate(frames):
        atoms.info["trajectory_kind"] = "neb_band"
        atoms.info["mliport_neb_image_index"] = index

    with pytest.raises(InvalidTrajectoryError, match="reaction-coordinate"):
        TrajectoryDataset.from_frames(
            frames,
            times_fs=[0, 1, 2, 3],
            positions_convention="unwrapped",
        )


def test_mliport_xdatcar_is_a_direct_msd_source(tmp_path) -> None:
    run = tmp_path / "run"
    vasp = run / "vasp"
    vasp.mkdir(parents=True)
    frames = _frames(4)
    frames[-1].positions[0, 0] = 5.2
    xdatcar = vasp / "XDATCAR"
    XdatcarWriter().write(xdatcar, frames)
    (run / "artifacts.json").write_text(
        json.dumps(
            {
                "schema": "mliport.md-artifacts/3",
                "trajectory": {
                    "frame_interval_fs": 10.0,
                    "positions_convention": "unwrapped",
                    "pbc": [True, True, True],
                },
                "artifacts": {
                    "xdatcar": {
                        "path": "vasp/XDATCAR",
                        "format": "vasp-xdatcar",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = TrajectoryDataset.load(xdatcar)
    assert dataset.metadata["source_format"] == "xdatcar"
    assert dataset.frame_interval_fs == 10.0
    assert dataset.positions_convention == "unwrapped"
    assert dataset.positions[-1, 0, 0] == pytest.approx(5.2)
    assert dataset.metadata["pbc_verified"] is True
    assert validate_trajectory(dataset).eligible_for_msd is True


def test_mliport_artifact_position_semantics_cannot_be_overridden(tmp_path) -> None:
    run = tmp_path / "run"
    raw = run / "raw"
    raw.mkdir(parents=True)
    with Trajectory(raw / "trajectory.traj", "w") as writer:
        for atoms in _frames():
            writer.write(atoms)
    (run / "artifacts.json").write_text(
        json.dumps(
            {
                "trajectory": {
                    "frame_interval_fs": 10.0,
                    "positions_convention": "unwrapped",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Refusing to reinterpret periodic image"):
        TrajectoryDataset.load(run, positions_convention="wrapped")


def test_nonuniform_time_is_not_replaced_by_median() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(),
        times_fs=[0, 10, 25, 35],
        positions_convention="unwrapped",
    )
    report = validate_trajectory(dataset)
    assert report.time.uniform is False
    assert report.time.frame_interval_fs is None
    assert dataset.frame_interval_fs is None
    with pytest.raises(InvalidTrajectoryError, match="uniform time axis"):
        require_analysis(dataset, "msd")


def test_unknown_position_convention_fails_transport_closed() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(), times_fs=[0, 10, 20, 30], positions_convention="unknown"
    )
    assert not validate_trajectory(dataset).eligible_for_transport
    with pytest.raises(InvalidTrajectoryError, match="Position convention"):
        require_analysis(dataset, "transport")


def test_default_analysis_view_excludes_equilibration() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(6),
        times_fs=np.arange(6) * 10,
        positions_convention="unwrapped",
        phases=[
            "equilibration",
            "equilibration",
            "production",
            "production",
            "production",
            "production",
        ],
    )
    production = dataset.analysis_view()
    all_frames = dataset.analysis_view(include_equilibration=True)
    assert production.nframes == 4
    assert production.times_fs.tolist() == [20, 30, 40, 50]
    assert all_frames.nframes == 6


def _write_run_with_md_csv_steps(run, steps, *, step_text=None) -> None:
    """Build a minimal valid mliport run whose ``raw/md.csv`` lists ``steps``.

    ``step_text`` optionally overrides the literal text written in the ``step``
    cell (to exercise exact text parsing of values that cannot survive a float).
    """

    raw = run / "raw"
    raw.mkdir(parents=True)
    with Trajectory(raw / "trajectory.traj", "w") as writer:
        for index in range(len(steps)):
            writer.write(
                Atoms(
                    "LiS",
                    positions=[[0.1 * index, 0, 0], [2, 2, 2]],
                    cell=[5, 5, 5],
                    pbc=True,
                )
            )
    with (raw / "md.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time_fs", "phase"])
        for index, step in enumerate(steps):
            cell = step if step_text is None else step_text[index]
            writer.writerow([cell, index * 10.0, "production"])
    (run / "artifacts.json").write_text(
        json.dumps(
            {
                "trajectory": {
                    "frame_interval_fs": 10.0,
                    "positions_convention": "unwrapped",
                }
            }
        ),
        encoding="utf-8",
    )


def test_fractional_md_csv_steps_are_rejected_not_truncated(tmp_path) -> None:
    """AUD-001: a fractional ``step`` column must fail closed, not be truncated.

    ``TrajectoryDataset.load`` cast the finite ``md.csv`` step column with
    ``astype(int)``, so steps ``0.5, 1.5, 2.5, 3.5`` silently loaded as
    ``0, 1, 2, 3`` and the fabricated indices were re-exported as the step
    axis.  Step indices are integers by contract, so the loader must refuse
    the input instead of inventing a step axis that is not in the file.
    """

    run = tmp_path / "run"
    _write_run_with_md_csv_steps(run, [0.5, 1.5, 2.5, 3.5])

    with pytest.raises(ValueError, match="step"):
        TrajectoryDataset.load(run)


def test_integral_float_md_csv_steps_still_load(tmp_path) -> None:
    """An integral float column (``0.0, 10.0, ...``) is a valid step axis."""

    run = tmp_path / "run"
    _write_run_with_md_csv_steps(run, [0.0, 10.0, 20.0, 30.0])

    dataset = TrajectoryDataset.load(run)

    assert dataset.steps is not None
    assert dataset.steps.tolist() == [0, 10, 20, 30]


def test_out_of_range_md_csv_steps_are_rejected(tmp_path) -> None:
    """A step value that overflows ``int64`` must not wrap silently."""

    run = tmp_path / "run"
    _write_run_with_md_csv_steps(run, [0.0, 1.0, 2.0, 1.0e300])

    with pytest.raises(ValueError, match="step"):
        TrajectoryDataset.load(run)


def test_md_csv_astronomical_step_exponent_is_rejected(tmp_path) -> None:
    """The bound check happens on the Decimal, before any huge int is built."""

    run = tmp_path / "run"
    _write_run_with_md_csv_steps(run, [0, 1], step_text=["0", "1e999999999"])

    with pytest.raises(ValueError, match="step"):
        TrajectoryDataset.load(run)


def test_from_frames_rejects_fractional_steps() -> None:
    """The in-memory constructor must not truncate fractional steps either."""

    with pytest.raises(ValueError, match="step"):
        TrajectoryDataset.from_frames(
            _frames(),
            times_fs=[0, 10, 20, 30],
            steps=[0.5, 1.5, 2.5, 3.5],
            positions_convention="unwrapped",
        )


def test_from_frames_preserves_large_integer_steps_exactly() -> None:
    """AUD-001 review: an integer step must never be rounded through float64.

    ``2**53 + 1`` is not representable as a float64, so the earlier
    ``np.asarray(..., dtype=float)`` turned it into ``2**53`` before the
    integrality check could see it.  Integer input must survive verbatim.
    """

    steps = [2**53 + 1, 2**53 + 2, 2**53 + 3]
    dataset = TrajectoryDataset.from_frames(
        _frames(3),
        times_fs=[0.0, 10.0, 20.0],
        steps=steps,
        positions_convention="unwrapped",
    )

    assert dataset.steps is not None
    assert dataset.steps.tolist() == steps


def test_from_frames_accepts_int64_min_and_rejects_overflow() -> None:
    """The int64 range is asymmetric; only genuine overflow is rejected."""

    dataset = TrajectoryDataset.from_frames(
        _frames(2),
        times_fs=[0.0, 10.0],
        steps=[-(2**63), -(2**63) + 1],
        positions_convention="unwrapped",
    )
    assert dataset.steps is not None
    assert dataset.steps.tolist() == [-(2**63), -(2**63) + 1]

    with pytest.raises(ValueError, match="step"):
        TrajectoryDataset.from_frames(
            _frames(2),
            times_fs=[0.0, 10.0],
            steps=[0, 2**63],
            positions_convention="unwrapped",
        )


def test_from_frames_preserves_mixed_integer_and_float_steps_exactly() -> None:
    """AUD-001 round 2: a mixed sequence must not round the integer.

    ``np.asarray([2**53 + 1, 2.0])`` promotes to ``float64`` and turns the
    integer into ``2**53`` *before* any dtype check could see it, so a
    heterogeneous sequence has to be validated element by element.
    """

    dataset = TrajectoryDataset.from_frames(
        _frames(2),
        times_fs=[0.0, 10.0],
        steps=[2**53 + 1, 2.0],
        positions_convention="unwrapped",
    )

    assert dataset.steps is not None
    assert dataset.steps.tolist() == [2**53 + 1, 2]


@pytest.mark.parametrize(
    "bad_steps",
    [
        [2**53 + 1, 0.5],  # fractional element next to an exact integer
        [2**63, 2.0],  # int64 overflow next to a legitimate float
        [0, float("nan")],  # non-finite element
        [0, "1"],  # non-numeric element
    ],
)
def test_from_frames_rejects_invalid_mixed_sequences(bad_steps) -> None:
    """Mixed input is validated element-wise; the bad element is never hidden."""

    with pytest.raises(ValueError, match="step"):
        TrajectoryDataset.from_frames(
            _frames(2),
            times_fs=[0.0, 10.0],
            steps=bad_steps,
            positions_convention="unwrapped",
        )


def test_md_csv_huge_integer_step_text_is_parsed_exactly(tmp_path) -> None:
    """AUD-001 review: the step column is validated on its text, not via float.

    ``float("9007199254740993")`` rounds to ``2**53``; the CSV reader must keep
    the exact integer from the file.
    """

    run = tmp_path / "run"
    _write_run_with_md_csv_steps(
        run,
        [0, 2**53 + 1, 2**53 + 2],
        step_text=["0", "9007199254740993", "9007199254740994"],
    )

    dataset = TrajectoryDataset.load(run)

    assert dataset.steps is not None
    assert dataset.steps.tolist() == [0, 2**53 + 1, 2**53 + 2]


def test_md_csv_float_literal_step_keeps_its_exact_integer_value(tmp_path) -> None:
    """``9007199254740993.0`` is an exact integer and must not be rounded."""

    run = tmp_path / "run"
    _write_run_with_md_csv_steps(
        run,
        [0, 2**53 + 1],
        step_text=["0.0", "9007199254740993.0"],
    )

    dataset = TrajectoryDataset.load(run)

    assert dataset.steps is not None
    assert dataset.steps.tolist() == [0, 2**53 + 1]


def test_md_csv_unparseable_step_makes_the_axis_unknown(tmp_path) -> None:
    """A blank/unparseable step cell keeps the historical "unknown" fallback."""

    run = tmp_path / "run"
    _write_run_with_md_csv_steps(run, [0, 1, 2], step_text=["0", "", "2"])

    dataset = TrajectoryDataset.load(run)

    # No explicit frame_stride_steps metadata is declared by the helper, so an
    # unknown column leaves the step axis as None rather than fabricating one.
    assert dataset.steps is None


def test_frame_slice_recomputes_time_and_step_spacing() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(6),
        times_fs=np.arange(6, dtype=float) * 10.0,
        steps=np.arange(6, dtype=int),
        frame_stride_steps=1,
        positions_convention="unwrapped",
    )

    sliced = dataset.slice_frames([0, 2, 4])

    assert sliced.times_fs.tolist() == [0.0, 20.0, 40.0]
    assert sliced.frame_interval_fs == pytest.approx(20.0)
    assert sliced.frame_stride_steps == 2
    assert sliced.metadata["transformations"][-1]["operation"] == "frame_slice"


def test_nonuniform_frame_slice_does_not_reuse_old_interval() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(6),
        times_fs=np.arange(6, dtype=float) * 10.0,
        positions_convention="unwrapped",
    )

    sliced = dataset.slice_frames([0, 1, 3])

    assert sliced.frame_interval_fs is None
    assert validate_trajectory(sliced).time.uniform is False


@pytest.mark.parametrize("indices", [(2, 1), (0, 0), (-1, 1), (0, 6)])
def test_frame_slice_rejects_nonchronological_or_out_of_range_indices(indices) -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(4), times_fs=[0, 10, 20, 30], positions_convention="unwrapped"
    )
    with pytest.raises((ValueError, IndexError)):
        dataset.slice_frames(indices)


def test_external_ase_momenta_supply_temperature_and_kinetic_energy() -> None:
    frames = _frames(3)
    for atoms in frames:
        atoms.set_velocities([[0.02, 0.0, 0.0], [0.0, 0.01, 0.0]])
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=[0, 2, 4],
        positions_convention="wrapped",
    )
    assert dataset.velocities is not None
    assert dataset.kinetic_energy_eV is not None
    assert dataset.temperature_K is not None
    assert np.all(dataset.kinetic_energy_eV > 0)
    assert np.all(dataset.temperature_K > 0)


def test_imported_ase_velocities_are_in_angstrom_per_fs() -> None:
    frames = _frames(4)
    for atoms in frames:
        atoms.set_velocities([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=[0, 1, 2, 3],
        positions_convention="unwrapped",
    )

    assert dataset.velocities is not None
    assert dataset.velocities[0, 0, 0] == pytest.approx(units.fs)


def test_nve_energy_drift_uses_production_per_atom_energy() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(5),
        times_fs=np.arange(5) * 1000.0,
        positions_convention="unwrapped",
        phases=[
            "equilibration",
            "production",
            "production",
            "production",
            "production",
        ],
        metadata={"resolved_config": {"run_options": {"ensemble": "NVE"}}},
    )
    time_ps = dataset.times_fs / 1000.0
    dataset.total_energy_eV = 2 * (-5.0 + 0.002 * time_ps)
    result = thermodynamic_diagnostics(dataset)
    assert result["summary"]["n_frames"] == 4
    assert result["summary"]["nve_total_energy_drift_eV_atom_ps"] == pytest.approx(
        0.002
    )
