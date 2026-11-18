"""Thermodynamic statistics must never hide how many frames they used.

``_statistics`` deliberately ignores non-finite values instead of propagating
them.  That definition is kept, but a trajectory whose ``raw/md.csv`` lost
cells (empty or unparsable values become NaN) must not produce a
plausible-looking mean with no trace of the loss.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

from mliport.analysis.schema import AnalysisRequest
from mliport.analysis.thermo import block_statistics, thermodynamic_diagnostics


def _dataset(n_frames=10, energy_eV=-10.0):
    frames = []
    for _ in range(n_frames):
        atoms = Atoms(
            "Na2", positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True
        )
        atoms.calc = SinglePointCalculator(
            atoms, energy=energy_eV, forces=np.zeros((2, 3))
        )
        frames.append(atoms)
    positions = np.zeros((n_frames, 2, 3))
    positions[:, 0] = [1, 1, 1]
    positions[:, 1] = [3, 1, 1]
    from mliport.analysis.dataset import TrajectoryDataset

    return TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames) * 20.0,
        positions_convention="wrapped",
        phases=["production"] * n_frames,
    )


def test_clean_trajectory_reports_full_frame_counts():
    dataset = _dataset(10)
    dataset.temperature_K = np.full(10, 300.0)
    result = thermodynamic_diagnostics(dataset)
    integrity = result["summary"]["data_integrity"]

    assert integrity["n_frames"] == 10
    temperature = integrity["columns"]["temperature_K"]
    assert temperature == {
        "frames": 10,
        "finite_frames": 10,
        "non_finite_frames": 0,
        "non_finite_fraction": 0.0,
        "statistics_available": True,
    }
    assert integrity["columns_with_non_finite_values"] == []
    assert integrity["columns_without_finite_values"] == []
    # Documented statistics are unchanged for a clean trajectory.
    assert result["summary"]["temperature_mean_K"] == pytest.approx(300.0)


def test_partially_non_finite_column_is_traceable():
    dataset = _dataset(10)
    temperature = np.full(10, 300.0)
    temperature[[2, 5, 7]] = np.nan
    dataset.temperature_K = temperature
    result = thermodynamic_diagnostics(dataset)

    summary = result["summary"]
    # The statistic itself keeps its documented definition: finite frames only.
    assert summary["temperature_mean_K"] == pytest.approx(300.0)
    assert summary["temperature_std_K"] == pytest.approx(0.0, abs=1e-12)
    integrity = summary["data_integrity"]["columns"]["temperature_K"]
    assert integrity["frames"] == 10
    assert integrity["finite_frames"] == 7
    assert integrity["non_finite_frames"] == 3
    assert integrity["non_finite_fraction"] == pytest.approx(0.3)
    assert summary["data_integrity"]["columns_with_non_finite_values"] == [
        "temperature_K"
    ]


def test_fully_non_finite_column_is_named_not_silently_omitted():
    dataset = _dataset(6)
    dataset.pressure_GPa = np.full(6, np.nan)
    result = thermodynamic_diagnostics(dataset)
    summary = result["summary"]

    assert "pressure_mean_GPa" not in summary
    integrity = summary["data_integrity"]
    assert integrity["columns_without_finite_values"] == ["pressure_GPa"]
    assert integrity["columns"]["pressure_GPa"]["statistics_available"] is False
    assert integrity["columns"]["pressure_GPa"]["non_finite_frames"] == 6


def test_block_statistics_records_skipped_values():
    values = np.array([1.0, np.nan, 3.0, 4.0, np.nan, np.nan], dtype=float)
    stats = block_statistics(values, block_size_frames=2)
    # Block means are [1.0, 3.5, nan]: the middle block has no finite value.
    assert stats["number_of_complete_blocks"] == 3
    assert stats["non_finite_frames"] == 3
    assert stats["blocks_without_finite_values"] == 1
    assert stats["blocks_used_for_mean"] == 2
    assert stats["mean_of_block_means"] == pytest.approx(2.25)  # (1.0 + 3.5) / 2


def test_corrupted_md_csv_is_visible_in_the_analysis_output(tmp_path: Path) -> None:
    """A run whose md.csv lost temperature cells must not look healthy."""

    from mliport.analysis.runner import run_analysis

    run = tmp_path / "run"
    raw = run / "raw"
    raw.mkdir(parents=True)
    n_frames = 8
    with Trajectory(raw / "trajectory.traj", "w") as writer:
        for index in range(n_frames):
            atoms = Atoms(
                "Na2", positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True
            )
            atoms.calc = SinglePointCalculator(
                atoms, energy=-10.0 - 0.01 * index, forces=np.zeros((2, 3))
            )
            writer.write(atoms)
    (run / "artifacts.json").write_text(
        '{"schema": "mliport.md-artifacts/3", "trajectory": '
        '{"positions_convention": "wrapped", "frame_interval_fs": 20.0}}',
        encoding="utf-8",
    )
    with (raw / "md.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time_fs", "temperature_K", "potential_energy_eV"])
        for index in range(n_frames):
            temperature = "" if index < 5 else "300.0"
            writer.writerow([index, index * 20.0, temperature, -10.0 - 0.01 * index])

    outcome = run_analysis(
        AnalysisRequest("thermo", str(run), parameters={"include_equilibration": True})
    )
    assert outcome["status"] == "success"
    summary = outcome["results"]["summary"]
    assert summary["temperature_mean_K"] == pytest.approx(300.0)
    integrity = summary["data_integrity"]["columns"]["temperature_K"]
    assert integrity["non_finite_frames"] == 5
    assert summary["data_integrity"]["columns_with_non_finite_values"] == [
        "temperature_K"
    ]


def test_block_statistics_drops_inf_as_well_as_nan():
    """``nanmean`` alone keeps ±Inf; block means are finite-value means."""

    values = np.array([1.0, np.inf, -np.inf, 2.0, np.nan, np.nan], dtype=float)
    stats = block_statistics(values, block_size_frames=2)
    # Blocks [1.0, inf] -> 1.0, [-inf, 2.0] -> 2.0, [nan, nan] -> no mean.
    np.testing.assert_allclose(stats["block_means"][:2], [1.0, 2.0])
    assert np.isnan(stats["block_means"][2])
    np.testing.assert_array_equal(stats["block_finite_counts"], [1, 1, 0])
    assert stats["mean_of_block_means"] == pytest.approx(1.5)
    assert stats["std_of_block_means"] == pytest.approx(np.std([1.0, 2.0], ddof=1))
    assert stats["non_finite_frames"] == 4
    assert stats["blocks_without_finite_values"] == 1
    assert stats["blocks_used_for_mean"] == 2


def test_block_statistics_counts_match_the_means_it_publishes():
    """Counts and block means must describe the same set of values."""

    values = np.array(
        [1.0, 2.0, np.inf, -np.inf, np.nan, np.inf, 3.0, 4.0, 5.0, 6.0],
        dtype=float,
    )
    stats = block_statistics(values, block_size_frames=2)
    counts = stats["block_finite_counts"]
    blocks = values.reshape(5, 2)
    assert counts.tolist() == [2, 0, 0, 2, 2]
    assert stats["non_finite_frames"] == int(np.count_nonzero(~np.isfinite(blocks)))
    assert stats["blocks_without_finite_values"] == int(np.count_nonzero(counts == 0))
    assert stats["blocks_used_for_mean"] == int(np.count_nonzero(counts > 0))
    published = stats["block_means"][np.isfinite(stats["block_means"])]
    assert published.shape[0] == stats["blocks_used_for_mean"]
    assert stats["mean_of_block_means"] == pytest.approx(published.mean())


def test_all_non_finite_blocks_produce_no_published_mean():
    values = np.array([np.nan, np.inf, -np.inf, np.nan], dtype=float)
    stats = block_statistics(values, block_size_frames=2)
    assert stats["blocks_used_for_mean"] == 0
    assert stats["blocks_without_finite_values"] == 2
    assert np.all(np.isnan(stats["block_means"]))
    assert np.isnan(stats["mean_of_block_means"])


def test_inf_temperature_is_not_a_valid_thermodynamic_mean():
    dataset = _dataset(6)
    temperature = np.full(6, 300.0)
    temperature[[1, 4]] = np.inf
    dataset.temperature_K = temperature
    result = thermodynamic_diagnostics(dataset, block_size_frames=2)
    summary = result["summary"]
    assert summary["temperature_mean_K"] == pytest.approx(300.0)
    integrity = summary["data_integrity"]["columns"]["temperature_K"]
    assert integrity["non_finite_frames"] == 2
    assert result["blocks"]["temperature_K"]["blocks_used_for_mean"] == 3
    np.testing.assert_allclose(
        result["blocks"]["temperature_K"]["block_means"], [300.0, 300.0, 300.0]
    )
