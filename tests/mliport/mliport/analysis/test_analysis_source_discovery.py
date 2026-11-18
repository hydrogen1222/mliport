"""A run directory without a trajectory must fail with an actionable message.

``mliport analyze <dir>`` used to hash the directory itself while computing the
analysis identity and die with ``[Errno 21] Is a directory``, hiding the real
problem.  Directory discovery is now shared between the loader and the cache
identity, so both name the expected files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io.trajectory import Trajectory

from mliport.analysis.dataset import (
    TrajectoryDataset,
    discover_trajectory_file,
    no_trajectory_message,
)
from mliport.analysis.runner import run_analysis, source_fingerprint
from mliport.analysis.schema import AnalysisRequest


def _single_point_run(path: Path) -> Path:
    """A run directory with results but no trajectory (e.g. a fixed structure)."""
    path.mkdir(parents=True)
    (path / "mliport_results.json").write_text(
        json.dumps({"status": "completed", "energy_eV": -1.0}), encoding="utf-8"
    )
    (path / "POSCAR").write_text(
        "Li\n1.0\n10 0 0\n0 10 0\n0 0 10\nLi\n1\ndirect\n0.5 0.5 0.5\n"
    )
    return path


def test_directory_without_trajectory_reports_the_expected_files(tmp_path) -> None:
    run = _single_point_run(tmp_path / "sp")
    assert discover_trajectory_file(run) is None
    with pytest.raises(FileNotFoundError) as excinfo:
        TrajectoryDataset.load(run)
    message = str(excinfo.value)
    assert message == no_trajectory_message(run)
    assert "raw/trajectory.traj" in message
    assert "manifest.json + trajectory.extxyz" in message


def test_fingerprint_does_not_hash_a_directory(tmp_path) -> None:
    run = _single_point_run(tmp_path / "sp")
    with pytest.raises(FileNotFoundError) as excinfo:
        source_fingerprint(run)
    assert "[Errno 21]" not in str(excinfo.value)
    assert "No supported trajectory found" in str(excinfo.value)


def test_analyze_on_a_trajectoryless_run_fails_actionably(tmp_path) -> None:
    run = _single_point_run(tmp_path / "sp")
    with pytest.raises(FileNotFoundError, match="No supported trajectory found"):
        run_analysis(AnalysisRequest("validate", str(run)))
    # No half-written analysis tree may be left behind for a request that
    # could never be dispatched.
    assert not (run / "analysis").exists()


def test_fingerprint_names_the_file_the_loader_would_read(tmp_path) -> None:
    run = tmp_path / "md"
    raw = run / "raw"
    raw.mkdir(parents=True)
    with Trajectory(raw / "trajectory.traj", "w") as writer:
        for index in range(3):
            writer.write(
                Atoms(
                    "Li",
                    positions=[[0.1 * (index + 1), 1.0, 1.0]],
                    cell=[10.0, 10.0, 10.0],
                    pbc=True,
                )
            )
    assert discover_trajectory_file(run) == raw / "trajectory.traj"
    fingerprint = source_fingerprint(run)
    assert fingerprint["path"] == str(raw / "trajectory.traj")
    assert fingerprint["sha256"]
    # and the loader agrees on which file the directory means
    dataset = TrajectoryDataset.load(run, frame_interval_fs=10.0)
    assert dataset.nframes == 3
    np.testing.assert_allclose(dataset.positions[0, 0], [0.1, 1.0, 1.0])
