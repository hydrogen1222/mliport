"""Tests for mliport writers (XDATCAR / trajectory frame reconstruction)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ase import Atoms

from mliport.writers.trajectory import TrajectoryWriter
from mliport.writers.xdatcar import XdatcarWriter

if TYPE_CHECKING:
    from pathlib import Path


def _tmp_path(tmp_path: Path, name: str) -> Path:
    return tmp_path / name


def test_xdatcar_write_from_md_with_atoms_objects(tmp_path: Path) -> None:
    frames = [
        Atoms(
            "H2O",
            positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        Atoms(
            "H2O",
            positions=[[0.1, 0, 0], [1.0, 0, 0], [0.1, 0.9, 0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
    ]
    out = _tmp_path(tmp_path, "traj1.XDATCAR")
    XdatcarWriter().write_from_md(
        out, [{"atoms": a, "step": i} for i, a in enumerate(frames)]
    )
    text = out.read_text(encoding="utf-8")
    assert "Direct configuration=" in text
    assert text.count("Direct configuration=") == 2


def test_xdatcar_write_from_md_with_positions_and_symbols(tmp_path: Path) -> None:
    """Regression: frames with only positions+symbols must work."""
    frames = [
        {
            "positions": [[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]],
            "symbols": ["H", "H", "O"],
            "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            "pbc": True,
        },
        {
            "positions": [[0.1, 0, 0], [1.0, 0, 0], [0.1, 0.9, 0]],
            "symbols": ["H", "H", "O"],
            "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            "pbc": True,
        },
    ]
    out = _tmp_path(tmp_path, "traj2.XDATCAR")
    XdatcarWriter().write_from_md(out, frames)
    text = out.read_text(encoding="utf-8")
    assert text.count("Direct configuration=") == 2
    # Element line from header. Note: ASE's Atoms(symbols=[...]) constructor
    # sorts atoms by default, so H H O collapses to the "H  O" / "2  1" blocks.
    lines = text.splitlines()
    assert lines[5].split() == ["H", "O"]
    assert lines[6].split() == ["2", "1"]


def test_xdatcar_write_from_md_positions_without_symbols_raises_clear_error(
    tmp_path: Path,
) -> None:
    """Regression: previously crashed with a confusing AttributeError
    ('dict' object has no attribute 'get_chemical_symbols')."""
    frames = [
        {"positions": [[0, 0, 0]], "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 5]]},
    ]
    out = _tmp_path(tmp_path, "traj3.XDATCAR")
    with pytest.raises(ValueError, match="symbols"):
        XdatcarWriter().write_from_md(out, frames)


def test_xdatcar_write_empty_trajectory_is_noop(tmp_path: Path) -> None:
    out = _tmp_path(tmp_path, "traj4.XDATCAR")
    XdatcarWriter().write_from_md(out, [])
    assert not out.exists()


def test_trajectory_frame_to_atoms_with_atoms() -> None:
    a = Atoms("He", positions=[[0, 0, 0]])
    result = TrajectoryWriter()._frame_to_atoms({"atoms": a})
    assert result is a


def test_trajectory_frame_to_atoms_with_symbols() -> None:
    result = TrajectoryWriter()._frame_to_atoms(
        {"positions": [[0, 0, 0]], "symbols": ["He"]}
    )
    assert result.get_chemical_formula() == "He"


def test_trajectory_frame_to_atoms_missing_symbols_raises_clear_error() -> None:
    """Regression: positions-only frames crashed with an obscure error."""
    with pytest.raises(ValueError, match="symbols"):
        TrajectoryWriter()._frame_to_atoms({"positions": [[0, 0, 0]]})


def test_trajectory_write_extxyz_with_dict_frames(tmp_path: Path) -> None:
    out = _tmp_path(tmp_path, "traj.extxyz")
    TrajectoryWriter().write_extxyz(
        [{"positions": [[0, 0, 0]], "symbols": ["He"]}], out
    )
    text = out.read_text(encoding="utf-8")
    assert "He" in text
