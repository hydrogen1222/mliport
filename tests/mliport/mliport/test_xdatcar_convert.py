"""Tests for XDATCAR -> standard VASP XDATCAR conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.io.trajectory import Trajectory

from mliport.writers.xdatcar import XdatcarWriter, convert_to_vasp_xdatcar

if TYPE_CHECKING:
    from pathlib import Path


def _mliport_style_xdatcar(path: Path, *, cross_boundary: bool = False) -> None:
    """Write a trajectory with the native (mliport) XDATCAR writer."""
    writer = XdatcarWriter()
    frames = []
    for step in range(0, 30, 10):
        pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        if cross_boundary and step > 0:
            # drift the first atom across the x = 4.0 boundary
            pos[0][0] = 4.0 + step * 0.01
        frames.append(Atoms("H2O", positions=pos, cell=[4.0, 4.0, 4.0], pbc=True))
    writer.write_header(frames[0], path)
    for i, frame in enumerate(frames):
        writer.append_frame(path, frame, step=i * 10)


def test_convert_output_matches_vasp_layout(tmp_path: Path) -> None:
    """Layout must match a real VASP XDATCAR (see /home/storm/vasp/MD_test)."""
    src = tmp_path / "XDATCAR"
    _mliport_style_xdatcar(src)
    out = convert_to_vasp_xdatcar(src)

    assert out == tmp_path / "XDATCAR.vasp"
    lines = out.read_text(encoding="utf-8").splitlines()

    # Comment line + scale factor (12-char right-aligned integer 1)
    assert lines[1] == "           1", repr(lines[1])
    # Lattice vectors: one leading space + 3 x 12-char fields (6 decimals),
    # matching VASP's ``1X,3F12.6`` output byte-for-byte.
    lattice = lines[2]
    assert lattice == "     4.000000    0.000000    0.000000", repr(lattice)
    # Element line: 5-char right-aligned symbols (ASE sorts atoms, so H2O
    # becomes one H block + one O block, exactly like VASP's own output)
    assert lines[5] == "    H    O", repr(lines[5])
    # Count line: one leading space + 5-char right-aligned integers
    assert lines[6] == "     2    1", repr(lines[6])
    # Frame marker: "Direct configuration=" + 12-char right-aligned index
    assert lines[7] == "Direct configuration=           1", repr(lines[7])
    # Coordinates: one leading space + 3 x 12-char fields (8 decimals)
    coord = lines[8]
    assert coord == "   0.00000000  0.00000000  0.00000000", repr(coord)
    # 3 frames total
    assert sum(1 for l in lines if l.startswith("Direct configuration=")) == 3


def test_convert_keeps_unwrapped_coordinates(tmp_path: Path) -> None:
    """VASP keeps unwrapped scaled coordinates (atoms travel across cell
    boundaries). The conversion must not fold coordinates back into [0,1)."""
    src = tmp_path / "XDATCAR"
    # Hand-built input whose first atom sits at scaled x = 1.025 (crossed the
    # x=4.0 boundary). This also exercises conversion of external files.
    src.write_text(
        "H2O\n"
        "1.0\n"
        "     4.000000    0.000000    0.000000\n"
        "     0.000000    4.000000    0.000000\n"
        "     0.000000    0.000000    4.000000\n"
        "    H    O\n"
        "     1    2\n"
        "Direct configuration=           1\n"
        "   1.02500000  0.00000000  0.00000000\n"
        "   0.50000000  0.00000000  0.00000000\n"
        "   0.00000000  0.50000000  0.00000000\n",
        encoding="utf-8",
    )
    out = convert_to_vasp_xdatcar(src)

    # atom 1 must still be at unwrapped x = 1.025 after conversion
    frames = read(out, index=":")
    scaled = frames[0].get_scaled_positions(wrap=False)
    assert scaled[0][0] > 1.0, scaled
    assert abs(scaled[0][0] - 1.025) < 1e-7, scaled


def test_native_writer_keeps_unwrapped_coordinates(tmp_path: Path) -> None:
    """New MD output must preserve boundary crossings before conversion."""
    src = tmp_path / "XDATCAR"
    _mliport_style_xdatcar(src, cross_boundary=True)

    frames = read(src, index=":")
    scaled = frames[-1].get_scaled_positions(wrap=False)
    assert scaled[0][0] == pytest.approx(1.05)


def test_convert_roundtrip_preserves_trajectory(tmp_path: Path) -> None:
    """ASE can read the converted file back with identical positions
    (up to the decimal precision of the format)."""
    src = tmp_path / "XDATCAR"
    _mliport_style_xdatcar(src, cross_boundary=True)
    out = convert_to_vasp_xdatcar(src)

    original = read(src, index=":")
    converted = read(out, index=":")
    assert len(converted) == len(original) == 3
    for orig, conv in zip(original, converted, strict=True):
        assert conv.cell[0][0] == 4.0
        assert np.allclose(
            conv.get_scaled_positions(),
            orig.get_scaled_positions(),
            atol=1e-7,
        )


def test_convert_is_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "XDATCAR"
    _mliport_style_xdatcar(src)
    first = convert_to_vasp_xdatcar(src)
    second = convert_to_vasp_xdatcar(first, tmp_path / "again.XDATCAR")
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_convert_real_vasp_xdatcar_is_noop(tmp_path: Path) -> None:
    """A real VASP XDATCAR (with unwrapped coords, VASP widths) must convert
    to the same content -- conversion is idempotent across sources."""
    vasp_content = (
        "unknown system\n"
        "           1\n"
        "     5.443702    0.000000    0.000000\n"
        "     0.000000    5.443702    0.000000\n"
        "     0.000000    0.000000    5.443702\n"
        "   Si\n"
        "     8\n"
        "Direct configuration=           1\n"
        "   0.75000000  0.75000000  0.25000000\n"
        "   0.00000000  0.50000000  0.50000000\n"
        "   0.75000000  0.25000000  0.75000000\n"
        "   0.00000000  0.00000000  0.00000000\n"
        "   0.25000000  0.75000000  0.75000000\n"
        "   0.50000000  0.50000000  0.00000000\n"
        "   0.25000000  0.25000000  0.25000000\n"
        "   0.50000000  0.00000000  0.50000000\n"
        "Direct configuration=           2\n"
        "   0.75021622  0.75015193  0.25082379\n"
        "   0.00027277  0.49909046  0.49998282\n"
        "   0.74951518  0.24991126  0.75002505\n"
        "   0.00000000  0.00000000  0.00000000\n"
        "   0.24999999  0.75000000  0.75000000\n"
        "   0.50000000  0.50000000  0.00000000\n"
        "   0.25000000  0.25000000  0.25000000\n"
        "   0.50000000  0.00000000  0.50000000\n"
    )
    src = tmp_path / "vasp.XDATCAR"
    src.write_text(vasp_content, encoding="utf-8")
    out = convert_to_vasp_xdatcar(src)
    converted = out.read_text(encoding="utf-8").splitlines()
    for orig_line, conv_line in zip(vasp_content.splitlines(), converted, strict=True):
        # comment line may be reformatted (formula vs "unknown system");
        # everything else must match exactly
        if orig_line.startswith("unknown"):
            continue
        assert orig_line == conv_line, f"{orig_line!r} != {conv_line!r}"


def test_convert_single_frame(tmp_path: Path) -> None:
    src = tmp_path / "XDATCAR"
    writer = XdatcarWriter()
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    writer.write_header(atoms, src)
    writer.append_frame(src, atoms)
    out = convert_to_vasp_xdatcar(src)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert sum(1 for l in lines if l.startswith("Direct configuration=")) == 1


def test_convert_creates_output_directory(tmp_path: Path) -> None:
    src = tmp_path / "XDATCAR"
    _mliport_style_xdatcar(src)
    destination = tmp_path / "exports" / "vasp" / "XDATCAR"

    assert convert_to_vasp_xdatcar(src, destination) == destination
    assert destination.is_file()


def test_convert_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert_to_vasp_xdatcar(tmp_path / "nope.XDATCAR")


def test_xdatcar_writer_rejects_variable_cell_before_writing(tmp_path: Path) -> None:
    frames = [
        Atoms("Li", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),
        Atoms("Li", positions=[[0, 0, 0]], cell=[5, 4, 4], pbc=True),
    ]
    output = tmp_path / "variable.XDATCAR"

    with pytest.raises(ValueError, match="variable cell"):
        XdatcarWriter().write(output, frames)

    assert not output.exists()


def test_convert_rejects_variable_cell_trajectory(tmp_path: Path) -> None:
    source = tmp_path / "variable.traj"
    frames = [
        Atoms("Li", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),
        Atoms("Li", positions=[[0, 0, 0]], cell=[5, 4, 4], pbc=True),
    ]
    with Trajectory(source, "w") as writer:
        for frame in frames:
            writer.write(frame)

    output = tmp_path / "variable.XDATCAR"
    with pytest.raises(ValueError, match="variable cell"):
        convert_to_vasp_xdatcar(source, output)

    assert not output.exists()
