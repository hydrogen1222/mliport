# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""VASP-syntax-compatible XDATCAR trajectory writer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms


def _validate_fixed_cell_trajectory(
    frames: list[Atoms], *, source: str = "trajectory"
) -> None:
    """Validate the fixed-cell contract required by standard XDATCAR."""

    if not frames:
        return
    reference = frames[0]
    reference_symbols = tuple(reference.get_chemical_symbols())
    reference_pbc = np.asarray(reference.pbc, dtype=bool)
    reference_cell = np.asarray(reference.cell.array, dtype=float)
    if not np.all(np.isfinite(reference_cell)):
        raise ValueError(f"{source} frame 0 has a non-finite cell")
    for index, atoms in enumerate(frames):
        if len(atoms) != len(reference):
            raise ValueError(
                f"{source} frame {index} changes atom count; XDATCAR requires "
                "a fixed atom count"
            )
        if tuple(atoms.get_chemical_symbols()) != reference_symbols:
            raise ValueError(
                f"{source} frame {index} changes atom identity/order; "
                "XDATCAR cannot represent that change"
            )
        if not np.array_equal(np.asarray(atoms.pbc, dtype=bool), reference_pbc):
            raise ValueError(
                f"{source} frame {index} changes PBC flags; XDATCAR requires "
                "consistent PBC semantics"
            )
        cell = np.asarray(atoms.cell.array, dtype=float)
        if not np.all(np.isfinite(cell)):
            raise ValueError(f"{source} frame {index} has a non-finite cell")
        if not np.array_equal(cell, reference_cell):
            raise ValueError(
                f"{source} frame {index} has a variable cell; standard XDATCAR "
                "supports only fixed-cell trajectories"
            )


class XdatcarWriter:
    """Write MD trajectory in VASP XDATCAR format.

    XDATCAR is a simple format for storing MD trajectories,
    compatible with VASP visualization tools.

    Example:
        >>> writer = XdatcarWriter()
        >>> writer.write_header(atoms[0], Path("XDATCAR"))
        >>> for frame in trajectory:
        ...     writer.append_frame(Path("XDATCAR"), frame)
    """

    def __init__(self):
        """Initialize XDATCAR writer."""
        self.header_written = False
        self.configuration_index = 0
        self._stream: TextIO | None = None
        self._stream_path: Path | None = None
        self._reference_cell: np.ndarray | None = None
        self._reference_symbols: tuple[str, ...] | None = None
        self._reference_pbc: np.ndarray | None = None

    def write_header(self, atoms: Atoms, output_path: Path | str) -> None:
        """Write XDATCAR header.

        Args:
            atoms: ASE Atoms object (template for structure)
            output_path: Output file path
        """
        self.close_stream()
        output_path = Path(output_path)
        _validate_fixed_cell_trajectory([atoms], source="XDATCAR")
        self._reference_cell = np.asarray(atoms.cell.array, dtype=float).copy()
        self._reference_symbols = tuple(atoms.get_chemical_symbols())
        self._reference_pbc = np.asarray(atoms.pbc, dtype=bool).copy()

        # VASP associates each count with a contiguous block of coordinates.
        # Preserve the actual atom order and therefore repeated symbol blocks;
        # Counter-based grouping corrupts interleaved structures.
        symbols = atoms.get_chemical_symbols()
        symbol_counts: list[tuple[str, int]] = []
        for symbol in symbols:
            if symbol_counts and symbol_counts[-1][0] == symbol:
                previous, count = symbol_counts[-1]
                symbol_counts[-1] = (previous, count + 1)
            else:
                symbol_counts.append((symbol, 1))

        # Match VASP's own fixed-width XDATCAR layout.  Apart from making the
        # file familiar to users, keeping this grammar exact matters for
        # readers that are less permissive than ASE.
        lines = [
            f"{atoms.get_chemical_formula():<40s}",
            f"{1:12d}",  # absolute lattice vectors below
        ]

        # Lattice vectors
        cell = atoms.cell
        for i in range(3):
            lines.append(" " + "".join(f"{value:12.6f}" for value in cell[i]))

        # Element symbols
        element_line = "".join(f"{symbol:>5s}" for symbol, _ in symbol_counts)
        lines.append(element_line)

        # Atom counts
        count_line = " " + "".join(f"{count:>5d}" for _, count in symbol_counts)
        lines.append(count_line)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        self.header_written = True
        self.configuration_index = 0

    def open_stream(self, output_path: Path | str) -> None:
        """Keep an initialized XDATCAR open for efficient frame appends."""
        if not self.header_written:
            raise RuntimeError("write_header() must be called before open_stream()")
        self.close_stream()
        self._stream_path = Path(output_path)
        self._stream = self._stream_path.open(
            "a", encoding="utf-8", buffering=1024 * 1024
        )

    def flush(self) -> None:
        """Flush a persistent append stream, if one is active."""
        if self._stream is not None:
            self._stream.flush()

    def close_stream(self) -> None:
        """Flush and close a persistent append stream."""
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
        self._stream = None
        self._stream_path = None

    def append_frame(
        self,
        output_path: Path | str,
        atoms: Atoms,
        step: int | None = None,
    ) -> None:
        """Append a trajectory frame to XDATCAR.

        Args:
            output_path: Output file path
            atoms: ASE Atoms object for this frame
            step: MD step number (optional)
        """
        output_path = Path(output_path)

        if not self.header_written:
            raise RuntimeError("write_header() must be called before append_frame()")
        if self._reference_cell is None:
            raise RuntimeError("XDATCAR header state is incomplete")
        _validate_fixed_cell_trajectory(
            [
                atoms,
            ],
            source="XDATCAR",
        )
        if len(atoms) != len(self._reference_symbols or ()):
            raise ValueError("XDATCAR frame changes atom count")
        if tuple(atoms.get_chemical_symbols()) != self._reference_symbols:
            raise ValueError(
                "XDATCAR frame changes atom identity/order; this is not "
                "representable in a fixed layout"
            )
        if not np.array_equal(np.asarray(atoms.pbc, dtype=bool), self._reference_pbc):
            raise ValueError("XDATCAR frame changes PBC flags")
        if not np.array_equal(
            np.asarray(atoms.cell.array, dtype=float), self._reference_cell
        ):
            raise ValueError(
                "XDATCAR frame has a variable cell; standard XDATCAR supports "
                "only fixed-cell trajectories"
            )

        # VASP writes continuous (unwrapped) direct coordinates in XDATCAR.
        # This is essential for diffusion/MSD: wrapping into [0, 1) destroys
        # the image history whenever an atom crosses a periodic boundary.
        scaled_pos = atoms.get_scaled_positions(wrap=False)

        self.configuration_index += 1
        # This exact marker is the VASP/ASE XDATCAR grammar.  The previous
        # custom "# Step:" comment made the streamed file unreadable as a
        # multi-frame XDATCAR.
        lines = [f"Direct configuration={self.configuration_index:12d}"]

        for pos in scaled_pos:
            lines.append(" " + "".join(f"{value:12.8f}" for value in pos))

        payload = "\n".join(lines) + "\n"
        if self._stream is not None and self._stream_path == output_path:
            self._stream.write(payload)
        else:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(payload)

    def write(
        self,
        output_path: Path | str,
        trajectory: list[Atoms],
        step_interval: int = 1,
    ) -> None:
        """Write complete trajectory to XDATCAR.

        Args:
            output_path: Output file path
            trajectory: List of ASE Atoms objects
            step_interval: Interval between recorded steps
        """
        if not trajectory:
            return

        output_path = Path(output_path)
        _validate_fixed_cell_trajectory(trajectory)

        # Write header from first frame
        self.write_header(trajectory[0], output_path)

        # Write all frames
        for i, atoms in enumerate(trajectory):
            step = i * step_interval
            self.append_frame(output_path, atoms, step=step)

    def write_from_md(
        self,
        output_path: Path | str,
        trajectory_data: list[dict[str, Any]],
        step_interval: int = 1,
    ) -> None:
        """Write trajectory from MD simulation data.

        Args:
            output_path: Output file path
            trajectory_data: List of frame dictionaries with 'atoms', 'step', 'energy'
            step_interval: Interval between recorded steps
        """
        if not trajectory_data:
            return

        frames = []
        reference_atoms = None
        for frame in trajectory_data:
            if "atoms" in frame:
                frame_atoms = frame["atoms"]
                frames.append(frame_atoms)
                if reference_atoms is None:
                    reference_atoms = frame_atoms
                continue
            if "positions" not in frame:
                raise ValueError("Trajectory data must contain 'atoms' or 'positions'")

            from ase import Atoms as AtomsClass

            symbols = frame.get(
                "symbols",
                None
                if reference_atoms is None
                else reference_atoms.get_chemical_symbols(),
            )
            if symbols is None:
                raise ValueError(
                    "Trajectory frame with 'positions' must also contain "
                    "'symbols' (or an 'atoms' object)."
                )
            if "cell" in frame:
                cell = frame["cell"]
            elif reference_atoms is not None:
                cell = reference_atoms.cell
            else:
                raise ValueError(
                    "Trajectory frame with 'positions' must also contain 'cell'"
                )
            frame_atoms = AtomsClass(
                symbols=symbols,
                positions=frame["positions"],
                cell=cell,
                pbc=frame.get(
                    "pbc", True if reference_atoms is None else reference_atoms.pbc
                ),
            )
            frames.append(frame_atoms)
            if reference_atoms is None:
                reference_atoms = frame_atoms

        self.write(Path(output_path), frames, step_interval=step_interval)


def convert_to_vasp_xdatcar(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert an XDATCAR trajectory to the standard VASP XDATCAR layout.

    Current mliport output already uses this layout.  The converter remains
    useful for older mliport files, ASE trajectories (especially ``.traj``),
    and VASP-style files produced by other tools.  When recovering an old MD
    run, prefer ``trajectory.traj`` over an already-wrapped legacy XDATCAR:
    text reformatting cannot reconstruct image information that was lost.

    Args:
        input_path: Any ASE-readable trajectory, including ``trajectory.traj``
            and a real or mliport XDATCAR (conversion is idempotent).
        output_path: Destination file. Defaults to ``<input>.vasp``.

    Returns:
        The path of the converted file.
    """
    from ase import Atoms as AtomsClass
    from ase.io import read

    frames = read(input_path, index=":")
    if isinstance(frames, AtomsClass):
        frames = [frames]
    if not frames:
        raise ValueError(f"No trajectory frames found in {input_path}")
    _validate_fixed_cell_trajectory(frames, source=str(input_path))

    if output_path is None:
        source = Path(input_path)
        output_path = source.with_name(source.name + ".vasp")

    # Element blocks: VASP associates each count with a contiguous run of
    # coordinates, so group consecutive equal symbols (same rule as the
    # native writer's header).
    symbols = frames[0].get_chemical_symbols()
    groups: list[tuple[str, int]] = []
    for symbol in symbols:
        if groups and groups[-1][0] == symbol:
            groups[-1] = (symbol, groups[-1][1] + 1)
        else:
            groups.append((symbol, 1))

    lines: list[str] = []
    # Comment line: VASP writes an arbitrary label; use the formula.
    lines.append(f"{frames[0].get_chemical_formula():<40s}")
    # Scale factor: lattice vectors below are absolute, so the factor is 1.
    lines.append(f"{1:12d}")
    # Lattice vectors: 12-char fields, 6 decimals, one leading space per row
    # (matches VASP's ``1X,3F12.6`` layout byte-for-byte).
    for vector in frames[0].cell:
        lines.append(" " + "".join(f"{component:12.6f}" for component in vector))
    # Element symbols / counts: 5-char right-aligned fields; the count row
    # carries the same leading space as the numeric rows.
    lines.append("".join(f"{symbol:>5s}" for symbol, _ in groups))
    lines.append(" " + "".join(f"{count:>5d}" for _, count in groups))
    # Frames with UNWRAPPED scaled coordinates (VASP convention).
    for index, atoms in enumerate(frames, start=1):
        lines.append(f"Direct configuration={index:12d}")
        for position in atoms.get_scaled_positions(wrap=False):
            lines.append(" " + "".join(f"{component:12.8f}" for component in position))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
