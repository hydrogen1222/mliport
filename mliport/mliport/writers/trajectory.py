# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ASE trajectory writer.

Writes trajectories in ASE's native format for analysis and visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ase.io import write
from ase.io.trajectory import TrajectoryWriter as AseTrajectoryWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms


class TrajectoryWriter:
    """Write trajectories in ASE format.

    Supports multiple ASE-compatible formats including:
    - ASE trajectory (.traj)
    - Extended XYZ (.extxyz)
    - XYZ (.xyz)

    Example:
        >>> writer = TrajectoryWriter()
        >>> writer.write(trajectory, Path("trajectory.traj"), format="traj")
        >>> writer.write(trajectory, Path("trajectory.xyz"), format="extxyz")
    """

    def __init__(self):
        """Initialize trajectory writer."""
        pass

    def write(
        self,
        trajectory: list[Atoms],
        output_path: Path | str,
        format: str = "traj",  # noqa: A002
    ) -> None:
        """Write trajectory to file.

        Args:
            trajectory: List of ASE Atoms objects
            output_path: Output file path
            format: Output format (traj, xyz, extxyz, etc.)

        Raises:
            ValueError: If format is not supported
        """
        output_path = Path(output_path)

        if not trajectory:
            return

        # Use ASE's write for multi-frame support
        write(output_path, trajectory, format=format)

    def write_single(
        self,
        atoms: Atoms,
        output_path: Path | str,
        format: str = "traj",  # noqa: A002
        append: bool = False,
    ) -> None:
        """Write single frame to trajectory file.

        Args:
            atoms: ASE Atoms object
            output_path: Output file path
            format: Output format
            append: Whether to append to existing file
        """
        output_path = Path(output_path)

        mode = "a" if append else "w"
        write(output_path, atoms, format=format, append=mode == "a")

    def write_ase_trajectory(
        self,
        trajectory: list[Atoms],
        output_path: Path | str,
    ) -> None:
        """Write in ASE's native binary trajectory format.

        This format preserves all Atoms information including
        velocities, energies, and custom arrays.

        Args:
            trajectory: List of ASE Atoms objects
            output_path: Output file path
        """
        output_path = Path(output_path)

        if not trajectory:
            return

        writer = AseTrajectoryWriter(output_path, mode="w")
        for atoms in trajectory:
            writer.write(atoms)
        writer.close()

    def write_extxyz(
        self,
        trajectory: list[Atoms] | list[dict[str, Any]],
        output_path: Path | str,
        columns: list[str] | None = None,
    ) -> None:
        """Write in extended XYZ format.

        Extended XYZ includes additional properties like forces and
        stress in the comment line.

        Args:
            trajectory: List of ASE Atoms or frame dictionaries
            output_path: Output file path
            columns: Additional columns to write
        """
        output_path = Path(output_path)

        if not trajectory:
            return

        # Convert frame dicts to Atoms if needed
        atoms_list = []
        for frame in trajectory:
            if isinstance(frame, dict):
                atoms = self._frame_to_atoms(frame)
                atoms_list.append(atoms)
            else:
                atoms_list.append(frame)

        # Write with extxyz format
        write(output_path, atoms_list, format="extxyz", columns=columns)

    def _frame_to_atoms(self, frame: dict[str, Any]) -> Atoms:
        """Convert frame dictionary to Atoms object.

        Args:
            frame: Frame dictionary with positions, cell, etc.

        Returns:
            ASE Atoms object
        """
        from ase import Atoms as AtomsClass

        if "atoms" in frame:
            return frame["atoms"]
        # At this point "atoms" is not in the frame (handled above), so the
        # frame must carry explicit symbols; a bare "positions"-only dict
        # previously crashed with an obscure AttributeError inside the
        # ``frame.get("symbols", ...)`` default expression.
        symbols = frame.get("symbols")
        if symbols is None:
            raise ValueError(
                "Trajectory frame must contain 'symbols' (or an 'atoms' object)."
            )
        atoms = AtomsClass(
            symbols=symbols,
            positions=frame["positions"],
            cell=frame.get("cell"),
            pbc=frame.get("pbc", True),
        )

        # Add info
        if "energy" in frame:
            atoms.info["energy"] = frame["energy"]
        if "forces" in frame:
            atoms.arrays["forces"] = frame["forces"]

        return atoms

    def write_xyz(
        self,
        trajectory: list[Atoms],
        output_path: Path | str,
    ) -> None:
        """Write in standard XYZ format.

        Simple XYZ format for compatibility with external tools.

        Args:
            trajectory: List of ASE Atoms objects
            output_path: Output file path
        """
        output_path = Path(output_path)

        if not trajectory:
            return

        write(output_path, trajectory, format="xyz")
