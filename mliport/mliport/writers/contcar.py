# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
VASP-style CONTCAR writer.

Writes the final/optimized structure in VASP POSCAR/CONTCAR format.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ase.io import write

if TYPE_CHECKING:
    from ase import Atoms

    import numpy as np


class ContcarWriter:
    """Write structure in VASP CONTCAR format.

        The CONTCAR file contains the optimized/final structure after
    gometry optimization or MD simulation.

        Example:
            >>> writer = ContcarWriter()
            >>> writer.write(atoms, Path("CONTCAR"), comment="Optimized structure")
    """

    def write(
        self,
        atoms: Atoms,
        output_path: Path | str,
        comment: str = "mliport optimized structure",
        direct: bool = False,
    ) -> None:
        """Write structure to CONTCAR file.

        Args:
            atoms: ASE Atoms object
            output_path: Output file path
            comment: Comment line for CONTCAR
            direct: Use direct (fractional) coordinates instead of Cartesian
        """
        output_path = Path(output_path)

        # ASE's vasp writer handles the format
        format_str = "vasp"

        # Write with ASE
        write(output_path, atoms, format=format_str, direct=direct)

        # Prepend comment if not using ASE's default
        if comment:
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            lines[0] = comment

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    def write_with_energy(
        self,
        atoms: Atoms,
        output_path: Path | str,
        energy: float | None = None,
        forces: "np.ndarray" | None = None,
    ) -> None:
        """Write structure with energy and forces in comment.

        Args:
            atoms: ASE Atoms object
            output_path: Output file path
            energy: Total energy to include in comment
            forces: Forces to store in atoms.info
        """
        output_path = Path(output_path)

        # Store results in atoms.info for reference
        if energy is not None:
            atoms.info["energy"] = energy
        if forces is not None:
            atoms.info["forces"] = forces

        comment = f"{atoms.get_chemical_formula()}"
        if energy is not None:
            comment += f" E={energy:.6f} eV"

        self.write(atoms, output_path, comment=comment)
