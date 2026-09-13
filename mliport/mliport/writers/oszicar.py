# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
VASP-style OSZICAR writer.

OSZICAR contains optimization progress information similar to VASP's OSZICAR file.
Tracks energy, forces, and convergence during geometry optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any


class OszicarWriter:
    """Write optimization progress in VASP OSZICAR-like format.

    Tracks the optimization trajectory with step number, energy,
    and maximum force for monitoring convergence.

    Example:
        >>> writer = OszicarWriter()
        >>> writer.write_header(Path("OSZICAR"))
        >>> writer.append_step(Path("OSZICAR"), step=0, energy=-123.45, forces=forces)
    """

    def __init__(self):
        """Initialize OSZICAR writer."""
        self.header_written = False

    def write_header(self, output_path: Path | str) -> None:
        """Write OSZICAR header.

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)

        header = [
            "=" * 70,
            " MLIP OPTIMIZATION PROGRESS",
            "=" * 70,
            "",
            f"{'Step':>8} {'Energy (eV)':>18} {'E/atom (eV)':>14} {'Fmax (eV/Å)':>14} {'FRMS (eV/Å)':>14}",
            "-" * 70,
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")

        self.header_written = True

    def append_step(
        self,
        output_path: Path | str,
        step: int,
        energy: float,
        forces: "np.ndarray",
        natoms: int,
        fmax_target: float | None = None,
    ) -> None:
        """Append optimization step to OSZICAR.

        Args:
            output_path: Output file path
            step: Optimization step number
            energy: Total energy in eV
            forces: Forces array (N, 3)
            natoms: Number of atoms
            fmax_target: Target fmax for convergence (optional)
        """
        output_path = Path(output_path)

        # Calculate force statistics
        force_mags = np.linalg.norm(forces, axis=1)
        fmax = np.max(force_mags)
        frms = np.sqrt(np.mean(force_mags**2))

        # Format line
        line = (
            f"{step:>8} {energy:>18.8f} {energy / natoms:>14.6f} "
            f"{fmax:>14.6f} {frms:>14.6f}"
        )

        # Add convergence indicator
        if fmax_target is not None:
            converged = "*" if fmax <= fmax_target else " "
            line += f"  [{converged}]"

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_footer(
        self, output_path: Path | str, converged: bool, nsteps: int
    ) -> None:
        """Write OSZICAR footer with final status.

        Args:
            output_path: Output file path
            converged: Whether optimization converged
            nsteps: Total number of steps
        """
        output_path = Path(output_path)

        footer = [
            "-" * 70,
            f"Optimization {'converged' if converged else 'not converged'} after {nsteps} steps",
            "=" * 70,
        ]

        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(footer) + "\n")

    def write(
        self,
        output_path: Path | str,
        trajectory: list[dict[str, Any]],
        fmax_target: float | None = None,
    ) -> None:
        """Write complete optimization trajectory to OSZICAR.

        Args:
            output_path: Output file path
            trajectory: List of step dictionaries with 'step', 'energy', 'forces'
            fmax_target: Target fmax for convergence
        """
        self.write_header(output_path)

        for step_data in trajectory:
            self.append_step(
                output_path,
                step=step_data["step"],
                energy=step_data["energy"],
                forces=step_data["forces"],
                natoms=step_data.get("natoms", len(step_data["forces"])),
                fmax_target=fmax_target,
            )

        # Determine convergence
        if trajectory:
            last_step = trajectory[-1]
            forces = last_step["forces"]
            force_mags = np.linalg.norm(forces, axis=1)
            fmax = np.max(force_mags)
            converged = fmax_target is not None and fmax <= fmax_target
            nsteps = last_step["step"]
            self.write_footer(output_path, converged, nsteps)
