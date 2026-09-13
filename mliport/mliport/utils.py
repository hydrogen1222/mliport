# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for mliport.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


def calculate_pressure(stress_voigt: "np.ndarray") -> float:
    """Calculate pressure from stress tensor (Voigt notation).

    Args:
        stress_voigt: Stress in Voigt notation [xx, yy, zz, yz, xz, xy] in eV/Å³

    Returns:
        Pressure in GPa
    """
    # Trace of stress / 3, convert eV/Å³ to GPa (1 eV/Å³ = 160.2177 GPa)
    pressure_ev_a3 = -(stress_voigt[0] + stress_voigt[1] + stress_voigt[2]) / 3.0
    pressure_gpa = pressure_ev_a3 * 160.2177
    return float(pressure_gpa)


def stress_voigt_to_tensor(stress_voigt: "np.ndarray") -> "np.ndarray":
    """Convert Voigt notation stress to 3x3 tensor.

    Args:
        stress_voigt: [xx, yy, zz, yz, xz, xy]

    Returns:
        3x3 stress tensor
    """
    tensor = np.zeros((3, 3))
    tensor[0, 0] = stress_voigt[0]  # xx
    tensor[1, 1] = stress_voigt[1]  # yy
    tensor[2, 2] = stress_voigt[2]  # zz
    tensor[1, 2] = tensor[2, 1] = stress_voigt[3]  # yz
    tensor[0, 2] = tensor[2, 0] = stress_voigt[4]  # xz
    tensor[0, 1] = tensor[1, 0] = stress_voigt[5]  # xy
    return tensor


def format_lattice(cell: "np.ndarray") -> str:
    """Format lattice vectors for output.

    Args:
        cell: 3x3 cell matrix

    Returns:
        Formatted string
    """
    lines = []
    for i in range(3):
        lines.append(f"  {cell[i][0]:12.6f}  {cell[i][1]:12.6f}  {cell[i][2]:12.6f}")
    return "\n".join(lines)


def get_atom_type_counts(atoms: Atoms) -> dict[str, int]:
    """Get dictionary of element symbol to count.

    Args:
        atoms: ASE Atoms object

    Returns:
        Dictionary mapping element symbols to counts
    """
    from collections import Counter

    symbols = atoms.get_chemical_symbols()
    return dict(Counter(symbols))


def format_atom_counts(atoms: Atoms) -> str:
    """Format atom counts as string.

    Args:
        atoms: ASE Atoms object

    Returns:
        Formatted string like "Li: 24, P: 4, S: 20"
    """
    counts = get_atom_type_counts(atoms)
    return ", ".join([f"{k}: {v}" for k, v in sorted(counts.items())])


def find_structure_file(directory: Path | str = ".") -> Path | None:
    """Find a structure file in directory.

    Searches for common structure file names in order of priority.

    Args:
        directory: Directory to search

    Returns:
        Path to structure file or None if not found
    """
    directory = Path(directory)

    # Priority order
    patterns = [
        "POSCAR",
        "CONTCAR",
        "*.cif",
        "*.xyz",
        "*.vasp",
        "POSCAR*",
    ]

    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]

    return None


def check_structure_valid(atoms: Atoms) -> tuple[bool, str]:
    """Check if structure is valid for calculation.

    Args:
        atoms: ASE Atoms object

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for zero positions
    if len(atoms) == 0:
        return False, "Structure has no atoms"

    # Check for NaN positions
    if np.any(np.isnan(atoms.positions)):
        return False, "Structure contains NaN positions"

    # Check cell. The *volume* (|det|) is what matters: a degenerate or
    # left-handed cell can have all-positive diagonal elements while its
    # determinant is zero/negative (e.g. linearly dependent lattice
    # vectors), so checking np.diag is not sufficient.
    if atoms.pbc.any():
        if atoms.cell.volume <= 0:
            return False, "Invalid cell (zero or negative volume)"

    return True, ""


def estimate_memory(atoms: Atoms, model_size: str = "small") -> dict[str, float]:
    """Estimate memory requirements for calculation.

    Args:
        atoms: ASE Atoms object
        model_size: Model size (small, medium, large)

    Returns:
        Dictionary with memory estimates in MB
    """
    natoms = len(atoms)

    # Rough estimates based on model size
    model_params = {
        "small": 100,  # million parameters
        "medium": 300,
        "large": 1000,
    }

    params = model_params.get(model_size, 100)

    # Model memory (4 bytes per float32 parameter)
    model_mb = params * 4

    # Activation memory (rough estimate)
    activation_mb = natoms * params * 0.001

    # Total
    total_mb = model_mb + activation_mb

    return {
        "model_mb": model_mb,
        "activation_mb": activation_mb,
        "total_mb": total_mb,
        "total_gb": total_mb / 1024,
    }
