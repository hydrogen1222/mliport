"""Rigid-rotation conventions for the invariance tier.

Column-vector convention throughout::

    x' = R x            positions (row-stored: pos' = pos @ R.T)
    F' = R F            forces  (row-stored: F'   = F @ R.T)
    sigma' = R sigma R^T

The stress transform is unit-tested against a synthetic tensor (CPU test,
``tests/.../test_science_harness.py``) before any real-model use, per the
taskbook's "synthetic tensor test first" rule for section 11.6.
"""

from __future__ import annotations

import numpy as np


def proper_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Right-handed rotation matrix about ``axis`` (Rodrigues formula)."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        msg = "rotation axis must be non-zero"
        raise ValueError(msg)
    k = axis / norm
    kx, ky, kz = k
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    theta = float(angle_rad)
    return (
        np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    )


def rotate_cell_and_positions(atoms, rotation: np.ndarray):
    """Return a new Atoms with cell and positions rotated by ``R``.

    Column-vector convention: rows of the stored arrays are multiplied as
    ``row @ R.T``; the cell matrix rows are the lattice vectors, so they are
    transformed with the same rule.  PBC flags are preserved.
    """
    import copy

    new = atoms.copy()
    new.set_positions(np.asarray(atoms.positions) @ rotation.T)
    new.set_cell(np.asarray(atoms.cell.array) @ rotation.T)
    return new


def rotate_vectors(forces: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Transform per-atom vectors (rows) as ``F' = R F``."""
    return np.asarray(forces) @ np.asarray(rotation).T


def voigt_to_matrix(voigt) -> np.ndarray:
    """ASE Voigt order ``xx, yy, zz, yz, xz, xy`` -> full symmetric 3x3."""
    v = np.asarray(voigt, dtype=float)
    xx, yy, zz, yz, xz, xy = v
    return np.array(
        [
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ]
    )


def matrix_to_voigt(mat: np.ndarray) -> np.ndarray:
    """Full symmetric 3x3 -> ASE Voigt order ``xx, yy, zz, yz, xz, xy``."""
    m = np.asarray(mat, dtype=float)
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]])


def rotate_stress_voigt(voigt, rotation: np.ndarray) -> np.ndarray:
    """Transform a stress tensor given in ASE Voigt order: ``R sigma R^T``."""
    mat = voigt_to_matrix(voigt)
    r = np.asarray(rotation, dtype=float)
    return matrix_to_voigt(r @ mat @ r.T)
