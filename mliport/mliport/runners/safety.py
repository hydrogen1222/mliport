"""Stateless numerical-safety helpers shared by the MD and confsearch runners.

These helpers were extracted from :class:`mliport.runners.md.MDRunner` without
changing their behaviour.  They validate one atomic state at a time, so both
the MD step observer and the confsearch cycle engine can enforce the same
contracts:

* :func:`validate_atomic_state` -- positions, cell, PBC, masses and momenta are
  finite and shape-consistent before anything reaches an ASE integrator.
* :func:`check_finite` -- energy and forces are finite; NaN/inf is always fatal.
* :func:`force_views` -- the raw model forces and the constraint-applied forces
  are validated separately, and the raw view is validated *before* a constraint
  projector transforms it.
* :func:`evaluate_force_abort` -- the ``fmax_abort`` decision is based on the
  raw (unprojected) model forces, so a frozen atom can never hide a large
  reaction force from the safety gate.
* :func:`validate_safe_point` -- the full per-frame safe point (atomic state,
  both force views, kinetic/total energy, temperature, volume).

Nothing here holds state or writes files, so the behaviour is identical for
every caller; ``MDRunner`` delegates to these functions and the confsearch
engine calls them directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from ase import Atoms


class ForceSafetyAbort(RuntimeError):
    """Raised after checkpointing a frame that exceeds ``fmax_abort``."""

    def __init__(
        self, *, step: int, max_force: float, atom_index: int, threshold: float
    ):
        self.step = step
        self.max_force = max_force
        self.atom_index = atom_index
        self.threshold = threshold
        super().__init__(
            f"Force safety abort at MD step {step}: atom {atom_index} has "
            f"raw model force |F|={max_force:.6g} eV/Angstrom, exceeding "
            f"fmax_abort={threshold:.6g} eV/Angstrom"
        )


@dataclass(frozen=True, slots=True)
class ForceAbortEvaluation:
    """Outcome of the raw-force safety gate for one frame."""

    max_force_raw: float
    atom_index: int
    threshold: float

    @property
    def abort(self) -> bool:
        """True when the largest raw per-atom force exceeds the threshold."""
        return self.max_force_raw > self.threshold


def finite_float(value: object, *, name: str) -> float:
    """Return a finite real scalar without accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric


def strict_integer(value: object, *, name: str) -> int:
    """Return an integer while rejecting bool and fractional real values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    raise TypeError(f"{name} must be an integer, got {value!r}")


def validate_atomic_state(atoms: Atoms, *, context: str) -> None:
    """Validate the complete atomic state before it reaches an integrator."""
    if len(atoms) == 0:
        raise ValueError(f"{context}: atoms must contain at least one atom")

    positions = np.asarray(atoms.positions, dtype=float)
    if positions.shape != (len(atoms), 3) or not np.all(np.isfinite(positions)):
        raise ValueError(
            f"{context}: positions must have shape ({len(atoms)}, 3) and "
            "contain only finite values"
        )

    cell = np.asarray(atoms.cell, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError(
            f"{context}: cell must have shape (3, 3) and contain only " "finite values"
        )
    pbc = np.asarray(atoms.pbc, dtype=bool)
    periodic_dimensions = int(np.count_nonzero(pbc))
    if periodic_dimensions and np.linalg.matrix_rank(cell[pbc]) < periodic_dimensions:
        raise ValueError(
            f"{context}: periodic cell vectors must be non-zero and linearly "
            "independent"
        )

    masses = np.asarray(atoms.get_masses(), dtype=float)
    if masses.shape != (len(atoms),) or not np.all(np.isfinite(masses)):
        raise ValueError(
            f"{context}: masses must have shape ({len(atoms)},) and contain "
            "only finite values"
        )
    if np.any(masses <= 0.0):
        raise ValueError(f"{context}: every atomic mass must be > 0")

    if atoms.has("momenta"):
        momenta = np.asarray(atoms.get_momenta(), dtype=float)
        if momenta.shape != (len(atoms), 3) or not np.all(np.isfinite(momenta)):
            raise ValueError(
                f"{context}: momenta must have shape ({len(atoms)}, 3) and "
                "contain only finite values"
            )
        velocities = momenta / masses[:, None]
        if not np.all(np.isfinite(velocities)):
            raise ValueError(f"{context}: velocities must contain only finite values")


def check_finite(
    energy: float,
    forces: Any | None = None,
    *,
    context: str = "calculation",
) -> None:
    """Abort the run if energy/forces are non-finite (NaN/inf).

    MLIPs can return NaN when atoms get too close or leave the training
    distribution. Without this guard ASE integrators propagate NaN to the end
    and write "successful" NaN results. Non-finite values are always fatal, so
    this raises unconditionally (plan safety.abort_on_nan).
    """
    if not np.isfinite(energy):
        raise RuntimeError(
            f"Non-finite energy ({energy!r}) during {context}; aborting "
            "before NaN is written to outputs."
        )
    if forces is not None:
        arr = np.asarray(forces, dtype=float)
        if arr.size and not np.all(np.isfinite(arr)):
            fmax = (
                float(np.max(np.linalg.norm(arr, axis=1)))
                if arr.ndim == 2
                else float("nan")
            )
            raise RuntimeError(
                f"Non-finite forces during {context} (max|F|={fmax}); "
                "aborting before NaN is written to outputs."
            )


def force_views(
    atoms: Atoms,
    energy: float,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite raw and constraint-applied force arrays."""
    raw = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
    expected_shape = (len(atoms), 3)
    if raw.shape != expected_shape:
        raise RuntimeError(
            f"Malformed raw forces during {context}: expected "
            f"{expected_shape}, got {raw.shape}"
        )
    # Validate the calculator result before asking a constraint projector
    # to transform it. Projecting Inf/NaN can emit warnings and obscure
    # which layer first violated the finite-state contract.
    check_finite(energy, raw, context=f"{context} (raw forces)")

    applied = np.asarray(atoms.get_forces(apply_constraint=True), dtype=float)
    if applied.shape != expected_shape:
        raise RuntimeError(
            f"Malformed constraint-applied forces during {context}: expected "
            f"{expected_shape}, got {applied.shape}"
        )
    check_finite(energy, applied, context=f"{context} (constraint-applied forces)")
    return raw, applied


def force_magnitudes(forces: np.ndarray) -> np.ndarray:
    """Per-atom force norms for an ``(N, 3)`` array, with a shape check."""
    array = np.asarray(forces, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise RuntimeError(f"Force array must have shape (N, 3), got {array.shape}")
    return np.linalg.norm(array, axis=1)


def evaluate_force_abort(
    forces_raw: np.ndarray, *, threshold: float
) -> ForceAbortEvaluation:
    """Evaluate the raw-force safety gate without applying it.

    The caller decides whether to checkpoint and abort, so the same evaluation
    serves MD (raise :class:`ForceSafetyAbort`) and confsearch (record
    ``md_force_abort`` and stop the cycle).
    """
    magnitudes = force_magnitudes(forces_raw)
    atom_index = int(np.argmax(magnitudes)) if magnitudes.size else 0
    max_force = float(magnitudes[atom_index]) if magnitudes.size else 0.0
    return ForceAbortEvaluation(
        max_force_raw=max_force,
        atom_index=atom_index,
        threshold=float(threshold),
    )


def validate_safe_point(
    atoms: Atoms,
    *,
    energy: float,
    forces_raw: np.ndarray,
    forces_applied: np.ndarray,
    temperature_fn: Callable[[Atoms], float],
    context: str,
) -> tuple[float, float, float]:
    """Validate positions, momenta and scalar observables at a safe point.

    ``temperature_fn`` is the caller's constraint-aware temperature definition:
    MD and confsearch freeze different sets, so the degrees of freedom are not
    shared here.  The historical evaluation order is preserved: the kinetic
    energy is read, then the temperature is computed from the same state, then
    the scalar observables are checked.

    Returns ``(kinetic_energy, temperature, total_energy)``.
    """
    validate_atomic_state(atoms, context=context)
    check_finite(energy, forces_raw, context=context)
    check_finite(energy, forces_applied, context=context)
    kinetic_energy = float(atoms.get_kinetic_energy())
    temperature = float(temperature_fn(atoms))
    total_energy = float(energy) + kinetic_energy
    volume = float(atoms.get_volume())
    for name, value in (
        ("kinetic energy", kinetic_energy),
        ("total energy", total_energy),
        ("temperature", temperature),
        ("volume", volume),
    ):
        if not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite {name} ({value!r}) during {context}; aborting"
            )
    # A zero cell volume is valid for isolated molecules and can also be
    # valid for lower-dimensional PBC. Only a 3D-periodic system requires
    # a non-zero bulk volume.
    if bool(np.asarray(atoms.pbc, dtype=bool).all()) and volume <= 0.0:
        raise RuntimeError(f"Non-positive volume ({volume!r}) during {context}")
    return kinetic_energy, temperature, total_energy


__all__ = [
    "ForceAbortEvaluation",
    "ForceSafetyAbort",
    "check_finite",
    "evaluate_force_abort",
    "finite_float",
    "force_magnitudes",
    "force_views",
    "strict_integer",
    "validate_atomic_state",
    "validate_safe_point",
]
