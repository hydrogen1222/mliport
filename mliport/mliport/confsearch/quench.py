"""Local minimization ("quench") and top-K refinement (spec sections 7.4, 10.3).

Every quench is a fixed-cell ASE FIRE run on the relax set ``R`` with only the
input ``FixAtoms`` constraints active.  The frame's velocities are dropped.
Non-finite values, a raw-force abort or a budget overrun are outcomes, not
exceptions: they are stored as ``quench_failed`` / ``quench_not_converged`` and
are never ranked and never become chain states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ase.constraints import FixAtoms
from ase.optimize import FIRE

from mliport.confsearch import status as cs_status
from mliport.confsearch.errors import ConfsearchInternalError
from mliport.runners.safety import (
    ForceSafetyAbort,
    evaluate_force_abort,
    force_views,
)

if TYPE_CHECKING:
    from ase import Atoms

    from mliport.confsearch.options import ConfsearchOptions

#: FIRE defaults recorded with every quench (ASE's own default maxstep).
FIRE_MAXSTEP_A = 0.2


@dataclass(frozen=True, slots=True)
class QuenchResult:
    """Outcome of one fixed-cell quench."""

    status: str
    atoms: Atoms
    energy_eV: float
    fmax_eV_A: float
    steps: int
    maxstep_A: float = FIRE_MAXSTEP_A

    @property
    def converged(self) -> bool:
        return self.status == cs_status.VALID or self.status == cs_status.REACTED

    @property
    def usable(self) -> bool:
        return self.status not in {
            cs_status.QUENCH_FAILED,
            cs_status.QUENCH_NOT_CONVERGED,
        }

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "energy_eV": float(self.energy_eV),
            "fmax_eV_A": float(self.fmax_eV_A),
            "steps": int(self.steps),
            "maxstep_A": float(self.maxstep_A),
        }


def quench(
    atoms: Atoms,
    calculator,
    options: ConfsearchOptions,
    *,
    fixed_indices: np.ndarray,
    fmax: float | None = None,
    steps: int | None = None,
) -> QuenchResult:
    """Minimize ``atoms`` at fixed cell with FIRE on the relax set.

    ``fixed_indices`` becomes the only constraint; everything else moves.  The
    input frame is copied, so the caller's structure is never modified.
    """
    work = atoms.copy()
    work.calc = calculator
    if "momenta" in work.arrays:
        del work.arrays["momenta"]
    fixed = np.asarray(fixed_indices, dtype=int)
    work.set_constraint(FixAtoms(indices=fixed) if fixed.size else None)

    budget = int(options.cs_quench_steps if steps is None else steps)
    target = float(options.cs_quench_fmax if fmax is None else fmax)
    threshold = float(options.fmax_abort)

    abort: dict[str, object] = {}

    def observe() -> None:
        energy = float(work.get_potential_energy())
        raw, applied = force_views(work, energy, context="confsearch quench")
        gate = evaluate_force_abort(raw, threshold=threshold)
        if gate.abort:
            abort["reason"] = "force"
            abort["max_force"] = gate.max_force_raw
            abort["atom_index"] = gate.atom_index
            raise ForceSafetyAbort(
                step=int(optimizer.nsteps),
                max_force=gate.max_force_raw,
                atom_index=gate.atom_index,
                threshold=threshold,
            )
        del applied

    optimizer = FIRE(work, logfile=None, maxstep=FIRE_MAXSTEP_A)
    optimizer.attach(observe, interval=1)
    status = cs_status.QUENCH_NOT_CONVERGED
    try:
        optimizer.run(fmax=target, steps=budget)
    except ForceSafetyAbort:
        status = cs_status.QUENCH_FAILED
    except (RuntimeError, ValueError, FloatingPointError):
        status = cs_status.QUENCH_FAILED
    else:
        if optimizer.converged():
            status = cs_status.VALID

    energy = float(work.get_potential_energy())
    if not np.isfinite(energy):
        status = cs_status.QUENCH_FAILED
    try:
        raw, _applied = force_views(work, energy, context="confsearch quench final")
    except (RuntimeError, ValueError):
        status = cs_status.QUENCH_FAILED
        fmax_value = float("nan")
    else:
        magnitudes = np.linalg.norm(raw, axis=1)
        free = np.setdiff1d(np.arange(len(work)), fixed)
        fmax_value = float(np.max(magnitudes[free])) if free.size else 0.0

    # A converged optimizer that left the fixed atoms moved is an internal bug.
    if fixed.size and status == cs_status.VALID:
        deltas = np.asarray(work.get_positions()) - np.asarray(atoms.get_positions())
        if np.any(deltas[fixed] != 0.0):
            raise ConfsearchInternalError(
                "A fixed atom moved during a confsearch quench"
            )

    return QuenchResult(
        status=status,
        atoms=work,
        energy_eV=energy,
        fmax_eV_A=fmax_value,
        steps=int(optimizer.nsteps),
    )


def refine(
    atoms: Atoms,
    calculator,
    options: ConfsearchOptions,
    *,
    fixed_indices: np.ndarray,
) -> QuenchResult:
    """Re-quench a representative at the tighter refinement criterion."""
    if not options.cs_refine:
        raise ConfsearchInternalError("refine() called with cs_refine disabled")
    return quench(
        atoms,
        calculator,
        options,
        fixed_indices=fixed_indices,
        fmax=options.cs_refine_fmax,
        steps=options.cs_refine_steps,
    )


__all__ = ["FIRE_MAXSTEP_A", "QuenchResult", "quench", "refine"]
