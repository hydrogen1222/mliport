"""Small, deterministic physical checks for opt-in backend validation."""

from __future__ import annotations

import numpy as np


def evaluate(atoms) -> dict:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
    if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
        raise ValueError("Nonfinite energy/force in validation")
    result = {"energy_eV": energy, "forces_eV_A": forces}
    if "stress" in atoms.calc.implemented_properties and atoms.cell.rank == 3:
        stress = np.asarray(atoms.get_stress(), dtype=float)
        if not np.all(np.isfinite(stress)):
            raise ValueError("Nonfinite stress in validation")
        result["stress_eV_A3"] = stress
    return result


def displacement_checks(atoms, *, force_atol: float, repeat_atol: float) -> dict:
    """A-B-A and six central differences at three predetermined displacements.

    Does not alter the caller's geometry or calculator; <=32 atoms only.
    Tolerances must be specified before evaluation, not inferred from errors.
    """
    if not 2 <= len(atoms) <= 32:
        raise ValueError("Validation requires 2–32 atoms")
    if (
        not np.isfinite([force_atol, repeat_atol]).all()
        or min(force_atol, repeat_atol) <= 0
    ):
        raise ValueError("Validation tolerances must be finite and positive")
    if atoms.constraints:
        raise ValueError("This bounded validation fixture requires unconstrained atoms")
    probe = atoms.copy()
    probe.calc = atoms.calc
    origin = probe.positions.copy()
    a = evaluate(probe)
    probe.positions[0, 0] += 0.03
    evaluate(probe)
    probe.positions[:] = origin
    repeated = evaluate(probe)
    repeat_errors = {
        key: float(np.max(np.abs(np.asarray(a[key]) - np.asarray(repeated[key]))))
        for key in a
    }
    samples = []
    for delta in (5e-4, 1e-3, 2e-3):
        errors = []
        for atom in (0, 1):
            for axis in range(3):
                probe.positions[:] = origin
                probe.positions[atom, axis] += delta
                plus = float(probe.get_potential_energy())
                probe.positions[atom, axis] -= 2 * delta
                minus = float(probe.get_potential_energy())
                numerical = -(plus - minus) / (2 * delta)
                error = abs(numerical - a["forces_eV_A"][atom, axis])
                if not np.isfinite(error):
                    raise ValueError("Nonfinite central difference")
                errors.append(float(error))
        samples.append(
            {"delta_A": delta, "errors_eV_A": errors, "max_error_eV_A": max(errors)}
        )
    return {
        "single_point": {
            "status": "passed",
            "energy_eV": a["energy_eV"],
            "max_force_eV_A": float(np.linalg.norm(a["forces_eV_A"], axis=1).max()),
            "stress_checked": "stress_eV_A3" in a,
        },
        "shared_calculator": {
            "status": "passed"
            if max(repeat_errors.values()) <= repeat_atol
            else "failed",
            "absolute_tolerance": repeat_atol,
            "repeat_errors": repeat_errors,
        },
        "energy_force_consistency": {
            "status": "passed"
            if all(s["max_error_eV_A"] <= force_atol for s in samples)
            else "failed",
            "force_atol_eV_A": force_atol,
            "free_dof_count": 6,
            "samples": samples,
        },
    }
