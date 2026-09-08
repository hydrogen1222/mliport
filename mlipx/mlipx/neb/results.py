"""Calculator-free NEB result summaries and barrier definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _max_atom_force(forces: np.ndarray) -> float:
    values = np.asarray(forces, dtype=float)
    if values.size == 0:
        return 0.0
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("forces must have shape (images, atoms, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("forces contain NaN or Inf")
    return float(np.linalg.norm(values, axis=2).max())


def barrier_metrics(energies_eV: np.ndarray) -> dict[str, float | int | str | None]:
    """Return sampled barriers without inventing a fitted saddle."""

    energies = np.asarray(energies_eV, dtype=float)
    if energies.ndim != 1 or len(energies) < 3:
        raise ValueError("NEB requires at least three image energies")
    if not np.all(np.isfinite(energies)):
        raise ValueError("NEB energies contain NaN or Inf")
    peak_index = int(np.argmax(energies))
    peak = float(energies[peak_index])
    internal = 0 < peak_index < len(energies) - 1
    return {
        "barrier_forward_sampled_eV": peak - float(energies[0]),
        "barrier_reverse_sampled_eV": peak - float(energies[-1]),
        "reaction_energy_eV": float(energies[-1] - energies[0]),
        "highest_energy_image_index": peak_index,
        "barrier_status": (
            "converged_path_estimate" if internal else "no_internal_saddle_identified"
        ),
        "barrier_forward_fitted_eV": None,
    }


@dataclass(slots=True)
class NEBResult:
    """In-memory result; output writers are intentionally a later PR."""

    status: str
    converged: bool
    energies_eV: np.ndarray
    physical_forces_eV_A: np.ndarray
    neb_forces_eV_A: np.ndarray
    stages: list[dict]
    climbing_image_index: int | None
    max_neb_force_eV_A: float
    climbing_physical_fmax_eV_A: float | None

    def to_dict(self) -> dict:
        metrics = barrier_metrics(self.energies_eV)
        metrics.update(
            {
                "schema": "mlipx.neb-results/1",
                "status": self.status,
                "converged": self.converged,
                "energy_unit": "eV",
                "barrier_unit": "eV",
                "climb": self.climbing_image_index is not None,
                "climbing_image_index": self.climbing_image_index,
                "max_neb_force_eV_A": self.max_neb_force_eV_A,
                "climbing_physical_fmax_eV_A": self.climbing_physical_fmax_eV_A,
                "stages": self.stages,
                "saddle_validation": "not_performed",
            }
        )
        return metrics
