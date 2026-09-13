"""Canonical framework-drift definitions (task book PR-F section 8).

Native MSD and kinisi transport must not silently use different physical
definitions of "the framework drift".  Every backend records both:

* ``drift_mode``       -- ``none``, ``arithmetic_mean`` or ``mass_weighted_com``
  (the reference centre definition);
* ``drift_reference``  -- ``none``, ``nonmobile``, ``all`` or ``indices``
  (which atoms define the drift).

Cross-backend comparisons are only meaningful when both fields agree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mliport.analysis.dataset import TrajectoryDataset

DRIFT_MODES = ("none", "arithmetic_mean", "mass_weighted_com")
DRIFT_REFERENCES = ("none", "nonmobile", "all", "indices")

_CENTER_DEFINITIONS = {
    "none": "no framework drift correction",
    "arithmetic_mean": (
        "unweighted arithmetic mean displacement of the reference atoms"
    ),
    "mass_weighted_com": (
        "mass-weighted center-of-mass displacement of the reference atoms"
    ),
}


def reference_indices(
    dataset: TrajectoryDataset,
    *,
    mobile: np.ndarray,
    drift_reference: str,
    drift_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Resolve the explicit drift-reference selection (no silent defaults)."""
    mode = str(drift_reference).lower()
    if mode not in DRIFT_REFERENCES:
        raise ValueError(
            f"drift_reference must be one of {DRIFT_REFERENCES}, got {mode!r}"
        )
    if mode == "none":
        if drift_indices is not None:
            raise ValueError("drift_indices is only valid with drift_reference=indices")
        return np.asarray([], dtype=int)
    if mode == "indices":
        if drift_indices is None:
            raise ValueError("drift_reference=indices requires drift_indices")
        reference = dataset.select(indices=drift_indices)
        if np.intersect1d(mobile, reference).size:
            raise ValueError("Drift reference indices overlap the mobile selection")
        return reference
    if mode == "all":
        return np.arange(dataset.natoms, dtype=int)
    reference = np.setdiff1d(np.arange(dataset.natoms), mobile)
    if len(reference) == 0:
        raise ValueError(
            "drift_reference=nonmobile requires at least one framework atom"
        )
    return reference


def drift_displacement(
    continuous_positions_A: np.ndarray,
    *,
    reference: np.ndarray,
    masses: np.ndarray,
    drift_mode: str,
) -> np.ndarray:
    """Per-frame drift displacement for the chosen centre definition."""
    mode = str(drift_mode).lower()
    if mode not in DRIFT_MODES:
        raise ValueError(f"drift_mode must be one of {DRIFT_MODES}, got {mode!r}")
    positions = np.asarray(continuous_positions_A, dtype=float)
    if mode == "none" or len(reference) == 0:
        return np.zeros((positions.shape[0], 3), dtype=float)
    displacement = positions[:, reference] - positions[0, reference]
    if mode == "arithmetic_mean":
        return np.mean(displacement, axis=1)
    weights = np.asarray(masses, dtype=float)[reference]
    return np.average(displacement, axis=1, weights=weights)


def drift_semantics(
    *,
    drift_mode: str,
    drift_reference: str,
    reference: np.ndarray,
    symbols: tuple[str, ...] | list[str],
) -> dict[str, object]:
    """Machine-readable drift identity recorded by every backend."""
    mode = str(drift_mode).lower()
    reference_mode = str(drift_reference).lower()
    return {
        "drift_mode": mode,
        "drift_reference": reference_mode,
        "reference_indices": np.asarray(reference, dtype=int),
        "reference_species": sorted({symbols[index] for index in reference}),
        "center_definition": _CENTER_DEFINITIONS[mode],
        "specification": (
            "explicit: drift_mode + drift_reference are recorded per backend; "
            "cross-backend comparison requires both to match"
        ),
    }


def drift_definitions_match(
    left: dict[str, object] | None, right: dict[str, object] | None
) -> bool:
    """True only when two recorded drift definitions are identical."""
    if not left or not right:
        return False
    return str(left.get("drift_mode")) == str(right.get("drift_mode")) and str(
        left.get("drift_reference")
    ) == str(right.get("drift_reference"))


__all__ = [
    "DRIFT_MODES",
    "DRIFT_REFERENCES",
    "drift_definitions_match",
    "drift_displacement",
    "drift_semantics",
    "reference_indices",
]
