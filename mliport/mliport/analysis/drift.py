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


#: Fraction of the mobile-ion mean-squared displacement above which an
#: uncorrected whole-system centre-of-mass motion is reported.
SYSTEM_COM_DRIFT_WARNING_FRACTION = 0.05


def system_com_drift_diagnostics(
    continuous_positions_A: np.ndarray,
    *,
    masses: np.ndarray,
    mobile: np.ndarray,
    drift_reference: str,
) -> dict[str, object]:
    """Quantify whole-system COM motion relative to the mobile-ion MSD.

    A thermostat that does not conserve a zero total momentum (or a run that
    never removed the initial COM momentum) moves every atom together. That
    collective translation is not ion transport, yet without a drift
    correction it adds ``|dR_COM|^2`` to every mobile-ion MSD. The ratio is
    evaluated at the longest time separation of the analysed window.
    """
    positions = np.asarray(continuous_positions_A, dtype=float)
    weights = np.asarray(masses, dtype=float)
    com = np.average(positions, axis=1, weights=weights)
    com_sq = np.sum((com - com[0]) ** 2, axis=1)
    mobile_disp = positions[:, mobile] - positions[0, mobile]
    mobile_msd = np.mean(np.sum(mobile_disp**2, axis=2), axis=1)
    final_mobile = float(mobile_msd[-1])
    final_com = float(com_sq[-1])
    max_com = float(np.sqrt(np.max(com_sq)))
    ratio = final_com / final_mobile if final_mobile > 0 else None
    corrected = str(drift_reference).lower() != "none"
    warning = None
    excessive = ratio is not None and ratio > SYSTEM_COM_DRIFT_WARNING_FRACTION
    if excessive and not corrected:
        warning = (
            "The whole-system centre of mass moved "
            f"{np.sqrt(final_com):.3g} A over the analysed window, which is "
            f"{100.0 * ratio:.1f}% of the uncorrected mobile-ion MSD. That "
            "collective translation is not ion transport. Re-run with "
            "--drift-reference nonmobile (or all), or use an MD COM policy that "
            "removes the centre-of-mass momentum."
        )
    return {
        "system_com_max_displacement_A": max_com,
        "system_com_final_displacement_A": float(np.sqrt(final_com)),
        "mobile_msd_final_A2": final_mobile,
        "system_com_sq_over_mobile_msd": ratio,
        "warning_fraction": SYSTEM_COM_DRIFT_WARNING_FRACTION,
        "drift_corrected": corrected,
        "warning": warning,
    }


__all__ = [
    "DRIFT_MODES",
    "DRIFT_REFERENCES",
    "SYSTEM_COM_DRIFT_WARNING_FRACTION",
    "drift_definitions_match",
    "drift_displacement",
    "drift_semantics",
    "reference_indices",
    "system_com_drift_diagnostics",
]
