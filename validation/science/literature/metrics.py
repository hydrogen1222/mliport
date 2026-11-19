"""Error metrics for literature-referenced MLIP benchmarks.

Every function in this module is pure NumPy and has no mliport import, so the
science checks (units, signs, known answers) run on a CPU-only CI machine
without any backend virtualenv.

Conventions (never changed silently):

* energies are **total** energies in eV per cell; per-atom errors divide by
  the *reference* cell's atom count;
* forces are eV / Angstrom, shape ``(n_atoms, 3)``;
* stress is the ASE Voigt 6-vector ``[xx, yy, zz, yz, xz, xy]`` in eV /
  Angstrom^3.  A reference that stores a **virial** tensor is never silently
  treated as stress: the importers record the virial separately and the
  evaluation reports stress as unavailable unless a dataset documents a
  verified stress label;
* a "group" (composition, formula, phase) is the unit for relative-energy
  metrics, because absolute energies from different exchange-correlation
  functionals are not comparable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

#: eV / Angstrom^3 -> GPa
EV_A3_TO_GPA = 160.21766208

#: Cosine/angle statistics are only meaningful above this reference magnitude.
COS_MIN_REF_FORCE_EV_A = 0.1

#: Reference force magnitude bins (eV / Angstrom).
FORCE_BINS = (0.1, 0.5, 1.0, 2.0)


def _as_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        msg = "metric requires at least one value"
        raise ValueError(msg)
    if not np.isfinite(array).all():
        msg = "metric input contains non-finite values"
        raise ValueError(msg)
    return array


def abs_stats(values: Any) -> dict[str, Any]:
    """Statistics of ``|x|`` plus the signed bias and RMSE."""
    signed = _as_float_array(values)
    magnitude = np.abs(signed)
    return {
        "n": int(magnitude.size),
        "mae": float(magnitude.mean()),
        "rmse": float(np.sqrt((signed**2).mean())),
        "median": float(np.median(magnitude)),
        "p95": float(np.percentile(magnitude, 95)),
        "p99": float(np.percentile(magnitude, 99)),
        "max": float(magnitude.max()),
        "bias": float(signed.mean()),
    }


def energy_per_atom_stats(
    e_ref: Sequence[float], e_model: Sequence[float], natoms: Sequence[int]
) -> dict[str, Any]:
    """Per-atom total-energy error statistics (eV/atom)."""
    ref = _as_float_array(e_ref)
    model = _as_float_array(e_model)
    counts = _as_float_array(natoms)
    if not (ref.shape == model.shape == counts.shape):
        msg = "e_ref, e_model and natoms must have the same length"
        raise ValueError(msg)
    return abs_stats((model - ref) / counts)


def vector_norm_stats(diff: np.ndarray) -> dict[str, Any]:
    """Statistics of per-atom vector magnitudes of ``diff``."""
    norms = np.linalg.norm(np.asarray(diff, dtype=float), axis=1)
    return abs_stats(norms)


def force_stats(
    f_ref: np.ndarray,
    f_model: np.ndarray,
    numbers: np.ndarray | None = None,
    symbols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Force error statistics, optionally stratified by element.

    Args:
        f_ref: reference forces, ``(n_atoms, 3)``, eV/Angstrom.
        f_model: model forces, same shape.
        numbers: atomic numbers of the same length (for per-element strata).
        symbols: element symbols aligned with ``numbers``; when omitted the
            strata are keyed by atomic number.

    Returns:
        Component and vector-norm errors, direction-angle and cosine
        statistics for non-negligible reference forces, errors stratified by
        reference force magnitude, and per-element breakdowns.
    """
    ref = _as_float_array(f_ref)
    model = _as_float_array(f_model)
    if ref.shape != model.shape or ref.ndim != 2 or ref.shape[1] != 3:
        msg = "f_ref and f_model must both have shape (n_atoms, 3)"
        raise ValueError(msg)
    diff = model - ref
    ref_norm = np.linalg.norm(ref, axis=1)
    model_norm = np.linalg.norm(model, axis=1)
    metrics: dict[str, Any] = {
        "n_atoms": int(ref.shape[0]),
        "component_mae_eV_A": float(np.abs(diff).mean()),
        "component_rmse_eV_A": float(np.sqrt((diff**2).mean())),
        "component_mae_by_axis_eV_A": {
            axis: float(np.abs(diff[:, i]).mean()) for i, axis in enumerate("xyz")
        },
        "vector_norm_error_eV_A": vector_norm_stats(diff),
        "reference_force_norm_eV_A": {
            "mean": float(ref_norm.mean()),
            "p95": float(np.percentile(ref_norm, 95)),
            "max": float(ref_norm.max()),
        },
        "model_force_norm_eV_A": {
            "mean": float(model_norm.mean()),
            "p95": float(np.percentile(model_norm, 95)),
            "max": float(model_norm.max()),
        },
    }

    mask = ref_norm >= COS_MIN_REF_FORCE_EV_A
    # a model that predicts exactly zero force has no defined direction; such
    # atoms are excluded instead of producing a NaN cosine
    mask = mask & (model_norm > 0.0)
    if mask.any():
        fr = ref[mask]
        fm = model[mask]
        cos = (fr * fm).sum(axis=1) / (
            np.linalg.norm(fr, axis=1) * np.linalg.norm(fm, axis=1)
        )
        cos = np.clip(cos, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos))
        metrics["direction"] = {
            "min_ref_force_eV_A": COS_MIN_REF_FORCE_EV_A,
            "n_atoms": int(mask.sum()),
            "cosine_mean": float(cos.mean()),
            "cosine_median": float(np.median(cos)),
            "cosine_min": float(cos.min()),
            "angle_deg_mean": float(angle_deg.mean()),
            "angle_deg_median": float(np.median(angle_deg)),
            "angle_deg_p95": float(np.percentile(angle_deg, 95)),
            "angle_deg_max": float(angle_deg.max()),
            "frac_angle_above_10deg": float((angle_deg > 10.0).mean()),
            "frac_angle_above_30deg": float((angle_deg > 30.0).mean()),
        }
    else:
        metrics["direction"] = {
            "min_ref_force_eV_A": COS_MIN_REF_FORCE_EV_A,
            "n_atoms": 0,
            "note": "no reference force reached the cosine threshold",
        }

    edges = [0.0, *FORCE_BINS, float("inf")]
    strata = []
    for lo, hi in pairwise(edges):
        selection = (ref_norm >= lo) & (ref_norm < hi)
        if not selection.any():
            continue
        strata.append(
            {
                "ref_force_lo_eV_A": lo,
                "ref_force_hi_eV_A": None if np.isinf(hi) else hi,
                "n_atoms": int(selection.sum()),
                "component_mae_eV_A": float(np.abs(diff[selection]).mean()),
                "component_rmse_eV_A": float(np.sqrt((diff[selection] ** 2).mean())),
                "vector_norm_mae_eV_A": float(
                    np.linalg.norm(diff[selection], axis=1).mean()
                ),
            }
        )
    metrics["by_ref_force_magnitude"] = strata

    if numbers is not None or symbols is not None:
        if numbers is not None:
            numbers = np.asarray(numbers)
            if numbers.shape[0] != ref.shape[0]:
                msg = "numbers must have one entry per atom"
                raise ValueError(msg)
        labels: Iterable[str]
        if symbols is not None:
            if len(symbols) != ref.shape[0]:
                msg = "symbols must have one entry per atom"
                raise ValueError(msg)
            labels = [str(s) for s in symbols]
        else:
            labels = [str(int(z)) for z in numbers]
        per_element: dict[str, Any] = {}
        for label in sorted(set(labels)):
            selection = np.array([lab == label for lab in labels])
            if not selection.any():
                continue
            per_element[label] = {
                "n_atoms": int(selection.sum()),
                "component_mae_eV_A": float(np.abs(diff[selection]).mean()),
                "component_rmse_eV_A": float(np.sqrt((diff[selection] ** 2).mean())),
                "vector_norm_mae_eV_A": float(
                    np.linalg.norm(diff[selection], axis=1).mean()
                ),
                "vector_norm_p95_eV_A": float(
                    np.percentile(np.linalg.norm(diff[selection], axis=1), 95)
                ),
                "reference_force_norm_mean_eV_A": float(ref_norm[selection].mean()),
            }
        metrics["by_element"] = per_element
    return metrics


def stress_stats(s_ref: np.ndarray, s_model: np.ndarray) -> dict[str, Any]:
    """Stress error statistics for ASE Voigt 6-vectors (eV/Angstrom^3).

    Hydrostatic and deviatoric components are separated because a uniform
    volume offset and a shape error are physically different failures.
    """
    ref = _as_float_array(s_ref)
    model = _as_float_array(s_model)
    if ref.shape != model.shape or ref.ndim != 2 or ref.shape[1] != 6:
        msg = "stress inputs must both have shape (n_structures, 6) in Voigt order"
        raise ValueError(msg)
    diff = model - ref
    hydro_ref = ref[:, :3].mean(axis=1)
    hydro_model = model[:, :3].mean(axis=1)
    dev_ref = ref.copy()
    dev_model = model.copy()
    dev_ref[:, :3] -= hydro_ref[:, None]
    dev_model[:, :3] -= hydro_model[:, None]
    return {
        "n_structures": int(ref.shape[0]),
        "voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
        "component_mae_eV_A3": float(np.abs(diff).mean()),
        "component_rmse_eV_A3": float(np.sqrt((diff**2).mean())),
        "component_mae_GPa": float(np.abs(diff).mean() * EV_A3_TO_GPA),
        "component_rmse_GPa": float(np.sqrt((diff**2).mean()) * EV_A3_TO_GPA),
        "hydrostatic_error_bias_eV_A3": float((hydro_model - hydro_ref).mean()),
        "hydrostatic_error_mae_GPa": float(
            np.abs(hydro_model - hydro_ref).mean() * EV_A3_TO_GPA
        ),
        "deviatoric_component_mae_eV_A3": float(np.abs(dev_model - dev_ref).mean()),
        "deviatoric_component_mae_GPa": float(
            np.abs(dev_model - dev_ref).mean() * EV_A3_TO_GPA
        ),
    }


def group_relative_energy_stats(
    e_ref: Sequence[float],
    e_model: Sequence[float],
    natoms: Sequence[int],
    groups: Sequence[str],
) -> dict[str, Any]:
    """Energy errors after removing the per-group mean (eV/atom).

    Absolute total energies from different exchange-correlation functionals
    differ by a large, arbitrary offset (and by pseudopotential convention).
    Comparing *relative* energies inside one chemical composition removes that
    offset, so the residual error measures the model's ability to describe
    structural/thermal energy differences.  The group mean is computed from
    the same set of frames for the reference and the model, so no reference
    value is ever leaked into the model prediction.
    """
    ref = _as_float_array(e_ref)
    model = _as_float_array(e_model)
    counts = _as_float_array(natoms)
    if not (ref.size == model.size == counts.size == len(groups)):
        msg = "e_ref, e_model, natoms and groups must have equal lengths"
        raise ValueError(msg)
    per_atom_ref = ref / counts
    per_atom_model = model / counts
    group_array = np.asarray([str(g) for g in groups])
    resid_ref = per_atom_ref.copy()
    resid_model = per_atom_model.copy()
    per_group: dict[str, Any] = {}
    for group in sorted(set(group_array)):
        selection = group_array == group
        ref_mean = per_atom_ref[selection].mean()
        model_mean = per_atom_model[selection].mean()
        resid_ref[selection] -= ref_mean
        resid_model[selection] -= model_mean
        if int(selection.sum()) >= 2:
            delta = resid_model[selection] - resid_ref[selection]
            per_group[group] = {
                "n_structures": int(selection.sum()),
                "n_atoms": int(counts[selection][0]),
                "reference_offset_eV_per_atom": float(ref_mean),
                "model_offset_eV_per_atom": float(model_mean),
                "offset_eV_per_atom": float(model_mean - ref_mean),
                "relative_rmse_eV_per_atom": float(np.sqrt((delta**2).mean())),
                "relative_mae_eV_per_atom": float(np.abs(delta).mean()),
            }
    delta_all = resid_model - resid_ref
    return {
        "definition": (
            "per-atom energy minus the mean per-atom energy of the same "
            "group (composition/formula); removes the arbitrary absolute "
            "offset between reference levels"
        ),
        "relative_energy_eV_per_atom": abs_stats(delta_all),
        "by_group": per_group,
    }


def formation_energies(
    e_total: Mapping[str, float],
    natoms: Mapping[str, int],
    x: Mapping[str, float],
    endpoint_low: str,
    endpoint_high: str,
) -> dict[str, float]:
    """LiPS-25 Task-1 formation energy per atom.

    The composition along the Li2S-P2S5 tie line is written as
    ``(Li2S)_x (P2S5)_(1-x)``, which contains ``7 - 4x`` atoms per formula
    unit.  Following the LiPS-25 reference implementation::

        Ef/atom = ( E_cell/(7-4x) - x E_Li2S^fu - (1-x) E_P2S5^fu ) / (7-4x)

    where ``E^fu`` are total energies per formula unit of the two end points,
    taken from the **same** (reference or model) energy set.  Both end points
    must therefore be present in the input mapping; a missing end point raises
    instead of using a hard-coded value.
    """
    for endpoint in (endpoint_low, endpoint_high):
        if endpoint not in e_total:
            msg = (
                f"formation energy needs the {endpoint!r} end point in the "
                "same energy set; refusing to substitute an external value"
            )
            raise ValueError(msg)
    if endpoint_low not in x or endpoint_high not in x:
        msg = "end points must carry their tie-line composition x"
        raise ValueError(msg)

    def per_formula_unit(key: str) -> float:
        return float(e_total[key]) / (float(natoms[key]) / (7.0 - 4.0 * float(x[key])))

    # ``endpoint_low`` is the x=0 end point (P2S5), ``endpoint_high`` the x=1
    # end point (Li2S); the formula uses x * E_Li2S + (1-x) * E_P2S5.
    low = per_formula_unit(endpoint_low)
    high = per_formula_unit(endpoint_high)
    out: dict[str, float] = {}
    for key, energy in e_total.items():
        if key not in x or key not in natoms:
            continue
        xv = float(x[key])
        n_fu = float(natoms[key]) / (7.0 - 4.0 * xv)
        energy_fu = float(energy) / n_fu
        out[key] = float((energy_fu - xv * high - (1.0 - xv) * low) / (7.0 - 4.0 * xv))
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 1-based."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def rank_agreement(
    reference: Sequence[float], model: Sequence[float]
) -> dict[str, Any]:
    """Spearman rho and the maximum rank shift between two orderings.

    Used for phase-stability rankings (e.g. LiPS-25 formation energies across
    the Li2S-P2S5 tie line).  Ties are handled with average ranks; a perfect
    ordering gives ``rho = 1`` and ``max_rank_shift = 0``.
    """
    ref = _as_float_array(reference)
    mdl = _as_float_array(model)
    if ref.shape != mdl.shape:
        msg = "reference and model rankings must have equal length"
        raise ValueError(msg)
    n = ref.size
    if n < 2:
        msg = "rank agreement needs at least two entries"
        raise ValueError(msg)
    ref_ranks = _rankdata(ref)
    mdl_ranks = _rankdata(mdl)
    ref_centered = ref_ranks - ref_ranks.mean()
    mdl_centered = mdl_ranks - mdl_ranks.mean()
    denominator = np.sqrt((ref_centered**2).sum() * (mdl_centered**2).sum())
    rho = float((ref_centered * mdl_centered).sum() / denominator)
    # Kendall tau-b on the rank order, also tie aware.
    concordant = discordant = ties_ref = ties_model = 0
    for i in range(n):
        for j in range(i + 1, n):
            dr = ref[i] - ref[j]
            dm = mdl[i] - mdl[j]
            if dr == 0 and dm == 0:
                ties_ref += 1
                ties_model += 1
            elif dr == 0:
                ties_ref += 1
            elif dm == 0:
                ties_model += 1
            elif (dr > 0) == (dm > 0):
                concordant += 1
            else:
                discordant += 1
    tau_denominator = np.sqrt(
        (concordant + discordant + ties_ref) * (concordant + discordant + ties_model)
    )
    tau = float((concordant - discordant) / tau_denominator) if tau_denominator else 0.0
    shift = np.abs(ref_ranks - mdl_ranks)
    return {
        "n": int(n),
        "spearman_rho": rho,
        "kendall_tau_b": tau,
        "max_rank_shift": float(shift.max()),
        "mean_rank_shift": float(shift.mean()),
    }


def convex_hull_keys(
    energies: Mapping[str, float], x: Mapping[str, float]
) -> list[str]:
    """Compositions on the lower convex hull of (x, formation energy).

    Returns the keys forming the hull in increasing ``x`` order, using the
    same monotone chain construction for the reference and the model.  A
    candidate composition is on the hull when the piecewise-linear
    interpolation between its hull neighbours does not lie below it.
    """
    points = sorted(
        ((float(x[k]), float(energies[k]), k) for k in energies if k in x),
        key=lambda item: (item[0], item[1]),
    )
    if not points:
        return []
    hull: list[tuple[float, float, str]] = []
    for point in points:
        while len(hull) >= 2:
            (x0, y0, _), (x1, y1, _) = hull[-2], hull[-1]
            # cross product of (p1-p0) and (p-p0); keep the lower hull
            cross = (x1 - x0) * (point[1] - y0) - (y1 - y0) * (point[0] - x0)
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(point)
    return [key for _, _, key in hull]


def hull_agreement(
    ref_energies: Mapping[str, float],
    model_energies: Mapping[str, float],
    x: Mapping[str, float],
) -> dict[str, Any]:
    """Compare which compositions are predicted to be thermodynamically stable."""
    ref_hull = convex_hull_keys(ref_energies, x)
    model_hull = convex_hull_keys(model_energies, x)
    shared = sorted(set(ref_energies) & set(model_energies) & set(x))
    return {
        "reference_hull": ref_hull,
        "model_hull": model_hull,
        "agreement": sorted(set(ref_hull) & set(model_hull)),
        "only_reference": sorted(set(ref_hull) - set(model_hull)),
        "only_model": sorted(set(model_hull) - set(ref_hull)),
        "n_compared": len(shared),
    }


def lattice_stats(cell_ref: np.ndarray, cell_model: np.ndarray) -> dict[str, Any]:
    """Volume / lattice-parameter errors between two 3x3 cells (Angstrom)."""
    ref = np.asarray(cell_ref, dtype=float)
    model = np.asarray(cell_model, dtype=float)
    if ref.shape != (3, 3) or model.shape != (3, 3):
        msg = "cells must be 3x3 matrices in Angstrom"
        raise ValueError(msg)
    volume_ref = float(abs(np.linalg.det(ref)))
    volume_model = float(abs(np.linalg.det(model)))
    lengths_ref = np.linalg.norm(ref, axis=1)
    lengths_model = np.linalg.norm(model, axis=1)
    return {
        "volume_ref_A3": volume_ref,
        "volume_model_A3": volume_model,
        "volume_rel_error": float((volume_model - volume_ref) / volume_ref),
        "lattice_length_mae_A": float(np.abs(lengths_model - lengths_ref).mean()),
        "lattice_lengths_ref_A": lengths_ref.tolist(),
        "lattice_lengths_model_A": lengths_model.tolist(),
    }
