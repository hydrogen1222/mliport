"""PBC-correct, directional, windowed mean-squared displacement analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.geometry import find_mic

from mliport.analysis.structure import cell_heights_A
from mliport.analysis.units import (
    diffusion_A2_fs_to_m2_s,
    diffusion_m2_s_to_cm2_s,
)
from mliport.analysis.drift import (
    drift_displacement,
    drift_semantics,
    reference_indices,
)
from mliport.analysis.validation import require_analysis

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from mliport.analysis.dataset import TrajectoryDataset

_VALID_AXES = {"x", "y", "z", "xy", "xz", "yz", "xyz"}
_AXIS_COLUMNS = {"x": 0, "y": 1, "z": 2}


def _normalize_axes(axes: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(axes, str):
        values = tuple(value.strip().lower() for value in axes.split(","))
    else:
        values = tuple(str(value).strip().lower() for value in axes)
    if not values or any(value not in _VALID_AXES for value in values):
        raise ValueError("axes must contain only x, y, z, xy, xz, yz, or xyz")
    if len(set(values)) != len(values):
        raise ValueError("axes contains duplicate selections")
    return values


def unwrap_positions(dataset: TrajectoryDataset) -> tuple[np.ndarray, dict[str, Any]]:
    """Return continuous Cartesian positions and an ambiguity diagnostic."""

    require_analysis(dataset, "msd")
    cell = dataset.cells[0]
    inverse = np.linalg.inv(cell)
    minimum_height = float(np.min(cell_heights_A(cell)))
    raw_steps = np.diff(dataset.positions, axis=0)
    if dataset.positions_convention == "wrapped":
        flat_steps = raw_steps.reshape(-1, 3)
        mic_steps, _ = find_mic(flat_steps, cell, pbc=dataset.pbc)
        cartesian_steps = np.asarray(mic_steps, dtype=float).reshape(raw_steps.shape)
        continuous = np.empty_like(dataset.positions, dtype=float)
        continuous[0] = dataset.positions[0]
        continuous[1:] = dataset.positions[0] + np.cumsum(cartesian_steps, axis=0)
        step_fractional = cartesian_steps @ inverse
        reconstruction = "consecutive ASE general minimum-image"
        exact_images_available = False
    elif dataset.positions_convention == "unwrapped":
        cartesian_steps = raw_steps
        step_fractional = cartesian_steps @ inverse
        continuous = dataset.positions.copy()
        reconstruction = "source unwrapped Cartesian coordinates"
        exact_images_available = True
    else:  # protected by validation, retained for a precise direct-call error
        raise ValueError("Position convention must be wrapped or unwrapped")
    cartesian_norms = np.linalg.norm(cartesian_steps, axis=-1)
    maximum_cartesian = float(np.max(cartesian_norms)) if cartesian_norms.size else 0.0
    maximum_fractional = (
        float(np.max(np.abs(step_fractional))) if step_fractional.size else 0.0
    )
    ratio = maximum_cartesian / (0.5 * minimum_height)
    if exact_images_available:
        level = "not_applicable_exact_unwrapped_source"
    elif ratio < 0.5:
        level = "comfortably_safe"
    elif ratio < 0.8:
        level = "warning"
    else:
        level = "strong_warning"
    diagnostics = {
        "positions_convention": dataset.positions_convention,
        "reconstruction": reconstruction,
        "exact_image_information_available": exact_images_available,
        "max_fractional_step_mic": maximum_fractional,
        "max_cartesian_step_mic_A": maximum_cartesian,
        "minimum_cell_height_A": minimum_height,
        "unwrap_safety_ratio": ratio,
        "unwrap_safety_level": level,
        "interpretation": (
            "For wrapped sources this ratio is a heuristic, not proof that hidden "
            "multiple cell crossings did not occur between saved frames."
        ),
    }
    return continuous, diagnostics


def displacement_trajectory(
    dataset: TrajectoryDataset,
    *,
    mobile_indices: Iterable[int],
    drift_reference: str = "none",
    drift_indices: Iterable[int] | None = None,
    drift_mode: str = "mass_weighted_com",
) -> dict[str, Any]:
    """Build raw/corrected mobile displacements with explicit drift semantics.

    The native MSD centre definition is the mass-weighted center of mass
    (``drift_mode="mass_weighted_com"``); it is recorded alongside the
    arithmetic-mean definition used by the kinisi transport backend so a
    cross-backend comparison can check that both agree (task book PR-F).
    """

    mobile = dataset.select(indices=mobile_indices)
    continuous, diagnostics = unwrap_positions(dataset)
    reference = reference_indices(
        dataset,
        mobile=mobile,
        drift_reference=drift_reference,
        drift_indices=drift_indices,
    )
    drift = drift_displacement(
        continuous,
        reference=reference,
        masses=dataset.masses,
        drift_mode=drift_mode,
    )
    raw_mobile_displacements = continuous[:, mobile] - continuous[0, mobile]
    corrected = raw_mobile_displacements - drift[:, None, :]
    semantics = drift_semantics(
        drift_mode=drift_mode,
        drift_reference=drift_reference,
        reference=reference,
        symbols=dataset.symbols,
    )
    return {
        "continuous_positions_A": continuous,
        "raw_mobile_displacements_A": raw_mobile_displacements,
        "mobile_displacements_A": corrected,
        "framework_drift_A": drift,
        "mobile_indices": mobile,
        # ``mode`` keeps its historical meaning (the reference selection)
        # for downstream consumers; the centre definition is ``drift_mode``.
        "drift_correction": {
            **semantics,
            "mode": str(drift_reference).lower(),
        },
        "drift_semantics": semantics,
        "unwrap_diagnostics": diagnostics,
    }


def direct_windowed_msd_components(displacements_A: np.ndarray) -> np.ndarray:
    """Reference O(T^2) windowed MSD for x/y/z independently."""

    positions = np.asarray(displacements_A, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("displacements_A must have shape (frames, atoms, 3)")
    result = np.empty((len(positions), 3), dtype=float)
    result[0] = 0.0
    for lag in range(1, len(positions)):
        delta = positions[lag:] - positions[:-lag]
        result[lag] = np.mean(delta**2, axis=(0, 1))
    return result


def fft_windowed_msd_components(displacements_A: np.ndarray) -> np.ndarray:
    """FFT-accelerated windowed MSD exactly matching the direct estimator."""

    positions = np.asarray(displacements_A, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("displacements_A must have shape (frames, atoms, 3)")
    n_frames = len(positions)
    transform = np.fft.rfft(positions, n=2 * n_frames, axis=0)
    autocorrelation = np.fft.irfft(
        transform * np.conjugate(transform), n=2 * n_frames, axis=0
    )[:n_frames]
    squared = positions**2
    prefix = np.concatenate(
        (np.zeros((1, positions.shape[1], 3)), np.cumsum(squared, axis=0)),
        axis=0,
    )
    result = np.empty((n_frames, 3), dtype=float)
    for lag in range(n_frames):
        origins = n_frames - lag
        later_sum = prefix[n_frames] - prefix[lag]
        earlier_sum = prefix[n_frames - lag]
        squared_displacement_sum = later_sum + earlier_sum - 2.0 * autocorrelation[lag]
        result[lag] = np.mean(squared_displacement_sum / origins, axis=0)
    result[0] = 0.0
    # Roundoff can create values around -1e-14 at lag zero/very short lags.
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(result)))) * 100
    result[np.abs(result) < tolerance] = 0.0
    return result


def _axis_msd(components: np.ndarray, axes: str) -> np.ndarray:
    columns = [_AXIS_COLUMNS[axis] for axis in axes]
    return np.sum(components[:, columns], axis=1)


ALPHA_ESTIMATOR_VERSION = "mliport msd local exponent/2"
#: Defaults for the local estimator.  These are configuration, not universal
#: scientific thresholds (review section 2.4); every value is recorded in the
#: result and therefore in the analysis cache fingerprint.
ALPHA_WINDOW_DECADES_DEFAULT = 0.25
ALPHA_MIN_POINTS_DEFAULT = 5
ALPHA_MIN_ORIGINS_DEFAULT = 8
ALPHA_CONSISTENCY_BAND_DEFAULT = 0.2
ALPHA_SENSITIVITY_WINDOWS_DECADES = (0.15, 0.25, 0.4)

REGIME_INSUFFICIENT = "insufficient_information"
REGIME_NOT_DIFFUSIVE = "not_diffusive"
REGIME_CONSISTENT = "consistent_with_diffusion"


def _log_log_alpha(lag_ps: np.ndarray, msd_A2: np.ndarray) -> np.ndarray:
    """Raw per-point log-log derivative, NaN outside continuous valid runs.

    The derivative never crosses a gap: zero/negative/non-finite MSD values
    split the series into independent continuous segments, each with its own
    gradient (review AL-04).  No epsilon is added to zero values.
    """
    values, _reasons = _raw_alpha_with_reasons(lag_ps, msd_A2)
    return values


def _continuous_segments(valid: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` index runs where ``valid`` is True."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, ok in enumerate(np.asarray(valid, dtype=bool)):
        if ok and start is None:
            start = index
        elif not ok and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(valid)))
    return segments


def _raw_alpha_with_reasons(
    lag_ps: np.ndarray,
    msd_A2: np.ndarray,
    *,
    max_lag_ps: float | None = None,
) -> tuple[np.ndarray, list[str | None]]:
    lag = np.asarray(lag_ps, dtype=float)
    msd = np.asarray(msd_A2, dtype=float)
    result = np.full(msd.shape, np.nan, dtype=float)
    reasons: list[str | None] = [
        "lag must be positive" if not (t > 0) else None for t in lag
    ]
    positive = (lag > 0) & np.isfinite(msd) & (msd > 0)
    if max_lag_ps is not None:
        beyond = lag > float(max_lag_ps)
        positive &= ~beyond
        for index in range(len(msd)):
            if beyond[index]:
                reasons[index] = f"lag exceeds alpha_max_lag_ps={float(max_lag_ps):g}"
    for index in range(len(msd)):
        if not np.isfinite(msd[index]):
            reasons[index] = "non-finite MSD"
        elif msd[index] <= 0:
            reasons[index] = "zero or negative MSD (no epsilon added)"
    for start, stop in _continuous_segments(positive):
        if stop - start < 3:
            for index in range(start, stop):
                reasons[index] = (
                    "continuous positive segment has fewer than three points"
                )
            continue
        log_time = np.log(lag[start:stop])
        log_msd = np.log(msd[start:stop])
        result[start:stop] = np.gradient(log_msd, log_time)
        for index in range(start, stop):
            reasons[index] = None
    return result, reasons


def _local_quadrature_weights(log_time: np.ndarray) -> np.ndarray:
    """Local ``Δlog(t)`` quadrature weights for the current window (PR-H A01).

    The weight of a point depends only on the local log-time spacing, not on
    how many points the whole segment happens to place in its decade.  A
    linear lag grid therefore contributes an integral over log time instead of
    letting the densely packed long-lag tail dominate the regression by
    population.
    """
    x = np.asarray(log_time, dtype=float)
    if x.size < 2:
        return np.ones_like(x)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    widths = np.diff(edges)
    return np.maximum(widths, np.finfo(float).tiny)


def _local_alpha_arrays(
    lag_ps: np.ndarray,
    msd_A2: np.ndarray,
    *,
    window_decades: float,
    min_points: int,
    max_lag_ps: float | None = None,
) -> dict[str, Any]:
    """Weighted local log-log regression with an explicit window width."""
    lag = np.asarray(lag_ps, dtype=float)
    msd = np.asarray(msd_A2, dtype=float)
    n_points = len(msd)
    alpha = np.full(n_points, np.nan, dtype=float)
    reasons: list[str | None] = [None] * n_points
    support = np.zeros(n_points, dtype=int)
    window_min = np.full(n_points, np.nan, dtype=float)
    window_max = np.full(n_points, np.nan, dtype=float)
    positive = (lag > 0) & np.isfinite(msd) & (msd > 0)
    if max_lag_ps is not None:
        beyond = lag > float(max_lag_ps)
        positive &= ~beyond
        for index in range(n_points):
            if beyond[index]:
                reasons[index] = f"lag exceeds alpha_max_lag_ps={float(max_lag_ps):g}"
    log_time = np.full(n_points, np.nan, dtype=float)
    log_msd = np.full(n_points, np.nan, dtype=float)
    log_time[positive] = np.log(lag[positive])
    log_msd[positive] = np.log(msd[positive])
    half_width = window_decades * np.log(10.0) / 2.0
    for start, stop in _continuous_segments(positive):
        segment = np.arange(start, stop)
        if len(segment) < min_points:
            for index in segment:
                reasons[index] = (
                    f"continuous segment has {len(segment)} points "
                    f"(< alpha_min_points={min_points})"
                )
            continue
        for index in segment:
            offsets = np.abs(log_time[segment] - log_time[index])
            selected = offsets <= half_width
            count = int(np.count_nonzero(selected))
            support[index] = count
            if count < min_points:
                reasons[index] = (
                    f"window contains {count} points "
                    f"(< alpha_min_points={min_points})"
                )
                continue
            x = log_time[segment][selected]
            y = log_msd[segment][selected]
            # Local quadrature weights: depends only on this window's spacing.
            weight = _local_quadrature_weights(x)
            if np.ptp(x) <= 0:
                reasons[index] = "window contains no log-time spread"
                continue
            design = np.column_stack([np.ones_like(x), x])
            root_weight = np.sqrt(weight)
            coefficients, *_ = np.linalg.lstsq(
                design * root_weight[:, None], y * root_weight, rcond=None
            )
            alpha[index] = float(coefficients[1])
            window_min[index] = float(np.exp(x.min()))
            window_max[index] = float(np.exp(x.max()))
    for index in range(n_points):
        if not np.isfinite(alpha[index]) and reasons[index] is None:
            reasons[index] = "point lies outside any continuous positive MSD segment"
    return {
        "alpha": alpha,
        "invalid_reason": reasons,
        "window_support_points": support,
        "window_lag_min_ps": window_min,
        "window_lag_max_ps": window_max,
    }


def alpha_local_analysis(
    lag_ps: np.ndarray,
    msd_A2: np.ndarray,
    *,
    time_origin_counts: np.ndarray | None = None,
    window_decades: float = ALPHA_WINDOW_DECADES_DEFAULT,
    min_points: int = ALPHA_MIN_POINTS_DEFAULT,
    min_origins: int = ALPHA_MIN_ORIGINS_DEFAULT,
    consistency_band: float = ALPHA_CONSISTENCY_BAND_DEFAULT,
    max_lag_ps: float | None = None,
    sensitivity_windows_decades: Iterable[float] = ALPHA_SENSITIVITY_WINDOWS_DECADES,
) -> dict[str, Any]:
    """Compute raw + local MSD log-log exponents with explicit validity.

    The raw pointwise derivative is kept for compatibility (and for consumers
    who want the un-smoothed signal) but is never the basis of a regime claim:
    the local fixed-window weighted regression carries ``alpha_local`` and the
    regime status.  No confidence interval is produced at all -- without
    independent replicates or validated time blocks a narrow interval would be
    fabricated (review section 2.4), so the fields are explicitly ``None``.
    """
    if window_decades <= 0:
        raise ValueError("alpha_window_decades must be positive")
    if min_points < 3:
        raise ValueError("alpha_min_points must be >= 3")
    if min_origins < 1:
        raise ValueError("alpha_min_origins must be >= 1")
    if consistency_band <= 0:
        raise ValueError("alpha_consistency_band must be positive")
    if max_lag_ps is not None and (not np.isfinite(max_lag_ps) or max_lag_ps <= 0):
        raise ValueError("alpha_max_lag_ps must be finite and positive when given")
    lag = np.asarray(lag_ps, dtype=float)
    msd = np.asarray(msd_A2, dtype=float)
    if lag.shape != msd.shape:
        raise ValueError("lag_ps and msd_A2 must have the same shape")
    if time_origin_counts is None:
        origins = np.full(lag.shape, np.nan, dtype=float)
    else:
        origins = np.asarray(time_origin_counts, dtype=float)
        if origins.shape != lag.shape:
            raise ValueError("time_origin_counts must match lag_ps shape")

    raw, raw_reasons = _raw_alpha_with_reasons(lag, msd, max_lag_ps=max_lag_ps)
    local = _local_alpha_arrays(
        lag,
        msd,
        window_decades=window_decades,
        min_points=min_points,
        max_lag_ps=max_lag_ps,
    )
    windows = sorted(
        {
            float(window_decades),
            *(float(window) for window in sensitivity_windows_decades),
        }
    )
    sensitivity: dict[float, np.ndarray] = {}
    for window in windows:
        sensitivity[window] = _local_alpha_arrays(
            lag,
            msd,
            window_decades=window,
            min_points=min_points,
            max_lag_ps=max_lag_ps,
        )["alpha"]
    stack = np.vstack([sensitivity[window] for window in windows])
    finite_stack = np.isfinite(stack)
    all_valid = np.all(finite_stack, axis=0)
    with np.errstate(invalid="ignore"):
        maxs = np.max(np.where(finite_stack, stack, -np.inf), axis=0)
        mins = np.min(np.where(finite_stack, stack, np.inf), axis=0)
    spread = np.where(all_valid, maxs - mins, np.nan)

    alpha_valid = np.isfinite(local["alpha"])
    regimes: list[str] = []
    reasons = list(local["invalid_reason"])
    for index in range(len(msd)):
        if (
            alpha_valid[index]
            and np.isfinite(origins[index])
            and origins[index] < min_origins
        ):
            # A window dominated by too few time origins carries insufficient
            # information even when the regression itself converged; time
            # origins are also not independent samples (review section 2.4).
            alpha_valid[index] = False
            reasons[index] = (
                f"only {int(origins[index])} time origins "
                f"(< alpha_min_origins={min_origins})"
            )
        if not alpha_valid[index]:
            regimes.append(REGIME_INSUFFICIENT)
            continue
        deviation = abs(float(local["alpha"][index]) - 1.0)
        unstable = bool(np.isfinite(spread[index]) and spread[index] > consistency_band)
        if deviation > consistency_band or unstable:
            regimes.append(REGIME_NOT_DIFFUSIVE)
        else:
            regimes.append(REGIME_CONSISTENT)

    return {
        "estimator_version": ALPHA_ESTIMATOR_VERSION,
        "parameters": {
            "window_decades": float(window_decades),
            "window_width_semantics": "full width in base-10 decades",
            "min_points": int(min_points),
            "min_origins": int(min_origins),
            "consistency_band": float(consistency_band),
            "weighting": ("local Δlog(t) quadrature weights inside the current window"),
            "sensitivity_windows_decades": windows,
            "max_lag_ps": None if max_lag_ps is None else float(max_lag_ps),
            "uncertainty_model": (
                "not_estimable: no independent replicates or validated time "
                "blocks were available, so no confidence interval is reported"
            ),
            "uncertainty_status": "not_estimable",
        },
        "alpha_raw": raw,
        "alpha_raw_valid": np.isfinite(raw),
        "alpha_raw_invalid_reason": raw_reasons,
        "alpha_local": local["alpha"],
        "alpha_valid": alpha_valid,
        "invalid_reason": reasons,
        "regime_status": regimes,
        "alpha_ci_low": None,
        "alpha_ci_high": None,
        "uncertainty_status": "not_estimable",
        "log_window_width_decades": float(window_decades),
        "window_lag_min_ps": local["window_lag_min_ps"],
        "window_lag_max_ps": local["window_lag_max_ps"],
        "window_support_points": local["window_support_points"],
        "alpha_window_range": spread,
        "time_origin_counts": origins,
        "time_origin_counts_are_not_independent_samples": True,
        "time_origin_counts_available": bool(np.any(np.isfinite(origins))),
        "sensitivity": {
            str(window): {
                "finite_fraction": float(np.mean(np.isfinite(values))),
                "finite_points": int(np.count_nonzero(np.isfinite(values))),
                "mean": (
                    float(np.mean(values[np.isfinite(values)]))
                    if np.any(np.isfinite(values))
                    else None
                ),
                "min": (
                    float(np.min(values[np.isfinite(values)]))
                    if np.any(np.isfinite(values))
                    else None
                ),
                "max": (
                    float(np.max(values[np.isfinite(values)]))
                    if np.any(np.isfinite(values))
                    else None
                ),
            }
            for window, values in sensitivity.items()
        },
    }


def _alpha_regime_summary(estimate: dict[str, Any]) -> dict[str, Any]:
    """Aggregate diagnostics that must accompany every regime statement."""
    regimes = list(estimate["regime_status"])
    total = max(1, len(regimes))
    counts = {
        regime: regimes.count(regime)
        for regime in (REGIME_INSUFFICIENT, REGIME_NOT_DIFFUSIVE, REGIME_CONSISTENT)
    }
    local = np.asarray(estimate["alpha_local"], dtype=float)
    finite_mask = np.isfinite(local)
    valid_mask = finite_mask & np.asarray(estimate["alpha_valid"], dtype=bool)
    # Every summary statistic is computed on finite & alpha_valid only; a
    # finite number that failed the validity contract must never enter a
    # summary or trend (task book PR-H A02).
    valid = local[valid_mask]
    origins = np.asarray(estimate["time_origin_counts"], dtype=float)
    origins_available = bool(estimate.get("time_origin_counts_available", False))
    if origins_available:
        known_origins = origins[np.isfinite(origins)]
        min_origins = float(estimate["parameters"]["min_origins"])
        fraction_with_few_origins = (
            float(np.mean(known_origins < min_origins)) if known_origins.size else None
        )
        origins_support_status = "known"
        origins_known_fraction = float(np.mean(np.isfinite(origins)))
    else:
        # NaN < threshold is False, so a naive computation would report 0%.
        fraction_with_few_origins = None
        origins_support_status = "unknown"
        origins_known_fraction = 0.0
    log_valid = valid_mask & (np.asarray(estimate["window_lag_min_ps"]) > 0)
    trend = None
    if np.count_nonzero(log_valid) >= 3:
        x = np.log(
            np.maximum(np.asarray(estimate["window_lag_max_ps"])[log_valid], 1e-300)
        )
        y = local[log_valid]
        if np.ptp(x) > 0:
            trend = float(np.polyfit(x, y, 1)[0])
    n_total = int(local.size)
    n_finite = int(np.count_nonzero(finite_mask))
    n_valid = int(np.count_nonzero(valid_mask))
    return {
        "estimator_version": estimate["estimator_version"],
        "parameters": estimate["parameters"],
        "regime_fractions": {regime: counts[regime] / total for regime in counts},
        "n_total": n_total,
        "n_finite": n_finite,
        "n_valid": n_valid,
        "valid_fraction": float(n_valid / n_total) if n_total else 0.0,
        "valid_points": n_valid,
        "alpha_local_mean": float(np.mean(valid)) if valid.size else None,
        "alpha_local_std": (float(np.std(valid, ddof=1)) if valid.size > 1 else None),
        "alpha_local_median": (float(np.median(valid)) if valid.size else None),
        "alpha_local_min": float(np.min(valid)) if valid.size else None,
        "alpha_local_max": float(np.max(valid)) if valid.size else None,
        "alpha_local_trend_per_log_time": trend,
        "alpha_window_range_max": (
            float(
                np.nanmax(
                    np.where(
                        valid_mask,
                        np.asarray(estimate["alpha_window_range"], dtype=float),
                        np.nan,
                    )
                )
            )
            if np.any(
                valid_mask
                & np.isfinite(np.asarray(estimate["alpha_window_range"], dtype=float))
            )
            else None
        ),
        "fraction_with_few_origins": fraction_with_few_origins,
        "origins_support_status": origins_support_status,
        "origins_known_fraction": origins_known_fraction,
        "time_origin_counts_are_not_independent_samples": True,
        "uncertainty_model": estimate["parameters"]["uncertainty_model"],
        "uncertainty_status": estimate["parameters"].get(
            "uncertainty_status", "not_estimable"
        ),
        "alpha_ci_low": None,
        "alpha_ci_high": None,
    }


def diagnostic_linear_diffusion_fit(
    lag_ps: np.ndarray,
    msd_A2: np.ndarray,
    *,
    axes: str,
    fit_start_ps: float,
    fit_stop_ps: float,
    time_origin_counts: np.ndarray | None = None,
    alpha_window_decades: float = ALPHA_WINDOW_DECADES_DEFAULT,
    alpha_min_points: int = ALPHA_MIN_POINTS_DEFAULT,
    alpha_min_origins: int = ALPHA_MIN_ORIGINS_DEFAULT,
    alpha_consistency_band: float = ALPHA_CONSISTENCY_BAND_DEFAULT,
    max_lag_ps: float | None = None,
) -> dict[str, Any]:
    """Explicit-range OLS diagnostic; not a publication uncertainty model.

    The linear OLS slope is retained for continuity, but the regime verdict
    now comes from the local log-log estimator: a constant MSD must never be
    reported as a credible diffusion coefficient just because ``R^2 == 1``
    (review AL-03).  ``insufficient_information``, ``not_diffusive`` and
    ``consistent_with_diffusion`` are separate states with the supporting
    valid fraction, alpha range, trend and window sensitivity attached.
    """

    if axes not in _VALID_AXES:
        raise ValueError("Invalid diffusion axes")
    if fit_start_ps < 0 or fit_stop_ps <= fit_start_ps:
        raise ValueError("fit_stop_ps must be greater than fit_start_ps >= 0")
    maximum_lag_ps = float(np.max(lag_ps))
    tolerance = max(1.0e-12, abs(maximum_lag_ps) * 1.0e-12)
    if fit_stop_ps > maximum_lag_ps + tolerance:
        raise ValueError(
            f"fit_stop_ps {fit_stop_ps:g} exceeds the maximum available lag "
            f"time {maximum_lag_ps:g} ps"
        )
    mask = (lag_ps >= fit_start_ps) & (lag_ps <= fit_stop_ps) & np.isfinite(msd_A2)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Diagnostic diffusion fit requires at least three lag points")
    x = lag_ps[mask]
    y = msd_A2[mask]
    slope_A2_ps, intercept_A2 = np.polyfit(x, y, 1)
    predicted = slope_A2_ps * x + intercept_A2
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    dimensions = len(axes)
    diffusion_A2_ps = slope_A2_ps / (2.0 * dimensions)
    diffusion_m2_s = diffusion_A2_fs_to_m2_s(diffusion_A2_ps / 1000.0)
    alpha = _log_log_alpha(lag_ps, msd_A2)
    finite_alpha = alpha[mask & np.isfinite(alpha)]

    estimate = alpha_local_analysis(
        lag_ps,
        msd_A2,
        time_origin_counts=time_origin_counts,
        window_decades=alpha_window_decades,
        min_points=alpha_min_points,
        min_origins=alpha_min_origins,
        consistency_band=alpha_consistency_band,
    )
    summary = _alpha_regime_summary(estimate)
    regimes = np.asarray(estimate["regime_status"], dtype=object)[mask]
    n_fit = int(np.count_nonzero(mask))
    consistent_fraction = float(np.mean(regimes == REGIME_CONSISTENT)) if n_fit else 0.0
    insufficient_fraction = (
        float(np.mean(regimes == REGIME_INSUFFICIENT)) if n_fit else 1.0
    )
    if insufficient_fraction > 0.5:
        regime_status = REGIME_INSUFFICIENT
    elif consistent_fraction > 0.5:
        regime_status = REGIME_CONSISTENT
    else:
        regime_status = REGIME_NOT_DIFFUSIVE

    constant_msd = (
        total <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(y)))) ** 2
    )
    fit_informative = bool(n_fit >= alpha_min_points and not constant_msd)
    if constant_msd:
        interpretation = (
            "MSD is constant within the fit window: the R^2 value is not "
            "defined as evidence and the OLS slope is zero by construction"
        )
    elif not fit_informative:
        interpretation = (
            "too few fit points for a meaningful linear diagnostic; R^2 is "
            "descriptive only"
        )
    else:
        interpretation = (
            "descriptive OLS fit quality only; regime_status comes from the "
            "local log-log estimator"
        )
    fit_alpha = np.asarray(estimate["alpha_local"], dtype=float)[mask]
    fit_alpha_finite = fit_alpha[np.isfinite(fit_alpha)]
    return {
        "estimator": "diagnostic_linear_diffusion_fit",
        "estimator_version": ALPHA_ESTIMATOR_VERSION,
        "publication_grade": False,
        "axes": axes,
        "dimensions": dimensions,
        "fit_start_ps": float(fit_start_ps),
        "fit_stop_ps": float(fit_stop_ps),
        "actual_fit_start_ps": float(x[0]),
        "actual_fit_stop_ps": float(x[-1]),
        "fit_points": n_fit,
        "slope_A2_ps": float(slope_A2_ps),
        "intercept_A2": float(intercept_A2),
        "r_squared": 1.0 - residual / total if total > 0 else None,
        "r_squared_interpretation": interpretation,
        "fit_informative": fit_informative,
        "D_diagnostic_m2_s": diffusion_m2_s,
        "D_diagnostic_cm2_s": diffusion_m2_s_to_cm2_s(diffusion_m2_s),
        "self_diffusion_coefficient_m2_s": diffusion_m2_s,
        "self_diffusion_coefficient_cm2_s": diffusion_m2_s_to_cm2_s(diffusion_m2_s),
        "mean_log_log_alpha_in_fit": (
            float(np.mean(finite_alpha)) if len(finite_alpha) else None
        ),
        "alpha_local_mean_in_fit": (
            float(np.mean(fit_alpha_finite)) if fit_alpha_finite.size else None
        ),
        "alpha_local_valid_fraction_in_fit": (
            float(np.mean(np.asarray(estimate["alpha_valid"], dtype=bool)[mask]))
            if n_fit
            else 0.0
        ),
        "alpha_local_finite_fraction_in_fit": (
            float(np.mean(np.isfinite(fit_alpha))) if n_fit else 0.0
        ),
        "regime_status": regime_status,
        "regime_fractions_in_fit": {
            REGIME_INSUFFICIENT: insufficient_fraction,
            REGIME_NOT_DIFFUSIVE: (
                float(np.mean(regimes == REGIME_NOT_DIFFUSIVE)) if n_fit else 0.0
            ),
            REGIME_CONSISTENT: consistent_fraction,
        },
        "alpha_diagnostics": summary,
        "diffusive_regime_warning": regime_status != REGIME_CONSISTENT,
    }


def calculate_msd(
    dataset: TrajectoryDataset,
    *,
    mobile_species: str | None = None,
    mobile_indices: Iterable[int] | None = None,
    axes: str | Iterable[str] = "xyz",
    drift_reference: str = "none",
    drift_indices: Iterable[int] | None = None,
    drift_mode: str = "mass_weighted_com",
    method: str = "fft",
    include_equilibration: bool = False,
    start: int | None = None,
    stop: int | None = None,
    fit_start_ps: float | None = None,
    fit_stop_ps: float | None = None,
    alpha_window_decades: float = ALPHA_WINDOW_DECADES_DEFAULT,
    alpha_min_points: int = ALPHA_MIN_POINTS_DEFAULT,
    alpha_min_origins: int = ALPHA_MIN_ORIGINS_DEFAULT,
    alpha_consistency_band: float = ALPHA_CONSISTENCY_BAND_DEFAULT,
    alpha_max_lag_ps: float | None = None,
) -> dict[str, Any]:
    """Calculate directional MSD, alpha estimates and an optional diagnostic fit.

    ``log_log_alpha_by_axes`` keeps its historical meaning (the raw per-point
    log-log derivative, now computed per continuous valid segment).  The
    ``alpha_estimates_by_axes`` / ``alpha_diagnostics_by_axes`` fields carry
    the explicitly named local estimator, its validity, window support and
    regime status; the new parameters are part of the analysis request and
    therefore of the cache fingerprint.
    """

    selected_axes = _normalize_axes(axes)
    if (fit_start_ps is None) != (fit_stop_ps is None):
        raise ValueError(
            "Both fit_start_ps and fit_stop_ps are required for a diagnostic fit"
        )
    view = dataset.analysis_view(
        include_equilibration=include_equilibration, start=start, stop=stop
    )
    mobile = view.select(mobile_species, indices=mobile_indices)
    prepared = displacement_trajectory(
        view,
        mobile_indices=mobile,
        drift_reference=drift_reference,
        drift_indices=drift_indices,
        drift_mode=drift_mode,
    )
    if method == "fft":
        components = fft_windowed_msd_components(prepared["mobile_displacements_A"])
    elif method == "direct":
        components = direct_windowed_msd_components(prepared["mobile_displacements_A"])
    else:
        raise ValueError("MSD method must be fft or direct")
    lag_ps = np.arange(view.nframes, dtype=float) * view.frame_interval_fs / 1000.0
    values = {axis: _axis_msd(components, axis) for axis in selected_axes}
    time_origin_counts = view.nframes - np.arange(view.nframes)
    alpha = {axis: _log_log_alpha(lag_ps, values[axis]) for axis in selected_axes}
    alpha_estimates = {
        axis: alpha_local_analysis(
            lag_ps,
            values[axis],
            time_origin_counts=time_origin_counts,
            window_decades=alpha_window_decades,
            min_points=alpha_min_points,
            min_origins=alpha_min_origins,
            consistency_band=alpha_consistency_band,
            max_lag_ps=alpha_max_lag_ps,
        )
        for axis in selected_axes
    }
    alpha_diagnostics = {
        axis: _alpha_regime_summary(alpha_estimates[axis]) for axis in selected_axes
    }
    if fit_start_ps is None:
        fit_window = None
        fits: dict[str, dict[str, Any]] = {}
    else:
        fit_window = {"start": float(fit_start_ps), "stop": float(fit_stop_ps)}
        fits = {
            axis: diagnostic_linear_diffusion_fit(
                lag_ps,
                values[axis],
                axes=axis,
                fit_start_ps=fit_start_ps,
                fit_stop_ps=fit_stop_ps,
                time_origin_counts=time_origin_counts,
                alpha_window_decades=alpha_window_decades,
                alpha_min_points=alpha_min_points,
                alpha_min_origins=alpha_min_origins,
                alpha_consistency_band=alpha_consistency_band,
                max_lag_ps=alpha_max_lag_ps,
            )
            for axis in selected_axes
        }
    return {
        "lag_time_ps": lag_ps,
        "time_origin_counts": time_origin_counts,
        "time_origin_counts_are_not_independent_samples": True,
        "msd_x_A2": components[:, 0],
        "msd_y_A2": components[:, 1],
        "msd_z_A2": components[:, 2],
        "msd_by_axes_A2": values,
        "log_log_alpha_by_axes": alpha,
        "alpha_estimates_by_axes": alpha_estimates,
        "alpha_diagnostics_by_axes": alpha_diagnostics,
        "alpha_estimator_version": ALPHA_ESTIMATOR_VERSION,
        "fit_window_ps": fit_window,
        "diagnostic_linear_diffusion_fits": fits,
        "method": f"{method}_windowed_msd",
        "mobile_species": mobile_species,
        "mobile_indices": mobile,
        "selected_axes": selected_axes,
        "analysis_phase": "all" if include_equilibration else "production",
        "drift_correction": prepared["drift_correction"],
        "drift_semantics": prepared["drift_semantics"],
        "framework_drift_A": prepared["framework_drift_A"],
        "unwrap_diagnostics": prepared["unwrap_diagnostics"],
    }
