"""Explicit multi-temperature Arrhenius fitting and extrapolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mliport.analysis.units import EV_TO_J, BOLTZMANN_J_K

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any


def fit_arrhenius(
    temperatures_K: Iterable[float],
    diffusivities_m2_s: Iterable[float],
    *,
    diffusivity_std_m2_s: Iterable[float] | None = None,
    extrapolate_temperatures_K: Iterable[float] = (),
    source_run_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fit ln(D)=ln(D0)-Ea/(k_B T), optionally weighted by D uncertainty."""

    temperatures = np.asarray(list(temperatures_K), dtype=float)
    diffusivities = np.asarray(list(diffusivities_m2_s), dtype=float)
    if temperatures.ndim != 1 or diffusivities.shape != temperatures.shape:
        raise ValueError("temperatures and diffusivities must be equal-length vectors")
    if len(temperatures) < 2:
        raise ValueError("Arrhenius fitting requires at least two temperatures")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0):
        raise ValueError("All temperatures must be finite and positive")
    if not np.all(np.isfinite(diffusivities)) or np.any(diffusivities <= 0):
        raise ValueError("All diffusivities must be finite and positive")
    if len(np.unique(temperatures)) != len(temperatures):
        raise ValueError(
            "Each Arrhenius point must represent an independent temperature run; "
            "duplicate temperatures are ambiguous"
        )
    sources = None if source_run_ids is None else tuple(source_run_ids)
    if sources is not None and len(sources) != len(temperatures):
        raise ValueError("source_run_ids must match the number of data points")
    x = 1.0 / temperatures
    y = np.log(diffusivities)
    sigma_y: np.ndarray | None = None
    sigma_D: np.ndarray | None = None
    if diffusivity_std_m2_s is not None:
        sigma_D = np.asarray(list(diffusivity_std_m2_s), dtype=float)
        if sigma_D.shape != diffusivities.shape:
            raise ValueError("diffusivity_std_m2_s must match diffusivities")
        if not np.all(np.isfinite(sigma_D)) or np.any(sigma_D <= 0):
            raise ValueError("Diffusivity standard deviations must be positive")
        sigma_y = sigma_D / diffusivities
        coefficients, covariance = np.polyfit(x, y, 1, w=1.0 / sigma_y, cov="unscaled")
        uncertainty_method = "weighted linear fit; sigma_lnD = sigma_D / D"
    else:
        coefficients = np.polyfit(x, y, 1)
        if len(temperatures) > 2:
            coefficients, covariance = np.polyfit(x, y, 1, cov=True)
        else:
            covariance = np.full((2, 2), np.nan)
        uncertainty_method = (
            "unweighted OLS covariance"
            if len(temperatures) > 2
            else "unavailable for two exact points"
        )
    slope, intercept = coefficients
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    kB_eV_K = BOLTZMANN_J_K / EV_TO_J
    activation_energy_eV = -float(slope) * kB_eV_K
    D0_m2_s = float(np.exp(intercept))
    slope_std = (
        float(np.sqrt(covariance[0, 0])) if np.isfinite(covariance[0, 0]) else None
    )
    intercept_std = (
        float(np.sqrt(covariance[1, 1])) if np.isfinite(covariance[1, 1]) else None
    )
    warnings: list[str] = []
    if len(temperatures) == 2:
        warnings.append(
            "Only two temperatures were supplied. The line is mathematically "
            "determined but does not provide a meaningful goodness-of-fit test."
        )
    extrapolations: list[dict[str, Any]] = []
    lower, upper = float(np.min(temperatures)), float(np.max(temperatures))
    for target in extrapolate_temperatures_K:
        target = float(target)
        if not np.isfinite(target) or target <= 0:
            raise ValueError("Extrapolation temperatures must be positive")
        vector = np.asarray([1.0 / target, 1.0])
        log_D = float(vector @ coefficients)
        log_std = (
            float(np.sqrt(vector @ covariance @ vector))
            if np.all(np.isfinite(covariance))
            else None
        )
        extrapolated = target < lower or target > upper
        item = {
            "temperature_K": target,
            "diffusivity_m2_s": float(np.exp(log_D)),
            "log_diffusivity_std": log_std,
            "extrapolated": extrapolated,
            "source_temperature_range_K": [lower, upper],
        }
        extrapolations.append(item)
        if extrapolated:
            warnings.append(
                f"{target:g} K lies outside the simulated temperature range "
                f"[{lower:g}, {upper:g}] K."
            )
    return {
        "model": "D(T) = D0 exp[-Ea/(k_B T)]",
        "temperatures_K": temperatures,
        "diffusivities_m2_s": diffusivities,
        "diffusivity_std_m2_s": (None if sigma_D is None else sigma_D),
        "inverse_temperature_K^-1": x,
        "ln_diffusivity": y,
        "ln_diffusivity_fit": predicted,
        "activation_energy_eV": activation_energy_eV,
        "activation_energy_std_eV": (
            slope_std * kB_eV_K if slope_std is not None else None
        ),
        "preexponential_factor_m2_s": D0_m2_s,
        "ln_preexponential_std": intercept_std,
        "parameter_covariance": covariance,
        "r_squared": 1.0 - residual / total if total > 0 else 1.0,
        "uncertainty_method": uncertainty_method,
        "number_of_independent_temperature_runs": int(len(temperatures)),
        "source_run_ids": sources,
        "extrapolations": extrapolations,
        "warnings": warnings,
    }
