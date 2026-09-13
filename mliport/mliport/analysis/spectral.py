"""Velocity autocorrelation and qualified VACF-derived spectra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mliport.analysis.validation import require_analysis

if TYPE_CHECKING:
    from typing import Any

    from mliport.analysis.dataset import TrajectoryDataset


def one_sided_cosine_taper(length: int) -> np.ndarray:
    """Raised-cosine taper with w(0)=1 and w(last)=0."""

    if length < 2:
        raise ValueError("A taper requires at least two points")
    index = np.arange(length, dtype=float)
    return 0.5 * (1.0 + np.cos(np.pi * index / (length - 1)))


def direct_vacf(velocities: np.ndarray) -> np.ndarray:
    """Reference O(T^2) time-origin averaged velocity autocorrelation."""

    velocity = np.asarray(velocities, dtype=float)
    if velocity.ndim != 3 or velocity.shape[-1] != 3:
        raise ValueError("velocities must have shape (frames, atoms, 3)")
    result = np.empty(len(velocity), dtype=float)
    for lag in range(len(velocity)):
        products = velocity[: len(velocity) - lag] * velocity[lag:]
        result[lag] = float(np.mean(np.sum(products, axis=2)))
    return result


def fft_vacf(velocities: np.ndarray) -> np.ndarray:
    """FFT autocorrelation exactly matching :func:`direct_vacf`."""

    velocity = np.asarray(velocities, dtype=float)
    if velocity.ndim != 3 or velocity.shape[-1] != 3:
        raise ValueError("velocities must have shape (frames, atoms, 3)")
    n_frames = len(velocity)
    transform = np.fft.rfft(velocity, n=2 * n_frames, axis=0)
    correlation = np.fft.irfft(
        transform * np.conjugate(transform), n=2 * n_frames, axis=0
    )[:n_frames]
    correlation /= (n_frames - np.arange(n_frames))[:, None, None]
    return np.mean(np.sum(correlation, axis=2), axis=1)


def calculate_vacf(
    dataset: TrajectoryDataset,
    *,
    species: str | None = None,
    method: str = "fft",
    include_equilibration: bool = False,
    start: int | None = None,
    stop: int | None = None,
) -> dict[str, Any]:
    """Calculate raw and normalized VACF from stored velocities only."""

    require_analysis(dataset, "vacf")
    view = dataset.analysis_view(
        include_equilibration=include_equilibration, start=start, stop=stop
    )
    selected = view.select(species)
    velocities = view.velocities[:, selected]
    if not np.all(np.isfinite(velocities)):
        raise ValueError("Stored velocities contain NaN or Inf")
    if method == "fft":
        vacf = fft_vacf(velocities)
    elif method == "direct":
        vacf = direct_vacf(velocities)
    else:
        raise ValueError("VACF method must be fft or direct")
    if not np.isfinite(vacf[0]) or np.isclose(vacf[0], 0.0):
        raise ValueError("VACF(0) is zero or non-finite and cannot be normalized")
    normalized = vacf / vacf[0]
    interval_fs = view.frame_interval_fs
    nyquist_THz = 500.0 / interval_fs
    return {
        "lag_time_fs": np.arange(view.nframes, dtype=float) * interval_fs,
        "vacf_raw_A2_fs2": vacf,
        "vacf_normalized": normalized,
        "species": species,
        "method": f"{method}_time_origin_averaged_vacf",
        "frame_interval_fs": interval_fs,
        "nyquist_THz": nyquist_THz,
        "nyquist_cm^-1": nyquist_THz * 33.35640951981521,
    }


def velocity_spectrum(
    vacf_result: dict[str, Any],
    *,
    taper: str = "one-sided-cosine",
    normalization: str = "normalized_area",
) -> dict[str, Any]:
    """Cosine/rFFT spectrum of a one-sided VACF without clipping negatives."""

    vacf = np.asarray(vacf_result["vacf_normalized"], dtype=float)
    interval_fs = float(vacf_result["frame_interval_fs"])
    if taper == "one-sided-cosine":
        window = one_sided_cosine_taper(len(vacf))
    elif taper == "none":
        window = np.ones(len(vacf), dtype=float)
    else:
        raise ValueError("taper must be one-sided-cosine or none")
    windowed = vacf * window
    raw = np.real(np.fft.rfft(windowed)) * interval_fs
    frequency_THz = np.fft.rfftfreq(len(vacf), d=interval_fs * 1.0e-15) / 1.0e12
    if normalization == "raw_spectrum":
        spectrum = raw.copy()
        normalization_note = "raw discrete rFFT real part multiplied by dt_fs"
    elif normalization == "normalized_area":
        area = float(np.trapezoid(raw, frequency_THz))
        if not np.isfinite(area) or np.isclose(area, 0.0):
            raise ValueError("Spectrum has zero/non-finite signed area")
        spectrum = raw / area
        normalization_note = "signed integral over frequency_THz equals one"
    else:
        raise ValueError("normalization must be raw_spectrum or normalized_area")
    return {
        "frequency_THz": frequency_THz,
        "frequency_cm^-1": frequency_THz * 33.35640951981521,
        "raw_spectrum": raw,
        "spectrum": spectrum,
        "window": window,
        "windowed_vacf": windowed,
        "taper": taper,
        "normalization": normalization,
        "normalization_note": normalization_note,
        "negative_fraction": float(np.mean(raw < 0)),
        "negative_min": float(np.min(raw)),
        "not_phonon_dos": True,
        "name": "VACF-derived velocity spectrum",
        "transform_convention": (
            "real part of rFFT of the one-sided normalized VACF after the "
            "declared taper; negative estimates are retained"
        ),
    }
