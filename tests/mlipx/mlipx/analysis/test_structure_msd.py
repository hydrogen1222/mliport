from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlipx.analysis import TrajectoryDataset
from mlipx.analysis.msd import (
    REGIME_CONSISTENT,
    REGIME_INSUFFICIENT,
    REGIME_NOT_DIFFUSIVE,
    alpha_local_analysis,
    calculate_msd,
    diagnostic_linear_diffusion_fit,
    direct_windowed_msd_components,
    fft_windowed_msd_components,
    unwrap_positions,
)
from mlipx.analysis.structure import (
    _periodic_gaussian_smooth,
    density_map,
    radial_distribution,
)


def _dataset(positions: np.ndarray, cell: np.ndarray, symbols: str = "LiS"):
    frames = [
        Atoms(symbols, positions=frame, cell=cell, pbc=True) for frame in positions
    ]
    return TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(len(frames), dtype=float),
        positions_convention="wrapped",
    )


def test_pbc_crossing_unwrap_known_answer() -> None:
    positions = np.asarray(
        [
            [[9.5, 1, 1], [5, 5, 5]],
            [[0.2, 1, 1], [5, 5, 5]],
            [[0.9, 1, 1], [5, 5, 5]],
            [[1.6, 1, 1], [5, 5, 5]],
        ]
    )
    unwrapped, diagnostic = unwrap_positions(_dataset(positions, np.eye(3) * 10))
    np.testing.assert_allclose(unwrapped[:, 0, 0], [9.5, 10.2, 10.9, 11.6])
    assert diagnostic["unwrap_safety_ratio"] < 0.5


def test_triclinic_crossing_unwrap_known_answer() -> None:
    cell = np.asarray([[5.0, 0, 0], [1.2, 4.5, 0], [0.4, 0.7, 4.0]])
    fractional = np.asarray(
        [
            [[0.92, 0.2, 0.3], [0.4, 0.4, 0.4]],
            [[0.04, 0.2, 0.3], [0.4, 0.4, 0.4]],
            [[0.16, 0.2, 0.3], [0.4, 0.4, 0.4]],
            [[0.28, 0.2, 0.3], [0.4, 0.4, 0.4]],
        ]
    )
    dataset = _dataset(fractional @ cell, cell)
    unwrapped, _ = unwrap_positions(dataset)
    expected_fractional_x = [0.92, 1.04, 1.16, 1.28]
    np.testing.assert_allclose(
        unwrapped[:, 0] @ np.linalg.inv(cell),
        np.column_stack((expected_fractional_x, np.full(4, 0.2), np.full(4, 0.3))),
    )


def test_skew_cell_unwrap_uses_general_minimum_image_not_fractional_rounding() -> None:
    cell = np.asarray([[5.0, 0, 0], [4.6, 1.2, 0], [0.4, 0.7, 4.0]])
    start_fractional = np.asarray([0.6, 0.6, 0.1])
    raw_fractional_step = np.asarray(
        [-0.3814205852010938, -0.32241417053740273, 0.5027611849508486]
    )
    start = start_fractional @ cell
    second = (start_fractional + raw_fractional_step) @ cell
    positions = np.asarray([[start], [second], [second], [second]])
    frames = [Atoms("Li", positions=frame, cell=cell, pbc=True) for frame in positions]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=[0.0, 1.0, 2.0, 3.0],
        positions_convention="wrapped",
    )

    continuous, _ = unwrap_positions(dataset)
    expected_step = np.asarray(
        [1.0108963635028174, 0.4650358248207108, -1.9889552601966054]
    )
    naive_step = (raw_fractional_step - np.rint(raw_fractional_step)) @ cell
    np.testing.assert_allclose(continuous[1, 0] - continuous[0, 0], expected_step)
    assert not np.allclose(expected_step, naive_step)


def test_fft_and_direct_windowed_msd_match() -> None:
    rng = np.random.default_rng(7)
    walk = np.cumsum(rng.normal(size=(64, 5, 3)), axis=0)
    direct = direct_windowed_msd_components(walk)
    fft = fft_windowed_msd_components(walk)
    np.testing.assert_allclose(fft, direct, rtol=1e-11, atol=1e-11)


def test_linear_msd_fit_known_self_diffusion_coefficient() -> None:
    lag_time_ps = np.arange(5, dtype=float)
    msd_xyz_A2 = 6.0 * lag_time_ps
    fit = diagnostic_linear_diffusion_fit(
        lag_time_ps,
        msd_xyz_A2,
        axes="xyz",
        fit_start_ps=1.0,
        fit_stop_ps=3.0,
    )
    assert fit["actual_fit_start_ps"] == 1.0
    assert fit["actual_fit_stop_ps"] == 3.0
    assert fit["self_diffusion_coefficient_m2_s"] == pytest.approx(1.0e-8)
    assert fit["self_diffusion_coefficient_cm2_s"] == pytest.approx(1.0e-4)

    with pytest.raises(ValueError, match="exceeds the maximum available lag time"):
        diagnostic_linear_diffusion_fit(
            lag_time_ps,
            msd_xyz_A2,
            axes="xyz",
            fit_start_ps=1.0,
            fit_stop_ps=5.0,
        )


def test_directional_msd_identity() -> None:
    rng = np.random.default_rng(9)
    walk = np.cumsum(rng.normal(size=(128, 4, 3)), axis=0)
    frames = [
        Atoms("Li4", positions=frame, cell=[1000, 1000, 1000], pbc=True)
        for frame in walk
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(len(frames)),
        positions_convention="unwrapped",
    )
    result = calculate_msd(dataset, mobile_species="Li", axes="x,y,z,xy,xyz")
    np.testing.assert_allclose(
        result["msd_by_axes_A2"]["xy"],
        result["msd_by_axes_A2"]["x"] + result["msd_by_axes_A2"]["y"],
    )
    np.testing.assert_allclose(
        result["msd_by_axes_A2"]["xyz"],
        result["msd_x_A2"] + result["msd_y_A2"] + result["msd_z_A2"],
    )
    assert result["fit_window_ps"] is None
    assert result["diagnostic_linear_diffusion_fits"] == {}


def test_simple_cubic_coordination_is_six_and_has_no_self_peak() -> None:
    grid = np.asarray(
        [[x, y, z] for x in range(3) for y in range(3) for z in range(3)],
        dtype=float,
    )
    dataset = TrajectoryDataset.from_frames(
        [Atoms("Li27", positions=grid, cell=[3, 3, 3], pbc=True)],
        times_fs=[0],
        positions_convention="unwrapped",
    )
    result = radial_distribution(
        dataset,
        center_species="Li",
        neighbor_species="Li",
        r_max_A=1.4,
        bins=140,
        cn_cutoff_A=1.1,
    )
    assert result["coordination_number_at_cutoff"] == 6.0
    assert result["ordered_neighbor_counts"][0] == 0
    peak_r = result["r_A"][np.argmax(result["g_center_neighbor"])]
    assert abs(peak_r - 1.0) <= 0.01


def test_random_same_species_rdf_tends_to_one_at_large_radius() -> None:
    rng = np.random.default_rng(21)
    n_atoms = 240
    frames = [
        Atoms(
            symbols=["Li"] * n_atoms,
            positions=rng.uniform(0, 15, size=(n_atoms, 3)),
            cell=[15, 15, 15],
            pbc=True,
        )
        for _ in range(5)
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(len(frames), dtype=float),
        positions_convention="unwrapped",
    )
    result = radial_distribution(
        dataset,
        center_species="Li",
        neighbor_species="Li",
        r_max_A=6.5,
        bins=65,
    )
    large_r = (result["r_A"] >= 3.0) & (result["r_A"] <= 6.0)
    np.testing.assert_allclose(
        np.mean(result["g_center_neighbor"][large_r]),
        1.0,
        rtol=0.04,
    )


def test_nonmobile_drift_correction_removes_rigid_translation() -> None:
    positions = np.asarray(
        [[[1 + 0.2 * frame, 1, 1], [4 + 0.2 * frame, 4, 4]] for frame in range(8)]
    )
    dataset = TrajectoryDataset.from_frames(
        [
            Atoms("LiS", positions=frame, cell=[20, 20, 20], pbc=True)
            for frame in positions
        ],
        times_fs=np.arange(len(positions), dtype=float),
        positions_convention="unwrapped",
    )
    raw = calculate_msd(dataset, mobile_species="Li", drift_reference="none")
    corrected = calculate_msd(
        dataset,
        mobile_species="Li",
        drift_reference="nonmobile",
    )
    assert raw["msd_by_axes_A2"]["xyz"][-1] > 0
    np.testing.assert_allclose(corrected["msd_by_axes_A2"]["xyz"], 0.0, atol=1e-13)
    np.testing.assert_allclose(corrected["framework_drift_A"][:, 0], np.arange(8) * 0.2)


def test_density_map_normalizations_survive_periodic_smoothing() -> None:
    frames = [
        Atoms(
            "LiS",
            positions=[[0.1, 0.1, 0.1], [2, 2, 2]],
            cell=[4, 4, 4],
            pbc=True,
        )
        for _ in range(4)
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=[0, 1, 2, 3],
        positions_convention="unwrapped",
    )
    result = density_map(
        dataset, mobile_species="Li", spacing_A=0.5, smoothing_sigma_A=0.3
    )
    assert np.isclose(np.sum(result["occupancy_probability"]), 1.0)
    assert np.isclose(
        np.sum(result["number_density_A^-3"]) * result["voxel_volume_A3"], 1.0
    )


def test_triclinic_gaussian_smoothing_uses_row_cell_reciprocal_vectors() -> None:
    cell = np.asarray([[4.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    size = 16
    coordinate = np.arange(size, dtype=float)[:, None, None]
    mode = np.cos(2.0 * np.pi * coordinate / size)
    mode = np.broadcast_to(mode, (size, size, size))
    density = 2.0 + mode

    smoothed = _periodic_gaussian_smooth(density, cell, sigma_A=1.0)
    measured_amplitude = 2.0 * np.mean((smoothed - np.mean(smoothed)) * mode)
    expected_amplitude = 0.1454886634865615

    assert measured_amplitude == pytest.approx(expected_amplitude, rel=1.0e-12)
    assert np.sum(smoothed) == pytest.approx(np.sum(density), rel=1.0e-14)


# ---------------------------------------------------------------------------
# PR5 / section 2: MSD local exponent diagnostics
# ---------------------------------------------------------------------------


def test_local_alpha_recovers_power_laws_and_constant_msd() -> None:
    t = np.logspace(0.0, 3.0, 400)
    for power in (0.0, 0.5, 1.0, 2.0):
        msd = 2.0 * t**power if power > 0 else np.full_like(t, 4.0)
        estimate = alpha_local_analysis(
            t, msd, time_origin_counts=np.full_like(t, 1000.0)
        )
        finite = np.asarray(estimate["alpha_local"], dtype=float)
        interior = finite[10:-10]
        assert np.all(np.isfinite(interior)), (power, estimate["invalid_reason"][:5])
        np.testing.assert_allclose(interior, power, atol=1e-9)
        assert np.all(
            np.asarray(estimate["regime_status"], dtype=object)[10:-10]
            == (REGIME_CONSISTENT if power == 1.0 else REGIME_NOT_DIFFUSIVE)
            if power in (0.0, 0.5, 1.0, 2.0)
            else True
        )


def test_raw_alpha_matches_historical_gradient_and_does_not_cross_gaps() -> None:
    t = np.arange(1, 21, dtype=float)
    msd = t**1.5
    msd[12:] = t[12:] ** 0.5  # a different exponent after the gap
    msd[9:12] = 0.0  # a gap of invalid (zero) MSD
    estimate = alpha_local_analysis(t, msd, time_origin_counts=t[::-1])
    raw = np.asarray(estimate["alpha_raw"], dtype=float)
    # valid runs: 0..8 and 12..19 -> segments of 9 and 8 points
    for start, stop in ((0, 9), (12, 20)):
        expected = np.gradient(np.log(msd[start:stop]), np.log(t[start:stop]))
        np.testing.assert_allclose(raw[start:stop], expected, rtol=1e-12)
    assert np.all(np.isnan(raw[9:12]))
    # no derivative is computed across the gap: the two segments keep their
    # own exponents (1.5 before, 0.5 after) instead of being blended
    np.testing.assert_allclose(raw[:8], 1.5, atol=1e-9)
    np.testing.assert_allclose(raw[12:], 0.5, atol=1e-9)
    assert (
        "zero or negative MSD (no epsilon added)"
        in estimate["alpha_raw_invalid_reason"][9]
    )
    # local estimates inside the gap are invalid with a readable reason
    assert not np.any(estimate["alpha_valid"][9:12])
    gap_reasons = np.asarray(estimate["invalid_reason"], dtype=object)[9:12]
    assert all(
        "continu" in (reason or "") or "window contains" in (reason or "")
        for reason in gap_reasons
    ), gap_reasons


def test_local_alpha_is_stable_under_tiny_relative_noise() -> None:
    rng = np.random.default_rng(20260912)
    t = np.arange(1, 10001, dtype=float)
    msd = t * np.exp(rng.normal(0, 0.001, len(t)))
    raw = np.asarray(
        alpha_local_analysis(t, msd, time_origin_counts=t[::-1])["alpha_raw"],
        dtype=float,
    )
    estimate = alpha_local_analysis(t, msd, time_origin_counts=t[::-1])
    local = np.asarray(estimate["alpha_local"], dtype=float)
    # the review's synthetic stress test: raw is unusable at long lag ...
    assert np.std(raw[9000:-1]) > 1.0
    # ... while the local estimator stays close to the true exponent 1
    assert np.std(local[9000:]) < 0.05
    assert abs(np.nanmean(local) - 1.0) < 1e-3
    assert np.mean(estimate["alpha_valid"]) > 0.99
    # no fabricated confidence interval
    assert estimate["alpha_ci_low"] is None and estimate["alpha_ci_high"] is None
    assert "no independent replicates" in estimate["parameters"]["uncertainty_model"]


def test_mean_alpha_near_one_is_not_enough_for_diffusion() -> None:
    """A wild oscillation with mean alpha ~ 1 must not be 'consistent'."""
    amplitude, frequency = 1.0, 25.0
    t = np.logspace(0.0, 4.0, 4000)
    msd = t * np.exp(amplitude * np.sin(frequency * np.log(t)))
    estimate = alpha_local_analysis(t, msd, time_origin_counts=np.full_like(t, 1000.0))
    local = np.asarray(estimate["alpha_local"], dtype=float)
    finite = local[np.isfinite(local)]
    assert abs(np.mean(finite) - 1.0) < 0.2  # the old mean-based gate would pass
    counts = {
        regime: estimate["regime_status"].count(regime)
        for regime in (REGIME_INSUFFICIENT, REGIME_NOT_DIFFUSIVE, REGIME_CONSISTENT)
    }
    assert counts[REGIME_CONSISTENT] < 0.5 * len(estimate["regime_status"])

    fit = diagnostic_linear_diffusion_fit(
        t,
        msd,
        axes="xyz",
        fit_start_ps=float(t[0]),
        fit_stop_ps=float(t[-1]),
        time_origin_counts=np.full_like(t, 1000.0),
    )
    assert fit["regime_status"] != REGIME_CONSISTENT
    assert fit["diffusive_regime_warning"] is True


def test_zero_msd_is_insufficient_information_not_negative_evidence() -> None:
    """The reviewer's probe: no usable alpha must warn, not report R^2=1,D=0."""
    t = np.arange(1, 10001, dtype=float)
    fit = diagnostic_linear_diffusion_fit(
        t,
        np.zeros_like(t),
        axes="xyz",
        fit_start_ps=1,
        fit_stop_ps=10000,
    )
    assert fit["regime_status"] == REGIME_INSUFFICIENT
    assert fit["diffusive_regime_warning"] is True
    assert fit["r_squared"] is None
    assert fit["fit_informative"] is False
    assert fit["mean_log_log_alpha_in_fit"] is None
    assert fit["alpha_local_valid_fraction_in_fit"] == 0.0
    assert "R^2" in fit["r_squared_interpretation"]


def test_alpha_is_invariant_under_time_unit_scaling() -> None:
    t = np.logspace(0.0, 3.0, 500)
    msd = 3.0 * t**0.7
    base = alpha_local_analysis(t, msd, time_origin_counts=np.full_like(t, 500.0))
    scaled = alpha_local_analysis(
        1000.0 * t, msd, time_origin_counts=np.full_like(t, 500.0)
    )
    base_local = np.asarray(base["alpha_local"], dtype=float)
    scaled_local = np.asarray(scaled["alpha_local"], dtype=float)
    mask = np.isfinite(base_local) & np.isfinite(scaled_local)
    np.testing.assert_allclose(base_local[mask], scaled_local[mask], atol=1e-12)


def test_alpha_window_sensitivity_and_origin_support_are_reported() -> None:
    t = np.logspace(0.0, 2.0, 300)
    msd = 2.0 * t
    estimate = alpha_local_analysis(
        t,
        msd,
        time_origin_counts=np.linspace(300.0, 1.0, 300),
        window_decades=0.25,
    )
    sensitivity = estimate["sensitivity"]
    assert set(sensitivity) == {"0.15", "0.25", "0.4"}
    for window in ("0.15", "0.25", "0.4"):
        assert sensitivity[window]["valid_points"] > 0
        assert sensitivity[window]["mean"] == pytest.approx(1.0, abs=1e-9)
    # late-lag support decays; with min_origins=8 the tail is insufficient
    assert not np.all(estimate["alpha_valid"])
    assert REGIME_INSUFFICIENT in estimate["regime_status"]
    assert any(
        "time origins" in (reason or "") for reason in estimate["invalid_reason"]
    )
    assert estimate["time_origin_counts_are_not_independent_samples"] is True


def test_calculate_msd_exposes_alpha_estimates_and_version() -> None:
    frames = [
        Atoms(
            "LiS",
            positions=[[0.2 * index, 0.0, 0.0], [5.0, 5.0, 5.0]],
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )
        for index in range(60)
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(60, dtype=float) * 10.0,
        positions_convention="unwrapped",
        md_timestep_fs=10.0,
        frame_stride_steps=1,
    )
    result = calculate_msd(
        dataset,
        mobile_species="Li",
        axes="xyz",
        fit_start_ps=0.1,
        fit_stop_ps=0.5,
        alpha_window_decades=0.3,
        alpha_min_origins=4,
    )
    assert result["alpha_estimator_version"].startswith("mlipx msd local exponent")
    estimate = result["alpha_estimates_by_axes"]["xyz"]
    assert estimate["log_window_width_decades"] == pytest.approx(0.3)
    assert estimate["parameters"]["window_width_semantics"].startswith("full width")
    assert "alpha_local" in estimate and "regime_status" in estimate
    diagnostics = result["alpha_diagnostics_by_axes"]["xyz"]
    assert set(diagnostics["regime_fractions"]) == {
        REGIME_INSUFFICIENT,
        REGIME_NOT_DIFFUSIVE,
        REGIME_CONSISTENT,
    }
    fit = result["diagnostic_linear_diffusion_fits"]["xyz"]
    assert "regime_status" in fit and "alpha_diagnostics" in fit


def test_alpha_max_lag_masks_beyond_without_touching_msd() -> None:
    t = np.logspace(0.0, 3.0, 400)
    msd = 2.0 * t
    result = alpha_local_analysis(
        t,
        msd,
        time_origin_counts=np.full_like(t, 1000.0),
        max_lag_ps=100.0,
    )
    beyond = t > 100.0
    assert not np.any(result["alpha_valid"][beyond])
    assert all(
        "alpha_max_lag_ps" in (reason or "")
        for reason in np.asarray(result["invalid_reason"], dtype=object)[beyond]
    )
    assert result["parameters"]["max_lag_ps"] == pytest.approx(100.0)
    # the raw MSD itself is untouched by the estimator mask
    np.testing.assert_allclose(msd, 2.0 * t)
