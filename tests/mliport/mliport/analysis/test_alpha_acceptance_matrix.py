"""PR-H acceptance matrix for the local alpha estimator (section 10).

Covers: local-window weighting (A01), validity-mask summaries (A02),
origin-support null semantics, the decade-boundary red test and the
required acceptance cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from mliport.analysis.msd import (
    REGIME_CONSISTENT,
    REGIME_INSUFFICIENT,
    REGIME_NOT_DIFFUSIVE,
    alpha_local_analysis,
)

_LOG_GRID = np.logspace(-1.0, 3.0, 400)


def _estimate(lag, msd, **kwargs):
    defaults = {"window_decades": 0.4, "min_points": 7, "min_origins": 1}
    defaults.update(kwargs)
    return alpha_local_analysis(lag, msd, **defaults)


def _valid_mean(estimate):
    local = np.asarray(estimate["alpha_local"], dtype=float)
    mask = np.isfinite(local) & np.asarray(estimate["alpha_valid"], dtype=bool)
    return float(np.mean(local[mask])) if np.any(mask) else None


def test_exact_power_laws_recover_exponent():
    for exponent in (0.0, 0.5, 1.0, 2.0):
        msd = 3.0 * _LOG_GRID**exponent if exponent else np.full_like(_LOG_GRID, 3.0)
        estimate = _estimate(_LOG_GRID, msd)
        mean = _valid_mean(estimate)
        assert mean == pytest.approx(exponent, abs=1e-6), exponent


def test_linear_msd_transitions_from_constant_to_alpha_one():
    """a+bt: alpha -> 0 while the intercept dominates, -> 1 once bt does."""
    estimate = _estimate(_LOG_GRID, 2.0 + 1.5 * _LOG_GRID)
    local = np.asarray(estimate["alpha_local"], dtype=float)
    valid = np.asarray(estimate["alpha_valid"], dtype=bool)
    assert np.mean(local[:100][valid[:100]]) < 0.5
    assert np.mean(local[-100:][valid[-100:]]) == pytest.approx(1.0, abs=0.01)


def test_all_zero_msd_is_insufficient_information():
    estimate = _estimate(_LOG_GRID, np.zeros_like(_LOG_GRID))
    assert not np.any(estimate["alpha_valid"])
    assert set(estimate["regime_status"]) == {REGIME_INSUFFICIENT}
    summary = _summary(estimate)
    assert summary["n_valid"] == 0
    assert summary["valid_fraction"] == 0.0
    assert summary["alpha_local_mean"] is None
    assert summary["alpha_local_trend_per_log_time"] is None


def test_nan_and_inf_points_are_masked_and_excluded_from_summary():
    msd = _LOG_GRID**1.0
    msd = msd.astype(float)
    msd[100] = np.nan
    msd[200] = np.inf
    estimate = _estimate(_LOG_GRID, msd)
    summary = __import__("mliport.analysis.msd", fromlist=["x"])._alpha_regime_summary(
        estimate
    )
    assert summary["n_total"] == len(msd)
    assert summary["n_finite"] <= summary["n_total"] - 2
    assert summary["n_valid"] <= summary["n_finite"]
    assert summary["valid_fraction"] == pytest.approx(
        summary["n_valid"] / summary["n_total"]
    )
    assert np.isnan(estimate["alpha_local"][100])
    assert not np.isfinite(estimate["alpha_local"][200])


def test_sparse_jump_is_not_silently_smoothed_into_diffusion():
    msd = _LOG_GRID**1.0
    jump_index = 250
    msd = msd.copy()
    msd[jump_index:] = 200.0  # discontinuous jump at one saved lag
    estimate = _estimate(_LOG_GRID, msd)
    local = np.asarray(estimate["alpha_local"], dtype=float)
    spikes = np.where(np.isfinite(local) & (local > 1.5))[0]
    assert spikes.size >= 1
    # the jump window is explicitly not diffusive, never a silent pass
    assert any(
        estimate["regime_status"][index] == REGIME_NOT_DIFFUSIVE for index in spikes
    )


def test_plateau_is_not_diffusive():
    msd = 1.0 - np.exp(-_LOG_GRID)
    estimate = _estimate(_LOG_GRID, msd)
    late = np.asarray(estimate["alpha_local"], dtype=float)[-40:]
    valid = np.asarray(estimate["alpha_valid"], dtype=bool)[-40:]
    assert np.any(valid)
    assert np.mean(late[valid]) < 0.5
    assert REGIME_CONSISTENT not in set(estimate["regime_status"][-40:])


def test_ballistic_to_diffusive_transition():
    ballistic = _LOG_GRID**2
    diffusive = 50.0 * _LOG_GRID
    msd = np.minimum(ballistic, diffusive)
    estimate = _estimate(_LOG_GRID, msd, window_decades=0.3)
    local = np.asarray(estimate["alpha_local"], dtype=float)
    valid = np.asarray(estimate["alpha_valid"], dtype=bool)
    early = local[:60][valid[:60]]
    late = local[-80:][valid[-80:]]
    assert np.any(early) and np.mean(early) > 1.6
    assert np.any(late) and np.mean(late) == pytest.approx(1.0, abs=0.2)


def test_curved_log_log_reports_a_trend():
    # msd = t * (1 + 0.5 log10 t): the local exponent decreases smoothly
    # from 1 + 0.5/ln(10) toward 1, so the recorded trend must be negative.
    curvature = 1.0 + 0.5 * np.log10(_LOG_GRID)
    estimate = _estimate(_LOG_GRID, curvature * _LOG_GRID)
    summary = _summary(estimate)
    assert summary["alpha_local_trend_per_log_time"] is not None
    assert summary["alpha_local_trend_per_log_time"] < -0.01


def test_decade_boundary_has_no_artificial_kink():
    """A linear lag grid must not produce a 10^k discontinuity (PR-H 10.2)."""
    lag = np.arange(1.0, 2001.0)
    msd = lag * (1.0 + 0.05 * np.sin(2.0 * np.pi * np.log10(lag)))
    estimate = alpha_local_analysis(
        lag, msd, window_decades=0.4, min_points=7, min_origins=1
    )
    local = np.asarray(estimate["alpha_local"], dtype=float)
    valid = np.asarray(estimate["alpha_valid"], dtype=bool)

    def at(center: float) -> float:
        index = int(np.argmin(np.abs(lag - center)))
        assert valid[index]
        return float(local[index])

    crossing = abs(at(1000.0) - 0.5 * (at(900.0) + at(1100.0)))
    indices = [
        i for i in range(1, len(lag) - 1) if valid[i - 1] and valid[i] and valid[i + 1]
    ]
    second = np.abs(
        local[indices]
        - 0.5 * (local[np.asarray(indices) - 1] + local[np.asarray(indices) + 1])
    )
    # the boundary second difference must not be an outlier of the sweep
    assert crossing <= max(2.0 * float(np.percentile(second, 95)), 0.01)


def test_time_scale_invariance():
    estimate_a = _estimate(_LOG_GRID, _LOG_GRID**1.3)
    estimate_b = _estimate(_LOG_GRID * 10.0, (_LOG_GRID * 10.0) ** 1.3)
    np.testing.assert_allclose(
        estimate_a["alpha_local"], estimate_b["alpha_local"], atol=1e-9, equal_nan=True
    )


def test_long_lag_support_collapse_marks_points_invalid():
    lag = np.logspace(-1.0, 1.0, 40)
    estimate = _estimate(lag, lag, window_decades=0.5)
    support = np.asarray(estimate["window_support_points"], dtype=int)
    valid = np.asarray(estimate["alpha_valid"], dtype=bool)
    assert support[-1] < support[len(support) // 2]
    assert not valid[-1]
    summary = _summary(estimate)
    assert summary["n_valid"] < summary["n_total"]


def test_invalid_finite_point_never_enters_a_summary():
    lag = np.logspace(-1.0, 3.0, 300)
    msd = lag * (1.0 + 0.5 * np.log10(lag))  # curved: alpha varies per point
    origins = np.full(300, 300.0)
    poisoned = 150
    origins[poisoned] = 1.0  # finite alpha, invalid support
    estimate = alpha_local_analysis(
        lag,
        msd,
        time_origin_counts=origins,
        window_decades=0.4,
        min_points=7,
        min_origins=8,
    )
    local = np.asarray(estimate["alpha_local"], dtype=float)
    valid = np.asarray(estimate["alpha_valid"], dtype=bool)
    assert np.isfinite(local[poisoned])
    assert not valid[poisoned]
    summary = _summary(estimate)
    valid_values = local[valid]
    assert summary["alpha_local_mean"] == pytest.approx(float(np.mean(valid_values)))
    assert summary["alpha_local_mean"] != pytest.approx(
        float(np.mean(local[np.isfinite(local)]))
    )
    assert summary["n_valid"] == int(np.count_nonzero(valid))
    assert summary["n_finite"] >= summary["n_valid"]


def test_missing_origin_counts_report_unknown_support_not_zero():
    estimate = _estimate(_LOG_GRID, _LOG_GRID)
    summary = _summary(estimate)
    assert estimate["time_origin_counts_available"] is False
    assert summary["fraction_with_few_origins"] is None
    assert summary["origins_support_status"] == "unknown"
    assert summary["valid_fraction"] > 0.0


def test_uncertainty_is_explicitly_not_estimable():
    estimate = _estimate(_LOG_GRID, _LOG_GRID)
    summary = _summary(estimate)
    assert estimate["alpha_ci_low"] is None
    assert estimate["alpha_ci_high"] is None
    assert estimate["uncertainty_status"] == "not_estimable"
    assert summary["uncertainty_status"] == "not_estimable"
    assert "not_estimable" in summary["uncertainty_model"]


def test_local_weighting_does_not_depend_on_outside_decade_population():
    """A01: the window weight may not come from whole-segment decade counts."""
    window = np.logspace(0.4, 1.5, 60)
    sparse_tail = np.logspace(1.6, 3.0, 60)
    dense_tail = np.linspace(35.0, 1000.0, 5000)

    def curved(t):
        return t**1.3 * (1.0 + 0.3 * np.sin(2.0 * np.pi * np.log10(t)))

    lag_sparse = np.concatenate([window, sparse_tail])
    lag_dense = np.concatenate([window, dense_tail])
    estimate_sparse = alpha_local_analysis(
        lag_sparse, curved(lag_sparse), window_decades=0.5, min_points=7, min_origins=1
    )
    estimate_dense = alpha_local_analysis(
        lag_dense, curved(lag_dense), window_decades=0.5, min_points=7, min_origins=1
    )
    index = 30  # window centred on the decade boundary, from the shared block
    assert estimate_sparse["alpha_local"][index] == pytest.approx(
        estimate_dense["alpha_local"][index], abs=1e-9
    )
    assert "local" in estimate_sparse["parameters"]["weighting"]


def test_regime_is_three_state_not_boolean():
    estimate = _estimate(_LOG_GRID, _LOG_GRID)
    assert set(estimate["regime_status"]) <= {
        REGIME_INSUFFICIENT,
        REGIME_NOT_DIFFUSIVE,
        REGIME_CONSISTENT,
    }


def test_full_range_lag_arrays_are_not_clipped():
    lag = np.logspace(-1.0, 3.0, 400)
    estimate = _estimate(lag, lag)
    assert len(estimate["alpha_local"]) == len(lag)
    assert len(estimate["window_support_points"]) == len(lag)
    assert len(estimate["time_origin_counts"]) == len(lag)


def _summary(estimate):
    from mliport.analysis.msd import _alpha_regime_summary

    return _alpha_regime_summary(estimate)
