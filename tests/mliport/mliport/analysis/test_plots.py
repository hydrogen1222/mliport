from __future__ import annotations

import numpy as np
import pytest

from mliport.analysis import plots


class _NamedTable:
    def __init__(self, columns):
        self._columns = columns
        self.columns = tuple(columns)

    def __getitem__(self, key):
        return self._columns[key]


def test_plot_msd_alpha_shows_full_range_raw_local_and_support(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    lag_time_ps = np.asarray([0.0, 20.0, 100.0, 180.0, 200.0])
    requested_axes = ("x", "y", "z", "xy", "xyz")
    raw = np.asarray([np.nan, 0.8, 1.0, 2.4, 1.1])
    local = np.asarray([np.nan, np.nan, 1.0, 2.4, -0.3])
    result = {
        "lag_time_ps": lag_time_ps,
        "fit_window_ps": {"start": 20.0, "stop": 180.0},
        "time_origin_counts": np.asarray([5.0, 4.0, 3.0, 2.0, 1.0]),
        "msd_by_axes_A2": {
            axes: np.asarray([0.0, 1.0, 4.0, 7.0, 8.0]) for axes in requested_axes
        },
        "log_log_alpha_by_axes": {axes: raw.copy() for axes in requested_axes},
        "alpha_estimates_by_axes": {
            axes: {
                "alpha_local": local.copy(),
                "window_support_points": np.asarray([0, 0, 4, 4, 3]),
            }
            for axes in requested_axes
        },
    }
    saved = {}

    def capture_figure(fig, output_stem):
        saved["figure"] = fig
        saved["output_stem"] = output_stem
        return []

    monkeypatch.setattr(plots, "_save", capture_figure)
    assert plots.plot_msd_alpha(result, tmp_path / "alpha") == []

    figure = saved["figure"]
    alpha_axis = figure.axes[0]
    labels = [line.get_label() for line in alpha_axis.lines]
    assert all(f"raw alpha {axes}" in labels for axes in requested_axes)
    assert all(f"alpha {axes}" in labels for axes in requested_axes)
    assert labels[-1] == "normal diffusion (alpha = 1)"

    # the default view keeps the complete data range (AL-02): 2.4 and -0.3
    # must be visible instead of being clipped to 0-2
    y_low, y_high = alpha_axis.get_ylim()
    assert y_low < -0.3 and y_high > 2.4
    # the fit window is shaded, not used to crop the axis
    x_low, x_high = alpha_axis.get_xlim()
    assert x_low <= 0.0 and x_high >= 200.0
    spans = [patch for patch in alpha_axis.patches if patch.get_width() > 0]
    assert spans, "the diagnostic fit window must be shaded"

    # support panel: time origins and window support counts
    support_axis = figure.axes[1]
    support_labels = [line.get_label() for line in support_axis.lines]
    assert any("not independent samples" in label for label in support_labels)
    assert len(support_axis.lines) >= 2
    plots._pyplot().close(figure)

    # explicit focus band clips only the VIEW and annotates the count
    assert (
        plots.plot_msd_alpha(result, tmp_path / "alpha_focus", focus_band=(0.0, 2.0))
        == []
    )
    focused = saved["figure"].axes[0]
    assert focused.get_ylim() == (0.0, 2.0)
    assert "outside this band" in focused.get_title()
    # the data arrays are untouched by the focused view
    np.testing.assert_array_equal(
        result["alpha_estimates_by_axes"]["x"]["alpha_local"], local
    )
    plots._pyplot().close(saved["figure"])

    # log-x toggle only changes display
    assert plots.plot_msd_alpha(result, tmp_path / "alpha_log", log_x=True) == []
    assert saved["figure"].axes[0].get_xscale() == "log"
    plots._pyplot().close(saved["figure"])


def test_plot_msd_alpha_accepts_legacy_result_without_estimates(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    result = {
        "lag_time_ps": np.asarray([0.0, 1.0, 2.0]),
        "fit_window_ps": None,
        "msd_by_axes_A2": {"xyz": np.asarray([0.0, 1.0, 2.0])},
        "log_log_alpha_by_axes": {"xyz": np.asarray([np.nan, 1.0, 1.0])},
    }
    saved = {}

    def capture_figure(fig, output_stem):
        saved["figure"] = fig
        return []

    monkeypatch.setattr(plots, "_save", capture_figure)
    assert plots.plot_msd_alpha(result, tmp_path / "legacy") == []
    assert saved["figure"].axes[0].lines
    plots._pyplot().close(saved["figure"])


def test_gemdat_distribution_uses_only_explicit_physical_field(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    saved = {}

    def capture_figure(fig, output_stem):
        saved["figure"] = fig
        saved["output_stem"] = output_stem
        return []

    monkeypatch.setattr(plots, "_save", capture_figure)
    table = _NamedTable(
        {
            "atom index": np.asarray([100, 200]),
            "residence_time_ps": np.asarray([0.1, 0.2]),
        }
    )
    assert (
        plots.plot_electrolyte_distribution(
            {"table": table},
            tmp_path / "residence",
            value_field="residence_time_ps",
            title="Residence-time distribution",
            xlabel="Residence time (ps)",
        )
        == []
    )
    axis = saved["figure"].axes[0]
    right_edges = [patch.get_x() + patch.get_width() for patch in axis.patches]
    assert right_edges
    assert max(right_edges) == pytest.approx(0.2, abs=1.0e-12)
    assert axis.get_xlabel() == "Residence time (ps)"
    plots._pyplot().close(saved["figure"])


def test_gemdat_distribution_warns_and_skips_unlabeled_or_missing_fields(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        plots,
        "_pyplot",
        lambda: pytest.fail("matplotlib must not be opened for an unsafe table"),
    )
    warning_sink = []
    with pytest.warns(RuntimeWarning, match="no explicit 'jump_distance_A'"):
        assert (
            plots.plot_electrolyte_distribution(
                {"table": _NamedTable({"atom index": np.asarray([0, 1])})},
                tmp_path / "jump-distance",
                value_field="jump_distance_A",
                title="Jump-distance distribution",
                xlabel="Jump distance (A)",
                warning_sink=warning_sink,
            )
            == []
        )
    assert len(warning_sink) == 1

    with pytest.warns(RuntimeWarning, match="unlabeled numeric array"):
        assert (
            plots.plot_electrolyte_distribution(
                {"table": np.asarray([[1.0, 2.0]])},
                tmp_path / "jump-rate",
                value_field="jump_rate_s^-1",
                title="Jump-rate distribution",
                xlabel="Jump rate (s^-1)",
            )
            == []
        )
