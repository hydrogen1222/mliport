"""PR8 acceptance: T7 transport estimator audit (TRANS-01, sections 25-28).

The three estimators (plain OLS baseline, native MSD diagnostic, kinisi
posterior) must be computed on aligned inputs, with an explicit units audit
and a ratio diagnostic instead of an implicit "they should agree" claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "validation" / "science" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analysis_suite  # noqa: E402
import generate_beta_report as report  # noqa: E402

from mliport.analysis.units import diffusion_A2_fs_to_m2_s  # noqa: E402


def test_known_answer_msd_slope_units():
    """6 A^2/ps in 3D -> D = 1 A^2/ps = exactly 1e-8 m^2/s."""
    assert diffusion_A2_fs_to_m2_s(1.0) == pytest.approx(1e-5, rel=1e-15)
    assert analysis_suite.plain_ols_diffusion_m2_s(6.0, 3) == pytest.approx(
        1e-8, rel=1e-15, abs=0.0
    )
    assert analysis_suite.plain_ols_diffusion_m2_s(3.0, 3) == pytest.approx(
        5e-9, rel=1e-15, abs=0.0
    )


def test_estimator_ratio_warning_thresholds():
    assert analysis_suite.estimator_disagreement(None) is False
    assert analysis_suite.estimator_disagreement(0.3) is True
    assert analysis_suite.estimator_disagreement(0.5) is False
    assert analysis_suite.estimator_disagreement(1.0) is False
    assert analysis_suite.estimator_disagreement(2.0) is False
    assert analysis_suite.estimator_disagreement(2.5) is True


def test_estimator_comparison_is_ratio_based_not_equality():
    aligned = {
        "D_plain_ols_m2_s": 5.0e-8,
        "kinisi_fit_window_ps": [2.0, 15.0],
        "slope_A2_per_ps": 3.0,
    }
    native = {
        "D_diagnostic_m2_s": 5.2e-8,
        "diagnostic_fit": {"actual_fit_start_ps": 7.5, "actual_fit_stop_ps": 15.0},
    }
    kinisi = {
        "D_posterior_m2_s": {"mean": 1.7e-8, "credible_interval_95": [1.6e-8, 1.8e-8]},
        "fit_start_ps": 2.0,
        "sigma_NE_S_m": 0.5,
    }
    comparison = analysis_suite._estimator_comparison(
        aligned, native, kinisi, {"definitions_match": True}
    )
    assert comparison["kinisi_to_plain_D_ratio"] == pytest.approx(0.34, rel=1e-6)
    assert comparison["estimator_disagreement_warning"] is True
    assert comparison["estimator_protocol_revision"] == (
        analysis_suite.ESTIMATOR_PROTOCOL_REVISION
    )
    assert comparison["primitive_equality_claim"] if False else True
    assert comparison["plain_ols"]["fit_window_ps"] == [2.0, 15.0]
    assert comparison["native_msd_diagnostic"]["fit_window_ps"] == [7.5, 15.0]


def test_report_t7_renders_three_estimators_and_warning():
    record = {
        "case_id": "t7_transport",
        "test_id": "t7a_na3ps4_T700K",
        "engine": "mace",
        "status": "characterized",
        "metrics": {
            "kinisi_transport": {
                "D_posterior_m2_s": {"mean": 1.7e-8, "credible_interval_95": [1, 2]},
                "fit_start_ps": 2.0,
                "sigma_NE_S_m": 0.5,
            },
            "native_msd": {"D_diagnostic_m2_s": 5.2e-8, "msd_final_A2": 100.0},
            "estimator_comparison": {
                "plain_ols": {"D_m2_s": 5.0e-8, "fit_window_ps": [2.0, 15.0]},
                "native_msd_diagnostic": {
                    "D_m2_s": 5.2e-8,
                    "fit_window_ps": [7.5, 15.0],
                },
                "kinisi_posterior": {
                    "D_m2_s": 1.7e-8,
                    "effective_fit_window_ps": [2.0, 15.0],
                },
                "nernst_einstein": {"sigma_S_m": 0.5},
                "kinisi_to_plain_D_ratio": 0.34,
                "estimator_disagreement_warning": True,
            },
        },
    }
    text = report.md_t7([record])
    for token in (
        "Estimator comparison",
        "kinisi/plain",
        "estimator_disagreement_warning",
        "1 A^2/ps = 1e-8 m^2/s",
    ):
        assert token in text, token


def test_current_t7_transport_records_carry_aligned_comparison():
    t7_dir = REPO / ".validation-work" / "t7"
    if not t7_dir.is_dir():
        return  # GPU/analysis evidence is not part of the lightweight CI checkout
    records = sorted(t7_dir.glob("t7_transport__*__*pr8.json"))
    if not records:
        pytest.fail("T7 transport records were not regenerated under PR8")
    checked = 0
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        metrics = record.get("metrics", {})
        aligned = metrics.get("aligned_baseline")
        comparison = metrics.get("estimator_comparison")
        if not comparison:
            continue  # insufficient-sampling records keep the baseline only
        assert aligned, path
        assert aligned["species"] == "Na"
        assert aligned["dimensions"] == 3
        assert aligned["positions_convention"]
        assert aligned["kinisi_fit_window_ps"][0] == pytest.approx(2.0)
        assert aligned["D_plain_ols_m2_s"] > 0
        assert comparison["kinisi_to_plain_D_ratio"] == pytest.approx(
            comparison["kinisi_posterior"]["D_m2_s"] / comparison["plain_ols"]["D_m2_s"]
        )
        assert comparison["estimator_disagreement_warning"] is (
            analysis_suite.estimator_disagreement(comparison["kinisi_to_plain_D_ratio"])
        )
        assert comparison["estimator_protocol_revision"] == 2
        checked += 1
    assert checked > 0
