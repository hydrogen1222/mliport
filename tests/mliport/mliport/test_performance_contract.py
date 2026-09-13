"""PR7 acceptance: T8 benchmark contract (PERF-01, task book sections 21-24).

The benchmark must separate startup, untimed warmup and timed repetitions,
report the full latency spread, and flag anomalous scaling instead of
silently publishing a median from three noisy samples.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "validation" / "science" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_beta_report as report  # noqa: E402
import performance_suite as perf  # noqa: E402


def _sp_record(natoms: int, median_s: float, status: str = "pass") -> dict:
    return {
        "case_id": f"t8_sp_scaling_{natoms}",
        "test_id": "t8_performance",
        "engine": "mace",
        "status": status,
        "metrics": {"natoms": natoms, "median_ms": median_s * 1000.0},
        "diagnostics": {},
    }


def test_perf01_contract_constants_are_enforced():
    assert perf.WARMUP_TARGET >= 5
    assert perf.WARMUP_MAX >= perf.WARMUP_TARGET
    assert perf.TIMED_REPS >= 20
    assert perf.ANOMALY_RATIO == 5.0


def test_summarize_timings_reports_full_spread():
    stats = perf.summarize_timings([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["n_timed"] == 5
    for key in ("median_ms", "mean_ms", "p05_ms", "p95_ms", "min_ms", "max_ms"):
        assert key in stats, key
    assert stats["median_ms"] == 3000.0
    assert stats["min_ms"] == 1000.0
    assert stats["max_ms"] == 5000.0


def test_classify_scaling_flags_five_x_anomaly():
    records = [
        _sp_record(32, 0.5),
        _sp_record(128, 0.05),
        _sp_record(512, 0.1),
    ]
    by_natoms = {
        rec["metrics"]["natoms"]: rec for rec in perf.classify_scaling(records)
    }
    assert by_natoms[32]["diagnostics"]["benchmark_anomaly"] is True
    assert by_natoms[32]["status"] == "characterized"
    assert (
        by_natoms[32]["diagnostics"]["scaling_verdict"] == "characterized_with_warning"
    )


def test_classify_scaling_marks_non_monotonic_scaling():
    records = [
        _sp_record(32, 0.01),
        _sp_record(128, 0.50),
        _sp_record(512, 0.02),
    ]
    by_natoms = {
        rec["metrics"]["natoms"]: rec for rec in perf.classify_scaling(records)
    }
    assert by_natoms[128]["diagnostics"]["scaling_verdict"] == (
        "characterized_with_warning"
    )
    assert by_natoms[128]["status"] == "characterized"
    assert "benchmark_anomaly" not in by_natoms[32]["diagnostics"]


def test_classify_scaling_keeps_healthy_scaling_clean():
    records = [
        _sp_record(32, 0.01),
        _sp_record(128, 0.04),
        _sp_record(512, 0.16),
    ]
    classified = perf.classify_scaling(records)
    assert all(rec["status"] == "pass" for rec in classified)
    assert all("benchmark_anomaly" not in rec["diagnostics"] for rec in classified)


def test_classify_scaling_ignores_within_noise_inversion():
    """A ~2% inversion (UMA 128 vs 32 atoms) is noise, not a warning."""
    records = [
        _sp_record(32, 0.1292),
        _sp_record(128, 0.1267),
        _sp_record(512, 0.2229),
    ]
    classified = perf.classify_scaling(records)
    assert all(record["status"] == "pass" for record in classified)
    assert all("scaling_verdict" not in record["diagnostics"] for record in classified)


def test_report_t8_renders_spread_and_anomaly():
    record = _sp_record(32, 0.5)
    record["metrics"].update(
        {
            "median_ms": 500.0,
            "mean_ms": 510.0,
            "p05_ms": 480.0,
            "p95_ms": 540.0,
            "min_ms": 470.0,
            "max_ms": 560.0,
            "n_timed": 20,
            "warmup_count": 5,
            "first_inference_ms": 600.0,
            "model_load_s": 2.5,
            "atoms_per_second_warm": 64.0,
        }
    )
    record["diagnostics"] = {
        "benchmark_anomaly": True,
        "scaling_verdict": "characterized_with_warning",
    }
    text = report.md_t8([record])
    for token in ("p05", "p95", "n_timed", "warmup", "benchmark_anomaly"):
        assert token in text, token


def test_current_campaign_t8_records_carry_the_full_contract():
    t8_dir = REPO / ".validation-work" / "t8"
    if not t8_dir.is_dir():
        return  # GPU evidence is not part of the lightweight CI checkout
    import json

    candidates = sorted(t8_dir.glob("t8_sp_scaling_*__t8_performance__*.json"))
    if not candidates:
        return  # the rerun writes them; CI has no GPU evidence
    checked = 0
    for path in candidates:
        record = json.loads(path.read_text(encoding="utf-8"))
        if "harness_commit" not in record.get("parameters", {}):
            continue  # pre-PERF-01 historical record, out of this contract
        metrics = record.get("metrics", {})
        if not metrics:
            continue
        for key in (
            "first_inference_ms",
            "warmup_count",
            "n_timed",
            "median_ms",
            "mean_ms",
            "p05_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
        ):
            assert key in metrics, f"{path.name}: missing {key}"
        assert metrics["n_timed"] >= perf.TIMED_REPS
        assert metrics["warmup_count"] >= perf.WARMUP_TARGET
        checked += 1
    if checked == 0:
        return  # no PERF-01 records yet (rerun happens on the GPU host)
