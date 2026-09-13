"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import time

import pytest

from mliport.runners.base import BaseRunner
from mliport.timing import RunTiming, append_timing_to_outputs, format_duration


def test_run_timing_uses_progress_phase_boundaries() -> None:
    timing = RunTiming(started_at=10.0)

    timing.observe_phase("loading_model", now=11.0)
    timing.observe_phase("running", now=13.0)
    timing.observe_phase("running", now=14.0)
    timing.observe_phase("writing_output", now=18.5)
    values = timing.finish(now=20.0)

    assert values["total_elapsed_time_s"] == pytest.approx(10.0)
    assert values["compute_time_s"] == pytest.approx(5.5)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (1.25, "1.25 s"),
        (65.5, "1m 5.50s"),
        (3661.0, "1h 1m 1.00s"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_append_timing_to_human_and_json_outputs(tmp_path) -> None:
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("END OF CALCULATION", encoding="utf-8")
    json_path = tmp_path / "mliport_results.json"
    json_path.write_text(
        json.dumps({"calculation": {"timing": {"calculation_time_s": 1.0}}}),
        encoding="utf-8",
    )
    values = {"total_elapsed_time_s": 12.5, "compute_time_s": 8.25}

    append_timing_to_outputs(tmp_path, values)

    outcar_text = outcar.read_text(encoding="utf-8")
    assert "FINAL TIMING SUMMARY" in outcar_text
    assert "Total elapsed time: 12.500000 s" in outcar_text
    assert "Compute time:       8.250000 s" in outcar_text

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["calculation"]["timing"]["calculation_time_s"] == 1.0
    assert data["calculation"]["timing"]["total_elapsed_time_s"] == 12.5
    assert data["calculation"]["timing"]["compute_time_s"] == 8.25


def test_append_timing_supports_layered_md_outputs(tmp_path) -> None:
    (tmp_path / "vasp").mkdir()
    (tmp_path / "raw").mkdir()
    outcar = tmp_path / "vasp" / "OUTCAR"
    outcar.write_text("MLIPORT MD", encoding="utf-8")
    results = tmp_path / "raw" / "mliport_results.json"
    results.write_text(json.dumps({"calculation": {"timing": {}}}), encoding="utf-8")
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": {
                    "outcar": {"path": "vasp/OUTCAR", "bytes": 0},
                    "results": {
                        "path": "raw/mliport_results.json",
                        "bytes": 0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    values = {"total_elapsed_time_s": 4.0, "compute_time_s": 3.0}

    append_timing_to_outputs(tmp_path, values)

    assert "FINAL TIMING SUMMARY" in outcar.read_text()
    assert json.loads(results.read_text())["calculation"]["timing"] == values
    manifest_data = json.loads(manifest.read_text())
    assert manifest_data["timing"] == values
    assert manifest_data["artifacts"]["outcar"]["bytes"] == outcar.stat().st_size


def test_runner_execute_adds_timing_and_delays_done_event(tmp_path) -> None:
    events = []
    logs = []

    class DummyRunner(BaseRunner):
        def run(self, atoms):
            self._emit_progress("loading_model", "Loading")
            self._emit_progress("running", "Computing")
            (self.output_dir / "OUTCAR").write_text("RESULT", encoding="utf-8")
            (self.output_dir / "mliport_results.json").write_text(
                json.dumps({"calculation": {"timing": {}}}),
                encoding="utf-8",
            )
            self._emit_progress("writing_output", "Writing")
            self._emit_progress("done", "Complete", extra={"energy": -1.0})
            return {"energy": -1.0, "time": 0.01}

    runner = DummyRunner(
        calculator=object(),
        output_dir=tmp_path,
        verbose=False,
        log_fn=lambda message, level: logs.append(message),
        progress_callback=events.append,
    )
    results = runner.execute(atoms=None, started_at=time.perf_counter())

    assert [event.phase for event in events] == [
        "loading_model",
        "running",
        "writing_output",
        "done",
    ]
    assert events[-1].extra["timing"] == results["timing"]
    assert "total_elapsed_time_s" in results["timing"]
    assert "compute_time_s" in results["timing"]
    assert any("Total elapsed" in line for line in logs)
    assert "FINAL TIMING SUMMARY" in (tmp_path / "OUTCAR").read_text(encoding="utf-8")
