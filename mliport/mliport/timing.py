"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Unified wall-clock timing and timing-output helpers for mliport runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass
class RunTiming:
    """Track end-to-end and model-ready-to-output wall-clock durations."""

    started_at: float
    compute_started_at: float | None = None
    compute_finished_at: float | None = None
    finished_at: float | None = None

    def observe_phase(self, phase: str, now: float | None = None) -> None:
        """Record calculation boundaries from runner progress phases."""
        timestamp = time.perf_counter() if now is None else now
        if phase == "running" and self.compute_started_at is None:
            self.compute_started_at = timestamp
        elif (
            phase == "writing_output"
            and self.compute_started_at is not None
            and self.compute_finished_at is None
        ):
            self.compute_finished_at = timestamp

    def mark_compute_started(self, now: float | None = None) -> None:
        """Explicitly mark a compute boundary, primarily for batch runs."""
        if self.compute_started_at is None:
            self.compute_started_at = time.perf_counter() if now is None else now

    def mark_compute_finished(self, now: float | None = None) -> None:
        """Explicitly mark the end of computation."""
        if self.compute_finished_at is None:
            self.compute_finished_at = time.perf_counter() if now is None else now

    def finish(self, now: float | None = None) -> dict[str, float]:
        """Finish the timer and return serializable timing values."""
        self.finished_at = time.perf_counter() if now is None else now
        if self.compute_started_at is None:
            self.compute_started_at = self.finished_at
        if self.compute_finished_at is None:
            self.compute_finished_at = self.finished_at

        return {
            "total_elapsed_time_s": max(0.0, self.finished_at - self.started_at),
            "compute_time_s": max(
                0.0, self.compute_finished_at - self.compute_started_at
            ),
        }


def format_duration(seconds: float) -> str:
    """Format a duration for concise human-readable terminal output."""
    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining:.2f}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remaining:.2f}s"


def timing_log_lines(timing: dict[str, float]) -> list[str]:
    """Build the standard timing summary shown by every interface."""
    total = timing["total_elapsed_time_s"]
    compute = timing["compute_time_s"]
    return [
        "",
        "Timing summary:",
        "  Total elapsed (run requested -> outputs finished): "
        f"{format_duration(total)}",
        "  Compute elapsed (model ready -> output writing): "
        f"{format_duration(compute)}",
    ]


def append_timing_to_outputs(
    output_dir: Path | str,
    timing: dict[str, float],
) -> None:
    """Append timing to human output and update machine-readable JSON output."""
    output_dir = Path(output_dir)
    outcar_paths = (output_dir / "OUTCAR", output_dir / "vasp" / "OUTCAR")
    for outcar_path in outcar_paths:
        if not outcar_path.exists():
            continue
        total = timing["total_elapsed_time_s"]
        compute = timing["compute_time_s"]
        block = [
            "",
            "",
            "-" * 80,
            " FINAL TIMING SUMMARY",
            "-" * 80,
            "",
            f"Total elapsed time: {total:.6f} s",
            f"Compute time:       {compute:.6f} s",
            "",
            "Total:   run requested -> all standard outputs finished",
            "Compute: model ready -> output writing started",
            "=" * 80,
        ]
        with open(outcar_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(block))
            handle.write("\n")

    json_paths = (
        output_dir / "mliport_results.json",
        output_dir / "raw" / "mliport_results.json",
        output_dir / "batch_summary.json",
    )
    for json_path in json_paths:
        if not json_path.exists():
            continue
        with open(json_path, encoding="utf-8") as handle:
            data: dict[str, Any] = json.load(handle)

        if json_path.name == "mliport_results.json":
            calculation = data.setdefault("calculation", {})
            calculation.setdefault("timing", {}).update(timing)
        else:
            data["timing"] = timing

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    manifest_path = output_dir / "artifacts.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            manifest: dict[str, Any] = json.load(handle)
        manifest["timing"] = timing
        for artifact in manifest.get("artifacts", {}).values():
            artifact_path = output_dir / artifact["path"]
            if artifact_path.exists():
                artifact["bytes"] = artifact_path.stat().st_size
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
