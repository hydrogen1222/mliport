"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from mliport.protocols import ProgressEvent


def test_progress_event_creation():
    event = ProgressEvent(
        phase="running",
        message="Calculating...",
        step=5,
        total_steps=100,
        extra={"energy": -123.4},
    )
    assert event.phase == "running"
    assert event.message == "Calculating..."
    assert event.step == 5
    assert event.total_steps == 100
    assert event.extra == {"energy": -123.4}


def test_progress_event_sp_no_steps():
    """SP calculation has no step/total — use None."""
    event = ProgressEvent(
        phase="loading_model",
        message="Loading model...",
        step=None,
        total_steps=None,
    )
    assert event.step is None
    assert event.total_steps is None


def test_progress_event_defaults():
    event = ProgressEvent(phase="done", message="Complete")
    assert event.step is None
    assert event.total_steps is None
    assert event.extra is None
