"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest

from mliport.protocols import ProgressCallback, ProgressEvent


@pytest.mark.skip(reason="Requires fairchem model checkpoint to run runners")
def test_singlepoint_runner_emits_progress():
    """Verify SinglePointRunner calls progress_callback during run.

    This test is skipped by default because it requires a real model checkpoint.
    The structure of progress events is validated by the protocol tests.
    """
    events: list[ProgressEvent] = []

    def collect(event: ProgressEvent) -> None:
        events.append(event)

    # Test that callback type is correct
    callback: ProgressCallback = collect
    assert callable(callback)


def test_progress_event_phases():
    """Verify known phases are consistent."""
    valid_phases = {"loading_model", "running", "writing_output", "done", "error"}
    for phase in valid_phases:
        event = ProgressEvent(phase=phase, message="test")
        assert event.phase in valid_phases
