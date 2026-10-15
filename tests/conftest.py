"""Shared pytest fixtures/configuration for mliport.

Deliberately imports nothing from torch/fairchem/ray so that test discovery
does not pull in heavy ML backends.
"""

from __future__ import annotations

import os

import pytest

#: GitHub keeps at most ten error annotations per workflow step.
_MAX_FAILURE_ANNOTATIONS = 9
_MAX_ANNOTATION_CHARS = 6000


def _escape_workflow_command(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    """Expose failures as GitHub annotations (readable through the checks API).

    Workflow logs are not always reachable from review tooling; annotations
    are. This is a no-op outside GitHub Actions.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    reports = [
        *terminalreporter.stats.get("failed", []),
        *terminalreporter.stats.get("error", []),
    ]
    if not reports:
        return
    names = "\n".join(report.nodeid for report in reports)
    terminalreporter.write_line(
        "::error title=pytest failures ("
        f"{len(reports)})::{_escape_workflow_command(names[:_MAX_ANNOTATION_CHARS])}"
    )
    for report in reports[:_MAX_FAILURE_ANNOTATIONS]:
        text = getattr(report, "longreprtext", "") or str(report.longrepr)
        tail = text[-_MAX_ANNOTATION_CHARS:]
        path = report.nodeid.split("::", 1)[0]
        line = ""
        location = getattr(report, "location", None)
        if location and isinstance(location[1], int):
            line = f",line={location[1] + 1}"
        terminalreporter.write_line(
            f"::error file={path}{line},title={_escape_workflow_command(report.nodeid)}"
            f"::{_escape_workflow_command(tail)}"
        )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test to run only on GPU workers")
    config.addinivalue_line(
        "markers", "cpu_and_gpu: mark test to run on both GPU and CPU workers"
    )
    config.addinivalue_line(
        "markers", "wheel_integration: build a wheel and install it into a clean venv"
    )


def pytest_runtest_setup(item):
    if "gpu" not in item.keywords or "cpu_and_gpu" in item.keywords:
        return

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")


@pytest.fixture(scope="session")
def water_xyz_file(tmp_path_factory):
    """Provide a reusable minimal water molecule XYZ file path."""
    contents = (
        "3\n"
        "water\n"
        "O 0.000000 0.000000 0.000000\n"
        "H 0.758602 0.000000 0.504284\n"
        "H -0.758602 0.000000 0.504284\n"
    )
    d = tmp_path_factory.mktemp("xyz_inputs")
    fpath = d / "water.xyz"
    fpath.write_text(contents)
    return str(fpath)
