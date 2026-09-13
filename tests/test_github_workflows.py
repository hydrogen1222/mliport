"""PR-K acceptance tests: GitHub Actions hygiene (section 13).

The workflows must use currently supported action majors (Node 24 runtime),
configure the uv cache against files that actually exist in the repository,
and keep the supported Python matrix at 3.10/3.11/3.12.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
SUPPORTED_PYTHONS = {"3.10", "3.11", "3.12"}

#: Minimum major that runs on a currently supported Node runtime / action API.
MINIMUM_MAJOR = {
    "actions/checkout": 5,
    "astral-sh/setup-uv": 8,
}


def _load(path: Path) -> dict:
    # PyYAML parses the ``on:`` key as a boolean; the jobs mapping is stable.
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _uses_entries(workflow: dict):
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "uses" in step:
                yield step["uses"], step.get("with", {})


def test_workflows_exist_and_only_use_supported_action_majors():
    assert WORKFLOWS
    for path in WORKFLOWS:
        workflow = _load(path)
        for uses, _with in _uses_entries(workflow):
            match = re.fullmatch(r"(?P<action>[^@]+)@v(?P<major>\d+)", uses)
            assert match, f"{path.name}: action {uses!r} must be pinned to a major tag"
            action = match.group("action")
            major = int(match.group("major"))
            if action in MINIMUM_MAJOR:
                assert major >= MINIMUM_MAJOR[action], (
                    f"{path.name}: {uses} predates the supported major "
                    f"v{MINIMUM_MAJOR[action]}"
                )


def test_uv_cache_globs_point_at_real_files():
    seen = 0
    for path in WORKFLOWS:
        workflow = _load(path)
        for uses, with_ in _uses_entries(workflow):
            if not uses.startswith("astral-sh/setup-uv"):
                continue
            assert with_.get("enable-cache") is True, path.name
            glob = with_.get("cache-dependency-glob")
            assert glob, f"{path.name}: setup-uv must configure the cache glob"
            assert (REPO / glob).is_file(), f"{path.name}: {glob} does not exist"
            seen += 1
    assert seen >= 3


def test_python_support_matrix_is_unchanged():
    workflow = _load(REPO / ".github" / "workflows" / "test.yml")
    versions = set()
    for job in workflow["jobs"].values():
        matrix = job.get("strategy", {}).get("matrix", {})
        for version in matrix.get("python-version", []):
            versions.add(str(version))
    assert versions
    assert versions <= SUPPORTED_PYTHONS
    assert versions >= SUPPORTED_PYTHONS


def test_no_cuda_or_driver_mutation_in_workflows():
    for path in WORKFLOWS:
        text = path.read_text(encoding="utf-8")
        assert "nvidia-driver" not in text
        assert "cuda-toolkit" not in text
        # torch is installed from the explicit CPU index only
        if "torch" in text:
            assert "download.pytorch.org/whl/cpu" in text, path.name


def test_no_removed_node16_or_node20_action_majors():
    """A regression guard for the deprecation warnings this PR fixes."""
    for path in WORKFLOWS:
        for uses, _with in _uses_entries(_load(path)):
            if uses.startswith("actions/checkout@v") and uses.endswith(
                ("v1", "v2", "v3", "v4")
            ):
                pytest.fail(f"{path.name}: {uses} uses a deprecated runtime")
            if uses.startswith("actions/upload-artifact@v") and uses.endswith(
                ("v1", "v2", "v3")
            ):
                pytest.fail(f"{path.name}: {uses} uses a deprecated runtime")
