"""PR-M acceptance tests: beta version policy (section 15).

While the README describes a beta candidate, the package version must be a
PEP 440 pre-release (for example ``2.0.0b1``), never a plain stable-looking
``2.0.0``; the installed metadata and the committed pyproject must agree.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PYPROJECT = REPO / "mliport" / "pyproject.toml"
PRERELEASE = re.compile(r"^\d+\.\d+\.\d+(a|b|rc)\d+$")


def _project_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    assert match, "pyproject.toml must declare a project version"
    return match.group(1)


def test_beta_candidate_uses_a_prerelease_version():
    assert "beta candidate" in (REPO / "README.md").read_text(encoding="utf-8")
    project_version = _project_version()
    assert PRERELEASE.match(project_version), (
        f"{project_version!r} looks stable; a beta candidate needs a "
        "pre-release version such as 2.0.0b1"
    )


def test_installed_metadata_matches_the_committed_pyproject():
    try:
        installed = version("mliport")
    except PackageNotFoundError:  # pragma: no cover - source-only environment
        return
    assert installed == _project_version()


def test_readme_does_not_claim_a_stable_release():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "version 2.0.0 stable" not in lowered
    assert "stable release" not in lowered
