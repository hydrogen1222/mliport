"""Regression tests for repository and release-artifact hygiene."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.check_repository_hygiene import (
    HygieneViolation,
    validate_artifact,
    validate_tracked_paths,
)


def test_current_tracked_repository_obeys_hygiene_policy() -> None:
    repo_root = Path(__file__).parents[1]
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    validate_tracked_paths(path for path in result.stdout.split("\0") if path)


@pytest.mark.parametrize(
    "tracked_path",
    [
        "revise.md",
        "uv.lock",
        "archive/legacy.py",
        "audit_round7.md",
        "mlipx_next_phase_NOT_FOR_REPO_2026.md",
        "mlipx_development_plan.md",
    ],
)
def test_repository_hygiene_rejects_local_or_unapproved_files(
    tracked_path: str,
) -> None:
    with pytest.raises(HygieneViolation):
        validate_tracked_paths([tracked_path])


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_distribution_hygiene_accepts_package_source(tmp_path: Path) -> None:
    wheel = tmp_path / "mlipx-2.0.0-py3-none-any.whl"
    _write_wheel(
        wheel,
        {
            "mlipx/__init__.py": b'__version__ = "2.0.0"\n',
            "mlipx-2.0.0.dist-info/LICENSE.md": b"MIT\n",
        },
    )
    validate_artifact(wheel)


@pytest.mark.parametrize(
    "member",
    [
        "mlipx/archive/legacy.py",
        "mlipx/checkpoints/model.pt",
        "mlipx/results/run.traj",
        "mlipx/revise.md",
        "mlipx/uv.lock",
        "mlipx/local_NOT_FOR_REPO_notes.md",
    ],
)
def test_distribution_hygiene_rejects_forbidden_members(
    tmp_path: Path, member: str
) -> None:
    wheel = tmp_path / "bad.whl"
    _write_wheel(wheel, {member: b"placeholder\n"})
    with pytest.raises(HygieneViolation):
        validate_artifact(wheel)


def test_distribution_hygiene_rejects_local_absolute_paths(tmp_path: Path) -> None:
    wheel = tmp_path / "bad-path.whl"
    _write_wheel(
        wheel,
        {"mlipx/example.py": b'MODEL = "/home/alice/private/model.pt"\n'},
    )
    with pytest.raises(HygieneViolation, match="local absolute path"):
        validate_artifact(wheel)
