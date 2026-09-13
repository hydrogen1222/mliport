#!/usr/bin/env python3
"""Fail closed when local-only material leaks into Git or distributions."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

ROOT_MARKDOWN_ALLOWLIST = frozenset(
    {
        "AGENTS.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE.md",
        "README.md",
        "CHANGELOG.md",
        "README_CN.md",
    }
)

_FORBIDDEN_TRACKED_EXACT = frozenset({"revise.md", "uv.lock"})
_FORBIDDEN_COMPONENTS = frozenset(
    {
        ".agent-local",
        ".local-plans",
        ".pytest_cache",
        "__pycache__",
        "archive",
        "audit-local",
        "checkpoints",
        "results",
    }
)
_FORBIDDEN_ARTIFACT_NAMES = frozenset(
    {
        ".env",
        ".pypirc",
        "config.yml",
        "config.yaml",
        "credentials",
        "credentials.json",
        "resolved_config.json",
        "revise.md",
        "settings.ini",
        "uv.lock",
    }
)
_FORBIDDEN_ARTIFACT_SUFFIXES = (
    ".aselmdb",
    ".lmdb",
    ".lmdb-lock",
    ".model",
    ".pt",
    ".pth",
    ".traj",
)
_LOCAL_ABSOLUTE_PATHS = (
    re.compile(rb"(?<![A-Za-z0-9])/(?:home|Users)/[A-Za-z0-9._-]+/"),
    re.compile(rb"(?i)(?<![A-Za-z0-9])[A-Z]:\\\\Users\\\\[A-Za-z0-9._-]+\\\\"),
)
_TEXT_SUFFIXES = frozenset(
    {".cfg", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
)


class HygieneViolation(ValueError):
    """Raised when repository or artifact hygiene checks fail."""


def _normalise_member(name: str) -> tuple[str, tuple[str, ...]]:
    normalised = name.replace("\\", "/")
    path = PurePosixPath(normalised)
    parts = tuple(part for part in path.parts if part not in {"", "."})
    return normalised, parts


def tracked_path_violations(paths: Iterable[str]) -> list[str]:
    """Return policy violations in a ``git ls-files`` path sequence."""
    violations: list[str] = []
    for raw in paths:
        normalised, parts = _normalise_member(raw)
        if not parts:
            continue
        name = parts[-1]
        if normalised in _FORBIDDEN_TRACKED_EXACT:
            violations.append(f"forbidden tracked path: {normalised}")
        if _FORBIDDEN_COMPONENTS.intersection(parts):
            violations.append(f"local/history directory is tracked: {normalised}")
        if "_NOT_FOR_REPO_" in name or "_development_plan" in name.lower():
            violations.append(f"transient plan is tracked: {normalised}")
        if (
            len(parts) == 1
            and name.lower().endswith(".md")
            and name not in ROOT_MARKDOWN_ALLOWLIST
        ):
            violations.append(f"root Markdown is not allowlisted: {normalised}")
    return sorted(set(violations))


def validate_tracked_paths(paths: Iterable[str]) -> None:
    """Raise when tracked repository paths violate the hygiene policy."""
    violations = tracked_path_violations(paths)
    if violations:
        raise HygieneViolation("\n".join(violations))


def artifact_member_violations(name: str) -> list[str]:
    """Return policy violations for one wheel/sdist member name."""
    normalised, parts = _normalise_member(name)
    if not parts:
        return []

    violations: list[str] = []
    if normalised.startswith("/") or re.match(r"^[A-Za-z]:/", normalised):
        violations.append(f"absolute artifact member path: {name}")
    if ".." in parts:
        violations.append(f"parent traversal in artifact member: {name}")
    if _FORBIDDEN_COMPONENTS.intersection(parts):
        violations.append(f"forbidden directory in artifact: {name}")

    basename = parts[-1]
    basename_lower = basename.lower()
    if basename_lower in _FORBIDDEN_ARTIFACT_NAMES:
        violations.append(f"forbidden file in artifact: {name}")
    if basename_lower.endswith(_FORBIDDEN_ARTIFACT_SUFFIXES):
        violations.append(f"model/data artifact packaged: {name}")
    if "_not_for_repo_" in basename_lower or "_development_plan" in basename_lower:
        violations.append(f"transient plan packaged: {name}")
    return violations


def artifact_content_violations(name: str, data: bytes) -> list[str]:
    """Return local absolute-path leaks from a text artifact member."""
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in _TEXT_SUFFIXES:
        return []
    return [
        f"local absolute path embedded in artifact member: {name}"
        for pattern in _LOCAL_ABSOLUTE_PATHS
        if pattern.search(data)
    ]


def validate_artifact(path: Path) -> None:
    """Inspect one wheel, zip, or source tarball without extracting it."""
    violations: list[str] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                violations.extend(artifact_member_violations(member.filename))
                if not member.is_dir():
                    violations.extend(
                        artifact_content_violations(
                            member.filename, archive.read(member.filename)
                        )
                    )
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive.getmembers():
                violations.extend(artifact_member_violations(member.name))
                if member.issym() or member.islnk():
                    violations.append(f"link member is not allowed: {member.name}")
                    continue
                if member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        violations.extend(
                            artifact_content_violations(member.name, extracted.read())
                        )
    else:
        raise HygieneViolation(f"unsupported distribution format: {path}")

    if violations:
        details = "\n".join(sorted(set(violations)))
        raise HygieneViolation(f"{path}:\n{details}")


def _tracked_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [path for path in result.stdout.split("\0") if path]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="*",
        type=Path,
        help="Optional built wheel/sdist files to inspect.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing .git (default: script parent).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run repository and optional distribution checks."""
    args = _build_parser().parse_args(argv)
    try:
        validate_tracked_paths(_tracked_paths(args.repo_root.resolve()))
        for artifact in args.artifacts:
            if not artifact.is_file():
                raise HygieneViolation(f"distribution does not exist: {artifact}")
            validate_artifact(artifact)
    except (HygieneViolation, subprocess.CalledProcessError) as exc:
        print(f"repository hygiene check failed:\n{exc}", file=sys.stderr)
        return 1

    suffix = f" and {len(args.artifacts)} distribution(s)" if args.artifacts else ""
    print(f"repository hygiene check passed{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
