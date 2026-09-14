#!/usr/bin/env python3
"""Validate mliport wheel/sdist contents before publication.

Usage::

    python scripts/check_package_contents.py dist/*.whl dist/*.tar.gz

The wheel must contain only the ``mliport`` package (plus its ``.dist-info``)
and the sdist must contain only one ``mliport-<version>/`` root.  Job state,
calculation results, build caches, virtual environments, native binaries and
repository-only trees are rejected.
"""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

#: Artifacts that must never ship, anywhere in a distribution.
DENIED_FRAGMENTS = (
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    "jobs/",
    "results/",
    "history/",
    "run.log",
    "resolved_config.json",
    "mliport_results.json",
    ".validation-",
    "local-data/",
    ".venv",
    "mliport/build/",
    "mliport/dist/",
)

DENIED_SUFFIXES = (".pyc", ".pyo", ".so", ".dll", ".dylib")

REQUIRED_WHEEL_MEMBERS = (
    "mliport/__init__.py",
    "mliport/data/compatibility/gpu_architecture_theory.json",
    "mliport/data/validation/mace-v100.json",
)

REQUIRED_WHEEL_METADATA = ("METADATA", "WHEEL", "entry_points.txt")


def _violations(names: list[str], *, wheel: bool) -> list[str]:
    problems: list[str] = []
    for name in names:
        normalized = name.replace("\\", "/")
        if normalized.endswith("/"):
            continue
        if any(fragment in normalized for fragment in DENIED_FRAGMENTS):
            problems.append(f"{name}: denied artifact")
            continue
        if normalized.endswith(DENIED_SUFFIXES):
            problems.append(f"{name}: native/binary artifact is not expected")
            continue
        if wheel:
            top = normalized.split("/", 1)[0]
            if top != "mliport" and not top.endswith(".dist-info"):
                problems.append(f"{name}: unexpected top-level wheel entry")
            if ".egg-info/" in normalized:
                problems.append(f"{name}: build metadata must not ship in a wheel")
    return problems


def check_wheel(names: list[str]) -> list[str]:
    """Return content violations for a wheel member list."""
    problems = _violations(names, wheel=True)
    missing = [member for member in REQUIRED_WHEEL_MEMBERS if member not in names]
    problems.extend(f"{member}: required wheel member is missing" for member in missing)
    dist_info = [name for name in names if ".dist-info/" in name]
    for suffix in REQUIRED_WHEEL_METADATA:
        if not any(name.endswith("/" + suffix) for name in dist_info):
            problems.append(f"dist-info/{suffix}: required metadata is missing")
    return problems


def check_sdist(names: list[str]) -> list[str]:
    """Return content violations for an sdist member list."""
    problems = _violations(names, wheel=False)
    roots = {name.replace("\\", "/").split("/", 1)[0] for name in names if name.strip()}
    if len(roots) != 1:
        problems.append(f"expected exactly one sdist root, found {sorted(roots)}")
        return problems
    root = next(iter(roots))
    if not root.startswith("mliport-"):
        problems.append(f"unexpected sdist root {root!r}")
    return problems


def _wheel_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def _sdist_members(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        return archive.getnames()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = False
    for artifact in args.artifacts:
        if not artifact.is_file():
            print(f"FAIL {artifact}: not a file")
            failed = True
            continue
        if artifact.suffix == ".whl":
            names = _wheel_members(artifact)
            problems = check_wheel(names)
        elif artifact.name.endswith((".tar.gz", ".tgz")):
            names = _sdist_members(artifact)
            problems = check_sdist(names)
        else:
            print(f"SKIP {artifact}: unknown artifact type")
            continue
        if problems:
            failed = True
            print(f"FAIL {artifact} ({len(names)} members)")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"PASS {artifact} ({len(names)} members)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
