"""Distribution content checks (task book PKG-T1/T2, STATE-T4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_package_contents import check_sdist, check_wheel


def _clean_wheel() -> list[str]:
    return [
        "mliport/__init__.py",
        "mliport/cli.py",
        "mliport/data/compatibility/gpu_architecture_theory.json",
        "mliport/data/validation/mace-v100.json",
        "mliport-2.0.0b4.dist-info/METADATA",
        "mliport-2.0.0b4.dist-info/WHEEL",
        "mliport-2.0.0b4.dist-info/entry_points.txt",
        "mliport-2.0.0b4.dist-info/RECORD",
    ]


def test_clean_wheel_passes() -> None:
    assert check_wheel(_clean_wheel()) == []


@pytest.mark.parametrize(
    "member, expected",
    [
        ("examples/__init__.py", "unexpected top-level"),
        ("tests/test_cli.py", "unexpected top-level"),
        ("mliport/jobs/deadbeef.json", "denied artifact"),
        ("mliport/results/run/out.log", "denied artifact"),
        ("mliport/__pycache__/cli.pyc", "denied"),
        ("mliport/libdeepmd_op_cuda.so", "not expected"),
    ],
)
def test_wheel_rejects_packaging_residue(member: str, expected: str) -> None:
    problems = check_wheel([*_clean_wheel(), member])
    assert any(expected in problem for problem in problems), problems


def test_wheel_requires_validation_data() -> None:
    names = [n for n in _clean_wheel() if "data/" not in n]
    problems = check_wheel(names)
    assert any("required wheel member is missing" in problem for problem in problems)


def test_wheel_requires_dist_info_metadata() -> None:
    names = [
        n
        for n in _clean_wheel()
        if not n.endswith(("/METADATA", "/WHEEL", "/entry_points.txt"))
    ]
    problems = check_wheel(names)
    assert sum("required metadata is missing" in p for p in problems) == 3


def _clean_sdist() -> list[str]:
    return [
        "mliport-2.0.0b4/PKG-INFO",
        "mliport-2.0.0b4/README.md",
        "mliport-2.0.0b4/pyproject.toml",
        "mliport-2.0.0b4/mliport/__init__.py",
        "mliport-2.0.0b4/mliport/cli.py",
    ]


def test_clean_sdist_passes() -> None:
    assert check_sdist(_clean_sdist()) == []


def test_sdist_rejects_job_state_and_second_root() -> None:
    problems = check_sdist(
        [
            *_clean_sdist(),
            "mliport-2.0.0b4/jobs/deadbeef.json",
            "examples/structures/li10gep2s12_primitive.vasp",
        ]
    )
    assert any("denied artifact" in problem for problem in problems)
    assert any("exactly one sdist root" in problem for problem in problems)


def test_built_artifacts_if_available() -> None:
    """Optional hook: CI passes a directory through MLIPORT_PACKAGE_ARTIFACTS."""
    import os

    from scripts.check_package_contents import main as check_main

    directory = os.environ.get("MLIPORT_PACKAGE_ARTIFACTS")
    if not directory:
        pytest.skip("MLIPORT_PACKAGE_ARTIFACTS is not set")
    artifacts = sorted(str(path) for path in Path(directory).glob("*"))
    assert artifacts
    assert check_main(artifacts) == 0
