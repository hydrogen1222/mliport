"""Rename hygiene guard (task book sections 13/15).

Scans tracked text files for the old project name and the retired brand
tagline. Legal residue is explicitly allowlisted: the migration guide, links
to it, the legacy evidence-namespace migration in the validation harness, and
negative assertions in tests that check the wrong name is absent.

Forbidden residue (the old name in production code, current CLI, package
metadata, installation paths or generic user docs) fails CI.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PATTERNS = (
    re.compile(r"mlipx", re.IGNORECASE),
    re.compile(r"MLIP\s+eXtended", re.IGNORECASE),
    re.compile(r"\.venv-uma", re.IGNORECASE),
    re.compile(r"hydrogen1222/mlipx", re.IGNORECASE),
    re.compile(r"pip install mlipx", re.IGNORECASE),
)

#: Files where the old name is legitimate archaeology or a legacy-compat
#: implementation detail.
ALLOWED_FILES = {
    "CHANGELOG.md",  # changelog mentions the old name as history
    "docs/migration-from-mlipx.md",  # the migration guide itself
    "docs/index.md",  # "migrate from the old mlipx name" table row
    "README.md",  # links to the migration guide
    "README_CN.md",
    "mliport/README.md",  # generated from the root README
    "docs/development.md",  # build check: "no mlipx namespace in the wheel"
    "validation/science/evidence/constants.py",  # legacy schema/namespace migration
    "validation/science/evidence/loader.py",  # legacy schema normalization
    "validation/science/reports/BETA_VALIDATION.md",  # historical rename note
    "validation/science/reports/BETA_VALIDATION_CN.md",
    "validation/science/scripts/generate_beta_report.py",  # renders that note
    "tests/mliport/mliport/test_science_harness.py",  # asserts legacy paths are accepted
    "tests/mliport/mliport/test_docs_contract.py",  # required tree includes the guide
    "tests/mliport/mliport/test_install_ux_contract.py",  # negative assertion
    "tests/test_rename_hygiene.py",  # this guard
}

#: The only old-name substring allowed in otherwise-clean files: links to the
#: migration guide (``migration-from-mlipx.md``).
ALLOWED_MATCH_CONTEXT = "migration-from-mlipx"


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def test_no_forbidden_rename_residue_in_tracked_files():
    violations: list[str] = []
    for relative in _tracked_files():
        path = REPO / relative
        text = _read_text(path)
        if text is None:
            continue
        for pattern in PATTERNS:
            for match in pattern.finditer(text):
                context = text[max(0, match.start() - 40) : match.end() + 40]
                if relative in ALLOWED_FILES:
                    continue
                if ALLOWED_MATCH_CONTEXT in context:
                    continue
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{relative}:{line}: {match.group(0)!r}")
    assert not violations, "forbidden rename residue:\n" + "\n".join(violations)


def test_current_brand_tagline_is_backend_neutral():
    cli = (REPO / "mliport" / "mliport" / "cli.py").read_text(encoding="utf-8")
    assert "backend-neutral MLIP workflows" in cli
    assert "MLIP eXtended" not in cli
    tui = (REPO / "mliport" / "mliport" / "tui" / "app.py").read_text(encoding="utf-8")
    assert "backend-neutral MLIP workflows" in tui
    outcar = (REPO / "mliport" / "mliport" / "writers" / "outcar.py").read_text(
        encoding="utf-8"
    )
    assert "backend-neutral MLIP workflows" in outcar


def test_package_description_is_backend_neutral():
    pyproject = (REPO / "mliport" / "pyproject.toml").read_text(encoding="utf-8")
    assert "backend-neutral" in pyproject.lower()
    assert "VASP-like CLI/TUI interface" not in pyproject
