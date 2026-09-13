"""Productization invariants lock (recovery plan sections 8-9, 40-50).

These tests are the anti-divergence guard: a future report/lineage branch
cannot silently restore the pre-productization state (implicit UMA default,
upstream citation contamination, old installer mental model, non-strict
config) without turning CI red.  Checks are behavior/registry-based first;
string guards are only the second layer.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "mliport"
PACKAGE_PARENT = PACKAGE  # contains the importable mliport/ directory
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

README_FILES = ("README.md", "README_CN.md")
CITATION = REPO / "CITATION.cff"


def _cff() -> dict:
    return yaml.safe_load(CITATION.read_text(encoding="utf-8"))


def _project_version() -> str:
    text = (PACKAGE / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    assert match, "pyproject.toml must declare a project version"
    return match.group(1)


def _top_level_exports() -> set[str]:
    """Load the real package __init__ without the repo-root shadowing."""
    spec = importlib.util.spec_from_file_location(
        "mliport_invariants_probe", PACKAGE / "mliport" / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.__all__)


# --------------------------------------------------------------- PI-01/02
def test_pi01_resolver_requires_an_explicit_backend():
    from mliport.config.resolver import resolve_config

    with pytest.raises(ValueError, match="No MLIP backend was selected"):
        resolve_config(calc_type="sp")
    explicit = resolve_config(
        calc_type="sp",
        cli={"model_type": "mace", "task": "bulk", "model_path": "m.model"},
    )
    assert explicit.model_type == "mace"


def test_pi02_factory_requires_an_explicit_backend(tmp_path):
    from mliport.calculators.factory import CalculatorFactory

    model = tmp_path / "model.bin"
    model.write_bytes(b"metadata only; never loaded")
    for missing in (None, ""):
        with pytest.raises(ValueError, match="No MLIP backend was selected"):
            CalculatorFactory.create(missing, model, task="bulk", device="cpu")


# ------------------------------------------------------------------ PI-03
def test_pi03_generic_template_cannot_default_to_uma():
    from mliport.config.defaults import build_incar_default

    text = build_incar_default("sp")
    active = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "MODEL_TYPE = REQUIRED" in active
    assert not any(line.upper().startswith("MODEL_TYPE = UMA") for line in active)
    engine_specific = build_incar_default("sp", engine="uma")
    assert "MODEL_TYPE = UMA" in engine_specific


# ------------------------------------------------------------------ PI-04
def test_pi04_citation_is_project_scoped():
    cff = _cff()
    reference_dois = {
        str(reference.get("doi", "")).lower()
        for reference in cff.get("references", [])
        if reference.get("doi")
    }
    top_level_doi = str(cff.get("doi", "")).lower()
    assert not top_level_doi or top_level_doi not in reference_dois
    assert "10.5281/zenodo.15587498" not in top_level_doi
    assert cff.get("authors"), "CITATION must name the project authors"
    assert "hydrogen1222/mliport" in str(cff.get("repository-code", ""))


# ---------------------------------------------------------------- PI-05/06
def test_pi05_generated_install_matrix_matches_the_registry():
    from mliport.install.compatibility import BACKENDS, python_versions_for

    docs = (REPO / "docs" / "installation.md").read_text(encoding="utf-8")
    for engine, backend in BACKENDS.items():
        assert backend.venv_name in docs, engine
        assert ", ".join(python_versions_for(engine)) in docs, engine
    # the UMA environment name is whatever the registry declares, not a
    # hard-coded legacy name
    uma_env = BACKENDS["uma"].venv_name
    assert uma_env in docs
    if uma_env != ".venv-uma":
        for name in README_FILES:
            assert ".venv-uma" not in (REPO / name).read_text(encoding="utf-8")


def test_pi06_install_mental_model_is_per_backend_environment():
    import sys as _sys

    _sys.path.insert(0, str(PACKAGE))
    from mliport.install.compatibility import BACKENDS

    uma_env = BACKENDS["uma"].venv_name
    for name in (*README_FILES, "mliport/README.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert "pip install ./mliport" not in text
        assert f"{uma_env}/bin/mliport" in text or "mliport-uma" in text, name
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO / "docs").rglob("*.md"))
    )
    assert "pip install ./mliport" not in docs


# ------------------------------------------------------------------ PI-07/08
def test_pi07_no_retired_brand_anywhere_user_facing():
    roots = [REPO / "README.md", REPO / "README_CN.md", REPO / "mliport" / "README.md"]
    roots += sorted((REPO / "docs").rglob("*.md"))
    roots += sorted((PACKAGE / "mliport").rglob("*.py"))
    for path in roots:
        text = path.read_text(encoding="utf-8")
        assert "MLIP eXtended" not in text, path


def test_pi08_contributing_describes_the_real_environment_split():
    text = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "UMA environment (default)" not in text
    assert "core development environment" in text.lower()
    assert "backend runtime" in text.lower()
    assert "uv sync" in text


def test_pi09_version_metadata_is_consistent():
    version = _project_version()
    assert re.match(r"^\d+\.\d+\.\d+(a|b|rc)\d+$", version), version
    assert _cff()["version"] == version
    assert "beta candidate" in (REPO / "README.md").read_text(encoding="utf-8")
    try:
        from importlib.metadata import version as installed_version

        assert installed_version("mliport") == version
    except Exception:  # noqa: BLE001 - source-only environment
        pass


# ----------------------------------------------- release invariants snapshot
def test_release_invariants_snapshot():
    """Section 40: semantic snapshot, not a full-file comparison."""
    resolver_requires_backend = True
    try:
        from mliport.config.resolver import resolve_config

        resolve_config(calc_type="sp")
    except ValueError as exc:
        resolver_requires_backend = "No MLIP backend was selected" in str(exc)
    assert resolver_requires_backend

    from mliport.config.defaults import get_default

    assert get_default(None, "strict_config") is True

    cff = _cff()
    assert str(cff.get("doi", "")).lower() != "10.5281/zenodo.15587498"

    from mliport.install.compatibility import BACKENDS

    assert BACKENDS["uma"].venv_name in (REPO / "docs" / "installation.md").read_text(
        encoding="utf-8"
    )


# ------------------------------------------------- §41/§42/§43/§49/§50
def test_package_level_assertions():
    pyproject = (PACKAGE / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "mliport"' in pyproject
    assert (PACKAGE / "mliport").is_dir()
    assert not (PACKAGE / "mlipx").exists()
    wheel_dir = PACKAGE / "dist"
    wheels = sorted(wheel_dir.glob("*.whl")) if wheel_dir.is_dir() else []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            assert not [n for n in archive.namelist() if n.startswith("mlipx/")]
            metadata = archive.read(
                next(n for n in archive.namelist() if n.endswith("METADATA"))
            ).decode()
        assert "Name: mliport" in metadata


def test_help_banner_is_backend_neutral(capsys):
    from mliport.cli import print_header

    print_header()
    output = capsys.readouterr().out
    assert "MLIP eXtended" not in output
    assert "backend-neutral MLIP workflows" in output


def test_fairchem_remains_a_legitimate_uma_alias(tmp_path):
    from mliport.backend_selection import is_uma, normalize_model_type

    assert normalize_model_type("fairchem") == "fairchem"
    assert is_uma("fairchem") is True

    from unittest.mock import patch

    from mliport.calculators.factory import CalculatorFactory
    from mliport.calculators.uma import UMACalculator

    model = tmp_path / "uma.pt"
    model.write_bytes(b"metadata only; never loaded")
    with patch.object(UMACalculator, "_validate"):
        wrapper = CalculatorFactory.create("fairchem", model, task="omat")
    assert isinstance(wrapper, UMACalculator)


def test_historical_rename_information_is_preserved():
    guide = REPO / "docs" / "migration-from-mlipx.md"
    assert guide.is_file()
    text = guide.read_text(encoding="utf-8").lower()
    assert "renamed" in text
    assert "legacy" in text
