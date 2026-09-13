"""Public-surface regression tests for the MLMD-focused project scope."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from mliport import api
from mliport.cli import create_parser
from mliport.config import get_schema


def test_analysis_v2_is_importable_without_calculators() -> None:
    module = importlib.import_module("mliport.analysis")
    assert hasattr(module, "TrajectoryDataset")
    code = """
import sys
import mliport.analysis
blocked = ('mliport.calculators', 'torch', 'fairchem', 'mace', 'deepmd', 'matgl')
assert not any(name.startswith(blocked) for name in sys.modules), sorted(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_declared_runner_exports_resolve_without_backend_imports() -> None:
    code = """
from mliport import EngineConfig, SinglePointRunner
assert EngineConfig.__name__ == 'EngineConfig'
assert SinglePointRunner.__name__ == 'SinglePointRunner'
"""
    package_root = Path(__file__).parents[3] / "mliport"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=package_root)


def test_analysis_v2_never_imports_archive() -> None:
    analysis = Path(__file__).parents[3] / "mliport" / "mliport" / "analysis"
    for path in analysis.glob("*.py"):
        assert "archive" not in path.read_text(encoding="utf-8")


def test_cli_exposes_analysis_v2_command() -> None:
    parser = create_parser()
    subcommands = next(
        action.choices for action in parser._actions if getattr(action, "choices", None)
    )
    assert "analyze" in subcommands


def test_adsorption_convenience_api_is_not_public() -> None:
    assert "calculate_adsorption_energy" not in api.__all__
    assert not hasattr(api, "calculate_adsorption_energy")


def test_equil_steps_is_a_strictly_valid_option() -> None:
    errors = get_schema().validate_dict(
        {"EQUIL_STEPS": 10}, strict=True, context="INCAR"
    )
    assert errors == []


def test_analysis_optional_dependencies_are_declared_as_extras() -> None:
    pyproject = Path(__file__).parents[3] / "mliport" / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "kinisi" in text
    assert "gemdat" in text
    assert "matplotlib" in text
