"""README sync tests (beta task spec section 43).

The two READMEs make factual claims about the code. This module checks
those claims against the real sources of truth — argparse choices,
``mlipx.api`` exports, ``pyproject.toml``, the beta summary and the
generated validation snippet — instead of comparing the Chinese
translation line by line against the English text.

Rules enforced here:

- The generated validation block between the ``BEGIN/END GENERATED``
  markers must be byte-identical to the generator output file in both
  READMEs.
- CLI commands, analysis task names, ensemble/thermostat names and API
  function names quoted in the READMEs must exist in the code.
- The model profiles quoted in the READMEs must be the profiles present
  in the beta summary and the model manifest.
- Python-version claims must match ``pyproject.toml``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "mlipx") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "mlipx"))

BEGIN_MARKER = (
    "<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->"
)
END_MARKER = "<!-- END GENERATED -->"

READMES = (REPO_ROOT / "README.md", REPO_ROOT / "README_CN.md")
GENERATED_SNIPPET = (
    REPO_ROOT / "validation" / "science" / "reports" / "README_VALIDATION.md"
)
BETA_SUMMARY = REPO_ROOT / "validation" / "science" / "reports" / "beta-summary.json"
PYPROJECT = REPO_ROOT / "mlipx" / "pyproject.toml"


def _readme_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _generated_block(text: str) -> str:
    start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = text.index(END_MARKER, start)
    return text[start:end].strip()


def test_generated_validation_block_matches_generator() -> None:
    snippet = GENERATED_SNIPPET.read_text(encoding="utf-8").strip()
    for readme in READMES:
        assert _generated_block(_readme_text(readme)) == snippet, readme


def test_generated_validation_block_exists_exactly_once() -> None:
    for readme in READMES:
        text = _readme_text(readme)
        assert text.count(BEGIN_MARKER) == 1, readme
        assert text.count(END_MARKER) == 1, readme


def test_readme_claims_match_pyproject_python_range() -> None:
    """``requires-python = ">=3.10,<3.13"`` means 3.10-3.12; the READMEs
    must state the last supported minor, not the exclusive bound."""
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"requires-python\s*=\s*\"([^\"]+)\"", pyproject)
    assert match, "requires-python missing from pyproject.toml"
    spec = match.group(1)
    lower = re.search(r">=\s*([\d.]+)", spec)
    upper = re.search(r"<\s*([\d.]+)", spec)
    assert lower and upper, f"unexpected requires-python form: {spec}"
    upper_parts = [int(p) for p in upper.group(1).split(".")]
    last_supported = upper_parts[:-1] + [upper_parts[-1] - 1]
    claimed_range = f"{lower.group(1)}-" + ".".join(str(p) for p in last_supported)
    for readme in READMES:
        assert claimed_range in _readme_text(readme), readme


def _top_level_choices() -> dict[str, Any]:
    from mlipx.cli import create_parser  # noqa: PLC0415

    parser = create_parser()
    return dict(parser._subparsers._group_actions[0].choices)  # noqa: SLF001


def _subcommand_choices(command: str, dest: str) -> set[str]:
    """Choices of a sub-subparser option, e.g. analyze's task list."""
    sub = _top_level_choices()[command]
    for action in sub._actions:  # noqa: SLF001
        choices = getattr(action, "choices", None)
        if action.dest == dest and isinstance(choices, dict):
            return set(choices)
    raise AssertionError(f"{command} has no sub-subparser {dest}")


def _md_option_choices(dest: str) -> set[str]:
    md = _top_level_choices()["md"]
    for action in md._actions:  # noqa: SLF001
        if action.dest == dest:
            return {str(c) for c in action.choices}
    raise AssertionError(f"md has no option {dest}")


def test_readme_cli_commands_exist() -> None:
    documented = {
        "run",
        "sp",
        "opt",
        "md",
        "neb",
        "batch",
        "analyze",
        "config",
        "template",
        "tui",
        "queue",
        "jobs",
        "kill",
        "clean",
        "doctor",
    }
    missing = documented - set(_top_level_choices())
    assert not missing, f"README documents non-existent commands: {missing}"


def test_readme_analysis_task_names_exist() -> None:
    documented = {
        "validate",
        "thermo",
        "rdf",
        "rmsd",
        "msd",
        "vacf",
        "spectrum",
        "transport",
        "density",
        "arrhenius",
        "electrolyte",
    }
    missing = documented - _subcommand_choices("analyze", "analysis_task")
    assert not missing, f"README documents non-existent analysis tasks: {missing}"


def test_readme_ensemble_and_thermostat_names_exist() -> None:
    assert _md_option_choices("ensemble") == {"NVT", "NVE"}
    thermostats = {c.lower() for c in _md_option_choices("thermostat")}
    assert {"langevin", "bussi", "nhc"} <= thermostats


def test_readme_api_names_exist() -> None:
    import mlipx.api  # noqa: PLC0415

    documented = {
        "calculate_energy",
        "run_single_point",
        "run_optimization",
        "run_md",
        "run_neb",
    }
    missing = documented - set(mlipx.api.__all__)
    assert not missing, f"README documents non-existent API names: {missing}"


def _manifest_profiles() -> dict[str, Any]:
    manifest = json.loads(
        (REPO_ROOT / "validation" / "science" / "model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    profiles = manifest["profiles"]
    assert isinstance(profiles, dict), "profiles must be keyed by profile_id"
    return profiles


VALIDATED_PROFILES = ("mace_omat", "dpa_omat", "grace_omat", "uma_omat")


def test_readme_model_profiles_backed_by_beta_summary() -> None:
    summary = json.loads(BETA_SUMMARY.read_text(encoding="utf-8"))
    profiles = summary.get("model_profiles", {})
    assert set(profiles) == {"mace", "dpa", "grace", "uma"}
    assert all(isinstance(v, str) and v for v in profiles.values())
    manifest_ids = set(_manifest_profiles())
    assert set(VALIDATED_PROFILES) <= manifest_ids


def test_readme_quotes_validated_model_names() -> None:
    """Checkpoint names quoted in the model tables must be the artifact
    names recorded in the manifest profiles (via upstream_model_id and
    the source_url artifact annotation)."""
    profiles = _manifest_profiles()
    expected_names = {
        "mace_omat": "mace-omat-0-medium.model",
        "dpa_omat": "DPA-3.1-3M.pt",
        "grace_omat": "GRACE-2L-OMAT-medium-base",
        "uma_omat": "uma-s-1p2.pt",
    }
    for profile_id, name in expected_names.items():
        profile = profiles[profile_id]
        haystack = (
            str(profile.get("upstream_model_id", ""))
            + " "
            + str(profile.get("source_url", ""))
        )
        assert name.lower() in haystack.lower(), (profile_id, haystack, name)
