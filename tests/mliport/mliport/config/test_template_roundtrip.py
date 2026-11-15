"""Template/run consistency contract (spec R3.2 of the #61 usage round).

Every ``mliport template <type>`` output, for every backend value, must be
accepted by the same validation and resolution that ``mliport run`` performs,
once the user has filled in the placeholders (``MODEL_PATH = REQUIRED`` and
friends). A template that generates a file the program then rejects is a bug:
the user should never have to discover the difference.

Tracking: #61.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from mliport.backend_selection import model_type_choices
from mliport.config.defaults import build_incar_default
from mliport.config.incar import IncarConfig
from mliport.config.resolver import resolve_config

TEMPLATE_TYPES = ("sp", "opt", "md", "neb", "confsearch")
BACKENDS = tuple(model_type_choices())

#: What a user types instead of ``REQUIRED``; every file exists, so a template
#: may not rely on a missing-model error to look consistent.
PLACEHOLDER_FILES = {
    "model.pt": "placeholder\n",
    "nep.txt": "placeholder\n",
    "starts.xyz": "1\nstart\nHe 0.0 0.0 0.0\n",
    "slab.vasp": "placeholder\n",
    "molecule.xyz": "1\nmol\nHe 0.0 0.0 0.0\n",
    "initial.vasp": "placeholder\n",
    "final.vasp": "placeholder\n",
}

#: Placeholder key -> the value a user fills in.
PLACEHOLDER_VALUES = {
    "MODEL_PATH": "model.pt",
    "CS_STARTS": "starts.xyz",
    "CS_ADSORBATE_INDICES": "0",
    "NEB_INITIAL": "initial.vasp",
    "NEB_FINAL": "final.vasp",
}


def _write_placeholder_files(folder: Path) -> None:
    for name, text in PLACEHOLDER_FILES.items():
        (folder / name).write_text(text, encoding="utf-8")


def _fill_placeholders(text: str) -> str:
    """Replace the template's ``REQUIRED`` placeholders with usable values."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if "=" not in stripped or stripped.startswith("#"):
            lines.append(line)
            continue
        key, _, rest = stripped.partition("=")
        key = key.strip().upper()
        value = rest.split("#", 1)[0].strip()
        if value != "REQUIRED":
            lines.append(line)
            continue
        if key not in PLACEHOLDER_VALUES:
            raise AssertionError(f"template placeholder {key} has no known value")
        lines.append(f"{key} = {PLACEHOLDER_VALUES[key]}")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("calc_type", TEMPLATE_TYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_template_round_trip_is_accepted_by_validation_and_resolution(
    tmp_path: Path, calc_type: str, backend: str
) -> None:
    _write_placeholder_files(tmp_path)
    filled = _fill_placeholders(build_incar_default(calc_type, engine=backend))
    incar_path = tmp_path / "INCAR.mliport"
    incar_path.write_text(filled, encoding="utf-8")

    config = IncarConfig.from_file(incar_path)
    assert config.validate() == []

    with warnings.catch_warnings():
        # An unknown key or a scope error is a hard failure, never a warning.
        warnings.simplefilter("error")
        resolved = resolve_config(calc_type=calc_type, incar=config)

    assert resolved.calc_type == calc_type
    assert Path(resolved.model_path).is_absolute()
    assert Path(resolved.model_path).parent == tmp_path


def test_templates_only_use_declared_placeholders() -> None:
    """The generator's placeholder vocabulary is explicit, not accidental."""
    seen: set[str] = set()
    for calc_type in TEMPLATE_TYPES:
        for backend in BACKENDS:
            for line in build_incar_default(calc_type, engine=backend).splitlines():
                stripped = line.strip()
                if "=" not in stripped or stripped.startswith("#"):
                    continue
                key, _, rest = stripped.partition("=")
                if rest.split("#", 1)[0].strip() == "REQUIRED":
                    seen.add(key.strip().upper())
    assert seen <= set(PLACEHOLDER_VALUES)
