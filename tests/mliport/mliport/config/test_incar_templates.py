"""INCAR template structure contract (#61 R3.1 / R3.4).

The template is generated from one option table (`mliport/config/schema.py`),
which also feeds `mliport config schema` and the generated key reference in
`docs/incar.md`. These tests pin the properties the specification attaches to
the template itself:

* fixed group order, each group appearing at most once,
* one `KEY = VALUE` line per key, each with a one-line English comment,
* only the keys of the selected calculation type,
* only the thermostat-specific keys of the selected thermostat,
* the NPT keys kept as a commented example in the MD template,
* the template/`config schema`/key-reference tables agreeing on the key set.
"""

from __future__ import annotations

import re

import pytest

from mliport.backend_selection import model_type_choices
from mliport.config.defaults import build_incar_default
from mliport.config.schema import (
    TEMPLATE_GROUP_ORDER,
    OptionSpec,
    get_schema,
)

CALC_TYPES = ("sp", "opt", "md", "neb", "confsearch")
ASSIGNMENT = re.compile(r"^(?P<key>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>.*?)\s*$")


def _active_lines(template: str) -> dict[str, str]:
    """Active (uncommented) ``KEY = VALUE`` lines, comments stripped."""
    values: dict[str, str] = {}
    for line in template.splitlines():
        if not line or line.startswith("#"):
            continue
        match = ASSIGNMENT.match(line)
        assert match, f"not an assignment: {line!r}"
        value = match.group("value").split("#", 1)[0].strip()
        values[match.group("key")] = value
    return values


def _group_headers(template: str) -> list[str]:
    """Section headers of a template, in the order they appear."""
    headers: list[str] = []
    for line in template.splitlines():
        name = line.removeprefix("# ").removesuffix(" Settings")
        if (
            line.startswith("# ")
            and line.endswith(" Settings")
            and name in (TEMPLATE_GROUP_ORDER)
        ):
            headers.append(name)
    return headers


@pytest.mark.parametrize("calc_type", CALC_TYPES)
@pytest.mark.parametrize("backend", list(model_type_choices()))
def test_each_group_appears_exactly_once_in_a_fixed_order(calc_type, backend) -> None:
    headers = _group_headers(build_incar_default(calc_type, engine=backend))
    assert len(headers) == len(set(headers)), headers
    order = [TEMPLATE_GROUP_ORDER.index(name) for name in headers]
    assert order == sorted(order), headers
    assert headers[0] == "Calculation"
    assert "Model" in headers


@pytest.mark.parametrize("calc_type", CALC_TYPES)
def test_every_key_has_a_one_line_comment(calc_type) -> None:
    """A user must never have to look a template key up to know what it is."""
    for line in build_incar_default(calc_type).splitlines():
        stripped = line.removeprefix("# ").strip()
        if not ASSIGNMENT.match(stripped):
            continue  # a comment, a section header or a blank line
        key = stripped.split("=", 1)[0].strip()
        if key in {"MODEL_TYPE", "MODEL_PATH"}:
            continue  # the model section carries its own explanatory block
        assert "#" in stripped, f"{calc_type}: {key} has no one-line comment"


def test_only_the_selected_thermostat_keys_are_written() -> None:
    template = build_incar_default("md")
    active = _active_lines(template)
    assert active["THERMOSTAT"] == "LANGEVIN"
    assert "FRICTION" in active
    for absent in ("BUSSI_TAU", "NHC_TDAMP", "NHC_TCHAIN", "NHC_TLOOP"):
        assert absent not in active
        assert f"# {absent}" not in template


def test_md_template_keeps_the_npt_keys_as_a_commented_example() -> None:
    template = build_incar_default("md")
    commented = {
        line.removeprefix("# ").split("=", 1)[0].strip()
        for line in template.splitlines()
        if line.startswith("# ") and "=" in line
    }
    assert {"MD_ENSEMBLE", "PRESSURE_GPA", "NPT_TDAMP", "NPT_PDAMP"} <= commented
    assert "# MD_ENSEMBLE = NPT" in template
    # ... and they are only an example: the active template stays NVT.
    assert _active_lines(template)["MD_ENSEMBLE"] == "NVT"
    assert "PRESSURE_GPA" not in _active_lines(template)


def test_template_only_contains_keys_of_its_calculation_type() -> None:
    md = set(_active_lines(build_incar_default("md")))
    opt = set(_active_lines(build_incar_default("opt")))
    neb = set(_active_lines(build_incar_default("neb")))
    confsearch = set(_active_lines(build_incar_default("confsearch")))
    assert {"TEMPERATURE", "THERMOSTAT", "STEPS"} <= md
    assert not {"TEMPERATURE"} & opt
    assert {"OPT_ALGO", "FMAX"} <= opt
    assert {"NEB_INITIAL", "NEB_FINAL", "NEB_IMAGES"} <= neb
    assert not {"WRITE_FORCES", "WRITE_STRESS"} & neb
    assert {"CS_STARTS", "CS_ADSORBATE_INDICES", "CS_TOP_K"} <= confsearch
    assert not {"TEMPERATURE", "THERMOSTAT"} & confsearch


def test_template_keys_and_documentation_rows_share_one_table() -> None:
    """`mliport config schema`, the templates and docs/incar.md use one table."""
    from pathlib import Path

    schema = get_schema()
    documented = set()
    docs = Path(__file__).parents[4] / "docs" / "incar.md"
    for key in re.findall(
        r"^\| `([A-Z][A-Z0-9_]*)` \|", docs.read_text(), re.MULTILINE
    ):
        documented.add(key)
    template_keys: set[str] = set()
    for calc_type in CALC_TYPES:
        for backend in model_type_choices():
            template_keys.update(
                _active_lines(build_incar_default(calc_type, engine=backend))
            )
            for line in build_incar_default(calc_type, engine=backend).splitlines():
                if not line.startswith("# "):
                    continue
                stripped = line.removeprefix("# ").strip()
                if ASSIGNMENT.match(stripped):
                    template_keys.add(stripped.split("=", 1)[0].strip())
    # ``CALCULATION`` is the documented alias used by the NEB template header.
    documented_keys = documented | {spec.incar_key for spec in schema.specs}
    documented_keys |= {
        alias for spec in schema.specs for alias in spec.aliases if alias.isupper()
    }
    unknown = sorted(key for key in template_keys if key not in documented_keys)
    assert not unknown, f"template keys missing from docs/incar.md: {unknown}"
    # ... and the reverse direction: every templated option is documented.
    templated = {spec.incar_key for spec in schema.specs if spec.template}
    missing = sorted(key for key in templated if key not in documented_keys)
    assert not missing, f"templated keys missing from docs/incar.md: {missing}"
    # and the reverse: every template key is a real option spelling
    known = {spec.incar_key for spec in schema.specs}
    known |= {
        alias for spec in schema.specs for alias in spec.aliases if alias.isupper()
    }
    assert template_keys <= known


def test_spec_cardinality_metadata_is_consistent() -> None:
    """Every templated key is grouped; every key is documented at most once."""
    schema = get_schema()
    for spec in schema.specs:
        assert isinstance(spec, OptionSpec)
        assert spec.group in TEMPLATE_GROUP_ORDER
        if spec.template:
            assert spec.description, f"{spec.name} is templated without a comment"
    groups = [spec.group for spec in schema.specs]
    assert all(groups), "every option belongs to a documented section"
