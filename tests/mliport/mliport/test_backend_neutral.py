"""PR2 acceptance tests: backend-neutral selection (ARCH-01/ARCH-02).

No implicit UMA default may exist anywhere: resolver, factory, EngineConfig,
INCAR template or TUI.  A backend must come from an explicit ``model_type``,
a model alias/profile, or the caller must get a clear fail-closed error.
The top-level API must not privilege UMA over the other backends.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mliport.calculators.factory import CalculatorFactory
from mliport.config.aliases import ModelAlias, parse_model_aliases
from mliport.config.defaults import build_incar_default
from mliport.config.resolver import resolve_config
from mliport.engine import EngineConfig

REPO = Path(__file__).resolve().parents[3]
_NO_BACKEND = "No MLIP backend was selected"


def _fake_model(tmp_path: Path, name: str = "model.bin") -> Path:
    model = tmp_path / name
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"metadata-only test model; never loaded")
    return model


def _incar_values(text: str) -> dict[str, str]:
    """Parse active ``KEY = VALUE`` lines from a written INCAR file."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.split("#", 1)[0].strip()
    return values


def _active_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


# --------------------------------------------------------------- RT-01
def test_factory_requires_an_explicit_backend(tmp_path):
    model = _fake_model(tmp_path)
    with pytest.raises(ValueError, match=_NO_BACKEND):
        CalculatorFactory.create(None, model, task="omat", device="cpu")
    with pytest.raises(ValueError, match=_NO_BACKEND):
        CalculatorFactory.create("", model, task="omat", device="cpu")


def test_factory_explicit_backends_and_fairchem_alias(tmp_path):
    from mliport.calculators.uma import UMACalculator

    model = _fake_model(tmp_path / "uma", "uma.pt")
    with patch.object(UMACalculator, "_validate"):
        uma = CalculatorFactory.create("uma", model, task="omat", device="cpu")
        alias = CalculatorFactory.create("fairchem", model, task="omat", device="cpu")
    assert isinstance(uma, UMACalculator)
    assert isinstance(alias, UMACalculator)

    mace_model = _fake_model(tmp_path / "mace", "m.model")
    from mliport.calculators.mace_calc import MACECalculatorWrapper

    with patch.object(MACECalculatorWrapper, "_validate", create=True):
        mace = CalculatorFactory.create("mace", mace_model, task="bulk", device="cpu")
    assert isinstance(mace, MACECalculatorWrapper)


# --------------------------------------------------------------- resolver
def test_resolver_requires_an_explicit_backend():
    with pytest.raises(ValueError, match=_NO_BACKEND):
        resolve_config(calc_type="sp")


def test_resolver_accepts_an_explicit_cli_backend():
    resolved = resolve_config(
        calc_type="sp",
        cli={"model_type": "mace", "task": "bulk", "model_path": "m.model"},
    )
    assert resolved.model_type == "mace"
    assert resolved.task == "bulk"


def test_resolver_model_alias_supplies_the_backend():
    alias = ModelAlias(name="m", engine="mace", path="m.model", task="bulk")
    resolved = resolve_config(
        calc_type="sp",
        model_aliases={"m": alias},
        model_alias_name="m",
    )
    assert resolved.model_type == "mace"


def test_model_alias_without_engine_is_rejected(tmp_path):
    settings = tmp_path / "settings.ini"
    settings.write_text(
        "[model:broken]\npath = m.model\n",
        encoding="utf-8",
    )
    import configparser

    parser = configparser.ConfigParser()
    parser.read(settings)
    with pytest.raises(ValueError, match="must declare engine"):
        parse_model_aliases(parser)


# ------------------------------------------------------------ EngineConfig
def test_engine_config_requires_an_explicit_backend():
    with pytest.raises(ValueError, match=_NO_BACKEND):
        EngineConfig(calc_type="sp", model_path=Path("model.bin"))


def test_engine_config_task_default_follows_the_backend():
    mace = EngineConfig(calc_type="sp", model_path=Path("m.model"), model_type="mace")
    assert mace.task == "bulk"
    uma = EngineConfig(calc_type="sp", model_path=Path("u.pt"), model_type="uma")
    assert uma.task == "omat"


# ---------------------------------------------------------------- RT-02
def test_template_without_engine_is_backend_neutral():
    text = build_incar_default("sp")
    active = _active_lines(text)
    assert "MODEL_TYPE = REQUIRED" in active
    assert "MODEL_PATH = REQUIRED" in active
    # no active (uncommented) backend or model path may be preselected
    assert not any(line.startswith("MODEL_TYPE = UMA") for line in active)
    assert not any(line.startswith("MODEL_PATH = uma") for line in active)
    lowered = text.lower()
    for engine in ("uma", "mace", "dpa", "grace"):
        assert engine in lowered, f"neutral template must show a {engine} example"


def test_template_engine_specific_fields():
    mace = build_incar_default("sp", engine="mace")
    assert "MODEL_TYPE = MACE" in mace
    assert "TASK = bulk" in mace
    dpa = build_incar_default("sp", engine="dpa")
    assert "MODEL_TYPE = DPA" in dpa
    assert "HEAD = Omat24" in dpa
    assert "TASK = bulk" in dpa
    grace = build_incar_default("sp", engine="grace")
    assert "MODEL_TYPE = GRACE" in grace
    assert "TASK = bulk" in grace
    uma = build_incar_default("sp", engine="uma")
    assert "MODEL_TYPE = UMA" in uma
    assert "TASK = omat" in uma


def test_cli_template_with_and_without_engine(tmp_path):
    from mliport.cli import main

    neutral = tmp_path / "INCAR.neutral"
    assert main(["template", "sp", "--output", str(neutral)]) == 0
    neutral_values = _incar_values(neutral.read_text(encoding="utf-8"))
    assert neutral_values["MODEL_TYPE"] == "REQUIRED"
    assert neutral_values["MODEL_PATH"] == "REQUIRED"

    mace = tmp_path / "INCAR.mace"
    assert main(["template", "sp", "--engine", "mace", "--output", str(mace)]) == 0
    mace_values = _incar_values(mace.read_text(encoding="utf-8"))
    assert mace_values["MODEL_TYPE"] == "MACE"
    assert mace_values["TASK"] == "bulk"


# ---------------------------------------------------------------- ARCH-02
def test_top_level_api_does_not_privilege_uma():
    """Inspect the real package without mutating sys.modules.

    Loading it under a private probe name avoids the repository-root
    namespace shadowing while keeping the test process clean.
    """
    import importlib.util

    init_path = REPO / "mliport" / "mliport" / "__init__.py"
    spec = importlib.util.spec_from_file_location("mliport_release_probe", init_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "UMACalculator" not in module.__all__
    with pytest.raises(AttributeError):
        module.UMACalculator  # noqa: B018


def test_calculators_package_exports_all_backends_symmetrically():
    from mliport.calculators import (
        DPACalculatorWrapper,
        GRACECalculatorWrapper,
        MACECalculatorWrapper,
        UMACalculator,
    )

    assert UMACalculator.__name__ == "UMACalculator"
    for wrapper in (
        MACECalculatorWrapper,
        DPACalculatorWrapper,
        GRACECalculatorWrapper,
    ):
        assert wrapper.__name__.endswith("CalculatorWrapper")


def test_legacy_calculator_module_is_a_compatibility_shim():
    from mliport.calculators import UMACalculator as LegacyUMA
    from mliport.calculators.uma import UMACalculator as NewUMA

    assert LegacyUMA is NewUMA
