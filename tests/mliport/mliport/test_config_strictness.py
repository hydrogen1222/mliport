"""PR2b acceptance tests: scientific config is strict by default (CFG-01).

A misspelled or cross-backend option must be a fatal configuration error.
The historical warning-only behaviour remains available as an explicit
opt-in (`strict_config = false` / `--lenient-config`), never as the default.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mliport.calculators.factory import CalculatorFactory
from mliport.cli import _build_cli_opts, create_parser
from mliport.config.defaults import BUILTIN_DEFAULTS
from mliport.config.resolver import resolve_config
from mliport.engine import EngineConfig


def test_strict_config_is_the_builtin_default():
    assert BUILTIN_DEFAULTS["general"]["strict_config"] is True
    config = EngineConfig(calc_type="sp", model_path=Path("m.model"), model_type="mace")
    assert config.strict_config is True


def test_resolver_unknown_key_is_fatal_by_default():
    with pytest.raises(ValueError, match="Strict config validation failed"):
        resolve_config(
            calc_type="sp",
            cli={
                "model_type": "mace",
                "model_path": "m.model",
                "task": "bulk",
                "not_a_real_key": 1,
            },
        )


def test_resolver_lenient_opt_in_warns_instead_of_failing():
    with pytest.warns(UserWarning, match="Unknown key"):
        resolved = resolve_config(
            calc_type="sp",
            cli={
                "model_type": "mace",
                "model_path": "m.model",
                "task": "bulk",
                "not_a_real_key": 1,
                "strict_config": False,
            },
        )
    assert "not_a_real_key" in resolved.unknown_options
    assert resolved.strict is False


def test_resolver_cross_backend_option_is_fatal_by_default():
    with pytest.raises(ValueError, match="default_dtype"):
        resolve_config(
            calc_type="sp",
            cli={
                "model_type": "dpa",
                "model_path": "m.pt",
                "task": "bulk",
                "head": "Omat24",
                "default_dtype": "float32",
            },
        )


def test_factory_unknown_and_cross_backend_keys_are_strict(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"metadata-only test model; never loaded")
    from mliport.calculators.dpa_calc import DPACalculatorWrapper

    with pytest.raises(ValueError, match="default_dtype"):
        CalculatorFactory.create(
            "dpa", model, task="bulk", strict=True, default_dtype="float32"
        )
    with pytest.raises(ValueError, match="Unknown calculator option"):
        CalculatorFactory.create(
            "dpa", model, task="bulk", strict=True, definitly_typo=1
        )
    # explicit lenient opt-in keeps the legacy warn-and-drop behaviour
    with (
        pytest.warns(UserWarning, match="not applicable"),
        patch.object(DPACalculatorWrapper, "_validate", create=True),
    ):
        CalculatorFactory.create(
            "dpa", model, task="bulk", strict=False, default_dtype="float32"
        )


def test_cli_lenient_config_flag_is_wired_into_the_resolver_layer():
    parser = create_parser()
    strict = parser.parse_args(
        ["sp", "s.cif", "--model", "m.pt", "--model-type", "mace"]
    )
    assert "strict_config" not in _build_cli_opts(strict, "sp")
    lenient = parser.parse_args(
        [
            "sp",
            "s.cif",
            "--model",
            "m.pt",
            "--model-type",
            "mace",
            "--lenient-config",
        ]
    )
    assert _build_cli_opts(lenient, "sp")["strict_config"] is False
