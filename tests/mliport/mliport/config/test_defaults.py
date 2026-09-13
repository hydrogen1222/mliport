"""Tests for mliport.config.defaults (single source of truth)."""

from __future__ import annotations

import pytest

from mliport.config.defaults import (
    BUILTIN_DEFAULTS,
    DEFAULT_DEVICE_BY_CALC_TYPE,
    build_incar_default,
    get_default,
    get_default_config,
)

# ---------------------------------------------------------------------------
# Built-in structure
# ---------------------------------------------------------------------------


def test_all_expected_scopes_present() -> None:
    expected = {
        "general",
        "resources",
        "batch",
        "output",
        "safety",
        "sp",
        "opt",
        "md",
        "neb",
        "calculator",
        "calculator.uma",
        "calculator.mace",
        "calculator.dpa",
        "calculator.grace",
    }
    actual = set(BUILTIN_DEFAULTS)
    missing = expected - actual
    assert not missing, f"Missing scopes: {missing}"


def test_no_inference_mode_in_calculator_scope() -> None:
    """inference_mode is model-level, not in the generic calculator scope."""
    assert "inference_mode" not in BUILTIN_DEFAULTS["calculator"]


def test_inference_mode_in_calc_type_scopes() -> None:
    assert BUILTIN_DEFAULTS["sp"]["inference_mode"] == "default"
    assert BUILTIN_DEFAULTS["opt"]["inference_mode"] == "default"
    assert BUILTIN_DEFAULTS["md"]["inference_mode"] == "turbo"
    assert BUILTIN_DEFAULTS["neb"]["inference_mode"] == "default"


def test_device_per_calc_type() -> None:
    assert DEFAULT_DEVICE_BY_CALC_TYPE["sp"] == "cpu"
    assert DEFAULT_DEVICE_BY_CALC_TYPE["opt"] == "cpu"
    assert DEFAULT_DEVICE_BY_CALC_TYPE["md"] == "cuda"
    assert DEFAULT_DEVICE_BY_CALC_TYPE["neb"] == "cpu"


def test_mace_default_dtype_is_float64() -> None:
    """Accuracy-first MACE default is float64."""
    assert BUILTIN_DEFAULTS["calculator.mace"]["default_dtype"] == "float64"


def test_md_thermostat_defaults_preserve_legacy_langevin() -> None:
    md = BUILTIN_DEFAULTS["md"]
    assert md["thermostat"] == "LANGEVIN"
    assert md["friction"] == 0.001
    assert md["bussi_tau"] == 1000.0
    assert md["nhc_tdamp"] == 100.0
    assert md["nhc_tchain"] == 3
    assert md["nhc_tloop"] == 1
    assert md["com_policy"] == "auto"


# ---------------------------------------------------------------------------
# get_default helper
# ---------------------------------------------------------------------------


def test_get_default_calc_type_scoped() -> None:
    assert get_default("opt", "fmax") == 0.05


def test_get_default_fallback_to_general() -> None:
    # scientific config is strict by default (CFG-01)
    assert get_default("opt", "strict_config") is True
    assert get_default("md", "strict_config") is True


def test_get_default_unknown_returns_fallback() -> None:
    assert get_default("sp", "nonexistent", "fallback") == "fallback"


def test_get_default_no_calc_type() -> None:
    assert get_default(None, "strict_config") is True


# ---------------------------------------------------------------------------
# INCAR template generation
# ---------------------------------------------------------------------------


def test_build_incar_default_sp() -> None:
    neutral = build_incar_default("sp")
    assert "CALC_TYPE = SP" in neutral
    assert "DEVICE = cpu" in neutral
    # no implicit backend or UMA-only inference mode in the neutral template
    assert "MODEL_TYPE = REQUIRED" in neutral
    assert "INFERENCE_MODE" not in neutral

    uma = build_incar_default("sp", engine="uma")
    assert "MODEL_TYPE = UMA" in uma
    assert "TASK = omat" in uma
    assert "INFERENCE_MODE = default" in uma


def test_build_incar_default_md() -> None:
    neutral = build_incar_default("md")
    assert "CALC_TYPE = MD" in neutral
    assert "DEVICE = cuda" in neutral
    assert "MODEL_TYPE = REQUIRED" in neutral

    uma = build_incar_default("md", engine="uma")
    assert "INFERENCE_MODE = turbo" in uma


def test_build_incar_default_neb() -> None:
    text = build_incar_default("neb")
    assert "CALCULATION = NEB" in text
    assert "NEB_INITIAL = initial.vasp" in text
    assert "NEB_FINAL = final.vasp" in text
    assert "NEB_IMAGES = 7" in text
    assert "DEVICE = cpu" in text
    assert "WRITE_STRESS" not in text


def test_build_incar_default_invalid_type() -> None:
    with pytest.raises(ValueError, match="Unknown calculation type"):
        build_incar_default("nonexistent")


def test_get_default_config_returns_incar() -> None:
    config = get_default_config("sp")
    assert config.get_str("CALC_TYPE", "").lower() == "sp"
