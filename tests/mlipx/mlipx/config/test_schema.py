"""Tests for mlipx.config.schema (OptionSpec registry and validation)."""

from __future__ import annotations

import pytest

from mlipx.config.schema import Schema, get_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schema() -> Schema:
    return get_schema()


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------


def test_schema_is_singleton(schema: Schema) -> None:
    assert get_schema() is schema


def test_schema_has_expected_specs(schema: Schema) -> None:
    """At minimum every key used by resolver/CLI must be registered."""
    names = schema.known_names()
    # Model-level keys
    for key in ("model_type", "model_path", "task", "device", "inference_mode"):
        assert key in names, f"missing core key {key!r}"
    # Run options
    for key in (
        "fmax",
        "max_steps",
        "optimizer",
        "temperature",
        "steps",
        "charge",
        "spin",
    ):
        assert key in names, f"missing run key {key!r}"
    # Calculator keys
    for key in ("default_dtype", "head"):
        assert key in names, f"missing calc key {key!r}"


# ---------------------------------------------------------------------------
# Typo suggestions
# ---------------------------------------------------------------------------


def test_suggest_exact_match(schema: Schema) -> None:
    assert "temperature" in schema.suggest("temperature")


def test_suggest_close_match(schema: Schema) -> None:
    suggestions = schema.suggest("temporature")
    assert "temperature" in suggestions


def test_suggest_case_insensitive(schema: Schema) -> None:
    suggestions = schema.suggest("TEMPERATURE")
    assert "temperature" in suggestions


def test_suggest_no_match(schema: Schema) -> None:
    assert schema.suggest("xyznonexistent") == []


# ---------------------------------------------------------------------------
# Canonical name resolution
# ---------------------------------------------------------------------------


def test_canonical_name_lowercase(schema: Schema) -> None:
    assert schema.canonical_name("model_type") == "model_type"


def test_canonical_name_alias(schema: Schema) -> None:
    """MODEL_TYPE is a registered alias -> model_type."""
    assert schema.canonical_name("MODEL_TYPE") == "model_type"
    assert schema.canonical_name("FMAX") == "fmax"


def test_canonical_name_unknown(schema: Schema) -> None:
    """Unknown key returns None (no canonical name registered)."""
    result = schema.canonical_name("unregistered_key")
    assert result is None or result == "unregistered_key"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def test_validate_known_key(schema: Schema) -> None:
    spec = schema.resolve("temperature")
    assert spec is not None
    assert spec.type is float


def test_validate_unknown_key(schema: Schema) -> None:
    assert schema.resolve("nonexistent_key_123") is None


def test_choices_constraint(schema: Schema) -> None:
    """inference_mode only accepts 'default' or 'turbo'."""
    spec = schema.resolve("inference_mode")
    assert spec is not None
    assert spec.choices == ("default", "turbo")


def test_device_is_string(schema: Schema) -> None:
    spec = schema.resolve("device")
    assert spec is not None
    assert spec.type is str


def test_schema_directed_zero_and_one_coercion(schema: Schema) -> None:
    head = schema.resolve("HEAD")
    threads = schema.resolve("CPU_THREADS")
    enabled = schema.resolve("WRITE_FORCES")
    assert head is not None and head.coerce("0") == "0"
    assert threads is not None and threads.coerce("1") == 1
    assert enabled is not None and enabled.coerce("0") is False


@pytest.mark.parametrize("value", [True, 3.7, "3.7"])
def test_integer_options_reject_bool_and_fractional_values(
    schema: Schema, value: object
) -> None:
    steps = schema.resolve("STEPS")
    assert steps is not None
    with pytest.raises(ValueError, match="expects int"):
        steps.coerce(value)


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-Inf", float("nan")])
def test_numeric_options_reject_nonfinite_values(schema: Schema, value: object) -> None:
    temperature = schema.resolve("TEMPERATURE")
    assert temperature is not None
    with pytest.raises(ValueError, match="finite"):
        temperature.coerce(value)


# ---------------------------------------------------------------------------
# Scope classification
# ---------------------------------------------------------------------------


def test_calc_scoped_keys_present(schema: Schema) -> None:
    """Keys scoped to specific calc types should be registered."""
    for key in ("fmax", "temperature", "ensemble", "sub_calc_type"):
        spec = schema.resolve(key)
        assert spec is not None, f"{key} not registered"


def test_calc_type_choices_match_engine(schema: Schema) -> None:
    """Regression: 'analyze' was a schema choice but the engine rejects it.
    The schema, IncarConfig.validate and CalculationEngine.VALID_CALC_TYPES
    must all agree on {sp, opt, md, batch}."""
    spec = schema.resolve("calc_type")
    assert spec is not None
    assert set(spec.choices) == {"sp", "opt", "md", "batch"}
