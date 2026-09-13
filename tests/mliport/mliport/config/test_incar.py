"""Tests for mliport.config.incar (INCAR parsing and validation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mliport.config.incar import IncarConfig


def test_parser_preserves_raw_tokens_until_schema_coercion() -> None:
    cfg = IncarConfig.from_string(
        "CALC_TYPE = SP\n"
        "FMAX = 0.05\n"
        "STEPS = 1000\n"
        "CELL_OPT = .TRUE.\n"
        "MODEL_PATH = uma-s-1.pt\n"
    )
    assert cfg["CALC_TYPE"] == "SP"
    assert cfg["FMAX"] == "0.05"
    assert cfg["STEPS"] == "1000"
    assert cfg["CELL_OPT"] == ".TRUE."
    assert cfg["MODEL_PATH"] == "uma-s-1.pt"
    assert cfg.get_float("FMAX") == 0.05
    assert cfg.get_int("STEPS") == 1000
    assert cfg.get_bool("CELL_OPT") is True


def test_numeric_zero_and_one_are_not_guessed_as_booleans() -> None:
    cfg = IncarConfig.from_string("HEAD = 0\nCPU_THREADS = 1\n")
    assert cfg["HEAD"] == "0"
    assert cfg["CPU_THREADS"] == "1"


def test_inline_comments_only_start_outside_quotes() -> None:
    cfg = IncarConfig.from_string(
        'MODEL_PATH = "models/a#b!c.model" # external comment\n'
        "HEAD = 'branch!with#marks' ! another comment\n"
    )
    assert cfg["MODEL_PATH"] == "models/a#b!c.model"
    assert cfg["HEAD"] == "branch!with#marks"


def test_quoted_comment_markers_round_trip_through_writer() -> None:
    original = IncarConfig(
        {
            "MODEL_PATH": "models/a#b!c.model",
            "HEAD": 'branch"one;two',
            "JOB_NAME": " leading and trailing ",
        }
    )
    restored = IncarConfig.from_string(original.to_string())
    assert dict(restored) == dict(original)


@pytest.mark.parametrize(
    "content",
    [
        "STEPS = 1; TEMPERATURE = 300\n",
        'MODEL_PATH = "unterminated\n',
        "THIS IS NOT AN ASSIGNMENT\n",
    ],
)
def test_unsupported_or_malformed_syntax_fails_closed(content: str) -> None:
    with pytest.raises(ValueError):
        IncarConfig.from_string(content)


def test_file_parser_records_path_line_and_base_dir(tmp_path: Path) -> None:
    incar = tmp_path / "inputs" / "INCAR.mliport"
    incar.parent.mkdir()
    incar.write_text("# header\nMODEL_PATH = models/a.pt\n", encoding="utf-8")

    cfg = IncarConfig.from_file(incar)

    origin = cfg.origin("MODEL_PATH")
    assert origin is not None
    assert origin.path == incar.resolve()
    assert origin.line == 2
    assert origin.base_dir == incar.parent.resolve()


def test_validate_accepts_supported_calc_types() -> None:
    for ct in ("SP", "OPT", "MD", "NEB", "BATCH"):
        cfg = IncarConfig.from_string(f"CALC_TYPE = {ct}\nMODEL_PATH = x.pt\n")
        assert cfg.validate() == [], f"{ct} should validate"


def test_validate_rejects_unsupported_calc_types() -> None:
    """Regression: phonon/analyze passed INCAR validation but crashed at
    engine construction with a late 'Unknown calc_type' error."""
    for ct in ("PHONON", "ANALYZE", "BOGUS"):
        cfg = IncarConfig.from_string(f"CALC_TYPE = {ct}\nMODEL_PATH = x.pt\n")
        errors = cfg.validate()
        assert len(errors) == 1, f"{ct}: {errors}"
        # Error message must quote the value properly (missing closing quote bug).
        assert f"Invalid CALC_TYPE '{ct.lower()}'. " in errors[0], errors[0]


def test_validate_accepts_calculation_alias_and_rejects_conflict() -> None:
    cfg = IncarConfig.from_string("CALCULATION = NEB\nMODEL_PATH = x.pt\n")
    assert cfg.validate() == []

    conflict = IncarConfig.from_string("CALC_TYPE = SP\nCALCULATION = NEB\n")
    assert any("Conflicting" in error for error in conflict.validate())


def test_validate_rejects_unsupported_optimizers() -> None:
    """Regression: gpmin/mdmin passed INCAR validation but raised at runner
    construction (OptimizationRunner only implements FIRE/BFGS/LBFGS)."""
    for algo in ("GPMIN", "MDMIN", "BOGUS"):
        cfg = IncarConfig.from_string(f"OPT_ALGO = {algo}\n")
        errors = cfg.validate()
        assert len(errors) == 1, f"{algo}: {errors}"
        assert f"Invalid OPT_ALGO '{algo.lower()}'. " in errors[0], errors[0]


def test_validate_accepts_supported_optimizers() -> None:
    for algo in ("FIRE", "BFGS", "LBFGS"):
        cfg = IncarConfig.from_string(f"OPT_ALGO = {algo}\n")
        assert cfg.validate() == []


def test_validate_ensemble_error_message_quotes_value() -> None:
    cfg = IncarConfig.from_string("MD_ENSEMBLE = NPT\n")
    errors = cfg.validate()
    assert len(errors) == 1
    assert "Invalid MD_ENSEMBLE 'npt'. " in errors[0], errors[0]


def test_validate_md_thermostat_names() -> None:
    for thermostat in ("LANGEVIN", "BUSSI", "NHC"):
        cfg = IncarConfig.from_string(f"THERMOSTAT = {thermostat}\n")
        assert cfg.validate() == []
    errors = IncarConfig.from_string("THERMOSTAT = BERENDSEN\n").validate()
    assert "Invalid THERMOSTAT 'berendsen'" in errors[0]


def test_validate_invalid_task() -> None:
    cfg = IncarConfig.from_string("TASK = not_a_task\n")
    errors = cfg.validate()
    assert len(errors) == 1
    assert "Invalid TASK 'not_a_task'" in errors[0]


def test_validate_invalid_device() -> None:
    cfg = IncarConfig.from_string("DEVICE = quantum\n")
    errors = cfg.validate()
    assert len(errors) == 1
    assert "Invalid DEVICE 'quantum'" in errors[0]


def test_validate_cuda_n_device_accepted() -> None:
    cfg = IncarConfig.from_string("DEVICE = cuda:3\n")
    assert cfg.validate() == []


def test_validate_invalid_model_type() -> None:
    cfg = IncarConfig.from_string("MODEL_TYPE = alpaca\n")
    errors = cfg.validate()
    assert len(errors) == 1
    assert "Invalid MODEL_TYPE 'alpaca'" in errors[0]


def test_validate_invalid_dtype() -> None:
    cfg = IncarConfig.from_string("DEFAULT_DTYPE = float16\n")
    errors = cfg.validate()
    assert len(errors) == 1
    assert "Invalid DEFAULT_DTYPE 'float16'" in errors[0]


def test_validate_multiple_errors_collected() -> None:
    cfg = IncarConfig.from_string("CALC_TYPE = PHONON\nMD_ENSEMBLE = NPT\n")
    errors = cfg.validate()
    assert len(errors) == 2
