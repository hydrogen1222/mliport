"""Tests for mlipx.config.resolver (layered config resolution)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mlipx.config.incar import IncarConfig
from mlipx.config.resolver import ResolvedValue, resolve_config
from mlipx.config.settings import load_settings

# ---------------------------------------------------------------------------
# Basic resolution (built-in only)
# ---------------------------------------------------------------------------


def test_builtin_sp_defaults() -> None:
    rc = resolve_config(calc_type="sp")
    assert rc.calc_type == "sp"
    assert rc.model_type == "uma"
    assert rc.device == "cpu"
    assert rc.inference_mode == "default"
    # model_type is a model-level attribute, not in sources dict
    assert rc.model_type == "uma"


def test_builtin_md_defaults() -> None:
    rc = resolve_config(calc_type="md")
    assert rc.model_type == "uma"
    assert rc.device == "cuda"
    assert rc.inference_mode == "turbo"
    # MD auto-seeds
    assert "seed" in rc.run_options
    assert rc.sources["seed"].source == "auto-generated"
    assert rc.run_options["thermostat"] == "LANGEVIN"
    assert rc.run_options["friction"] == 0.001


def test_md_thermostat_options_are_run_options_only() -> None:
    values = {
        "thermostat": "NHC",
        "friction": 0.002,
        "bussi_tau": 800.0,
        "nhc_tdamp": 120.0,
        "nhc_tchain": 4,
        "nhc_tloop": 2,
    }
    for model_type in ("uma", "mace", "dpa", "grace"):
        rc = resolve_config(
            calc_type="md",
            cli={"model_type": model_type, **values},
        )
        assert values.items() <= rc.run_options.items()
        assert set(values).isdisjoint(rc.calculator_options)


def test_builtin_opt_defaults() -> None:
    rc = resolve_config(calc_type="opt")
    assert rc.run_options.get("fmax") == 0.05
    assert rc.run_options.get("optimizer") == "FIRE"
    assert rc.run_options.get("max_steps") == 500


# ---------------------------------------------------------------------------
# Source tracking
# ---------------------------------------------------------------------------


def test_source_tracking_cli_override() -> None:
    rc = resolve_config(calc_type="sp", cli={"device": "cuda:0"})
    assert rc.device == "cuda:0"
    assert rc.sources["device"] == ResolvedValue("cuda:0", "CLI")


def test_molecular_charge_and_spin_are_run_options() -> None:
    rc = resolve_config(
        calc_type="md",
        cli={"task": "omol", "charge": -1, "spin": 2},
    )
    assert rc.run_options["charge"] == -1
    assert rc.run_options["spin"] == 2
    assert rc.sources["charge"] == ResolvedValue(-1, "CLI")
    assert rc.sources["spin"] == ResolvedValue(2, "CLI")


def test_source_tracking_settings_override() -> None:
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text("[general]\ndevice = cuda:0\n")
        s = load_settings(explicit=str(ini))
        rc = resolve_config(calc_type="sp", settings=s)
        assert rc.device == "cuda:0"
        source = rc.sources["device"]
        assert source.source == "settings.ini"


def test_source_tracking_cli_beats_settings() -> None:
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text("[general]\ndevice = cuda:0\n")
        s = load_settings(explicit=str(ini))
        rc = resolve_config(calc_type="sp", settings=s, cli={"device": "cpu"})
        assert rc.device == "cpu"
        assert rc.sources["device"] == ResolvedValue("cpu", "CLI")


def test_only_current_calculation_section_is_loaded() -> None:
    """[opt] must not leak into MD, and [md] must not leak into OPT."""
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text("[md]\ntemperature = 650\n" "[opt]\nmax_steps = 17\n")
        settings = load_settings(explicit=str(ini))
        md = resolve_config(calc_type="md", settings=settings)
        opt = resolve_config(calc_type="opt", settings=settings)
        assert md.run_options["temperature"] == 650
        assert "max_steps" not in md.run_options
        assert opt.run_options["max_steps"] == 17
        assert "temperature" not in opt.run_options


def test_cli_selected_engine_uses_matching_engine_defaults() -> None:
    """CLI model_type is known before the settings-level engine block is read."""
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text(
            "[engine:mace]\n"
            "task = bulk\n"
            "device = cuda:1\n"
            "default_dtype = float64\n"
        )
        settings = load_settings(explicit=str(ini))
        rc = resolve_config(
            calc_type="sp",
            settings=settings,
            cli={"model_type": "mace", "model_path": "m.model"},
        )
        assert rc.task == "bulk"
        assert rc.device == "cuda:1"
        assert rc.calculator_options["default_dtype"] == "float64"


def test_grace_gpu_memory_policy_is_a_calculator_option() -> None:
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text(
            "[engine:grace]\n"
            "task = bulk\n"
            "gpu_memory_growth = true\n"
            "gpu_memory_limit_mb = 6144\n"
        )
        settings = load_settings(explicit=str(ini))
        rc = resolve_config(
            calc_type="md",
            settings=settings,
            cli={"model_type": "grace", "model_path": "grace_model"},
        )

        assert rc.calculator_options["gpu_memory_growth"] is True
        assert rc.calculator_options["gpu_memory_limit_mb"] == 6144
        assert "gpu_memory_limit_mb" not in rc.run_options


def test_engine_builtins_are_materialized_in_resolved_config() -> None:
    mace = resolve_config(
        calc_type="md", cli={"model_type": "mace", "model_path": "m.model"}
    )
    grace = resolve_config(
        calc_type="md", cli={"model_type": "grace", "model_path": "grace"}
    )

    assert mace.calculator_options["default_dtype"] == "float64"
    assert mace.sources["default_dtype"].source == (
        "built-in defaults (calculator.mace)"
    )
    assert mace.inference_mode == "default"
    assert grace.calculator_options["neighbor_cache"] is True
    assert grace.calculator_options["neighbor_skin"] == 1.5
    assert grace.calculator_options["gpu_memory_growth"] is True
    assert grace.inference_mode == "default"


def test_non_uma_turbo_mode_fails_instead_of_being_ignored() -> None:
    with pytest.raises(ValueError, match="UMA-only"):
        resolve_config(
            calc_type="md",
            cli={
                "model_type": "dpa",
                "model_path": "dpa.pt",
                "inference_mode": "turbo",
            },
        )


# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------


def test_model_alias_sets_engine_and_task() -> None:
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text(
            "[model:mace_test]\n"
            "engine = mace\n"
            "path = ./fake.model\n"
            "task = bulk\n"
            "dtype = float64\n"
        )
        s = load_settings(explicit=str(ini))
        rc = resolve_config(calc_type="sp", settings=s, model_alias_name="mace_test")
        assert rc.model_type == "mace"
        assert rc.task == "bulk"
        assert rc.calculator_options.get("default_dtype") == "float64"


def test_model_alias_path_resolves_relative_to_ini() -> None:
    with tempfile.TemporaryDirectory() as d:
        model_file = Path(d) / "fake.model"
        model_file.touch()
        ini = Path(d) / "settings.ini"
        ini.write_text(
            "[model:mace_mpa0]\n"
            "engine = mace\n"
            "path = ./fake.model\n"
            "task = bulk\n"
        )
        s = load_settings(explicit=str(ini))
        rc = resolve_config(calc_type="sp", settings=s, model_alias_name="mace_mpa0")
        assert rc.model_path == str(model_file.resolve())


def test_model_alias_path_uses_declaring_file_not_highest_priority_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_dir = tmp_path / "user"
    project_dir = tmp_path / "project"
    user_dir.mkdir()
    project_dir.mkdir()
    user_settings = user_dir / "settings.ini"
    project_settings = project_dir / "settings.ini"
    user_settings.write_text(
        "[model:shared]\nengine = mace\npath = models/model.pt\ntask = bulk\n",
        encoding="utf-8",
    )
    project_settings.write_text("[general]\ndevice = cpu\n", encoding="utf-8")
    monkeypatch.setenv("MLIPX_SETTINGS", str(user_settings))

    settings = load_settings(explicit=project_settings, cwd=project_dir)
    rc = resolve_config(calc_type="sp", settings=settings, model_alias_name="shared")

    assert rc.model_path == str((user_dir / "models/model.pt").resolve())
    assert rc.sources["model_path"].base_dir == str(user_dir.resolve())


def test_incar_types_and_relative_model_path_follow_schema(tmp_path: Path) -> None:
    incar_path = tmp_path / "input" / "INCAR.mlipx"
    incar_path.parent.mkdir()
    incar_path.write_text(
        "MODEL_TYPE = MACE\n"
        "MODEL_PATH = models/mace.model\n"
        "HEAD = 0\n"
        "CPU_THREADS = 1\n"
        "FMAX = 1D-3\n",
        encoding="utf-8",
    )
    incar = IncarConfig.from_file(incar_path)

    rc = resolve_config(calc_type="opt", incar=incar)

    assert rc.model_path == str((incar_path.parent / "models/mace.model").resolve())
    assert rc.calculator_options["head"] == "0"
    assert rc.settings["torch_num_threads"] == 1
    assert rc.run_options["fmax"] == pytest.approx(1.0e-3)
    assert rc.sources["model_path"].location == f"{incar_path.resolve()}:2"


def test_cli_relative_model_path_uses_explicit_cli_base(tmp_path: Path) -> None:
    cli_dir = tmp_path / "cli"
    settings_dir = tmp_path / "settings"
    cli_dir.mkdir()
    settings_dir.mkdir()
    settings_path = settings_dir / "settings.ini"
    settings_path.write_text("[general]\ndevice = cpu\n", encoding="utf-8")

    rc = resolve_config(
        calc_type="sp",
        settings=load_settings(explicit=settings_path, cwd=cli_dir),
        cli={"model_path": "models/cli.pt"},
        cli_base_dir=cli_dir,
    )

    assert rc.model_path == str((cli_dir / "models/cli.pt").resolve())
    assert rc.sources["model_path"].base_dir == str(cli_dir.resolve())


def test_strict_unknown_key_never_becomes_known_settings() -> None:
    incar = IncarConfig.from_string("STRICT_CONFIG = true\nTEMPRATURE = 999\n")
    with pytest.raises(ValueError, match="TEMPRATURE|temprature"):
        resolve_config(calc_type="md", incar=incar)


@pytest.mark.parametrize("value", ["NaN", "Inf", "-Inf"])
def test_resolver_rejects_nonfinite_physical_values(value: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        resolve_config(calc_type="md", incar={"TEMPERATURE": value})


def test_cli_dtype_overrides_alias() -> None:
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text(
            "[model:mace_test]\n"
            "engine = mace\n"
            "path = ./fake.model\n"
            "dtype = float32\n"
        )
        s = load_settings(explicit=str(ini))
        # CLI dtype=float64 overrides alias dtype=float32
        rc = resolve_config(
            calc_type="sp",
            settings=s,
            model_alias_name="mace_test",
            cli={"default_dtype": "float64"},
        )
        assert rc.calculator_options.get("default_dtype") == "float64"
        assert rc.sources["default_dtype"] == ResolvedValue("float64", "CLI")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def test_profile_applies_overrides() -> None:
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text("[profile:gpu_prod]\n" "device = cuda:1\n" "max_steps = 1000\n")
        s = load_settings(explicit=str(ini))
        rc = resolve_config(calc_type="opt", settings=s, profile_name="gpu_prod")
        assert rc.device == "cuda:1"
        assert rc.run_options.get("max_steps") == 1000


# ---------------------------------------------------------------------------
# Settings bag - non-run keys don't pollute run_options
# ---------------------------------------------------------------------------


def test_output_keys_land_in_settings() -> None:
    """INCAR output keys (WRITE_FORCES, etc.) go to settings, not run_options."""
    rc = resolve_config(calc_type="sp", incar={"WRITE_FORCES": True})
    assert "write_forces" not in rc.run_options
    assert rc.settings.get("write_forces") is True


def test_unsupported_output_format_fails_closed() -> None:
    with pytest.raises(ValueError, match="only VASP"):
        resolve_config(calc_type="sp", incar={"OUTPUT_FORMAT": "xyz"})


# ---------------------------------------------------------------------------
# as_dict / JSON roundtrip
# ---------------------------------------------------------------------------


def test_as_dict_is_serializable() -> None:
    rc = resolve_config(calc_type="md", cli={"temperature": 500.0})
    d = rc.as_dict()
    assert d["calc_type"] == "md"
    assert d["run_options"]["temperature"] == 500.0
    # Must be JSON-serializable (ResolvedValue converted via default=str)
    json.dumps(d, default=str)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_cli_is_noop() -> None:
    rc_default = resolve_config(calc_type="sp")
    rc_empty = resolve_config(calc_type="sp", cli={})
    assert rc_default.device == rc_empty.device
    assert rc_default.inference_mode == rc_empty.inference_mode


def test_none_cli_is_noop() -> None:
    rc = resolve_config(calc_type="sp", cli=None)
    assert rc.device == "cpu"


def test_batch_calc_type() -> None:
    rc = resolve_config(calc_type="batch")
    assert rc.calc_type == "batch"
    assert rc.device == "cpu"


def test_sp_no_auto_seed() -> None:
    """Only MD auto-generates a seed."""
    rc = resolve_config(calc_type="sp")
    assert "seed" not in rc.run_options


def test_settings_sp_section_is_loaded() -> None:
    """Regression: the [sp] section was missing from _settings_layer, so
    settings.ini device/inference_mode for sp runs were silently ignored."""
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text("[sp]\ndevice = cuda:1\ninference_mode = turbo\n")
        s = load_settings(explicit=str(ini))
        rc = resolve_config(calc_type="sp", settings=s)
        assert rc.device == "cuda:1"
        assert rc.inference_mode == "turbo"
        assert rc.sources["device"].source == "settings.ini"
        assert rc.sources["inference_mode"].source == "settings.ini"


def test_settings_sp_section_does_not_leak_into_other_calc_types() -> None:
    """[sp] device/inference_mode must not leak into opt/md/batch runs."""
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text("[sp]\ndevice = cuda:1\ninference_mode = turbo\n")
        s = load_settings(explicit=str(ini))
        # opt/batch default to inference_mode="default"; if [sp] leaked they
        # would become "turbo". md defaults to "turbo" but device must stay
        # "cuda" (not the [sp] "cuda:1").
        for ct, expected_device, expected_mode in (
            ("opt", "cpu", "default"),
            ("md", "cuda", "turbo"),
            ("batch", "cpu", "default"),
        ):
            rc = resolve_config(calc_type=ct, settings=s)
            assert rc.device == expected_device, f"{ct} leaked [sp] device"
            assert (
                rc.inference_mode == expected_mode
            ), f"{ct} leaked [sp] inference_mode"
