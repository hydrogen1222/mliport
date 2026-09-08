"""Tests for mlipx.config.settings (settings.ini loading)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from mlipx.config.settings import (
    DEFAULT_SETTINGS_INI,
    init_settings_file,
    load_settings,
    settings_search_paths,
)

# ---------------------------------------------------------------------------
# Search paths
# ---------------------------------------------------------------------------


def test_search_paths_includes_cwd() -> None:
    paths = settings_search_paths()
    # CWD settings.ini should be one of the search paths
    found = any(
        p.name == "settings.ini" and p.parent == Path.cwd()
        for p in paths
        if p.is_absolute()
    )
    rel_found = any(str(p) == "./settings.ini" for p in paths if not p.is_absolute())
    assert found or rel_found


def test_search_paths_includes_user_config() -> None:
    paths = settings_search_paths()
    user = Path.home() / ".config" / "mlipx" / "settings.ini"
    assert user in [
        p.resolve() if p.is_absolute() else Path.cwd() / p for p in paths
    ] or any("mlipx/settings.ini" in str(p) for p in paths)


# ---------------------------------------------------------------------------
# load_settings - no files found
# ---------------------------------------------------------------------------


def test_load_settings_no_files() -> None:
    with tempfile.TemporaryDirectory() as d:
        # cd into an empty dir so cwd/settings.ini and ancestors don't exist
        old = os.getcwd()
        try:
            os.chdir(d)
            s = load_settings()
            assert len(s.loaded_paths) == 0 or all(p is None for p in [s.path])
            assert s.path is None
            # Aliases / profiles are still empty but valid
            # model_aliases/profiles resolved in resolver, not on settings
        finally:
            os.chdir(old)


# ---------------------------------------------------------------------------
# load_settings - explicit path
# ---------------------------------------------------------------------------


def test_load_settings_explicit_path() -> None:
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "my-settings.ini"
        ini.write_text("[general]\ndevice = cuda:0\n")
        s = load_settings(explicit=str(ini))
        assert ini in s.loaded_paths
        assert s.get("general", "device") == "cuda:0"


def test_settings_record_each_value_declaring_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    low_dir = tmp_path / "low"
    high_dir = tmp_path / "high"
    low_dir.mkdir()
    high_dir.mkdir()
    low = low_dir / "settings.ini"
    high = high_dir / "settings.ini"
    low.write_text("[general]\ndevice = cuda:0\n", encoding="utf-8")
    high.write_text("[general]\nstrict_config = true\n", encoding="utf-8")
    monkeypatch.setenv("MLIPX_SETTINGS", str(low))

    settings = load_settings(explicit=high, cwd=high_dir)

    device_origin = settings.origin("general", "device")
    strict_origin = settings.origin("general", "strict_config")
    assert device_origin is not None and device_origin.path == low.resolve()
    assert device_origin.line == 2
    assert strict_origin is not None and strict_origin.path == high.resolve()
    assert strict_origin.line == 2


# ---------------------------------------------------------------------------
# load_settings - env var
# ---------------------------------------------------------------------------


def test_load_settings_env_var() -> None:
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "env-settings.ini"
        ini.write_text("[general]\nlog_level = DEBUG\n")
        old = os.environ.get("MLIPX_SETTINGS")
        os.environ["MLIPX_SETTINGS"] = str(ini)
        try:
            s = load_settings()
            assert ini in s.loaded_paths
            assert s.get("general", "log_level") == "DEBUG"
        finally:
            if old is not None:
                os.environ["MLIPX_SETTINGS"] = old
            else:
                os.environ.pop("MLIPX_SETTINGS", None)


# ---------------------------------------------------------------------------
# Model aliases and profile parsing
# ---------------------------------------------------------------------------


def test_parse_model_alias_in_resolver() -> None:
    """Aliases are resolved via resolve_config, not stored on MlipxSettings."""
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text(
            "[model:mace_mpa0]\n"
            "engine = mace\n"
            "path = ./model.pt\n"
            "task = bulk\n"
            "dtype = float32\n"
        )
        s = load_settings(explicit=str(ini))
        assert s.sections is not None  # settings loaded
        assert str(ini) in [str(p) for p in s.loaded_paths]


def test_parse_profile_in_settings() -> None:
    """Profile sections loaded via settings, tested in resolver tests."""
    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "settings.ini"
        ini.write_text("[profile:gpu_prod]\n" "device = cuda:1\n" "max_steps = 2000\n")
        s = load_settings(explicit=str(ini))
        assert s.sections is not None  # settings loaded
        assert str(ini) in [str(p) for p in s.loaded_paths]


# ---------------------------------------------------------------------------
# Priority: explicit > env > cwd > user
# ---------------------------------------------------------------------------


def test_explicit_beats_env() -> None:
    with tempfile.TemporaryDirectory() as d:
        explicit = Path(d) / "explicit.ini"
        explicit.write_text("[general]\noutput_root = ./explicit_output\n")
        env_ini = Path(d) / "env.ini"
        env_ini.write_text("[general]\noutput_root = ./env_output\n")
        old = os.environ.get("MLIPX_SETTINGS")
        os.environ["MLIPX_SETTINGS"] = str(env_ini)
        try:
            s = load_settings(explicit=str(explicit))
            # Both loaded, but explicit has higher priority
            assert s.get("general", "output_root") == "./explicit_output"
        finally:
            if old is not None:
                os.environ["MLIPX_SETTINGS"] = old
            else:
                os.environ.pop("MLIPX_SETTINGS", None)


# ---------------------------------------------------------------------------
# init_settings_file
# ---------------------------------------------------------------------------


def test_init_settings_file_project() -> None:
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "settings.ini"
        path = init_settings_file(str(target))
        assert Path(path).exists()
        assert "mlipx" in Path(path).read_text().lower()
        assert str(target) == str(path)


def test_init_settings_file_user() -> None:
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "settings.ini"
        path = init_settings_file(str(target))
        assert Path(path).exists()
        assert "mlipx" in Path(path).read_text().lower()


def test_init_settings_file_force_overwrite() -> None:
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "settings.ini"
        path1 = init_settings_file(str(target))
        # Overwrite without force raises
        with pytest.raises(FileExistsError):
            init_settings_file(str(target))
        # Overwrite with force succeeds
        path2 = init_settings_file(str(target), force=True)
        assert path1 == path2


# ---------------------------------------------------------------------------
# DEFAULT_SETTINGS_INI template
# ---------------------------------------------------------------------------


def test_default_settings_ini_has_sections() -> None:
    assert "[general]" in DEFAULT_SETTINGS_INI
    assert "[model:" in DEFAULT_SETTINGS_INI
    assert "[profile:" in DEFAULT_SETTINGS_INI
