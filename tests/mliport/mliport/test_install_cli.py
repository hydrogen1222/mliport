"""Tests for the interactive Python-native installer CLI."""

from __future__ import annotations

import pytest

from mliport.install import cli
from mliport.install.plan import InstallPlanError


def test_executor_uses_managed_source_environment(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("UV_INDEX_URL", "https://wrong.example/simple")
    monkeypatch.setenv("UV_FIND_LINKS", "https://wrong.example/wheels")
    monkeypatch.setattr(cli, "detect_gpus", lambda: None)
    monkeypatch.setattr(cli, "_existing_venv_python_mismatch", lambda *_args: False)
    calls = []

    def execute(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", execute)
    assert (
        cli.main(
            [
                "--engines",
                "mace",
                "--device",
                "cpu",
                "--source",
                "official",
                "--skip-doctor",
                "--non-interactive",
            ]
        )
        == 0
    )
    assert calls
    for _argv, env in calls:
        assert "UV_INDEX_URL" not in env
        assert "UV_FIND_LINKS" not in env
        assert env["UV_NO_CONFIG"] == "1"


def test_china_source_prompt_accepts_number_after_invalid_input(capsys) -> None:
    answers = iter(["not-a-number", "6", "3"])

    selected = cli._prompt_china_source(lambda _prompt: next(answers))

    assert selected == "china-ustc"
    captured = capsys.readouterr()
    assert "1) China: TUNA" in captured.err
    assert "5) Official" in captured.err
    assert captured.err.count("Invalid choice") == 2


def test_china_source_prompt_empty_response_keeps_historical_default() -> None:
    assert cli._prompt_china_source(lambda _prompt: "") == "china"


def test_china_source_prompt_eof_fails_closed() -> None:
    def _eof(_prompt: str) -> str:
        raise EOFError

    with pytest.raises(InstallPlanError, match="cancelled"):
        cli._prompt_china_source(_eof)


def test_china_source_interactive_selection_reaches_plan(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "detect_gpus", lambda: None)
    monkeypatch.setattr(cli, "_can_prompt_for_source", lambda: True)
    monkeypatch.setattr(cli, "_prompt_china_source", lambda: "china-ustc")
    monkeypatch.setattr(cli, "_existing_venv_python_mismatch", lambda *_args: False)

    result = cli.main(
        [
            "--engines",
            "dpa",
            "--device",
            "cpu",
            "--source",
            "china",
            "--skip-doctor",
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Selected source: China: USTC" in output
    assert "Source   : china-ustc" in output
    assert "torch==2.10.0+cpu" in output
    assert "mirrors.aliyun.com/pytorch-wheels/cpu" in output
    assert "mirrors.ustc.edu.cn/pypi/simple" in output


def test_china_source_non_interactive_never_prompts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "detect_gpus", lambda: None)
    monkeypatch.setattr(cli, "_can_prompt_for_source", lambda: True)
    monkeypatch.setattr(
        cli,
        "_prompt_china_source",
        lambda: pytest.fail("non-interactive mode must not prompt"),
    )
    monkeypatch.setattr(cli, "_existing_venv_python_mismatch", lambda *_args: False)

    result = cli.main(
        [
            "--engines",
            "uma",
            "--device",
            "cpu",
            "--source",
            "china",
            "--non-interactive",
            "--skip-doctor",
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Source   : china" in output
    assert "tuna.tsinghua.edu.cn" in output


def test_china_source_non_terminal_input_never_prompts(monkeypatch) -> None:
    monkeypatch.setattr(cli, "detect_gpus", lambda: None)
    monkeypatch.setattr(cli, "_can_prompt_for_source", lambda: False)
    monkeypatch.setattr(
        cli,
        "_prompt_china_source",
        lambda: pytest.fail("non-terminal input must not prompt"),
    )
    monkeypatch.setattr(cli, "_existing_venv_python_mismatch", lambda *_args: False)

    assert (
        cli.main(
            [
                "--engines",
                "uma",
                "--device",
                "cpu",
                "--source",
                "china",
                "--skip-doctor",
                "--dry-run",
            ]
        )
        == 0
    )


def test_cancelled_interactive_source_returns_cli_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "detect_gpus", lambda: None)
    monkeypatch.setattr(cli, "_can_prompt_for_source", lambda: True)
    monkeypatch.setattr(
        cli,
        "_prompt_china_source",
        lambda: (_ for _ in ()).throw(InstallPlanError("Source selection cancelled.")),
    )

    assert cli.main(["--source", "china", "--dry-run"]) == 2
    assert "Source selection cancelled" in capsys.readouterr().err
