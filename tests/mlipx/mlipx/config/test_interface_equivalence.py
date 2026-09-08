"""Cross-interface typed-config equivalence regressions."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mlipx.api import _build_api_cli
from mlipx.cli import _build_cli_opts, create_parser
from mlipx.config import IncarConfig, resolve_config
from mlipx.queue import parse_task_file


def _fingerprint(resolved) -> dict:
    return {
        "calc_type": resolved.calc_type,
        "model_type": resolved.model_type,
        "model_path": resolved.model_path,
        "task": resolved.task,
        "device": resolved.device,
        "inference_mode": resolved.inference_mode,
        "calculator_options": dict(resolved.calculator_options),
        "run_options": dict(resolved.run_options),
        "settings": dict(resolved.settings),
    }


def test_cli_incar_api_and_queue_resolve_to_same_typed_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "models").mkdir()
    model = tmp_path / "models/mace.model"
    model.write_text("placeholder", encoding="utf-8")
    structure = tmp_path / "structure.xyz"
    structure.write_text("1\nplaceholder\nH 0 0 0\n", encoding="utf-8")

    parser = create_parser()
    args = parser.parse_args(
        [
            "opt",
            str(structure),
            "--model",
            "models/mace.model",
            "--model-type",
            "mace",
            "--task",
            "bulk",
            "--device",
            "cpu",
            "--head",
            "0",
            "--cpu-threads",
            "1",
            "--fmax",
            "1e-3",
            "--max-steps",
            "4",
            "--no-cell-opt",
        ]
    )
    cli_values = _build_cli_opts(args, "opt")
    cli_values["model_path"] = args.model
    cli_resolved = resolve_config(
        calc_type="opt", cli=cli_values, cli_base_dir=tmp_path
    )

    incar = IncarConfig.from_string(
        "MODEL_TYPE = MACE\n"
        "MODEL_PATH = models/mace.model\n"
        "TASK = bulk\n"
        "DEVICE = cpu\n"
        "HEAD = 0\n"
        "CPU_THREADS = 1\n"
        "FMAX = 1D-3\n"
        "MAX_STEPS = 4\n"
        "CELL_OPT = 0\n",
        base_dir=tmp_path,
    )
    incar_resolved = resolve_config(calc_type="opt", incar=incar)

    api_values = _build_api_cli(
        "opt",
        "mace",
        "bulk",
        "cpu",
        None,
        None,
        "0",
        {
            "torch_num_threads": 1,
            "fmax": 1.0e-3,
            "max_steps": 4,
            "cell_opt": False,
        },
    )
    api_values["model_path"] = "models/mace.model"
    api_resolved = resolve_config(
        calc_type="opt", cli=api_values, cli_base_dir=tmp_path
    )

    task_file = tmp_path / "tasks.json"
    task_file.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "calc_type": "opt",
                        "structure": "structure.xyz",
                        "model": "models/mace.model",
                        "model_type": "mace",
                        "task": "bulk",
                        "device": "cpu",
                        "options": {
                            "HEAD": 0,
                            "CPU_THREADS": 1,
                            "FMAX": "1D-3",
                            "MAX_STEPS": 4,
                            "CELL_OPT": "0",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    task = parse_task_file(task_file)["tasks"][0]
    queue_resolved = resolve_config(
        calc_type=task["calc_type"],
        cli={
            "model_type": task["model_type"],
            "model_path": task["model"],
            "task": task["task"],
            "device": task["device"],
            **task["options"],
        },
        cli_base_dir=tmp_path,
    )

    expected = _fingerprint(cli_resolved)
    assert _fingerprint(incar_resolved) == expected
    assert _fingerprint(api_resolved) == expected
    assert _fingerprint(queue_resolved) == expected
    assert expected["calculator_options"]["head"] == "0"
    assert expected["settings"]["torch_num_threads"] == 1
    assert expected["run_options"]["fmax"] == pytest.approx(1.0e-3)


def test_resolved_config_is_immutable() -> None:
    resolved = resolve_config(calc_type="opt", cli={"fmax": 0.02})

    with pytest.raises(TypeError):
        resolved.run_options["fmax"] = 0.5
    with pytest.raises(FrozenInstanceError):
        resolved.device = "cuda"
