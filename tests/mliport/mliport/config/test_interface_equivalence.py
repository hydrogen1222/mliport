"""Cross-interface typed-config equivalence regressions."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mliport.api import _build_api_cli
from mliport.backend_selection import NO_BACKEND_SELECTED_MESSAGE
from mliport.cli import _build_cli_opts, create_parser
from mliport.config import IncarConfig
from mliport.config import resolve_config as _resolve_config
from mliport.queue import parse_task_file


def resolve_config(*, calc_type, **kwargs):
    """Legacy equivalence tests: UMA is an explicit test fixture choice."""
    try:
        return _resolve_config(calc_type=calc_type, **kwargs)
    except ValueError as exc:
        if NO_BACKEND_SELECTED_MESSAGE not in str(exc):
            raise
        cli = dict(kwargs.pop("cli", None) or {})
        cli["model_type"] = "uma"
        return _resolve_config(calc_type=calc_type, cli=cli, **kwargs)


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


def test_neb_cli_incar_api_and_queue_share_one_typed_resolution(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    for endpoint in ("initial.vasp", "final.vasp"):
        (tmp_path / endpoint).write_text("placeholder", encoding="utf-8")
    common = {
        "model_type": "mace",
        "model_path": "model.pt",
        "task": "bulk",
        "device": "cpu",
        "neb_initial": "initial.vasp",
        "neb_final": "final.vasp",
        "n_intermediate_images": 3,
        "climb": True,
        "neb_spring": 0.2,
        "max_steps": 4,
        "checkpoint_interval": 2,
    }

    args = create_parser().parse_args(
        [
            "neb",
            "--initial",
            "initial.vasp",
            "--final",
            "final.vasp",
            "--model",
            "model.pt",
            "--model-type",
            "mace",
            "--task",
            "bulk",
            "--device",
            "cpu",
            "--images",
            "3",
            "--climb",
            "--spring",
            "0.2",
            "--max-steps",
            "4",
            "--checkpoint-interval",
            "2",
            "--output",
            "out",
        ]
    )
    cli_values = _build_cli_opts(args, "neb")
    cli_values["model_path"] = args.model
    cli_resolved = resolve_config(
        calc_type="neb", cli=cli_values, cli_base_dir=tmp_path
    )

    incar = IncarConfig.from_string(
        "CALCULATION = NEB\n"
        "MODEL_TYPE = MACE\n"
        "MODEL_PATH = model.pt\n"
        "TASK = bulk\n"
        "DEVICE = cpu\n"
        "NEB_INITIAL = initial.vasp\n"
        "NEB_FINAL = final.vasp\n"
        "NEB_IMAGES = 3\n"
        "NEB_CLIMB = 1\n"
        "NEB_SPRING = 0.2\n"
        "MAX_STEPS = 4\n"
        "NEB_CHECKPOINT_INTERVAL = 2\n",
        base_dir=tmp_path,
    )
    incar_resolved = resolve_config(calc_type="neb", incar=incar)

    api_values = _build_api_cli(
        "neb",
        "mace",
        "bulk",
        "cpu",
        None,
        None,
        None,
        {key: value for key, value in common.items() if key != "model_type"},
    )
    api_resolved = resolve_config(
        calc_type="neb", cli=api_values, cli_base_dir=tmp_path
    )

    task_file = tmp_path / "neb-tasks.json"
    task_file.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "calc_type": "neb",
                        "initial": "initial.vasp",
                        "final": "final.vasp",
                        "model": "model.pt",
                        "model_type": "mace",
                        "task": "bulk",
                        "device": "cpu",
                        "options": {
                            "NEB_IMAGES": "3",
                            "NEB_CLIMB": "1",
                            "NEB_SPRING": "0.2",
                            "MAX_STEPS": "4",
                            "NEB_CHECKPOINT_INTERVAL": "2",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    queued = parse_task_file(task_file)["tasks"][0]
    queue_resolved = resolve_config(
        calc_type="neb",
        cli={
            "model_type": queued["model_type"],
            "model_path": queued["model"],
            "task": queued["task"],
            "device": queued["device"],
            "neb_initial": queued["initial"],
            "neb_final": queued["final"],
            **queued["options"],
        },
        cli_base_dir=tmp_path,
    )

    expected = _fingerprint(cli_resolved)
    assert _fingerprint(incar_resolved) == expected
    assert _fingerprint(api_resolved) == expected
    assert _fingerprint(queue_resolved) == expected
