"""Interface equivalence across every supported submission surface.

One logical request -- expressed through the CLI, the Python API, an INCAR-style
job mapping, and a queue task spec -- must resolve to the same typed request
(model identity, task/head, dtype, device, structure, physical run options,
calculator options). The TUI submits through the shared ``build_mliport_command``
queue helper, so a TUI-built command is compared against the CLI resolver too
(direct argv-drift between TUI and queue is pinned by ``test_tui.py``).

What is compared: the resolved request (``ResolvedConfig`` model/task/device/
inference_mode plus ``calculator_options``/``run_options``/``settings``/
``unknown_options`` and the ``EngineConfig`` identity fields). Output/working
directory and display name are execution placement, not request content, and
are excluded; per-key layer provenance (``sources``) is intentionally allowed
to differ because it records *where* each value came from.

This file does not execute models: equivalence is a property of the resolver
layer. CPU execution semantics are covered by the product runner tests
(``test_cli.py``, ``test_neb_workflow.py``) and the validation harness tiers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write as ase_write

from mliport.cli import _resolve_engine_config, create_parser
from mliport.config import resolve_config
from mliport.engine import EngineConfig

_MACE = "mace"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _make_structure(tmp_path: Path) -> Path:
    atoms = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.8, 1.8, 1.8]],
        cell=[3.6, 3.6, 3.6],
        pbc=True,
    )
    path = tmp_path / "structure.vasp"
    ase_write(path, atoms, format="vasp")
    return path


def _make_model(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    path.write_bytes(b"interface-equivalence-model")
    return path


def _make_endpoints(tmp_path: Path) -> tuple[Path, Path]:
    initial = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.8, 1.8, 1.8]],
        cell=[3.6, 3.6, 3.6],
        pbc=True,
    )
    final = initial.copy()
    final.positions[1, 0] += 0.9
    initial_path = tmp_path / "initial.vasp"
    final_path = tmp_path / "final.vasp"
    ase_write(initial_path, initial, format="vasp")
    ase_write(final_path, final, format="vasp")
    return initial_path, final_path


def _identity_projection(engine: EngineConfig) -> dict:
    """Comparable identity view of one resolved request (placement excluded)."""
    return {
        "calc_type": engine.calc_type,
        "model_path": str(engine.model_path),
        "model_type": engine.model_type,
        "task": engine.task,
        "device": engine.device,
        "inference_mode": engine.inference_mode,
        "strict_config": engine.strict_config,
        "calculator_options": dict(engine.calculator_options),
        "run_options": dict(engine.run_options),
        "settings": dict(engine.settings),
    }


def _md_options() -> dict[str, object]:
    return {
        "ensemble": "NVT",
        "temperature": 500,
        "timestep": 2.0,
        "steps": 40,
        "thermostat": "LANGEVIN",
        "friction": 0.02,
        "save_interval": 10,
        "seed": 20260911,
        "com_policy": "none",
        "default_dtype": "float32",
        "head": "small",
    }


def _opt_options() -> dict[str, object]:
    return {
        "fmax": 0.05,
        "max_steps": 25,
        "cell_opt": False,
        "write_outcar": True,
        "default_dtype": "float32",
        "head": "small",
    }


def _neb_options() -> dict[str, object]:
    return {
        "n_intermediate_images": 3,
        "climb": True,
        "neb_method": "improvedtangent",
        "neb_interpolation": "linear",
        "neb_spring": 0.1,
        "checkpoint_interval": 5,
        "default_dtype": "float32",
        "head": "small",
    }


def _identity_cli(calc_type: str, structure: Path, model: Path) -> list[str]:
    argv = [calc_type]
    if structure is not None:
        argv.append(str(structure))
    argv += [
        "--model",
        str(model),
        "--model-type",
        _MACE,
        "--task",
        "bulk",
        "--device",
        "cpu",
    ]
    return argv


_FLAG_BY_KEY = {
    "ensemble": "--ensemble",
    "temperature": "--temp",
    "timestep": "--timestep",
    "steps": "--steps",
    "thermostat": "--thermostat",
    "friction": "--friction",
    "save_interval": "--save-interval",
    "seed": "--seed",
    "com_policy": "--com-policy",
    "default_dtype": "--dtype",
    "head": "--head",
    "fmax": "--fmax",
    "max_steps": "--max-steps",
    "cell_opt": "--no-cell-opt",
    "write_outcar": "--write-outcar",
    "n_intermediate_images": "--images",
    "climb": "--climb",
    "neb_method": "--method",
    "neb_interpolation": "--interpolation",
    "neb_spring": "--spring",
    "checkpoint_interval": "--checkpoint-interval",
}


def _argv_options(options: dict[str, object]) -> list[str]:
    argv: list[str] = []
    for key, value in options.items():
        flag = _FLAG_BY_KEY[key]
        if value is True:
            argv.append(flag)
        elif value is False:
            continue  # negated flags are appended explicitly below
        else:
            argv += [flag, str(value)]
    return argv


def _resolve_via_cli(
    tmp_path: Path, calc_type: str, options: dict[str, object]
) -> EngineConfig:
    structure = None if calc_type == "neb" else _make_structure(tmp_path)
    model = _make_model(tmp_path)
    argv = _identity_cli(calc_type, structure, model)
    argv += _argv_options(options)
    if calc_type == "neb":
        initial, final = _make_endpoints(tmp_path)
        argv += [
            "--initial",
            str(initial),
            "--final",
            str(final),
            "--output",
            str(tmp_path / "out"),
        ]
    args = create_parser().parse_args(argv)
    engine, _resolved, _settings = _resolve_engine_config(args, calc_type)
    return engine


def _resolve_via_cli_with_resolved(
    tmp_path: Path, calc_type: str, options: dict[str, object]
):
    structure = None if calc_type == "neb" else _make_structure(tmp_path)
    model = _make_model(tmp_path)
    argv = _identity_cli(calc_type, structure, model)
    argv += _argv_options(options)
    if calc_type == "neb":
        initial, final = _make_endpoints(tmp_path)
        argv += [
            "--initial",
            str(initial),
            "--final",
            str(final),
            "--output",
            str(tmp_path / "out"),
        ]
    args = create_parser().parse_args(argv)
    engine, resolved, _settings = _resolve_engine_config(args, calc_type)
    return engine, resolved


def _resolve_via_api(
    tmp_path: Path, calc_type: str, options: dict[str, object]
) -> EngineConfig:
    from mliport.api import _api_resolve_full

    model = _make_model(tmp_path)
    cli = {
        "model_type": _MACE,
        "task": "bulk",
        "device": "cpu",
        **options,
    }
    kwargs: dict[str, object] = {
        "model_path": str(model),
        "output_dir": str(tmp_path / "out"),
        "job_name": "eq-test",
    }
    if calc_type == "neb":
        initial, final = _make_endpoints(tmp_path)
        cli["neb_initial"] = str(initial)
        cli["neb_final"] = str(final)
    engine, _resolved = _api_resolve_full(
        calc_type,
        kwargs["model_path"],
        cli,
        kwargs["output_dir"],
        kwargs["job_name"],
        None,
        None,
        None,
        None,  # resolver default (strict since CFG-01)
    )
    return engine


def _resolve_via_api_with_resolved(
    tmp_path: Path, calc_type: str, options: dict[str, object]
):
    from mliport.api import _api_resolve_full

    model = _make_model(tmp_path)
    cli = {
        "model_type": _MACE,
        "task": "bulk",
        "device": "cpu",
        **options,
    }
    if calc_type == "neb":
        initial, final = _make_endpoints(tmp_path)
        cli["neb_initial"] = str(initial)
        cli["neb_final"] = str(final)
    engine, resolved = _api_resolve_full(
        calc_type,
        str(model),
        cli,
        str(tmp_path / "out"),
        "eq-test",
        None,
        None,
        None,
        False,
    )
    return engine, resolved


def _resolve_via_incar(
    tmp_path: Path, calc_type: str, options: dict[str, object]
) -> EngineConfig:
    model = _make_model(tmp_path)
    incar = dict(options)
    cli = {
        "model_path": str(model),
        "model_type": _MACE,
        "task": "bulk",
        "device": "cpu",
    }
    if calc_type == "neb":
        initial, final = _make_endpoints(tmp_path)
        incar["neb_initial"] = str(initial)
        incar["neb_final"] = str(final)
    resolved = resolve_config(
        calc_type=calc_type,
        incar=incar,
        cli=cli,
        incar_base_dir=str(tmp_path),
        cli_base_dir=str(tmp_path),
    )
    return EngineConfig.from_resolved(resolved)


def _resolve_via_queue(
    tmp_path: Path, calc_type: str, options: dict[str, object]
) -> EngineConfig:
    from mliport.queue import build_mliport_command

    structure = None if calc_type == "neb" else _make_structure(tmp_path)
    model = _make_model(tmp_path)
    kwargs: dict[str, object] = {
        "calc_type": calc_type,
        "structure": None if structure is None else str(structure),
        "model": str(model),
        "model_type": _MACE,
        "task": "bulk",
        "device": "cpu",
        "output_dir": str(tmp_path / "out"),
        "job_name": "eq-test",
        "options": dict(options),
    }
    if calc_type == "neb":
        initial, final = _make_endpoints(tmp_path)
        kwargs["initial"] = str(initial)
        kwargs["final"] = str(final)
    argv = build_mliport_command(**kwargs)
    assert argv[:3] == [sys.executable, "-m", "mliport.cli"], argv[:3]
    args = create_parser().parse_args(argv[3:])
    engine, _resolved, _settings = _resolve_engine_config(args, calc_type)
    return engine


def assert_equivalent(engine_a: EngineConfig, engine_b: EngineConfig) -> None:
    reference = _identity_projection(engine_a)
    other = _identity_projection(engine_b)
    for field, value_a in reference.items():
        value_b = other[field]
        assert value_a == value_b, (
            f"{field}: {value_a!r} != {value_b!r} "
            f"(a run_options={reference['run_options']})"
        )


# ---------------------------------------------------------------------------
# equivalence: sp / opt / md / neb
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("calc_type", ["sp", "opt", "md", "neb"])
def test_request_resolution_is_interface_independent(
    tmp_path: Path, calc_type: str
) -> None:
    """The same logical request resolves identically through CLI, API, INCAR
    mapping, and the queue task spec (the TUI submits via the same helper)."""
    options = {
        "sp": {"default_dtype": "float32", "head": "small", "write_outcar": True},
        "opt": _opt_options(),
        "md": _md_options(),
        "neb": _neb_options(),
    }[calc_type]

    engines = {
        "cli": _resolve_via_cli(tmp_path, calc_type, options),
        "api": _resolve_via_api(tmp_path, calc_type, options),
        "incar": _resolve_via_incar(tmp_path, calc_type, options),
        "queue": _resolve_via_queue(tmp_path, calc_type, options),
    }
    reference = _identity_projection(engines["cli"])
    for name, engine in engines.items():
        assert _identity_projection(engine) == reference, name

    # strong comparison on the resolved option bags and settings
    cli_engine = engines["cli"]
    for name in ("api", "incar", "queue"):
        assert_equivalent(cli_engine, engines[name])

    # model identity must survive every interface unchanged
    for engine in engines.values():
        assert engine.model_type == _MACE
        assert engine.task == "bulk"
        assert engine.device == "cpu"
        assert Path(engine.model_path).name == "model.pt"


def test_resolved_option_bags_carry_all_request_parameters(
    tmp_path: Path,
) -> None:
    """Every physically meaningful request parameter lands in a typed bag."""
    _engine, resolved = _resolve_via_api_with_resolved(tmp_path, "md", _md_options())
    run_options = resolved.run_options
    for key, value in _md_options().items():
        if key in {"default_dtype", "head"}:
            assert resolved.calculator_options[key] == value
        else:
            assert run_options[key] == value
    assert resolved.unknown_options == {}
    assert resolved.model_type == _MACE


def test_neb_mapping_and_checkpoint_policy_resolve_identically(
    tmp_path: Path,
) -> None:
    """NEB mapping/winding/checkpoint policy survive interface translation."""
    _engine, resolved = _resolve_via_api_with_resolved(tmp_path, "neb", _neb_options())
    run_options = resolved.run_options
    assert run_options["n_intermediate_images"] == 3
    assert run_options["climb"] is True
    assert run_options["neb_method"] == "improvedtangent"
    assert run_options["checkpoint_interval"] == 5
    assert Path(run_options["neb_initial"]).name == "initial.vasp"
    assert Path(run_options["neb_final"]).name == "final.vasp"


# ---------------------------------------------------------------------------
# TUI: the generated argv must resolve to the same typed request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_tui_md_command_resolves_like_api_request(tmp_path: Path) -> None:
    """A TUI-built MD argv parses through the CLI resolver to the same typed
    request the API produces for the same logical configuration."""
    from mliport.api import _api_resolve_full
    from mliport.tui import MliportApp
    from mliport.tui.run_screen import RunScreen

    structure = _make_structure(tmp_path)
    model = _make_model(tmp_path)
    options = _md_options()
    output_dir = str(tmp_path / "tui-out")

    app = MliportApp()
    app.config.update(
        {
            "calc_type": "md",
            "structure_file": str(structure),
            "model_file": str(model),
            "model_type": _MACE,
            "task": "bulk",
            "device": "cpu",
            "output_dir": output_dir,
            "inference_mode": "default",
            **options,
        }
    )

    async with app.run_test(size=(80, 40)):
        screen = RunScreen()
        screen._job_id = "eq-tui-md"
        command = screen._build_command()

    assert command[:3] == [sys.executable, "-m", "mliport.cli"]
    args = create_parser().parse_args(command[3:])
    tui_engine, tui_resolved, _settings = _resolve_engine_config(args, "md")

    cli = {"model_type": _MACE, "task": "bulk", "device": "cpu", **options}
    api_engine, api_resolved = _api_resolve_full(
        "md",
        str(model),
        cli,
        output_dir,
        "eq-tui-md",
        None,
        None,
        None,
        None,  # resolver default (strict since CFG-01)
    )

    assert_equivalent(tui_engine, api_engine)
    assert dict(tui_resolved.calculator_options) == dict(
        api_resolved.calculator_options
    )
    assert dict(tui_resolved.run_options) == dict(api_resolved.run_options)
