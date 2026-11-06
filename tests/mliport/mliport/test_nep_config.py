"""NEP dual-runtime configuration and identity contract (PR #61 sections C/D).

The scientific requirement behind these tests is that the *runtime* is an
explicit user decision.  ``NEP_RUNTIME=ase`` and ``NEP_RUNTIME=gpumd`` execute
the same ``nep.txt`` through two different code paths with different numeric
precision (float64 in-process versus float32 inside a spawned ``gpumd``
process) and completely different per-step cost.  If mliport ever inferred,
defaulted or silently fell back from one to the other, two runs that look
identical in ``resolved_config.json`` would not be comparable evidence.

Nothing here needs ``calorine`` or a GPU: the tests exercise the resolver,
schema, factory routing, engine validation, entry-point parity, queue command
builder, INCAR template and doctor inventory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mliport.api import run_single_point
from mliport.backend_selection import (
    CALCULATOR_OPTION_KEYS,
    NEP_CALC_TYPES_BY_RUNTIME,
    SUPPORTED_BACKENDS,
    model_type_choices,
    nep_calc_types,
    normalize_device_for_nep,
    require_nep_support,
)
from mliport.calculators.factory import CalculatorFactory
from mliport.cli import create_parser
from mliport.config import get_schema, resolve_config
from mliport.config.defaults import build_incar_default
from mliport.config.incar import IncarConfig
from mliport.engine import CalculationEngine, EngineConfig
from mliport.queue import build_mliport_command

# A syntactically valid NEP4 header.  The numbers are the radial/angular
# cutoff (Angstrom) followed by the neighbour capacities; only the header is
# needed for configuration tests, no inference happens here.
NEP_MODEL_TEXT = "nep4 3 Ba Zr O\ncutoff 8 4 232 26\n"


def _nep_model(tmp_path: Path, name: str = "nep.txt") -> Path:
    model = tmp_path / name
    model.write_text(NEP_MODEL_TEXT, encoding="utf-8")
    return model


def _gpumd_executable(tmp_path: Path) -> Path:
    binary = tmp_path / "gpumd"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


# --------------------------------------------------------------- backend table


def test_nep_is_a_first_class_backend():
    assert "nep" in SUPPORTED_BACKENDS
    assert "nep" in model_type_choices()
    # ``fairchem`` stays the single alias and appears exactly once.
    assert model_type_choices().count("fairchem") == 1
    assert set(CALCULATOR_OPTION_KEYS) == {"uma", "mace", "dpa", "grace", "nep"}
    assert CALCULATOR_OPTION_KEYS["nep"] == {"nep_runtime", "nep_gpumd_command"}


def test_every_public_entry_point_accepts_the_same_model_types():
    """CLI, schema, INCAR and install inventory must not drift (section D.1)."""
    choices = set(model_type_choices())
    parser = create_parser()
    # argparse stores the choices iterable verbatim; inspect each subparser that
    # declares --model-type so a new backend cannot appear in one only.
    declared = [
        set(action.choices or ())
        for action in _iter_actions(parser)
        if "--model-type" in getattr(action, "option_strings", ())
    ]
    assert declared, "no --model-type option found"
    for entry in declared:
        assert choices <= entry, entry

    model_type_spec = get_schema().resolve("model_type")
    assert set(model_type_spec.choices) == choices

    incar = IncarConfig(
        {"MODEL_TYPE": "NEP", "MODEL_PATH": "/tmp/nep.txt", "NEP_RUNTIME": "ase"},
        source_name="INCAR",
    )
    assert incar.validate() == [] or all(
        "MODEL_PATH" not in error for error in incar.validate()
    )


def _iter_actions(parser):
    """Yield every argparse action of the parser and its subparsers."""
    stack = [parser]
    while stack:
        current = stack.pop()
        yield from current._actions
        for action in current._actions:
            if hasattr(action, "choices") and isinstance(action.choices, dict):
                stack.extend(action.choices.values())


# ------------------------------------------------------------------ resolver


def test_resolver_requires_an_explicit_nep_runtime(tmp_path):
    model = _nep_model(tmp_path)
    with pytest.raises(ValueError, match="NEP requires an explicit execution runtime"):
        resolve_config(
            calc_type="sp",
            cli={"model_type": "nep", "model_path": str(model), "task": "bulk"},
        )


def test_resolver_records_runtime_and_normalised_device(tmp_path):
    model = _nep_model(tmp_path)
    resolved = resolve_config(
        calc_type="sp",
        cli={
            "model_type": "nep",
            "model_path": str(model),
            "task": "bulk",
            "nep_runtime": "ASE",
            "device": "gpu",
        },
    )
    assert resolved.model_type == "nep"
    assert resolved.calculator_options["nep_runtime"] == "ase"
    # ``gpu`` is a documented synonym that resolves to the canonical ``cuda``.
    assert resolved.device == "cuda"
    # The runtime's provenance is recorded, not implicit.
    assert "nep_runtime" in resolved.sources
    assert "device" in resolved.sources


@pytest.mark.parametrize(
    ("runtime", "device", "expected"),
    [
        ("ase", "cpu", "cpu"),
        ("ase", "cuda", "cuda"),
        ("ase", "cuda:1", "cuda:1"),
        ("ase", "gpu", "cuda"),
        ("gpumd", "cuda", "cuda"),
        ("gpumd", "cuda:0", "cuda:0"),
    ],
)
def test_normalize_device_for_nep_matrix(runtime, device, expected):
    assert normalize_device_for_nep(runtime, device) == expected


@pytest.mark.parametrize(
    ("runtime", "device", "message"),
    [
        ("gpumd", "cpu", "no CPU implementation"),
        ("ase", "mps", "Invalid device"),
        ("ase", "cuda:x", "CUDA index"),
    ],
)
def test_normalize_device_for_nep_rejects(runtime, device, message):
    with pytest.raises(ValueError, match=message):
        normalize_device_for_nep(runtime, device)


def test_gpumd_cpu_request_is_never_downgraded_to_cpunep(tmp_path):
    """The rejected combination must not be silently rerouted (section C)."""
    model = _nep_model(tmp_path)
    with pytest.raises(ValueError, match="no CPU implementation"):
        resolve_config(
            calc_type="sp",
            cli={
                "model_type": "nep",
                "model_path": str(model),
                "task": "bulk",
                "nep_runtime": "gpumd",
                "device": "cpu",
            },
        )


def test_nep_model_path_must_be_a_file(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    with pytest.raises(ValueError, match="is a directory"):
        resolve_config(
            calc_type="sp",
            cli={
                "model_type": "nep",
                "model_path": str(directory),
                "task": "bulk",
                "nep_runtime": "ase",
            },
        )
    with pytest.raises(FileNotFoundError, match="NEP model file not found"):
        resolve_config(
            calc_type="sp",
            cli={
                "model_type": "nep",
                "model_path": str(tmp_path / "missing.txt"),
                "task": "bulk",
                "nep_runtime": "ase",
            },
        )


@pytest.mark.parametrize(
    ("runtime", "calc_type", "supported"),
    [
        ("ase", "sp", True),
        ("ase", "opt", True),
        ("ase", "md", True),
        ("ase", "neb", True),
        ("ase", "batch", True),
        ("ase", "confsearch", True),
        ("gpumd", "sp", True),
        ("gpumd", "opt", True),
        ("gpumd", "md", True),
        ("gpumd", "neb", False),
        ("gpumd", "batch", False),
        ("gpumd", "confsearch", False),
    ],
)
def test_capability_matrix_is_enforced(runtime, calc_type, supported):
    assert calc_type in nep_calc_types(runtime) if supported else True
    if supported:
        require_nep_support(runtime, calc_type)
    else:
        with pytest.raises(ValueError, match="does not support calc_type"):
            require_nep_support(runtime, calc_type)


def test_gpumd_rejection_names_the_ase_runtime_not_a_fallback():
    with pytest.raises(ValueError) as excinfo:
        require_nep_support("gpumd", "neb")
    message = str(excinfo.value)
    assert "NEP_RUNTIME=ase" in message
    assert "never reroutes a runtime silently" in message


def test_declared_capability_matrix_covers_every_calc_type():
    known = {"sp", "opt", "md", "neb", "batch", "confsearch"}
    for runtime, supported in NEP_CALC_TYPES_BY_RUNTIME.items():
        assert set(supported) <= known, (runtime, supported)
    # GPUMD is strictly a subset: nothing is invented for the native path.
    assert NEP_CALC_TYPES_BY_RUNTIME["gpumd"] < NEP_CALC_TYPES_BY_RUNTIME["ase"]


def test_nep_runtime_is_rejected_for_other_engines():
    with pytest.raises(ValueError, match="not applicable to engine 'mace'"):
        resolve_config(
            calc_type="sp",
            cli={
                "model_type": "mace",
                "model_path": "model.bin",
                "task": "bulk",
                "nep_runtime": "ase",
            },
        )


def test_nep_does_not_accept_torch_thread_or_uma_mode_options(tmp_path):
    model = _nep_model(tmp_path)
    with pytest.raises(ValueError, match="inference_mode"):
        resolve_config(
            calc_type="sp",
            cli={
                "model_type": "nep",
                "model_path": str(model),
                "task": "bulk",
                "nep_runtime": "ase",
                "inference_mode": "turbo",
            },
        )


def test_engine_rejects_torch_num_threads_for_nep(tmp_path):
    model = _nep_model(tmp_path)
    with pytest.raises(ValueError, match="TORCH_NUM_THREADS"):
        CalculationEngine.from_config(
            EngineConfig(
                calc_type="sp",
                model_path=model,
                model_type="nep",
                task="bulk",
                device="cpu",
                calculator_options={"nep_runtime": "ase"},
                torch_num_threads=4,
            )
        )


def test_engine_requires_gpumd_command_for_the_native_runtime(tmp_path):
    model = _nep_model(tmp_path)
    with pytest.raises(ValueError, match="NEP_GPUMD_COMMAND"):
        CalculationEngine.from_config(
            EngineConfig(
                calc_type="md",
                model_path=model,
                model_type="nep",
                task="bulk",
                device="cuda",
                calculator_options={"nep_runtime": "gpumd"},
            )
        )


def test_engine_rejects_gpumd_neb_before_creating_outputs(tmp_path):
    model = _nep_model(tmp_path)
    binary = _gpumd_executable(tmp_path)
    with pytest.raises(ValueError, match="does not support calc_type"):
        CalculationEngine.from_config(
            EngineConfig(
                calc_type="neb",
                model_path=model,
                model_type="nep",
                task="bulk",
                device="cuda",
                calculator_options={
                    "nep_runtime": "gpumd",
                    "nep_gpumd_command": str(binary),
                },
            )
        )
    assert not (tmp_path / "neb").exists()


# -------------------------------------------------------------------- factory


def test_factory_routes_the_ase_runtime_to_the_nep_wrapper(tmp_path):
    from mliport.calculators.nep_calc import NEPCalculatorWrapper

    model = _nep_model(tmp_path)
    wrapper = CalculatorFactory.create(
        "nep",
        model,
        device="cpu",
        task="bulk",
        strict=True,
        nep_runtime="ase",
    )
    assert isinstance(wrapper, NEPCalculatorWrapper)
    info = wrapper.info()
    assert info["model_type"] == "nep"
    assert info["nep_runtime"] == "ase"
    assert info["default_dtype"] == "float64"
    assert info["per_call_subprocess"] is False


def test_factory_refuses_to_build_an_ase_calculator_for_the_native_runtime(tmp_path):
    model = _nep_model(tmp_path)
    binary = _gpumd_executable(tmp_path)
    with pytest.raises(ValueError, match="native gpumd process"):
        CalculatorFactory.create(
            "nep",
            model,
            device="cuda",
            task="bulk",
            strict=True,
            nep_runtime="gpumd",
            nep_gpumd_command=str(binary),
        )


def test_gpunep_route_requires_a_gpumd_command(tmp_path):
    model = _nep_model(tmp_path)
    with pytest.raises(ValueError, match="gpumd_command"):
        CalculatorFactory.create(
            "nep",
            model,
            device="cuda",
            task="bulk",
            strict=True,
            nep_runtime="ase",
        )


def test_gpunep_route_reports_float32_and_per_call_subprocess(tmp_path):
    from mliport.calculators.nep_calc import NEPCalculatorWrapper

    model = _nep_model(tmp_path)
    binary = _gpumd_executable(tmp_path)
    wrapper = NEPCalculatorWrapper(
        model_path=model,
        device="cuda:0",
        task="bulk",
        nep_runtime="ase",
        gpumd_command=binary,
    )
    info = wrapper.info()
    assert info["default_dtype"] == "float32"
    assert info["per_call_subprocess"] is True
    assert info["actual_device_type"] == "cuda"
    assert info["actual_device_logical_index"] == 0


# ------------------------------------------------------------ engine dispatch


def test_engine_dispatches_gpumd_to_the_native_runner(tmp_path, monkeypatch):
    """``NEP_RUNTIME=gpumd`` must never construct an ASE calculator."""
    model = _nep_model(tmp_path)
    binary = _gpumd_executable(tmp_path)
    engine = CalculationEngine.from_config(
        EngineConfig(
            calc_type="sp",
            model_path=model,
            model_type="nep",
            task="bulk",
            device="cuda",
            calculator_options={
                "nep_runtime": "gpumd",
                "nep_gpumd_command": str(binary),
            },
        )
    )
    assert engine.uses_native_gpumd is True

    def _explode(*_args, **_kwargs):
        raise AssertionError("the native runtime must not build an ASE calculator")

    monkeypatch.setattr(engine, "_create_calculator", _explode)
    runner = engine._create_gpumd_runner()
    assert type(runner).__name__ == "NativeGPUMDRunner"
    identity = runner.calculator
    assert identity.task == "bulk"
    assert identity.info()["nep_runtime"] == "gpumd"
    # The identity object is not an ASE calculator and says so.
    with pytest.raises(RuntimeError, match="not an ASE calculator"):
        identity.get_calculator()


def test_engine_keeps_the_ase_runtime_on_the_ase_runner(tmp_path):
    model = _nep_model(tmp_path)
    engine = CalculationEngine.from_config(
        EngineConfig(
            calc_type="sp",
            model_path=model,
            model_type="nep",
            task="bulk",
            device="cpu",
            calculator_options={"nep_runtime": "ase"},
        )
    )
    assert engine.nep_runtime == "ase"
    assert engine.uses_native_gpumd is False


def test_gpunep_emits_the_coordination_cost_warning(tmp_path):
    model = _nep_model(tmp_path)
    binary = _gpumd_executable(tmp_path)
    engine = CalculationEngine.from_config(
        EngineConfig(
            calc_type="sp",
            model_path=model,
            model_type="nep",
            task="bulk",
            device="cuda",
            calculator_options={
                "nep_runtime": "ase",
                "nep_gpumd_command": str(binary),
            },
        )
    )
    with pytest.warns(UserWarning, match="subprocess"):
        engine._create_calculator()


# -------------------------------------------------------- entry-point parity


def test_cli_api_incar_queue_agree_on_the_nep_config(tmp_path):
    model = _nep_model(tmp_path)
    binary = _gpumd_executable(tmp_path)

    resolved = resolve_config(
        calc_type="md",
        cli={
            "model_type": "nep",
            "model_path": str(model),
            "task": "bulk",
            "device": "cuda:0",
            "nep_runtime": "gpumd",
            "nep_gpumd_command": str(binary),
        },
    )
    command = build_mliport_command(
        "md",
        "/tmp/s.xyz",
        str(model),
        model_type="nep",
        python="/usr/bin/python",
        output_dir="/tmp/out",
        options={
            "task": "bulk",
            "device": "cuda:0",
            "nep_runtime": "gpumd",
            "nep_gpumd_command": str(binary),
        },
    )
    joined = " ".join(command)
    assert "--nep-runtime gpumd" in joined
    assert f"--gpumd-command {binary}" in joined
    assert resolved.calculator_options["nep_runtime"] == "gpumd"


def test_incar_template_declares_the_runtime_as_required():
    template = build_incar_default("sp", "nep")
    assert "NEP_RUNTIME" in template
    assert "NEP_GPUMD_COMMAND" in template
    # The template must not pick a runtime silently for the user.
    assert "NEP_RUNTIME = ase" in template
    assert "required" in template


def test_incar_rejects_an_unknown_model_type():
    incar = IncarConfig({"MODEL_TYPE": "nep2"}, source_name="INCAR")
    errors = incar.validate()
    assert any("Invalid MODEL_TYPE" in error for error in errors)


# ------------------------------------------------------------------- doctor


def test_doctor_declares_nep_with_both_runtimes():
    from mliport.doctor import _ENGINE_SPECS

    spec = _ENGINE_SPECS["nep"]
    assert spec["distribution"] == "calorine"
    assert spec["module"] == "calorine.calculators"
    # NEP is not a torch/tensorflow framework backend.
    assert spec["framework"] == "numpy"
    assert "gpumd" in spec["install"]


# -------------------------------------------------------------- model record


def test_model_record_carries_the_nep_runtime_identity(tmp_path, monkeypatch):
    """Two runtimes of one nep.txt must not be interchangeable evidence."""
    from mliport import model_record

    model = _nep_model(tmp_path)
    resolved = resolve_config(
        calc_type="sp",
        cli={
            "model_type": "nep",
            "model_path": str(model),
            "task": "bulk",
            "nep_runtime": "ase",
        },
    )

    class _Stub:
        task = "bulk"
        inference_mode = "default"

        def info(self):
            return {
                "model_type": "nep",
                "model_path": str(model),
                "task": "bulk",
                "nep_runtime": "ase",
                "default_dtype": "float64",
                "per_call_subprocess": False,
                "nep_element_types": ["Ba", "Zr", "O"],
                "nep_model_version": "nep4",
                "backend_version": "4.1",
                "framework_versions": {"calorine": "4.1"},
                "actual_device_type": "cpu",
            }

    record = model_record.build_model_record(_Stub(), resolved)
    assert record["nep_runtime"] == "ase"
    assert record["numeric_precision"] == "float64"
    assert record["per_call_subprocess"] is False
    assert record["nep_element_types"] == ["Ba", "Zr", "O"]
    assert record["nep_model_version"] == "nep4"
    assert record["backend_distribution"] == "calorine"
    assert record["model_sha256"] == model_record.hash_model_path(
        model, error_class=ValueError
    )


# ---------------------------------------------------------- API surface


def test_api_exposes_nep_runtime_and_forwards_it(tmp_path, monkeypatch):
    """The Python API must accept ``nep_runtime`` explicitly, not via a bag."""
    import inspect

    from mliport.api import run_md, run_optimization

    for function in (run_single_point, run_optimization, run_md):
        signature = inspect.signature(function)
        assert "nep_runtime" in signature.parameters
        assert "nep_gpumd_command" in signature.parameters

    model = _nep_model(tmp_path)
    captured: dict[str, object] = {}

    def _fake_resolve(
        calc_type,
        model_path,
        cli,
        output_dir,
        job_name,
        settings_path,
        model_alias,
        profile,
        strict_config,
        *,
        overwrite=False,
    ):
        from mliport.config.resolver import resolve_config as real_resolve

        cli = dict(cli)
        if model_path:
            cli.setdefault("model_path", model_path)
        captured["cli"] = dict(cli)
        captured["resolved"] = real_resolve(
            calc_type=calc_type,
            cli=cli,
            model_alias_name=model_alias,
            profile_name=profile,
        )
        return EngineConfig.from_resolved(captured["resolved"])

    monkeypatch.setattr("mliport.api._api_resolve", _fake_resolve)

    class _Stop(Exception):
        pass

    def _stop(*_args, **_kwargs):
        raise _Stop

    monkeypatch.setattr(CalculationEngine, "run", _stop)
    with pytest.raises(_Stop):
        run_single_point(
            structure=_minimal_atoms(),
            model_path=str(model),
            model_type="nep",
            task="bulk",
            device="cpu",
            nep_runtime="ase",
        )
    assert captured["cli"]["nep_runtime"] == "ase"


def _minimal_atoms():
    from ase import Atoms

    return Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


# ------------------------------------------------------------ docs parity


def test_backend_docs_exist_in_both_languages():
    docs = Path(__file__).resolve().parents[3] / "docs" / "backends"
    english = docs / "nep.md"
    chinese = docs / "nep.zh.md"
    assert english.is_file()
    assert chinese.is_file()
    text = english.read_text(encoding="utf-8")
    for required in (
        "NEP_RUNTIME",
        "gpumd",
        "--gpumd-command",
        "float32",
        "float64",
        "unwrapped",
    ):
        assert required in text, required


def test_capability_matrix_declares_the_nep_runtimes():
    matrix = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "validation"
            / "science"
            / "capability_matrix.json"
        ).read_text(encoding="utf-8")
    )
    text = json.dumps(matrix)
    assert "NEP" in text
