"""Tests for the NEP (Calorine) calculator wrapper.

Pure CPU and backend-free: no ``calorine`` or GPUMD install is required. The
lazy backend import is simulated by stubbing ``calorine.calculators`` in
``sys.modules``, so the tests can assert exactly which constructor arguments
each runtime receives and that no import happens before first use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mliport.base_calculator import BaseMLIPCalculator
from mliport.calculators.nep_calc import (
    NEPCalculatorWrapper,
    parse_nep_model_header,
)

#: Minimal but structurally faithful NEP4 header: version + element count +
#: elements, then the cutoff row (radial/angular cutoffs plus the two neighbour
#: list capacities that must not be mistaken for cutoffs).
NEP_MODEL_TEXT = "nep4 3 Ba Ti O\ncutoff 8 4 60 0\nn_max 8 6\n"


def _write_model(
    tmp_path: Path, text: str = NEP_MODEL_TEXT, name: str = "nep.txt"
) -> Path:
    """Write a ``nep.txt`` model file into ``tmp_path``."""
    model = tmp_path / name
    model.write_text(text, encoding="utf-8")
    return model


def _make_executable(tmp_path: Path, name: str = "gpumd") -> Path:
    """Create an executable stand-in for the GPUMD binary."""
    executable = tmp_path / name
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _stub_calorine(
    monkeypatch, *, implemented_properties=("energy", "forces", "stress")
) -> dict[str, list[dict[str, object]]]:
    """Install fake ``calorine.calculators`` classes and record their calls."""
    calls: dict[str, list[dict[str, object]]] = {"cpu": [], "gpu": []}

    class _FakeNEP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.implemented_properties = list(implemented_properties)

    class _CPUNEP(_FakeNEP):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            calls["cpu"].append(kwargs)

    class _GPUNEP(_FakeNEP):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            calls["gpu"].append(kwargs)

    monkeypatch.setitem(sys.modules, "calorine", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "calorine.calculators",
        SimpleNamespace(CPUNEP=_CPUNEP, GPUNEP=_GPUNEP),
    )
    return calls


def _hide_calorine(monkeypatch) -> None:
    """Simulate an environment without ``calorine`` installed."""
    monkeypatch.setitem(sys.modules, "calorine", None)
    monkeypatch.setitem(sys.modules, "calorine.calculators", None)


# ---------------------------------------------------------------------------
# Contract / construction
# ---------------------------------------------------------------------------


def test_wrapper_implements_base_contract(tmp_path):
    wrapper = NEPCalculatorWrapper(_write_model(tmp_path))
    assert isinstance(wrapper, BaseMLIPCalculator)
    assert wrapper.task == "bulk"
    assert wrapper.inference_mode == "default"


@pytest.mark.parametrize("runtime", ["ase", "ASE", " ase ", "Ase"])
def test_ase_runtime_spellings_are_accepted(runtime, tmp_path):
    wrapper = NEPCalculatorWrapper(_write_model(tmp_path), nep_runtime=runtime)
    assert wrapper.info()["nep_runtime"] == "ase"


@pytest.mark.parametrize("runtime", ["", "auto", "cupy", "torch", "ase2"])
@pytest.mark.parametrize("model_name", ["nep.txt"])
def test_unknown_runtime_is_rejected(runtime, model_name, tmp_path):
    with pytest.raises(ValueError, match="nep_runtime") as excinfo:
        NEPCalculatorWrapper(
            _write_model(tmp_path, name=model_name), nep_runtime=runtime
        )
    message = str(excinfo.value)
    assert "ase" in message
    assert "gpumd" in message


def test_gpumd_runtime_is_rejected_and_points_at_native_runner(tmp_path):
    """The native runtime is a separate runner, never a secret ASE fallback."""
    with pytest.raises(ValueError) as excinfo:
        NEPCalculatorWrapper(_write_model(tmp_path), nep_runtime=" GPUMD ")
    message = str(excinfo.value)
    assert "not an ASE calculator" in message
    assert "mliport.gpumd" in message


@pytest.mark.parametrize("task", ["bulk", "BULK", " molecule "])
def test_valid_tasks_are_accepted(task, tmp_path):
    wrapper = NEPCalculatorWrapper(_write_model(tmp_path), task=task)
    assert wrapper.task == task.strip().lower()


@pytest.mark.parametrize("task", ["omat", "", "periodic", "bulk-molecule"])
def test_invalid_task_is_rejected(task, tmp_path):
    with pytest.raises(ValueError, match="task"):
        NEPCalculatorWrapper(_write_model(tmp_path), task=task)


def test_missing_model_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="NEP model file not found"):
        NEPCalculatorWrapper(tmp_path / "nep.txt")


def test_directory_model_path_is_rejected(tmp_path):
    """A directory has no single model identity, so it cannot be recorded."""
    model_dir = tmp_path / "nep_model"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="directory") as excinfo:
        NEPCalculatorWrapper(model_dir)
    assert "single model identity" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Device validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "expected_type", "expected_index"),
    [
        ("cpu", "cpu", None),
        ("CPU", "cpu", None),
        ("cuda", "cuda", 0),
        ("gpu", "cuda", 0),
        ("cuda:0", "cuda", 0),
        (" CUDA:3 ", "cuda", 3),
    ],
)
def test_device_normalization(device, expected_type, expected_index, tmp_path):
    wrapper = NEPCalculatorWrapper(
        _write_model(tmp_path), device=device, gpumd_command=_make_executable(tmp_path)
    )
    info = wrapper.info()
    assert info["device"] == expected_type
    assert info["actual_device_type"] == expected_type
    assert info["actual_device_logical_index"] == expected_index


@pytest.mark.parametrize(
    "device", ["mps", "xpu", "cuda:abc", "cuda:", "", "cuda:-1", "cuda:1:2", "tpu"]
)
def test_invalid_device_is_rejected(device, tmp_path):
    with pytest.raises(ValueError, match="Invalid NEP device"):
        NEPCalculatorWrapper(
            _write_model(tmp_path),
            device=device,
            gpumd_command=_make_executable(tmp_path),
        )


def test_bare_integer_device_is_rejected(tmp_path):
    """``--device 0`` must not quietly mean ``cuda:0``."""
    with pytest.raises(ValueError, match="Invalid NEP device"):
        NEPCalculatorWrapper(
            _write_model(tmp_path), device=0, gpumd_command=_make_executable(tmp_path)
        )


@pytest.mark.parametrize(
    "device",
    ["cuda", "cuda:1", pytest.param("gpu", id="gpu-synonym")],
)
def test_cuda_requires_explicit_gpumd_command(device, tmp_path):
    with pytest.raises(ValueError, match="gpumd_command") as excinfo:
        NEPCalculatorWrapper(_write_model(tmp_path), device=device)
    assert "GPUNEP" in str(excinfo.value)


def test_cpu_needs_no_gpumd_command(tmp_path):
    wrapper = NEPCalculatorWrapper(_write_model(tmp_path), device="cpu")
    assert wrapper.info()["gpumd_command"] is None


# ---------------------------------------------------------------------------
# GPUMD command validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "/usr/bin/gpumd && rm -rf /",
        "/usr/bin/gpumd > output",
        "echo hi | /usr/bin/gpumd",
        "GPUMD=/usr/bin/gpumd; $GPUMD",
        "`/usr/bin/gpumd`",
        "/usr/bin/gpumd < input",
        "$HOME/bin/gpumd",
        "/usr/bin/gpumd (x)",
        "/usr/bin/gpumd\nrm -rf /",
    ],
)
def test_gpumd_command_shell_string_is_rejected(command, tmp_path):
    """GPUNEP builds its own shell line; a command string is an injection."""
    with pytest.raises(ValueError, match="shell command string"):
        NEPCalculatorWrapper(
            _write_model(tmp_path), device="cuda", gpumd_command=command
        )


def test_gpumd_command_must_be_executable(tmp_path):
    gpumd = tmp_path / "gpumd"
    gpumd.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    gpumd.chmod(0o644)
    with pytest.raises(ValueError, match="not executable"):
        NEPCalculatorWrapper(_write_model(tmp_path), device="cuda", gpumd_command=gpumd)


def test_missing_gpumd_command_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="GPUMD executable not found"):
        NEPCalculatorWrapper(
            _write_model(tmp_path),
            device="cuda",
            gpumd_command=tmp_path / "no-gpumd",
        )


def test_gpumd_command_directory_is_rejected(tmp_path):
    command_dir = tmp_path / "gpumd-bin"
    command_dir.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        NEPCalculatorWrapper(
            _write_model(tmp_path), device="cuda", gpumd_command=command_dir
        )


def test_gpumd_command_path_with_spaces_is_accepted(tmp_path):
    gpumd = _make_executable(tmp_path, name="my gpumd binary")
    wrapper = NEPCalculatorWrapper(
        _write_model(tmp_path), device="cuda", gpumd_command=gpumd
    )
    assert wrapper.info()["gpumd_command"] == str(gpumd)


def test_gpumd_command_is_validated_even_on_cpu(tmp_path):
    """A bad path must not be ignored just because CPUNEP will not use it."""
    with pytest.raises(FileNotFoundError, match="GPUMD executable not found"):
        NEPCalculatorWrapper(
            _write_model(tmp_path), device="cpu", gpumd_command=tmp_path / "nope"
        )


# ---------------------------------------------------------------------------
# Lazy import and construction arguments
# ---------------------------------------------------------------------------


def test_calorine_is_imported_lazily(tmp_path, monkeypatch):
    """Construction must work with no calorine; get_calculator raises ImportError."""
    _hide_calorine(monkeypatch)
    wrapper = NEPCalculatorWrapper(_write_model(tmp_path))
    assert wrapper._calculator is None
    with pytest.raises(ImportError) as excinfo:
        wrapper.get_calculator()
    message = str(excinfo.value)
    assert "calorine" in message
    assert ".venv-nep" in message


def test_cpunep_receives_model_filename_and_is_cached(tmp_path, monkeypatch):
    calls = _stub_calorine(monkeypatch)
    model = _write_model(tmp_path)
    wrapper = NEPCalculatorWrapper(model, device="cpu")

    calculator = wrapper.get_calculator()
    assert calls["cpu"] == [{"model_filename": str(model)}]
    assert calls["gpu"] == []
    assert wrapper.get_calculator() is calculator
    assert len(calls["cpu"]) == 1


@pytest.mark.parametrize(
    ("device", "expected_index"),
    [("cuda", 0), ("gpu", 0), ("cuda:0", 0), ("cuda:2", 2)],
)
def test_gpunep_receives_command_and_gpu_index(
    device, expected_index, tmp_path, monkeypatch
):
    calls = _stub_calorine(monkeypatch)
    model = _write_model(tmp_path)
    gpumd = _make_executable(tmp_path)
    wrapper = NEPCalculatorWrapper(model, device=device, gpumd_command=gpumd)

    wrapper.get_calculator()
    assert calls["gpu"] == [
        {
            "model_filename": str(model),
            "command": str(gpumd),
            "gpu_identifier_index": expected_index,
        }
    ]
    assert calls["cpu"] == []


# ---------------------------------------------------------------------------
# has_stress / info()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("properties", "expected"),
    [
        (("energy", "forces", "stress"), True),
        (("energy", "energies", "forces", "stress", "stresses"), True),
        (("energy", "forces", "energies", "stresses"), False),
    ],
)
def test_has_stress_follows_implemented_properties(
    properties, expected, tmp_path, monkeypatch
):
    """has_stress is derived from the live calculator, never hard-coded."""
    _stub_calorine(monkeypatch, implemented_properties=properties)
    wrapper = NEPCalculatorWrapper(_write_model(tmp_path))
    assert wrapper.has_stress is expected
    info = wrapper.info()
    assert info["implemented_properties"] == list(properties)
    assert info["has_stress"] is expected


def test_info_before_build_reports_none_for_live_fields(tmp_path, monkeypatch):
    """info() must never import the backend or crash before construction."""
    _hide_calorine(monkeypatch)
    wrapper = NEPCalculatorWrapper(_write_model(tmp_path), task="molecule")
    info = wrapper.info()

    assert info["model_type"] == "nep"
    assert info["nep_runtime"] == "ase"
    assert info["task"] == "molecule"
    assert info["device"] == "cpu"
    assert info["implemented_properties"] is None
    assert info["has_stress"] is None
    assert info["backend_distribution"] == "calorine"
    assert set(info["framework_versions"]) == {"calorine", "ase"}
    assert info["default_dtype"] == "float64"
    assert info["actual_device_type"] == "cpu"
    assert info["actual_device_logical_index"] is None
    assert info["nep_element_types"] == ["Ba", "Ti", "O"]
    assert info["nep_model_version"] == "nep4"
    assert info["per_call_subprocess"] is False
    assert wrapper._calculator is None


def test_info_for_cpunep(tmp_path, monkeypatch):
    _stub_calorine(monkeypatch)
    wrapper = NEPCalculatorWrapper(_write_model(tmp_path))
    wrapper.get_calculator()
    info = wrapper.info()

    assert info["default_dtype"] == "float64"
    assert info["per_call_subprocess"] is False
    assert info["actual_device_type"] == "cpu"
    assert info["actual_device_logical_index"] is None
    assert info["has_stress"] is True
    assert info["nep_cutoff_radial"] == 8.0
    assert info["nep_cutoff_angular"] == 4.0


def test_info_for_gpunep(tmp_path, monkeypatch):
    _stub_calorine(monkeypatch)
    gpumd = _make_executable(tmp_path)
    wrapper = NEPCalculatorWrapper(
        _write_model(tmp_path), device="cuda:1", gpumd_command=gpumd
    )
    wrapper.get_calculator()
    info = wrapper.info()

    assert info["default_dtype"] == "float32"
    assert info["per_call_subprocess"] is True
    assert info["actual_device_type"] == "cuda"
    assert info["actual_device_logical_index"] == 1
    assert info["gpumd_command"] == str(gpumd)
    assert info["has_stress"] is True


def test_info_reports_malformed_header_without_raising(tmp_path, monkeypatch):
    """A broken model file is reported in-band instead of exploding a writer."""
    _hide_calorine(monkeypatch)
    model = tmp_path / "nep.txt"
    model.write_text("definitely not a nep model\n", encoding="utf-8")
    wrapper = NEPCalculatorWrapper(model)

    info = wrapper.info()
    assert info["nep_element_types"] is None
    assert info["nep_model_version"] is None
    assert "Malformed NEP model header" in info["nep_model_header_error"]

    with pytest.raises(ValueError, match="Malformed NEP model header"):
        wrapper.require_elements(["Ba"])


# ---------------------------------------------------------------------------
# parse_nep_model_header
# ---------------------------------------------------------------------------


def test_parse_nep_model_header_reads_the_header(tmp_path):
    header = parse_nep_model_header(_write_model(tmp_path))

    assert header["element_types"] == ["Ba", "Ti", "O"]
    assert header["model_version"] == "nep4"
    assert header["model_token"] == "nep4"
    assert header["cutoff_radial"] == 8.0
    assert header["cutoff_angular"] == 4.0
    # The trailing numbers are neighbour-list capacities, not cutoffs.
    assert header["max_neighbours_radial"] == 60.0
    assert header["max_neighbours_angular"] == 0.0


def test_parse_nep_model_header_handles_suffixed_version_and_zbl_row(tmp_path):
    model = _write_model(tmp_path, "nep4_zbl 2 Ba O\nzbl 0.75 1.5\ncutoff 6 4 86 26\n")
    header = parse_nep_model_header(model)

    assert header["model_token"] == "nep4_zbl"
    assert header["model_version"] == "nep4"
    assert header["element_types"] == ["Ba", "O"]
    assert header["cutoff_radial"] == 6.0
    assert header["cutoff_angular"] == 4.0


def test_parse_nep_model_header_accepts_minimal_cutoff_row(tmp_path):
    header = parse_nep_model_header(_write_model(tmp_path, "nep4 1 C\ncutoff 8 4\n"))
    assert header["cutoff_radial"] == 8.0
    assert header["cutoff_angular"] == 4.0
    assert header["max_neighbours_radial"] is None


def test_parse_nep_model_header_allows_missing_cutoff_row(tmp_path):
    header = parse_nep_model_header(_write_model(tmp_path, "nep4 1 C\n"))
    assert header["element_types"] == ["C"]
    assert header["cutoff_radial"] is None
    assert header["cutoff_angular"] is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n\n   \n",
        "draw 3 Ba Ti O\n",
        "nep4\n",
        "nep4 3\n",
        "nep4 3 Ba 7\n",
        "nep4 3 Ba Ti O\ncutoff 8\n",
        "nep4 3 Ba Ti O\ncutoff 8 x 60 0\n",
        "nep4 3 Ba Ti O\ncutoff -8 4\n",
    ],
)
def test_parse_nep_model_header_rejects_malformed_models(text, tmp_path):
    model = _write_model(tmp_path, text)
    with pytest.raises(ValueError):
        parse_nep_model_header(model)


def test_parse_nep_model_header_rejects_unreadable_path(tmp_path):
    with pytest.raises(ValueError, match="Cannot read NEP model file"):
        parse_nep_model_header(tmp_path / "missing.txt")


# ---------------------------------------------------------------------------
# require_elements
# ---------------------------------------------------------------------------


def test_require_elements_accepts_supported_elements(tmp_path):
    """Works without calorine installed: the header parse is stdlib only."""
    wrapper = NEPCalculatorWrapper(_write_model(tmp_path))
    assert wrapper.require_elements(["Ba", "Ti", "O", "Ba"]) is None


def test_require_elements_rejects_unsupported_elements(tmp_path):
    wrapper = NEPCalculatorWrapper(_write_model(tmp_path))
    with pytest.raises(ValueError) as excinfo:
        wrapper.require_elements(["Ba", "Fe", "O", "Fe"])
    message = str(excinfo.value)
    assert "['Fe']" in message
    assert "Ba" in message
    assert "Ti" in message
