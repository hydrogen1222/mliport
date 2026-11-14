"""The callable-check probe must work on the CPU and on any structure.

Spec #61 (model-support §6.2): ``scripts/validate_runtime.py`` stops assuming
``cuda:0``, UMA's ``omat`` task and a Cu cell. These tests drive the probe with
a stub calculator: no backend weights, no GPU and no downloads.
"""

from __future__ import annotations

import json

import pytest
from ase.calculators.lj import LennardJones

from scripts import validate_runtime


class StubWrapper:
    """Minimal stand-in for a backend wrapper (no torch, no weights)."""

    def __init__(self, model_path, *, info):
        self._calculator = LennardJones()
        self._info = {"model_path": str(model_path), **info}

    def get_calculator(self):
        return self._calculator

    def info(self):
        return dict(self._info)


def _stub_info(device, **overrides):
    info = {
        "model_type": "mace",
        "task": "bulk",
        "actual_device_type": "cpu" if device == "cpu" else "cuda",
        "actual_device_logical_index": None if device == "cpu" else 0,
        "actual_device_uuid": None if device == "cpu" else "GPU-stub-uuid",
        "model_precision": ["float64"],
        "default_dtype": "float64",
        "backend_version": "0.3.16",
        "framework_versions": {"torch": "2.8.0"},
        "implemented_properties": ["energy", "forces", "stress"],
    }
    info.update(overrides)
    return info


def _install_stub(monkeypatch, device, **overrides):
    def create(backend, model_path, **kwargs):
        return StubWrapper(model_path, info=_stub_info(device, **overrides))

    monkeypatch.setattr(validate_runtime.CalculatorFactory, "create", create)


# ------------------------------------------------------------ arguments
def test_device_is_required_and_typed(tmp_path):
    with pytest.raises(SystemExit):
        validate_runtime.parse_args(["mace", str(tmp_path / "m"), str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        validate_runtime.parse_args(["mace", "m", "o", "--device", "gpu"])
    with pytest.raises(SystemExit):
        validate_runtime.parse_args(["mace", "m", "o", "--device", "cuda"])
    assert (
        validate_runtime.parse_args(["mace", "m", "o", "--device", "cpu"]).device
        == "cpu"
    )
    parsed = validate_runtime.parse_args(["mace", "m", "o", "--device", "cuda:2"])
    assert parsed.device == "cuda:2"


def test_uma_task_must_be_explicit_and_others_default_to_bulk():
    with pytest.raises(SystemExit):
        validate_runtime.parse_args(["uma", "m", "o", "--device", "cpu"])
    parsed = validate_runtime.parse_args(
        ["uma", "m", "o", "--device", "cpu", "--task", "omat"]
    )
    assert parsed.task == "omat"
    parsed = validate_runtime.parse_args(["dpa", "m", "o", "--device", "cpu"])
    assert parsed.task == "bulk"


def test_evidence_id_names_the_device_type():
    assert (
        validate_runtime.evidence_id("cpu", "mace", "a" * 64, "float64", True)
        == f"cpu-mace-{'a' * 12}-float64-cacheTrue"
    )
    assert (
        validate_runtime.evidence_id("cuda:1", "dpa", "b" * 64, "native", False)
        == f"cuda-dpa-{'b' * 12}-native-cacheFalse"
    )


# --------------------------------------------------------- device block
def test_cpu_identity_needs_no_gpu_uuid():
    identity = validate_runtime.device_identity("cpu", _stub_info("cpu"))
    assert identity["actual_device_type"] == "cpu"
    assert "gpu_uuid_sha256" not in identity


def test_cpu_identity_rejects_a_silent_gpu_run():
    with pytest.raises(RuntimeError, match="Requested cpu"):
        validate_runtime.device_identity("cpu", _stub_info("cuda"))


def test_cuda_identity_keeps_the_uuid_confirmation():
    identity = validate_runtime.device_identity("cuda:0", _stub_info("cuda"))
    assert identity["gpu_uuid_sha256"]

    broken = _stub_info("cuda", actual_device_uuid=None)
    with pytest.raises(RuntimeError, match="UUID"):
        validate_runtime.device_identity("cuda:0", broken)

    wrong_ordinal = _stub_info("cuda", actual_device_logical_index=1)
    with pytest.raises(RuntimeError, match="cuda:1"):
        validate_runtime.device_identity("cuda:0", wrong_ordinal)

    cuda_reported_cpu = _stub_info("cpu")
    with pytest.raises(RuntimeError, match="Requested cuda:0"):
        validate_runtime.device_identity("cuda:0", cuda_reported_cpu)


# ------------------------------------------------------------ CPU run
def test_cpu_run_writes_a_report_without_a_gpu_block(tmp_path, monkeypatch):
    model = tmp_path / "model.model"
    model.write_bytes(b"stub model")
    output = tmp_path / "report.json"
    _install_stub(monkeypatch, "cpu")
    validate_runtime.main(
        ["mace", str(model), str(output), "--device", "cpu", "--no-cache"]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "mliport.runtime-validation/2"
    assert report["device"] == "cpu"
    assert report["task"] == "bulk"
    assert report["evidence_id"].startswith("cpu-mace-")
    assert "gpu" not in report
    assert "gpu_uuid_sha256" not in report["runtime"]
    assert "peak_vram_MiB" not in report
    assert report["fixture"]["fixture"] == "built-in fcc Cu"
    assert report["status"] == "passed"
    assert report["workloads"]["single_point"]["status"] == "passed"


def test_cuda_run_records_the_gpu_block(tmp_path, monkeypatch):
    model = tmp_path / "model.model"
    model.write_bytes(b"stub model")
    output = tmp_path / "report.json"
    _install_stub(monkeypatch, "cuda")
    monkeypatch.setattr(
        validate_runtime, "gpu_record", lambda: {"name": "stub", "driver": "1"}
    )
    monkeypatch.setattr(validate_runtime, "peak_vram_MiB", lambda framework: 1.5)
    validate_runtime.main(
        ["mace", str(model), str(output), "--device", "cuda:0", "--no-cache"]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["evidence_id"].startswith("cuda-mace-")
    assert report["runtime"]["gpu_uuid_sha256"]
    assert report["gpu"] == {"name": "stub", "driver": "1"}
    assert report["peak_vram_MiB"] == 1.5


def test_structure_option_replaces_the_cu_fixture(tmp_path, monkeypatch):
    model = tmp_path / "model.model"
    model.write_bytes(b"stub model")
    structure = tmp_path / "cell.xyz"
    structure.write_text(
        '2\nLattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0" pbc="T T T"\n'
        "Ar 0.0 0.0 0.0\nAr 1.5 0.0 0.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    _install_stub(monkeypatch, "cpu")
    validate_runtime.main(
        [
            "mace",
            str(model),
            str(output),
            "--device",
            "cpu",
            "--structure",
            str(structure),
            "--task",
            "bulk",
            "--no-cache",
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["fixture"]["fixture"] == "user structure"
    assert report["fixture"]["symbols"] == ["Ar"]
    assert report["fixture"]["atoms"] == 2
    assert len(report["fixture"]["structure_sha256"]) == 64
    assert str(structure) not in json.dumps(report)


@pytest.mark.parametrize("backend", ["dpa", "grace"])
def test_isolated_backends_need_process_level_device_isolation(backend):
    """These adapters refuse to run in a process that never isolated devices."""
    with pytest.raises(SystemExit, match="CUDA_VISIBLE_DEVICES"):
        validate_runtime.check_process_isolation(backend, {})
    validate_runtime.check_process_isolation(backend, {"CUDA_VISIBLE_DEVICES": ""})
    for backend in ("mace", "uma"):
        validate_runtime.check_process_isolation(backend, {})


def test_grace_memory_cap_is_only_sent_to_a_gpu(tmp_path, monkeypatch):
    """A GPU-only GRACE option must not be passed on a CPU run."""
    model = tmp_path / "model.model"
    model.write_bytes(b"stub model")
    output = tmp_path / "report.json"
    seen = {}

    def create(backend, model_path, **kwargs):
        seen.update(kwargs)
        return StubWrapper(model_path, info=_stub_info("cpu", model_type="grace"))

    monkeypatch.setattr(validate_runtime.CalculatorFactory, "create", create)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    validate_runtime.main(
        ["grace", str(model), str(output), "--device", "cpu", "--no-cache"]
    )
    assert seen["gpu_memory_limit_mb"] is None
