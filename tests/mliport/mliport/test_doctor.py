"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path

from mliport.doctor import (
    _installed_dependency_conflicts,
    _should_warn_torch_mismatch,
    format_diagnostics,
    run_diagnostics,
)


def _controlled_environment(monkeypatch, **versions):
    installed = {"mliport": "2.0.0", **versions}
    monkeypatch.setattr(
        "mliport.doctor._distribution_version",
        lambda name: installed.get(name),
    )
    monkeypatch.setattr(
        "mliport.doctor._installed_dependency_conflicts", lambda names: []
    )
    monkeypatch.setattr("mliport.doctor.metadata.requires", lambda name: [])


def _mock_4090_runtime(monkeypatch, arch_list, *, total_memory=24564 * 1024**2):
    from mliport.gpu_setup import GpuInfo

    _controlled_environment(
        monkeypatch,
        **{"fairchem-core": "2.21.0", "torch": "2.8.0+cu128"},
    )
    monkeypatch.setattr(
        "mliport.doctor.detect_gpus",
        lambda: [GpuInfo("NVIDIA GeForce RTX 4090", 8, 9, "580", 24564)],
    )
    monkeypatch.setattr(
        "mliport.doctor._runtime_probe",
        lambda engine, device, framework=None: {
            "ok": True,
            "framework": "torch",
            "torch_version": "2.8.0+cu128",
            "cuda_runtime": "12.8",
            "cuda_available": True,
            "device_count": 1,
            "arch_list": arch_list,
            "gpus": [
                {
                    "name": "NVIDIA GeForce RTX 4090",
                    "major": 8,
                    "minor": 9,
                    "total_memory": total_memory,
                }
            ],
        },
    )


def test_doctor_runs_without_crashing():
    checks, failures = run_diagnostics()
    assert isinstance(checks, list)
    assert isinstance(failures, int)
    assert len(checks) > 0
    for c in checks:
        assert "name" in c
        assert "status" in c
        assert c["status"] in ("ok", "fail", "warn", "skip")


def test_doctor_format_output(monkeypatch):
    _controlled_environment(monkeypatch)
    monkeypatch.setattr("mliport.doctor.detect_gpus", lambda: None)
    checks, _ = run_diagnostics()
    output = format_diagnostics(checks)
    assert "mliport Environment Diagnostic" in output
    assert "Inventory complete" in output
    assert isinstance(output, str)
    assert len(output) > 0


def test_doctor_with_nonexistent_model_fails_closed(monkeypatch):
    _controlled_environment(monkeypatch)
    monkeypatch.setattr("mliport.doctor.detect_gpus", lambda: None)
    checks, failures = run_diagnostics(model_path="/nonexistent/model.pt")
    model_check = [c for c in checks if c["name"] == "Model file"]
    assert len(model_check) == 1
    assert model_check[0]["status"] == "fail"
    assert failures == 1


def test_inventory_never_imports_optional_backends(monkeypatch):
    _controlled_environment(
        monkeypatch,
        **{"deepmd-kit": "3.1.3", "tensorpotential": "0.6.0"},
    )
    monkeypatch.setattr("mliport.doctor.detect_gpus", lambda: None)

    def unexpected_probe(engine, device, *, framework=None):
        raise AssertionError("inventory mode must not import a backend")

    monkeypatch.setattr("mliport.doctor._runtime_probe", unexpected_probe)
    checks, failures = run_diagnostics()
    names = {c["name"]: c for c in checks}
    assert names["Engine runtime"]["status"] == "skip"
    assert names["NVIDIA driver"]["status"] == "skip"
    assert failures == 0


def test_cpu_target_does_not_require_cuda(monkeypatch):
    _controlled_environment(monkeypatch, tensorpotential="0.6.0")
    monkeypatch.setattr("mliport.doctor.detect_gpus", lambda: None)
    monkeypatch.setattr(
        "mliport.doctor._runtime_probe",
        lambda engine, device, framework=None: {
            "ok": True,
            "framework": "tensorflow",
            "tensorflow_version": "2.20.0",
            "cuda_available": False,
            "device_count": 0,
        },
    )

    checks, failures = run_diagnostics(engine="grace", device="cpu")
    names = {c["name"]: c for c in checks}

    assert names["NVIDIA driver"]["status"] == "skip"
    assert names["Compute device"] == {
        "name": "Compute device",
        "value": "CPU",
        "status": "ok",
    }
    assert failures == 0


def test_dpa_pb_model_selects_tensorflow_runtime_probe(monkeypatch, tmp_path):
    _controlled_environment(monkeypatch, **{"deepmd-kit": "3.1.3"})
    monkeypatch.setattr("mliport.doctor.detect_gpus", lambda: None)
    model = tmp_path / "legacy.pb"
    model.write_bytes(b"placeholder")
    observed = {}

    def fake_probe(engine, device, *, framework=None):
        observed.update(engine=engine, device=device, framework=framework)
        return {
            "ok": True,
            "framework": "tensorflow",
            "tensorflow_version": "2.20.0",
            "cuda_available": False,
            "device_count": 0,
        }

    monkeypatch.setattr("mliport.doctor._runtime_probe", fake_probe)
    monkeypatch.setattr(
        "mliport.doctor._model_probe",
        lambda *args, **kwargs: {"ok": True, "task": "bulk"},
    )

    _, failures = run_diagnostics(
        model_path=model,
        engine="dpa",
        device="cpu",
        task="bulk",
    )

    assert observed == {
        "engine": "dpa",
        "device": "cpu",
        "framework": "tensorflow",
    }
    assert failures == 0


def test_model_probe_requires_explicit_task(monkeypatch, tmp_path):
    _controlled_environment(monkeypatch, **{"deepmd-kit": "3.1.3"})
    monkeypatch.setattr("mliport.doctor.detect_gpus", lambda: None)
    monkeypatch.setattr(
        "mliport.doctor._runtime_probe",
        lambda *args, **kwargs: {
            "ok": True,
            "framework": "torch",
            "torch_version": "2.10.0",
            "cuda_available": False,
            "device_count": 0,
            "arch_list": [],
            "gpus": [],
        },
    )
    monkeypatch.setattr(
        "mliport.doctor._model_probe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("model load must not run without an explicit task")
        ),
    )
    model = tmp_path / "model.pt"
    model.write_bytes(b"placeholder")

    checks, failures = run_diagnostics(
        model_path=model,
        engine="dpa",
        device="cpu",
    )

    load = next(check for check in checks if check["name"] == "Model load")
    assert load["status"] == "fail"
    assert "explicit --task" in load["detail"]
    assert failures == 1


def test_model_and_structure_probe_is_reported(monkeypatch, tmp_path):
    _controlled_environment(monkeypatch, **{"mace-torch": "0.3.16"})
    monkeypatch.setattr("mliport.doctor.detect_gpus", lambda: None)
    monkeypatch.setattr(
        "mliport.doctor._runtime_probe",
        lambda *args, **kwargs: {
            "ok": True,
            "framework": "torch",
            "torch_version": "2.6.0",
            "cuda_available": False,
            "device_count": 0,
            "arch_list": [],
            "gpus": [],
        },
    )
    model = tmp_path / "model.model"
    structure = tmp_path / "POSCAR"
    model.write_bytes(b"placeholder")
    structure.write_text("placeholder", encoding="utf-8")
    observed = {}

    def fake_model_probe(engine, device, **kwargs):
        observed.update(engine=engine, device=device, **kwargs)
        return {
            "ok": True,
            "task": "bulk",
            "active_head": "PBE",
            "model_precision": "float64",
            "evaluation": {
                "atoms": 2,
                "energy_eV": -1.25,
                "max_force_eV_A": 0.05,
                "stress_checked": True,
            },
        }

    monkeypatch.setattr("mliport.doctor._model_probe", fake_model_probe)
    checks, failures = run_diagnostics(
        model_path=model,
        engine="mace",
        device="cpu",
        task="bulk",
        head="PBE",
        structure_path=structure,
    )

    names = {check["name"]: check for check in checks}
    assert names["Model load"]["status"] == "ok"
    assert names["Single-point smoke"]["status"] == "ok"
    assert observed["model_path"] == Path(model)
    assert observed["structure_path"] == Path(structure)
    assert observed["head"] == "PBE"
    assert failures == 0


def test_cuda_target_fails_when_framework_cannot_see_gpu(monkeypatch):
    from mliport.gpu_setup import GpuInfo

    _controlled_environment(monkeypatch, **{"mace-torch": "0.3.16", "torch": "2.6.0"})
    monkeypatch.setattr(
        "mliport.doctor.detect_gpus",
        lambda: [GpuInfo("NVIDIA P104-100", 6, 1, "580.167.08", 8192)],
    )
    monkeypatch.setattr(
        "mliport.doctor._runtime_probe",
        lambda engine, device, framework=None: {
            "ok": True,
            "framework": "torch",
            "torch_version": "2.6.0+cu124",
            "cuda_runtime": "12.4",
            "cuda_available": False,
            "device_count": 0,
            "arch_list": [],
            "gpus": [],
        },
    )

    checks, failures = run_diagnostics(engine="mace", device="cuda")
    names = {c["name"]: c for c in checks}

    assert names["Engine runtime"]["status"] == "ok"
    assert names["CUDA runtime"]["status"] == "fail"
    assert failures == 1


def _unexpected_failures(checks: list[dict]) -> list[dict]:
    """Failures other than the documented interpreter-vs-installer-pin one.

    The UMA installer pin (fairchem-core==2.21.0) requires Python >=3.11, so
    on a 3.10 interpreter doctor reports "UMA installer Python constraint"
    as fail by design. These tests exercise the GPU recommendation logic,
    not that constraint.
    """
    return [
        check
        for check in checks
        if check["status"] == "fail"
        and check["name"] != "UMA installer Python constraint"
    ]


def test_doctor_verifies_uma_torch_recommendation_on_4090(monkeypatch):
    """Regression: installer verification must not mix dict and object APIs."""
    _mock_4090_runtime(
        monkeypatch,
        ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
    )

    checks, failures = run_diagnostics(engine="uma", device="cuda")
    runtime_gpu = next(c for c in checks if c["name"] == "Runtime GPU 0")

    assert runtime_gpu["status"] == "ok"
    assert _unexpected_failures(checks) == []


def test_doctor_reports_missing_4090_kernel_without_crashing(monkeypatch):
    """A wheel without an Ada-compatible kernel must fail closed with advice."""
    _mock_4090_runtime(monkeypatch, ["sm_70"])

    checks, failures = run_diagnostics(engine="uma", device="cuda")
    runtime_gpu = next(c for c in checks if c["name"] == "Runtime GPU 0")

    assert runtime_gpu["status"] == "fail"
    assert "do not support sm_89" in runtime_gpu["detail"]
    assert "torch 2.8.0+cu128" in runtime_gpu["detail"]
    unexpected = _unexpected_failures(checks)
    assert len(unexpected) == 1 and unexpected[0]["name"] == "Runtime GPU 0"


def test_doctor_p40_dpa_recommends_exact_deepmd_cu126_build(monkeypatch):
    """A wrong cu128 DPA wheel must not receive UMA's torch 2.8 advice."""

    from mliport.gpu_setup import GpuInfo

    _controlled_environment(
        monkeypatch,
        **{"deepmd-kit": "3.1.3", "torch": "2.10.0+cu128"},
    )
    monkeypatch.setattr(
        "mliport.doctor.detect_gpus",
        lambda: [GpuInfo("Tesla P40", 6, 1, "580.95.05", 22912)],
    )
    monkeypatch.setattr(
        "mliport.doctor._runtime_probe",
        lambda engine, device, framework=None: {
            "ok": True,
            "framework": "torch",
            "torch_version": "2.10.0+cu128",
            "cuda_runtime": "12.8",
            "cuda_available": True,
            "device_count": 1,
            "arch_list": ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
            "gpus": [
                {
                    "name": "Tesla P40",
                    "major": 6,
                    "minor": 1,
                    "total_memory": 22912 * 1024**2,
                }
            ],
        },
    )

    checks, failures = run_diagnostics(engine="dpa", device="cuda")
    runtime_gpu = next(check for check in checks if check["name"] == "Runtime GPU 0")

    assert runtime_gpu["status"] == "fail"
    assert "2.10.0+cu126" in runtime_gpu["detail"]
    assert "torch==2.10.0+cu126" in runtime_gpu["detail"]
    assert "2.8.0" not in runtime_gpu["detail"]
    assert "does NOT match" in runtime_gpu["detail"]
    assert failures == 1


def test_doctor_falls_back_to_nvidia_smi_when_container_reports_zero_vram(
    monkeypatch,
):
    """A container runtime's zero memory value means unknown, not zero VRAM."""
    _mock_4090_runtime(
        monkeypatch,
        ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
        total_memory=0,
    )

    checks, failures = run_diagnostics(engine="uma", device="cuda")
    names = {check["name"]: check for check in checks}

    assert "24.0 GB" in names["Runtime GPU 0"]["value"]
    assert "Runtime GPU 0 VRAM" not in names
    assert _unexpected_failures(checks) == []


def test_backend_import_failure_is_not_reported_as_installed_runtime(monkeypatch):
    _controlled_environment(monkeypatch, **{"deepmd-kit": "3.1.3", "torch": "2.10.0"})
    monkeypatch.setattr("mliport.doctor.detect_gpus", lambda: None)
    monkeypatch.setattr(
        "mliport.doctor._runtime_probe",
        lambda engine, device, framework=None: {
            "ok": False,
            "error_type": "ImportError",
            "error": "undefined symbol: broken_abi",
        },
    )

    checks, failures = run_diagnostics(engine="dpa", device="cpu")
    runtime = next(c for c in checks if c["name"] == "Engine runtime")
    output = format_diagnostics(checks)

    assert runtime["status"] == "fail"
    assert "broken_abi" in runtime["detail"]
    assert failures == sum(c["status"] == "fail" for c in checks) == 1
    assert "1 required check failed" in output
    assert "2 required checks failed" not in output


def test_dependency_conflicts_detect_mace_e3nn_mismatch(monkeypatch):
    def fake_requires(name):
        return {
            "mace-torch": ["e3nn==0.4.4", "torch>=1.12"],
            "fairchem-core": ["e3nn>=0.5"],
        }[name]

    versions = {"e3nn": "0.6.0", "torch": "2.6.0"}
    monkeypatch.setattr("mliport.doctor.metadata.requires", fake_requires)
    monkeypatch.setattr("mliport.doctor.metadata.version", lambda name: versions[name])

    conflicts = _installed_dependency_conflicts(("mace-torch", "fairchem-core"))

    assert conflicts == ["mace-torch requires e3nn==0.4.4, but 0.6.0 is installed"]


def test_torch_recommendation_is_uma_specific():
    """DPA's ABI-matched Torch must not be replaced with UMA's recommendation."""
    assert _should_warn_torch_mismatch(
        uma_installed=True,
        installed_torch="2.10.0+cu126",
        recommended_torch="2.6.0+cu124",
    )
    assert not _should_warn_torch_mismatch(
        uma_installed=False,
        installed_torch="2.10.0+cu126",
        recommended_torch="2.6.0+cu124",
    )


def test_feature_inventory_flags_missing_optional_and_required(monkeypatch):
    """Doctor must report core/tui/analysis/transport/electrolyte/plotting."""
    import mliport.doctor as doctor

    def fake_version(name: str):
        return None if name == "kinisi" else "1.0"

    monkeypatch.setattr(doctor, "_distribution_version", fake_version)
    checks = {c["name"]: c for c in doctor.feature_inventory()}
    assert set(checks) >= {
        "Feature: core",
        "Feature: tui",
        "Feature: analysis",
        "Feature: transport",
        "Feature: electrolyte",
        "Feature: plotting",
    }
    assert checks["Feature: transport"]["status"] == "warn"
    assert checks["Feature: analysis"]["status"] == "ok"

    monkeypatch.setattr(
        doctor, "_distribution_version", lambda name: None if name == "numpy" else "1.0"
    )
    core = {c["name"]: c for c in doctor.feature_inventory()}["Feature: core"]
    assert core["status"] == "warn"
    assert core["required"] is True


def test_doctor_json_reports_features_and_failures(capsys):
    """Acceptance harnesses consume `doctor --json` instead of scraping text."""
    import json

    from mliport.cli import main

    rc = main(["doctor", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "mliport.doctor-report/1"
    assert isinstance(payload["failures"], int)
    names = {check["name"] for check in payload["checks"]}
    assert "Feature: core" in names
    assert "Feature: analysis" in names
    assert "Feature: transport" in names
    assert "Feature: electrolyte" in names
