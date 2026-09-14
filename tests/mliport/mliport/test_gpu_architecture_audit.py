"""GPU architecture audit tests (task book ARCH-T1..T6).

These tests are synthetic: they plan installations for compute capabilities
without owning that hardware. They never claim hardware verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mliport.install.compatibility import (
    ARCH_PROFILES,
    BACKENDS,
    BackendArchProfile,
    effective_cuda_channel,
)
from mliport.install.hardware import GpuInfo
from mliport.install.plan import InstallPlanError, generate_plan

REPO = Path(__file__).resolve().parents[3]


def _gpu(major: int, minor: int, name: str = "synthetic") -> GpuInfo:
    return GpuInfo(
        name=name,
        cc_major=major,
        cc_minor=minor,
        driver_version="580.173.02",
        vram_mib=16384,
        uuid=f"GPU-synthetic-{major}{minor}",
        physical_index=0,
    )


def _joined(plan) -> str:
    return "\n".join(" ".join(step.argv) for step in plan.steps)


@pytest.mark.parametrize("cc, expected_arch", [((6, 0), "pascal"), ((6, 1), "pascal")])
def test_arch_t1_t2_pascal_sm60_and_sm61_plan(cc, expected_arch):
    """ARCH-T1/T2: Pascal sm60 and sm61 both plan the cu126 legacy route."""
    for engine in ("mace", "uma"):
        plan = generate_plan([_gpu(*cc)], engines=[engine], device="cuda")
        assert plan.gpu_arch == expected_arch
        joined = _joined(plan)
        assert "torch==2.8.0+cu126" in joined, engine
        assert any("needs smoke test" in w for w in plan.warnings), engine

    dpa = generate_plan([_gpu(*cc)], engines=["dpa"], device="cuda")
    dpa_joined = _joined(dpa)
    assert "torch==2.10.0+cu126" in dpa_joined
    assert "mpich" in dpa_joined

    grace = generate_plan([_gpu(*cc)], engines=["grace"], device="cuda")
    grace_joined = _joined(grace)
    assert "tensorflow[and-cuda]==2.20.0" in grace_joined
    assert "+cu126" not in grace_joined


def test_arch_t3_volta_sm70_plan():
    """ARCH-T3: Volta uses the cu126 legacy route for all torch engines."""
    plan = generate_plan(
        [_gpu(7, 0, "V100")], engines=["mace", "dpa", "uma"], device="cuda"
    )
    assert plan.gpu_arch == "volta"
    joined = _joined(plan)
    assert "torch==2.8.0+cu126" in joined
    assert "torch==2.10.0+cu126" in joined


@pytest.mark.parametrize(
    "cc, expected_arch",
    [
        ((7, 5), "turing"),
        ((8, 0), "ampere"),
        ((8, 6), "ampere"),
        ((8, 9), "ada"),
        ((9, 0), "hopper"),
        ((10, 0), "blackwell"),
        ((12, 0), "blackwell"),
    ],
)
def test_modern_arches_use_cu128(cc, expected_arch):
    plan = generate_plan([_gpu(*cc)], engines=["mace", "dpa", "uma"], device="cuda")
    assert plan.gpu_arch == expected_arch
    joined = _joined(plan)
    assert "torch==2.8.0+cu128" in joined
    assert "torch==2.10.0+cu128" in joined


def test_grace_hopper_and_blackwell_are_experimental():
    for cc in ((9, 0), (10, 0), (12, 0)):
        plan = generate_plan([_gpu(*cc)], engines=["grace"], device="cuda")
        assert any("EXPERIMENTAL" in w for w in plan.warnings), cc


def test_arch_t4_unknown_compute_capability_fails_closed():
    """ARCH-T4: unknown CC never gets a guessed plan."""
    with pytest.raises(InstallPlanError):
        generate_plan([_gpu(11, 0)], engines=["mace"], device="cuda")

    auto = generate_plan([_gpu(11, 0)], engines=["mace"], device="auto")
    assert auto.gpu_arch == "cpu"
    assert any("unsupported" in w for w in auto.warnings)


def test_arch_t5_future_torch_version_requires_audit_mapping():
    """ARCH-T5: legacy/future torch versions need an explicit channel audit."""
    from mliport.install.compatibility import _torch_modern_cuda

    assert _torch_modern_cuda("2.8.0") == "cu128"
    assert _torch_modern_cuda("2.10.0") == "cu128"
    with pytest.raises(ValueError, match="audit|channel|version"):
        _torch_modern_cuda("2.15.0")

    future = BackendArchProfile(framework_version="2.15.0")
    with pytest.raises(ValueError):
        effective_cuda_channel(BACKENDS["mace"], ARCH_PROFILES["turing"], future)


def test_arch_t6_cu126_wheels_use_exact_local_version():
    """ARCH-T6: legacy plans pin the +cu126 local version, not bare PyPI."""
    joined = _joined(
        generate_plan([_gpu(6, 1)], engines=["mace", "dpa"], device="cuda")
    )
    assert "torch==2.8.0+cu126" in joined
    assert "torch==2.10.0+cu126" in joined
    assert "torch==2.8.0 " not in joined
    assert "torch==2.10.0 " not in joined


def test_dependabot_framework_upgrades_require_manual_review():
    """PyTorch 2.15 drops cu126; framework updates need a manual audit."""
    data = yaml.safe_load((REPO / ".github" / "dependabot.yml").read_text())
    pip_updates = [
        update for update in data["updates"] if update.get("package-ecosystem") == "pip"
    ]
    assert pip_updates
    ignored = set()
    for update in pip_updates:
        for rule in update.get("ignore", []) or []:
            ignored.add(rule.get("dependency-name"))
    for name in ("torch", "fairchem-core", "deepmd-kit", "tensorflow"):
        assert name in ignored, (
            f"{name} must require a manual architecture compatibility review; "
            f"see the PyTorch 2.15 CUDA 12.6 cutoff"
        )


def test_packaged_audit_matches_the_repository_report():
    """The doctor reads a packaged copy; it must match the audit source."""
    repo = REPO / "validation" / "compatibility" / "gpu_architecture_theory.json"
    packaged = (
        REPO
        / "mliport"
        / "mliport"
        / "data"
        / "compatibility"
        / "gpu_architecture_theory.json"
    )
    assert packaged.read_bytes() == repo.read_bytes()


def test_architecture_evidence_files_exist_and_are_consistent():
    report_dir = REPO / "validation" / "compatibility"
    report = report_dir / "GPU_ARCHITECTURE_THEORY_REPORT.md"
    data_path = report_dir / "gpu_architecture_theory.json"
    assert report.is_file()
    assert data_path.is_file()
    data = json.loads(data_path.read_text(encoding="utf-8"))
    assert data["schema"] == "mliport.gpu-architecture-theory/1"
    levels = {
        "hardware_verified",
        "binary_theory_verified",
        "resolver_only",
        "experimental",
        "unsupported",
    }
    for entry in data["entries"]:
        assert entry["support_classification"] in levels
        assert entry["real_hardware_tested"] is False or (
            entry["architecture"] == "volta" and entry["backend"] in BACKENDS
        )
