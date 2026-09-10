"""Tests for mlipx.install.plan — installation plan generation."""

from __future__ import annotations

import pytest

from mlipx.install.hardware import GpuInfo
from mlipx.install.plan import (
    InstallPlanError,
    generate_plan,
    normalize_engines,
)


def test_backend_python_intersection_before_clean_plan():
    with pytest.raises(InstallPlanError, match="fairchem-core==2.21.0"):
        generate_plan(None, ["uma"], python_version="3.10", device="cpu", clean=True)
    for py in ("3.11", "3.12"):
        assert generate_plan(None, ["uma"], python_version=py, device="cpu").steps
    for engine in ("mace", "dpa", "grace"):
        assert generate_plan(None, [engine], python_version="3.10", device="cpu").steps


def _gpu(name: str, major: int, minor: int, vram: int = 8192) -> GpuInfo:
    return GpuInfo(
        name=name,
        cc_major=major,
        cc_minor=minor,
        driver_version="580",
        vram_mib=vram,
    )


def _pip_cmds(plan, stage: str = "pip"):
    return [s for s in plan.steps if s.stage == stage]


def _all_argv(plan) -> str:
    return " ".join(" ".join(s.argv) for s in plan.steps)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_engines_dedup_and_alias() -> None:
    assert normalize_engines(["uma", "mace", "uma", "fairchem"]) == ["uma", "mace"]


def test_normalize_engines_unknown_raises() -> None:
    with pytest.raises(InstallPlanError):
        normalize_engines(["foo"])


def test_normalize_engines_empty_raises() -> None:
    with pytest.raises(InstallPlanError):
        normalize_engines([])


# ---------------------------------------------------------------------------
# UMA explicit install (never uv sync)
# ---------------------------------------------------------------------------


def test_uma_explicit_install_v100() -> None:
    """V100: UMA must use torch 2.8.0 + cu126 + fairchem-core, no uv sync."""
    plan = generate_plan([_gpu("V100", 7, 0)], ["uma"], verify=False)
    argv = _all_argv(plan)
    assert "uv sync" not in argv
    assert "torch==2.8.0" in argv
    assert "cu126" in argv
    assert "fairchem-core==2.21.0" in argv
    assert "-e ./mlipx" in argv


def test_uma_explicit_install_cpu() -> None:
    plan = generate_plan(None, ["uma"], device="cpu", verify=False)
    argv = _all_argv(plan)
    assert "uv sync" not in argv
    assert "torch==2.8.0+cpu" in argv
    assert "https://download.pytorch.org/whl/cpu" in argv
    assert "fairchem-core==2.21.0" in argv


# ---------------------------------------------------------------------------
# Architecture → CUDA channel
# ---------------------------------------------------------------------------


def test_v100_plan_all_engines() -> None:
    """V100 (Volta) → all torch engines use cu126; GRACE uses TF."""
    plan = generate_plan([_gpu("V100", 7, 0)], verify=False)
    argv = _all_argv(plan)
    assert "cu126" in argv
    assert "tensorflow[and-cuda]==2.20.0" in argv


def test_p40_dpa_plan_pins_exact_legacy_torch_build() -> None:
    """P40 must never resolve the same-version cu128 wheel from another index."""

    plan = generate_plan(
        [_gpu("Tesla P40", 6, 1)],
        ["dpa"],
        source="china",
        verify=False,
    )
    torch_step = next(
        step
        for step in plan.steps
        if any(arg.startswith("torch==") for arg in step.argv)
    )
    assert "torch==2.10.0+cu126" in torch_step.argv
    assert "--find-links" in torch_step.argv
    assert "cu126" in " ".join(torch_step.argv)
    assert "torch==2.10.0" not in torch_step.argv


def test_4090_plan_modern() -> None:
    """Ada (sm_89) → torch engines use cu128 (modern)."""
    plan = generate_plan([_gpu("RTX 4090", 8, 9)], ["uma", "mace", "dpa"], verify=False)
    argv = _all_argv(plan)
    assert "cu128" in argv
    assert "cu126" not in argv
    assert "torch==2.10.0+cu128" in argv
    assert not any("DPA" in w and "needs smoke test" in w for w in plan.warnings)


def test_mixed_gpu_uses_oldest() -> None:
    """4090 + P100 → conservative Pascal (cu126) selection."""
    plan = generate_plan(
        [_gpu("RTX 4090", 8, 9), _gpu("P100", 6, 0)], ["uma"], verify=False
    )
    assert plan.gpu_arch == "pascal"
    argv = _all_argv(plan)
    assert "cu126" in argv


def test_maxwell_experimental() -> None:
    plan = generate_plan([_gpu("TITAN X", 5, 2)], ["uma"], verify=False)
    assert plan.gpu_arch == "maxwell"
    assert any("EXPERIMENTAL" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# Device semantics (fail closed)
# ---------------------------------------------------------------------------


def test_auto_no_gpu_cpu_ok() -> None:
    plan = generate_plan(None, ["uma"], device="auto", verify=False)
    assert plan.gpu_arch == "cpu"


def test_cuda_no_gpu_fails() -> None:
    with pytest.raises(InstallPlanError):
        generate_plan(None, ["uma"], device="cuda")


def test_cuda_kepler_fails() -> None:
    with pytest.raises(InstallPlanError):
        generate_plan([_gpu("K80", 3, 7)], ["uma"], device="cuda")


def test_auto_kepler_cpu_fallback_with_warning() -> None:
    plan = generate_plan([_gpu("K80", 3, 7)], ["uma"], device="auto", verify=False)
    assert plan.gpu_arch == "cpu"
    assert any("unsupported" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# Python version validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["3.9", "3.13", "2.7"])
def test_invalid_python_raises(bad: str) -> None:
    with pytest.raises(InstallPlanError):
        generate_plan(None, ["uma"], python_version=bad, device="cpu")


def test_invalid_device_fails_closed() -> None:
    with pytest.raises(InstallPlanError, match="Unknown device"):
        generate_plan(None, ["uma"], device="gpu")


def test_invalid_source_uses_public_plan_error() -> None:
    with pytest.raises(InstallPlanError, match="Unknown source profile"):
        generate_plan(None, ["uma"], source="nope", device="cpu")


# ---------------------------------------------------------------------------
# clean / verify
# ---------------------------------------------------------------------------


def test_clean_only_removes_target_venv() -> None:
    plan = generate_plan(None, ["mace"], device="cpu", clean=True, verify=False)
    clean_steps = [s for s in plan.steps if s.stage == "clean"]
    assert len(clean_steps) == 1
    # Only the known .venv-mace path is removed.
    assert clean_steps[0].argv == ["rm", "-rf", ".venv-mace"]


def test_skip_doctor_removes_verify_steps() -> None:
    plan = generate_plan(None, ["mace"], device="cpu", verify=False)
    assert not plan.has_verify_steps
    plan2 = generate_plan(None, ["mace"], device="cpu", verify=True)
    assert plan2.has_verify_steps


# ---------------------------------------------------------------------------
# Offline / custom
# ---------------------------------------------------------------------------


def test_offline_plan_has_no_urls() -> None:
    plan = generate_plan(
        [_gpu("V100", 7, 0)],
        ["uma", "mace", "dpa", "grace"],
        source="offline",
        verify=False,
    )
    for step in plan.steps:
        joined = " ".join(step.argv)
        assert "http://" not in joined
        assert "https://" not in joined
    # Every pip step must carry --offline.
    for step in _pip_cmds(plan):
        assert "--offline" in step.argv


def test_offline_cpu_plan_has_no_urls() -> None:
    plan = generate_plan(
        None,
        ["uma", "mace", "dpa", "grace"],
        source="offline",
        device="cpu",
        verify=False,
    )
    for step in plan.steps:
        joined = " ".join(step.argv)
        assert "http://" not in joined
        assert "https://" not in joined
    for step in _pip_cmds(plan):
        assert "--offline" in step.argv


def test_custom_source_no_overrides() -> None:
    plan = generate_plan(
        [_gpu("V100", 7, 0)], ["uma", "mace"], source="custom", verify=False
    )
    for step in plan.steps:
        joined = " ".join(step.argv)
        assert "--index-url" not in joined
        assert "--find-links" not in joined


def test_custom_cpu_source_no_overrides() -> None:
    plan = generate_plan(None, ["dpa"], source="custom", device="cpu", verify=False)
    argv = _all_argv(plan)
    assert "torch==2.10.0+cpu" in argv
    assert "--index-url" not in argv
    assert "--find-links" not in argv


# ---------------------------------------------------------------------------
# china source — all packages use tuna, torch uses aliyun
# ---------------------------------------------------------------------------


def test_china_source_all_packages_use_mirror() -> None:
    plan = generate_plan(
        [_gpu("V100", 7, 0)],
        ["uma", "mace", "dpa", "grace"],
        source="china",
        verify=False,
    )
    for step in plan.steps:
        joined = " ".join(step.argv)
        # A standalone torch install step has a bare "torch==X" token.
        is_torch = any(a.startswith("torch==") for a in step.argv)
        if is_torch:
            # torch wheel → Aliyun; transitive packages → selected PyPI mirror
            assert "aliyun" in joined
            assert "tuna.tsinghua" in joined
        elif step.stage == "pip":
            # non-torch pip steps (mace-torch, deepmd-kit, tf, fairchem)
            # must use the tuna PyPI mirror
            assert "tuna.tsinghua" in joined
            assert "pypi.org" not in joined


def test_china_cpu_source_uses_selected_mirrors() -> None:
    plan = generate_plan(
        None,
        ["dpa"],
        source="china-ustc",
        device="cpu",
        verify=False,
    )
    torch_step, backend_step = _pip_cmds(plan)
    assert "torch==2.10.0+cpu" in torch_step.argv
    assert "mirrors.aliyun.com/pytorch-wheels/cpu" in " ".join(torch_step.argv)
    assert "mirrors.ustc.edu.cn/pypi/simple" in " ".join(torch_step.argv)
    assert "mirrors.ustc.edu.cn/pypi/simple" in " ".join(backend_step.argv)


# ---------------------------------------------------------------------------
# InstallStep uses argv, not shell strings
# ---------------------------------------------------------------------------


def test_steps_are_argv_lists() -> None:
    plan = generate_plan(None, ["mace"], device="cpu", verify=False)
    for step in plan.steps:
        assert isinstance(step.argv, list)
        assert all(isinstance(a, str) for a in step.argv)
