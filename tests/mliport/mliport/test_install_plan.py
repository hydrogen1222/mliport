"""Tests for mliport.install.plan — installation plan generation."""

from __future__ import annotations

import pytest

from mliport.install.hardware import GpuInfo
from mliport.install.plan import (
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
    assert "-e ./mliport" in argv


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
        if step.stage == "pip" and any(arg.startswith("torch==") for arg in step.argv)
    )
    assert "torch==2.11.0+cu126" in torch_step.argv
    assert "--find-links" in torch_step.argv
    assert "cu126" in " ".join(torch_step.argv)
    assert "torch==2.11.0" not in torch_step.argv


def test_dpa_source_build_is_pinned_and_uses_selected_torch():
    from mliport.install.compatibility import BACKENDS

    backend = BACKENDS["dpa"]
    plan = generate_plan([_gpu("V100", 7, 0)], ["dpa"], verify=False)
    build = next(s for s in plan.steps if backend.requirement in s.argv)
    assert "--no-build-isolation" in build.argv
    assert build.env["DP_ENABLE_TENSORFLOW"] == "0"
    assert build.env["DP_ENABLE_PYTORCH"] == "1"
    assert build.env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DEEPMD_KIT"] == backend.version
    build_index = plan.steps.index(build)
    assert any("scikit-build-core>=1,<2" in s.argv for s in plan.steps[:build_index])
    constraints = next(s for s in plan.steps if s.stage == "constraints")
    assert f"deepmd-kit=={backend.version}" in " ".join(constraints.argv)
    venv = next(s for s in plan.steps if s.stage == "venv")
    assert "--allow-existing" in venv.argv
    assert "--clear" not in venv.argv


def test_4090_plan_modern() -> None:
    """Ada (sm_89) → torch engines use cu128 (modern)."""
    plan = generate_plan([_gpu("RTX 4090", 8, 9)], ["uma", "mace", "dpa"], verify=False)
    argv = _all_argv(plan)
    assert "cu128" in argv
    assert "cu126" not in argv
    assert "torch==2.11.0+cu128" in argv
    assert any("DPA" in w and "needs smoke test" in w for w in plan.warnings)


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


@pytest.mark.parametrize("bad", ["3.9", "3.14", "2.7"])
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
    from mliport.install.compatibility import BACKENDS

    backend = BACKENDS["dpa"]
    dpa = next(s for s in plan.steps if f"deepmd-kit=={backend.version}" in s.argv)
    assert dpa.argv[dpa.argv.index("--find-links") + 1] == "packages"
    assert "--no-build-isolation" not in dpa.argv


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
    assert "torch==2.11.0+cpu" in argv
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
        is_torch = step.stage == "pip" and any(
            a.startswith("torch==") for a in step.argv
        )
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
    torch_step, backend_step = _pip_cmds(plan)[0], _pip_cmds(plan)[-1]
    assert "torch==2.11.0+cpu" in torch_step.argv
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


# ---------------------------------------------------------------------------
# NEP: mliport itself must be installed into .venv-nep (tracking #61)
# ---------------------------------------------------------------------------

_NEP_PIP_ARGV_START = ("uv", "pip", "install")


def _nep_install(plan):
    steps = [s for s in _pip_cmds(plan) if "-e" in s.argv]
    assert len(steps) == 1, [s.argv for s in _pip_cmds(plan)]
    return steps[0]


def test_nep_plan_installs_editable_mliport_with_pinned_calorine() -> None:
    plan = generate_plan(None, ["nep"], device="cpu")
    step = _nep_install(plan)
    argv = step.argv
    assert tuple(argv[:3]) == _NEP_PIP_ARGV_START
    assert argv[argv.index("--python") + 1] == ".venv-nep/bin/python"
    assert argv[argv.index("-e") + 1] == "./mliport[analysis-all]"
    assert "calorine==4.1" in argv
    # NumPy/ASE come from the resolver (mliport requires ase>=3.26), never from
    # hand-written pins that could contradict it.
    assert not any(a.startswith(("numpy", "ase")) for a in argv)
    assert "--constraint" in argv
    assert "calorine==4.1" in plan_constraints_text(plan)


def plan_constraints_text(plan) -> str:
    step = next(s for s in plan.steps if s.stage == "constraints")
    return step.argv[-1]


def test_nep_plan_stage_order_ends_with_launchable_verify() -> None:
    plan = generate_plan(None, ["nep"], device="cpu")
    stages = [s.stage for s in plan.steps]
    assert stages == ["venv", "constraints", "pip", "check", "inventory", "verify"]
    # mliport is installed before anything that executes .venv-nep/bin/mliport
    verify = plan.steps[-1]
    assert verify.argv[0] == ".venv-nep/bin/mliport"
    assert verify.argv[1:4] == ["doctor", "--engine", "nep"]
    first_mliport_install = stages.index("pip")
    assert first_mliport_install < stages.index("verify")


def test_nep_plan_does_not_rely_on_ignored_pip_environment() -> None:
    """The managed source profiles strip PIP_*/UV_* env, so mirror selection must
    be argv-based; the old PIP_EXTRA_INDEX_URL env on the NEP step was dead."""
    for source in ("official", "china", "offline", "custom"):
        plan = generate_plan(None, ["nep"], device="cpu", source=source)
        for step in plan.steps:
            assert not any(k.startswith(("PIP_", "UV_")) for k in step.env), source


def test_nep_plan_honours_china_mirror() -> None:
    plan = generate_plan(None, ["nep"], device="cpu", source="china")
    argv = _nep_install(plan).argv
    assert "--index-url" in argv
    assert argv[argv.index("--index-url") + 1].startswith("https://mirrors.tuna.")
    assert "--offline" not in argv


def test_nep_plan_honours_offline_and_custom() -> None:
    offline = generate_plan(None, ["nep"], device="cpu", source="offline")
    assert "--offline" in _nep_install(offline).argv
    for step in offline.steps:
        joined = " ".join(step.argv)
        assert "http://" not in joined and "https://" not in joined

    custom = generate_plan(None, ["nep"], device="cpu", source="custom")
    joined = " ".join(_nep_install(custom).argv)
    assert "--index-url" not in joined and "--extra-index-url" not in joined


def test_nep_official_plan_has_no_mirror_flags() -> None:
    argv = _nep_install(generate_plan(None, ["nep"], device="cpu")).argv
    assert "--index-url" not in argv and "--offline" not in argv


def test_nep_clean_removes_only_its_own_environment() -> None:
    plan = generate_plan(None, ["nep"], device="cpu", clean=True)
    clean = [s for s in plan.steps if s.stage == "clean"]
    assert [s.argv for s in clean] == [["rm", "-rf", ".venv-nep"]]


def test_nep_is_opt_in_and_isolated_from_other_backends() -> None:
    from mliport.install.cli import _parser

    assert "nep" not in _parser().parse_args([]).engines
    assert _parser().parse_args(["--engines", "nep"]).engines == ["nep"]

    both = generate_plan(None, ["mace", "nep"], device="cpu", clean=True)
    alone_mace = generate_plan(None, ["mace"], device="cpu", clean=True)
    alone_nep = generate_plan(None, ["nep"], device="cpu", clean=True)
    # Adding NEP appends its own steps; the MACE steps are byte-identical.
    mace_in_both = [s.argv for s in both.steps if ".venv-mace" in " ".join(s.argv)]
    assert mace_in_both == [s.argv for s in alone_mace.steps]
    nep_in_both = [s.argv for s in both.steps if ".venv-nep" in " ".join(s.argv)]
    assert nep_in_both == [s.argv for s in alone_nep.steps]
    # no step touches another backend's environment
    for step in both.steps:
        joined = " ".join(step.argv)
        assert not (".venv-mace" in joined and ".venv-nep" in joined)


@pytest.mark.parametrize("engine", ["mace", "dpa", "grace", "uma"])
def test_other_backends_install_mliport_exactly_once_and_never_calorine(engine) -> None:
    plan = generate_plan(None, [engine], device="cpu")
    editable = [s for s in _pip_cmds(plan) if "-e" in s.argv]
    assert len(editable) == 1
    assert "calorine" not in _all_argv(plan)


def test_nep_plan_still_warns_that_gpumd_is_not_built() -> None:
    plan = generate_plan(None, ["nep"], device="cpu")
    text = " ".join(plan.warnings)
    assert "gpumd" in text
    assert "does not build it" in text
    assert "MLIPORT_GPUMD_COMMAND" in text
    assert "NEP_GPUMD_COMMAND" in text
