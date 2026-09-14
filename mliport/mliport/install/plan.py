# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Installation plan generation.

Given detected GPUs, requested engines, and a source profile, this module
produces an :class:`InstallPlan` — an ordered list of :class:`InstallStep`
objects, each holding an ``argv`` list (never a shell string).  The executor
runs each step with ``shell=False``.

Every mliport engine — including UMA — is installed **explicitly**: create a
venv, install the pinned torch/TF wheel for the detected architecture, install
the pinned backend package, and finally install mliport editable.  The UMA
runtime is **never** installed via ``uv sync``/``uv sync --frozen``.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from mliport.install.compatibility import (
    BACKENDS,
    FRAMEWORK_PYTHON,
    PROJECT_REQUIRES_PYTHON,
    SUPPORTED_PYTHON_VERSIONS,
    BackendSpec,
    effective_cuda_channel,
    get_backend_arch_profile,
)
from mliport.install.hardware import GpuInfo, classify_gpu, _pick_oldest
from mliport.install.sources import (
    SourceProfile,
    build_package_source_args,
    build_torch_source_args,
    resolve_source,
)

# Python versions supported by mliport (requires-python >=3.10,<3.13).


# Known engine keys (for normalization).
_ENGINE_KEYS = set(BACKENDS)


class InstallPlanError(Exception):
    """Raised for invalid installation configuration (fail closed)."""


@dataclass
class InstallStep:
    """One step in an installation plan.

    Attributes:
        stage: ``"venv"``, ``"clean"``, ``"pip"``, or ``"verify"``.
        description: One-line description for logging.
        argv: Argument vector executed with ``shell=False``.
        env: Extra environment variables to set for this step.
    """

    stage: str
    description: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class InstallPlan:
    """Complete installation plan.

    Attributes:
        gpu_arch: Detected architecture name (or ``"cpu"``).
        gpu_cc: Compute capability string (or ``""``).
        source: Resolved source profile name.
        python_version: Python version for the isolated venvs.
        steps: Ordered list of :class:`InstallStep`.
        warnings: Human-readable warnings to show before execution.
    """

    gpu_arch: str
    gpu_cc: str
    source: str
    python_version: str
    steps: list[InstallStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_verify_steps(self) -> bool:
        return any(s.stage == "verify" for s in self.steps)


def normalize_engines(engines: Sequence[str]) -> list[str]:
    """Normalize and deduplicate an engine list.

    - strips whitespace, lowercases
    - maps ``fairchem`` → ``uma``
    - deduplicates preserving order
    - raises :class:`InstallPlanError` on unknown engine or empty list
    """
    seen: set[str] = set()
    result: list[str] = []
    for raw in engines:
        name = str(raw).strip().lower()
        if name == "fairchem":
            name = "uma"
        if name not in _ENGINE_KEYS:
            raise InstallPlanError(
                f"Unknown engine '{name}'. "
                f"Choose from: {', '.join(sorted(_ENGINE_KEYS))}"
            )
        if name not in seen:
            seen.add(name)
            result.append(name)
    if not result:
        raise InstallPlanError("At least one engine must be requested.")
    return result


def validate_python_version(version: str) -> str:
    """Validate a Python version against mliport's supported range."""
    v = str(version).strip()
    if v not in SUPPORTED_PYTHON_VERSIONS:
        raise InstallPlanError(
            f"Python version '{v}' is not supported. mliport requires Python "
            f"3.10–3.12; choose one of: {', '.join(SUPPORTED_PYTHON_VERSIONS)}"
        )
    return v


def _uv_pip(profile: SourceProfile, python: str, *args: str) -> list[str]:
    """Build a ``uv pip install`` argv with source/offline handling."""
    argv: list[str] = ["uv", "pip", "install"]
    if profile.offline:
        argv.append("--offline")
    argv += ["--python", python]
    argv += list(args)
    return argv


def _venv_step(backend: BackendSpec, python_version: str) -> InstallStep:
    return InstallStep(
        stage="venv",
        description=f"Create {backend.venv_name} for {backend.label}",
        argv=["uv", "venv", "--python", python_version, backend.venv_name],
    )


def _clean_step(backend: BackendSpec) -> InstallStep:
    return InstallStep(
        stage="clean",
        description=f"Remove existing {backend.venv_name} (--clean)",
        # Only the known venv path from the compatibility matrix is removed.
        argv=["rm", "-rf", backend.venv_name],
    )


def _verify_step(backend: BackendSpec, device: str) -> InstallStep:
    return InstallStep(
        stage="verify",
        description=f"Verify {backend.label}",
        argv=[
            f"{backend.venv_name}/bin/mliport",
            "doctor",
            "--engine",
            backend.engine,
            "--device",
            device,
        ],
    )


def _torch_steps(
    backend: BackendSpec,
    arch_name: str,
    bp,
    profile: SourceProfile,
    cuda_tag: str | None,
    *,
    cpu: bool,
) -> list[InstallStep]:
    """Torch install steps (shared by UMA/MACE/DPA)."""
    steps: list[InstallStep] = []
    torch_ver = bp.framework_version
    python = f"{backend.venv_name}/bin/python"

    if cpu:
        # CPU wheels also carry a PEP 440 local tag. Route them through the
        # selected source profile so china/custom/offline keep their promised
        # source semantics instead of silently contacting pytorch.org.
        torch_argv = _uv_pip(profile, python, f"torch=={torch_ver}+cpu")
        torch_argv += build_torch_source_args(profile, "cpu")
        steps.append(
            InstallStep(
                stage="pip",
                description=f"Install torch {torch_ver}+cpu for {backend.label}",
                argv=torch_argv,
            )
        )
    else:
        assert cuda_tag is not None
        source_args = build_torch_source_args(profile, cuda_tag)
        # Pin the PEP 440 local version as well as the public version. Without
        # ``+cu126``/``+cu128``, a flat mirror or custom source can select a
        # different CUDA build with the same public version (for example,
        # torch 2.10.0+cu128 on a Pascal GPU that requires cu126).
        torch_argv = _uv_pip(profile, python, f"torch=={torch_ver}+{cuda_tag}")
        torch_argv += source_args
        steps.append(
            InstallStep(
                stage="pip",
                description=(
                    f"Install torch {torch_ver}+{cuda_tag} for {backend.label}"
                ),
                argv=torch_argv,
            )
        )

    # Backend package(s) + editable mliport with all user features.
    # ``analysis-all`` covers scipy/matplotlib (analysis), kinisi
    # (transport) and gemdat (electrolyte); textual is a core dependency, so
    # this is the complete user-facing feature set for each environment.
    pkg_argv = _uv_pip(
        profile,
        python,
        "-e",
        "./mliport[analysis-all]",
        *backend.install_packages(),
    )
    pkg_argv += build_package_source_args(profile)
    steps.append(
        InstallStep(
            stage="pip",
            description=f"Install {backend.label} (pinned backend + editable mliport)",
            argv=pkg_argv,
        )
    )
    return steps


def _grace_steps(
    backend: BackendSpec,
    bp,
    profile: SourceProfile,
    *,
    cpu: bool,
) -> list[InstallStep]:
    """GRACE (TensorFlow) install steps."""
    python = f"{backend.venv_name}/bin/python"
    tf_ver = bp.framework_version
    extra_pkgs = bp.extra_packages

    if cpu:
        pkg_argv = _uv_pip(
            profile,
            python,
            "-e",
            "./mliport[analysis-all]",
            f"tensorflow=={tf_ver}",
            backend.requirement,
        )
    else:
        pkg_argv = _uv_pip(
            profile,
            python,
            "-e",
            "./mliport[analysis-all]",
            f"tensorflow[and-cuda]=={tf_ver}",
            backend.requirement,
            *extra_pkgs,
        )
    pkg_argv += build_package_source_args(profile)
    return [
        InstallStep(
            stage="pip",
            description=f"Install {backend.label} (TF {tf_ver})",
            argv=pkg_argv,
        )
    ]


def generate_plan(
    gpus: Sequence[GpuInfo] | None,
    engines: Sequence[str] = ("uma", "mace", "dpa", "grace"),
    *,
    source: str = "auto",
    python_version: str = "3.12",
    device: str = "auto",
    clean: bool = False,
    verify: bool = True,
) -> InstallPlan:
    """Generate an installation plan for the given hardware and engines.

    Args:
        gpus: Detected GPUs (``None`` or empty = CPU-only).
        engines: Backend engines to install (``"uma"``, ``"mace"``,
            ``"dpa"``, ``"grace"``).
        source: Source profile name (``"auto"``, ``"official"``, one of the
            ``"china"`` / explicit ``"china-*"`` profiles, ``"offline"``,
            or ``"custom"``).
        python_version: Python version for the isolated venvs (3.10–3.12).
        device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
        clean: If ``True``, remove each target venv before recreating it.
        verify: If ``True``, append a ``doctor`` verify step per engine.

    Returns:
        An :class:`InstallPlan` ready for execution.

    Raises:
        InstallPlanError: On invalid configuration (unknown engine, empty
            engine list, invalid Python version, ``device=cuda`` without a
            supported GPU, unknown source).
    """
    py_ver = validate_python_version(python_version)
    if device not in {"auto", "cuda", "cpu"}:
        raise InstallPlanError(
            f"Unknown device '{device}'. Choose from: auto, cuda, cpu"
        )
    try:
        src = resolve_source(source)
    except ValueError as exc:
        raise InstallPlanError(str(exc)) from exc
    engine_list = normalize_engines(engines)
    for engine in engine_list:
        backend = BACKENDS[engine]
        constraints = [
            ("mliport", PROJECT_REQUIRES_PYTHON),
            (backend.requirement, backend.requires_python),
        ]
        for bp in backend.arch_profiles.values():
            key = (backend.framework, bp.framework_version)
            if key not in FRAMEWORK_PYTHON:
                raise InstallPlanError(f"Unknown framework Python/wheel support: {key}")
            constraints.append(("==".join(key) + " wheels", FRAMEWORK_PYTHON[key]))
        for origin, constraint in constraints:
            if py_ver not in constraint:
                raise InstallPlanError(
                    f"Python {py_ver} is incompatible with {origin}: "
                    f"requires-python {constraint}. No environment changes were made."
                )

    # ---- Resolve device / architecture (fail closed for cuda) ----
    is_cpu: bool
    arch_name: str
    cc_str = ""
    arch = None

    if device == "cuda":
        if not gpus:
            raise InstallPlanError(
                "device=cuda was requested but no NVIDIA GPU was detected. "
                "Use device=auto (CPU fallback) or install on a CUDA machine."
            )
        oldest = _pick_oldest(gpus)
        cc_str = f"{oldest.cc_major}.{oldest.cc_minor}"
        arch = classify_gpu(oldest.cc_major, oldest.cc_minor)
        if arch is None:
            raise InstallPlanError(
                f"device=cuda was requested but GPU compute capability "
                f"{cc_str} is unsupported (Kepler or unknown). Use device=auto "
                f"for a CPU fallback, or upgrade the GPU."
            )
        is_cpu = False
        arch_name = arch.name
    elif device == "cpu":
        is_cpu = True
        arch_name = "cpu"
    else:  # auto
        if not gpus:
            is_cpu = True
            arch_name = "cpu"
        else:
            oldest = _pick_oldest(gpus)
            cc_str = f"{oldest.cc_major}.{oldest.cc_minor}"
            arch = classify_gpu(oldest.cc_major, oldest.cc_minor)
            if arch is None:
                # Unsupported GPU (e.g. Kepler): CPU fallback with warning.
                is_cpu = True
                arch_name = "cpu"
            else:
                is_cpu = False
                arch_name = arch.name

    plan = InstallPlan(
        gpu_arch=arch_name,
        gpu_cc=cc_str,
        source=src.name,
        python_version=py_ver,
    )

    if device == "auto" and not is_cpu and not gpus:
        # Shouldn't happen (auto with gpus sets is_cpu False), kept for safety.
        pass
    if arch is None and device == "auto" and gpus:
        plan.warnings.append(
            f"GPU compute capability {cc_str} is unsupported; falling back "
            f"to CPU-only installation. Use device=cuda to fail instead."
        )
    if src.offline:
        plan.warnings.append(
            "Source profile 'offline': no network access will be used. "
            "Only locally cached wheels are consulted."
        )

    # ---- Build steps per engine ----
    for engine in engine_list:
        backend = BACKENDS[engine]
        if clean:
            plan.steps.append(_clean_step(backend))
        plan.steps.append(_venv_step(backend, py_ver))

        if engine == "grace":
            if is_cpu:
                plan.warnings.append(
                    "TensorPotential 0.6.0 on Linux requires tensorflow[and-cuda] "
                    "even for CPU execution. This is not a CUDA-free minimal install."
                )
            bp = get_backend_arch_profile("grace", arch_name) if not is_cpu else None
            if is_cpu:
                # Use any arch profile's framework version (all use TF 2.20).
                bp = next(iter(BACKENDS["grace"].arch_profiles.values()))
            else:
                assert bp is not None, f"No profile for {backend.label} on {arch_name}"
                if bp.status == "experimental":
                    plan.warnings.append(
                        f"{backend.label} on {arch_name}: EXPERIMENTAL — "
                        f"upstream does not officially support this GPU. {bp.notes}"
                    )
                elif bp.status == "needs_smoke_test":
                    plan.warnings.append(
                        f"{backend.label} on {arch_name}: needs smoke test — "
                        f"upstream constraints are satisfied but mliport has not "
                        f"yet verified this combination on real hardware."
                    )
            plan.steps.extend(_grace_steps(backend, bp, src, cpu=is_cpu))
        else:
            # torch backend (UMA / MACE / DPA)
            bp = get_backend_arch_profile(engine, arch_name) if not is_cpu else None
            cuda_tag: str | None = None
            if not is_cpu:
                assert arch is not None
                assert bp is not None, f"No profile for {backend.label} on {arch_name}"
                cuda_tag = effective_cuda_channel(backend, arch, bp)
                if bp.status == "experimental":
                    plan.warnings.append(
                        f"{backend.label} on {arch_name}: EXPERIMENTAL — "
                        f"upstream does not officially support this GPU. {bp.notes}"
                    )
                elif bp.status == "needs_smoke_test":
                    plan.warnings.append(
                        f"{backend.label} on {arch_name}: needs smoke test — "
                        f"upstream constraints are satisfied but mliport has not "
                        f"yet verified this combination on real hardware."
                    )
            else:
                # CPU: use the framework version from any arch profile
                bp = next(iter(BACKENDS[engine].arch_profiles.values()))
            plan.steps.extend(
                _torch_steps(backend, arch_name, bp, src, cuda_tag, cpu=is_cpu)
            )

        # One immutable constraint set across every resolver stage. In
        # particular, installing the backend must not replace the CUDA build
        # selected by the earlier framework stage.
        framework_pin = f"{backend.framework}=={bp.framework_version}"
        if backend.framework == "torch":
            framework_pin += "+cpu" if is_cpu else f"+{cuda_tag}"
        pins = [framework_pin, backend.requirement, *bp.extra_packages]
        pins.extend(pkg for pkg in backend.install_packages()[1:] if "==" in pkg)
        constraint_path = f"{backend.venv_name}/mliport-constraints.txt"
        constraint_text = "\n".join(dict.fromkeys(pins)) + "\n"
        first_pip = next(
            i
            for i, step in enumerate(plan.steps)
            if step.stage == "pip"
            and step.argv[step.argv.index("--python") + 1]
            == f"{backend.venv_name}/bin/python"
        )
        plan.steps.insert(
            first_pip,
            InstallStep(
                stage="constraints",
                description=f"Freeze {backend.label} dependency contract",
                argv=[
                    f"{backend.venv_name}/bin/python",
                    "-c",
                    "from pathlib import Path; import sys; "
                    "Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')",
                    constraint_path,
                    constraint_text,
                ],
            ),
        )
        for step in plan.steps[first_pip + 1 :]:
            if step.stage == "pip":
                step.argv.extend(["--constraint", constraint_path])
        python = f"{backend.venv_name}/bin/python"
        plan.steps.append(
            InstallStep(
                stage="check",
                description=f"Check {backend.label} dependency consistency",
                argv=["uv", "pip", "check", "--python", python],
            )
        )
        plan.steps.append(
            InstallStep(
                stage="inventory",
                description=f"Record {backend.label} final inventory",
                argv=[
                    python,
                    "-m",
                    "mliport.install.inventory",
                    "--write",
                    "--constraints",
                    constraint_path,
                ],
            )
        )
        if verify:
            plan.steps.append(_verify_step(backend, "cpu" if is_cpu else device))

    return plan


def plan_to_json(plan: InstallPlan) -> str:
    """Serialize an :class:`InstallPlan` to JSON."""
    return json.dumps(
        {
            "gpu_arch": plan.gpu_arch,
            "gpu_cc": plan.gpu_cc,
            "source": plan.source,
            "python_version": plan.python_version,
            "warnings": plan.warnings,
            "steps": [
                {
                    "stage": s.stage,
                    "description": s.description,
                    "argv": s.argv,
                    "env": s.env,
                }
                for s in plan.steps
            ],
        },
        indent=2,
        ensure_ascii=False,
    )


def render_plan_shell(plan: InstallPlan) -> str:
    """Render a plan as human-readable shell lines (for ``--dry-run``)."""
    lines = [
        f"GPU arch : {plan.gpu_arch}",
        f"Source   : {plan.source}",
        f"Python   : {plan.python_version}",
        "",
    ]
    for s in plan.steps:
        lines.append(f"  [{s.stage}] {s.description}")
        lines.append(f"    $ {shlex.join(s.argv)}")
    return "\n".join(lines)
