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
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from mliport.install.compatibility import (
    ALL_BACKENDS,
    BACKENDS,
    CUDA_CHANNELS,
    FRAMEWORK_PYTHON,
    PROJECT_REQUIRES_PYTHON,
    SUPPORTED_PYTHON_VERSIONS,
    BackendSpec,
    effective_cuda_channel,
    get_backend_arch_profile,
    lookup_backend,
)
from mliport.install.hardware import GpuInfo, _pick_oldest, classify_gpu
from mliport.install.sources import (
    SourceProfile,
    build_package_source_args,
    build_torch_source_args,
    resolve_source,
)

# Python versions supported by mliport (requires-python >=3.10,<3.14).


# Known engine keys (for normalization).
_ENGINE_KEYS = set(ALL_BACKENDS)


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
        supported = "–".join(
            (SUPPORTED_PYTHON_VERSIONS[0], SUPPORTED_PYTHON_VERSIONS[-1])
        )
        raise InstallPlanError(
            f"Python version '{v}' is not supported. mliport requires Python "
            f"{supported}; choose one of: {', '.join(SUPPORTED_PYTHON_VERSIONS)}"
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
    """The venv step.

    ``--allow-existing`` only keeps ``uv`` from refusing a path that is already
    there; it is **not** a compatibility decision, and on its own it would let
    uv initialise a venv inside a foreign directory or silently replace a venv
    with another Python version.  The decision is made beforehand by
    :func:`mliport.install.venv_contract.inspect_target_venv` (called from
    :mod:`mliport.install.cli`): a compatible environment is reused, an
    incompatible one is refused with ``--clean`` and nothing is executed.
    """
    return InstallStep(
        stage="venv",
        description=f"Create {backend.venv_name} for {backend.label}",
        argv=[
            "uv",
            "venv",
            "--python",
            python_version,
            "--allow-existing",
            backend.venv_name,
        ],
    )


def _venv_python_path(backend: BackendSpec, *, windows: bool | None = None) -> str:
    """Python executable path inside an installer-managed virtual environment."""
    is_windows = os.name == "nt" if windows is None else windows
    if is_windows:
        return str(PureWindowsPath(backend.venv_name) / "Scripts" / "python.exe")
    return f"{backend.venv_name}/bin/python"


def _venv_cli_path(backend: BackendSpec, *, windows: bool | None = None) -> str:
    """Console-script path inside an installer-managed virtual environment."""
    is_windows = os.name == "nt" if windows is None else windows
    if is_windows:
        return str(PureWindowsPath(backend.venv_name) / "Scripts" / "mliport.exe")
    return f"{backend.venv_name}/bin/mliport"


def _clean_step(backend: BackendSpec) -> InstallStep:
    """Remove only the matrix-owned venv, using the current Python on every OS."""
    remove_path = (
        "from pathlib import Path; import shutil,sys; "
        "p=Path(sys.argv[1]); "
        "shutil.rmtree(p) if p.is_dir() and not p.is_symlink() "
        "else p.unlink(missing_ok=True)"
    )
    return InstallStep(
        stage="clean",
        description=f"Remove existing {backend.venv_name} (--clean)",
        argv=[sys.executable, "-c", remove_path, backend.venv_name],
    )


def _verify_step(backend: BackendSpec, device: str) -> InstallStep:
    return InstallStep(
        stage="verify",
        description=f"Verify {backend.label}",
        argv=[
            _venv_cli_path(backend),
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
    python = _venv_python_path(backend)

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
    from_source = bool(backend.source_url) and not profile.offline
    if from_source and backend.build_packages:
        steps.append(
            InstallStep(
                stage="pip",
                description=f"Install {backend.label} source-build requirements",
                argv=(
                    _uv_pip(profile, python, *backend.build_packages)
                    + build_package_source_args(profile)
                ),
            )
        )
    pkg_argv = _uv_pip(
        profile,
        python,
        "-e",
        "./mliport[analysis-all]",
        *(
            backend.install_packages()
            if not profile.offline or not backend.source_url
            else (f"{backend.distribution}=={backend.version}", *backend.install_extra)
        ),
    )
    if from_source and backend.build_packages:
        # Build against the exact architecture-specific torch installed above;
        # isolation could download another CUDA/ABI variant or TensorFlow.
        pkg_argv.append("--no-build-isolation")
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if not cpu and cuda_root:
            # Torch's CMake package still uses FindCUDA's legacy variable.
            # Forward the user's explicit toolkit location to both finders.
            for key in ("CUDA_TOOLKIT_ROOT_DIR", "CUDAToolkit_ROOT"):
                pkg_argv.extend(["--config-setting", f"cmake.define.{key}={cuda_root}"])
    pkg_argv += build_package_source_args(profile)
    if profile.offline and backend.source_url:
        # Source snapshots have no PyPI release. Offline installs consume an
        # explicitly prepared wheelhouse, without retaining a remote Git URL.
        pkg_argv.extend(["--find-links", "packages"])
    steps.append(
        InstallStep(
            stage="pip",
            description=f"Install {backend.label} (pinned backend + editable mliport)",
            argv=pkg_argv,
            env=dict(backend.build_env) if from_source else {},
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
    python = _venv_python_path(backend)
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


def _nep_steps(
    backend: BackendSpec,
    profile: SourceProfile,
    *,
    cpu: bool,
) -> list[InstallStep]:
    """NEP (Calorine) install steps.

    Calorine is a compiled C++ extension (numpy + ASE), not a deep-learning
    framework, so there is no framework wheel to pin.  Like every other
    backend, the isolated environment gets the pinned backend package **and**
    an editable mliport in a single resolver run: the ``verify`` stage calls
    ``<venv>/bin/mliport doctor``, which cannot exist unless mliport itself is
    installed into that environment.  One resolve (rather than separate
    ``calorine`` / ``mliport`` installs) keeps ASE and NumPy consistent between
    the two; NumPy is deliberately left to the resolver because neither mliport
    nor Calorine pins it.  The source profile is applied exactly as for the
    other backends (mirror index, ``--offline``), so ``--source china`` /
    ``offline`` / ``custom`` keep their promised semantics.

    The native ``gpumd`` binary is a separate build that mliport deliberately
    does NOT perform automatically: it needs a matching CUDA toolkit and an
    architecture flag, and a silently mis-architected binary would fail at run
    time rather than install time.  The plan therefore installs the ASE runtime
    and records the prerequisite for the native runtime in a warning, which
    ``doctor --engine nep`` re-checks afterwards.
    """
    python = _venv_python_path(backend)
    argv = _uv_pip(
        profile,
        python,
        "-e",
        "./mliport[analysis-all]",
        *backend.install_packages(),
    )
    argv += build_package_source_args(profile)
    return [
        InstallStep(
            stage="pip",
            description=(
                f"Install {backend.label} ASE runtime "
                "(pinned calorine + editable mliport)"
            ),
            argv=argv,
        )
    ]


def generate_plan(
    gpus: Sequence[GpuInfo] | None,
    engines: Sequence[str] = ("uma", "mace", "dpa", "grace", "nep"),
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
            ``"dpa"``, ``"grace"``, ``"nep"``).
        source: Source profile name (``"auto"``, ``"official"``, one of the
            ``"china"`` / explicit ``"china-*"`` profiles, ``"offline"``,
            or ``"custom"``).
        python_version: Python version for the isolated venvs (3.10–3.13).
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
        backend = lookup_backend(engine)
        assert backend is not None
        constraints = [
            ("mliport", PROJECT_REQUIRES_PYTHON),
            (backend.requirement, backend.requires_python),
        ]
        for bp in backend.arch_profiles.values():
            if backend.framework not in {"torch", "tensorflow"}:
                # NEP: the pin is the CUDA toolkit used to build gpumd, not a
                # Python framework wheel, so there is no FRAMEWORK_PYTHON entry.
                continue
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

    route_fallback_warning: str | None = None
    route_profile_warnings: list[str] = []
    if not is_cpu and gpus:
        incompatible_routes: list[str] = []
        profile_warnings: list[str] = []
        unknown_gpus = [
            gpu for gpu in gpus if classify_gpu(gpu.cc_major, gpu.cc_minor) is None
        ]
        if unknown_gpus:
            devices = ", ".join(
                f"{gpu.name} (CC {gpu.cc_major}.{gpu.cc_minor})"
                for gpu in unknown_gpus
            )
            incompatible_routes.append(
                f"unrecognized GPU compute capability for {devices}"
            )

        for engine in engine_list:
            backend = lookup_backend(engine)
            assert backend is not None
            for gpu in gpus:
                gpu_arch = classify_gpu(gpu.cc_major, gpu.cc_minor)
                if gpu_arch is None:
                    continue
                gpu_profile = get_backend_arch_profile(engine, gpu_arch.name)
                if gpu_profile is None:
                    incompatible_routes.append(
                        f"{backend.label} has no compatibility profile for "
                        f"{gpu.name} ({gpu_arch.name})"
                    )
                elif not gpu_profile.gpu_runtime_available:
                    incompatible_routes.append(
                        f"{backend.label} has no usable GPU runtime wheel for "
                        f"{gpu_arch.label} ({gpu.name})"
                    )
                elif gpu_profile.status == "experimental":
                    profile_warnings.append(
                        f"{backend.label} on {gpu_arch.name}: EXPERIMENTAL — "
                        f"upstream does not officially support this GPU. "
                        f"{gpu_profile.notes}"
                    )
                elif gpu_profile.status == "needs_smoke_test":
                    profile_warnings.append(
                        f"{backend.label} on {gpu_arch.name}: needs smoke test — "
                        "upstream constraints are satisfied but mliport has not "
                        "yet verified this combination on real hardware."
                    )

            if backend.framework != "torch":
                continue
            profile = get_backend_arch_profile(engine, arch_name)
            if profile is None:
                incompatible_routes.append(
                    f"{backend.label} has no CUDA profile for {arch_name}"
                )
                continue
            channel_name = effective_cuda_channel(backend, arch, profile)
            channel = CUDA_CHANNELS.get(channel_name)
            if channel is None:
                incompatible_routes.append(
                    f"{backend.label} selects unregistered CUDA channel {channel_name}"
                )
                continue
            unsupported = [
                gpu
                for gpu in gpus
                if not channel.sm_min
                <= (gpu.cc_major, gpu.cc_minor)
                <= channel.sm_max
            ]
            if unsupported:
                devices = ", ".join(
                    f"{gpu.name} (sm_{gpu.cc_major}{gpu.cc_minor})"
                    for gpu in unsupported
                )
                incompatible_routes.append(
                    f"{backend.label} wheel {channel_name} does not cover {devices}"
                )
        if incompatible_routes:
            detail = "; ".join(dict.fromkeys(incompatible_routes))
            if device == "cuda":
                raise InstallPlanError(
                    "No compatible GPU runtime route exists for every visible GPU "
                    f"and selected backend: {detail}. Use --device cpu or select "
                    "a compatible GPU visibility set with CUDA_VISIBLE_DEVICES."
                )
            is_cpu = True
            arch_name = "cpu"
            arch = None
            route_fallback_warning = (
                "No compatible GPU runtime route exists for every visible GPU and "
                f"selected backend; --device auto is falling back to CPU. {detail}"
            )
        else:
            route_profile_warnings = list(dict.fromkeys(profile_warnings))

    plan = InstallPlan(
        gpu_arch=arch_name,
        gpu_cc=cc_str,
        source=src.name,
        python_version=py_ver,
    )
    if arch is None and device == "auto" and gpus and not route_fallback_warning:
        plan.warnings.append(
            f"GPU compute capability {cc_str} is unsupported; falling back "
            f"to CPU-only installation. Use device=cuda to fail instead."
        )
    if route_fallback_warning:
        plan.warnings.append(route_fallback_warning)
    plan.warnings.extend(route_profile_warnings)
    if src.offline:
        plan.warnings.append(
            "Source profile 'offline': no network access will be used. "
            "Only locally cached wheels are consulted."
        )

    # ---- Build steps per engine ----
    for engine in engine_list:
        backend = lookup_backend(engine)
        assert backend is not None
        if clean:
            plan.steps.append(_clean_step(backend))
        plan.steps.append(_venv_step(backend, py_ver))

        if engine == "nep":
            plan.warnings.append(
                "NEP: the native runtime needs a separately built `gpumd` "
                "binary (CUDA toolkit + matching -arch=sm_XX). mliport does not "
                "build it automatically. Without it only the CPU ASE runtime "
                "(`NEP_RUNTIME=ase` on cpu) works. After building, set "
                "NEP_GPUMD_COMMAND for mliport runs and make `gpumd` visible to "
                "`mliport doctor --engine nep` (on PATH or MLIPORT_GPUMD_COMMAND)."
            )
            plan.steps.extend(_nep_steps(backend, src, cpu=is_cpu))
        elif engine == "grace":
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
                    warning = (
                        f"{backend.label} on {arch_name}: EXPERIMENTAL — "
                        f"upstream does not officially support this GPU. {bp.notes}"
                    )
                    if warning not in plan.warnings:
                        plan.warnings.append(warning)
                elif bp.status == "needs_smoke_test":
                    warning = (
                        f"{backend.label} on {arch_name}: needs smoke test — "
                        "upstream constraints are satisfied but mliport has not "
                        "yet verified this combination on real hardware."
                    )
                    if warning not in plan.warnings:
                        plan.warnings.append(warning)
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
                    warning = (
                        f"{backend.label} on {arch_name}: EXPERIMENTAL — "
                        f"upstream does not officially support this GPU. {bp.notes}"
                    )
                    if warning not in plan.warnings:
                        plan.warnings.append(warning)
                elif bp.status == "needs_smoke_test":
                    warning = (
                        f"{backend.label} on {arch_name}: needs smoke test — "
                        "upstream constraints are satisfied but mliport has not "
                        "yet verified this combination on real hardware."
                    )
                    if warning not in plan.warnings:
                        plan.warnings.append(warning)
            else:
                # CPU: use the framework version from any arch profile
                bp = next(iter(backend.arch_profiles.values()))
            plan.steps.extend(
                _torch_steps(backend, arch_name, bp, src, cuda_tag, cpu=is_cpu)
            )

        # One immutable constraint set across every resolver stage. In
        # particular, installing the backend must not replace the CUDA build
        # selected by the earlier framework stage.
        pins = []
        if backend.framework in {"torch", "tensorflow"}:
            framework_pin = f"{backend.framework}=={bp.framework_version}"
            if backend.framework == "torch":
                framework_pin += "+cpu" if is_cpu else f"+{cuda_tag}"
            pins.append(framework_pin)
        pins.extend([f"{backend.distribution}=={backend.version}", *bp.extra_packages])
        pins.extend(pkg for pkg in backend.install_packages()[1:] if "==" in pkg)
        constraint_path = f"{backend.venv_name}/mliport-constraints.txt"
        constraint_text = "\n".join(dict.fromkeys(pins)) + "\n"
        first_pip = next(
            i
            for i, step in enumerate(plan.steps)
            if step.stage == "pip"
            and step.argv[step.argv.index("--python") + 1]
            == _venv_python_path(backend)
        )
        plan.steps.insert(
            first_pip,
            InstallStep(
                stage="constraints",
                description=f"Freeze {backend.label} dependency contract",
                argv=[
                    _venv_python_path(backend),
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
        python = _venv_python_path(backend)
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
