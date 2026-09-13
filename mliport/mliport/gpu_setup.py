# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Hardware GPU detection and PyTorch build recommendation (backward-compat).

This module is the **legacy public API** for ``mliport setup`` and
``mliport doctor``.  It delegates entirely to :mod:`mliport.install`; there is
**no second copy of installation logic** here.

New code should use :mod:`mliport.install` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mliport.install.compatibility import BACKENDS, get_backend_arch_profile
from mliport.install.hardware import (
    MIN_VRAM_MIB_WARN,
    GpuInfo,
    cc_arch_name,
    classify_gpu,
    detect_gpus,
    _pick_oldest,
)
from mliport.install.plan import InstallPlanError, generate_plan, render_plan_shell

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Re-exports (backward-compatible)
# ---------------------------------------------------------------------------
__all__ = [
    "GpuInfo",
    "MIN_VRAM_MIB_WARN",
    "TorchRecommendation",
    "cc_arch_name",
    "detect_gpus",
    "engine_install_commands",
    "format_setup_report",
    "recommend_torch",
    "setup_report_json",
]


@dataclass
class TorchRecommendation:
    """Legacy PyTorch build recommendation for a given compute capability."""

    version: str
    index_url: str
    cu_tag: str
    supported: bool
    rationale: str
    install_commands: list[str] = field(default_factory=list)
    pyproject_snippet: str = ""


def recommend_torch(
    cc_major: int, cc_minor: int, *, engine: str = "uma"
) -> TorchRecommendation:
    """Legacy: recommend a PyTorch build for the given compute capability.

    Kept for backward compatibility with ``mliport doctor``. ``engine`` defaults
    to UMA for callers of the historical two-argument API, while doctor passes
    the explicitly selected engine so DPA/MACE advice preserves their own
    framework ABI pins.
    """
    arch = classify_gpu(cc_major, cc_minor)
    sm = f"sm_{cc_major}{cc_minor}"
    arch_name = cc_arch_name(cc_major, cc_minor)

    if arch is None:
        return TorchRecommendation(
            version="",
            index_url="",
            cu_tag="",
            supported=False,
            rationale=(
                f"{arch_name} ({sm}) has NO prebuilt PyTorch wheel. "
                f"Use --device cpu or upgrade to a Maxwell (GTX 900) GPU."
            ),
        )

    backend = BACKENDS.get(engine)
    if backend is None or backend.framework != "torch":
        return TorchRecommendation(
            version="",
            index_url="",
            cu_tag="",
            supported=False,
            rationale=f"No PyTorch compatibility profile for engine {engine!r}.",
        )

    bp = get_backend_arch_profile(engine, arch.name)
    if bp is None:
        return TorchRecommendation(
            version="",
            index_url="",
            cu_tag="",
            supported=False,
            rationale=f"No {backend.label} profile for {arch_name} ({sm}).",
        )

    # Derive the CUDA channel via the matrix.
    from mliport.install.compatibility import effective_cuda_channel

    cuda_tag = effective_cuda_channel(backend, arch, bp)
    from mliport.install.compatibility import CUDA_CHANNELS

    channel = CUDA_CHANNELS.get(cuda_tag)
    url = f"https://download.pytorch.org/whl/{cuda_tag}" if channel else ""

    if arch.experimental:
        rationale = (
            f"{backend.label} on {arch_name} ({sm}): EXPERIMENTAL. "
            f"torch {bp.framework_version}"
            f"+{cuda_tag} via legacy CUDA channel. Upstream does not test this GPU."
        )
    elif bp.mliport_verified:
        rationale = (
            f"{backend.label} on {arch_name} ({sm}): mliport-verified with torch "
            f"{bp.framework_version}+{cuda_tag}."
        )
    else:
        rationale = (
            f"{backend.label} on {arch_name} ({sm}): torch "
            f"{bp.framework_version}+{cuda_tag} (needs smoke test)."
        )

    return TorchRecommendation(
        version=f"{bp.framework_version}+{cuda_tag}",
        index_url=url,
        cu_tag=cuda_tag,
        supported=True,
        rationale=rationale,
        install_commands=[
            "# Recommended PyTorch build (see compatibility matrix):",
            f"uv pip install torch=={bp.framework_version}+{cuda_tag} "
            f"--index-url {url}",
        ],
        pyproject_snippet="",
    )


def engine_install_commands(gpus: Sequence[GpuInfo] | None, engine: str) -> list[str]:
    """Return copy-paste install commands for one mliport engine.

    Delegates to :func:`mliport.install.plan.generate_plan` — there is no
    independent installation logic here.
    """
    gpu_list = list(gpus) if gpus else None
    try:
        plan = generate_plan(
            gpus=gpu_list,
            engines=[engine],
            device="auto",
            verify=False,
        )
    except InstallPlanError as e:
        return [f"# {engine}: {e}"]
    # Render the plan as shell lines (strip the header, keep commands).
    shell = render_plan_shell(plan)
    lines = shell.splitlines()
    # Drop the leading "GPU arch / Source / Python" info lines.
    cmd_lines = [ln for ln in lines if ln.startswith("    $ ")]
    return [ln[6:] for ln in cmd_lines]


def format_setup_report(gpus: list[GpuInfo] | None) -> str:
    """Format a human-readable GPU setup report (delegates to the matrix)."""
    width = 68
    lines: list[str] = []
    lines.append("")
    lines.append("=" * width)
    lines.append(" mliport GPU Setup — compatibility report")
    lines.append("=" * width)
    lines.append("")

    if not gpus:
        lines.append("  No NVIDIA GPU detected (nvidia-smi unavailable).")
        lines.append("  mliport can still run with --device cpu.")
        lines.append("")
        lines.append("  Fastest path — one-command installer:")
        lines.append("    ./scripts/install_mliport.sh --device cpu")
        lines.append("")
        _append_per_engine(lines, gpus)
        lines.append("=" * width)
        return "\n".join(lines)

    oldest = _pick_oldest(gpus)
    arch = classify_gpu(oldest.cc_major, oldest.cc_minor)

    for i, gpu in enumerate(gpus):
        a = classify_gpu(gpu.cc_major, gpu.cc_minor)
        arch_label = a.label if a else "unknown"
        vram_gb = gpu.vram_mib / 1024
        lines.append(f"  GPU {i}: {gpu.name}")
        lines.append(
            f"        Architecture : {arch_label} (CC {gpu.compute_capability}, {gpu.sm})"
        )
        lines.append(f"        VRAM         : {vram_gb:.1f} GB")
        lines.append(f"        Driver       : {gpu.driver_version}")
        if gpu.vram_mib < MIN_VRAM_MIB_WARN:
            lines.append(
                f"        ! VRAM below {MIN_VRAM_MIB_WARN // 1024} GB — small "
                f"systems only."
            )
        if a and a.experimental:
            lines.append(
                f"        ! {a.label} is EXPERIMENTAL — upstream frameworks "
                f"may not include official kernels."
            )
        lines.append("")

    lines.append("-" * width)
    if arch is None:
        lines.append("  NOT SUPPORTED — no prebuilt PyTorch wheel for this GPU.")
        lines.append("  Use --device cpu.")
        lines.append("=" * width)
        return "\n".join(lines)

    lines.append(f"  Architecture profile : {arch.label}")
    lines.append(f"  Compute capability   : {oldest.sm}")
    if arch.experimental:
        lines.append("  Status               : EXPERIMENTAL (best-effort only)")
    lines.append("")

    # Per-backend status table from the matrix.
    lines.append("  Compatibility matrix (backend × arch):")
    lines.append("")
    lines.append(f"  {'Backend':<10} {'Version':<12} {'FW':<10} {'Status':<18}")
    lines.append(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*18}")
    for engine_name in ("uma", "mace", "dpa", "grace"):
        backend = BACKENDS[engine_name]
        bp = get_backend_arch_profile(engine_name, arch.name)
        if bp is None:
            continue
        lines.append(
            f"  {backend.engine:<10} {backend.version:<12} "
            f"{bp.framework_version:<10} {bp.status:<18}"
        )
    lines.append("")

    lines.append("  Fastest path — one-command installer:")
    lines.append("    ./scripts/install_mliport.sh")
    lines.append("")
    _append_per_engine(lines, gpus)

    lines.append("  If the download fails, check your network / proxy settings.")
    lines.append("  Then verify:  mliport doctor")
    lines.append("=" * width)
    return "\n".join(lines)


def _append_per_engine(lines: list[str], gpus: list[GpuInfo] | None) -> None:
    """Append per-engine copy-paste commands."""
    lines.append("  Per-engine install commands (from the compatibility matrix):")
    lines.append("")
    for engine_name, label in (
        ("uma", "UMA (FAIRChem)"),
        ("mace", "MACE"),
        ("dpa", "DPA / DeepMD"),
        ("grace", "GRACE"),
    ):
        lines.append(f"  --- {label} ---")
        for cmd in engine_install_commands(gpus, engine_name):
            lines.append(f"    {cmd}")
        lines.append("")


def setup_report_json(gpus: list[GpuInfo] | None) -> dict[str, Any]:
    """Build a JSON-serializable dict of the setup report (for --json)."""
    if not gpus:
        return {"gpus": [], "recommended": None, "has_gpu": False}

    oldest = _pick_oldest(gpus)
    arch = classify_gpu(oldest.cc_major, oldest.cc_minor)
    rec = recommend_torch(oldest.cc_major, oldest.cc_minor)

    result: dict[str, Any] = {
        "has_gpu": True,
        "gpus": [
            {
                "index": i,
                "name": g.name,
                "architecture": cc_arch_name(g.cc_major, g.cc_minor),
                "compute_capability": g.compute_capability,
                "sm": g.sm,
                "driver_version": g.driver_version,
                "vram_mib": g.vram_mib,
                "low_vram": g.vram_mib < MIN_VRAM_MIB_WARN,
            }
            for i, g in enumerate(gpus)
        ],
        "arch_profile": arch.name if arch else None,
        "recommended": {
            "version": rec.version,
            "index_url": rec.index_url,
            "cu_tag": rec.cu_tag,
            "supported": rec.supported,
            "rationale": rec.rationale,
            "install_commands": rec.install_commands,
            "pyproject_snippet": rec.pyproject_snippet,
        }
        if rec
        else None,
        "backends": {},
    }

    if arch:
        for engine_name in ("uma", "mace", "dpa", "grace"):
            backend = BACKENDS[engine_name]
            bp = get_backend_arch_profile(engine_name, arch.name)
            if bp:
                result["backends"][engine_name] = {
                    "version": backend.version,
                    "framework_version": bp.framework_version,
                    "status": bp.status,
                    "upstream_supported": bp.upstream_supported,
                    "workloads": bp.workload_support,
                    "notes": bp.notes,
                }

    return result
