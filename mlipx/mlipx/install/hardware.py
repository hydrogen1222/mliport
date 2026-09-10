# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU hardware detection via ``nvidia-smi`` — works *before* PyTorch is installed.

Unlike :mod:`mlipx.gpu_compat` (which does compute-capability *math* against an
already-installed PyTorch's ``arch_list``), this module detects the physical
hardware and maps a GPU's compute capability to an architecture profile.

The single source of GPU classification is :data:`mlipx.install.compatibility.ARCH_PROFILES`.
:func:`cc_arch_name` and :func:`classify_gpu` both delegate to it so there is
only one place to update when a new GPU architecture is added.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from mlipx.install.compatibility import classify_gpu
from mlipx.devices import VisibleGpu, visible_gpus

# Minimum VRAM (MiB) to comfortably run the UMA-s model (~1.1 GB) on small
# systems. Below this we warn (still allowed).
MIN_VRAM_MIB_WARN = 2048


@dataclass
class GpuInfo:
    """One physical CUDA GPU as reported by ``nvidia-smi``."""

    name: str
    cc_major: int
    cc_minor: int
    driver_version: str
    vram_mib: int
    uuid: str | None = None
    physical_index: int | None = None
    logical_index: int | None = None

    @property
    def compute_capability(self) -> str:
        """Compute capability as ``X.Y`` (e.g. ``"6.1"``)."""
        return f"{self.cc_major}.{self.cc_minor}"

    @property
    def sm(self) -> str:
        """Compute capability as ``sm_XY`` (e.g. ``"sm_61"``)."""
        return f"sm_{self.cc_major}{self.cc_minor}"


def cc_arch_name(major: int, minor: int) -> str:
    """Map a compute capability to a human architecture name.

    Delegates to :func:`classify_gpu` so GPU classification lives in exactly
    one place (the compatibility matrix).  Unsupported/unknown parts are
    identified separately (Kepler and older vs unknown future parts).
    """
    profile = classify_gpu(major, minor)
    if profile is not None:
        return profile.label
    if major == 3:
        return "Kepler"
    if major < 3:
        return f"unknown (sm_{major}{minor})"
    return f"unknown (sm_{major}{minor})"


def detect_gpus() -> list[GpuInfo] | None:
    """Detect CUDA GPUs via ``nvidia-smi`` (no PyTorch needed).

    Returns:
        List of :class:`GpuInfo` (one per GPU), or ``None`` if ``nvidia-smi``
        is unavailable (no NVIDIA driver installed). Never raises.
    """
    query = (
        "name,compute_cap,driver_version,memory.total,index,uuid",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query[0]}",
                *query[1:],
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    if result.returncode != 0:
        return None

    gpus: list[GpuInfo] = []
    import csv as _csv
    import io as _io

    for parts in _csv.reader(_io.StringIO(result.stdout)):
        parts = [p.strip() for p in parts]
        if not parts or all(p == "" for p in parts):
            continue
        if len(parts) != 6:
            continue
        name, cc, driver, vram = parts[0], parts[1], parts[2], parts[3]
        try:
            cc_major_str, _, cc_minor_str = cc.partition(".")
            cc_major = int(cc_major_str)
            cc_minor = int(cc_minor_str) if cc_minor_str else 0
            vram_mib = int(vram)
            physical_index = int(parts[4])
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                name=name,
                cc_major=cc_major,
                cc_minor=cc_minor,
                driver_version=driver,
                vram_mib=vram_mib,
                physical_index=physical_index,
                uuid=parts[5],
            )
        )

    try:
        visible = visible_gpus(
            [
                VisibleGpu(
                    i,
                    gpu.uuid,
                    gpu.physical_index,
                    gpu.name,
                    (gpu.cc_major, gpu.cc_minor),
                )
                for i, gpu in enumerate(gpus)
            ]
        )
    except RuntimeError:
        return None
    by_uuid = {gpu.uuid: gpu for gpu in gpus}
    selected = []
    for entry in visible:
        gpu = by_uuid[entry.uuid]
        gpu.logical_index = entry.logical_index
        selected.append(gpu)
    return selected or None


def _pick_oldest(gpus: Sequence[GpuInfo]) -> GpuInfo:
    """Return the GPU with the lowest compute capability (most conservative)."""
    return min(gpus, key=lambda g: (g.cc_major, g.cc_minor))
