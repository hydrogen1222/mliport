# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
mliport installation infrastructure — GPU detection, compatibility matrix,
source profiles, and installation plan generation.

This subpackage is deliberately kept free of torch/fairchem/tensorflow
imports so it can run *before* any MLIP backend is installed.
"""

from __future__ import annotations

from mliport.install.hardware import (
    GpuInfo,
    cc_arch_name,
    classify_gpu,
    detect_gpus,
)
from mliport.install.compatibility import (
    ArchProfile,
    BackendArchProfile,
    BackendSpec,
    CudaChannel,
    ARCH_PROFILES,
    BACKENDS,
    CUDA_CHANNELS,
    effective_cuda_channel,
    select_cuda_channel,
)
from mliport.install.sources import (
    SourceProfile,
    SOURCE_PROFILES,
    resolve_source,
    build_package_source_args,
    build_torch_source_args,
    build_offline_args,
)
from mliport.install.plan import (
    InstallPlan,
    InstallPlanError,
    InstallStep,
    generate_plan,
    normalize_engines,
    plan_to_json,
    render_plan_shell,
    validate_python_version,
)

__all__ = [
    # hardware
    "GpuInfo",
    "cc_arch_name",
    "classify_gpu",
    "detect_gpus",
    # compatibility
    "ArchProfile",
    "BackendArchProfile",
    "BackendSpec",
    "CudaChannel",
    "ARCH_PROFILES",
    "BACKENDS",
    "CUDA_CHANNELS",
    "effective_cuda_channel",
    "select_cuda_channel",
    # sources
    "SourceProfile",
    "SOURCE_PROFILES",
    "resolve_source",
    "build_package_source_args",
    "build_torch_source_args",
    "build_offline_args",
    # plan
    "InstallPlan",
    "InstallPlanError",
    "InstallStep",
    "generate_plan",
    "normalize_engines",
    "plan_to_json",
    "render_plan_shell",
    "validate_python_version",
]
