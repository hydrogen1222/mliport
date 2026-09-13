# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU compute-capability compatibility helpers.

PyTorch ships SASS kernels for a fixed set of architectures
(``torch.cuda.get_arch_list()``). A device of compute capability ``sm_XZ`` can
run any kernel compiled for ``sm_XY`` where ``Y <= Z`` (binary compatibility
within the same major revision). This means, for example, that a P104-100
(``sm_61``) can run kernels compiled for ``sm_60`` even though ``sm_61`` is not
literally present in the arch list — the strict ``gpu_cc in arch_list`` check is
a false negative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parse_sm(token: str) -> tuple[int, int] | None:
    """Parse an ``sm_XY`` token into a ``(major, minor)`` pair.

    Args:
        token: Architecture token such as ``"sm_60"`` or ``"compute_70"``.

    Returns:
        ``(major, minor)`` for ``sm_*`` tokens, else ``None`` (e.g. for
        ``compute_*`` PTX-only entries which are not SASS binaries).
    """
    if not token.startswith("sm_"):
        return None
    num = token[3:]
    if not num.isdigit() or len(num) < 2:
        return None
    return int(num[:-1]), int(num[-1])


def arch_supports_device(gpu_cc: str, arch_list: Sequence[str]) -> bool:
    """Check whether any compiled kernel can run on the given GPU.

    A kernel compiled for ``sm_XY`` runs on a device of compute capability
    ``sm_XZ`` when ``X`` matches and ``Y <= Z`` (minor-revision binary
    compatibility within the same major). An empty arch list is treated as
    "unknown / no restriction".

    Args:
        gpu_cc: Device compute capability as ``sm_XY`` (e.g. ``"sm_61"``).
        arch_list: ``torch.cuda.get_arch_list()`` output.

    Returns:
        True if the PyTorch build can execute on this GPU.
    """
    if not arch_list:
        return True

    gpu = _parse_sm(gpu_cc)
    if gpu is None:
        # Unrecognized device token; fall back to exact-match only.
        return gpu_cc in arch_list

    gpu_major, gpu_minor = gpu
    for token in arch_list:
        arch = _parse_sm(token)
        if arch is None:
            continue
        arch_major, arch_minor = arch
        if arch_major == gpu_major and arch_minor <= gpu_minor:
            return True

    return gpu_cc in arch_list
