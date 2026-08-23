# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Source profiles — where to download packages from.

PyPI packages (fairchem-core, mace-torch, deepmd-kit, tensorflow, …) and
PyTorch CUDA wheels are handled through **separate** channels.  The installer
never touches the user's ``~/.config/uv/uv.toml``; it sets ``UV_NO_CONFIG=1``
and passes index/find-links through CLI flags / environment variables.

Source profiles:

    ``auto``
        Resolves to ``official`` (lightweight connectivity probing is
        explicitly out of scope for now).
    ``official``
        PyPI: pypi.org.  PyTorch: download.pytorch.org.
    ``china``
        Interactive China-mirror alias in the CLI (TUNA is the deterministic
        default for non-interactive callers).  The explicit ``china-*``
        profiles select TUNA, Aliyun, USTC, or Tencent Cloud for PyPI.
        PyTorch wheels use mirrors.aliyun.com via ``--find-links``, because
        that mirror is a flat HTML listing, not a PEP 503 /simple/ registry.
    ``offline``
        No network at all.  Uses ``uv pip install --offline`` so only the
        local uv/pip cache is consulted.  No install command may contain a URL.
    ``custom``
        The user supplies ``UV_INDEX_URL`` / ``UV_EXTRA_INDEX_URL`` /
        ``UV_FIND_LINKS`` etc. via the environment.  The installer passes them
        through unchanged and does not add its own index overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceProfile:
    """One package source configuration.

    Attributes:
        name: Machine-readable key (``"official"``, ``"china"`` …).
        label: Human-readable label.
        pypi_index: PyPI index URL for ordinary packages (``None`` = default).
        pypi_extra_index: Optional extra PyPI index.
        pytorch_index: PyTorch CUDA wheel index URL (``None`` = use
            ``download.pytorch.org/whl/{cuda_tag}``).
        pytorch_find_links: Flat HTML mirror for PyTorch wheels
            (``None`` = no find-links).  Used for Aliyun.
        offline: If ``True``, every install step must use uv's ``--offline``
            flag and no command may contain a URL.
        env: Environment variables to set during installation.
    """

    name: str
    label: str
    pypi_index: str | None = None
    pypi_extra_index: str | None = None
    pytorch_index: str | None = None
    pytorch_find_links: str | None = None
    offline: bool = False
    env: dict[str, str] = field(default_factory=dict)


# Official PyTorch CUDA index template.  ``{cuda_tag}`` is substituted with
# the channel name (``cu126``, ``cu128`` …).
_PYTORCH_OFFICIAL = "https://download.pytorch.org/whl/{cuda_tag}"
# Aliyun PyTorch flat mirror.  ``{cuda_tag}`` is substituted the same way.
_PYTORCH_ALIYUN = "https://mirrors.aliyun.com/pytorch-wheels/{cuda_tag}/"

# Ordered choices shown by ``--source china`` in an interactive terminal.
# Keep the stable, long-standing TUNA profile first so non-interactive callers
# and an empty menu response preserve the historical behavior.
CHINA_SOURCE_CHOICES: tuple[str, ...] = (
    "china",
    "china-aliyun",
    "china-ustc",
    "china-tencent",
    "official",
)

SOURCE_PROFILES: dict[str, SourceProfile] = {
    "auto": SourceProfile(
        name="auto",
        label="auto (→ official)",
        # auto currently resolves to official; connectivity probing is out of
        # scope (see plan §86).
    ),
    "official": SourceProfile(
        name="official",
        label="Official (PyPI + pytorch.org)",
    ),
    "china": SourceProfile(
        name="china",
        label="China: TUNA PyPI + Aliyun PyTorch",
        pypi_index="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        # Aliyun PyTorch mirror is a flat HTML listing, not a PEP 503
        # registry, so it must be passed via --find-links, not --index-url.
        pytorch_find_links=_PYTORCH_ALIYUN,
    ),
    "china-aliyun": SourceProfile(
        name="china-aliyun",
        label="China: Aliyun PyPI + Aliyun PyTorch",
        pypi_index="https://mirrors.aliyun.com/pypi/simple/",
        pytorch_find_links=_PYTORCH_ALIYUN,
    ),
    "china-ustc": SourceProfile(
        name="china-ustc",
        label="China: USTC PyPI + Aliyun PyTorch",
        pypi_index="https://mirrors.ustc.edu.cn/pypi/simple/",
        pytorch_find_links=_PYTORCH_ALIYUN,
    ),
    "china-tencent": SourceProfile(
        name="china-tencent",
        label="China: Tencent Cloud PyPI + Aliyun PyTorch",
        pypi_index="https://mirrors.cloud.tencent.com/pypi/simple/",
        pytorch_find_links=_PYTORCH_ALIYUN,
    ),
    "offline": SourceProfile(
        name="offline",
        label="Offline (uv/pip cache only, no network)",
        offline=True,
        env={"UV_NO_CONFIG": "1", "UV_OFFLINE": "1"},
    ),
    "custom": SourceProfile(
        name="custom",
        label="Custom (user-provided env vars)",
        # No index overrides here: the user's UV_INDEX_URL / UV_FIND_LINKS /
        # UV_EXTRA_INDEX_URL are inherited from the process environment.
    ),
}


def resolve_source(name: str) -> SourceProfile:
    """Resolve a source profile name to its :class:`SourceProfile`.

    ``"auto"`` currently resolves to ``"official"``.

    Args:
        name: Source profile name.

    Returns:
        The resolved :class:`SourceProfile`.

    Raises:
        ValueError: If the profile name is unknown.
    """
    if name == "auto":
        name = "official"
    if name not in SOURCE_PROFILES:
        raise ValueError(
            f"Unknown source profile '{name}'. "
            f"Choose from: {', '.join(SOURCE_PROFILES)}"
        )
    return SOURCE_PROFILES[name]


# ---------------------------------------------------------------------------
# uv pip source argument builders
# ---------------------------------------------------------------------------


def build_package_source_args(profile: SourceProfile) -> list[str]:
    """Return ``uv pip install`` source args for ordinary PyPI packages.

    For ``china`` this adds ``--index-url <tuna>`` so every ordinary package
    (fairchem-core, mace-torch, deepmd-kit, tensorflow, …) actually uses the
    China PyPI mirror, not just torch.
    """
    args: list[str] = []
    if profile.name == "custom":
        # custom inherits the user's UV_INDEX_URL / UV_EXTRA_INDEX_URL.
        return args
    if profile.pypi_index:
        args += ["--index-url", profile.pypi_index]
    if profile.pypi_extra_index:
        args += ["--extra-index-url", profile.pypi_extra_index]
    return args


def build_torch_source_args(profile: SourceProfile, cuda_tag: str) -> list[str]:
    """Return ``uv pip install`` source args for a PyTorch wheel.

    ``cuda_tag`` is the wheel channel name (``cpu``, ``cu126``, ``cu128`` …).
    For China profiles this uses ``--find-links`` against the Aliyun flat
    mirror; for ``official`` it uses ``--index-url`` against
    download.pytorch.org. Offline profiles return no source args
    (``--offline`` handles everything).
    """
    if profile.offline or profile.name == "custom":
        # offline: --offline handles everything.  custom: inherit the user's
        # UV_INDEX_URL / UV_FIND_LINKS from the environment, don't override.
        return []
    if profile.pytorch_find_links:
        # The flat wheel page supplies torch itself; its transitive packages
        # must still use the PyPI mirror selected by the user instead of
        # silently falling back to the default index.
        return [
            *build_package_source_args(profile),
            "--find-links",
            profile.pytorch_find_links.format(cuda_tag=cuda_tag),
        ]
    if profile.pytorch_index:
        return ["--index-url", profile.pytorch_index]
    # official default
    return ["--index-url", _PYTORCH_OFFICIAL.format(cuda_tag=cuda_tag)]


def build_offline_args(profile: SourceProfile) -> list[str]:
    """Return ``uv pip install`` offline flags for an offline profile."""
    if profile.offline:
        return ["--offline"]
    return []
