# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Declarative compatibility matrix for mliport engines.

**Architecture profiles** describe GPU families (Maxwell, Pascal, …).
**CUDA channels** describe PyTorch wheel distribution families (cu126, cu128, …).
**Backend specs** describe each MLIP engine's upstream version, constraints,
and per-architecture profile (framework version, CUDA channel, verification
status).

Two concepts are deliberately separate:

* An architecture profile determines *which CUDA channel* is appropriate
  for a given GPU (e.g. Pascal → cu126 Legacy).
* The CUDA channel determines *which wheel URL* to use.
* The backend spec determines *which framework version* to install, and
  whether that combination has been smoke-tested by mliport.

Maintenance rule: when adding a new backend version or changing a torch/TF
pin, update this file.  The installer, ``mliport setup``, ``doctor``, and the
docs all derive their information from this single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from packaging.specifiers import SpecifierSet

PROJECT_REQUIRES_PYTHON = SpecifierSet(">=3.10,<3.14")
#: Python versions mliport's installer may target.  Backend specs narrow this
#: list; the installer fails closed when a requested version is incompatible
#: with any requested backend.
SUPPORTED_PYTHON_VERSIONS: tuple[str, ...] = ("3.10", "3.11", "3.12", "3.13")
# Fixed-version PyPI Requires-Python and published CPython wheel tags.
# Update together with framework pins; an unknown pin requires a new audit.
FRAMEWORK_PYTHON = {
    ("torch", "2.8.0"): SpecifierSet(">=3.9,<3.14"),
    ("torch", "2.10.0"): SpecifierSet(">=3.10,<3.15"),
    ("torch", "2.11.0"): SpecifierSet(">=3.10,<3.15"),
    ("tensorflow", "2.20.0"): SpecifierSet(">=3.9,<3.14"),
}

# ---------------------------------------------------------------------------
# Architecture profiles (GPU families)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchProfile:
    """One GPU architecture family.

    Attributes:
        name: Machine-readable key (``"pascal"``, ``"volta"`` …).
        label: Human-readable name (``"Pascal"``).
        cc_set: Explicit set of known compute capabilities in this family.
            Using an explicit set (rather than a continuous range) avoids
            misclassifying unknown future compute capabilities.
        experimental: If ``True``, mliport considers this GPU best-effort;
            the upstream frameworks may not ship official kernels for it.
    """

    name: str
    label: str
    cc_set: tuple[tuple[int, int], ...]
    experimental: bool = False


ARCH_PROFILES: dict[str, ArchProfile] = {
    "maxwell": ArchProfile(
        name="maxwell",
        label="Maxwell",
        cc_set=((5, 0), (5, 2)),
        experimental=True,
    ),
    "pascal": ArchProfile(
        name="pascal",
        label="Pascal",
        cc_set=((6, 0), (6, 1)),
    ),
    "volta": ArchProfile(
        name="volta",
        label="Volta",
        cc_set=((7, 0),),
    ),
    "turing": ArchProfile(
        name="turing",
        label="Turing",
        cc_set=((7, 5),),
    ),
    "ampere": ArchProfile(
        name="ampere",
        label="Ampere",
        cc_set=((8, 0), (8, 6)),
    ),
    "ada": ArchProfile(
        name="ada",
        label="Ada Lovelace",
        cc_set=((8, 9),),
    ),
    "hopper": ArchProfile(
        name="hopper",
        label="Hopper",
        cc_set=((9, 0),),
    ),
    # Blackwell: explicit known parts only.  Unknown future compute
    # capabilities (e.g. 11.x) are NOT auto-classified as Blackwell — they
    # should surface as "unknown / needs update" instead.
    "blackwell": ArchProfile(
        name="blackwell",
        label="Blackwell",
        cc_set=((10, 0), (12, 0)),
    ),
}


def classify_gpu(cc_major: int, cc_minor: int) -> ArchProfile | None:
    """Map a compute capability to its architecture profile.

    Returns ``None`` for unknown/unsupported compute capabilities (Kepler and
    older, or unknown future parts) rather than guessing.
    """
    cc = (cc_major, cc_minor)
    for profile in ARCH_PROFILES.values():
        if cc in profile.cc_set:
            return profile
    return None


# ---------------------------------------------------------------------------
# CUDA wheel channels (PyTorch distribution families)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CudaChannel:
    """One CUDA wheel distribution channel.

    Attributes:
        name: Machine-readable key (``"cu126"``, ``"cu128"`` …).
        label: Human-readable label (``"CUDA 12.6 (Legacy)"``).
        sm_min: Lowest SM the channel's kernels support (inclusive).
        sm_max: Highest SM the channel's kernels support (inclusive).
    """

    name: str
    label: str
    sm_min: tuple[int, int]
    sm_max: tuple[int, int]


CUDA_CHANNELS: dict[str, CudaChannel] = {
    "cu126": CudaChannel(
        name="cu126",
        label="CUDA 12.6 (Legacy)",
        sm_min=(5, 0),
        sm_max=(9, 0),
    ),
    "cu128": CudaChannel(
        name="cu128",
        label="CUDA 12.8",
        sm_min=(7, 5),
        sm_max=(12, 0),
    ),
    "cu130": CudaChannel(
        name="cu130",
        label="CUDA 13.0",
        sm_min=(7, 5),
        sm_max=(12, 0),
    ),
}

# CUDA channels that may contain the legacy sm_50/60/70 kernels.
_LEGACY_CUDA_CHANNELS = ("cu126",)


# ---------------------------------------------------------------------------
# Torch version → modern CUDA channel mapping
# ---------------------------------------------------------------------------
# For legacy GPUs (Maxwell/Pascal/Volta) the channel is always cu126.
# For modern GPUs (Turing+) the channel depends on the torch version:
#   torch 2.8.0 / 2.10.0 / 2.11.0 → cu128
# Other releases require an explicit wheel compatibility audit.
# This table is the single central mapping; update it when a new torch
# version changes its modern CUDA channel.


def _torch_modern_cuda(version: str) -> str:
    """Return the modern CUDA channel for a given torch version."""
    mapping = {
        "2.8.0": "cu128",
        "2.10.0": "cu128",
        "2.11.0": "cu128",
    }
    if version not in mapping:
        raise ValueError(
            f"Unknown torch version {version!r}: CUDA wheel compatibility "
            "requires a resolver update"
        )
    return mapping[version]


def select_cuda_channel(arch: ArchProfile, torch_version: str) -> str:
    """Derive the CUDA channel name for a torch backend on an architecture.

    Legacy GPUs (Maxwell/Pascal/Volta) always use ``cu126``.  Modern GPUs
    (Turing+) use the torch version's modern CUDA channel.
    """
    modern = _torch_modern_cuda(torch_version)
    if arch.name in ("maxwell", "pascal", "volta"):
        return "cu126"
    return modern


# ---------------------------------------------------------------------------
# Backend specs
# ---------------------------------------------------------------------------

Status = Literal["verified", "needs_smoke_test", "experimental"]


@dataclass(frozen=True)
class BackendArchProfile:
    """One backend × architecture combination.

    ``cuda_channel`` is optional: for torch backends it is derived from the
    architecture + framework version via :func:`select_cuda_channel`, unless
    an explicit override is supplied here.  GRACE (TensorFlow) leaves it as
    ``None``/``""``.

    Attributes:
        framework_version: Torch or TF version string (e.g. ``"2.8.0"``).
        cuda_channel: Optional explicit CUDA channel override.
        upstream_supported: The upstream project declares this combination
            as supported.
        extra_packages: Additional pip packages for this combination.
        notes: Human-readable rationale.
    """

    framework_version: str
    cuda_channel: str | None = None
    upstream_supported: bool = True
    extra_packages: tuple[str, ...] = ()
    notes: str = ""

    @property
    def status(self) -> Status:
        """Architecture compatibility is not a workload certification."""
        if not self.upstream_supported:
            return "experimental"
        return "needs_smoke_test"

    @property
    def mliport_verified(self) -> bool:
        """Compatibility shim: never make a model-independent verified claim."""
        return False

    @property
    def workload_support(self) -> dict:
        """Exact model/runtime evidence is resolved separately by capabilities."""
        return {
            workload: {"status": "not_run", "evidence_id": None}
            for workload in (
                "installation",
                "single_point",
                "short_md",
                "energy_force_consistency",
                "neb",
                "neighbor_cache",
            )
        }


@dataclass(frozen=True)
class BackendSpec:
    """One MLIP engine backend.

    ``distribution`` and ``version`` pin installed metadata. ``source_url``
    optionally selects an immutable upstream commit when a released wheel
    cannot load the requested checkpoints. ``install_extra`` holds additional
    runtime packages; build requirements and environment stay explicit.

    Attributes:
        engine: Machine-readable key (``"uma"``, ``"mace"`` …).
        label: Human-readable name.
        distribution: pip package name.
        version: Upstream version string.
        framework: ``"torch"`` or ``"tensorflow"``.
        upstream_constraint: The framework version constraint declared by the
            upstream project (PEP 440 specifier).
        arch_profiles: Per-architecture-profile entries.
        venv_name: Filesystem name for the isolated venv.
        install_extra: Extra pip packages installed alongside this backend
            (e.g. ``("e3nn==0.4.4",)`` for MACE).  The backend's own package
            (``distribution==version``) is appended automatically.
        source_url: Optional Git URL ending in a full commit SHA.
        build_packages: Dependencies installed before a non-isolated source build.
        build_env: Explicit source-build policy, without replacing torch/TF pins.
    """

    engine: str
    label: str
    distribution: str
    version: str
    framework: str
    upstream_constraint: str
    requires_python: SpecifierSet
    arch_profiles: dict[str, BackendArchProfile] = field(default_factory=dict)
    venv_name: str = ""
    install_extra: tuple[str, ...] = ()
    source_url: str | None = None
    build_packages: tuple[str, ...] = ()
    build_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.venv_name:
            object.__setattr__(self, "venv_name", f".venv-{self.engine}")
        if not self.distribution or not self.version:
            raise ValueError(
                f"BackendSpec '{self.engine}' needs a non-empty distribution "
                f"and version"
            )
        if self.source_url and not re.search(r"@[0-9a-f]{40}$", self.source_url):
            raise ValueError("Backend source must pin a full immutable Git commit")

    @property
    def requirement(self) -> str:
        """The pinned backend package requirement (``dist==version``)."""
        if self.source_url:
            return f"{self.distribution} @ {self.source_url}"
        return f"{self.distribution}=={self.version}"

    def install_packages(self) -> tuple[str, ...]:
        """Full list of packages to install for this backend.

        Includes the backend's own pinned package plus any ``install_extra``.
        """
        return (self.requirement, *self.install_extra)


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

BACKENDS: dict[str, BackendSpec] = {
    # ── UMA (FAIRChem) ────────────────────────────────────────────────
    "uma": BackendSpec(
        engine="uma",
        label="UMA (FAIRChem)",
        distribution="fairchem-core",
        version="2.21.0",
        framework="torch",
        upstream_constraint="torch~=2.8.0",
        requires_python=SpecifierSet(">=3.11,<3.14"),
        venv_name=".venv",
        arch_profiles={
            "maxwell": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=False,
                notes="Experimental. fairchem-core 2.21.0 → torch~=2.8.0. "
                "Maxwell via cu126 legacy channel; upstream does not test this GPU.",
            ),
            "pascal": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="torch 2.8.0+cu126 includes sm60 kernel. "
                "fairchem-core 2.21.0 constraint torch~=2.8.0 satisfied.",
            ),
            "volta": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Architecture profile: V100 (sm_70). "
                "torch 2.8.0+cu126 includes sm70 kernel.",
            ),
            "turing": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128) for Turing+.",
            ),
            "ampere": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128) for Ampere.",
            ),
            "ada": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Architecture profile: RTX 4090. Modern CUDA (cu128) for Ada.",
            ),
            "hopper": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128) for Hopper.",
            ),
            "blackwell": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128) for Blackwell.",
            ),
        },
    ),
    # ── MACE ──────────────────────────────────────────────────────────
    "mace": BackendSpec(
        engine="mace",
        label="MACE",
        distribution="mace-torch",
        version="0.3.16",
        framework="torch",
        upstream_constraint="torch>=1.12",
        requires_python=SpecifierSet(">=3.9"),
        install_extra=("e3nn==0.4.4",),
        arch_profiles={
            "maxwell": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=False,
                notes="Experimental. MACE 0.3.16 has torch>=1.12, no upper bound. "
                "Maxwell via cu126; upstream does not test this GPU.",
            ),
            "pascal": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="torch 2.8.0+cu126 includes sm60. "
                "MACE 0.3.16 explicitly fixed torch 2.8 compile issues.",
            ),
            "volta": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Architecture profile: V100 (sm_70). "
                "torch 2.8.0+cu126 includes sm70 kernel.",
            ),
            "turing": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128).",
            ),
            "ampere": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128).",
            ),
            "ada": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Architecture profile: RTX 4090 with modern CUDA (cu128).",
            ),
            "hopper": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128).",
            ),
            "blackwell": BackendArchProfile(
                framework_version="2.8.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128).",
            ),
        },
    ),
    # ── DPA / DeepMD ──────────────────────────────────────────────────
    "dpa": BackendSpec(
        engine="dpa",
        label="DPA (DeepMD-kit)",
        distribution="deepmd-kit",
        version="3.2.1.dev63+g0edeaef56",
        framework="torch",
        upstream_constraint="torch==2.11.0",
        requires_python=SpecifierSet(">=3.10"),
        # DPA4's SO(3) operations and the upstream neighbor-list path need
        # the PyTorch extra's e3nn / vesin dependencies. Keep optional fused
        # nvalchemi kernels out of the portable eager runtime. MPICH metadata
        # is needed when importing the wheel's compiled PyTorch operators.
        install_extra=(
            "e3nn==0.6.0",
            "vesin[torch]==0.6.2",
            "vesin-torch==0.6.2",
            "mpich>=5.0,<6",
        ),
        # The released 3.2.0 wheel rejects the gaussian/fix basis in the
        # published DPA4C checkpoints. Use the upstream implementation of
        # that basis, with an immutable source and metadata version.
        source_url=(
            "git+https://github.com/deepmodeling/deepmd-kit.git@"
            "0edeaef5617e9878edeb2346bf37bb6a035a4f51"
        ),
        build_packages=(
            "scikit-build-core>=1,<2",
            "setuptools-scm>=8,<11",
            "hatch-fancy-pypi-readme>=22.1,<26",
            "dependency-groups>=1,<2",
            "cmake>=3.24,<5",
            "ninja>=1.11,<2",
        ),
        build_env={
            "DP_ENABLE_TENSORFLOW": "0",
            "DP_ENABLE_PYTORCH": "1",
            "DP_VARIANT": "cpu",
            "CMAKE_BUILD_PARALLEL_LEVEL": "4",
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DEEPMD_KIT": "3.2.1.dev63+g0edeaef56",
        },
        arch_profiles={
            "maxwell": BackendArchProfile(
                framework_version="2.11.0",
                upstream_supported=False,
                notes="Experimental. The DPA4 runtime uses torch==2.11.0. "
                "Maxwell via cu126; must be smoke-tested on real hardware.",
            ),
            "pascal": BackendArchProfile(
                framework_version="2.11.0",
                upstream_supported=True,
                notes="P40 installation exposed a CUDA-wheel selection bug; "
                "torch 2.11.0+cu126 includes sm60 for sm61 compatibility and "
                "must be pinned including the +cu126 local version. Runtime "
                "model smoke retest is still required.",
            ),
            "volta": BackendArchProfile(
                framework_version="2.11.0",
                upstream_supported=True,
                notes="Architecture profile: V100 (sm_70). "
                "torch 2.11.0+cu126 includes sm70 kernel.",
            ),
            "turing": BackendArchProfile(
                framework_version="2.11.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128).",
            ),
            "ampere": BackendArchProfile(
                framework_version="2.11.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128).",
            ),
            "ada": BackendArchProfile(
                framework_version="2.11.0",
                upstream_supported=True,
                notes="Architecture profile: RTX 4090 with modern CUDA (cu128).",
            ),
            "hopper": BackendArchProfile(
                framework_version="2.11.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128).",
            ),
            "blackwell": BackendArchProfile(
                framework_version="2.11.0",
                upstream_supported=True,
                notes="Modern CUDA (cu128).",
            ),
        },
    ),
    # ── NEP ───────────────────────────────────────────────────────────
    # NEP is a potential family, not a framework-specific network.  The ASE
    # runtime needs the compiled ``calorine`` extension (numpy + ASE); the
    # native runtime needs the separately built ``gpumd`` binary.  There is no
    # torch/tensorflow wheel to pin, so ``framework`` is ``"numpy"`` and the
    # arch profiles describe the CUDA toolkit the gpumd binary must target.
    "nep": BackendSpec(
        engine="nep",
        label="NEP (Calorine / GPUMD)",
        distribution="calorine",
        version="4.1",
        framework="numpy",
        upstream_constraint="calorine>=3.5; gpumd built from source",
        requires_python=SpecifierSet(">=3.9"),
        arch_profiles={
            "maxwell": BackendArchProfile(
                framework_version="12.9",
                upstream_supported=False,
                notes="gpumd must be built for sm_50; CUDA 13 dropped Maxwell. "
                "Not verified by mliport.",
            ),
            "pascal": BackendArchProfile(
                framework_version="12.9",
                upstream_supported=True,
                notes="Build gpumd with CUDA_ARCH=-arch=sm_60.",
            ),
            "volta": BackendArchProfile(
                framework_version="12.9",
                upstream_supported=True,
                notes="V100 (sm_70). Build gpumd with CUDA_ARCH=-arch=sm_70; "
                "verified locally with GPUMD 5.9.1 + CUDA 12.9.",
            ),
            "turing": BackendArchProfile(
                framework_version="12.9",
                upstream_supported=True,
                notes="Build gpumd with CUDA_ARCH=-arch=sm_75.",
            ),
            "ampere": BackendArchProfile(
                framework_version="12.9",
                upstream_supported=True,
                notes="Build gpumd with CUDA_ARCH=-arch=sm_80.",
            ),
            "ada": BackendArchProfile(
                framework_version="12.9",
                upstream_supported=True,
                notes="Build gpumd with CUDA_ARCH=-arch=sm_89.",
            ),
            "hopper": BackendArchProfile(
                framework_version="12.9",
                upstream_supported=True,
                notes="Build gpumd with CUDA_ARCH=-arch=sm_90.",
            ),
            "blackwell": BackendArchProfile(
                framework_version="12.9",
                upstream_supported=False,
                notes="gpumd must be built for sm_100/sm_120 with a CUDA "
                "toolkit that supports Blackwell. Not verified by mliport.",
            ),
        },
    ),
    # ── GRACE ─────────────────────────────────────────────────────────
    "grace": BackendSpec(
        engine="grace",
        label="GRACE",
        distribution="tensorpotential",
        version="0.6.0",
        framework="tensorflow",
        upstream_constraint="tensorflow<=2.20",
        requires_python=SpecifierSet(">=3.9"),
        arch_profiles={
            # All GPU archs use TF 2.20.0 because tensorpotential 0.6.0
            # pins tensorflow<=2.20.  The only difference is Maxwell:
            # TF 2.20 official wheel build target starts at sm_60.
            "maxwell": BackendArchProfile(
                framework_version="2.20.0",
                upstream_supported=False,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="Experimental. TF 2.20 official wheel build target "
                "starts at sm_60 (Pascal); Maxwell sm_50 not in the wheel. "
                "May need source-built TF.",
            ),
            "pascal": BackendArchProfile(
                framework_version="2.20.0",
                upstream_supported=True,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "Official wheel includes sm_60 target.",
            ),
            "volta": BackendArchProfile(
                framework_version="2.20.0",
                upstream_supported=True,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="Architecture profile: V100 (sm_70). "
                "TF 2.20.0 + CUDA 12.5 + cuDNN 9.3.",
            ),
            "turing": BackendArchProfile(
                framework_version="2.20.0",
                upstream_supported=True,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "sm75 via sm70 SASS forward-compat.",
            ),
            "ampere": BackendArchProfile(
                framework_version="2.20.0",
                upstream_supported=True,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "sm86 via sm80 SASS forward-compat.",
            ),
            "ada": BackendArchProfile(
                framework_version="2.20.0",
                upstream_supported=True,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="Architecture profile: RTX 4090. TF 2.20.0 + CUDA 12.5 + "
                "cuDNN 9.3; sm89 explicitly compiled.",
            ),
            "hopper": BackendArchProfile(
                framework_version="2.20.0",
                upstream_supported=False,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "compute_90 via PTX JIT (experimental).",
            ),
            "blackwell": BackendArchProfile(
                framework_version="2.20.0",
                upstream_supported=False,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "Blackwell not in TF 2.20 build target; PTX JIT may work.",
            ),
        },
    ),
}


def get_backend_arch_profile(engine: str, arch_name: str) -> BackendArchProfile | None:
    """Look up the backend×arch profile entry."""
    backend = BACKENDS.get(engine)
    if backend is None:
        return None
    return backend.arch_profiles.get(arch_name)


def effective_cuda_channel(
    backend: BackendSpec, arch: ArchProfile, bp: BackendArchProfile
) -> str:
    """Return the effective CUDA channel for a backend on an architecture.

    Torch backends derive the channel from the architecture + framework
    version unless an explicit override is set.  TensorFlow backends (GRACE)
    have no torch CUDA channel and return ``""``.
    """
    if backend.framework != "torch":
        return ""
    if bp.cuda_channel:
        return bp.cuda_channel
    return select_cuda_channel(arch, bp.framework_version)


def framework_versions(backend: BackendSpec) -> tuple[str, ...]:
    """Framework versions this backend is pinned to across architectures."""
    return tuple(
        sorted(
            {profile.framework_version for profile in backend.arch_profiles.values()}
        )
    )


def python_versions_for(engine: str) -> tuple[str, ...]:
    """Supported Python versions for one backend (registry intersection).

    Combines mliport's own range, the backend's ``requires-python`` and every
    framework wheel's Python range.  The result is the authoritative list the
    installer enforces and the docs render.
    """
    backend = BACKENDS[engine]
    constraint = PROJECT_REQUIRES_PYTHON & backend.requires_python
    for version in framework_versions(backend):
        key = (backend.framework, version)
        if key in FRAMEWORK_PYTHON:
            constraint &= FRAMEWORK_PYTHON[key]
    return tuple(v for v in SUPPORTED_PYTHON_VERSIONS if v in constraint)


def runtime_stack_text(engine: str, lang: str = "en") -> str:
    """Human-readable runtime stack of one backend for the generated docs.

    For torch/TensorFlow backends the version is the pinned framework wheel.
    NEP has no framework wheel: Calorine sits on NumPy/ASE (resolved by the
    installer, not pinned here), and the only version in its arch profiles is
    the **CUDA Toolkit** the separately built ``gpumd`` binary targets.  Those
    two must never be rendered as one "numpy 12.9".
    """
    backend = BACKENDS[engine]
    versions = "/".join(framework_versions(backend))
    if backend.framework == "numpy":
        if lang == "zh":
            return (
                f"`{backend.requirement}`（NumPy/ASE 由安装器解析；"
                f"gpumd 需用 CUDA Toolkit {versions} 编译）"
            )
        return (
            f"`{backend.requirement}` (NumPy/ASE, resolved by the installer; "
            f"gpumd built with CUDA Toolkit {versions})"
        )
    return f"`{backend.requirement}` ({backend.framework} {versions})"


def backend_environment_rows(lang: str = "en") -> list[dict[str, str]]:
    """One row per backend for the generated documentation matrix."""
    rows: list[dict[str, str]] = []
    for engine, backend in BACKENDS.items():
        versions = python_versions_for(engine)
        rows.append(
            {
                "engine": engine,
                "label": backend.label,
                "environment": backend.venv_name,
                "python": ", ".join(versions),
                "runtime": runtime_stack_text(engine, lang),
            }
        )
    return rows


__all__ = [
    "BACKENDS",
    "FRAMEWORK_PYTHON",
    "PROJECT_REQUIRES_PYTHON",
    "SUPPORTED_PYTHON_VERSIONS",
    "backend_environment_rows",
    "framework_versions",
    "python_versions_for",
    "runtime_stack_text",
]
