"""Tests for mliport.install.compatibility — matrix invariants."""

from __future__ import annotations

import pytest
from packaging.specifiers import SpecifierSet

from mliport.install.compatibility import (
    ARCH_PROFILES,
    BACKENDS,
    CUDA_CHANNELS,
    BackendSpec,
    classify_gpu,
    effective_cuda_channel,
    select_cuda_channel,
)

# ---------------------------------------------------------------------------
# Blackwell classification
# ---------------------------------------------------------------------------


def test_blackwell_known_parts_only() -> None:
    """Unknown future CC (e.g. 11.x) must NOT be auto-classified as Blackwell."""
    assert classify_gpu(10, 0) is not None
    assert classify_gpu(10, 0).name == "blackwell"
    assert classify_gpu(12, 0).name == "blackwell"
    # Unknown future parts → None (needs update), never guessed.
    assert classify_gpu(11, 0) is None
    assert classify_gpu(10, 8) is None
    assert classify_gpu(5, 1) is None  # not a real Maxwell part


def test_cc_arch_name_consistent_with_classify() -> None:
    from mliport.install.hardware import cc_arch_name

    # Every known arch profile must be reachable via cc_arch_name with its label.
    for name, prof in ARCH_PROFILES.items():
        cc = prof.cc_set[0]
        assert cc_arch_name(cc[0], cc[1]) == prof.label
    # Kepler and unknown are separate.
    assert cc_arch_name(3, 7) == "Kepler"
    assert "unknown" in cc_arch_name(4, 0)


# ---------------------------------------------------------------------------
# CUDA channel derivation (no duplicate hand-written data)
# ---------------------------------------------------------------------------


def test_select_cuda_channel_legacy_vs_modern() -> None:
    legacy = {"maxwell", "pascal", "volta"}
    for prof in ARCH_PROFILES.values():
        if prof.name in legacy:
            assert select_cuda_channel(prof, "2.8.0") == "cu126"
        else:
            assert select_cuda_channel(prof, "2.8.0") == "cu128"


def test_torch_channel_mapping() -> None:
    from mliport.install.compatibility import _torch_modern_cuda

    assert _torch_modern_cuda("2.8.0") == "cu128"
    assert _torch_modern_cuda("2.10.0") == "cu128"


@pytest.mark.parametrize("version", ["2.8.1", "2.12.0", "2.13.1", "99.0.0", "invalid"])
def test_unknown_torch_channel_fails_on_every_architecture(version):
    for arch in ARCH_PROFILES.values():
        with pytest.raises(ValueError, match="resolver update"):
            select_cuda_channel(arch, version)


# ---------------------------------------------------------------------------
# Matrix invariants
# ---------------------------------------------------------------------------


def test_backend_keys_match() -> None:
    for key, backend in BACKENDS.items():
        assert backend.engine == key
        assert backend.version  # non-empty
        assert backend.distribution  # non-empty
        assert backend.framework in ("torch", "tensorflow")


def test_arch_profile_keys_exist() -> None:
    for backend in BACKENDS.values():
        for arch_name in backend.arch_profiles:
            assert (
                arch_name in ARCH_PROFILES
            ), f"{backend.engine} references unknown arch '{arch_name}'"


def test_backend_requirement_matches_distribution_version() -> None:
    for backend in BACKENDS.values():
        assert backend.requirement == f"{backend.distribution}=={backend.version}"


def test_framework_version_nonempty() -> None:
    for backend in BACKENDS.values():
        for bp in backend.arch_profiles.values():
            assert bp.framework_version


def test_verified_implies_upstream_supported() -> None:
    for backend in BACKENDS.values():
        for bp in backend.arch_profiles.values():
            if bp.mliport_verified:
                assert bp.upstream_supported


def test_experimental_not_verified() -> None:
    for backend in BACKENDS.values():
        for arch_name, bp in backend.arch_profiles.items():
            if ARCH_PROFILES[arch_name].experimental or not bp.upstream_supported:
                assert bp.status == "experimental"


def test_cuda_channel_supports_arch() -> None:
    """The effective CUDA channel must support the architecture's SM."""
    for backend in BACKENDS.values():
        for arch_name, bp in backend.arch_profiles.items():
            if backend.framework != "torch":
                continue
            arch = ARCH_PROFILES[arch_name]
            tag = effective_cuda_channel(backend, arch, bp)
            assert tag in CUDA_CHANNELS, f"unknown channel {tag}"
            channel = CUDA_CHANNELS[tag]
            for cc in arch.cc_set:
                assert channel.sm_min <= cc <= channel.sm_max, (
                    f"{backend.engine} x {arch_name} uses {tag} which does not "
                    f"support CC {cc}"
                )


def test_status_derivation() -> None:
    assert (
        BackendSpec(
            engine="x",
            label="x",
            distribution="pkg",
            version="1",
            framework="torch",
            upstream_constraint="torch>=1",
            requires_python=SpecifierSet(">=3.10"),
        ).arch_profiles
        == {}
    )
    # covered via backend arch profiles below
    for backend in BACKENDS.values():
        for bp in backend.arch_profiles.values():
            if not bp.upstream_supported:
                assert bp.status == "experimental"
            elif bp.mliport_verified:
                assert bp.status == "verified"
            else:
                assert bp.status == "needs_smoke_test"


def test_verified_platforms_match_real_hardware() -> None:
    """No architecture-wide claim may substitute for exact workload evidence."""

    for backend in BACKENDS.values():
        for bp in backend.arch_profiles.values():
            assert not bp.mliport_verified
            assert all(
                record["evidence_id"] is None for record in bp.workload_support.values()
            )
