"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from mliport.gpu_compat import arch_supports_device


def test_exact_match_supported():
    assert arch_supports_device("sm_61", ["sm_50", "sm_60", "sm_61", "sm_70"])


def test_sm60_covers_sm61():
    """sm_60 kernels are binary-compatible with sm_61 (same major, minor <=)."""
    assert arch_supports_device("sm_61", ["sm_50", "sm_60", "sm_70", "sm_75"])


def test_sm50_does_not_cover_sm61():
    """sm_50 is a different major (5 vs 6) — NOT binary-compatible with sm_61."""
    assert not arch_supports_device("sm_61", ["sm_50", "sm_70"])


def test_torch28_wheel_rejects_sm61():
    """Stock torch 2.8 arch list has no Pascal kernel -> sm_61 unsupported."""
    arch_list = [
        "sm_70",
        "sm_75",
        "sm_80",
        "sm_86",
        "sm_90",
        "sm_100",
        "sm_120",
        "compute_120",
    ]
    assert not arch_supports_device("sm_61", arch_list)


def test_higher_minor_not_compatible():
    """sm_61 kernels do NOT run on sm_60 (compiled minor must be <= device minor)."""
    assert not arch_supports_device("sm_60", ["sm_61", "sm_70"])


def test_different_major_not_compatible():
    assert not arch_supports_device("sm_61", ["sm_70", "sm_75", "sm_80"])


def test_empty_arch_list_is_permissive():
    assert arch_supports_device("sm_61", [])
    assert arch_supports_device("sm_61", None)


def test_compute_only_entries_ignored():
    """compute_* tokens are PTX, not SASS; only sm_* counts for compatibility."""
    assert not arch_supports_device("sm_61", ["compute_70", "compute_80"])


def test_modern_gpu_supported():
    assert arch_supports_device("sm_89", ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89"])
    assert arch_supports_device("sm_86", ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89"])
