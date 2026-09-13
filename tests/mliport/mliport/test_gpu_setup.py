"""Tests for mliport.gpu_setup (nvidia-smi parsing and recommendations)."""

from __future__ import annotations

from unittest import mock

import pytest

from mliport.gpu_setup import (
    GpuInfo,
    cc_arch_name,
    detect_gpus,
    engine_install_commands,
    recommend_torch,
)


@pytest.fixture(autouse=True)
def _unmasked_probe(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_detect_gpus_parses_plain_csv() -> None:
    fake = mock.Mock(
        returncode=0,
        stdout="NVIDIA GeForce RTX 4090,8.9,535.104.05,24564,0,GPU-test\n",
    )
    with mock.patch("mliport.install.hardware.subprocess.run", return_value=fake):
        gpus = detect_gpus()
    assert gpus is not None and len(gpus) == 1
    g = gpus[0]
    assert g.name == "NVIDIA GeForce RTX 4090"
    assert (g.cc_major, g.cc_minor) == (8, 9)
    assert g.vram_mib == 24564
    assert g.driver_version == "535.104.05"


def test_detect_gpus_handles_quoted_commas_in_name() -> None:
    fake = mock.Mock(
        returncode=0,
        stdout='"Tesla V100-SXM2-16GB, Special",7.0,535.104.05,16384,0,GPU-test\n',
    )
    with mock.patch("mliport.install.hardware.subprocess.run", return_value=fake):
        gpus = detect_gpus()
    assert gpus is not None and len(gpus) == 1
    assert gpus[0].name == "Tesla V100-SXM2-16GB, Special"
    assert (gpus[0].cc_major, gpus[0].cc_minor) == (7, 0)


def test_detect_gpus_skips_malformed_rows() -> None:
    fake = mock.Mock(
        returncode=0,
        stdout="bad row without enough fields\n"
        "NVIDIA A100,8.0,535.104.05,40960,0,GPU-test\n",
    )
    with mock.patch("mliport.install.hardware.subprocess.run", return_value=fake):
        gpus = detect_gpus()
    assert gpus is not None and len(gpus) == 1
    assert gpus[0].name == "NVIDIA A100"


def test_detect_gpus_missing_nvidia_smi() -> None:
    with mock.patch(
        "mliport.install.hardware.subprocess.run",
        side_effect=FileNotFoundError("nvidia-smi"),
    ):
        assert detect_gpus() is None


def test_detect_gpus_nonzero_returncode() -> None:
    fake = mock.Mock(returncode=1, stdout="")
    with mock.patch("mliport.install.hardware.subprocess.run", return_value=fake):
        assert detect_gpus() is None


def test_detect_gpus_empty_output() -> None:
    fake = mock.Mock(returncode=0, stdout="")
    with mock.patch("mliport.install.hardware.subprocess.run", return_value=fake):
        assert detect_gpus() is None


def test_cc_arch_name_families() -> None:
    assert cc_arch_name(3, 7) == "Kepler"
    assert cc_arch_name(5, 2) == "Maxwell"
    assert cc_arch_name(6, 1) == "Pascal"
    assert cc_arch_name(7, 0) == "Volta"
    assert cc_arch_name(7, 5) == "Turing"
    assert cc_arch_name(8, 0) == "Ampere"
    assert cc_arch_name(8, 9) == "Ada Lovelace"
    assert cc_arch_name(9, 0) == "Hopper"
    assert cc_arch_name(10, 0) == "Blackwell"
    assert cc_arch_name(12, 0) == "Blackwell"


def test_recommend_torch_kepler_unsupported() -> None:
    rec = recommend_torch(3, 7)
    assert rec.supported is False


def test_recommend_torch_pascal_uses_cu126() -> None:
    """Pascal (sm_61) → torch 2.8.0+cu126 via legacy channel."""
    rec = recommend_torch(6, 1)
    assert rec.supported is True
    assert "2.8.0" in rec.version
    assert "cu126" in rec.version


def test_recommend_torch_pascal_dpa_preserves_deepmd_abi_pin() -> None:
    rec = recommend_torch(6, 1, engine="dpa")
    assert rec.supported is True
    assert rec.version == "2.10.0+cu126"
    assert "torch==2.10.0+cu126" in "\n".join(rec.install_commands)


def test_recommend_torch_blackwell_uses_modern() -> None:
    """Blackwell (sm_100) → torch 2.8.0+cu128 via modern channel."""
    rec = recommend_torch(10, 0)
    assert rec.supported is True
    assert "2.8.0" in rec.version
    assert "cu128" in rec.version


def _gpu(cc_major: int, cc_minor: int) -> GpuInfo:
    return GpuInfo(
        name="fake",
        cc_major=cc_major,
        cc_minor=cc_minor,
        driver_version="1.0",
        vram_mib=8192,
    )


def test_engine_install_commands_pascal_mace_uses_cu126() -> None:
    """Pascal (sm_61) MACE → torch 2.8.0+cu126."""
    cmds = engine_install_commands([_gpu(6, 1)], "mace")
    joined = "\n".join(cmds)
    assert "torch==2.8.0" in joined
    assert "cu126" in joined
    assert "e3nn==0.4.4" in joined


def test_engine_install_commands_pascal_dpa_uses_cu126() -> None:
    """Pascal DPA → torch 2.10.0+cu126."""
    cmds = engine_install_commands([_gpu(6, 1)], "dpa")
    joined = "\n".join(cmds)
    assert "torch==2.10.0+cu126" in joined
    assert "cu126" in joined


def test_engine_install_commands_volta_dpa_uses_cu126() -> None:
    """Volta (sm_70) DPA → torch 2.10.0+cu126."""
    cmds = engine_install_commands([_gpu(7, 0)], "dpa")
    joined = "\n".join(cmds)
    assert "torch==2.10.0" in joined
    assert "cu126" in joined


def test_engine_install_commands_ada_mace_uses_modern() -> None:
    """Ada (sm_89) MACE → torch 2.8.0+cu128."""
    cmds = engine_install_commands([_gpu(8, 9)], "mace")
    joined = "\n".join(cmds)
    assert "torch==2.8.0" in joined
    assert "cu128" in joined


def test_engine_install_commands_blackwell_uses_cu128() -> None:
    cmds = engine_install_commands([_gpu(10, 0)], "mace")
    joined = "\n".join(cmds)
    assert "torch==2.8.0" in joined
    assert "cu128" in joined


def test_engine_install_commands_cpu_only_uses_cpu_wheels() -> None:
    mace = "\n".join(engine_install_commands(None, "mace"))
    assert "https://download.pytorch.org/whl/cpu" in mace
    grace = "\n".join(engine_install_commands(None, "grace"))
    assert "tensorflow==" in grace
    assert "[and-cuda]" not in grace


def test_engine_install_commands_uma_explicit() -> None:
    """UMA is installed explicitly (torch + fairchem-core + editable mliport),
    never via `uv sync --frozen`."""
    cmds = engine_install_commands(None, "uma")
    joined = "\n".join(cmds)
    assert "uv sync" not in joined
    assert "fairchem-core==2.21.0" in joined
    assert "-e ./mliport" in joined


def test_engine_install_commands_maxwell_experimental() -> None:
    """Maxwell (sm_52) should still give commands but mark experimental."""
    cmds = engine_install_commands([_gpu(5, 2)], "uma")
    joined = "\n".join(cmds)
    # Should still produce explicit UMA install (torch + fairchem-core).
    assert "fairchem-core==2.21.0" in joined
    assert "torch==2.8.0" in joined
