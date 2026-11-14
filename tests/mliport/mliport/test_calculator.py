"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mliport.calculators.uma import UMACalculator


class TestUMACalculatorValidation:
    """Tests for UMACalculator parameter validation (no model needed)."""

    def test_invalid_task_raises(self):
        with (
            patch.object(Path, "exists", return_value=True),
            pytest.raises(ValueError, match="Invalid task"),
        ):
            UMACalculator(model_path="fake.pt", task="invalid")

    def test_invalid_inference_mode_raises(self):
        with (
            patch.object(Path, "exists", return_value=True),
            pytest.raises(ValueError, match="Invalid inference mode"),
        ):
            UMACalculator(model_path="fake.pt", inference_mode="invalid")

    def test_resource_settings_stored(self):
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(
                model_path="fake.pt",
                torch_num_threads=4,
                activation_checkpointing=False,
            )
            assert calc.torch_num_threads == 4
            assert calc.activation_checkpointing is False

    def test_default_resource_settings(self):
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt")
            assert calc.torch_num_threads is None
            assert calc.activation_checkpointing is None

    def test_task_case_insensitive(self):
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt", task="OMAT")
            assert calc.task == "omat"


class TestGpuIndex:
    """Plan section 3.2: GPU compat checks must probe the *requested* device,
    not always physical device 0."""

    @pytest.mark.parametrize(
        ("device,expected"),
        [
            ("cuda", 0),
            ("cuda:0", 0),
            ("cuda:1", 1),
            ("cuda:7", 7),
            ("gpu", 0),
            ("cpu", 0),
            ("CUDA:3", 3),  # case-insensitive
        ],
    )
    def test_gpu_index_parsing(self, device, expected):
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt", device=device)
            assert calc._gpu_index() == expected

    def test_backend_device_selects_index_then_uses_fairchem_cuda_token(self):
        with (
            patch.object(UMACalculator, "_validate"),
            patch("torch.cuda.set_device") as set_device,
        ):
            calc = UMACalculator(model_path="fake.pt", device="cuda:3")
            assert calc._backend_device() == "cuda"
        set_device.assert_called_once_with(3)

    @pytest.mark.parametrize("device", ["cpu", "cuda", "gpu", "cuda:2"])
    def test_validate_accepts_documented_devices(self, tmp_path, device):
        model = tmp_path / "model.pt"
        model.write_text("x")
        UMACalculator(model, device=device)

    def test_validate_rejects_malformed_device(self, tmp_path):
        model = tmp_path / "model.pt"
        model.write_text("x")
        with pytest.raises(ValueError, match="Invalid device"):
            UMACalculator(model, device="cuda:x")

    def test_check_gpu_compat_probes_requested_index(self, monkeypatch):
        fake_cuda = MagicMock()
        fake_cuda.is_available.return_value = True
        # sm_80 is supported by modern wheels -> no raise.
        fake_cuda.get_device_capability.return_value = (8, 0)
        fake_cuda.get_arch_list.return_value = ["sm_80"]
        fake_cuda.get_device_name.return_value = "FakeGPU"
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt", device="cuda:2")
            calc._check_gpu_compatibility()  # supported -> should not raise
        # Must have probed logical device 2, not 0.
        fake_cuda.get_device_capability.assert_called_once_with(2)
        # get_device_name only fires on the unsupported path (tested below).

    def test_check_gpu_compat_unsupported_uses_requested_index(self, monkeypatch):
        fake_cuda = MagicMock()
        fake_cuda.is_available.return_value = True
        # Pascal sm_61 with no sm_60 kernel in the wheel -> unsupported.
        fake_cuda.get_device_capability.return_value = (6, 1)
        fake_cuda.get_arch_list.return_value = ["sm_80"]
        fake_cuda.get_device_name.return_value = "PascalGPU"
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt", device="cuda:2")
            with pytest.raises(RuntimeError, match="NOT SUPPORTED"):
                calc._check_gpu_compatibility()
        # Both probes must use the requested logical index 2, not 0.
        fake_cuda.get_device_capability.assert_called_once_with(2)
        fake_cuda.get_device_name.assert_called_once_with(2)

    def test_compile_supported_probes_requested_index(self, monkeypatch):
        fake_cuda = MagicMock()
        fake_cuda.is_available.return_value = True
        # Pascal sm_61 -> unsupported by triton (major < 7).
        fake_cuda.get_device_capability.return_value = (6, 1)
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt", device="cuda:3")
            assert calc._compile_supported() is False
        fake_cuda.get_device_capability.assert_called_once_with(3)


class TestNoImplicitAccelerationPresets:
    """UMA Nitro is Ampere+ only, so mliport must never select it.

    Spec #61 (model-support §6.4): fairchem-core 2.24.0 added a Nitro inference
    path that cannot run on Volta. mliport exposes exactly two presets and
    records the effective one; nothing can silently pick a third.
    """

    def test_only_the_two_supported_presets_exist(self):
        modes = set(UMACalculator.VALID_INFERENCE_MODES)
        assert modes == {"default", "turbo"}
        assert "nitro" not in modes

    def test_presets_map_to_upstream_presets_without_nitro(self):
        """default -> batch and turbo -> multigpu reproduce the old numbers."""
        mapping = dict(UMACalculator.INFERENCE_PRESETS)
        assert mapping == {"default": "batch", "turbo": "multigpu"}

    def test_nitro_is_rejected_by_the_calculator(self):
        with (
            patch.object(Path, "exists", return_value=True),
            pytest.raises(ValueError, match="Invalid inference mode"),
        ):
            UMACalculator(model_path="fake.pt", inference_mode="nitro")

    def test_nitro_is_rejected_by_the_configuration_schema(self, tmp_path):
        from mliport.config import resolve_config

        model = tmp_path / "uma.pt"
        model.write_bytes(b"stub")
        with pytest.raises(ValueError):
            resolve_config(
                calc_type="sp",
                cli={
                    "model_type": "uma",
                    "model_path": str(model),
                    "device": "cpu",
                    "inference_mode": "nitro",
                },
            )

    def test_the_effective_preset_is_recorded(self):
        """Provenance must say which preset ran, not just that one was allowed."""
        from mliport.config import resolve_config

        resolved = resolve_config(
            calc_type="sp",
            cli={
                "model_type": "uma",
                "model_path": "uma.pt",
                "device": "cpu",
                "inference_mode": "turbo",
            },
        )
        assert resolved.inference_mode == "turbo"
        assert str(resolved.sources["inference_mode"].value) == "turbo"


class _FakeSettings:
    def __init__(self, **kwargs):
        self.tf32 = kwargs.get("tf32", False)
        self.compile = kwargs.get("compile", False)
        self.nitro = kwargs.get("nitro", False)
        self.activation_checkpointing = True
        self.torch_num_threads = None


def _install_fake_fairchem(monkeypatch, guess):
    """Register stub fairchem modules so load_predictor() runs without weights."""
    modules = {}
    for name in (
        "fairchem",
        "fairchem.core",
        "fairchem.core.units",
        "fairchem.core.units.mlip_unit",
        "fairchem.core.units.mlip_unit.api",
        "fairchem.core.units.mlip_unit.api.inference",
    ):
        modules[name] = types.ModuleType(name)

    recorded = {}

    def load_predict_unit(path=None, device=None, inference_settings=None):
        recorded["settings"] = inference_settings
        return types.SimpleNamespace(model=None, direct_forces=False)

    modules["fairchem.core.units.mlip_unit"].load_predict_unit = load_predict_unit
    modules[
        "fairchem.core.units.mlip_unit.api.inference"
    ].guess_inference_settings = guess
    modules["fairchem.core.units"].mlip_unit = modules["fairchem.core.units.mlip_unit"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return recorded


class TestInferencePresetMapping:
    """The upstream preset that really runs is chosen by mliport, not upstream."""

    def test_default_maps_to_the_upstream_batch_preset(self, monkeypatch):
        recorded = _install_fake_fairchem(
            monkeypatch, lambda name: _FakeSettings() if name == "batch" else None
        )
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt", device="cpu")
            calc.load_predictor()
        assert calc._effective_preset == "batch"
        assert recorded["settings"].nitro is False

    def test_turbo_maps_to_the_upstream_multigpu_preset(self, monkeypatch):
        recorded = _install_fake_fairchem(
            monkeypatch,
            lambda name: _FakeSettings(tf32=True, compile=True)
            if name == "multigpu"
            else None,
        )
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(
                model_path="fake.pt", device="cpu", inference_mode="turbo"
            )
            calc.load_predictor()
        assert calc._effective_preset == "multigpu"
        assert recorded["settings"].tf32 is True
        assert recorded["settings"].nitro is False

    def test_a_preset_that_turns_nitro_on_is_refused(self, monkeypatch):
        _install_fake_fairchem(monkeypatch, lambda name: _FakeSettings(nitro=True))
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt", device="cpu")
            with pytest.raises(RuntimeError, match="Nitro"):
                calc.load_predictor()

    def test_a_missing_upstream_preset_fails_closed(self, monkeypatch):
        def guess(name):
            raise ValueError("inference_settings='batch' is not a preset")

        _install_fake_fairchem(monkeypatch, guess)
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt", device="cpu")
            with pytest.raises(RuntimeError, match="fairchem-core 2.25.0"):
                calc.load_predictor()
