"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_minimal_config(self):
        from mlipx.engine import EngineConfig

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
        )
        assert config.calc_type == "sp"
        assert config.options == {}
        assert config.detach is False

    def test_config_with_options(self):
        from mlipx.engine import EngineConfig

        config = EngineConfig(
            calc_type="opt",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
            options={"fmax": 0.02, "cell_opt": True},
        )
        assert config.options["fmax"] == 0.02
        assert config.options["cell_opt"] is True

    def test_config_detach(self):
        from mlipx.engine import EngineConfig

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
            detach=True,
        )
        assert config.detach is True

    def test_from_resolved_routes_md_safety_threshold(self):
        from mlipx.config.resolver import resolve_config
        from mlipx.engine import EngineConfig

        resolved = resolve_config(calc_type="md")
        config = EngineConfig.from_resolved(resolved)
        assert config.run_options["fmax_abort"] == 20.0

    def test_from_resolved_routes_neb_safety_threshold(self):
        from mlipx.config.resolver import resolve_config
        from mlipx.engine import EngineConfig

        resolved = resolve_config(calc_type="neb")
        config = EngineConfig.from_resolved(resolved)
        assert config.run_options["fmax_abort"] == 20.0

    @pytest.mark.parametrize("model_type", ["uma", "mace", "dpa", "grace"])
    def test_from_resolved_routes_backend_thread_count(self, model_type):
        from mlipx.config.resolver import resolve_config
        from mlipx.engine import EngineConfig

        resolved = resolve_config(
            calc_type="sp",
            cli={
                "model_type": model_type,
                "model_path": "model.pt",
                "torch_num_threads": 3,
            },
        )
        config = EngineConfig.from_resolved(resolved)
        assert config.torch_num_threads == 3

    def test_from_resolved_routes_uma_activation_checkpointing_override(self):
        from mlipx.config.resolver import resolve_config
        from mlipx.engine import EngineConfig

        resolved = resolve_config(
            calc_type="sp",
            cli={
                "model_type": "uma",
                "model_path": "model.pt",
                "activation_checkpointing": False,
            },
        )
        config = EngineConfig.from_resolved(resolved)
        assert config.activation_checkpointing is False


class TestCalculationEngineSetup:
    """Tests for CalculationEngine construction and config validation."""

    def test_engine_from_config_sp(self):
        from mlipx.engine import CalculationEngine, EngineConfig

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
        )
        with patch.object(Path, "exists", return_value=True):
            engine = CalculationEngine.from_config(config)
            assert engine.config.calc_type == "sp"

    def test_invalid_calc_type_raises(self):
        from mlipx.engine import CalculationEngine, EngineConfig

        config = EngineConfig(
            calc_type="invalid",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
        )
        with pytest.raises(ValueError, match="Unknown calc_type"):
            CalculationEngine.from_config(config)

    def test_engine_config_options_validation_unknown_key_warns(self):
        """Unknown options keys should not raise, just warn."""
        from mlipx.engine import CalculationEngine, EngineConfig

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("model.pt"),
            task="omat",
            device="cpu",
            inference_mode="default",
            output_dir=Path("./results"),
            options={"made_up_key": 42},
        )
        # Should not raise
        with patch.object(Path, "exists", return_value=True):
            engine = CalculationEngine.from_config(config)
            assert engine is not None

    @pytest.mark.parametrize("model_type", ["uma", "mace", "dpa"])
    def test_pytorch_thread_count_is_applied_before_calculator_creation(
        self, model_type
    ):
        from mlipx.engine import CalculationEngine, EngineConfig

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("model.pt"),
            model_type=model_type,
            task="omat" if model_type == "uma" else "bulk",
            torch_num_threads=2,
        )
        engine = CalculationEngine.from_config(config)

        with (
            patch("torch.set_num_threads") as set_num_threads,
            patch(
                "mlipx.calculators.factory.CalculatorFactory.create",
                return_value=object(),
            ),
        ):
            engine._create_calculator()

        set_num_threads.assert_called_once_with(2)

    def test_grace_thread_count_reaches_tensorflow_calculator(self):
        from mlipx.engine import CalculationEngine, EngineConfig

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("grace-model"),
            model_type="grace",
            task="bulk",
            torch_num_threads=2,
        )
        engine = CalculationEngine.from_config(config)

        with patch(
            "mlipx.calculators.factory.CalculatorFactory.create",
            return_value=object(),
        ) as create:
            engine._create_calculator()

        assert create.call_args.kwargs["cpu_threads"] == 2

    def test_engine_forwards_all_thermostat_options_to_md_runner(self, tmp_path):
        from mlipx.engine import CalculationEngine, EngineConfig

        config = EngineConfig(
            calc_type="md",
            model_path=Path("model.pt"),
            output_dir=tmp_path,
            run_options={
                "thermostat": "NHC",
                "friction": 0.002,
                "bussi_tau": 800.0,
                "nhc_tdamp": 120.0,
                "nhc_tchain": 4,
                "nhc_tloop": 2,
                "com_policy": "none",
                "pre_relax": False,
            },
        )
        runner = CalculationEngine.from_config(config)._create_runner(Mock())

        assert runner.thermostat == "nhc"
        assert runner.friction > 0
        assert runner.bussi_tau > 0
        assert runner.nhc_tdamp > 0
        assert runner.nhc_tchain == 4
        assert runner.nhc_tloop == 2
        assert runner.com_policy == "none"

    @pytest.mark.parametrize("model_type", ["uma", "mace", "dpa", "grace"])
    def test_legacy_md_defaults_remain_nvt_langevin_for_every_backend(self, model_type):
        from ase import units

        from mlipx.config.resolver import resolve_config
        from mlipx.engine import CalculationEngine, EngineConfig

        resolved = resolve_config(
            calc_type="md",
            cli={"model_type": model_type, "model_path": "model"},
        )
        runner = CalculationEngine.from_config(
            EngineConfig.from_resolved(resolved)
        )._create_runner(Mock())

        assert runner.ensemble == "nvt"
        assert runner.thermostat == "langevin"
        assert runner.friction * units.fs == pytest.approx(0.001)


def test_engine_creates_live_log_and_tail_hint(tmp_path):
    from mlipx.engine import CalculationEngine, EngineConfig

    config = EngineConfig(
        calc_type="sp",
        model_path=Path("model.pt"),
        output_dir=tmp_path,
        job_name="job-01",
    )
    engine = CalculationEngine.from_config(config)
    runner = Mock()
    runner.execute.return_value = {"energy": -1.0}
    messages = []

    with (
        patch.object(engine, "_create_calculator", return_value=object()),
        patch.object(engine, "_create_runner", return_value=runner),
    ):
        engine.run(
            atoms=object(),
            log_fn=lambda message, level: messages.append(message),
            started_at=123.0,
        )

    expected_log = (tmp_path / "job-01" / "run.log").resolve()
    assert engine.run_log_path == expected_log
    assert expected_log.exists()
    assert "tail -f" in expected_log.read_text(encoding="utf-8")
    assert any("Follow live output:" in message for message in messages)
    runner.execute.assert_called_once()
    assert runner.execute.call_args.kwargs["started_at"] == 123.0


def test_live_run_logger_batches_step_records_without_losing_them(tmp_path):
    from mlipx.logger import LiveRunLogger

    callbacks = []
    log_path = tmp_path / "run.log"
    with LiveRunLogger(
        log_path,
        callback=lambda message, level: callbacks.append((message, level)),
    ) as logger:
        logger.write_buffered("Step 0/2")
        logger.write_buffered("Step 1/2")
        assert callbacks == []
        assert logger._buffered_records == 2

        # A normal status message also flushes all preceding step records.
        logger("MD simulation completed")
        assert callbacks == [("MD simulation completed", "info")]
        assert logger._buffered_records == 0
        text = log_path.read_text(encoding="utf-8")
        assert text.index("Step 0/2") < text.index("Step 1/2")
        assert text.index("Step 1/2") < text.index("MD simulation completed")

        logger.write_buffered("Step 2/2")

    assert "Step 2/2" in log_path.read_text(encoding="utf-8")


def test_python_api_forwards_all_thermostat_options(monkeypatch):
    from ase import Atoms

    from mlipx import api

    captured = {}

    def fake_resolve(calc_type, model_path, cli, *args, **kwargs):
        captured.update(cli)
        return object()

    class DummyEngine:
        def run(self, atoms, **kwargs):
            return {"ok": True}

    monkeypatch.setattr(api, "_api_resolve", fake_resolve)
    monkeypatch.setattr(
        api.CalculationEngine,
        "from_config",
        lambda config: DummyEngine(),
    )

    result = api.run_md(
        Atoms("Ar", positions=[[0.0, 0.0, 0.0]]),
        "model.pt",
        thermostat="NHC",
        friction=0.002,
        bussi_tau=800.0,
        nhc_tdamp=120.0,
        nhc_tchain=4,
        nhc_tloop=2,
        com_policy="none",
        verbose=False,
    )

    assert result == {"ok": True}
    assert (
        captured.items()
        >= {
            "thermostat": "NHC",
            "friction": 0.002,
            "bussi_tau": 800.0,
            "nhc_tdamp": 120.0,
            "nhc_tchain": 4,
            "nhc_tloop": 2,
            "com_policy": "none",
        }.items()
    )


def test_python_neb_api_forwards_typed_options_to_public_engine(tmp_path, monkeypatch):
    from ase import Atoms

    from mlipx import api

    captured = {}
    resolved = Mock(calc_type="neb", model_path=str(tmp_path / "model.pt"))

    def fake_resolve(calc_type, model_path, cli, *args, **kwargs):
        captured["calc_type"] = calc_type
        captured["cli"] = dict(cli)
        return object(), resolved

    class DummyEngine:
        def run_neb(self, actual_resolved, **kwargs):
            captured["resolved"] = actual_resolved
            captured["run"] = kwargs
            return {"status": "completed", "converged": True}

    monkeypatch.setattr(api, "_api_resolve_full", fake_resolve)
    monkeypatch.setattr(
        api.CalculationEngine, "from_config", lambda config: DummyEngine()
    )
    initial = Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    final = initial.copy()
    final.positions[0, 0] = 1.0

    result = api.run_neb(
        initial,
        final,
        "model.pt",
        output_dir=tmp_path / "run",
        n_intermediate_images=3,
        spring=0.2,
        climb=True,
        verbose=False,
    )

    assert result["converged"] is True
    assert captured["calc_type"] == "neb"
    assert captured["cli"]["n_intermediate_images"] == 3
    assert captured["cli"]["neb_spring"] == pytest.approx(0.2)
    assert captured["cli"]["climb"] is True
    assert captured["resolved"] is resolved


class TestMaceDtypeDefaults:
    """MACE defaults to accuracy-first float64 for every calculation type."""

    @pytest.mark.parametrize("calc_type", ["sp", "opt", "md", "neb"])
    def test_mace_default_dtype_is_float64_for_all_calc_types(self, calc_type):
        from mlipx.engine import CalculationEngine, EngineConfig

        config = EngineConfig(
            calc_type=calc_type,
            model_path=Path("mace.model"),
            model_type="mace",
            task="bulk",
            device="cpu",
            output_dir=Path("./results"),
        )
        engine = CalculationEngine.from_config(config)
        captured: dict = {}

        def fake_create(model_type, model_path, device, task, strict=False, **kwargs):
            captured.update(kwargs)
            return object()

        with patch(
            "mlipx.calculators.factory.CalculatorFactory.create",
            side_effect=fake_create,
        ):
            engine._create_calculator()
        assert captured.get("default_dtype") == "float64"

    def test_mace_explicit_dtype_float64_not_overridden(self):
        from mlipx.engine import CalculationEngine, EngineConfig

        config = EngineConfig(
            calc_type="sp",
            model_path=Path("mace.model"),
            model_type="mace",
            task="bulk",
            device="cpu",
            output_dir=Path("./results"),
            calculator_options={"default_dtype": "float64"},
        )
        engine = CalculationEngine.from_config(config)
        captured: dict = {}

        def fake_create(model_type, model_path, device, task, strict=False, **kwargs):
            captured.update(kwargs)
            return object()

        with patch(
            "mlipx.calculators.factory.CalculatorFactory.create",
            side_effect=fake_create,
        ):
            engine._create_calculator()
        assert captured.get("default_dtype") == "float64"
