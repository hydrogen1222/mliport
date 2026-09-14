"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from mliport.jobs import JobManager, JobStatus
from mliport.tui.analysis_screen import AnalysisScreen
from mliport.tui.app import MliportApp
from mliport.tui.config_screen import ConfigScreen
from mliport.tui.jobs_screen import JobDetailScreen, JobsScreen
from mliport.tui.run_screen import RunScreen


@pytest.mark.asyncio()
async def test_analysis_screen_progressive_disclosure() -> None:
    app = MliportApp()
    async with app.run_test(size=(100, 100)) as pilot:
        screen = AnalysisScreen()
        await app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#analysis-mobile-input").display is False
        assert screen.query_one("#analysis-charge-input").display is False
        screen.query_one("#analysis-task-select").value = "transport"
        await pilot.pause()
        assert screen.query_one("#analysis-mobile-input").display is True
        assert screen.query_one("#analysis-drift-select").display is True
        assert screen.query_one("#analysis-charge-input").display is True
        assert screen.query_one("#analysis-lag-step-input").display is True
        assert screen.query_one("#analysis-lag-stop-input").display is True
        assert screen.query_one("#analysis-temperature-input").display is True
        assert screen.query_one("#analysis-collective-switch").display is True
        assert screen.query_one("#analysis-frame-interval-input").display is True
        assert screen.query_one("#analysis-rdf-center-input").display is False

        screen.query_one("#analysis-task-select").value = "rdf"
        await pilot.pause()
        assert screen.query_one("#analysis-lag-step-input").display is False
        assert screen.query_one("#analysis-temperature-input").display is False
        assert screen.query_one("#analysis-frame-interval-input").display is False

        screen.query_one("#analysis-task-select").value = "electrolyte"
        await pilot.pause()
        assert screen.query_one("#analysis-sites-input").display is True
        assert screen.query_one("#analysis-charge-input").display is False
        assert screen.query_one("#analysis-lag-step-input").display is False
        assert screen.query_one("#analysis-frame-interval-input").display is False


@pytest.mark.asyncio()
async def test_analysis_transport_parameters_include_lag_and_source_overrides() -> None:
    app = MliportApp()
    async with app.run_test(size=(100, 100)) as pilot:
        screen = AnalysisScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#analysis-task-select").value = "transport"
        await pilot.pause()

        screen.query_one("#analysis-drift-select").value = "nonmobile"
        screen.query_one("#analysis-axes-input").value = "xyz"
        screen.query_one("#analysis-charge-input").value = "1"
        screen.query_one("#analysis-fit-input").value = "40"
        screen.query_one("#analysis-lag-step-input").value = "2"
        screen.query_one("#analysis-lag-stop-input").value = "200"
        screen.query_one("#analysis-temperature-input").value = "700"
        screen.query_one("#analysis-collective-switch").value = True
        screen.query_one("#analysis-positions-convention-select").value = "wrapped"
        screen.query_one("#analysis-frame-interval-input").value = "10"
        parameters = screen._parameters("transport")

    assert parameters == {
        "mobile_species": "Li",
        "drift_reference": "nonmobile",
        "dimensions": "xyz",
        "ionic_charge_e": 1.0,
        "fit_start_ps": 40.0,
        "lag_step_ps": 2.0,
        "lag_stop_ps": 200.0,
        "temperature_K": 700.0,
        "collective_conductivity": True,
        "positions_convention": "wrapped",
        "frame_interval_fs": 10.0,
        "parser_memory_limit_gib": 4.0,
        "random_seed": 0,
    }


@pytest.mark.asyncio()
async def test_analysis_transport_blank_overrides_are_omitted() -> None:
    app = MliportApp()
    async with app.run_test(size=(100, 100)) as pilot:
        screen = AnalysisScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#analysis-task-select").value = "transport"
        await pilot.pause()
        screen.query_one("#analysis-charge-input").value = "1"
        screen.query_one("#analysis-fit-input").value = "40"
        parameters = screen._parameters("transport")

    assert "positions_convention" not in parameters
    assert "frame_interval_fs" not in parameters
    assert "temperature_K" not in parameters


@pytest.mark.asyncio()
async def test_analysis_transport_lag_pair_validation() -> None:
    app = MliportApp()
    async with app.run_test(size=(100, 100)) as pilot:
        screen = AnalysisScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#analysis-task-select").value = "transport"
        await pilot.pause()
        screen.query_one("#analysis-charge-input").value = "1"
        screen.query_one("#analysis-fit-input").value = "40"
        screen.query_one("#analysis-lag-step-input").value = "2"
        with pytest.raises(
            ValueError,
            match="Transport lag step and lag stop must be provided together",
        ):
            screen._parameters("transport")


@pytest.mark.asyncio()
async def test_cpu_tui_run_screen_queues_without_mount_error(tmp_path: Path) -> None:
    """Regression: the TUI CPU path must mount and enqueue cleanly."""
    from ase import Atoms
    from ase.io import write

    structure = tmp_path / "structure.xyz"
    model = tmp_path / "model.pt"
    write(structure, Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]))
    model.write_text("placeholder")
    app = MliportApp()
    app.config.update(
        {
            "calc_type": "sp",
            "structure_file": str(structure),
            "model_file": str(model),
            # explicit backend: the TUI has no implicit UMA default
            "model_type": "uma",
            "task": "omat",
            "device": "cpu",
            "output_dir": str(tmp_path / "results"),
        }
    )
    async with app.run_test(size=(100, 60)) as pilot:
        screen = RunScreen()
        screen._job_manager = JobManager(jobs_dir=tmp_path / "jobs")
        await app.push_screen(screen)
        await pilot.pause()
        assert screen._job_id is not None
        job = screen._job_manager.get_job(screen._job_id)
        assert job is not None
        assert job["status"] == "pending"
        assert job["device"] == "cpu"
        assert "Failed to queue" not in str(screen.status.render())


@pytest.mark.asyncio()
async def test_md_ensemble_and_options_persisted(tmp_path: Path) -> None:
    """MD ensemble, timestep and save interval are saved to app config."""
    structure = tmp_path / "structure.cif"
    model = tmp_path / "model.pt"
    structure.write_text("")
    model.write_text("")

    app = MliportApp()
    app.update_config("calc_type", "md")
    # explicit backend: the TUI has no implicit UMA default
    app.update_config("model_type", "uma")
    app.update_config("task", "omat")

    async with app.run_test(size=(80, 80)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        config_screen.query_one("#structure-input").value = str(structure)
        config_screen.query_one("#model-input").value = str(model)
        config_screen.query_one("#device-input").value = "cuda:1"
        config_screen.query_one("#inference-mode-select").value = "turbo"
        config_screen.query_one("#torch-threads-input").value = "6"
        config_screen.query_one("#activation-checkpointing-select").value = "off"
        config_screen.query_one("#timestep-input").value = "2.5"
        config_screen.query_one("#save-interval-input").value = "25"
        config_screen.query_one("#friction-input").value = "0.004"
        config_screen.query_one("#pre-relax-steps-input").value = "12"
        config_screen.query_one("#pre-relax-fmax-input").value = "0.08"
        config_screen.query_one("#seed-input").value = "42"
        config_screen.query_one("#velocity-policy-select").value = "initialize"
        config_screen.query_one("#com-policy-select").value = "constraint"
        config_screen.query_one("#fmax-abort-input").value = "15"
        config_screen.query_one("#nve").value = True
        await pilot.pause()

        # Avoid actually mounting RunScreen in this unit test.
        app.push_screen = Mock()
        config_screen._save_and_run()

    assert app.get_config("ensemble") == "NVE"
    assert app.get_config("device") == "cuda:1"
    assert app.get_config("inference_mode") == "turbo"
    assert app.get_config("torch_num_threads") == 6
    assert app.get_config("activation_checkpointing") is False
    assert app.get_config("timestep") == 2.5
    assert app.get_config("save_interval") == 25
    # NVE hides and ignores thermostat-specific fields.
    assert app.get_config("friction") == 0.001
    assert app.get_config("pre_relax_steps") == 12
    assert app.get_config("pre_relax_fmax") == 0.08
    assert app.get_config("seed") == 42
    assert app.get_config("velocity_policy") == "initialize"
    assert app.get_config("com_policy") == "constraint"
    assert app.get_config("fmax_abort") == 15.0
    assert isinstance(app.get_config("run_started_at"), float)


@pytest.mark.asyncio()
async def test_opt_values_loaded_from_config() -> None:
    """ConfigScreen loads previously saved opt values when re-composed."""
    app = MliportApp()
    app.update_config("calc_type", "opt")
    app.update_config("fmax", 0.02)
    app.update_config("max_steps", 100)
    app.update_config("optimizer", "BFGS")
    app.update_config("cell_opt", True)

    async with app.run_test(size=(80, 80)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        assert config_screen.query_one("#fmax-input").value == "0.02"
        assert config_screen.query_one("#max-steps-input").value == "100"
        assert config_screen.query_one("#optimizer-select").value == "BFGS"
        assert config_screen.query_one("#cell-opt").value is True


@pytest.mark.asyncio()
async def test_md_values_loaded_from_config() -> None:
    """ConfigScreen loads previously saved md values when re-composed."""
    app = MliportApp()
    app.update_config("calc_type", "md")
    app.update_config("ensemble", "NVE")
    app.update_config("temperature", 400.0)
    app.update_config("timestep", 2.0)
    app.update_config("steps", 2000)
    app.update_config("save_interval", 5)
    app.update_config("pre_relax", False)
    app.update_config("com_policy", "none")

    async with app.run_test(size=(80, 80)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        assert config_screen.query_one("#nve").value is True
        assert config_screen.query_one("#temp-input").value == "400.0"
        assert config_screen.query_one("#timestep-input").value == "2.0"
        assert config_screen.query_one("#steps-input").value == "2000"
        assert config_screen.query_one("#save-interval-input").value == "5"
        assert config_screen.query_one("#pre-relax").value is False
        assert config_screen.query_one("#com-policy-select").value == "none"


@pytest.mark.asyncio()
async def test_backend_resource_controls_follow_selected_engine() -> None:
    """TUI exposes resource controls without forwarding invalid cross-engine options."""
    app = MliportApp()
    # start from an explicit UMA selection, then switch to DPA below
    app.update_config("model_type", "uma")
    app.update_config("task", "omat")

    async with app.run_test(size=(100, 100)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        assert config_screen.query_one("#device-input").value == "cpu"
        assert config_screen.query_one("#inference-mode-select").disabled is False
        assert (
            config_screen.query_one("#activation-checkpointing-select").disabled
            is False
        )
        assert config_screen.query_one("#dtype-select").disabled is True
        assert config_screen.query_one("#head-input").disabled is True
        assert config_screen.query_one("#inference-mode-select").display is True
        assert config_screen.query_one("#head-input").display is False
        assert config_screen.query_one("#dtype-select").display is False

        config_screen.query_one("#model-type-select").value = "dpa"
        await pilot.pause()
        assert config_screen.query_one("#inference-mode-select").disabled is True
        assert (
            config_screen.query_one("#activation-checkpointing-select").disabled is True
        )
        assert config_screen.query_one("#dtype-select").disabled is True
        assert config_screen.query_one("#head-input").disabled is False
        assert config_screen.query_one("#torch-threads-input").disabled is False
        assert config_screen.query_one("#inference-mode-select").display is False
        assert config_screen.query_one("#head-input").display is True
        assert config_screen.query_one("#dtype-select").display is False

        config_screen.query_one("#model-type-select").value = "grace"
        await pilot.pause()
        assert config_screen.query_one("#head-input").disabled is True
        assert config_screen.query_one("#head-input").display is False
        assert config_screen.query_one("#torch-threads-input").disabled is False
        assert config_screen.query_one("#grace-neighbor-cache-row").display is True
        assert config_screen.query_one("#grace-neighbor-skin-input").disabled is False


@pytest.mark.asyncio()
async def test_grace_cache_options_reach_background_command() -> None:
    app = MliportApp()
    app.config.update(
        {
            "calc_type": "md",
            "structure_file": "/tmp/structure.vasp",
            "model_file": "/tmp/grace-model",
            "model_type": "grace",
            "task": "bulk",
            "device": "cuda:0",
            "output_dir": "/tmp/out",
            "neighbor_cache": False,
            "neighbor_skin": 2.25,
            "steps": 5,
        }
    )

    async with app.run_test(size=(80, 40)):
        screen = RunScreen()
        screen._job_id = "test-grace-cache"
        command = screen._build_command()

    assert "--no-neighbor-cache" in command
    assert command[command.index("--neighbor-skin") + 1] == "2.25"


@pytest.mark.asyncio()
async def test_molecular_charge_and_spin_controls_are_task_aware(
    tmp_path: Path,
) -> None:
    structure = tmp_path / "molecule.xyz"
    model = tmp_path / "uma.pt"
    structure.write_text("")
    model.write_text("")
    app = MliportApp()
    # "omol" is a UMA task family: select the backend explicitly
    app.update_config("model_type", "uma")
    app.update_config("task", "omat")

    async with app.run_test(size=(100, 100)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        assert config_screen.query_one("#charge-input").display is False
        assert config_screen.query_one("#spin-input").display is False

        config_screen.query_one("#task-select").value = "omol"
        await pilot.pause()
        assert config_screen.query_one("#charge-input").display is True
        assert config_screen.query_one("#spin-input").display is True
        assert "Multiplicity" in str(config_screen.query_one("#spin-label").render())

        config_screen.query_one("#structure-input").value = str(structure)
        config_screen.query_one("#model-input").value = str(model)
        config_screen.query_one("#charge-input").value = "-1"
        config_screen.query_one("#spin-input").value = "2"
        app.push_screen = Mock()
        config_screen._save_and_run()

    assert app.get_config("charge") == -1
    assert app.get_config("spin") == 2


@pytest.mark.asyncio()
async def test_run_command_contains_tui_resource_and_md_options() -> None:
    """Every visible TUI option must reach the background CLI command."""
    app = MliportApp()
    app.config.update(
        {
            "calc_type": "md",
            "structure_file": "/tmp/structure.vasp",
            "model_file": "/tmp/model.pt",
            "model_type": "uma",
            "task": "omat",
            "device": "cuda:1",
            "output_dir": "/tmp/out",
            "inference_mode": "turbo",
            "torch_num_threads": 6,
            "activation_checkpointing": False,
            "ensemble": "NVE",
            "temperature": 500.0,
            "timestep": 0.5,
            "steps": 10,
            "friction": 0.002,
            "save_interval": 2,
            "pre_relax": False,
            "pre_relax_steps": 7,
            "pre_relax_fmax": 0.07,
            "seed": 42,
            "velocity_policy": "initialize",
            "com_policy": "auto",
            "fmax_abort": 12.0,
        }
    )

    async with app.run_test(size=(80, 40)):
        screen = RunScreen()
        screen._job_id = "test-job"
        command = screen._build_command()

    for expected in (
        "--device",
        "cuda:1",
        "--inference-mode",
        "turbo",
        "--cpu-threads",
        "6",
        "--no-activation-checkpointing",
        "--velocity-policy",
        "initialize",
        "--com-policy",
        "auto",
        "--fmax-abort",
        "12.0",
        "--seed",
        "42",
        "--no-pre-relax",
    ):
        assert expected in command
    assert "--thermostat" not in command
    assert "--friction" not in command


@pytest.mark.asyncio()
async def test_run_command_contains_only_active_nhc_options() -> None:
    app = MliportApp()
    app.config.update(
        {
            "calc_type": "md",
            "structure_file": "/tmp/structure.vasp",
            "model_file": "/tmp/model.pt",
            "model_type": "mace",
            "task": "bulk",
            "device": "cpu",
            "output_dir": "/tmp/out",
            "ensemble": "NVT",
            "thermostat": "NHC",
            "com_policy": "none",
            "steps": 5,
            "friction": 0.003,
            "bussi_tau": 700.0,
            "nhc_tdamp": 120.0,
            "nhc_tchain": 4,
            "nhc_tloop": 2,
        }
    )

    async with app.run_test(size=(80, 40)):
        screen = RunScreen()
        screen._job_id = "test-nhc"
        command = screen._build_command()

    assert command[command.index("--thermostat") + 1] == "NHC"
    assert command[command.index("--nhc-tdamp") + 1] == "120.0"
    assert command[command.index("--nhc-tchain") + 1] == "4"
    assert command[command.index("--nhc-tloop") + 1] == "2"
    assert command[command.index("--com-policy") + 1] == "none"
    assert "--friction" not in command
    assert "--bussi-tau" not in command


@pytest.mark.asyncio()
async def test_md_thermostat_controls_are_dynamic_and_task_label_is_accurate() -> None:
    app = MliportApp()
    app.update_config("calc_type", "md")
    # "Task Type" is the UMA label; select UMA explicitly (then switch to DPA)
    app.update_config("model_type", "uma")
    app.update_config("task", "omat")

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()

        assert "Task Type" in str(screen.query_one("#task-type-label").render())
        assert screen.query_one("#friction-input").display is True
        assert screen.query_one("#bussi-tau-input").display is False
        assert screen.query_one("#nhc-tdamp-input").display is False

        screen.query_one("#thermostat-select").value = "BUSSI"
        await pilot.pause()
        assert screen.query_one("#friction-input").display is False
        assert screen.query_one("#bussi-tau-input").display is True

        screen.query_one("#thermostat-select").value = "NHC"
        await pilot.pause()
        assert screen.query_one("#bussi-tau-input").display is False
        assert screen.query_one("#nhc-tdamp-input").display is True
        assert screen.query_one("#nhc-tchain-input").display is True
        assert screen.query_one("#nhc-tloop-input").display is True

        screen.query_one("#nve").value = True
        await pilot.pause()
        assert screen.query_one("#thermostat-select").display is False
        assert screen.query_one("#nhc-tdamp-input").display is False

        screen.query_one("#model-type-select").value = "dpa"
        await pilot.pause()
        assert "System Type" in str(screen.query_one("#task-type-label").render())
        assert "TensorFlow" in str(screen.query_one("#engine-options-note").render())


@pytest.mark.asyncio()
async def test_md_invalid_numeric_inputs_block_submission(tmp_path: Path) -> None:
    structure = tmp_path / "structure.cif"
    model = tmp_path / "model.pt"
    structure.write_text("")
    model.write_text("")
    app = MliportApp()
    app.update_config("calc_type", "md")

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#structure-input").value = str(structure)
        screen.query_one("#model-input").value = str(model)
        screen.notify = Mock()
        app.push_screen = Mock()

        cases = [
            ("LANGEVIN", "#temp-input", "bad"),
            ("LANGEVIN", "#timestep-input", "0"),
            ("LANGEVIN", "#steps-input", "1.5"),
            ("LANGEVIN", "#save-interval-input", "0"),
            ("LANGEVIN", "#friction-input", "0"),
            ("BUSSI", "#bussi-tau-input", "bad"),
            ("BUSSI", "#temp-input", "0"),
            ("NHC", "#nhc-tdamp-input", "0"),
            ("NHC", "#temp-input", "0"),
            ("NHC", "#nhc-tchain-input", "0"),
            ("NHC", "#nhc-tloop-input", "bad"),
            ("LANGEVIN", "#pre-relax-steps-input", "bad"),
            ("LANGEVIN", "#pre-relax-fmax-input", "nan"),
            ("LANGEVIN", "#fmax-abort-input", "bad"),
        ]
        defaults = {
            "#temp-input": "300",
            "#timestep-input": "1",
            "#steps-input": "5",
            "#save-interval-input": "1",
            "#friction-input": "0.001",
            "#bussi-tau-input": "1000",
            "#nhc-tdamp-input": "100",
            "#nhc-tchain-input": "3",
            "#nhc-tloop-input": "1",
            "#pre-relax-steps-input": "50",
            "#pre-relax-fmax-input": "0.1",
            "#fmax-abort-input": "20",
        }
        for thermostat, selector, invalid in cases:
            for input_selector, value in defaults.items():
                screen.query_one(input_selector).value = value
            screen.query_one("#thermostat-select").value = thermostat
            screen.query_one("#com-policy-select").value = (
                "none" if thermostat == "NHC" else "auto"
            )
            screen.query_one(selector).value = invalid
            screen._save_and_run()
            app.push_screen.assert_not_called()
            assert screen.notify.called
            screen.notify.reset_mock()


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    ("ensemble", "thermostat", "com_policy"),
    [
        ("NVT", "NHC", "auto"),
        ("NVT", "BUSSI", "initialize_only"),
        ("NVE", "LANGEVIN", "initialize_only"),
    ],
)
async def test_md_unsupported_com_policy_blocks_tui_submission(
    tmp_path: Path,
    ensemble: str,
    thermostat: str,
    com_policy: str,
) -> None:
    structure = tmp_path / "structure.cif"
    model = tmp_path / "model.pt"
    structure.write_text("")
    model.write_text("")
    app = MliportApp()
    app.update_config("calc_type", "md")

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#structure-input").value = str(structure)
        screen.query_one("#model-input").value = str(model)
        screen.query_one("#thermostat-select").value = thermostat
        screen.query_one("#com-policy-select").value = com_policy
        screen.query_one(f"#{ensemble.lower()}").value = True
        await pilot.pause()
        screen.notify = Mock()
        app.push_screen = Mock()

        screen._save_and_run()

        app.push_screen.assert_not_called()
        screen.notify.assert_called_once()


@pytest.mark.asyncio()
async def test_run_command_contains_tui_charge_and_spin() -> None:
    app = MliportApp()
    app.config.update(
        {
            "calc_type": "sp",
            "structure_file": "/tmp/molecule.xyz",
            "model_file": "/tmp/uma.pt",
            "model_type": "uma",
            "task": "omol",
            "device": "cpu",
            "output_dir": "/tmp/out",
            "charge": -1,
            "spin": 2,
        }
    )

    async with app.run_test(size=(80, 40)):
        screen = RunScreen()
        screen._job_id = "test-molecule"
        command = screen._build_command()

    assert command[command.index("--charge") + 1] == "-1"
    assert command[command.index("--spin") + 1] == "2"


@pytest.mark.asyncio()
async def test_md_switch_row_does_not_clip_or_overlap() -> None:
    """The pre-relax label and switch have a full row and no ghost control."""
    app = MliportApp()
    app.update_config("calc_type", "md")

    async with app.run_test(size=(80, 40)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        pre_relax = config_screen.query_one("#pre-relax")
        row = pre_relax.parent
        label = row.query_one("Label")

        assert row.has_class("switch-row")
        assert row.region.height >= pre_relax.region.height
        assert row.region.y <= label.region.y < row.region.bottom
        assert len(config_screen.query("#detach-switch")) == 0

        before = pre_relax.value
        pre_relax.focus()
        await pilot.press("space")
        await pilot.pause()
        assert pre_relax.value is not before


@pytest.mark.asyncio()
async def test_job_detail_screen_displays_log() -> None:
    """JobDetailScreen writes supplied log text into the Log widget."""
    app = MliportApp()

    async with app.run_test(size=(80, 80)) as pilot:
        detail = JobDetailScreen("test-job", "line one\nline two")
        await app.push_screen(detail)
        await pilot.pause()

        log = detail.query_one("#job-detail-log")
        assert len(log.lines) == 2


@pytest.mark.asyncio()
async def test_jobs_screen_handles_empty_store_and_remounts(tmp_path: Path) -> None:
    """An empty jobs screen mounts, refreshes, and remounts without crashing."""
    app = MliportApp()
    jobs_screen = JobsScreen()
    jobs_screen._job_manager = JobManager(jobs_dir=tmp_path)

    async with app.run_test(size=(80, 40)) as pilot:
        await app.push_screen(jobs_screen)
        await pilot.pause()

        table = jobs_screen.query_one("#jobs-table")
        assert len(table.columns) == 7
        assert table.row_count == 0
        assert jobs_screen.query_one("#pause-job-btn")
        assert jobs_screen.query_one("#resume-job-btn")
        assert jobs_screen._jobs_refresh_timer is not None

        app.pop_screen()
        await pilot.pause()
        assert jobs_screen._jobs_refresh_timer is None

        await app.push_screen(jobs_screen)
        await pilot.pause()
        assert len(table.columns) == 7
        assert table.row_count == 0


@pytest.mark.asyncio()
async def test_jobs_screen_uses_job_id_as_row_key(tmp_path: Path) -> None:
    """Selecting a table row resolves to the persisted job ID."""
    manager = JobManager(jobs_dir=tmp_path)
    job_id = str(uuid.uuid4())
    manager._write_job_state(
        job_id,
        status=JobStatus.RUNNING,
        calc_type="sp",
        structure="/tmp/POSCAR",
        formula="H2",
        natoms=2,
        pid=123,
        device="cpu",
        display_name="job-123",
    )

    app = MliportApp()
    jobs_screen = JobsScreen()
    jobs_screen._job_manager = manager
    async with app.run_test(size=(80, 40)) as pilot:
        await app.push_screen(jobs_screen)
        await pilot.pause()

        table = jobs_screen.query_one("#jobs-table")
        assert [row_key.value for row_key in table.rows] == [job_id]
        assert table.get_row_at(0)[:2] == ["job-123", job_id]


@pytest.mark.asyncio()
async def test_jobs_screen_refresh_preserves_selected_job(tmp_path: Path) -> None:
    """Auto-refresh must not move the cursor back to the first running job."""
    manager = JobManager(jobs_dir=tmp_path)
    records = (
        (str(uuid.uuid4()), "a-running", JobStatus.RUNNING, 123),
        (str(uuid.uuid4()), "b-pending", JobStatus.PENDING, 0),
    )
    for job_id, display_name, status, pid in records:
        manager._write_job_state(
            job_id,
            status=status,
            calc_type="sp",
            structure="/tmp/POSCAR",
            formula="H2",
            natoms=2,
            pid=pid,
            device="cpu",
            display_name=display_name,
        )

    app = MliportApp()
    jobs_screen = JobsScreen()
    jobs_screen._job_manager = manager
    async with app.run_test(size=(80, 40)) as pilot:
        await app.push_screen(jobs_screen)
        await pilot.pause()

        table = jobs_screen.query_one("#jobs-table")
        pending_id = records[1][0]
        table.move_cursor(row=table.get_row_index(pending_id))
        assert table.get_row_at(table.cursor_row)[0] == "b-pending"
        jobs_screen._refresh_table()
        assert table.get_row_at(table.cursor_row)[0] == "b-pending"


def test_run_screen_unmount_keeps_background_job_running() -> None:
    """Leaving RunScreen stops UI refresh without cancelling the job."""
    screen = RunScreen()
    timer = Mock()
    manager = Mock()
    screen._refresh_timer = timer
    screen._job_manager = manager

    screen.on_unmount()

    timer.stop.assert_called_once()
    assert screen._refresh_timer is None
    manager.kill_job.assert_not_called()


def test_jobs_screen_unmount_cancels_timer() -> None:
    """JobsScreen cancels its refresh timer when the screen is unmounted."""
    screen = JobsScreen()
    timer = Mock()
    screen._jobs_refresh_timer = timer

    screen.on_unmount()

    timer.stop.assert_called_once()
    assert screen._jobs_refresh_timer is None


# ---------------------------------------------------------------------------
# NEB (R3-D): TUI consumes the shared queue helper; no TUI-private defaults.
# ---------------------------------------------------------------------------


def _write_periodic_endpoints(
    tmp_path: Path,
    cell: list[list[float]],
    initial_positions: list[list[float]],
    final_positions: list[list[float]],
    symbols: str = "H2",
) -> tuple[Path, Path]:
    """Write periodic endpoints as extxyz so cell/PBC survive the round trip."""
    from ase import Atoms
    from ase.io import write

    initial = tmp_path / "initial_periodic.extxyz"
    final = tmp_path / "final_periodic.extxyz"
    write(
        initial,
        Atoms(symbols, positions=initial_positions, cell=cell, pbc=True),
        format="extxyz",
    )
    write(
        final,
        Atoms(symbols, positions=final_positions, cell=cell, pbc=True),
        format="extxyz",
    )
    return initial, final


def _write_neb_endpoints(tmp_path: Path) -> tuple[Path, Path, Path]:
    from ase import Atoms
    from ase.io import write

    initial = tmp_path / "initial.xyz"
    final = tmp_path / "final.xyz"
    model = tmp_path / "model.pt"
    write(initial, Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))
    write(final, Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.20, 0.0, 0.0]]))
    model.write_text("placeholder")
    return initial, final, model


@pytest.mark.asyncio()
async def test_main_screen_lists_neb_entry() -> None:
    from textual.widgets import ListView

    app = MliportApp()
    async with app.run_test(size=(80, 24)):
        items = [
            item.id
            for item in app.screen.query_one("#calc-type-list", ListView).children
        ]
    assert "neb" in items


@pytest.mark.asyncio()
async def test_neb_run_command_matches_shared_queue_helper(tmp_path: Path) -> None:
    """TUI-built NEB argv must equal the CLI/queue/API command for the same
    configuration (both go through build_mliport_command)."""
    from mliport.queue import build_mliport_command

    initial, final, model = _write_neb_endpoints(tmp_path)
    output_dir = str(tmp_path / "out")
    app = MliportApp()
    app.config.update(
        {
            "calc_type": "neb",
            "structure_file": str(initial),
            "neb_final": str(final),
            "neb_atom_map": [1, 0],
            "neb_image_shifts": [[0, 0, 0]],
            "neb_resume": None,
            "model_file": str(model),
            "model_type": "mace",
            "task": "bulk",
            "device": "cpu",
            "output_dir": output_dir,
            "n_intermediate_images": 5,
            "climb": True,
            "neb_interpolation": "idpp",
            "idpp_mic": False,
            "path_convention": "mic",
            "neb_spring": 0.2,
            "fmax": 0.05,
            "max_steps": 200,
            "endpoint_policy": "relax",
            "allow_unvalidated_neb": True,
            "inference_mode": "default",
            "fmax_abort": 15,
        }
    )

    async with app.run_test(size=(80, 40)):
        screen = RunScreen()
        screen._job_id = "test-neb"
        command = screen._build_command()

    expected = build_mliport_command(
        calc_type="neb",
        structure=str(initial),
        model=str(model),
        model_type="mace",
        task="bulk",
        device="cpu",
        output_dir=output_dir,
        job_name="test-neb",
        options={
            "n_intermediate_images": 5,
            "climb": True,
            "neb_interpolation": "idpp",
            "idpp_mic": False,
            "path_convention": "mic",
            "neb_spring": 0.2,
            "fmax": 0.05,
            "max_steps": 200,
            "endpoint_policy": "relax",
            "allow_unvalidated_neb": True,
            "inference_mode": "default",
            "default_dtype": "float64",
            "fmax_abort": 15,
        },
        initial=str(initial),
        final=str(final),
        atom_map=[1, 0],
        image_shifts=[[0, 0, 0]],
    )
    assert command == expected

    # Every visible NEB control reaches the argv with its CLI flag.
    assert command[command.index("--initial") + 1] == str(initial)
    assert command[command.index("--final") + 1] == str(final)
    assert command[command.index("--images") + 1] == "5"
    assert "--climb" in command
    assert command[command.index("--interpolation") + 1] == "idpp"
    assert "--no-idpp-mic" in command
    assert command[command.index("--spring") + 1] == "0.2"
    assert command[command.index("--fmax") + 1] == "0.05"
    assert command[command.index("--max-steps") + 1] == "200"
    assert command[command.index("--endpoint-policy") + 1] == "relax"
    assert "--allow-unvalidated-neb" in command
    assert command[command.index("--atom-map") + 1] == "1,0"
    assert command[command.index("--image-shifts") + 1] == "0,0,0"
    # Blank/None options must not hard-code TUI defaults.
    assert "--pre-fmax" not in command
    assert "--min-distance" not in command


@pytest.mark.asyncio()
async def test_neb_resume_command_locks_original_directory(tmp_path: Path) -> None:
    resume_path = str(tmp_path / "run" / "checkpoints" / "latest")
    app = MliportApp()
    app.config.update(
        {
            "calc_type": "neb",
            "structure_file": None,
            "neb_final": None,
            "neb_atom_map": None,
            "neb_image_shifts": None,
            "neb_resume": resume_path,
            "model_file": None,
            "model_type": "mace",
            "device": "cpu",
            "output_dir": str(tmp_path / "out"),
        }
    )

    async with app.run_test(size=(80, 40)):
        screen = RunScreen()
        screen._job_id = "test-neb-resume"
        command = screen._build_command()

    assert command[command.index("--resume") + 1] == resume_path
    assert "--initial" not in command
    assert "--final" not in command
    assert "--model" not in command
    # Resume infers and locks the original run directory.
    assert "--output" not in command
    assert "--name" not in command


@pytest.mark.asyncio()
async def test_neb_run_screen_queues_without_mount_error(tmp_path: Path) -> None:
    initial, final, model = _write_neb_endpoints(tmp_path)
    app = MliportApp()
    app.config.update(
        {
            "calc_type": "neb",
            "structure_file": str(initial),
            "neb_final": str(final),
            "neb_atom_map": None,
            "neb_image_shifts": None,
            "neb_resume": None,
            "model_file": str(model),
            "model_type": "uma",
            "task": "omat",
            "device": "cpu",
            "output_dir": str(tmp_path / "results"),
        }
    )
    async with app.run_test(size=(100, 60)) as pilot:
        screen = RunScreen()
        screen._job_manager = JobManager(jobs_dir=tmp_path / "jobs")
        await app.push_screen(screen)
        await pilot.pause()
        assert screen._job_id is not None
        job = screen._job_manager.get_job(screen._job_id)
        assert job is not None
        assert job["status"] == "pending"
        assert "Failed to queue" not in str(screen.status.render())


@pytest.mark.asyncio()
async def test_neb_config_screen_sections_and_idpp_toggle(tmp_path: Path) -> None:
    initial, final, _model = _write_neb_endpoints(tmp_path)
    app = MliportApp()
    app.update_config("calc_type", "neb")

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()

        # Plan R3-D sections: endpoints/mapping, path, endpoint policy,
        # convergence/safety.
        for selector in (
            "#final-structure-input",
            "#neb-resume-input",
            "#atom-map-input",
            "#image-shifts-input",
            "#neb-preview",
            "#neb-interpolation-select",
            "#neb-climb-switch",
            "#neb-path-convention-select",
            "#endpoint-policy-select",
            "#endpoint_fmax-input",
            "#fmax-input",
            "#max_steps-input",
            "#checkpoint_interval-input",
            "#fmax_abort-input",
            "#allow-unvalidated-neb-switch",
            "#neb-validation-note",
        ):
            screen.query_one(selector)

        # Linear interpolation hides the IDPP-only controls.
        assert screen.query_one("#idpp_fmax-input").display is False
        screen.query_one("#neb-interpolation-select").value = "idpp"
        await pilot.pause()
        assert screen.query_one("#idpp_fmax-input").display is True
        assert screen.query_one("#idpp-mic-switch").display is True

        # Endpoint preview summarizes composition and sampled displacement.
        screen.query_one("#structure-input").value = str(initial)
        screen.query_one("#final-structure-input").value = str(final)
        screen._update_neb_preview()
        preview = str(screen.query_one("#neb-preview").render())
        assert "Initial: H2" in preview
        assert "Final:   H2" in preview
        assert "Preview displacement" in preview

        # Resume mode ignores endpoint inputs.
        screen.query_one("#neb-resume-input").value = str(tmp_path / "run")
        screen._update_neb_option_states()
        preview = str(screen.query_one("#neb-preview").render())
        assert "Resume mode" in preview


@pytest.mark.asyncio()
async def test_neb_invalid_atom_map_blocks_submission(tmp_path: Path) -> None:
    initial, final, model = _write_neb_endpoints(tmp_path)
    app = MliportApp()
    app.update_config("calc_type", "neb")
    app.update_config("model_type", "mace")

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#structure-input").value = str(initial)
        screen.query_one("#model-input").value = str(model)
        screen.query_one("#final-structure-input").value = str(final)
        screen.notify = Mock()
        app.push_screen = Mock()

        for invalid in ("abc", "0,0", "0,5", "0"):
            screen.query_one("#atom-map-input").value = invalid
            screen._save_and_run()
            app.push_screen.assert_not_called()
            assert screen.notify.called
            screen.notify.reset_mock()

        # A valid permutation passes validation and launches the run screen.
        screen.query_one("#atom-map-input").value = "1,0"
        screen._save_and_run()
        app.push_screen.assert_called_once()


@pytest.mark.asyncio()
async def test_neb_image_shifts_require_unwrapped_convention(tmp_path: Path) -> None:
    initial, final, model = _write_neb_endpoints(tmp_path)
    app = MliportApp()
    app.update_config("calc_type", "neb")
    app.update_config("model_type", "mace")

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#structure-input").value = str(initial)
        screen.query_one("#model-input").value = str(model)
        screen.query_one("#final-structure-input").value = str(final)
        screen.query_one("#neb-path-convention-select").value = "mic"
        screen.query_one("#image-shifts-input").value = "0,0,0; 0,0,0"
        screen.notify = Mock()
        app.push_screen = Mock()

        screen._save_and_run()
        app.push_screen.assert_not_called()
        assert screen.notify.called
        assert "unwrapped" in str(screen.notify.call_args)

        # Switching to unwrapped lets the same shifts through.
        screen.query_one("#neb-path-convention-select").value = "unwrapped"
        screen._save_and_run()
        app.push_screen.assert_called_once()
        assert app.get_config("neb_image_shifts") == [[0, 0, 0], [0, 0, 0]]


def _render_neb_preview(screen: ConfigScreen) -> str:
    screen._update_neb_preview()
    return str(screen.query_one("#neb-preview").render())


@pytest.mark.asyncio()
async def test_neb_preview_uses_mic_for_periodic_crossing(tmp_path: Path) -> None:
    """RC-05: 9.9 → 0.1 in a 10 Å cell previews as the +0.2 Å MIC hop, and the
    production band preparation produces the same intended displacement."""
    import numpy as np
    from ase.io import read

    from mliport.neb import NEBOptions, prepare_band

    app = MliportApp()
    app.update_config("calc_type", "neb")
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    initial, final = _write_periodic_endpoints(
        tmp_path,
        cell,
        [[9.9, 5.0, 5.0], [5.0, 5.0, 5.0]],
        [[0.1, 5.0, 5.0], [5.0, 5.0, 5.0]],
    )

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#structure-input").value = str(initial)
        screen.query_one("#final-structure-input").value = str(final)
        screen.query_one("#neb-path-convention-select").value = "mic"
        screen.query_one("#n_intermediate_images-input").value = "4"
        preview = _render_neb_preview(screen)
        assert "0.200 Å" in preview
        assert "-9.800" not in preview
        assert "Per-segment estimate (5 segments): 0.040 Å" in preview
        assert "Winding: none" in preview

        # Production parity: the band's final image is the MIC-lifted endpoint.
        initial_atoms = read(initial)
        final_atoms = read(final)
        band = prepare_band(
            initial_atoms, final_atoms, NEBOptions(n_intermediate_images=4)
        )
        assert np.allclose(
            band.final.positions,
            initial_atoms.positions + np.array([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )


@pytest.mark.asyncio()
async def test_neb_preview_reports_explicit_winding(tmp_path: Path) -> None:
    """RC-05: an explicit +1 lattice shift previews as a nonzero winding hop
    even though the wrapped endpoints look almost identical."""
    app = MliportApp()
    app.update_config("calc_type", "neb")
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    positions = [[0.1, 5.0, 5.0], [0.4, 5.0, 5.0]]
    initial, final = _write_periodic_endpoints(tmp_path, cell, positions, positions)

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#structure-input").value = str(initial)
        screen.query_one("#final-structure-input").value = str(final)
        screen.query_one("#neb-path-convention-select").value = "unwrapped"
        screen.query_one("#image-shifts-input").value = "1,0,0; 0,0,0"
        preview = _render_neb_preview(screen)
        assert "max 10.000 Å at atom 0 (H)" in preview
        assert "Winding: present" in preview


@pytest.mark.asyncio()
async def test_neb_preview_uses_general_triclinic_mic(tmp_path: Path) -> None:
    """RC-05: the preview resolves the general triclinic MIC instead of a
    per-component fractional wrap."""
    import numpy as np
    from ase.geometry import find_mic

    app = MliportApp()
    app.update_config("calc_type", "neb")
    cell = [[10.0, 0.0, 0.0], [5.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    initial, final = _write_periodic_endpoints(
        tmp_path,
        cell,
        [[0.0, 5.0, 5.0]],
        [[8.9, 5.0, 5.0]],
        symbols="H",
    )

    # The raw displacement is (8.9, 0, 0). The general MIC shortens it to
    # ~1.1 Å along -a, while a naive per-component fractional wrap would give
    # a much longer vector (~5.8 Å).
    mic, _ = find_mic(np.array([[8.9, 0.0, 0.0]]), cell=np.array(cell), pbc=True)
    mic_norm = float(np.linalg.norm(mic[0]))
    assert abs(mic_norm - 1.1) < 1.0e-6

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#structure-input").value = str(initial)
        screen.query_one("#final-structure-input").value = str(final)
        screen.query_one("#neb-path-convention-select").value = "mic"
        preview = _render_neb_preview(screen)
        assert f"max {mic_norm:.3f} Å at atom 0 (H)" in preview
        assert "5.8" not in preview


@pytest.mark.asyncio()
async def test_neb_preview_applies_atom_map_before_image_shifts(
    tmp_path: Path,
) -> None:
    """RC-05: after atom mapping, image shifts are indexed in the initial
    identity order, so the shifted atom is the mapped one."""
    app = MliportApp()
    app.update_config("calc_type", "neb")
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    initial, final = _write_periodic_endpoints(
        tmp_path,
        cell,
        [[9.0, 5.0, 5.0], [2.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
        [[2.5, 5.0, 5.0], [5.5, 5.0, 5.0], [0.5, 5.0, 5.0]],
        symbols="H3",
    )

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#structure-input").value = str(initial)
        screen.query_one("#final-structure-input").value = str(final)
        screen.query_one("#neb-path-convention-select").value = "unwrapped"
        # atom_map: initial atom 0 -> final index 2 (x: 9.0 -> 0.5), and the
        # shift row for initial atom 0 lifts the hop to +1.5 Å instead of the
        # raw -8.5 Å wrapped difference.
        screen.query_one("#atom-map-input").value = "2,0,1"
        screen.query_one("#image-shifts-input").value = "1,0,0; 0,0,0; 0,0,0"
        preview = _render_neb_preview(screen)
        assert "max 1.500 Å at atom 0 (H)" in preview
        assert "Winding: present" in preview


@pytest.mark.asyncio()
async def test_neb_run_screen_live_status_reads_engine_artifacts(
    tmp_path: Path,
) -> None:
    """The run screen reports NEB progress from engine artifacts only (no
    TUI-side re-evaluation, no runner stdout parsing)."""
    import json

    initial, final, model = _write_neb_endpoints(tmp_path)
    output_dir = tmp_path / "results"
    run_dir = output_dir / "job-neb"
    checkpoint_dir = run_dir / "checkpoints" / "step_000003"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema": "mliport.neb-checkpoint/1",
                "stage": "neb",
                "stage_step": 12,
                "resume_fingerprint": {"neb_options": {"max_steps": 1000}},
            }
        )
    )
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "schema": "mliport.neb-artifacts/1",
                "status": "running",
                "artifacts": {
                    "checkpoint": {"path": str(checkpoint_dir), "kind": "neb_band"},
                    "result": None,
                },
            }
        )
    )
    app = MliportApp()
    app.config.update(
        {
            "calc_type": "neb",
            "structure_file": str(initial),
            "neb_final": str(final),
            "neb_resume": None,
            "model_file": str(model),
            "model_type": "uma",
            "device": "cpu",
            "output_dir": str(output_dir),
        }
    )
    async with app.run_test(size=(100, 60)) as pilot:
        screen = RunScreen()
        screen._job_manager = JobManager(jobs_dir=tmp_path / "jobs")
        await app.push_screen(screen)
        await pilot.pause()
        assert screen._job_id is not None
        screen._job_id = "job-neb"

        screen._refresh_neb_status("running")
        running_text = str(screen.status.render())
        assert "stage neb" in running_text
        assert "step 12" in running_text
        # The budget belongs to THIS stage; a combined total percentage would
        # be invented (review R09).
        assert "CI budget 1000 steps" in running_text

        # Pre-CI stage reports its own budget, not the CI budget.
        (checkpoint_dir / "checkpoint.json").write_text(
            json.dumps(
                {
                    "schema": "mliport.neb-checkpoint/1",
                    "stage": "neb_pre",
                    "stage_step": 5,
                    "resume_fingerprint": {
                        "neb_options": {"pre_max_steps": 300, "max_steps": 1000}
                    },
                }
            )
        )
        screen._refresh_neb_status("running")
        pre_text = str(screen.status.render())
        assert "stage neb_pre" in pre_text
        assert "pre-CI budget 300 steps" in pre_text
        (checkpoint_dir / "checkpoint.json").write_text(
            json.dumps(
                {
                    "schema": "mliport.neb-checkpoint/1",
                    "stage": "neb",
                    "stage_step": 12,
                    "resume_fingerprint": {"neb_options": {"max_steps": 1000}},
                }
            )
        )

        # Completed run: converged barrier summary comes from neb_results.json.
        (run_dir / "neb_results.json").write_text(
            json.dumps(
                {
                    "schema": "mliport.neb-results/2",
                    "status": "completed",
                    "converged": True,
                    "barrier_forward_sampled_eV": 0.42,
                    "max_neb_force_eV_A": 0.02,
                    "climbing_image_index": 3,
                }
            )
        )
        (run_dir / "artifacts.json").write_text(
            json.dumps(
                {
                    "schema": "mliport.neb-artifacts/1",
                    "status": "completed",
                    "artifacts": {
                        "checkpoint": {
                            "path": str(checkpoint_dir),
                            "kind": "neb_band",
                        },
                        "result": str(run_dir / "neb_results.json"),
                    },
                }
            )
        )
        screen._refresh_neb_status("done")
        done_text = str(screen.status.render())
        assert "converged" in done_text
        assert "0.42" in done_text
        assert "climbing image 3" in done_text

        # Not-converged completion points the user at resume.
        (run_dir / "neb_results.json").write_text(
            json.dumps(
                {
                    "schema": "mliport.neb-results/2",
                    "status": "not_converged",
                    "converged": False,
                }
            )
        )
        (run_dir / "artifacts.json").write_text(
            json.dumps(
                {
                    "schema": "mliport.neb-artifacts/1",
                    "status": "not_converged",
                    "artifacts": {
                        "checkpoint": {
                            "path": str(checkpoint_dir),
                            "kind": "neb_band",
                        },
                        "result": str(run_dir / "neb_results.json"),
                    },
                }
            )
        )
        screen._refresh_neb_status("done")
        assert "NOT converged" in str(screen.status.render())
        assert "resume" in str(screen.status.render()).lower()


@pytest.mark.asyncio()
async def test_config_save_requires_explicit_backend(tmp_path: Path) -> None:
    """TUI config must not silently default an unselected backend to UMA."""
    structure = tmp_path / "s.cif"
    structure.write_text("dummy", encoding="utf-8")
    model = tmp_path / "m.pt"
    model.write_text("dummy", encoding="utf-8")

    app = MliportApp()
    app.update_config("calc_type", "sp")
    app.update_config("model_type", None)
    async with app.run_test(size=(80, 80)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#structure-input").value = str(structure)
        screen.query_one("#model-input").value = str(model)
        screen.query_one("#output-input").value = str(tmp_path / "out")
        notifications: list[str] = []
        screen.notify = lambda message, **kwargs: notifications.append(str(message))
        screen._save_and_run()

    assert notifications, "saving without a backend must notify the user"
    assert "backend" in notifications[0].lower()
    assert not app.get_config("model_type")


@pytest.mark.asyncio()
async def test_tui_jobs_screen_is_empty_in_new_state(
    monkeypatch, tmp_path: Path
) -> None:
    """STATE-T3: the TUI jobs screen must be empty for a brand-new state."""
    monkeypatch.setenv("MLIPORT_JOBS_DIR", str(tmp_path / "jobs"))
    app = MliportApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen("jobs")
        await pilot.pause()
        table = app.screen.query_one("#jobs-table")
        assert len(table.rows) == 0
