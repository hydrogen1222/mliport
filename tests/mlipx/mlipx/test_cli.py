"""Tests for mlipx CLI (argparse flags, backward compat, config subcommands)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from mlipx.cli import create_parser, main

# ---------------------------------------------------------------------------
# Parser creation
# ---------------------------------------------------------------------------


def test_parser_has_all_subcommands() -> None:
    parser = create_parser()
    # Minimal smoke: parse --help on the root parser
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_transport_parser_accepts_explicit_lag_grid() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "analyze",
            "RUN",
            "transport",
            "--mobile",
            "Li",
            "--charge",
            "1",
            "--fit-start-ps",
            "40",
            "--lag-step-ps",
            "1",
            "--lag-stop-ps",
            "200",
        ]
    )
    assert args.lag_step_ps == 1.0
    assert args.lag_stop_ps == 200.0


def test_analysis_parser_maps_collective_and_gemdat_semantics() -> None:
    parser = create_parser()
    transport = parser.parse_args(
        [
            "analyze",
            "RUN",
            "transport",
            "--mobile",
            "Li",
            "--charge",
            "1",
            "--fit-start-ps",
            "40",
            "--collective-conductivity",
            "--collective-system-particles",
            "4",
            "--jump-diffusion",
        ]
    )
    assert transport.collective_conductivity is True
    assert transport.collective_system_particles == 4
    assert transport.jump_diffusion is True

    electrolyte = parser.parse_args(
        [
            "analyze",
            "RUN",
            "electrolyte",
            "--mobile",
            "Li",
            "--sites",
            "sites.cif",
            "--drift-reference",
            "indices",
            "--drift-indices",
            "3,4",
            "--jump-dimensions",
            "1",
            "--percolation-axes",
            "xz",
        ]
    )
    assert electrolyte.mobile_species == "Li"
    assert electrolyte.sites_path == "sites.cif"
    assert electrolyte.jump_dimensions == 1
    assert electrolyte.percolation_axes == "xz"
    assert electrolyte.drift_reference == "indices"
    assert electrolyte.drift_indices == "3,4"


def test_cmd_analyze_preserves_gemdat_parameter_names(monkeypatch, capsys) -> None:
    from mlipx import cli as cli_module

    captured = {}

    def fake_run_analysis(request):
        captured.update(request.parameters)
        return {"status": "success", "analysis_id": "test", "output_dir": "."}

    monkeypatch.setattr("mlipx.analysis.runner.run_analysis", fake_run_analysis)
    args = create_parser().parse_args(
        [
            "analyze",
            "RUN",
            "electrolyte",
            "--mobile",
            "Li",
            "--sites",
            "sites.cif",
            "--drift-reference",
            "indices",
            "--drift-indices",
            "3,4",
            "--jump-dimensions",
            "2",
            "--percolation-axes",
            "xz",
        ]
    )
    assert cli_module.cmd_analyze(args) == 0
    assert captured["mobile_species"] == "Li"
    assert captured["sites_path"] == "sites.cif"
    assert captured["drift_reference"] == "indices"
    assert captured["drift_indices"] == [3, 4]
    assert captured["jump_dimensions"] == 2
    assert captured["percolation_axes"] == "xz"
    assert "discover_sites_from_density" in captured
    capsys.readouterr()


@pytest.mark.parametrize("queue_command", ["pause", "resume", "status"])
def test_queue_control_commands_parse(queue_command: str) -> None:
    parser = create_parser()
    args = parser.parse_args(["queue", queue_command])
    assert args.command == "queue"
    assert args.queue_command == queue_command


@pytest.mark.parametrize("queue_command", ["pause", "resume"])
def test_queue_job_control_accepts_job_id(queue_command: str) -> None:
    parser = create_parser()
    args = parser.parse_args(["queue", queue_command, "job-2"])
    assert args.queue_command == queue_command
    assert args.job_id == "job-2"


def test_sp_parser_accepts_known_flags() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "sp",
            "struct.xyz",
            "--model",
            "model.pt",
            "--model-type",
            "mace",
            "--task",
            "bulk",
            "--device",
            "cuda:0",
            "--charge",
            "-1",
            "--spin",
            "2",
            "--inference-mode",
            "turbo",
            "--cpu-threads",
            "6",
            "--no-activation-checkpointing",
            "--dtype",
            "float64",
            "--head",
            "some_head",
            "--model-alias",
            "mace_mpa0",
            "--profile",
            "gpu_prod",
            "--output",
            "./out",
        ]
    )
    assert args.command == "sp"
    assert args.structure == "struct.xyz"
    assert args.model == "model.pt"
    assert args.model_type == "mace"
    assert args.task == "bulk"
    assert args.device == "cuda:0"
    assert args.charge == -1
    assert args.spin == 2
    assert args.inference_mode == "turbo"
    assert args.torch_num_threads == 6
    assert args.activation_checkpointing is False
    assert args.default_dtype == "float64"
    assert args.head == "some_head"
    assert args.model_alias == "mace_mpa0"
    assert args.profile == "gpu_prod"
    assert args.output == "./out"


def test_opt_parser_accepts_optimisation_flags() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "opt",
            "struct.xyz",
            "--model",
            "model.pt",
            "--fmax",
            "0.02",
            "--max-steps",
            "100",
            "--optimizer",
            "BFGS",
            "--cell-opt",
            "--no-fix-symmetry",
        ]
    )
    assert args.fmax == 0.02
    assert args.max_steps == 100
    assert args.optimizer == "BFGS"
    assert args.cell_opt is True
    assert args.fix_symmetry is False  # --no-fix-symmetry


def test_opt_fix_symmetry_default_is_none() -> None:
    """BooleanOptionalAction without explicit flag defaults to None."""
    parser = create_parser()
    args = parser.parse_args(["opt", "struct.xyz", "--model", "m.pt"])
    assert args.fix_symmetry is None
    assert args.cell_opt is None


def test_md_parser_accepts_ensemble_flags() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "md",
            "struct.xyz",
            "--model",
            "model.pt",
            "--ensemble",
            "NVE",
            "--temp",
            "500",
            "--timestep",
            "2.0",
            "--steps",
            "5000",
            "--thermostat",
            "NHC",
            "--friction",
            "0.005",
            "--bussi-tau",
            "800",
            "--nhc-tdamp",
            "120",
            "--nhc-tchain",
            "4",
            "--nhc-tloop",
            "2",
            "--save-interval",
            "25",
            "--seed",
            "42",
            "--velocity-policy",
            "initialize",
            "--com-policy",
            "none",
            "--fmax-abort",
            "15",
            "--pre-relax",
            "--no-pre-relax",
        ]
    )
    assert args.ensemble == "NVE"
    assert args.temp == 500.0
    assert args.timestep == 2.0
    assert args.steps == 5000
    assert args.thermostat == "NHC"
    assert args.friction == 0.005
    assert args.bussi_tau == 800.0
    assert args.nhc_tdamp == 120.0
    assert args.nhc_tchain == 4
    assert args.nhc_tloop == 2
    assert args.save_interval == 25
    assert args.seed == 42
    assert args.velocity_policy == "initialize"
    assert args.com_policy == "none"
    assert args.fmax_abort == 15.0


def test_md_thermostat_flags_default_to_resolver() -> None:
    parser = create_parser()
    args = parser.parse_args(["md", "struct.xyz", "--model", "model.pt"])
    assert args.thermostat is None
    assert args.friction is None
    assert args.bussi_tau is None
    assert args.nhc_tdamp is None
    assert args.nhc_tchain is None
    assert args.nhc_tloop is None
    assert args.com_policy is None


def test_neb_parser_keeps_scientific_defaults_in_resolver() -> None:
    args = create_parser().parse_args(
        [
            "neb",
            "--initial",
            "initial.vasp",
            "--final",
            "final.vasp",
            "--model",
            "model.pt",
            "--images",
            "5",
            "--climb",
            "--spring",
            "0.2",
            "--output",
            "results/hop",
        ]
    )

    assert args.command == "neb"
    assert args.n_intermediate_images == 5
    assert args.climb is True
    assert args.neb_spring == pytest.approx(0.2)
    assert args.fmax is None
    assert args.device is None


def test_neb_command_uses_typed_resolver_and_public_engine(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    initial = _write_poscar(tmp_path)
    final = tmp_path / "final.vasp"
    final.write_text(initial.read_text(encoding="utf-8"), encoding="utf-8")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    captured = {}

    class DummyEngine:
        def run_neb(self, resolved, **kwargs):
            captured["resolved"] = resolved
            captured.update(kwargs)
            return {
                "status": "completed",
                "converged": True,
                "run_id": "run-id",
                "latest_checkpoint": "checkpoint",
                "barrier_forward_sampled_eV": 0.0,
            }

    monkeypatch.setattr(
        "mlipx.cli.CalculationEngine.from_config", lambda config: DummyEngine()
    )

    rc = main(
        [
            "neb",
            "--initial",
            str(initial),
            "--final",
            str(final),
            "--model",
            str(model),
            "--images",
            "1",
            "--output",
            str(tmp_path / "neb-run"),
        ]
    )

    assert rc == 0
    assert captured["resolved"].calc_type == "neb"
    assert captured["resolved"].run_options["n_intermediate_images"] == 1
    assert captured["initial"].pbc.all()
    assert captured["final"].pbc.all()
    capsys.readouterr()


def test_grace_memory_and_md_output_flags_parse() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "md",
            "struct.xyz",
            "--model",
            "grace_model",
            "--model-type",
            "grace",
            "--gpu-memory-limit-mb",
            "6144",
            "--no-write-outcar",
            "--no-write-xdatcar",
        ]
    )

    assert args.gpu_memory_limit_mb == 6144
    assert args.write_outcar is False
    assert args.write_xdatcar is False


def test_batch_parser_accepts_pattern_and_workers() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "batch",
            "input_dir/",
            "--model",
            "model.pt",
            "--pattern",
            "*.poscar",
        ]
    )
    assert args.command == "batch"
    assert args.input_dir == "input_dir/"
    assert args.pattern == "*.poscar"


def test_device_accepts_cuda_n() -> None:
    """--device no longer restricted by argparse choices; accepts 'cuda:1'."""
    parser = create_parser()
    args = parser.parse_args(["sp", "s.xyz", "--model", "m.pt", "--device", "cuda:2"])
    assert args.device == "cuda:2"


def test_model_optional_with_model_alias() -> None:
    """--model is not required when --model-alias is provided."""
    parser = create_parser()
    args = parser.parse_args(
        ["sp", "s.xyz", "--model-alias", "mace_mpa0", "--output", "./out"]
    )
    assert args.model is None
    assert args.model_alias == "mace_mpa0"


def test_model_required_without_model_alias() -> None:
    """Without --model or --model-alias, the handler should error. Parser itself
    doesn't enforce this (--model is default=None), but the resolver does."""
    parser = create_parser()
    args = parser.parse_args(["sp", "s.xyz", "--output", "./out"])
    assert args.model is None
    assert args.model_alias is None


# ---------------------------------------------------------------------------
# config subcommand
# ---------------------------------------------------------------------------


def test_config_paths() -> None:
    """config paths should run without error."""
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["config", "paths"])
            assert rc == 0
        finally:
            os.chdir(old)


def test_config_schema() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["config", "schema"])
            assert rc == 0
        finally:
            os.chdir(old)


def test_config_show() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["config", "show"])
            assert rc == 0
        finally:
            os.chdir(old)


def test_config_init_project() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["config", "init", "--project"])
            assert rc == 0
            assert (Path(d) / "settings.ini").exists()
        finally:
            os.chdir(old)


def test_config_init_force() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            main(["config", "init", "--project"])
            rc = main(["config", "init", "--project", "--force"])
            assert rc == 0
        finally:
            os.chdir(old)


def test_generated_settings_validates() -> None:
    """`config init` must never generate a file rejected by `config validate`."""
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            assert main(["config", "init", "--project"]) == 0
            assert main(["config", "validate", "settings.ini"]) == 0
        finally:
            os.chdir(old)


# ---------------------------------------------------------------------------
# CLI backward compat: calc_type → temperature alias mapping
# ---------------------------------------------------------------------------


def test_md_temp_alias() -> None:
    """--temp is the historical flag for temperature."""
    parser = create_parser()
    args = parser.parse_args(["md", "s.xyz", "--model", "m.pt", "--temp", "400"])
    assert args.temp == 400.0


def test_md_prints_banner_once(tmp_path, monkeypatch, capsys) -> None:
    """The dispatcher owns the banner; calculation handlers must not repeat it."""
    structure = _write_poscar(tmp_path)
    model = tmp_path / "model.pt"
    model.touch()

    class DummyEngine:
        def run(self, atoms, **kwargs):
            return {}

    monkeypatch.setattr(
        "mlipx.cli.CalculationEngine.from_config",
        lambda config: DummyEngine(),
    )

    rc = main(
        [
            "md",
            str(structure),
            "--model",
            str(model),
            "--steps",
            "0",
            "--output",
            str(tmp_path / "results"),
        ]
    )

    assert rc == 0
    assert capsys.readouterr().out.count("(mlipx - MLIP eXtended)") == 1


def test_batch_parser_basic_flags() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "batch",
            "dir/",
            "--model",
            "m.pt",
            "--pattern",
            "*.cif",
        ]
    )
    assert args.command == "batch"
    assert args.input_dir == "dir/"
    assert args.pattern == "*.cif"


# ---------------------------------------------------------------------------
# --settings global option
# ---------------------------------------------------------------------------


def test_settings_global_option_accepted() -> None:
    parser = create_parser()
    args = parser.parse_args(
        ["--settings", "/tmp/x.ini", "sp", "s.xyz", "--model", "m.pt"]
    )
    assert args.settings == "/tmp/x.ini"
    assert args.command == "sp"


# ---------------------------------------------------------------------------
# template command
# ---------------------------------------------------------------------------


def test_template_sp() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["template", "sp", "--output", "INCAR.sp"])
            assert rc == 0
            content = Path("INCAR.sp").read_text()
            assert "SP" in content
            assert "CALC_TYPE" in content
        finally:
            os.chdir(old)


# ---------------------------------------------------------------------------
# doctor / setup smoke (these need additional deps; just check exit)
# ---------------------------------------------------------------------------


def test_doctor_parser_accepts_engine_device_model_and_smoke_options() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "doctor",
            "--engine",
            "dpa",
            "--device",
            "cuda:2",
            "--model",
            "model.pt",
            "--task",
            "bulk",
            "--head",
            "Domains_SSE_PBE",
            "--structure",
            "POSCAR",
            "--dtype",
            "float32",
        ]
    )

    assert args.engine == "dpa"
    assert args.device == "cuda:2"
    assert args.model == "model.pt"
    assert args.task == "bulk"
    assert args.head == "Domains_SSE_PBE"
    assert args.structure == "POSCAR"
    assert args.default_dtype == "float32"


def test_doctor_parser_rejects_invalid_device() -> None:
    parser = create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["doctor", "--device", "cuda:bad"])
    assert exc.value.code == 2


def test_doctor_exits_gracefully() -> None:
    # Doctor should not crash even if optional deps are missing.
    rc = main(["doctor"])
    # Can be 0 (all ok) or non-zero (some missing), but not a crash
    assert isinstance(rc, int)


# ---------------------------------------------------------------------------
# `mlipx run` INCAR calc-type dispatch (regression)
# ---------------------------------------------------------------------------


def _write_poscar(directory: Path) -> Path:
    poscar = directory / "POSCAR"
    poscar.write_text(
        "He\n"
        "1.0\n"
        "5.0 0.0 0.0\n"
        "0.0 5.0 0.0\n"
        "0.0 0.0 5.0\n"
        "He\n"
        "1\n"
        "Direct\n"
        "0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    return poscar


def test_run_incar_calc_type_batch_fails_with_guidance(capsys) -> None:
    """Regression: CALC_TYPE=BATCH passed validation, loaded the structure,
    then crashed with 'Unknown calc_type: batch' deep inside the engine.
    It must now fail fast with guidance toward the batch subcommand."""
    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        _write_poscar(dpath)
        (dpath / "INCAR.mlipx").write_text(
            "CALC_TYPE = BATCH\nMODEL_PATH = model.pt\nTASK = omat\n",
            encoding="utf-8",
        )
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["run", "--incar", "INCAR.mlipx", "--structure", "POSCAR"])
            out = capsys.readouterr().out
            assert rc == 1
            assert "cannot be executed by 'mlipx run'" in out
            assert "mlipx batch" in out
        finally:
            os.chdir(old)


def test_run_incar_calc_type_phonon_rejected_at_validation(capsys) -> None:
    """Regression: CALC_TYPE=PHONON passed INCAR validation but the engine
    rejects it; it must now be rejected by the validator itself."""
    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        _write_poscar(dpath)
        (dpath / "INCAR.mlipx").write_text(
            "CALC_TYPE = PHONON\nMODEL_PATH = model.pt\n", encoding="utf-8"
        )
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["run", "--incar", "INCAR.mlipx", "--structure", "POSCAR"])
            out = capsys.readouterr().out
            assert rc == 1
            assert "Invalid CALC_TYPE" in out
        finally:
            os.chdir(old)


def test_run_incar_calculation_neb_uses_endpoint_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    initial = _write_poscar(tmp_path)
    final = tmp_path / "final.vasp"
    final.write_text(initial.read_text(encoding="utf-8"), encoding="utf-8")
    incar = tmp_path / "INCAR.mlipx"
    incar.write_text(
        "CALCULATION = NEB\n"
        "MODEL_PATH = model.pt\n"
        "TASK = omat\n"
        "NEB_INITIAL = POSCAR\n"
        "NEB_FINAL = final.vasp\n"
        "NEB_IMAGES = 1\n",
        encoding="utf-8",
    )
    captured = {}

    class DummyEngine:
        def run_neb(self, resolved, **kwargs):
            captured["resolved"] = resolved
            captured.update(kwargs)
            return {"converged": True}

    monkeypatch.setattr(
        "mlipx.cli.CalculationEngine.from_config", lambda config: DummyEngine()
    )

    rc = main(
        [
            "run",
            "--incar",
            str(incar),
            "--output",
            str(tmp_path / "run"),
        ]
    )

    assert rc == 0
    assert captured["resolved"].calc_type == "neb"
    assert captured["resolved"].run_options["neb_initial"] == str(initial.resolve())
    assert captured["resolved"].run_options["neb_final"] == str(final.resolve())
    capsys.readouterr()
