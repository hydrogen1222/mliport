"""Tests for mliport CLI (argparse flags, backward compat, config subcommands)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from mliport.cli import create_parser, main
from mliport.run_status import (
    EXIT_CANCELLED,
    EXIT_ERROR,
    EXIT_NOT_CONVERGED,
    RunOutcome,
    classify_result,
)

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
    from mliport import cli as cli_module

    captured = {}

    def fake_run_analysis(request):
        captured.update(request.parameters)
        return {"status": "success", "analysis_id": "test", "output_dir": "."}

    monkeypatch.setattr("mliport.analysis.runner.run_analysis", fake_run_analysis)
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
        "mliport.cli.CalculationEngine.from_config", lambda config: DummyEngine()
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
            "--model-type",
            "uma",
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
            # the backend contract is fail-closed: no implicit UMA
            assert main(["config", "show"]) == 1
            Path("settings.ini").write_text(
                "[md]\nMODEL_TYPE = UMA\nMODEL_PATH = model.pt\nTASK = omat\n",
                encoding="utf-8",
            )
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
        "mliport.cli.CalculationEngine.from_config",
        lambda config: DummyEngine(),
    )

    rc = main(
        [
            "md",
            str(structure),
            "--model",
            str(model),
            "--model-type",
            "uma",
            "--steps",
            "0",
            "--output",
            str(tmp_path / "results"),
        ]
    )

    assert rc == 0
    assert (
        capsys.readouterr().out.count("(mliport - backend-neutral MLIP workflows)") == 1
    )


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
# `mliport run` INCAR calc-type dispatch (regression)
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
        (dpath / "INCAR.mliport").write_text(
            "CALC_TYPE = BATCH\nMODEL_PATH = model.pt\nTASK = omat\n",
            encoding="utf-8",
        )
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["run", "--incar", "INCAR.mliport", "--structure", "POSCAR"])
            out = capsys.readouterr().out
            assert rc == 1
            assert "cannot be executed by 'mliport run'" in out
            assert "mliport batch" in out
        finally:
            os.chdir(old)


def test_run_incar_calc_type_phonon_rejected_at_validation(capsys) -> None:
    """Regression: CALC_TYPE=PHONON passed INCAR validation but the engine
    rejects it; it must now be rejected by the validator itself."""
    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        _write_poscar(dpath)
        (dpath / "INCAR.mliport").write_text(
            "CALC_TYPE = PHONON\nMODEL_PATH = model.pt\n", encoding="utf-8"
        )
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(["run", "--incar", "INCAR.mliport", "--structure", "POSCAR"])
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
    incar = tmp_path / "INCAR.mliport"
    incar.write_text(
        "CALCULATION = NEB\n"
        "MODEL_TYPE = UMA\n"
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
        "mliport.cli.CalculationEngine.from_config", lambda config: DummyEngine()
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


# ---------------------------------------------------------------------------
# R07: unified success/not_converged/failed/cancelled exit contract
# ---------------------------------------------------------------------------


class _ResultEngine:
    """Engine stub returning a fixed runner result (or raising)."""

    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.kwargs = {}

    def run(self, atoms, **kwargs):
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return self.result


def _run_opt_with_engine(tmp_path, monkeypatch, engine, capsys):
    structure = _write_poscar(tmp_path)
    model = tmp_path / "model.pt"
    model.touch()
    monkeypatch.setattr(
        "mliport.cli.CalculationEngine.from_config", lambda config: engine
    )
    rc = main(
        [
            "opt",
            str(structure),
            "--model",
            str(model),
            "--model-type",
            "uma",
            "--output",
            str(tmp_path / "results"),
        ]
    )
    return rc, capsys.readouterr()


def test_unconverged_opt_exits_2_and_keeps_partial_reason(
    tmp_path, monkeypatch, capsys
):
    """R07 red case: converged=False must not exit 0."""
    engine = _ResultEngine(
        {
            "converged": False,
            "nsteps": 0,
            "energy": -1.0,
            "forces": None,
            "failure_reason": "MAX_STEPS=0",
        }
    )
    rc, captured = _run_opt_with_opt_result = _run_opt_with_engine(
        tmp_path, monkeypatch, engine, capsys
    )
    assert rc == EXIT_NOT_CONVERGED == 2
    assert "not converged" in captured.out.lower()


def test_completed_opt_exits_0(tmp_path, monkeypatch, capsys):
    engine = _ResultEngine({"converged": True, "nsteps": 3})
    rc, _captured = _run_opt_with_engine(tmp_path, monkeypatch, engine, capsys)
    assert rc == 0


def test_engine_exception_exits_1(tmp_path, monkeypatch, capsys):
    engine = _ResultEngine(exc=RuntimeError("backend exploded"))
    rc, captured = _run_opt_with_engine(tmp_path, monkeypatch, engine, capsys)
    assert rc == EXIT_ERROR
    assert "backend exploded" in captured.out


def test_keyboard_interrupt_exits_130(tmp_path, monkeypatch, capsys):
    engine = _ResultEngine(exc=KeyboardInterrupt())
    rc, captured = _run_opt_with_engine(tmp_path, monkeypatch, engine, capsys)
    assert rc == EXIT_CANCELLED == 130
    assert "cancelled" in captured.err.lower()


def test_engine_gets_a_cancellation_event(tmp_path, monkeypatch, capsys):
    """SIGTERM/SIGINT must reach the runner through a cooperative event."""
    engine = _ResultEngine({"converged": True})
    _run_opt_with_engine(tmp_path, monkeypatch, engine, capsys)
    assert engine.kwargs.get("cancel_event") is not None
    assert engine.kwargs["cancel_event"].is_set() is False


def test_classify_result_defaults_to_not_converged_for_convergence_types():
    assert classify_result({}, calc_type="opt") is RunOutcome.NOT_CONVERGED
    assert classify_result({}, calc_type="neb") is RunOutcome.NOT_CONVERGED
    # MD/SP have no convergence concept: an empty result is a completion
    assert classify_result({}, calc_type="md") is RunOutcome.COMPLETED
    assert (
        classify_result({"converged": False}, calc_type="opt")
        is RunOutcome.NOT_CONVERGED
    )
    assert (
        classify_result({"status": "cancelled"}, calc_type="md") is RunOutcome.CANCELLED
    )


def test_run_incar_unconverged_opt_exits_2(tmp_path, monkeypatch, capsys):
    """The real INCAR path (MAX_STEPS=0, strict FMAX) must return 2."""
    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        structure = _write_poscar(dpath)
        (dpath / "model.pt").touch()
        incar = dpath / "INCAR.mliport"
        incar.write_text(
            "CALC_TYPE = OPT\n"
            "MODEL_TYPE = UMA\n"
            "MODEL_PATH = model.pt\n"
            "TASK = omat\n"
            "MAX_STEPS = 0\n"
            "FMAX = 1e-8\n",
            encoding="utf-8",
        )
        engine = _ResultEngine({"converged": False, "nsteps": 0})
        monkeypatch.setattr(
            "mliport.cli.CalculationEngine.from_config", lambda config: engine
        )
        old = os.getcwd()
        try:
            os.chdir(d)
            rc = main(
                ["run", "--incar", "INCAR.mliport", "--structure", str(structure)]
            )
        finally:
            os.chdir(old)
        assert rc == 2
        assert "not converged" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# R13: output directory + writer switches reach the runner
# ---------------------------------------------------------------------------


def test_incar_output_dir_is_supported_and_written_into_job_dir(
    tmp_path, monkeypatch, capsys
):
    structure = _write_poscar(tmp_path)
    (tmp_path / "model.pt").touch()
    incar = tmp_path / "INCAR.mliport"
    incar.write_text(
        "CALC_TYPE = SP\n"
        "MODEL_TYPE = UMA\n"
        "MODEL_PATH = model.pt\n"
        "TASK = omat\n"
        "JOB_NAME = job1\n"
        "OUTPUT_DIR = out_root\n"
        "WRITE_OUTCAR = False\n",
        encoding="utf-8",
    )
    engine = _ResultEngine({})
    monkeypatch.setattr(
        "mliport.cli.CalculationEngine.from_config", lambda config: engine
    )
    import warnings

    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            rc = main(["run", "--incar", str(incar), "--structure", str(structure)])
    finally:
        os.chdir(old)
    assert rc == 0
    assert not [
        w
        for w in record
        if "Unknown key" in str(w.message) and "OUTPUT_DIR" in str(w.message)
    ], "OUTPUT_DIR must be a recognised INCAR field, not an unknown-key warning"
    run_dir = tmp_path / "out_root" / "job1"
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    assert resolved["settings"]["write_outcar"] is False
    # the parent output directory must not be used as the resolved-config home
    assert not (tmp_path / "out_root" / "resolved_config.json").exists()


def test_two_jobs_keep_their_own_resolved_configs(tmp_path, monkeypatch):
    structure = _write_poscar(tmp_path)
    (tmp_path / "model.pt").touch()
    engine = _ResultEngine({})
    monkeypatch.setattr(
        "mliport.cli.CalculationEngine.from_config", lambda config: engine
    )
    for name in ("job_a", "job_b"):
        incar = tmp_path / f"INCAR.{name}"
        incar.write_text(
            "CALC_TYPE = SP\n"
            "MODEL_TYPE = UMA\n"
            "MODEL_PATH = model.pt\n"
            "TASK = omat\n"
            f"JOB_NAME = {name}\n"
            f"OUTPUT_DIR = jobs\n",
            encoding="utf-8",
        )
        old = os.getcwd()
        try:
            os.chdir(tmp_path)
            rc = main(["run", "--incar", str(incar), "--structure", str(structure)])
        finally:
            os.chdir(old)
        assert rc == 0
    root = tmp_path / "jobs"
    for name in ("job_a", "job_b"):
        config_path = root / name / "resolved_config.json"
        assert config_path.is_file()
        assert json.loads(config_path.read_text())["settings"]["job_name"] == name
    assert not (root / "resolved_config.json").exists()


@pytest.mark.parametrize(
    ("setting", "runner_attr", "value"),
    [
        ("write_outcar", "write_outcar", False),
        ("write_forces", "write_forces", False),
        ("write_stress", "write_stress", False),
        ("write_json", "write_json", False),
    ],
)
def test_output_switches_reach_the_runner(tmp_path, setting, runner_attr, value):
    """R13: a resolved output switch must actually reach the runner object."""
    from mliport.engine import CalculationEngine, EngineConfig

    config = EngineConfig(
        calc_type="sp",
        model_path=tmp_path / "model.pt",
        model_type="mace",
        task="bulk",
        device="cpu",
        output_dir=tmp_path,
        settings={setting: value},
    )
    engine = CalculationEngine.from_config(config)
    runner = engine._create_runner(calculator=object())
    assert getattr(runner, runner_attr) is value


def test_write_outcar_defaults_to_true(tmp_path):
    from mliport.engine import CalculationEngine, EngineConfig

    config = EngineConfig(
        calc_type="opt",
        model_path=tmp_path / "model.pt",
        model_type="mace",
        task="bulk",
        device="cpu",
        output_dir=tmp_path,
    )
    runner = CalculationEngine.from_config(config)._create_runner(calculator=object())
    assert runner.write_outcar is True


# ---------------------------------------------------------------------------
# R08: SIGTERM becomes cooperative cancellation with a bounded exit
# ---------------------------------------------------------------------------

_SIGTERM_SCRIPT = """
import sys
import time

from mliport.cli import _run_product
from mliport.protocols import CancellationRequested


class CooperativeEngine:
    def run(self, atoms, log_fn=None, started_at=None, cancel_event=None):
        deadline = time.time() + 15
        while time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise CancellationRequested("test cancellation")
            time.sleep(0.05)
        return {"converged": True}


rc = _run_product(CooperativeEngine(), None, calc_type="sp", started_at=0.0)
print(f"RC {rc}")
sys.exit(rc)
"""

_SIGTERM_IGNORING_SCRIPT = """
import sys
import time

from mliport.cli import _run_product


class IgnoringEngine:
    def run(self, atoms, log_fn=None, started_at=None, cancel_event=None):
        time.sleep(30)
        return {"converged": True}


rc = _run_product(IgnoringEngine(), None, calc_type="sp", started_at=0.0)
print(f"RC {rc}")
sys.exit(rc)
"""


def test_sigterm_requests_cooperative_cancellation(tmp_path):
    """A queue SIGTERM must stop at a safe point and exit 130, not kill hard."""
    env = dict(os.environ)
    env["MLIPORT_CANCEL_GRACE_SECONDS"] = "20"
    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(_SIGTERM_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        time.sleep(1.0)
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == EXIT_CANCELLED, (proc.returncode, stdout, stderr)
    assert "RC 130" in stdout


def test_sigterm_watchdog_bounds_a_stuck_cancellation(tmp_path):
    """A runner that ignores cancellation is force-exited after the grace period."""
    env = dict(os.environ)
    env["MLIPORT_CANCEL_GRACE_SECONDS"] = "1.5"
    started = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(_SIGTERM_IGNORING_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        # give the child time to import mliport and install its handler
        time.sleep(1.5)
        proc.send_signal(signal.SIGTERM)
        _stdout, _stderr = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert time.monotonic() - started < 12
    assert proc.returncode == EXIT_CANCELLED


def test_md_output_switches_reach_the_runner(tmp_path):
    from mliport.engine import CalculationEngine, EngineConfig

    config = EngineConfig(
        calc_type="md",
        model_path=tmp_path / "model.pt",
        model_type="mace",
        task="bulk",
        device="cpu",
        output_dir=tmp_path,
        run_options={"steps": 1, "pre_relax": False},
        settings={"write_trajectory": False, "write_xdatcar": False},
    )
    runner = CalculationEngine.from_config(config)._create_runner(calculator=object())
    assert runner.write_trajectory is False
    assert runner.write_xdatcar is False


def test_cli_version_flag(capsys) -> None:
    """`mliport --version` is part of the documented install/check protocol."""
    from importlib.metadata import version as package_version

    from mliport.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out.strip()
    assert output.startswith("mliport ")
    assert package_version("mliport") in output


def test_tui_fails_closed_without_a_tty(capsys) -> None:
    """The interactive TUI must not hang when no terminal is attached.

    Fresh-install acceptance found that `mliport tui` under a non-TTY session
    waited for a terminal until the acceptance driver killed it.
    """
    from mliport.cli import main

    rc = main(["tui"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "requires an interactive terminal" in captured.err


def test_isolation_backends_reexec_in_a_fresh_process(monkeypatch, capsys):
    """GRACE/DPA commands restart with CUDA_VISIBLE_DEVICES when unset."""
    from types import SimpleNamespace

    import mliport.cli as cli

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MLIPORT_DEVICE_ISOLATION_REEXEC", raising=False)
    captured: dict = {}

    def fake_execvpe(path, argv, env):
        captured.update(path=path, argv=argv, env=env)
        raise SystemExit(99)

    monkeypatch.setattr(cli.os, "execvpe", fake_execvpe)
    import mliport.devices as devices

    monkeypatch.setattr(
        devices,
        "query_physical_gpus",
        lambda: [devices.VisibleGpu(0, "GPU-test", 0, "V100", (7, 0))],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli._maybe_reexec_for_device_isolation(
            SimpleNamespace(model_type="grace", device="cuda")
        )
    assert excinfo.value.code == 99
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] != ""
    assert captured["env"]["MLIPORT_DEVICE_ISOLATION_REEXEC"] == "1"
    assert "isolated device visibility" in capsys.readouterr().err

    # An already isolated process is not restarted.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cli._maybe_reexec_for_device_isolation(
        SimpleNamespace(model_type="dpa", device="cuda")
    )

    # Backends without the restriction are never restarted.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for model_type in ("mace", "uma"):
        cli._maybe_reexec_for_device_isolation(
            SimpleNamespace(model_type=model_type, device="cuda")
        )

    # CPU DPA/GRACE isolates with an empty visibility list.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit):
        cli._maybe_reexec_for_device_isolation(
            SimpleNamespace(model_type="grace", device="cpu")
        )
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""
