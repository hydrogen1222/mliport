# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mliport project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""
Command-line interface for mliport.

Provides subcommands for different calculation types:
- run: Run from INCAR configuration file
- sp: Single point calculation
- opt: Geometry optimization
- md: Molecular dynamics
- neb: Fixed-cell NEB/CI-NEB workflow
- batch: Batch processing
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
import threading
import time
from pathlib import Path

from ase.io import read

from mliport.config import (
    IncarConfig,
    get_default_config,
    get_schema,
    resolve_config,
)
from mliport.config.settings import init_settings_file
from mliport.engine import CalculationEngine, EngineConfig
from mliport.protocols import CancellationRequested
from mliport.run_status import (
    EXIT_CANCELLED,
    EXIT_ERROR,
    RunOutcome,
    classify_exception,
    classify_result,
    describe,
    exit_code_for,
)

#: Seconds the first SIGTERM/SIGINT gives the runners to stop at a safe point
#: and publish a cancelled status before the watchdog forces a bounded exit.
CANCEL_GRACE_SECONDS = float(os.environ.get("MLIPORT_CANCEL_GRACE_SECONDS", "30"))


def _run_product(
    engine: CalculationEngine, atoms, *, calc_type: str, started_at: float
):
    """Run a product calculation with cancellation and unified exit codes."""
    cancel_event = threading.Event()
    watchdog: threading.Timer | None = None

    def _start_watchdog() -> None:
        nonlocal watchdog
        if watchdog is not None:
            return
        watchdog = threading.Timer(CANCEL_GRACE_SECONDS, _force_cancelled_exit)
        watchdog.daemon = True
        watchdog.start()

    previous: dict[int, object] = {}

    def _handler(signum, frame):  # noqa: ARG001
        if cancel_event.is_set():
            signal.signal(signum, previous.get(signum, signal.SIG_DFL))
            raise KeyboardInterrupt
        cancel_event.set()
        print(
            "\nCancellation requested; stopping at the next safe point "
            "(press again to force).",
            file=sys.stderr,
        )
        _start_watchdog()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        try:
            signal.signal(sig, _handler)
        except ValueError:  # non-main thread (library use): keep default
            pass
    try:
        result = engine.run(
            atoms,
            log_fn=_console_log,
            started_at=started_at,
            cancel_event=cancel_event,
        )
    except (KeyboardInterrupt, CancellationRequested) as exc:
        print(f"\n{describe(classify_exception(exc))}.", file=sys.stderr)
        return exit_code_for(classify_exception(exc))
    except Exception as exc:  # noqa: BLE001 - reported as a failed run
        print(f"Error: {exc}")
        return EXIT_ERROR
    finally:
        if watchdog is not None:
            watchdog.cancel()
        for sig, prev in previous.items():
            try:
                signal.signal(sig, prev)
            except ValueError:
                pass
    outcome = classify_result(result, calc_type=calc_type)
    if outcome is RunOutcome.NOT_CONVERGED:
        print(
            f"Warning: {calc_type.upper()} {describe(outcome)}; inspect the "
            "output directory before using the structure."
        )
    return exit_code_for(outcome)


def _force_cancelled_exit() -> None:
    print(
        f"Error: cancellation did not finish within {CANCEL_GRACE_SECONDS:g}s; "
        "forcing exit.",
        file=sys.stderr,
    )
    os._exit(EXIT_CANCELLED)


def _doctor_device_argument(value: str) -> str:
    """Argparse validator for doctor device targets, including ``cuda:N``."""
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "cuda", "gpu"}:
        return normalized
    if normalized.startswith("cuda:") and normalized[5:].isdigit():
        return normalized
    raise argparse.ArgumentTypeError("device must be auto, cpu, cuda, gpu, or cuda:N")


def _mliport_version() -> str:
    """Installed mliport version for `--version` and the doctor JSON report."""
    try:
        from importlib.metadata import version as _dist_version

        return _dist_version("mliport")
    except Exception:  # noqa: BLE001 - source-only / unusual environments
        return "0.0.0"


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="mliport",
        description="mliport - VASP-like CLI for MLIP models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from INCAR file
  mliport run

  # Single point calculation
  mliport sp structure.cif --model uma-s-1.pt --task omat

  # Geometry optimization with cell relaxation
  mliport opt structure.cif --cell-opt --fmax 0.02

  # Molecular dynamics (NVT)
  mliport md structure.cif --ensemble NVT --temp 300 --steps 10000

  # Fixed-cell NEB with complete-band checkpoints
  mliport neb --initial initial.vasp --final final.vasp --model model.pt --output results/hop

  # Batch processing
  mliport batch structures/ --pattern "*.cif" --output results/

  # Run environment diagnostic (recommended after install!)
  mliport doctor --engine uma --device auto

  # Generate template INCAR
  mliport template sp
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"mliport {_mliport_version()}",
        help="Show the installed mliport version and exit.",
    )

    # Global option: explicit settings.ini (plan section 4.2). Must precede
    # the subcommand, e.g. `mliport --settings path.ini sp ...`.
    parser.add_argument(
        "--settings",
        type=str,
        default=None,
        help="Path to a settings.ini file (overrides MLIPORT_SETTINGS env / "
        "./settings.ini / ~/.config/mliport/settings.ini).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    def _add_resolver_args(
        p: argparse.ArgumentParser, *, trajectory_output_flags: bool = True
    ) -> None:
        """Add config-resolver args shared by sp/opt/md/neb/batch."""
        p.add_argument(
            "--charge",
            type=int,
            default=None,
            metavar="INTEGER",
            help="Total molecular charge stored in atoms.info (UMA omol default: 0).",
        )
        p.add_argument(
            "--spin",
            type=int,
            default=None,
            metavar="INTEGER",
            help=(
                "Molecular spin metadata; UMA omol interprets it as spin "
                "multiplicity (default: 1)."
            ),
        )
        p.add_argument(
            "--inference-mode",
            type=str,
            default=None,
            choices=["default", "turbo"],
            help="UMA inference mode (default: task-specific; turbo is rejected for other engines).",
        )
        p.add_argument(
            "--cpu-threads",
            "--torch-num-threads",
            dest="torch_num_threads",
            type=int,
            metavar="N",
            default=None,
            help="CPU intra-op threads (PyTorch for UMA/MACE and DPA PyTorch "
            "models; TensorFlow for GRACE; DPA .pb uses its TensorFlow backend "
            "settings; default: backend/system).",
        )
        p.add_argument(
            "--activation-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="UMA GPU memory-saving activation checkpointing.",
        )
        p.add_argument(
            "--gpu-memory-growth",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "GRACE: grow TensorFlow GPU memory on demand instead of "
                "reserving the whole device (default: enabled)."
            ),
        )
        p.add_argument(
            "--gpu-memory-limit-mb",
            type=int,
            default=None,
            metavar="MIB",
            help="GRACE: hard per-process TensorFlow GPU memory limit in MiB.",
        )
        p.add_argument(
            "--neighbor-cache",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "GRACE: verlet-style neighbour-list cache (extended cutoff + "
                "exact per-step re-filter while preserving periodic-image "
                "semantics; default: enabled)."
            ),
        )
        p.add_argument(
            "--neighbor-skin",
            type=float,
            default=None,
            metavar="ANGSTROM",
            help=(
                "GRACE: neighbour-cache skin in Å; the neighbour table is "
                "rebuilt when any atom moves more than skin/2 (default: 1.5)."
            ),
        )
        p.add_argument(
            "--dtype",
            "--default-dtype",
            dest="default_dtype",
            type=str,
            default=None,
            choices=["float32", "float64"],
            help="MACE model dtype (default: float64; use float32 explicitly for speed).",
        )
        p.add_argument(
            "--head",
            type=str,
            default=None,
            help="MACE head or DeepMD/DPA branch name.",
        )
        if trajectory_output_flags:
            p.add_argument(
                "--write-outcar",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Write the VASP-like OUTCAR output (use --no-write-outcar to reduce MD I/O).",
            )
            p.add_argument(
                "--write-xdatcar",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Write XDATCAR (use --no-write-xdatcar to keep only the canonical trajectory).",
            )
            p.add_argument(
                "--write-trajectory",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Write the canonical ASE trajectory (recommended for reproducible MD).",
            )
        p.add_argument(
            "--model-alias",
            type=str,
            default=None,
            help="Model alias defined in settings.ini [model:NAME].",
        )
        p.add_argument(
            "--profile",
            type=str,
            default=None,
            help="Reusable profile from settings.ini [profile:NAME].",
        )

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run calculation from INCAR file",
        description="Read configuration from INCAR file and run calculation",
    )
    run_parser.add_argument(
        "-i",
        "--incar",
        type=str,
        default="INCAR.mliport",
        help="Path to INCAR configuration file (default: INCAR.mliport)",
    )
    run_parser.add_argument(
        "-s",
        "--structure",
        type=str,
        default=None,
        help="Structure file (default: POSCAR, CONTCAR, or from INCAR)",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Output directory (default: INCAR OUTPUT_DIR, else the current "
            "directory)"
        ),
    )

    # sp command
    sp_parser = subparsers.add_parser(
        "sp",
        help="Single point calculation",
        description="Calculate energy, forces, and stress",
    )
    sp_parser.add_argument(
        "structure",
        type=str,
        help="Input structure file (CIF, XYZ, POSCAR, etc.)",
    )
    sp_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (or a model-alias name; see --model-alias).",
    )
    sp_parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type; required unless a model alias/profile " "provides it.",
    )
    sp_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task type. UMA: omat/omol/...; others: bulk/molecule (default: engine-specific).",
    )
    sp_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for calculation: cpu, cuda, gpu or cuda:N (default: cpu).",
    )
    sp_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: settings OUTPUT_DIR, else '.')",
    )
    sp_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )
    _add_resolver_args(sp_parser)
    # opt command
    opt_parser = subparsers.add_parser(
        "opt",
        help="Geometry optimization",
        description="Optimize atomic positions and optionally cell parameters",
    )
    opt_parser.add_argument(
        "structure",
        type=str,
        help="Input structure file",
    )
    opt_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (or a model-alias name; see --model-alias).",
    )
    opt_parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type; required unless a model alias/profile " "provides it.",
    )
    opt_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task type. UMA: omat/omol/...; others: bulk/molecule (default: engine-specific).",
    )
    opt_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for calculation: cpu, cuda, gpu or cuda:N (default: cpu).",
    )
    opt_parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Force convergence threshold in eV/Angstrom (default: 0.05).",
    )
    opt_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum optimization steps (default: 500).",
    )
    opt_parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["FIRE", "BFGS", "LBFGS"],
        help="Optimization algorithm (default: FIRE).",
    )
    opt_parser.add_argument(
        "--cell-opt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optimize cell parameters (requires stress support).",
    )
    opt_parser.add_argument(
        "--fix-symmetry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Preserve crystal symmetry during optimization.",
    )
    opt_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: settings OUTPUT_DIR, else '.')",
    )
    opt_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )
    _add_resolver_args(opt_parser)

    # md command
    md_parser = subparsers.add_parser(
        "md",
        help="Molecular dynamics",
        description="Run MD simulation (NVT or NVE ensemble)",
    )
    md_parser.add_argument(
        "structure",
        type=str,
        help="Input structure file",
    )
    md_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (or a model-alias name; see --model-alias).",
    )
    md_parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type; required unless a model alias/profile " "provides it.",
    )
    md_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task type. UMA: omat/omol/...; others: bulk/molecule (default: engine-specific).",
    )
    md_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for calculation: cpu, cuda, gpu or cuda:N (default: cuda).",
    )
    md_parser.add_argument(
        "--ensemble",
        type=str,
        default=None,
        choices=["NVT", "NVE"],
        help="MD ensemble (default: NVT).",
    )
    md_parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Temperature in Kelvin (default: 300).",
    )
    md_parser.add_argument(
        "--timestep",
        type=float,
        default=None,
        help="Time step in femtoseconds (default: 1.0).",
    )
    md_parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of production MD steps (default: 1000).",
    )
    md_parser.add_argument(
        "--equilibration-steps",
        "--equil-steps",
        dest="equilibration_steps",
        type=int,
        default=None,
        help=("Same-ensemble equilibration steps before production (default: 0)."),
    )
    md_parser.add_argument(
        "--thermostat",
        type=str,
        default=None,
        choices=["LANGEVIN", "BUSSI", "NHC"],
        help="NVT thermostat (default: LANGEVIN).",
    )
    md_parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help="Langevin friction coefficient in fs^-1 (default: 0.001).",
    )
    md_parser.add_argument(
        "--bussi-tau",
        type=float,
        default=None,
        help="Bussi/CSVR coupling time in fs (default: 1000.0).",
    )
    md_parser.add_argument(
        "--nhc-tdamp",
        type=float,
        default=None,
        help="Nose-Hoover-chain damping time in fs (default: 100.0).",
    )
    md_parser.add_argument(
        "--nhc-tchain",
        type=int,
        default=None,
        help="Nose-Hoover chain length (default: 3).",
    )
    md_parser.add_argument(
        "--nhc-tloop",
        type=int,
        default=None,
        help="Nose-Hoover thermostat substeps (default: 1).",
    )
    md_parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Interval for saving trajectory frames (default: 10).",
    )
    md_parser.add_argument(
        "--pre-relax",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pre-relax the structure before MD (default: enabled for NVT, disabled for NVE).",
    )
    md_parser.add_argument(
        "--pre-relax-steps",
        type=int,
        default=None,
        help="Maximum pre-relaxation steps (default: 50).",
    )
    md_parser.add_argument(
        "--pre-relax-fmax",
        type=float,
        default=None,
        help="Pre-relaxation force threshold in eV/Angstrom (default: 0.1).",
    )
    md_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible MD (default: auto-generated and recorded).",
    )
    md_parser.add_argument(
        "--velocity-policy",
        choices=["auto", "initialize", "preserve"],
        default=None,
        help="Velocity initialization policy (default: auto).",
    )
    md_parser.add_argument(
        "--com-policy",
        choices=["auto", "none", "initialize_only", "constraint"],
        default=None,
        help="Center-of-mass policy (default: auto; NHC requires explicit none).",
    )
    md_parser.add_argument(
        "--fmax-abort",
        type=float,
        default=None,
        help="Large-force abort threshold in eV/Angstrom (default: 20).",
    )
    md_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: settings OUTPUT_DIR, else '.')",
    )
    md_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )
    _add_resolver_args(md_parser)

    # Fixed-cell NEB command. Scientific defaults stay in config/defaults.py;
    # argparse uses None so settings/profile/checkpoint layers remain effective.
    neb_parser = subparsers.add_parser(
        "neb",
        help="Fixed-cell NEB/CI-NEB reaction path",
        description=(
            "Run a serial-image fixed-cell NEB with atomic full-band checkpoints, "
            "geometry resume, and VASP path export."
        ),
    )
    neb_parser.add_argument("--initial", default=None, help="Initial endpoint file")
    neb_parser.add_argument("--final", default=None, help="Final endpoint file")
    neb_parser.add_argument(
        "--resume",
        default=None,
        help="Complete checkpoint, checkpoints directory, or NEB run directory",
    )
    neb_parser.add_argument(
        "--model",
        default=None,
        help="Model path (or alias); restored from the checkpoint for resume",
    )
    neb_parser.add_argument(
        "--model-type",
        default=None,
        choices=["uma", "mace", "dpa", "grace"],
    )
    neb_parser.add_argument("--task", default=None)
    neb_parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, gpu, or cuda:N (default: cpu)",
    )
    neb_parser.add_argument(
        "--images",
        dest="n_intermediate_images",
        type=int,
        default=None,
        help="Number of intermediate images (endpoints are additional)",
    )
    neb_parser.add_argument(
        "--climb", action=argparse.BooleanOptionalAction, default=None
    )
    neb_parser.add_argument(
        "--method", dest="neb_method", choices=["improvedtangent"], default=None
    )
    neb_parser.add_argument(
        "--interpolation",
        dest="neb_interpolation",
        choices=["linear", "idpp"],
        default=None,
    )
    neb_parser.add_argument(
        "--path-convention", choices=["mic", "unwrapped"], default=None
    )
    neb_parser.add_argument("--spring", dest="neb_spring", type=float, default=None)
    neb_parser.add_argument("--fmax", type=float, default=None)
    neb_parser.add_argument("--max-steps", type=int, default=None)
    neb_parser.add_argument("--pre-fmax", dest="neb_pre_fmax", type=float, default=None)
    neb_parser.add_argument(
        "--pre-max-steps", dest="neb_pre_max_steps", type=int, default=None
    )
    neb_parser.add_argument("--maxstep", dest="neb_maxstep", type=float, default=None)
    neb_parser.add_argument(
        "--endpoint-policy", choices=["validate", "relax"], default=None
    )
    neb_parser.add_argument("--endpoint-fmax", type=float, default=None)
    neb_parser.add_argument("--endpoint-steps", type=int, default=None)
    neb_parser.add_argument("--idpp-fmax", type=float, default=None)
    neb_parser.add_argument("--idpp-steps", type=int, default=None)
    neb_parser.add_argument(
        "--idpp-mic", action=argparse.BooleanOptionalAction, default=None
    )
    neb_parser.add_argument("--min-distance", dest="neb_min_distance", type=float)
    neb_parser.add_argument("--checkpoint-interval", type=int, default=None)
    neb_parser.add_argument(
        "--allow-unvalidated-neb", action=argparse.BooleanOptionalAction, default=None
    )
    neb_parser.add_argument("--fmax-abort", type=float, default=None)
    neb_parser.add_argument(
        "--atom-map",
        default=None,
        help="0-based final indices in initial-atom order, comma-separated",
    )
    neb_parser.add_argument(
        "--image-shifts",
        default=None,
        help="Integer lattice shifts per atom, e.g. '0,0,0;1,0,0'",
    )
    neb_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="New run directory; resume infers and locks the original directory",
    )
    neb_parser.add_argument(
        "--name",
        "-n",
        default=None,
        help="Optional new-run subdirectory (used by queue jobs)",
    )
    _add_resolver_args(neb_parser, trajectory_output_flags=False)

    # batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch processing",
        description="Process multiple structures in batch mode",
    )
    batch_parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing structure files",
    )
    batch_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (or a model-alias name; see --model-alias).",
    )
    batch_parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type; required unless a model alias/profile " "provides it.",
    )
    batch_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task type. UMA: omat/omol/...; others: bulk/molecule (default: engine-specific).",
    )
    batch_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for calculation: cpu, cuda, gpu or cuda:N (default: cpu).",
    )
    batch_parser.add_argument(
        "--calc-type",
        type=str,
        default=None,
        choices=["sp", "opt"],
        help="Sub-calculation type for the sweep (default: sp).",
    )
    batch_parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help=(
            "File glob to match. If omitted, discovers *.cif, *.xyz, *.vasp "
            "and POSCAR*."
        ),
    )
    batch_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="batch_results",
        help="Output directory (default: batch_results)",
    )
    batch_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )
    _add_resolver_args(batch_parser)

    # config command (plan section 4.2 / 9 / 17.6 / 24)
    for _model_parser in (
        run_parser,
        sp_parser,
        opt_parser,
        md_parser,
        neb_parser,
        batch_parser,
    ):
        _model_parser.add_argument(
            "--lenient-config",
            action="store_true",
            help=(
                "Downgrade unknown/cross-backend configuration keys to "
                "warnings (legacy behaviour). The default is strict."
            ),
        )

    config_parser = subparsers.add_parser(
        "config",
        help="Inspect and manage mliport configuration",
        description="Show resolved config, settings search paths, validate "
        "settings.ini, or explain where a parameter value comes from.",
    )
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_show = config_sub.add_parser("show", help="Show the resolved configuration")
    config_show.add_argument(
        "--lenient-config",
        action="store_true",
        help="Downgrade unknown configuration keys to warnings.",
    )
    config_sub.add_parser("paths", help="List settings.ini search paths")
    config_init = config_sub.add_parser("init", help="Create a settings.ini")
    config_init.add_argument(
        "--project",
        action="store_true",
        help="Write ./settings.ini",
    )
    config_init.add_argument(
        "--user",
        action="store_true",
        help="Write the user-level settings.ini",
    )
    config_init.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Explicit output path (overrides --project/--user)",
    )
    config_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing file",
    )
    config_validate = config_sub.add_parser(
        "validate", help="Validate a settings.ini file"
    )
    config_validate.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="settings.ini path (default: resolved search)",
    )
    config_explain = config_sub.add_parser(
        "explain",
        help="Explain why a parameter has its resolved value",
    )
    config_explain.add_argument("key", type=str, help="Parameter name")
    config_show = config_sub.add_parser("schema", help="List recognised option keys")
    config_show.add_argument(
        "--strict",
        action="store_true",
        help="Only show keys valid in strict mode",
    )

    template_parser = subparsers.add_parser(
        "template",
        help="Generate template INCAR files",
        description="Generate template configuration files",
    )
    template_parser.add_argument(
        "type",
        choices=["sp", "opt", "md", "neb"],
        help="Type of template to generate",
    )
    template_parser.add_argument(
        "--engine",
        choices=["uma", "fairchem", "mace", "dpa", "grace"],
        default=None,
        help=(
            "Render backend-specific model fields. Omit for a "
            "backend-neutral template (no implicit UMA default)."
        ),
    )
    template_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file name (default: INCAR.{type})",
    )

    # tui command
    subparsers.add_parser(
        "tui",
        help="Launch interactive TUI mode",
        description="Launch interactive terminal UI for visual configuration",
    )

    # convert-xdatcar command: re-emit a trajectory in the exact VASP
    # XDATCAR layout (unwrapped coordinates, VASP column widths).
    xdatcar_parser = subparsers.add_parser(
        "convert-xdatcar",
        help="Export an ASE-readable trajectory as a standard XDATCAR",
        description=(
            "Export trajectory.traj, or re-emit an mliport/VASP XDATCAR, in "
            "the layout VASP writes: unwrapped scaled coordinates and VASP "
            "column widths (12-char fields, 8 decimals). For legacy mliport "
            "runs prefer trajectory.traj because it retains image history."
        ),
    )
    xdatcar_parser.add_argument(
        "input",
        type=str,
        help="Input ASE-readable trajectory (prefer trajectory.traj for old runs)",
    )
    xdatcar_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file (default: <input>.vasp)",
    )

    # Analysis v2: explicit scientific subcommands rather than a monolithic
    # list of loosely coupled tasks.
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run validated trajectory analysis",
        description="Analyze an mliport run, ASE trajectory, or XDATCAR.",
    )
    analyze_parser.add_argument(
        "run",
        type=str,
        help="mliport run directory or trajectory file",
    )
    analyze_sub = analyze_parser.add_subparsers(dest="analysis_task", required=True)

    def _analysis_source_options(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--positions-convention",
            choices=["wrapped", "unwrapped", "unknown"],
            default=None,
            help=(
                "Coordinate convention for external trajectories. A value that "
                "conflicts with trusted mliport artifact metadata is rejected."
            ),
        )
        p.add_argument(
            "--frame-interval-fs",
            type=float,
            default=None,
            help="Explicit saved-frame interval when trajectory metadata lacks time.",
        )
        p.add_argument(
            "--force", action="store_true", help="Recompute an existing analysis ID."
        )

    def _analysis_range_options(p: argparse.ArgumentParser) -> None:
        _analysis_source_options(p)
        p.add_argument(
            "--include-equilibration",
            action="store_true",
            help="Include equilibration frames (production-only is the default).",
        )
        p.add_argument("--start-frame", type=int, default=None)
        p.add_argument("--stop-frame", type=int, default=None)

    validate_parser = analyze_sub.add_parser(
        "validate", help="Validate trajectory semantics and task eligibility"
    )
    _analysis_source_options(validate_parser)

    thermo_parser = analyze_sub.add_parser(
        "thermo", help="Thermodynamic diagnostics and NVE energy drift"
    )
    _analysis_range_options(thermo_parser)
    thermo_parser.add_argument("--block-size-frames", type=int, default=None)

    rdf_parser = analyze_sub.add_parser("rdf", help="Bulk partial RDF and coordination")
    _analysis_range_options(rdf_parser)
    rdf_parser.add_argument("--center", dest="center_species", required=True)
    rdf_parser.add_argument("--neighbor", dest="neighbor_species", required=True)
    rdf_parser.add_argument("--rmax", dest="r_max_A", type=float, default=None)
    rdf_parser.add_argument("--bins", type=int, default=200)
    rdf_parser.add_argument("--cn-cutoff", dest="cn_cutoff_A", type=float, default=None)
    rdf_parser.add_argument("--stride", type=int, default=1)

    rmsd_parser = analyze_sub.add_parser(
        "rmsd", help="Periodic-displacement RMSD/RMSF crystal diagnostic"
    )
    _analysis_range_options(rmsd_parser)
    rmsd_parser.add_argument("--species", default=None)
    rmsd_parser.add_argument(
        "--drift-reference",
        choices=["none", "nonmobile", "indices"],
        default="none",
    )
    rmsd_parser.add_argument("--drift-indices", default=None)

    msd_parser = analyze_sub.add_parser("msd", help="Directional windowed MSD")
    _analysis_range_options(msd_parser)
    msd_parser.add_argument("--mobile", dest="mobile_species", required=True)
    msd_parser.add_argument(
        "--axes", default="xyz", help="One or comma-separated x/y/z/xy/xz/yz/xyz"
    )
    msd_parser.add_argument(
        "--drift-reference",
        choices=["none", "nonmobile", "all", "indices"],
        default="none",
    )
    msd_parser.add_argument(
        "--drift-mode",
        choices=["none", "arithmetic_mean", "mass_weighted_com"],
        default="mass_weighted_com",
        help=(
            "Explicit native-MSD centre definition (default mass-weighted COM); "
            "the recorded definition must match kinisi for cross-backend "
            "comparison."
        ),
    )
    msd_parser.add_argument("--drift-indices", default=None)
    msd_parser.add_argument("--method", choices=["fft", "direct"], default="fft")
    msd_parser.add_argument("--fit-start-ps", type=float, default=None)
    msd_parser.add_argument("--fit-stop-ps", type=float, default=None)
    msd_parser.add_argument(
        "--alpha-window-decades",
        type=float,
        default=None,
        help=(
            "Full width (base-10 decades) of the local log-log regression "
            "window for the MSD exponent; default 0.25. Recorded in the result."
        ),
    )
    msd_parser.add_argument(
        "--alpha-min-points",
        type=int,
        default=None,
        help="Minimum points inside a local alpha window (default 5).",
    )
    msd_parser.add_argument(
        "--alpha-min-origins",
        type=int,
        default=None,
        help=(
            "Minimum time origins before an alpha point is reported as "
            "sufficient information (default 8; origins are not independent "
            "samples)."
        ),
    )
    msd_parser.add_argument(
        "--alpha-consistency-band",
        type=float,
        default=None,
        help=(
            "Half-band around alpha=1 accepted as consistent with diffusion "
            "(default 0.2); outside it the point is not_diffusive."
        ),
    )
    msd_parser.add_argument(
        "--alpha-max-lag-ps",
        type=float,
        default=None,
        help=(
            "Ignore lag times above this value for the alpha estimator (and "
            "record it in the result); raw MSD data are unaffected."
        ),
    )
    msd_parser.add_argument(
        "--alpha-log-x",
        action="store_true",
        help="Plot the alpha figure with a logarithmic lag-time axis.",
    )
    msd_parser.add_argument(
        "--alpha-focus-band",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help=(
            "Focused alpha view (view only): the exported values are never "
            "cropped and the figure annotates how many points fall outside."
        ),
    )

    transport_parser = analyze_sub.add_parser(
        "transport", help="kinisi tracer diffusion and NE conductivity"
    )
    _analysis_source_options(transport_parser)
    transport_parser.add_argument("--mobile", dest="mobile_species", required=True)
    transport_parser.add_argument(
        "--charge", dest="ionic_charge_e", type=float, required=True
    )
    transport_parser.add_argument("--fit-start-ps", type=float, required=True)
    transport_parser.add_argument(
        "--lag-step-ps",
        type=float,
        default=None,
        help=(
            "Spacing of the explicit kinisi lag-time grid in ps. "
            "This sparsifies lag times, not trajectory frames, and must be "
            "used together with --lag-stop-ps."
        ),
    )
    transport_parser.add_argument(
        "--lag-stop-ps",
        type=float,
        default=None,
        help=(
            "Maximum lag time for the explicit kinisi lag-time grid in ps. "
            "This sparsifies lag times, not trajectory frames, and must be "
            "used together with --lag-step-ps."
        ),
    )
    transport_parser.add_argument("--dimensions", default="xyz")
    transport_parser.add_argument(
        "--drift-reference",
        choices=["none", "nonmobile", "all", "indices"],
        default="none",
    )
    transport_parser.add_argument(
        "--drift-mode",
        choices=["none", "arithmetic_mean", "mass_weighted_com"],
        default="arithmetic_mean",
        help=(
            "Explicit framework-drift centre definition. kinisi transport "
            "pre-applies it once and records it; cross-backend comparisons "
            "require matching definitions."
        ),
    )
    transport_parser.add_argument("--drift-indices", default=None)
    transport_parser.add_argument("--temperature-K", type=float, default=None)
    transport_parser.add_argument("--collective-conductivity", action="store_true")
    transport_parser.add_argument(
        "--collective-system-particles",
        type=int,
        default=1,
        help=(
            "kinisi index-ordered statistical groups for collective/jump analysis; "
            "these are not independent MD replicas (default: 1)."
        ),
    )
    transport_parser.add_argument(
        "--jump-diffusion",
        action="store_true",
        help="Also run kinisi JumpDiffusionAnalyzer as a total/jump diagnostic.",
    )
    transport_parser.add_argument("--random-seed", type=int, default=0)
    transport_parser.add_argument("--n-samples", type=int, default=1000)
    transport_parser.add_argument("--n-walkers", type=int, default=32)
    transport_parser.add_argument("--n-burn", type=int, default=500)
    transport_parser.add_argument("--n-thin", type=int, default=10)
    transport_parser.add_argument(
        "--parser-memory-limit-gib",
        type=float,
        default=4.0,
        help=(
            "Fail before kinisi parsing when the estimated trajectory-parser "
            "peak exceeds this many GiB (default: 4)."
        ),
    )
    transport_parser.add_argument(
        "--allow-reconstructed-fallback",
        action="store_true",
        help=(
            "Diagnostic only: permit transport to fall back to kinisi's "
            "unverified from_ase displacement reconstruction when the exact "
            "adapter is unavailable. Results are marked "
            "publication_grade=false."
        ),
    )

    density_parser = analyze_sub.add_parser(
        "density", help="Periodic 3-D mobile-ion density map"
    )
    _analysis_range_options(density_parser)
    density_parser.add_argument("--mobile", dest="mobile_species", required=True)
    density_parser.add_argument("--spacing", dest="spacing_A", type=float, default=0.25)
    density_parser.add_argument("--smoothing-sigma-A", type=float, default=None)
    density_parser.add_argument("--stride", type=int, default=1)

    arrhenius_parser = analyze_sub.add_parser(
        "arrhenius", help="Fit independent multi-temperature diffusion results"
    )
    _analysis_source_options(arrhenius_parser)
    arrhenius_parser.add_argument(
        "--temperature",
        dest="temperatures_K",
        type=float,
        action="append",
        required=True,
    )
    arrhenius_parser.add_argument(
        "--diffusivity",
        dest="diffusivities_m2_s",
        type=float,
        action="append",
        required=True,
    )
    arrhenius_parser.add_argument(
        "--diffusivity-std", dest="diffusivity_std_m2_s", type=float, action="append"
    )
    arrhenius_parser.add_argument(
        "--extrapolate-temperature",
        dest="extrapolate_temperatures_K",
        type=float,
        action="append",
        default=[],
    )
    arrhenius_parser.add_argument(
        "--source-run-id", dest="source_run_ids", action="append"
    )

    electrolyte_parser = analyze_sub.add_parser(
        "electrolyte", help="GEMDAT site, jump, and percolation mechanisms"
    )
    _analysis_source_options(electrolyte_parser)
    electrolyte_parser.add_argument("--mobile", dest="mobile_species", required=True)
    site_group = electrolyte_parser.add_mutually_exclusive_group(required=True)
    site_group.add_argument("--sites", dest="sites_path", default=None)
    site_group.add_argument("--discover-sites-from-density", action="store_true")
    electrolyte_parser.add_argument("--temperature-K", type=float, default=None)
    electrolyte_parser.add_argument("--resolution-A", type=float, default=0.5)
    electrolyte_parser.add_argument("--background-level", type=float, default=0.1)
    electrolyte_parser.add_argument("--site-radius-A", type=float, default=None)
    electrolyte_parser.add_argument("--minimal-residence", type=int, default=0)
    electrolyte_parser.add_argument(
        "--drift-reference",
        choices=["none", "nonmobile", "all", "indices"],
        default="none",
    )
    electrolyte_parser.add_argument(
        "--drift-mode",
        choices=["none", "arithmetic_mean", "mass_weighted_com"],
        default="arithmetic_mean",
        help="Explicit GEMDAT framework-drift centre definition.",
    )
    electrolyte_parser.add_argument("--drift-indices", default=None)
    electrolyte_parser.add_argument(
        "--jump-dimensions", type=int, choices=[1, 2, 3], default=3
    )
    electrolyte_parser.add_argument("--percolation-axes", default="xyz")

    for name in ("vacf", "spectrum"):
        spectral_parser = analyze_sub.add_parser(
            name,
            help=(
                "Velocity autocorrelation"
                if name == "vacf"
                else "Qualified VACF-derived velocity spectrum"
            ),
        )
        _analysis_range_options(spectral_parser)
        spectral_parser.add_argument("--species", default=None)
        spectral_parser.add_argument(
            "--method", choices=["fft", "direct"], default="fft"
        )
        if name == "spectrum":
            spectral_parser.add_argument(
                "--taper",
                choices=["one-sided-cosine", "none"],
                default="one-sided-cosine",
            )
            spectral_parser.add_argument(
                "--normalization",
                choices=["normalized_area", "raw_spectrum"],
                default="normalized_area",
            )

    # queue command: Slurm-like submission and scheduling of background jobs.
    queue_parser = subparsers.add_parser(
        "queue",
        help="Submit and schedule queued background jobs",
        description=(
            "Slurm-like job queue: submit tasks from a JSON file, then run "
            "a scheduler that executes them one at a time (default) or up to "
            "--max-concurrent at once. Each task may use its own Python "
            "environment / engine / model."
        ),
    )
    queue_sub = queue_parser.add_subparsers(dest="queue_command", required=True)
    queue_submit = queue_sub.add_parser(
        "submit",
        help="Enqueue tasks from a JSON task file",
        description=(
            "Read a JSON task file and add every task to the queue with "
            "status PENDING. Start the scheduler with 'mliport queue start' "
            "to execute them."
        ),
    )
    queue_submit.add_argument(
        "task_file",
        type=str,
        help="Path to the JSON task file",
    )
    queue_start = queue_sub.add_parser(
        "start",
        help="Start the queue scheduler",
        description=(
            "Promote queued (PENDING) jobs to RUNNING and run them to "
            "completion, respecting --max-concurrent."
        ),
    )
    queue_start.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="How many jobs may run at once (default: 1, single GPU).",
    )
    queue_start.add_argument(
        "--poll",
        type=float,
        default=5.0,
        help="Scheduler poll interval in seconds (default: 5).",
    )
    queue_start.add_argument(
        "--foreground",
        action="store_true",
        help="Run the scheduler in the foreground (Ctrl-C to stop).",
    )
    queue_sub.add_parser(
        "stop",
        help="Stop a background scheduler",
    )
    queue_sub.add_parser(
        "status",
        help="Show queue and scheduler status",
    )
    queue_pause = queue_sub.add_parser(
        "pause",
        help="Pause one pending job, or the whole pending queue",
    )
    queue_pause.add_argument(
        "job_id",
        nargs="?",
        help="Specific pending job ID; omit to pause all pending dispatch",
    )
    queue_resume = queue_sub.add_parser(
        "resume",
        help="Resume one paused job, or the whole pending queue",
    )
    queue_resume.add_argument(
        "job_id",
        nargs="?",
        help="Specific paused job ID; omit to resume all pending dispatch",
    )

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run environment diagnostic checks",
        description=(
            "Inventory installed engines without importing them, or validate one "
            "selected engine/device runtime."
        ),
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable doctor report as JSON.",
    )
    doctor_parser.add_argument(
        "--engine",
        choices=["auto", "uma", "mace", "dpa", "grace"],
        default="auto",
        help=(
            "Engine to runtime-check. auto (default) performs a side-effect-free "
            "package inventory only."
        ),
    )
    doctor_parser.add_argument(
        "--device",
        type=_doctor_device_argument,
        default="auto",
        metavar="DEVICE",
        help="Target auto, cpu, cuda, gpu, or cuda:N (used with --engine).",
    )
    doctor_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model checkpoint to load (requires --engine and --task).",
    )
    doctor_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Explicit model task/PBC semantic used by the model probe.",
    )
    doctor_parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="Explicit MACE head or DPA/DeepMD branch for the model probe.",
    )
    doctor_parser.add_argument(
        "--structure",
        type=str,
        default=None,
        help="Optional structure for a real energy/force single-point smoke test.",
    )
    doctor_parser.add_argument(
        "--dtype",
        dest="default_dtype",
        choices=["float32", "float64"],
        default="float64",
        help="MACE dtype used by the model probe (default: float64).",
    )

    # setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Detect your GPU and get the matching PyTorch install command",
        description=(
            "Detect NVIDIA GPUs via nvidia-smi (works even before PyTorch is "
            "installed) and print the exact torch version + install commands "
            "for your card. Supports Maxwell (GTX 900 series) through Hopper; "
            "Blackwell (RTX 50) gets torch 2.8+; Kepler (GTX 700) is flagged "
            "unsupported."
        ),
    )
    setup_parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of the text report",
    )

    # jobs command
    jobs_parser = subparsers.add_parser("jobs", help="List background jobs")
    jobs_parser.add_argument(
        "--refresh", type=int, default=0, help="Auto-refresh interval in seconds"
    )

    # kill command
    kill_parser = subparsers.add_parser("kill", help="Kill a background job")
    kill_parser.add_argument("job_id", help="Job ID to kill")

    # clean command
    subparsers.add_parser("clean", help="Remove completed/failed job records")

    return parser


def print_header():
    """Print mliport header."""
    print("=" * 80)
    print(" MLIPORT".center(80))
    print(" (mliport - backend-neutral MLIP workflows)".center(80))
    print("=" * 80)
    print()


def _run_started_at(args: argparse.Namespace) -> float:
    """Return the command-entry timestamp, including parsing and input loading."""
    return getattr(args, "_run_started_at", time.perf_counter())


def _console_log(message: str, level: str = "info") -> None:
    """Print a live engine log message immediately."""
    print(message, flush=True)


def _load_settings(args: argparse.Namespace):
    """Load settings.ini honouring --settings / MLIPORT_SETTINGS / cwd / user."""
    from mliport.config.settings import load_settings  # noqa: PLC0415

    return load_settings(explicit=getattr(args, "settings", None))


def _build_cli_opts(args: argparse.Namespace, calc_type: str) -> dict:
    """Collect non-None CLI args into a canonical option dict.

    Only explicitly-provided values are included so that unspecified
    parameters fall through to settings/profile/built-in defaults
    (plan section 17.6: argparse defaults -> None, resolved by ConfigResolver).
    """
    opts: dict = {}
    if getattr(args, "lenient_config", False):
        opts["strict_config"] = False
    for key in (
        "model_type",
        "task",
        "device",
        "charge",
        "spin",
        "inference_mode",
        "torch_num_threads",
        "activation_checkpointing",
        "gpu_memory_growth",
        "gpu_memory_limit_mb",
        "neighbor_cache",
        "neighbor_skin",
        "default_dtype",
        "head",
        "write_outcar",
        "write_xdatcar",
        "write_trajectory",
    ):
        value = getattr(args, key, None)
        if value is not None:
            opts[key] = value
    if calc_type == "opt":
        for key in ("fmax", "max_steps", "optimizer", "cell_opt", "fix_symmetry"):
            value = getattr(args, key, None)
            if value is not None:
                opts[key] = value
    elif calc_type == "md":
        if getattr(args, "temp", None) is not None:
            opts["temperature"] = args.temp
        for key in (
            "ensemble",
            "timestep",
            "steps",
            "equilibration_steps",
            "thermostat",
            "friction",
            "bussi_tau",
            "nhc_tdamp",
            "nhc_tchain",
            "nhc_tloop",
            "save_interval",
            "pre_relax",
            "pre_relax_steps",
            "pre_relax_fmax",
            "seed",
            "velocity_policy",
            "com_policy",
            "fmax_abort",
        ):
            value = getattr(args, key, None)
            if value is not None:
                opts[key] = value
    elif calc_type == "neb":
        if getattr(args, "initial", None) is not None:
            opts["neb_initial"] = args.initial
        if getattr(args, "final", None) is not None:
            opts["neb_final"] = args.final
        for key in (
            "n_intermediate_images",
            "climb",
            "neb_method",
            "neb_interpolation",
            "path_convention",
            "neb_spring",
            "fmax",
            "max_steps",
            "neb_pre_fmax",
            "neb_pre_max_steps",
            "neb_maxstep",
            "endpoint_policy",
            "endpoint_fmax",
            "endpoint_steps",
            "idpp_fmax",
            "idpp_steps",
            "idpp_mic",
            "neb_min_distance",
            "checkpoint_interval",
            "allow_unvalidated_neb",
            "fmax_abort",
        ):
            value = getattr(args, key, None)
            if value is not None:
                opts[key] = value
    elif calc_type == "batch":
        # batch `--calc-type` selects the sub-calculation (sp/opt).
        if getattr(args, "calc_type", None) is not None:
            opts["sub_calc_type"] = args.calc_type
        for key in ("pattern",):
            value = getattr(args, key, None)
            if value is not None:
                opts[key] = value
    return opts


def _resolve_engine_config(
    args: argparse.Namespace,
    calc_type: str,
    *,
    model_path: str | None = None,
    incar_layer: dict | None = None,
    output_dir: str | None = None,
    job_name: str | None = None,
) -> tuple:
    """Resolve a full config and build an EngineConfig from it.

    Returns ``(EngineConfig, ResolvedConfig, MliportSettings)``.
    """
    from mliport.config.aliases import parse_model_aliases  # noqa: PLC0415

    settings = _load_settings(args)
    aliases = parse_model_aliases(settings.parser)

    model_alias = getattr(args, "model_alias", None)
    cli = _build_cli_opts(args, calc_type)

    # `--model` may be a filesystem path OR a model-alias name (plan section
    # 5.1: `--model mace_mpa0`). An explicit --model-alias wins; otherwise a
    # value matching a known alias is treated as the alias.
    if model_path is not None:
        cli.setdefault("model_path", model_path)
    else:
        model_arg = getattr(args, "model", None)
        if model_arg is not None:
            if model_alias is None and model_arg in aliases:
                model_alias = model_arg
            else:
                cli.setdefault("model_path", model_arg)
        elif model_alias is None and not incar_layer:
            raise SystemExit(
                "Error: one of --model PATH or --model-alias NAME is required."
            )

    resolved = resolve_config(
        calc_type=calc_type,
        settings=settings,
        model_alias_name=model_alias,
        profile_name=getattr(args, "profile", None),
        incar=incar_layer,
        cli=cli,
        cli_base_dir=Path.cwd(),
    )
    engine_config = EngineConfig.from_resolved(resolved)
    # Output-directory precedence (review R13): an explicit argument wins,
    # then an explicit CLI --output, then INCAR/settings OUTPUT_DIR, then ".".
    # The argparse default for --output is None precisely so an INCAR
    # OUTPUT_DIR is not silently shadowed by the current-directory default.
    explicit_output = (
        output_dir if output_dir is not None else getattr(args, "output", None)
    )
    if explicit_output is None:
        explicit_output = resolved.settings.get("output_dir")
    engine_config.output_dir = (
        Path(explicit_output if explicit_output else ".").expanduser().resolve()
    )
    engine_config.job_name = (
        job_name if job_name is not None else getattr(args, "name", None)
    )
    _maybe_reexec_for_device_isolation(resolved)
    return engine_config, resolved, settings


_ISOLATION_BACKENDS = {"dpa", "grace"}


def _device_visibility_value(device: str) -> str:
    """Value for CUDA_VISIBLE_DEVICES that matches the requested device."""
    import subprocess  # noqa: PLC0415

    dev = str(device).lower()
    if dev == "cpu":
        return ""
    index = 0
    if ":" in dev:
        with contextlib.suppress(ValueError):
            index = int(dev.split(":", 1)[1])
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if index < len(uuids):
            return uuids[index]
    except (OSError, subprocess.SubprocessError):
        pass
    # The isolation guard accepts any single non-empty token; the runtime UUID
    # check still compares the framework-reported UUID with the requested one.
    return str(index)


def _maybe_reexec_for_device_isolation(resolved) -> None:
    """Restart DPA/GRACE commands in a process with isolated GPU visibility.

    Their ASE adapters cannot select a GPU after the framework is imported, so
    the documented per-environment CLI commands would otherwise fail on a
    machine that has not exported CUDA_VISIBLE_DEVICES. Re-exec keeps the
    user-facing command and arguments unchanged (fresh-install acceptance
    finding, LGPS round).
    """
    from mliport.devices import visibility_is_isolated  # noqa: PLC0415

    if resolved.model_type not in _ISOLATION_BACKENDS:
        return
    if os.environ.get("MLIPORT_DEVICE_ISOLATION_REEXEC") == "1":
        return
    if visibility_is_isolated(resolved.device):
        return
    value = _device_visibility_value(resolved.device)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = value
    env["MLIPORT_DEVICE_ISOLATION_REEXEC"] = "1"
    print(
        f"Note: restarting with isolated device visibility "
        f"(CUDA_VISIBLE_DEVICES={value!r}) for {resolved.model_type.upper()}.",
        file=sys.stderr,
    )
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and os.access(argv0, os.X_OK):
        os.execvpe(argv0, sys.argv, env)
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "mliport.cli", *sys.argv[1:]],
        env,
    )


def _final_run_dir(config: EngineConfig) -> Path:
    """Directory the calculation actually writes into (job subdir included)."""
    if config.job_name:
        return Path(config.output_dir) / config.job_name
    return Path(config.output_dir)


def _emit_resolved_config(resolved, output_dir: Path) -> None:
    """Atomically write resolved_config.json into the FINAL run directory.

    The file records the provenance of one job; writing it to the parent
    output directory let sibling jobs overwrite each other (review R13).
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    if not resolved.settings.get("write_resolved_config", True):
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "resolved_config.json"
    payload = json.dumps(resolved.as_dict(), indent=2, default=str) + "\n"
    tmp = output_dir / f".resolved_config.json.tmp-{os.getpid()}"
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        print(
            f"Warning: could not write {target}: {exc}",
            file=sys.stderr,
        )


def cmd_run(args: argparse.Namespace) -> int:
    started_at = _run_started_at(args)
    # Load configuration
    incar_path = Path(args.incar).expanduser()
    # Backward-compat: fall back to legacy INCAR.uma if the default is missing.
    if not incar_path.exists() and args.incar == "INCAR.mliport":
        legacy = Path("INCAR.uma")
        if legacy.exists():
            incar_path = legacy
    if not incar_path.exists():
        print(f"Error: INCAR file not found: {incar_path}")
        return 1

    config = IncarConfig.from_file(incar_path)

    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    calc_type = config.get_str("CALC_TYPE", config.get_str("CALCULATION", "sp")).lower()
    if calc_type == "neb":
        job_name = config.get_str("JOB_NAME", None)
        try:
            engine_config, resolved, _settings = _resolve_engine_config(
                args,
                "neb",
                incar_layer=config,
                output_dir=args.output,
                job_name=job_name,
            )
            initial_path = resolved.run_options.get("neb_initial")
            final_path = resolved.run_options.get("neb_final")
            if not initial_path or not final_path:
                raise ValueError("NEB INCAR requires both NEB_INITIAL and NEB_FINAL")
            initial = read(initial_path)
            final = read(final_path)
            engine = CalculationEngine.from_config(engine_config)
            result = engine.run_neb(
                resolved,
                initial=initial,
                final=final,
                log_fn=_console_log,
            )
            outcome = classify_result(result, calc_type="neb")
            if outcome is RunOutcome.NOT_CONVERGED:
                print(
                    "Warning: NEB did not converge; resume from the latest "
                    "checkpoint to continue."
                )
            return exit_code_for(outcome)
        except (KeyboardInterrupt, CancellationRequested) as exc:
            print(f"\n{describe(classify_exception(exc))}.", file=sys.stderr)
            return exit_code_for(classify_exception(exc))
        except Exception as exc:
            print(f"Error: {exc}")
            return EXIT_ERROR

    # Determine structure file
    structure_file = args.structure
    if structure_file is None:
        # Try common defaults
        for default in ["POSCAR", "CONTCAR", "structure.cif", "structure.xyz"]:
            if Path(default).exists():
                structure_file = default
                break

    if structure_file is None:
        print("Error: No structure file specified and no default found")
        return 1

    structure_path = Path(structure_file).expanduser().resolve()
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    # Read structure
    print(f"Reading structure from: {structure_path}")
    try:
        atoms = read(structure_path)
    except Exception as e:
        print(f"Error reading structure: {e}")
        return 1

    print(f"System: {atoms.get_chemical_formula()}")
    print(f"Atoms: {len(atoms)}")
    print()

    # Determine calculation type from INCAR (authoritative for the `run` flow).
    # `mliport run` dispatches single-structure runs only. A BATCH INCAR was
    # previously accepted by validation and then crashed with a confusing
    # "Unknown calc_type: batch" after model loading. Fail fast with guidance.
    if calc_type not in {"sp", "opt", "md"}:
        print(f"Error: CALC_TYPE={calc_type!r} cannot be executed by 'mliport run'.")
        if calc_type == "batch":
            print(
                "       Batch calculations use the 'mliport batch' subcommand instead."
            )
        return 1
    job_name = config.get_str("JOB_NAME", None)
    # The INCAR dict (UPPER keys) is passed as a resolver layer; aliases are
    # canonicalised automatically (MODEL_TYPE -> model_type, FMAX -> fmax ...).
    engine_config, resolved, _settings = _resolve_engine_config(
        args,
        calc_type,
        incar_layer=config,
        output_dir=args.output,
        job_name=job_name,
    )

    engine = CalculationEngine.from_config(engine_config)
    _emit_resolved_config(resolved, _final_run_dir(engine_config))
    return _run_product(engine, atoms, calc_type=calc_type, started_at=started_at)


def cmd_sp(args: argparse.Namespace) -> int:
    """Execute 'sp' command."""
    started_at = _run_started_at(args)

    structure_path = Path(args.structure).expanduser().resolve()
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    config, resolved, _settings = _resolve_engine_config(args, "sp")

    print(f"System: reading from {structure_path}")

    try:
        atoms = read(structure_path)
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")

        engine = CalculationEngine.from_config(config)
        _emit_resolved_config(resolved, _final_run_dir(config))
        return _run_product(engine, atoms, calc_type="sp", started_at=started_at)
    except Exception as e:
        print(f"Error: {e}")
        return EXIT_ERROR


def cmd_opt(args: argparse.Namespace) -> int:
    """Execute 'opt' command."""
    started_at = _run_started_at(args)

    structure_path = Path(args.structure).expanduser().resolve()
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    config, resolved, _settings = _resolve_engine_config(args, "opt")

    print(f"Reading structure from: {structure_path}")

    try:
        atoms = read(structure_path)
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")

        engine = CalculationEngine.from_config(config)
        _emit_resolved_config(resolved, _final_run_dir(config))
        return _run_product(engine, atoms, calc_type="opt", started_at=started_at)
    except Exception as e:
        print(f"Error: {e}")
        return EXIT_ERROR


def cmd_md(args: argparse.Namespace) -> int:
    """Execute 'md' command."""
    started_at = _run_started_at(args)

    structure_path = Path(args.structure).expanduser().resolve()
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    config, resolved, _settings = _resolve_engine_config(args, "md")

    print(f"Reading structure from: {structure_path}")

    try:
        atoms = read(structure_path)
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")

        engine = CalculationEngine.from_config(config)
        _emit_resolved_config(resolved, _final_run_dir(config))
        return _run_product(engine, atoms, calc_type="md", started_at=started_at)
    except Exception as e:
        print(f"Error: {e}")
        return EXIT_ERROR


def _parse_neb_atom_map(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        parsed = [int(token.strip()) for token in value.split(",")]
    except ValueError as exc:
        raise ValueError("--atom-map must contain comma-separated integers") from exc
    if not parsed or any(not token.strip() for token in value.split(",")):
        raise ValueError("--atom-map cannot contain empty entries")
    return parsed


def _parse_neb_image_shifts(value: str | None) -> list[list[int]] | None:
    if value is None:
        return None
    rows: list[list[int]] = []
    try:
        for row in value.split(";"):
            values = [int(token.strip()) for token in row.split(",")]
            if len(values) != 3 or any(not token.strip() for token in row.split(",")):
                raise ValueError
            rows.append(values)
    except ValueError as exc:
        raise ValueError(
            "--image-shifts must contain one x,y,z integer row per atom, "
            "separated by semicolons"
        ) from exc
    if not rows:
        raise ValueError("--image-shifts cannot be empty")
    return rows


def cmd_neb(args: argparse.Namespace) -> int:
    """Execute a new or geometry-resumed fixed-cell NEB workflow."""
    from mliport.neb.workflow import (  # noqa: PLC0415
        checkpoint_resolved_layer,
        checkpoint_run_directory,
    )

    try:
        if args.resume is not None:
            if args.initial is not None or args.final is not None:
                raise ValueError("--resume cannot be combined with --initial/--final")
            if args.atom_map is not None or args.image_shifts is not None:
                raise ValueError(
                    "--resume restores atom mapping/winding from the checkpoint"
                )
            if args.name is not None:
                raise ValueError("--name cannot change the directory of a resumed run")
            if args.model_alias is not None or args.profile is not None:
                raise ValueError(
                    "Resume restores the original alias/profile values; use direct "
                    "CLI overrides, which are checked by the fingerprint"
                )
            run_dir = checkpoint_run_directory(args.resume)
            if args.output is not None:
                requested_output = Path(args.output).expanduser().resolve()
                if requested_output != run_dir:
                    raise ValueError(
                        f"--output must be the original resume directory {run_dir}"
                    )
            checkpoint_layer = checkpoint_resolved_layer(args.resume)
            engine_config, resolved, _settings = _resolve_engine_config(
                args,
                "neb",
                incar_layer=checkpoint_layer,
                output_dir=str(run_dir),
                job_name=None,
            )
            initial = final = None
            atom_map = image_shifts = None
        else:
            if args.initial is None or args.final is None:
                raise ValueError("A new NEB run requires --initial and --final")
            if args.output is None:
                raise ValueError(
                    "A new NEB run requires an explicit --output directory"
                )
            engine_config, resolved, _settings = _resolve_engine_config(args, "neb")
            initial_path = resolved.run_options.get("neb_initial")
            final_path = resolved.run_options.get("neb_final")
            if not initial_path or not final_path:
                raise ValueError("Typed NEB endpoint paths were not resolved")
            initial = read(initial_path)
            final = read(final_path)
            atom_map = _parse_neb_atom_map(args.atom_map)
            image_shifts = _parse_neb_image_shifts(args.image_shifts)

        engine = CalculationEngine.from_config(engine_config)
        result = engine.run_neb(
            resolved,
            initial=initial,
            final=final,
            resume=args.resume,
            atom_map=atom_map,
            image_shifts=image_shifts,
            log_fn=_console_log,
        )
        print(f"NEB status: {result['status']}")
        print(f"Run ID: {result['run_id']}")
        print(f"Checkpoint: {result['latest_checkpoint']}")
        if result.get("barrier_forward_sampled_eV") is not None:
            print(
                "Sampled forward barrier: "
                f"{result['barrier_forward_sampled_eV']:.8f} eV"
            )
        outcome = classify_result(result, calc_type="neb")
        if outcome is RunOutcome.NOT_CONVERGED:
            print(
                "Warning: NEB did not converge; resume from the latest "
                "checkpoint to continue."
            )
        return exit_code_for(outcome)
    except (KeyboardInterrupt, CancellationRequested) as exc:
        print(f"\n{describe(classify_exception(exc))}.", file=sys.stderr)
        return exit_code_for(classify_exception(exc))
    except Exception as exc:
        print(f"Error: {exc}")
        return EXIT_ERROR


def cmd_batch(args: argparse.Namespace) -> int:
    """Execute 'batch' command."""
    started_at = _run_started_at(args)

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    config, resolved, _settings = _resolve_engine_config(args, "batch")
    pattern = resolved.run_options.get("pattern")

    try:
        _emit_resolved_config(
            resolved,
            config.output_dir / (config.job_name or "")
            if config.job_name
            else config.output_dir,
        )
        engine = CalculationEngine.from_config(config)
        if "pattern" in resolved.run_options:
            files = sorted(input_dir.glob(pattern))
        else:
            # Match the formats supported by BatchRunner/read(), not only CIF.
            # The old CLI duplicated discovery but forgot XYZ/VASP/POSCAR, so
            # valid directories were reported as empty.
            files = sorted(
                {
                    *input_dir.glob("*.cif"),
                    *input_dir.glob("*.xyz"),
                    *input_dir.glob("*.vasp"),
                    *input_dir.glob("POSCAR*"),
                }
            )
        if not files:
            if "pattern" in resolved.run_options:
                print(f"No files matching '{pattern}' found in {input_dir}")
            else:
                print(f"No supported structure files found in {input_dir}")
            return 1
        print(f"Found {len(files)} structure files")
        summary = engine.run_batch(
            files,
            log_fn=_console_log,
            started_at=started_at,
        )
        if summary["failed"] > 0:
            return 1
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Execute 'config' subcommands (plan section 4.2 / 9 / 17.6)."""
    sub = args.config_command

    if sub == "paths":
        settings = _load_settings(args)
        print("settings.ini search paths (high priority first):")
        for path in settings.searched:
            marker = "  (loaded)" if path in settings.loaded_paths else ""
            print(f"  {path}{marker}")
        if not settings.loaded_paths:
            print("  (no settings.ini found; using built-in defaults)")
        return 0

    if sub == "init":
        if args.output:
            target = args.output
        elif args.user:
            target = "user"
        else:
            # default to project-level when neither flag is given
            target = "project"
        try:
            path = init_settings_file(target, force=args.force)
        except FileExistsError as exc:
            print(f"Error: {exc} (use --force to overwrite)")
            return 1
        print(f"Wrote settings.ini: {path}")
        return 0

    if sub == "validate":
        explicit = args.path
        settings = (
            _load_settings(argparse.Namespace(settings=explicit))
            if explicit
            else _load_settings(args)
        )
        # Re-parse strictly to surface parser errors.
        import configparser  # noqa: PLC0415

        parser = configparser.ConfigParser(interpolation=None)
        paths_to_check = [Path(explicit)] if explicit else settings.loaded_paths
        if not paths_to_check:
            print("No settings.ini found to validate.")
            return 1
        errors: list[str] = []
        for p in paths_to_check:
            try:
                parser.read(p, encoding="utf-8")
            except (configparser.Error, OSError) as exc:
                errors.append(f"{p}: {exc}")
        # Schema-validate every section's keys.
        schema = get_schema()
        known = schema.known_names()
        for section in parser.sections():
            for key, _value in parser.items(section):
                if key.lower() not in known and not section.startswith(
                    ("engine:", "model:", "profile:")
                ):
                    suggestion = schema.suggest(key)
                    hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
                    errors.append(f"[{section}] unknown key {key!r}.{hint}")
        if errors:
            print("settings.ini validation errors:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print(f"settings.ini OK ({len(paths_to_check)} file(s)).")
        return 0

    if sub == "explain":
        # Explain uses a representative md resolve; the source trace is what matters.
        settings = _load_settings(args)
        resolved = resolve_config(
            calc_type="md",
            settings=settings,
            cli=(
                {"strict_config": False}
                if getattr(args, "lenient_config", False)
                else None
            ),
        )
        print(resolved.explain(args.key))
        return 0

    if sub == "schema":
        schema = get_schema()
        print(f"{'name':24s} {'type':8s} {'scopes':24s} aliases")
        print("-" * 80)
        for spec in schema.specs:
            scopes = ",".join(sorted(spec.scopes))
            aliases = ",".join(sorted(spec.aliases)) if spec.aliases else ""
            print(f"{spec.name:24s} {spec.type.__name__:8s} {scopes:24s} {aliases}")
        return 0

    # sub == "show"
    settings = _load_settings(args)
    resolved = resolve_config(calc_type="md", settings=settings)
    print("Resolved configuration (calc_type=md, no CLI overrides):")
    print(f"  settings.ini: {resolved.settings_path or '(built-in defaults)'}")
    print(f"  model_type  : {resolved.model_type}")
    print(f"  task        : {resolved.task}")
    print(f"  device      : {resolved.device}")
    print(f"  inference_mode: {resolved.inference_mode}")
    print(f"  calculator_options: {resolved.calculator_options}")
    print(f"  run_options: {resolved.run_options}")
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    """Execute 'template' command."""
    config = get_default_config(args.type, engine=args.engine)

    output_file = args.output
    if output_file is None:
        output_file = f"INCAR.{args.type}"

    config.write(output_file)
    print(f"Template written to: {output_file}")

    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    """Launch interactive TUI mode."""
    import sys  # noqa: PLC0415

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "Error: mliport tui requires an interactive terminal. "
            "Run it from a terminal, or use `mliport tui --help` and the "
            "Python API in headless environments.",
            file=sys.stderr,
        )
        return 1
    try:
        from mliport.tui import MliportApp  # noqa: PLC0415
    except ImportError as e:
        print("Error: TUI mode requires textual.")
        print("Install with: pip install textual")
        print(f"Import error: {e}")
        return 1

    app = MliportApp()
    return app.run()


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run environment diagnostic checks."""
    import json  # noqa: PLC0415

    from mliport.doctor import run_diagnostics, format_diagnostics  # noqa: PLC0415

    checks, failures = run_diagnostics(
        model_path=args.model,
        engine=args.engine,
        device=args.device,
        task=args.task,
        head=args.head,
        structure_path=args.structure,
        default_dtype=args.default_dtype,
    )
    if getattr(args, "json", False):
        payload = {
            "schema": "mliport.doctor-report/1",
            "mliport_version": _mliport_version(),
            "engine": args.engine,
            "device": args.device,
            "failures": failures,
            "checks": checks,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(format_diagnostics(checks))
    return 0 if failures == 0 else 1


def cmd_setup(args: argparse.Namespace) -> int:
    """Detect the GPU and print the matching PyTorch install commands."""
    import json  # noqa: PLC0415

    from mliport.gpu_setup import (  # noqa: PLC0415
        detect_gpus,
        format_setup_report,
        setup_report_json,
    )

    gpus = detect_gpus()
    # CPU-only detection/report is a successful outcome, not an error.
    if args.json:
        print(json.dumps(setup_report_json(gpus), indent=2, ensure_ascii=False))
        return 0
    print(format_setup_report(gpus))
    return 0


def cmd_convert_xdatcar(args: argparse.Namespace) -> int:
    """Export an ASE-readable trajectory in the standard VASP layout."""
    from mliport.writers.xdatcar import convert_to_vasp_xdatcar  # noqa: PLC0415

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input trajectory not found: {input_path}")
        return 1
    try:
        out = convert_to_vasp_xdatcar(input_path, args.output)
    except Exception as exc:
        print(f"Error: conversion failed: {exc}")
        return 1
    print(f"Converted to VASP-standard XDATCAR: {out}")
    return 0


def _parse_index_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("drift indices must be comma-separated integers") from exc
    if not parsed:
        raise ValueError("drift indices cannot be empty")
    return parsed


def _print_transport_summary(results: dict) -> None:
    """Print a compact, human-readable transport posterior summary."""

    tracer = results.get("tracer_diffusion") or {}
    d_post = tracer.get("D_posterior_m2_s") or {}
    lag = tracer.get("lag_grid") or {}
    ne = results.get("nernst_einstein") or {}
    sigma = ne.get("sigma_NE_tracer_posterior_mS_cm") or {}
    print(f"Production mean T: {results.get('temperature_mean_K', float('nan')):g} K")
    print("Tracer diffusion:")
    d_mean = d_post.get("mean")
    d_std = d_post.get("std")
    d_ci = d_post.get("credible_interval_95") or [None, None]
    if d_mean is not None:
        print(f"  D = {d_mean:.6e} m^2/s")
    if d_std is not None:
        print(f"  posterior SD = {d_std:.6e} m^2/s")
    if d_ci[0] is not None and d_ci[1] is not None:
        print(f"  95% credible interval = [{d_ci[0]:.6e}, {d_ci[1]:.6e}] m^2/s")
    fit_start = tracer.get("fit_start_ps")
    fit_stop = tracer.get("fit_stop_ps")
    if fit_start is not None and fit_stop is not None:
        print("Fit window:")
        print(f"  {fit_start:g} - {fit_stop:g} ps")
    mode = lag.get("mode")
    n_total = lag.get("n_lag_points_total")
    nominal = lag.get("nominal_step_ps")
    if nominal is None:
        nominal = lag.get("requested_step_ps")
    print("Kinisi lag grid:")
    if mode == "custom" and nominal is not None:
        print(f"  custom, nominal step {nominal:g} ps, {n_total} total points")
    else:
        print(f"  {mode}, {n_total} total points")
    sigma_mean = sigma.get("mean")
    sigma_ci = sigma.get("credible_interval_95") or [None, None]
    if sigma_mean is not None:
        print("Nernst-Einstein tracer conductivity:")
        print(f"  sigma_NE = {sigma_mean:.6e} mS/cm")
        if sigma_ci[0] is not None and sigma_ci[1] is not None:
            print(
                f"  95% credible interval = [{sigma_ci[0]:.6e}, {sigma_ci[1]:.6e}] mS/cm"
            )
    collective = results.get("collective_conductivity") or {}
    coll = collective.get("sigma_collective_mS_cm_posterior") or {}
    if coll.get("mean") is not None:
        print(
            "Collective Einstein ionic conductivity (trajectory/charge-model quantity):"
        )
        print(f"  sigma_collective = {coll['mean']:.6e} mS/cm")
        interval = coll.get("credible_interval_95") or [None, None]
        if interval[0] is not None:
            print(
                f"  95% credible interval = [{interval[0]:.6e}, {interval[1]:.6e}] mS/cm"
            )
        dsigma = collective.get("D_sigma_posterior_m2_s") or {}
        if dsigma.get("mean") is not None:
            print(f"  D_sigma = {dsigma['mean']:.6e} m^2/s")
        haven = results.get("haven_ratio") or {}
        if haven.get("point_estimate") is not None:
            print(f"  Haven ratio H_R = {haven['point_estimate']:.6g}")
        corr = results.get("correlation_factor") or {}
        if corr.get("point_estimate") is not None:
            print(f"  correlation factor = {corr['point_estimate']:.6g}")
    jump = results.get("jump_diffusion") or {}
    if jump:
        d_j = jump.get("D_J_posterior_m2_s") or {}
        if d_j.get("mean") is not None:
            print(f"Jump/total displacement diagnostic D_J = {d_j['mean']:.6e} m^2/s")
    for warning in results.get("warnings", []):
        print(f"WARNING: {warning}")


def cmd_analyze(args: argparse.Namespace) -> int:
    """Execute one explicit Analysis v2 subcommand."""

    from mliport.analysis.runner import run_analysis  # noqa: PLC0415
    from mliport.analysis.schema import AnalysisRequest  # noqa: PLC0415

    excluded = {
        "command",
        "settings",
        "run",
        "analysis_task",
        "force",
        "_run_started_at",
    }
    parameters = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded and value is not None
    }
    if args.analysis_task == "transport" and (
        (args.lag_step_ps is None) != (args.lag_stop_ps is None)
    ):
        print("Error: --lag-step-ps and --lag-stop-ps must be provided together")
        return 1
    if "start_frame" in parameters:
        parameters["start"] = parameters.pop("start_frame")
    if "stop_frame" in parameters:
        parameters["stop"] = parameters.pop("stop_frame")
    if "drift_indices" in parameters:
        try:
            parameters["drift_indices"] = _parse_index_list(parameters["drift_indices"])
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
    # Argparse store_true defaults are meaningful for scientific semantics,
    # except the site-source flag which must remain explicit False when a CIF
    # was selected.
    try:
        outcome = run_analysis(
            AnalysisRequest(
                task=args.analysis_task,
                source=args.run,
                parameters=parameters,
                force=bool(args.force),
            )
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Analysis {outcome['status']}: {outcome['analysis_id']}")
    print(f"Output: {outcome['output_dir']}")
    if outcome.get("reused"):
        print("Existing completed result reused (pass --force to recompute).")
    if args.analysis_task == "validate":
        result = outcome.get("results") or {}
        time_report = result.get("time", {})
        print(f"Frames: {result.get('n_frames')}")
        print(f"Production frames: {result.get('production_frames')}")
        print(f"3-D PBC: {result.get('three_dimensional_pbc')}")
        print(f"Fixed cell: {result.get('fixed_cell')}")
        print(f"Uniform sampling: {time_report.get('uniform')}")
        print(f"Frame interval (fs): {time_report.get('frame_interval_fs')}")
        print(f"MSD/transport eligible: {result.get('eligible_for_transport')}")
        print(f"VACF eligible: {result.get('eligible_for_vacf')}")
        for warning in result.get("warnings", []):
            print(f"WARNING: {warning}")
    elif args.analysis_task == "transport":
        _print_transport_summary(outcome.get("results") or {})
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    """Execute 'queue' subcommands (submit/start/stop/status)."""
    from mliport.jobs import JobManager  # noqa: PLC0415
    from mliport.queue import (  # noqa: PLC0415
        QueueScheduler,
        pause_pending_job,
        pause_scheduler,
        resume_paused_job,
        resume_scheduler,
        scheduler_status,
        start_scheduler,
        stop_scheduler,
        submit_task_file,
    )

    sub = args.queue_command

    if sub == "submit":
        try:
            mgr = JobManager()
            job_ids, max_concurrent = submit_task_file(mgr, args.task_file)
        except Exception as exc:
            print(f"Error: {exc}")
            return 1
        print(f"Queued {len(job_ids)} task(s) with status PENDING.")
        for job_id in job_ids:
            job = mgr.get_job(job_id)
            display_name = job.get("display_name", job_id) if job else job_id
            print(f"  - {display_name} [{job_id}]")
        print(f"max_concurrent: {max_concurrent} (set in the task file)")
        print("Start the scheduler with: mliport queue start")
        return 0

    if sub == "start":
        try:
            if args.foreground:
                scheduler = QueueScheduler(
                    max_concurrent=args.max_concurrent,
                    poll_interval=args.poll,
                )
                print(
                    "Scheduler running in foreground ",
                    f"(max_concurrent={args.max_concurrent}, ",
                    f"poll={args.poll}s). Ctrl-C to stop.",
                )
                try:
                    scheduler.run_forever()
                except KeyboardInterrupt:
                    pass
                print("Scheduler stopped.")
                return 0
            pid = start_scheduler(
                jobs_dir=JobManager().jobs_dir,
                max_concurrent=args.max_concurrent,
                poll_interval=args.poll,
            )
        except Exception as exc:
            print(f"Error: {exc}")
            return 1
        print(f"Scheduler started in background (PID {pid}).")
        print("Stop it with: mliport queue stop")
        return 0

    if sub == "stop":
        stopped = stop_scheduler(jobs_dir=JobManager().jobs_dir)
        if stopped:
            print("Scheduler stopped.")
        else:
            print("No scheduler was running.")
        return 0

    if sub == "pause":
        if args.job_id:
            mgr = JobManager()
            job = mgr.get_job(args.job_id)
            if job is None:
                print(f"Error: job not found: {args.job_id}")
                return 1
            if job.get("status") != "pending":
                print(
                    f"Error: job {args.job_id} has status {job.get('status')!r}; "
                    "only pending jobs can be paused."
                )
                return 1
            pause_pending_job(mgr.jobs_dir, args.job_id)
            print(f"Job paused: {args.job_id}. Other pending jobs are unchanged.")
            return 0
        paused = pause_scheduler(jobs_dir=JobManager().jobs_dir)
        if paused:
            print("Queue paused: running jobs continue; pending jobs will wait.")
        else:
            print("Queue is already paused.")
        return 0

    if sub == "resume":
        if args.job_id:
            mgr = JobManager()
            job = mgr.get_job(args.job_id)
            if job is None:
                print(f"Error: job not found: {args.job_id}")
                return 1
            if job.get("status") != "paused":
                print(
                    f"Error: job {args.job_id} has status {job.get('status')!r}; "
                    "only paused jobs can be resumed."
                )
                return 1
            resume_paused_job(mgr.jobs_dir, args.job_id)
            print(f"Job resumed: {args.job_id}. It re-enters the pending queue.")
            return 0
        resumed = resume_scheduler(jobs_dir=JobManager().jobs_dir)
        if resumed:
            print("Queue resumed: pending jobs may now be launched.")
        else:
            print("Queue was not paused.")
        return 0

    # sub == "status"
    mgr = JobManager()
    summary = mgr.queue_summary()
    status = scheduler_status(jobs_dir=mgr.jobs_dir)
    if status["running"] and status.get("paused"):
        print(f"Scheduler: PAUSED (PID {status['pid']}; running jobs continue)")
    elif status["running"]:
        print(f"Scheduler: RUNNING (PID {status['pid']})")
    else:
        suffix = " (queue paused)" if status.get("paused") else ""
        print(f"Scheduler: not running{suffix}")
    print(
        f"Queued:     {summary['pending']} pending, {summary['paused']} paused, "
        f"{summary['claimed']} claimed, {summary['running']} running"
    )
    print(
        f"Finished:   {summary['done']} done, {summary['failed']} failed, ",
        f"{summary['cancelled']} cancelled",
    )
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    from mliport.jobs import JobManager  # noqa: PLC0415

    mgr = JobManager()
    jobs = mgr.list_jobs()
    if not jobs:
        print("No jobs found.")
        return 0
    print(
        f"{'Name':<24} {'Run ID':<36} {'Status':<12} "
        f"{'Type':<6} {'Formula':<12} {'Device'}"
    )
    print("-" * 110)
    for j in jobs:
        print(
            f"{j.get('display_name', j['job_id']):<24} {j['job_id']:<36} "
            f"{j['status']:<12} {j.get('calc_type', ''):<6} "
            f"{j.get('formula', ''):<12} {j.get('device', '')}"
        )
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    """Kill a background job."""
    from mliport.jobs import JobManager  # noqa: PLC0415

    mgr = JobManager()
    ok = mgr.kill_job(args.job_id)
    if ok:
        print(f"Killed: {args.job_id}")
        return 0
    else:
        print(f"Failed to kill: {args.job_id}")
        return 1


def cmd_clean(args: argparse.Namespace) -> int:
    """Remove completed/failed job records."""
    from mliport.jobs import JobManager  # noqa: PLC0415

    mgr = JobManager()
    removed = mgr.clean()
    if removed:
        print(f"Removed {len(removed)} completed/failed job records.")
    else:
        print("No completed/failed jobs to clean.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point: Ctrl-C anywhere resolves to exit code 130."""
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return EXIT_CANCELLED


def _main(argv: list[str] | None = None) -> int:
    run_started_at = time.perf_counter()
    # Check if running in TUI mode (no command, or explicit 'tui' command)
    if argv is None:
        argv = sys.argv[1:]

    # If no arguments provided, launch TUI by default
    if len(argv) == 0:
        try:
            from mliport.tui import MliportApp  # noqa: PLC0415

            print("Launching interactive TUI mode...")
            print("(Use --help for command-line interface)")
            time.sleep(0.5)
            app = MliportApp()
            return app.run()
        except ImportError:
            # Fall back to CLI help if textual not installed
            pass

    parser = create_parser()
    args = parser.parse_args(argv)
    args._run_started_at = run_started_at

    # Handle TUI command
    if args.command == "tui":
        return cmd_tui(args)

    if args.command is None:
        parser.print_help()
        return 1

    # --json output must be clean (no banner) for scripting. The `config`
    # subcommands also produce machine/script-oriented output, so suppress the
    # banner for them too.
    suppress_banner = (
        (args.command == "setup" and getattr(args, "json", False))
        or (args.command == "doctor" and getattr(args, "json", False))
        or args.command in {"config", "analyze"}
    )
    if not suppress_banner:
        print_header()

    # Dispatch to appropriate command handler
    commands = {
        "run": cmd_run,
        "sp": cmd_sp,
        "opt": cmd_opt,
        "md": cmd_md,
        "neb": cmd_neb,
        "batch": cmd_batch,
        "config": cmd_config,
        "template": cmd_template,
        "doctor": cmd_doctor,
        "setup": cmd_setup,
        "convert-xdatcar": cmd_convert_xdatcar,
        "analyze": cmd_analyze,
        "queue": cmd_queue,
        "jobs": cmd_jobs,
        "kill": cmd_kill,
        "clean": cmd_clean,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Error: Unknown command: {args.command}")
        return 1

    try:
        return handler(args)
    except ValueError as exc:
        # Configuration/argument errors (including the fail-closed
        # "no MLIP backend was selected") are user-facing, never a traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
