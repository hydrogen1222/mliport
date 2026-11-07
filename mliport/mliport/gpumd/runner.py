"""Native GPUMD runner: one external process per SP / OPT / MD calculation.

This is deliberately **not** an ASE calculator.  GPUMD integrates its own MD
and performs its own minimisation, so wrapping it as an ASE ``Calculator``
would silently re-implement (or mix) two different algorithms and units of
truth (PR #61 §C).  :class:`NEPGPUMDIdentity` is an identity object that tells
writers *which* model and executable produced a result; calling
:meth:`NEPGPUMDIdentity.get_calculator` is a hard error.

The runner keeps all raw outputs in ``<output_dir>/native/`` (never deletes
them), writes a normalised ``trajectory.extxyz`` and a ``manifest.json``, and
returns the same result-dictionary shape as the ASE-backed runners.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from mliport.gpumd.audit import MDDumpAudit, MDScheduleReport
from mliport.gpumd.input import (
    build_model_xyz,
    build_run_in,
    parse_nep_model,
    run_in_provenance,
    validate_elements,
)
from mliport.gpumd.parse import (
    GPUMDIncompleteRunError,
    GPUMDParseError,
    NormalizedTrajectoryWriter,
    iter_extxyz_frames,
    optimization_result,
    parse_extxyz_frames,
    parse_thermo_out,
    read_thermo_header,
    single_point_from_dump,
    voigt_stress,
    write_normalized_trajectory,
)
from mliport.gpumd.process import (
    ProcessOutcome,
    gpumd_identity,
    resolve_gpumd_command,
    run_gpumd,
)
from mliport.gpumd.provenance import build_gpumd_provenance, write_provenance
from mliport.protocols import ProgressEvent
from mliport.runners.base import BaseRunner

if TYPE_CHECKING:
    from ase import Atoms

    from mliport.gpumd.parse import Frame

__all__ = ["NEPGPUMDIdentity", "NativeGPUMDRunner"]

_CALC_TYPE_ALIASES = {
    "sp": "sp",
    "single_point": "sp",
    "singlepoint": "sp",
    "opt": "opt",
    "optimization": "opt",
    "md": "md",
}

_MANIFEST_SCHEMA = "mliport.gpumd-manifest/1"

#: A zero-time-step single point must reproduce the input structure to this
#: wrapped-image tolerance (Angstrom); anything larger means GPUMD moved atoms.
_SP_GEOMETRY_TOLERANCE_A = 1.0e-6


def _normalise_calc_type(value: str) -> str:
    """Normalise a calculation-type string or raise ``ValueError``."""
    key = str(value).strip().lower()
    if key in _CALC_TYPE_ALIASES:
        return _CALC_TYPE_ALIASES[key]
    raise ValueError(
        f"Unsupported calc_type {value!r} for the native GPUMD runner. "
        "Choose one of: sp, opt, md."
    )


def _engine_value(engine: Any, names: tuple[str, ...], default: Any = None) -> Any:
    """Read the first present value from an engine object or mapping.

    The integrator owns the concrete engine type, so this helper accepts an
    object with attributes, a mapping, and the nested ``config`` / ``options``
    containers that mliport engines conventionally expose.
    """
    containers: list[Any] = [engine]
    for container_name in ("config", "options", "calculator_options"):
        for container in list(containers):
            if isinstance(container, Mapping):
                nested = container.get(container_name)
            else:
                nested = getattr(container, container_name, None)
            if nested is not None:
                containers.append(nested)
    for container in containers:
        if isinstance(container, Mapping):
            for name in names:
                if name in container and container[name] is not None:
                    return container[name]
            continue
        for name in names:
            try:
                value = getattr(container, name, None)
            except Exception:  # noqa: BLE001 - engine properties may fail
                value = None
            if value is not None:
                return value
    return default


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks (never held in memory)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """JSON fallback for numpy scalars/arrays and paths."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


@dataclass
class NEPGPUMDIdentity:
    """Identity of a native GPUMD calculation (not an ASE calculator).

    This object exists so the existing writers/provenance helpers can ask for
    ``.task``, ``.info()`` and ``.implemented_properties`` without pretending
    that a ``gpumd`` process can be evaluated per structure.  It is *not* a
    calculator: :meth:`get_calculator` always raises.

    Attributes:
        task: mliport task label (``bulk`` for a periodic structure).
        model_path: NEP model file.
        gpumd_path: Resolved GPUMD executable.
        gpumd_version: Version reported by the binary banner, if available.
        implemented_properties: Properties the external process produces.
        extra: Additional identity fields (NEP element types, cutoffs, ...).
    """

    task: str = "bulk"
    model_path: str | None = None
    gpumd_path: str | None = None
    gpumd_version: str | None = None
    implemented_properties: tuple[str, ...] = ("energy", "forces", "stress")
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_stress(self) -> bool:
        """Native GPUMD always reports a stress tensor."""
        return True

    @property
    def inference_mode(self) -> str:
        """Inference mode label used by writers."""
        return "native_gpumd_process"

    def info(self) -> dict[str, Any]:
        """Return the model/executable identity as a plain mapping."""
        payload: dict[str, Any] = {
            "model_type": "nep",
            "task": self.task,
            "model_path": self.model_path,
            "backend_distribution": "gpumd",
            "backend_version": self.gpumd_version,
            "implementation": "native_gpumd_process",
            "is_ase_calculator": False,
            "implemented_properties": list(self.implemented_properties),
            "numeric_precision": "float32",
            "dtype_requested": None,
            "dtype_effective": "float32",
            "dispersion": None,
        }
        payload.update(self.extra)
        return payload

    def get_calculator(self) -> Any:
        """Always raise: native GPUMD is not an ASE calculator."""
        raise RuntimeError("native GPUMD is an external process, not an ASE calculator")


class NativeGPUMDRunner(BaseRunner):
    """Run SP, optimisation or MD in one native ``gpumd`` process.

    The ``engine`` object (owned by the integrator) must expose, as attributes
    or mapping keys:

    * ``gpumd_command`` -- executable path or argv list (also accepted as an
      explicit ``gpumd_command=`` constructor argument);
    * ``nep_path`` -- NEP model file (also accepted explicitly);
    * optional ``task`` (default ``"bulk"``), ``device``,
      ``cuda_visible_devices`` and ``env``.

    Native options mirror the ASE runners so the engine can construct either
    runner with the same common keyword arguments.

    Example:
        >>> runner = NativeGPUMDRunner(
        ...     engine, calc_type="md", ensemble="NVT", temperature=300,
        ...     timestep_fs=1.0, steps=1000, thermostat="LANGEVIN",
        ...     friction_per_fs=0.01, dump_interval=10,
        ... )
        >>> result = runner.execute(atoms)
    """

    def __init__(
        self,
        engine: Any,
        *,
        calc_type: str,
        output_dir: Path | str = ".",
        gpumd_command: str | Path | list[str] | None = None,
        nep_path: str | Path | None = None,
        task: str | None = None,
        device: str | None = None,
        cuda_visible_devices: str | None = None,
        env: Mapping[str, str] | None = None,
        calculator: Any | None = None,
        ensemble: str | None = None,
        temperature: float | None = None,
        timestep_fs: float | None = None,
        timestep: float | None = None,
        steps: int | None = None,
        thermostat: str | None = None,
        friction_per_fs: float | None = None,
        friction: float | None = None,
        bussi_tau_fs: float | None = None,
        bussi_tau: float | None = None,
        nhc_tdamp_fs: float | None = None,
        nhc_tdamp: float | None = None,
        save_interval: int | None = None,
        seed: int | None = None,
        fmax: float = 0.05,
        max_steps: int = 500,
        optimizer: str = "FIRE",
        dump_interval: int | None = None,
        restart_interval: int | None = None,
        unwrapped: bool = True,
        timeout_s: float | None = None,
        verbose: bool = True,
        job_name: str | None = None,
        log_fn: Any | None = None,
        progress_callback: Any | None = None,
        cancel_event: Any | None = None,
        charge: int | None = None,
        spin: int | None = None,
        write_outcar: bool = True,
        write_forces: bool = True,
        write_stress: bool = True,
        write_json: bool = True,
        identity_extra: Mapping[str, Any] | None = None,
        allow_existing_native: bool = False,
    ):
        """Initialise the runner; no GPUMD process is launched yet."""
        self.calc_type = _normalise_calc_type(calc_type)

        if gpumd_command is None:
            gpumd_command = _engine_value(
                engine,
                (
                    "gpumd_command",
                    "nep_gpumd_command",
                    "gpumd_executable",
                    "executable",
                    "command",
                ),
            )
        if gpumd_command is None and isinstance(engine, (str, Path)):
            gpumd_command = engine
        if gpumd_command is None:
            raise ValueError(
                "NativeGPUMDRunner needs a GPUMD executable: pass "
                "gpumd_command=... or give the engine a 'gpumd_command' field"
            )
        if isinstance(gpumd_command, (list, tuple)):
            self._gpumd_argv: list[str] | None = [str(part) for part in gpumd_command]
            self.gpumd_path = Path(self._gpumd_argv[0]).expanduser()
        else:
            self._gpumd_argv = None
            self.gpumd_path = Path(str(gpumd_command)).expanduser()

        if nep_path is None:
            nep_path = _engine_value(
                engine, ("nep_path", "nep_model", "model_path", "potential")
            )
        if nep_path is None:
            raise ValueError(
                "NativeGPUMDRunner needs a NEP model file: pass nep_path=..."
            )
        self.nep_path = Path(str(nep_path)).expanduser().resolve()
        if not self.nep_path.is_file():
            raise ValueError(f"NEP model file not found: {self.nep_path}")

        self.task = str(
            task if task is not None else _engine_value(engine, ("task",), "bulk")
        )
        self.device = (
            device if device is not None else _engine_value(engine, ("device",), "cuda")
        )
        self.cuda_visible_devices = (
            cuda_visible_devices
            if cuda_visible_devices is not None
            else _engine_value(
                engine,
                ("cuda_visible_devices", "CUDA_VISIBLE_DEVICES"),
            )
        )
        if env is not None:
            self.env = {str(key): str(value) for key, value in env.items()}
        else:
            engine_env = _engine_value(engine, ("env", "environment"))
            self.env = (
                {str(key): str(value) for key, value in engine_env.items()}
                if isinstance(engine_env, Mapping)
                else None
            )

        # ``calculator`` is accepted only so the engine can pass the same
        # common keyword dict as for the ASE runners.  It is deliberately
        # ignored: native GPUMD is an external process, not an ASE calculator.
        self._ignored_calculator = calculator is not None
        self.ensemble = ensemble
        self.temperature = temperature
        self.timestep_fs = timestep_fs if timestep_fs is not None else timestep
        self.steps = steps
        self.thermostat = thermostat
        self.friction_per_fs = (
            friction_per_fs if friction_per_fs is not None else friction
        )
        self.bussi_tau_fs = bussi_tau_fs if bussi_tau_fs is not None else bussi_tau
        self.nhc_tdamp_fs = nhc_tdamp_fs if nhc_tdamp_fs is not None else nhc_tdamp
        self.seed = seed
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        self.optimizer = optimizer
        self.dump_interval = (
            dump_interval if dump_interval is not None else save_interval
        )
        self.restart_interval = restart_interval
        self.unwrapped = bool(unwrapped)
        self.allow_existing_native = bool(allow_existing_native)
        self.timeout_s = timeout_s

        # Accepted for constructor compatibility with the ASE runners; the
        # native runner writes raw GPUMD files plus the normalised trajectory
        # and manifest instead of OUTCAR/CONTCAR.
        self.write_outcar = write_outcar
        self.write_forces = write_forces
        self.write_stress = write_stress
        self.write_json = write_json

        extra = {"nep_runtime": "gpumd"}
        if identity_extra:
            extra.update({str(k): v for k, v in identity_extra.items()})
        self.calculator = NEPGPUMDIdentity(
            task=self.task,
            model_path=str(self.nep_path),
            gpumd_path=str(self.gpumd_path),
            extra=extra,
        )

        super().__init__(
            self.calculator,
            output_dir=output_dir,
            verbose=verbose,
            job_name=job_name,
            log_fn=log_fn,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            charge=charge,
            spin=spin,
        )

        if self._ignored_calculator:
            self.log(
                "Ignoring the ASE calculator argument; native GPUMD uses the "
                "external executable and NEP model identity instead",
                level="warning",
            )

        self.native_dir = self.output_dir / "native"
        self.trajectory_path = self.output_dir / "trajectory.extxyz"
        self.manifest_path = self.output_dir / "manifest.json"
        self.provenance_path = self.native_dir / "provenance.json"

        self._nep_info: dict[str, Any] | None = None
        self._run_in_text: str | None = None
        self._run_in_provenance: dict[str, Any] | None = None
        self._provenance: dict[str, Any] | None = None
        self._last_outcome: ProcessOutcome | None = None
        self._last_command: list[str] | None = None
        self._result: dict[str, Any] | None = None
        self._run_entry: dict[str, Any] | None = None
        self._md_report: MDScheduleReport | None = None
        # True only after *this* run committed a verified trajectory.  A file
        # that merely exists at ``trajectory_path`` may belong to an earlier run
        # in the same output directory and must never be advertised as ours.
        self._trajectory_written = False
        self._error: BaseException | None = None
        self._secondary_errors: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Execute the native SP/OPT/MD run and return the normalised result."""
        self.print_header(f"NATIVE GPUMD {self.calc_type.upper()}")
        self._emit_progress("loading_model", "Validating structure and NEP model...")
        prepared = self._prepare_atoms(atoms)
        self._nep_info = parse_nep_model(self.nep_path)
        validate_elements(prepared, self._nep_info["element_types"])
        self.calculator.extra.update(
            {
                "nep_model_format": self._nep_info["model_format"],
                "nep_model_version": self._nep_info["model_version"],
                "nep_element_types": list(self._nep_info["element_types"]),
                "nep_cutoffs": dict(self._nep_info["cutoffs"]),
            }
        )
        self._claim_native_directory()
        self.native_dir.mkdir(parents=True, exist_ok=True)
        model_xyz_path = self.native_dir / "model.xyz"
        build_model_xyz(prepared, model_xyz_path)
        self._run_in_text = build_run_in(**self._run_in_kwargs())
        run_in_path = self.native_dir / "run.in"
        run_in_path.write_text(self._run_in_text, encoding="utf-8")
        self._run_in_provenance = run_in_provenance(**self._run_in_kwargs())

        self._emit_progress(
            "running", f"Launching native GPUMD for {self.calc_type.upper()}..."
        )
        try:
            outcome = self._execute_process()
            self._last_outcome = outcome
            if outcome.status == "cancelled":
                self._result = self._cancelled_result()
                return self._finalize(self._result)
            if outcome.status == "timeout":
                raise RuntimeError(
                    "Native GPUMD timed out "
                    f"after {outcome.wall_time_s:.1f} s; raw files were kept in "
                    f"{self.native_dir}"
                )
            if outcome.returncode != 0:
                raise RuntimeError(self._process_failure_message(outcome))

            try:
                self._result = self._parse_result(prepared)
            except BaseException as exc:  # noqa: BLE001 - failure is a result
                self._error = exc
                self._result = self._failed_result(exc)
            return self._finalize(self._result)
        except BaseException as exc:
            self._error = exc
            self._write_manifest_safely("failed")
            raise

    #: Files whose presence means ``native/`` already holds a finished run.
    _PREVIOUS_RUN_MARKERS: ClassVar[tuple[str, ...]] = (
        "run.in",
        "model.xyz",
        "thermo.out",
        "traj.xyz",
        "sp.xyz",
        "opt.xyz",
        "restart.xyz",
    )

    def _claim_native_directory(self) -> None:
        """Never mix two runs inside one ``native/`` directory.

        GPUMD's ``dump_xyz`` and ``dump_thermo`` **append**. Re-running into a
        directory that already holds a previous run therefore produced a
        ``sp.xyz`` with two frames (the SP parser correctly refused it) and
        would silently concatenate two trajectories for MD — i.e. analysis
        over frames from two different potentials/runs.  Both outcomes are
        unacceptable, so a stale directory is refused unless the caller
        explicitly allowed overwriting, and even then the old tree is moved
        aside (never deleted) as ``native.attempt-<n>``.
        """
        if not self.native_dir.exists():
            return
        stale = [
            name
            for name in self._PREVIOUS_RUN_MARKERS
            if (self.native_dir / name).exists()
        ]
        if not stale:
            return
        if not self.allow_existing_native:
            raise RuntimeError(
                f"{self.native_dir} already contains output from a previous run "
                f"({', '.join(sorted(stale))}). GPUMD appends to dump_xyz and "
                "thermo.out, so running here would mix two trajectories. Use a "
                "fresh output directory, or pass --overwrite to move the old "
                "raw output aside."
            )
        index = 1
        while (self.output_dir / f"native.attempt-{index}").exists():
            index += 1
        archived = self.output_dir / f"native.attempt-{index}"
        self.native_dir.rename(archived)
        self.log(
            f"Previous raw GPUMD output moved to {archived} (--overwrite); "
            "nothing was deleted."
        )

    # ------------------------------------------------------------------
    # Process execution and provenance
    # ------------------------------------------------------------------
    def _run_in_kwargs(self) -> dict[str, Any]:
        """Arguments shared by build_run_in and run_in_provenance."""
        traj_filename = {
            "sp": "sp.xyz",
            "opt": "opt.xyz",
            "md": "traj.xyz",
        }[self.calc_type]
        dump_interval = self.dump_interval
        if self.calc_type == "md" and dump_interval is None:
            dump_interval = 10
        return {
            "calc_type": self.calc_type,
            "nep_path": self.nep_path,
            "ensemble": self.ensemble,
            "temperature": self.temperature,
            "timestep_fs": self.timestep_fs,
            "steps": self.steps,
            "thermostat": self.thermostat,
            "friction_per_fs": self.friction_per_fs,
            "bussi_tau_fs": self.bussi_tau_fs,
            "nhc_tdamp_fs": self.nhc_tdamp_fs,
            "seed": self.seed,
            "fmax": self.fmax,
            "max_steps": self.max_steps,
            "optimizer": self.optimizer,
            "dump_interval": dump_interval,
            "traj_filename": traj_filename,
            "restart_interval": self.restart_interval,
            "unwrapped": self.unwrapped,
        }

    def _execute_process(self) -> ProcessOutcome:
        """Resolve the executable, record provenance and launch GPUMD."""
        if self._gpumd_argv is not None:
            command = list(self._gpumd_argv)
        else:
            command = resolve_gpumd_command(self.gpumd_path)
        self._last_command = command
        identity = gpumd_identity(command[0], env=self.env)
        self.calculator.gpumd_path = identity.get("real_path", command[0])
        self.calculator.gpumd_version = identity.get("version")
        self.calculator.extra["gpumd_sha256"] = identity.get("sha256")
        self.calculator.extra["gpumd_version_error"] = identity.get("version_error")

        assert self._run_in_text is not None  # noqa: S101 - set by run()
        self._provenance = build_gpumd_provenance(
            gpumd_path=command[0],
            nep_path=self.nep_path,
            run_in_text=self._run_in_text,
            model_xyz_path=self.native_dir / "model.xyz",
            device_requested=self.device,
            cuda_visible_devices=self.cuda_visible_devices,
            gpumd_identity_record=identity,
            extra={
                "run_in_provenance": self._run_in_provenance,
                "identity": self.calculator.info(),
            },
        )
        write_provenance(self.provenance_path, self._provenance)

        # Pin the child to the leased GPU.  Recording the UUID in provenance is
        # not enough: without CUDA_VISIBLE_DEVICES the child would see every
        # device and pick device 0, which may be a GPU another job holds a lease
        # on.  CUDA accepts the UUID form directly.
        child_env = {str(k): str(v) for k, v in (self.env or {}).items()}
        if self.cuda_visible_devices:
            child_env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)

        return run_gpumd(
            command,
            self.native_dir,
            timeout_s=self.timeout_s,
            cancel_event=self._cancel_event,
            log_fn=self.log,
            env=child_env or None,
        )

    @staticmethod
    def _process_failure_message(outcome: ProcessOutcome) -> str:
        """Build a concise but diagnostic process-failure message."""
        stdout_tail: list[str] = []
        stderr_tail: list[str] = []
        try:
            stdout_tail = (
                Path(outcome.stdout_path)
                .read_text(encoding="utf-8", errors="replace")
                .strip()
                .splitlines()[-8:]
            )
        except OSError:
            pass
        try:
            stderr_tail = (
                Path(outcome.stderr_path)
                .read_text(encoding="utf-8", errors="replace")
                .strip()
                .splitlines()[-8:]
            )
        except OSError:
            pass
        return (
            f"Native GPUMD exited with return code {outcome.returncode}. "
            f"Raw files are kept in {outcome.cwd}. "
            f"stdout tail: {stdout_tail}; stderr tail: {stderr_tail}"
        )

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------
    def _base_result(self) -> dict[str, Any]:
        """Common result fields shared by every outcome."""
        return {
            "calc_type": self.calc_type,
            "status": "failed",
            "converged": False if self.calc_type == "opt" else None,
            "energy": None,
            "energy_per_atom": None,
            "forces": None,
            "stress": None,
            "natoms": None,
            "trajectory_path": (
                str(self.trajectory_path) if self._trajectory_written else None
            ),
            "native_dir": str(self.native_dir),
            "provenance": self._provenance,
        }

    def _failed_result(self, exc: BaseException) -> dict[str, Any]:
        """Build a failed result that preserves the first exception."""
        result = self._base_result()
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["secondary_errors"] = list(self._secondary_errors)
        if isinstance(exc, GPUMDParseError):
            result["parse_error"] = str(exc)
        return result

    def _cancelled_result(self) -> dict[str, Any]:
        """Build a cancelled result (raw files stay on disk)."""
        result = self._base_result()
        result["status"] = "cancelled"
        result["error"] = None
        result["secondary_errors"] = list(self._secondary_errors)
        return result

    def _parse_result(self, atoms: Atoms) -> dict[str, Any]:
        """Dispatch to the calculation-specific parser."""
        if self.calc_type == "sp":
            return self._parse_sp_result(atoms)
        if self.calc_type == "opt":
            return self._parse_opt_result(atoms)
        return self._parse_md_result(atoms)

    def _parse_sp_result(self, atoms: Atoms) -> dict[str, Any]:
        """Parse ``sp.xyz`` into the standard single-point result."""
        dump = self.native_dir / "sp.xyz"
        parsed = single_point_from_dump(dump)
        expected_symbols = list(atoms.get_chemical_symbols())
        if parsed["symbols"] != expected_symbols:
            raise GPUMDParseError(
                "GPUMD reordered atoms: input species "
                f"{expected_symbols} vs dump {parsed['symbols']}"
            )
        self._check_finite(
            atoms,
            parsed["energy"],
            parsed["forces"],
            context="native GPUMD single point",
        )
        positions_unchanged, max_delta = self._require_unchanged_geometry(atoms, parsed)
        frames = parse_extxyz_frames(dump)
        if frames and frames[0].step is None:
            frames[0].step = 0
        write_normalized_trajectory(frames, self.trajectory_path, source_path=dump)
        forces = np.asarray(parsed["forces"], dtype=float)
        fmax_component = float(np.max(np.abs(forces)))
        fmax_norm = float(np.max(np.linalg.norm(forces, axis=1)))
        result = self._base_result()
        result.update(
            {
                "status": "completed",
                "energy": parsed["energy"],
                "energy_per_atom": parsed["energy_per_atom"],
                "forces": forces,
                "stress": parsed["stress"],
                "natoms": parsed["natoms"],
                "fmax_component": fmax_component,
                "fmax_norm": fmax_norm,
                "positions_unchanged": positions_unchanged,
                "positions_max_delta_A": max_delta,
                "positions_convention": parsed["positions_convention"],
                "trajectory_path": str(self.trajectory_path),
                "native": {
                    "dump": str(dump),
                    "stress_source": parsed["stress_source"],
                },
            }
        )
        self._run_entry = self._run_entry_for("sp")
        self._trajectory_written = True
        return result

    def _parse_opt_result(self, atoms: Atoms) -> dict[str, Any]:
        """Parse ``opt.xyz`` into the standard optimisation result."""
        dump = self.native_dir / "opt.xyz"
        parsed = optimization_result(
            dump,
            force_tolerance=self.fmax,
            max_steps=self.max_steps,
            stdout_path=self.native_dir / "stdout.log",
            run_in_path=self.native_dir / "run.in",
        )
        self._check_finite(
            atoms,
            parsed["energy"],
            parsed["forces"],
            context="native GPUMD optimization final",
        )
        frames = parse_extxyz_frames(dump)
        if frames and parsed["nsteps"] is not None:
            frames[-1].step = int(parsed["nsteps"])
        write_normalized_trajectory(frames, self.trajectory_path, source_path=dump)
        converged = parsed["converged"] is True
        status = "completed" if converged else "not_converged"
        result = self._base_result()
        result.update(
            {
                "status": status,
                "converged": converged,
                "energy": parsed["energy"],
                "energy_per_atom": parsed["energy_per_atom"],
                "forces": np.asarray(parsed["forces"], dtype=float),
                "stress": parsed["stress"],
                "natoms": parsed["natoms"],
                "nsteps": parsed["nsteps"],
                "fmax": self.fmax,
                "fmax_component": parsed["fmax_component"],
                "fmax_norm": parsed["fmax_norm"],
                "force_tolerance_eV_A": parsed["force_tolerance_eV_A"],
                "max_steps": parsed["max_steps"],
                "max_steps_reached": parsed["max_steps_reached"],
                "converged_reason": parsed["converged_reason"],
                "optimizer": parsed["optimizer"],
                "trajectory_path": str(self.trajectory_path),
                "native": {
                    "dump": str(dump),
                    "stdout": str(self.native_dir / "stdout.log"),
                },
            }
        )
        self._run_entry = self._run_entry_for("opt")
        self._trajectory_written = True
        return result

    def _parse_md_result(self, atoms: Atoms) -> dict[str, Any]:
        """Verify ``traj.xyz`` + ``thermo.out`` and convert them in one pass.

        The dump is streamed exactly once: every frame is checked by
        :class:`~mliport.gpumd.audit.MDDumpAudit`, written to the normalised
        trajectory and dropped, so memory is bounded by one frame however long
        the run was.  The normalised trajectory is published only after the
        whole dump has been accepted as the complete product of ``run.in``;
        exit status 0 alone never makes a run ``completed``.
        """
        dump = self.native_dir / "traj.xyz"
        thermo_path = self.native_dir / "thermo.out"
        provenance = self._run_in_provenance or {}
        steps = int(provenance["steps"])
        dump_interval = int(provenance["dump_interval"])
        dt_fs = float(provenance["timestep_fs"])
        thermo = parse_thermo_out(thermo_path)
        thermo_header = read_thermo_header(thermo_path)
        temperatures = np.asarray(thermo["T"], dtype=float)
        if temperatures.size == 0:
            raise GPUMDParseError(f"{thermo_path}: no temperature rows")
        final_temperature = float(temperatures[-1])
        if not math.isfinite(final_temperature):
            raise GPUMDParseError(f"{thermo_path}: final temperature is not finite")
        if thermo_header["num_atoms"] not in (None, len(atoms)):
            raise GPUMDIncompleteRunError(
                f"{thermo_path}: thermo.out states {thermo_header['num_atoms']} "
                f"atoms but the input structure has {len(atoms)}"
            )
        audit = MDDumpAudit(
            steps=steps,
            dump_interval=dump_interval,
            dt_fs=dt_fs,
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.cell.array,
            thermo_rows=int(temperatures.size),
            thermo_dump_interval=thermo_header["dump_thermo"],
            thermo_dt_output_fs=thermo_header["dt_output_fs"],
            source=str(dump),
        )
        final: Frame | None = None
        with NormalizedTrajectoryWriter(
            self.trajectory_path, dt_fs=dt_fs, source_path=dump
        ) as writer:
            for frame in iter_extxyz_frames(dump):
                audit.observe_frame(frame)
                writer.write(frame)
                final = frame
            report = audit.verify()
            if final is None:  # pragma: no cover - verify() rejects empty dumps
                raise GPUMDIncompleteRunError(f"{dump}: the dump contains no frames")
            self._check_finite(
                atoms,
                final.energy_eV,
                final.forces,
                context="native GPUMD MD final frame",
            )
            writer.commit()
        self._trajectory_written = True
        self._md_report = report
        unwrapped_available = report.positions_convention == "unwrapped"
        msd_eligible = unwrapped_available
        msd_reason = (
            None
            if msd_eligible
            else (
                "GPUMD dump_xyz did not emit unwrapped_position for every "
                "frame; wrapped positions are not MSD-eligible"
            )
        )
        stress = None
        if final.stress_eV_A3 is not None:
            stress = voigt_stress(final.stress_eV_A3)
        result = self._base_result()
        result.update(
            {
                "status": "completed",
                "converged": None,
                "energy": final.energy_eV,
                "energy_per_atom": final.energy_eV / report.natoms,
                "forces": final.forces,
                "stress": stress,
                "natoms": report.natoms,
                "temperature": final_temperature,
                "temperature_mean": float(np.mean(temperatures)),
                "temperature_min": float(np.min(temperatures)),
                "temperature_max": float(np.max(temperatures)),
                "nsteps": self.steps,
                "nframes": report.nframes,
                "steps_per_frame": report.steps.tolist(),
                "times_fs": report.times_fs.tolist(),
                "md_schedule": report.to_dict(),
                "unwrapped_available": unwrapped_available,
                "msd_eligible": msd_eligible,
                "msd_ineligible_reason": msd_reason,
                "positions_convention": report.positions_convention,
                "thermo_summary": {
                    "nrows": int(temperatures.size),
                    "T_final": final_temperature,
                    "T_mean": float(np.mean(temperatures)),
                    "KE_final": float(thermo["KE"][-1]),
                    "PE_final": float(thermo["PE"][-1]),
                },
                "trajectory_path": str(self.trajectory_path),
                "native": {
                    "dump": str(dump),
                    "thermo": str(thermo_path),
                },
            }
        )
        self._run_entry = self._run_entry_for("md")
        self._run_entry.update(
            {
                "dump_interval": dump_interval,
                "nframes": report.nframes,
                "unwrapped_available": unwrapped_available,
                "msd_eligible": msd_eligible,
                "msd_ineligible_reason": msd_reason,
            }
        )
        return result

    def _run_entry_for(self, calc_type: str) -> dict[str, Any]:
        """Return the manifest run entry common to every calculation type."""
        native = self.native_dir
        dump_name = {"sp": "sp.xyz", "opt": "opt.xyz", "md": "traj.xyz"}[calc_type]
        entry: dict[str, Any] = {
            "index": 0,
            "calc_type": calc_type,
            "raw_trajectory": self._relative(native / dump_name),
            "thermo": self._relative(native / "thermo.out")
            if (native / "thermo.out").exists()
            else None,
            "restart": self._relative(native / "restart.xyz")
            if (native / "restart.xyz").exists()
            else None,
        }
        if calc_type == "md":
            entry.update(
                {
                    "steps": self.steps,
                    "dt_fs": self.timestep_fs,
                    "ensemble": (
                        None if self.ensemble is None else str(self.ensemble).upper()
                    ),
                    "thermostat": self.thermostat,
                    "thermostat_mapping": (
                        None
                        if self._run_in_provenance is None
                        else self._run_in_provenance.get("thermostat_mapping")
                    ),
                    "T_coup": (
                        None
                        if self._run_in_provenance is None
                        else self._run_in_provenance.get("T_coup")
                    ),
                    "seed": self.seed,
                }
            )
        else:
            entry.update({"steps": 1, "dt_fs": 0.0})
        return entry

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _positions_unchanged(
        reference: Atoms,
        candidate: Any,
        cell: Any,
    ) -> tuple[bool, float]:
        """Compare wrapped fractional coordinates of input and dump positions.

        GPUMD wraps ``pos`` into the cell, so the wrapped-image difference is
        the physically meaningful check.  The returned maximum is in Angstrom.
        """
        reference_positions = np.asarray(reference.get_positions(), dtype=float)
        candidate_positions = np.asarray(candidate, dtype=float)
        if candidate_positions.shape != reference_positions.shape:
            return False, float("inf")
        matrix = np.asarray(cell, dtype=float)
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return False, float("inf")
        reference_fractional = reference_positions @ inverse
        candidate_fractional = candidate_positions @ inverse
        delta = candidate_fractional - reference_fractional
        delta -= np.round(delta)
        delta_cartesian = delta @ matrix
        max_delta = float(np.max(np.abs(delta_cartesian)))
        return max_delta <= _SP_GEOMETRY_TOLERANCE_A, max_delta

    def _require_unchanged_geometry(
        self, atoms: Atoms, parsed: Mapping[str, Any]
    ) -> tuple[bool, float]:
        """Refuse a zero-time-step single point whose geometry is not the input.

        ``run.in`` for a single point uses ``time_step 0``, so the dumped
        positions and cell must reproduce the input structure.  If they do not,
        the energy and forces describe a different structure than the one that
        was asked for; reporting them as ``completed`` would put a wrong number
        into a ranking.  The check is made against the dump's own cell as well,
        because wrapped fractional coordinates are only comparable when both
        sides use the same cell.

        Returns:
            ``(True, max_wrapped_displacement_A)`` when the geometry is unchanged.

        Raises:
            GPUMDParseError: If any atom moved, or the cell changed, by more
                than ``_SP_GEOMETRY_TOLERANCE_A``.
        """
        positions_unchanged, max_delta = self._positions_unchanged(
            atoms, parsed["positions"], parsed["cell"]
        )
        dumped_cell = np.asarray(parsed["cell"], dtype=float)
        input_cell = np.asarray(atoms.cell.array, dtype=float)
        if dumped_cell.shape != input_cell.shape:
            cell_change = float("inf")
        else:
            cell_change = float(np.max(np.abs(dumped_cell - input_cell)))
        if not positions_unchanged or cell_change > _SP_GEOMETRY_TOLERANCE_A:
            raise GPUMDParseError(
                "native GPUMD single point does not reproduce the input "
                f"structure: largest wrapped atomic displacement {max_delta:.3e} A, "
                f"largest cell change {cell_change:.3e} A (tolerance "
                f"{_SP_GEOMETRY_TOLERANCE_A:g} A) although run.in uses time_step 0. "
                "The dumped energy and forces belong to a different structure than "
                "the one requested, so no single-point result is reported."
            )
        return positions_unchanged, max_delta

    def _relative(self, path: str | Path) -> str:
        """Path relative to ``output_dir`` for the manifest."""
        candidate = Path(path)
        try:
            return candidate.resolve().relative_to(self.output_dir.resolve()).as_posix()
        except ValueError:
            return str(candidate)

    def _collect_artifacts(self) -> dict[str, dict[str, Any]]:
        """Describe every raw/normalised output that exists on disk."""
        candidates: dict[str, tuple[Path, str]] = {
            "model_xyz": (self.native_dir / "model.xyz", "extxyz-gpumd-model"),
            "run_in": (self.native_dir / "run.in", "gpumd-run-in"),
            "stdout": (self.native_dir / "stdout.log", "text"),
            "stderr": (self.native_dir / "stderr.log", "text"),
            "provenance": (
                self.provenance_path,
                "mliport-gpumd-provenance",
            ),
            "thermo": (self.native_dir / "thermo.out", "gpumd-thermo"),
            "trajectory_raw": (
                self.native_dir / "traj.xyz",
                "gpumd-extxyz-dump",
            ),
            "sp_dump": (self.native_dir / "sp.xyz", "gpumd-extxyz-dump"),
            "opt_dump": (self.native_dir / "opt.xyz", "gpumd-extxyz-dump"),
            "restart": (self.native_dir / "restart.xyz", "gpumd-restart"),
            "neighbor": (self.native_dir / "neighbor.out", "text"),
        }
        if self._trajectory_written:
            # A trajectory.extxyz that this run did not commit may belong to an
            # earlier run in the same output directory; it is never advertised.
            candidates["trajectory"] = (
                self.trajectory_path,
                "mliport-normalized-extxyz",
            )
        artifacts: dict[str, dict[str, Any]] = {}
        for name, (path, file_format) in candidates.items():
            if not path.is_file():
                continue
            artifacts[name] = {
                "path": self._relative(path),
                "format": file_format,
                "bytes": path.stat().st_size,
            }
        return artifacts

    def _result_summary(self) -> dict[str, Any]:
        """Select JSON-safe result scalars for the manifest."""
        if self._result is None:
            return {}
        summary: dict[str, Any] = {}
        scalar_keys = (
            "calc_type",
            "status",
            "converged",
            "energy",
            "energy_per_atom",
            "fmax_component",
            "fmax_norm",
            "temperature",
            "temperature_mean",
            "nframes",
            "nsteps",
            "natoms",
            "unwrapped_available",
            "msd_eligible",
            "positions_unchanged",
            "positions_max_delta_A",
            "positions_convention",
            "converged_reason",
        )
        for key in scalar_keys:
            if key not in self._result:
                continue
            value = self._result[key]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                continue
            summary[key] = value
        return summary

    def _trajectory_block(self) -> dict[str, Any] | None:
        """Describe the committed normalised trajectory for ``TrajectoryDataset``.

        ``None`` unless this run committed a verified trajectory.  The block
        records only what was *verified* (frame count, recorded times and steps,
        position convention, the MD schedule audit); the loader refuses a
        trajectory whose own frames contradict it.
        """
        if not self._trajectory_written:
            return None
        block: dict[str, Any] = {
            "path": self._relative(self.trajectory_path),
            "format": "extxyz",
            "sha256": _file_sha256(self.trajectory_path),
            "bytes": self.trajectory_path.stat().st_size,
        }
        if self._md_report is not None:
            report = self._md_report
            block.update(
                {
                    "nframes": report.nframes,
                    "natoms": report.natoms,
                    "positions_convention": report.positions_convention,
                    "dt_fs": report.dt_fs,
                    "frame_interval_fs": report.frame_interval_fs,
                    "first_step": report.first_step,
                    "last_step": report.last_step,
                    "first_time_fs": report.first_time_fs,
                    "last_time_fs": report.last_time_fs,
                    "schedule": report.to_dict(),
                }
            )
        return block

    def _manifest_payload(self, status: str) -> dict[str, Any]:
        """Build the manifest payload for the current run."""
        msd_eligible = None
        msd_reason = None
        if self._result is not None:
            msd_eligible = self._result.get("msd_eligible")
            msd_reason = self._result.get("msd_ineligible_reason")
        error = None
        if self._error is not None:
            error = {
                "type": type(self._error).__name__,
                "message": str(self._error),
            }
        run_in_sha = (
            None
            if self._run_in_text is None
            else hashlib.sha256(self._run_in_text.encode("utf-8")).hexdigest()
        )
        payload: dict[str, Any] = {
            "schema": _MANIFEST_SCHEMA,
            "status": status,
            "calc_type": self.calc_type,
            "producer": {
                "name": "mliport",
                "version": (
                    None
                    if self._provenance is None
                    else self._provenance.get("mliport_version")
                ),
            },
            "native_dir": self._relative(self.native_dir),
            "trajectory_path": (
                self._relative(self.trajectory_path)
                if self._trajectory_written
                else None
            ),
            "trajectory": self._trajectory_block(),
            "command": self._last_command,
            "process": (
                None if self._last_outcome is None else self._last_outcome.to_dict()
            ),
            "run_in": {
                "path": self._relative(self.native_dir / "run.in"),
                "sha256": run_in_sha,
                "provenance": self._run_in_provenance,
            },
            "model_xyz": {
                "path": self._relative(self.native_dir / "model.xyz"),
                "sha256": (
                    None
                    if self._provenance is None
                    else self._provenance.get("model_xyz", {}).get("sha256")
                ),
            },
            "runs": [] if self._run_entry is None else [self._run_entry],
            "result": self._result_summary(),
            "provenance": self._provenance,
            "msd_eligible": msd_eligible,
            "msd_ineligible_reason": msd_reason,
            "error": error,
            "secondary_errors": list(self._secondary_errors),
            "artifacts": self._collect_artifacts(),
        }
        return payload

    def _write_manifest(self, status: str) -> Path:
        """Write ``manifest.json`` deterministically."""
        payload = self._manifest_payload(status)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.manifest_path

    def _write_manifest_safely(self, status: str) -> None:
        """Write the manifest without masking a primary exception."""
        try:
            self._write_manifest(status)
        except Exception as exc:  # noqa: BLE001 - secondary error by contract
            record = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self._secondary_errors.append(record)
            self.log(f"Could not write manifest: {exc}", level="error")

    def _finalize(self, result: dict[str, Any]) -> dict[str, Any]:
        """Emit the terminal progress event and write the manifest."""
        status = result["status"]
        if status == "completed":
            self._emit_progress(
                "done",
                f"Native GPUMD {self.calc_type.upper()} complete",
                extra={
                    "energy": result.get("energy"),
                    "status": status,
                },
            )
        elif status == "cancelled":
            self._pending_terminal_event = ProgressEvent(
                phase="cancelled",
                message="Native GPUMD calculation cancelled",
                extra={"status": status},
            )
        elif status == "not_converged":
            self._emit_progress(
                "done",
                "Native GPUMD optimization finished but did not converge",
                extra={"status": status, "converged": False},
            )
        else:
            self._pending_terminal_event = ProgressEvent(
                phase="error",
                message=f"Native GPUMD calculation failed: {result.get('error')}",
                extra={"status": status, "error": result.get("error")},
            )
        self._write_manifest_safely(status)
        return result
