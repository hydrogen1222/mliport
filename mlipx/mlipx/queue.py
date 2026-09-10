"""Slurm-like job queue for mlipx.

Background calculations can be *queued* instead of launched immediately:

* tasks are submitted with status ``PENDING`` (``mlipx queue submit tasks.json``
  or the TUI), each carrying its own interpreter (``python``), engine
  (``model_type``), model, structure, calc type and options;
* a scheduler process (``mlipx queue start``) promotes queued jobs to
  ``RUNNING`` one at a time -- by default at most one job runs concurrently
  (single GPU), which can be raised with ``--max-concurrent N`` for multi-GPU
  machines (Slurm-like);
* when a job finishes (DONE/FAILED), the scheduler automatically starts the
  next queued job.

This module also owns the task-file format and the shared
``build_mlipx_command`` helper used by both the CLI queue commands and the
TUI, so the two interfaces cannot drift apart.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mlipx.config import get_schema
from mlipx.devices import VisibleGpu, visible_gpus
from mlipx.jobs import JobManager, JobStatus, _lock_file, _unlock_file

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, BinaryIO

#: Task file keys that are per-task run options vs. structural fields.
_STRUCTURAL_KEYS = {
    "name",
    "python",
    "calc_type",
    "structure",
    "initial",
    "final",
    "resume",
    "atom_map",
    "image_shifts",
    "model",
    "model_type",
    "task",
    "device",
    "output_dir",
    "options",
}
_VALID_CALC_TYPES = {"sp", "opt", "md", "neb"}
_VALID_ENGINES = {"uma", "fairchem", "mace", "dpa", "grace"}
_CALC_SCOPES = {"sp", "opt", "md", "neb", "batch"}

#: option key -> CLI flag mapping for the shared command builder.
_OPT_FLAGS: dict[str, tuple[str, ...]] = {
    "charge": ("--charge",),
    "spin": ("--spin",),
    "inference_mode": ("--inference-mode",),
    "activation_checkpointing": (
        "--activation-checkpointing",
        "--no-activation-checkpointing",
    ),
    "torch_num_threads": ("--cpu-threads",),
    "gpu_memory_growth": ("--gpu-memory-growth", "--no-gpu-memory-growth"),
    "gpu_memory_limit_mb": ("--gpu-memory-limit-mb",),
    "neighbor_cache": ("--neighbor-cache", "--no-neighbor-cache"),
    "neighbor_skin": ("--neighbor-skin",),
    "default_dtype": ("--dtype",),
    "head": ("--head",),
    "write_outcar": ("--write-outcar", "--no-write-outcar"),
    "write_xdatcar": ("--write-xdatcar", "--no-write-xdatcar"),
    "write_trajectory": ("--write-trajectory", "--no-write-trajectory"),
    # opt
    "fmax": ("--fmax",),
    "max_steps": ("--max-steps",),
    "optimizer": ("--optimizer",),
    "cell_opt": ("--cell-opt", "--no-cell-opt"),
    "fix_symmetry": ("--fix-symmetry", "--no-fix-symmetry"),
    # md
    "ensemble": ("--ensemble",),
    "temperature": ("--temp",),
    "timestep": ("--timestep",),
    "steps": ("--steps",),
    "equilibration_steps": ("--equilibration-steps",),
    "thermostat": ("--thermostat",),
    "friction": ("--friction",),
    "bussi_tau": ("--bussi-tau",),
    "nhc_tdamp": ("--nhc-tdamp",),
    "nhc_tchain": ("--nhc-tchain",),
    "nhc_tloop": ("--nhc-tloop",),
    "save_interval": ("--save-interval",),
    "pre_relax": ("--pre-relax", "--no-pre-relax"),
    "pre_relax_steps": ("--pre-relax-steps",),
    "pre_relax_fmax": ("--pre-relax-fmax",),
    "velocity_policy": ("--velocity-policy",),
    "com_policy": ("--com-policy",),
    "fmax_abort": ("--fmax-abort",),
    "seed": ("--seed",),
    # neb
    "n_intermediate_images": ("--images",),
    "climb": ("--climb", "--no-climb"),
    "neb_method": ("--method",),
    "neb_interpolation": ("--interpolation",),
    "path_convention": ("--path-convention",),
    "neb_spring": ("--spring",),
    "neb_pre_fmax": ("--pre-fmax",),
    "neb_pre_max_steps": ("--pre-max-steps",),
    "neb_maxstep": ("--maxstep",),
    "endpoint_policy": ("--endpoint-policy",),
    "endpoint_fmax": ("--endpoint-fmax",),
    "endpoint_steps": ("--endpoint-steps",),
    "idpp_fmax": ("--idpp-fmax",),
    "idpp_steps": ("--idpp-steps",),
    "idpp_mic": ("--idpp-mic", "--no-idpp-mic"),
    "neb_min_distance": ("--min-distance",),
    "checkpoint_interval": ("--checkpoint-interval",),
    "allow_unvalidated_neb": ("--allow-unvalidated-neb", "--no-allow-unvalidated-neb"),
}


def _strict_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(f"{label} must be an integer, got {value!r}")


def _freeze_declared_path(
    value: Any,
    *,
    base_dir: Path,
    label: str,
    must_exist: bool,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: missing required path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise ValueError(f"{label} not found: {path}")
    return str(path)


def _coerce_task_options(
    options: dict[str, Any],
    *,
    calc_type: str,
    model_type: str,
    label: str,
) -> dict[str, Any]:
    schema = get_schema()
    cleaned: dict[str, Any] = {}
    engine_scope = f"calculator.{model_type}"
    for raw_key, raw_value in options.items():
        key = str(raw_key)
        if key in _STRUCTURAL_KEYS:
            raise ValueError(
                f"{label}: option {key!r} is a structural key, not an option"
            )
        spec = schema.resolve(key)
        if spec is None:
            suggestion = schema.suggest(key)
            hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
            raise ValueError(f"{label}: unknown option {key!r}.{hint}")
        canonical = spec.name
        if canonical in {"calc_type", "device", "model_path", "model_type", "task"}:
            raise ValueError(
                f"{label}: option {key!r} is a structural key, not an option"
            )
        if calc_type == "neb" and canonical in {
            "write_outcar",
            "write_xdatcar",
            "write_trajectory",
        }:
            raise ValueError(
                f"{label}: option {key!r} is not applicable to the mandatory "
                "NEB checkpoint/result output"
            )
        if canonical not in _OPT_FLAGS:
            raise ValueError(
                f"{label}: option {key!r} has no supported queue/CLI representation"
            )
        calc_scopes = set(spec.scopes) & _CALC_SCOPES
        if calc_scopes and calc_type not in calc_scopes:
            raise ValueError(
                f"{label}: option {key!r} is not valid for calc_type={calc_type!r}"
            )
        calculator_scopes = {
            scope for scope in spec.scopes if scope.startswith("calculator.")
        }
        if calculator_scopes and engine_scope not in calculator_scopes:
            raise ValueError(
                f"{label}: option {key!r} is not valid for engine={model_type!r}"
            )
        if canonical == "fmax_abort" and calc_type not in {"md", "neb"}:
            raise ValueError(f"{label}: option {key!r} is only valid for MD/NEB")
        try:
            value = spec.coerce(raw_value)
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc
        errors = spec.validate_value(value)
        if errors:
            raise ValueError(f"{label}: " + "; ".join(errors))
        if canonical in cleaned and cleaned[canonical] != value:
            raise ValueError(f"{label}: conflicting aliases for option {canonical!r}")
        cleaned[canonical] = value
    return cleaned


def build_mlipx_command(
    calc_type: str,
    structure: str | None,
    model: str | None,
    model_type: str = "uma",
    task: str | None = None,
    device: str = "cpu",
    output_dir: str = "./results",
    job_name: str | None = None,
    options: dict[str, Any] | None = None,
    python: str | None = None,
    *,
    initial: str | None = None,
    final: str | None = None,
    resume: str | None = None,
    atom_map: list[int] | None = None,
    image_shifts: list[list[int]] | None = None,
) -> list[str]:
    """Build the argv for a ``mlipx <calc_type> ...`` calculation run.

    ``python`` selects the interpreter the calculation must run under (each
    engine has its own virtual environment, e.g. ``.venv`` for UMA,
    ``.venv-grace`` for GRACE). Defaults to the current interpreter.

    ``options`` maps mlipx option names (see :data:`_OPT_FLAGS`) to values;
    only keys known to the schema are forwarded, so a GRACE task never gets
    UMA-only flags and vice versa.
    """
    cmd = [python or sys.executable, "-m", "mlipx.cli", calc_type]
    if calc_type == "neb":
        if resume is not None:
            if initial is not None or final is not None:
                raise ValueError("NEB resume cannot also specify endpoints")
            cmd.extend(["--resume", resume])
        else:
            if initial is None or final is None:
                raise ValueError("A queued NEB run requires initial and final")
            cmd.extend(["--initial", initial, "--final", final])
            if atom_map is not None:
                cmd.extend(["--atom-map", ",".join(str(index) for index in atom_map)])
            if image_shifts is not None:
                encoded_shifts = ";".join(
                    ",".join(str(component) for component in row)
                    for row in image_shifts
                )
                cmd.extend(["--image-shifts", encoded_shifts])
    else:
        if structure is None:
            raise ValueError(f"A queued {calc_type} run requires a structure")
        cmd.append(structure)
    if model is not None:
        cmd.extend(["--model", model])
    cmd.extend(["--model-type", model_type, "--device", device])
    if resume is None:
        cmd.extend(["--output", output_dir])
    if task:
        cmd.extend(["--task", task])
    if job_name and resume is None:
        cmd.extend(["--name", job_name])

    engine = model_type.lower()
    for key, value in sorted((options or {}).items()):
        if key not in _OPT_FLAGS or value is None:
            continue
        # UMA-only options must not leak to other engines; MACE/DPA head and
        # dtype are engine-specific too. The CLI itself already ignores
        # inapplicable flags, but keeping the argv clean is friendlier.
        if key in {"inference_mode", "activation_checkpointing"} and engine not in {
            "uma",
            "fairchem",
        }:
            continue
        if key == "default_dtype" and engine != "mace":
            continue
        if key == "head" and engine not in {"mace", "dpa"}:
            continue
        if (
            key
            in {
                "gpu_memory_growth",
                "gpu_memory_limit_mb",
                "neighbor_cache",
                "neighbor_skin",
            }
            and engine != "grace"
        ):
            continue
        flags = _OPT_FLAGS[key]
        if isinstance(value, bool):
            cmd.append(flags[0] if value else flags[1])
        else:
            cmd.extend([flags[0], str(value)])
    return cmd


def parse_task_file(path: str | Path) -> dict[str, Any]:
    """Parse and validate a queue task file (JSON).

    Schema::

        {
          "max_concurrent": 1,               // optional, >= 1
          "tasks": [
            {
              "name": "opt-1",               // optional display name
              "python": "/abs/.venv/bin/python",  // optional, per-task env
              "calc_type": "opt",            // sp | opt | md | neb
              "structure": "/abs/a.cif",     // required, must exist
              "model": "/abs/uma-s-1.pt",    // required, must exist
              "model_type": "uma",           // optional
              "task": "omat",                // optional
              "device": "cuda:0",            // optional
              "output_dir": "/abs/out",      // optional
              "options": {"fmax": 0.05}      // optional run options
            }
          ]
        }

    Raises:
        ValueError: with a human-readable message on any problem.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Task file not found: {path}")
    base_dir = path.parent
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Task file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Task file must be a JSON object with a 'tasks' list")

    max_concurrent = _strict_integer(
        data.get("max_concurrent", 1), label="max_concurrent"
    )
    if max_concurrent < 1:
        raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("Task file must contain a non-empty 'tasks' list")

    tasks: list[dict[str, Any]] = []
    for index, task in enumerate(raw_tasks):
        if not isinstance(task, dict):
            raise ValueError(f"tasks[{index}] must be an object")
        label = f"tasks[{index}]"
        unknown_structural = set(task) - _STRUCTURAL_KEYS
        if unknown_structural:
            raise ValueError(
                f"{label}: unknown task field(s): {sorted(unknown_structural)}"
            )
        calc_type = str(task.get("calc_type", "")).lower()
        if calc_type not in _VALID_CALC_TYPES:
            raise ValueError(
                f"{label}: calc_type must be one of "
                f"{sorted(_VALID_CALC_TYPES)}, got {task.get('calc_type')!r}"
            )
        initial = final = resume = None
        atom_map = image_shifts = None
        checkpoint_layer: dict[str, Any] = {}
        if calc_type == "neb":
            if task.get("resume") is not None:
                if any(
                    task.get(key) is not None
                    for key in (
                        "structure",
                        "initial",
                        "final",
                        "atom_map",
                        "image_shifts",
                    )
                ):
                    raise ValueError(
                        f"{label}: NEB resume cannot also specify structure/endpoints"
                    )
                resume = _freeze_declared_path(
                    task.get("resume"),
                    base_dir=base_dir,
                    label=f"{label}: resume",
                    must_exist=True,
                )
                from mlipx.neb.workflow import (  # noqa: PLC0415
                    checkpoint_resolved_layer,
                    checkpoint_run_directory,
                )
                from mlipx.neb.io import resolve_checkpoint_path  # noqa: PLC0415

                checkpoint_layer = checkpoint_resolved_layer(resume)
                resume_root = checkpoint_run_directory(resume)
                requested_output = task.get("output_dir")
                if requested_output is not None:
                    frozen_output = _freeze_declared_path(
                        requested_output,
                        base_dir=base_dir,
                        label=f"{label}: output_dir",
                        must_exist=False,
                    )
                    if Path(frozen_output) != resume_root:
                        raise ValueError(
                            f"{label}: resumed output_dir must remain {resume_root}"
                        )
                output_dir = str(resume_root)
                structure = str(resolve_checkpoint_path(resume) / "band.traj")
            else:
                if task.get("structure") is not None:
                    raise ValueError(
                        f"{label}: NEB uses initial/final instead of structure"
                    )
                initial = _freeze_declared_path(
                    task.get("initial"),
                    base_dir=base_dir,
                    label=f"{label}: initial",
                    must_exist=True,
                )
                final = _freeze_declared_path(
                    task.get("final"),
                    base_dir=base_dir,
                    label=f"{label}: final",
                    must_exist=True,
                )
                raw_map = task.get("atom_map")
                if raw_map is not None:
                    if not isinstance(raw_map, list) or not raw_map:
                        raise ValueError(f"{label}: atom_map must be a non-empty list")
                    atom_map = [
                        _strict_integer(value, label=f"{label}: atom_map")
                        for value in raw_map
                    ]
                raw_shifts = task.get("image_shifts")
                if raw_shifts is not None:
                    if not isinstance(raw_shifts, list) or not raw_shifts:
                        raise ValueError(
                            f"{label}: image_shifts must be a non-empty list"
                        )
                    image_shifts = []
                    for row in raw_shifts:
                        if not isinstance(row, list) or len(row) != 3:
                            raise ValueError(
                                f"{label}: every image_shifts row must have 3 integers"
                            )
                        image_shifts.append(
                            [
                                _strict_integer(value, label=f"{label}: image_shifts")
                                for value in row
                            ]
                        )
                structure = initial
                output_dir = _freeze_declared_path(
                    task.get("output_dir", "./results"),
                    base_dir=base_dir,
                    label=f"{label}: output_dir",
                    must_exist=False,
                )
        else:
            if any(
                task.get(key) is not None
                for key in (
                    "initial",
                    "final",
                    "resume",
                    "atom_map",
                    "image_shifts",
                )
            ):
                raise ValueError(
                    f"{label}: initial/final/resume are only valid for NEB"
                )
            structure = _freeze_declared_path(
                task.get("structure"),
                base_dir=base_dir,
                label=f"{label}: structure",
                must_exist=True,
            )
            output_dir = _freeze_declared_path(
                task.get("output_dir", "./results"),
                base_dir=base_dir,
                label=f"{label}: output_dir",
                must_exist=False,
            )
        model = _freeze_declared_path(
            task.get("model", checkpoint_layer.get("model_path")),
            base_dir=base_dir,
            label=f"{label}: model",
            must_exist=True,
        )
        model_type = str(
            task.get("model_type", checkpoint_layer.get("model_type", "uma"))
        ).lower()
        if model_type not in _VALID_ENGINES:
            raise ValueError(
                f"{label}: model_type must be one of {sorted(_VALID_ENGINES)}, "
                f"got {task.get('model_type')!r}"
            )
        python = task.get("python")
        if python is not None:
            python = _freeze_declared_path(
                python,
                base_dir=base_dir,
                label=f"{label}: python",
                must_exist=True,
            )
        name = str(task.get("name") or f"{calc_type}-{index + 1}")
        if not name.strip():
            raise ValueError(f"{label}: task display name must not be empty")

        options = task.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError(f"{label}: 'options' must be an object")
        cleaned = _coerce_task_options(
            options,
            calc_type=calc_type,
            model_type=model_type,
            label=label,
        )
        tasks.append(
            {
                "name": name,
                "python": python,
                "calc_type": calc_type,
                "structure": structure,
                "initial": initial,
                "final": final,
                "resume": resume,
                "atom_map": atom_map,
                "image_shifts": image_shifts,
                "model": model,
                "model_type": model_type,
                "task": str(task.get("task", checkpoint_layer.get("task", ""))).lower()
                or None,
                "device": str(
                    task.get("device", checkpoint_layer.get("device", "cpu"))
                ),
                "output_dir": output_dir,
                "options": cleaned,
            }
        )

    return {"max_concurrent": max_concurrent, "tasks": tasks}


def submit_task_file(mgr: JobManager, path: str | Path) -> tuple[list[str], int]:
    """Parse ``path`` and enqueue every task as a PENDING job.

    Returns ``(job_ids, max_concurrent)``.
    """
    parsed = parse_task_file(path)
    job_ids: list[str] = []
    for task in parsed["tasks"]:
        job_id = mgr.new_job_id()
        cmd = build_mlipx_command(
            calc_type=task["calc_type"],
            structure=task["structure"],
            model=task["model"],
            model_type=task["model_type"],
            task=task["task"],
            device=task["device"],
            output_dir=task["output_dir"],
            job_name=job_id,
            options=task["options"],
            python=task["python"],
            initial=task["initial"],
            final=task["final"],
            resume=task["resume"],
            atom_map=task["atom_map"],
            image_shifts=task["image_shifts"],
        )
        formula, natoms = _probe_structure(task["structure"])
        mgr.enqueue(
            job_id=job_id,
            calc_type=task["calc_type"],
            structure=task["structure"],
            formula=formula,
            natoms=natoms,
            device=task["device"],
            cmd=cmd,
            python=task["python"],
            display_name=task["name"],
        )
        job_ids.append(job_id)
    return job_ids, parsed["max_concurrent"]


def _probe_structure(structure: str) -> tuple[str, int]:
    """Best-effort formula/atom count for the jobs table (never raises)."""
    try:
        from ase.io import read

        atoms = read(structure)
        return atoms.get_chemical_formula(), len(atoms)
    except Exception:
        return "?", 0


def resolve_device_uuid(device: str) -> str | None:
    """Resolve a CUDA ordinal to the immutable physical GPU UUID.

    CPU jobs do not take an exclusive device lease. CUDA ordinals are resolved
    through ``CUDA_VISIBLE_DEVICES`` before querying ``nvidia-smi`` so two
    differently expressed ordinals cannot lease the same physical GPU.
    """
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return None
    match = re.fullmatch(r"(?:gpu|cuda)(?::(\d+))?", normalized)
    if match is None:
        raise ValueError(
            f"Cannot acquire device lease for {device!r}; expected cpu, gpu, "
            "cuda, or cuda:N"
        )
    local_ordinal = int(match.group(1) or 0)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Cannot resolve {device!r} to a GPU UUID; nvidia-smi failed"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(
            f"Cannot resolve {device!r} to a GPU UUID: nvidia-smi {detail}"
        )

    inventory: list[VisibleGpu] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1]:
            raise RuntimeError(f"Malformed nvidia-smi GPU inventory line: {line!r}")
        inventory.append(
            VisibleGpu(len(inventory), fields[1], int(fields[0]), "", None)
        )
    visible = visible_gpus(inventory)
    if local_ordinal >= len(visible):
        raise RuntimeError(
            f"CUDA device {device!r} is not present in CUDA_VISIBLE_DEVICES"
        )
    return visible[local_ordinal].uuid


class QueueScheduler:
    """Poll the job directory and launch queued (PENDING) jobs.

    Concurrency is capped at ``max_concurrent`` RUNNING jobs; each launched
    job is executed by :mod:`mlipx.job_worker` under the interpreter recorded
    on the job, so queued tasks can mix engines with different virtual
    environments.
    """

    def __init__(
        self,
        jobs_dir: str | Path | None = None,
        max_concurrent: int = 1,
        poll_interval: float = 5.0,
        device_uuid_resolver: Callable[[str], str | None] = resolve_device_uuid,
        stale_claim_seconds: float = 60.0,
    ):
        self.mgr = JobManager(jobs_dir)
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int):
            raise TypeError("max_concurrent must be an integer")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = max_concurrent
        self.poll_interval = float(poll_interval)
        if not math.isfinite(self.poll_interval) or self.poll_interval <= 0:
            raise ValueError("poll_interval must be finite and > 0")
        self.device_uuid_resolver = device_uuid_resolver
        self.stale_claim_seconds = float(stale_claim_seconds)
        if not math.isfinite(self.stale_claim_seconds) or self.stale_claim_seconds <= 0:
            raise ValueError("stale_claim_seconds must be finite and > 0")
        self._stop_requested = False

    def request_stop(self) -> None:
        """Ask the scheduler to exit after the current poll cycle."""
        self._stop_requested = True

    def _reap_stale_running(self) -> None:
        """Mark RUNNING jobs whose worker process is gone as FAILED.

        Without this, a worker that dies without updating its job state
        (kill -9, backend crash, OOM killer) leaves the job stuck in RUNNING
        forever and permanently occupies a concurrency slot, blocking the
        whole queue.

        Two cases are covered:
        * workers spawned by *this* scheduler process -- reaped via waitpid
          so they cannot linger as zombies;
        * workers spawned by another process (legacy JobManager.submit) --
          detected with a liveness probe on their recorded PID.
        """
        # 1) Reap our own exited worker children (zombies would otherwise
        #    keep the PID "alive" for the probe below).
        while True:
            try:
                reaped = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break  # no children at all
            if reaped[0] == 0:
                break  # no child has exited yet
            for job in self.mgr.list_jobs():
                if job.get("status") == "running" and job.get("pid") == reaped[0]:
                    self.mgr.update_status(
                        job["job_id"],
                        JobStatus.FAILED,
                        expected_statuses={JobStatus.RUNNING},
                        expected_pid=reaped[0],
                        error=(
                            "Worker process exited unexpectedly "
                            f"(status {reaped[1]})"
                        ),
                    )
                    break
        # 2) RUNNING jobs whose recorded PID no longer exists (workers that
        #    were spawned by a different process than this scheduler).
        for job in self.mgr.list_jobs():
            if job.get("status") != "running":
                continue
            pid = job.get("pid")
            if not self.mgr.running_process_matches(job):
                self.mgr.update_status(
                    job["job_id"],
                    JobStatus.FAILED,
                    expected_statuses={JobStatus.RUNNING},
                    expected_pid=pid,
                    error="Worker process exited unexpectedly",
                )

    def run_once(self) -> int:
        """Launch as many queued jobs as the concurrency limit allows.

        Returns the number of jobs started in this pass.
        """
        self.mgr.recover_stale_claims(max_age_seconds=self.stale_claim_seconds)
        self._reap_stale_running()
        # Pausing the queue is deliberately different from stopping the
        # scheduler: existing RUNNING workers continue, while PENDING jobs
        # remain untouched until the queue is resumed.
        if queue_paused(self.mgr.jobs_dir):
            return 0
        launched = 0
        active = self.mgr.count_by_status("running") + self.mgr.count_by_status(
            "claimed"
        )
        while active < self.max_concurrent:
            # Re-check between launches so a pause request cannot cause a
            # second pending job to start after the first one in this pass.
            if queue_paused(self.mgr.jobs_dir):
                break
            claimed_job: dict[str, Any] | None = None
            claim_token: str | None = None
            for job in self.mgr.pending_jobs():
                job_id = job["job_id"]
                cmd = job.get("cmd") or []
                if not cmd:
                    self.mgr.update_status(
                        job_id,
                        JobStatus.FAILED,
                        expected_statuses={JobStatus.PENDING},
                        error="Job has no command recorded",
                    )
                    continue
                try:
                    device_uuid = self.device_uuid_resolver(
                        str(job.get("device", "cpu"))
                    )
                except Exception as exc:
                    self.mgr.update_status(
                        job_id,
                        JobStatus.FAILED,
                        expected_statuses={JobStatus.PENDING},
                        error=f"Could not acquire device lease: {exc}",
                    )
                    continue
                token = self.mgr.new_job_id()
                try:
                    claimed_job = self.mgr.claim_job(
                        job_id,
                        claim_token=token,
                        owner_pid=os.getpid(),
                        device_uuid=device_uuid,
                    )
                except ValueError as exc:
                    self.mgr.update_status(
                        job_id,
                        JobStatus.FAILED,
                        expected_statuses={JobStatus.PENDING},
                        error=f"Job must be resubmitted with an internal UUID: {exc}",
                    )
                    continue
                if claimed_job is not None:
                    claim_token = token
                    break
            if claimed_job is None or claim_token is None:
                break

            job_id = claimed_job["job_id"]
            cmd = claimed_job["cmd"]
            try:
                proc = self.mgr._spawn_worker(job_id, cmd, claim_token=claim_token)
            except OSError as exc:
                self.mgr.update_status(
                    job_id,
                    JobStatus.FAILED,
                    expected_statuses={JobStatus.CLAIMED},
                    claim_token=claim_token,
                    error=f"Could not spawn worker: {exc}",
                )
                continue
            if not self.mgr.mark_running(job_id, proc.pid, claim_token=claim_token):
                self.mgr._kill_process(proc.pid)
                self.mgr.update_status(
                    job_id,
                    JobStatus.FAILED,
                    expected_statuses={JobStatus.CLAIMED},
                    claim_token=claim_token,
                    error="Could not promote claimed job to running",
                )
                continue
            active += 1
            launched += 1
        return launched

    def _run_loop(self, stop_file: str | Path | None = None) -> None:
        stop_path = Path(stop_file) if stop_file else None
        while not self._stop_requested:
            if stop_path is not None and stop_path.exists():
                break
            self.run_once()
            time.sleep(self.poll_interval)

    def run_forever(
        self,
        stop_file: str | Path | None = None,
        *,
        inherited_lock: BinaryIO | None = None,
    ) -> None:
        """Poll while holding the process-wide singleton scheduler lock."""
        if inherited_lock is not None:
            self._run_loop(stop_file)
            return
        lock_path = scheduler_lock_file(self.mgr.jobs_dir)
        try:
            handle = lock_path.open("a+b")
            _lock_file(handle, blocking=False)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError("A scheduler is already running") from exc
        pid_file = scheduler_pid_file(self.mgr.jobs_dir)
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()).encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
            _atomic_write_text(pid_file, str(os.getpid()))
            self._run_loop(stop_file)
        finally:
            if _scheduler_lock_owner(self.mgr.jobs_dir) == os.getpid():
                pid_file.unlink(missing_ok=True)
            _unlock_file(handle)
            handle.close()


def scheduler_pid_file(jobs_dir: str | Path) -> Path:
    """PID file for a background scheduler, next to the jobs directory."""
    return Path(jobs_dir).parent / "scheduler.pid"


def scheduler_lock_file(jobs_dir: str | Path) -> Path:
    """OS-lock backing file used to guarantee one scheduler process."""
    return Path(jobs_dir).parent / "scheduler.lock"


def scheduler_pause_file(jobs_dir: str | Path) -> Path:
    """Persistent control file that pauses only pending queue dispatch."""
    return Path(jobs_dir).parent / "scheduler.paused"


def queue_paused(jobs_dir: str | Path) -> bool:
    """Return whether pending jobs are currently prevented from launching."""
    return scheduler_pause_file(jobs_dir).exists()


def pause_scheduler(jobs_dir: str | Path) -> bool:
    """Pause dispatching PENDING jobs without affecting RUNNING workers."""
    path = scheduler_pause_file(jobs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.open("x", encoding="utf-8").close()
    except FileExistsError:
        return False
    return True


def resume_scheduler(jobs_dir: str | Path) -> bool:
    """Resume dispatching PENDING jobs. Returns whether it was paused."""
    path = scheduler_pause_file(jobs_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def pause_pending_job(jobs_dir: str | Path, job_id: str) -> bool:
    """Pause one queued job without affecting other pending or running jobs."""
    return JobManager(jobs_dir).pause_pending(job_id)


def resume_paused_job(jobs_dir: str | Path, job_id: str) -> bool:
    """Resume one paused job so it can re-enter normal FIFO dispatch."""
    return JobManager(jobs_dir).resume_paused(job_id)


def start_scheduler(
    jobs_dir: str | Path,
    max_concurrent: int = 1,
    poll_interval: float = 5.0,
) -> int:
    """Launch a detached background scheduler; returns its PID."""
    if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int):
        raise TypeError("max_concurrent must be an integer")
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")
    poll_interval = float(poll_interval)
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("poll_interval must be finite and > 0")
    if os.name == "nt":  # pragma: no cover - POSIX HPC path is authoritative
        raise RuntimeError(
            "Reliable background scheduler lock handoff is unavailable on Windows; "
            "use 'mlipx queue start --foreground'."
        )
    pid_file = scheduler_pid_file(jobs_dir)
    lock_path = scheduler_lock_file(jobs_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+b")
    try:
        _lock_file(lock_handle, blocking=False)
    except BlockingIOError as exc:
        lock_handle.close()
        existing = (
            pid_file.read_text(encoding="utf-8").strip() if pid_file.exists() else "?"
        )
        raise RuntimeError(
            f"A scheduler is already running (PID {existing}); "
            "stop it with 'mlipx queue stop' first."
        ) from exc
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlipx.queue_daemon",
                str(Path(jobs_dir).resolve()),
                "--max-concurrent",
                str(max_concurrent),
                "--poll",
                str(poll_interval),
                "--lock-fd",
                str(lock_handle.fileno()),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(lock_handle.fileno(),),
        )
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(str(proc.pid).encode("ascii"))
        lock_handle.flush()
        os.fsync(lock_handle.fileno())
        _atomic_write_text(pid_file, str(proc.pid))
    except Exception:
        if proc is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        _unlock_file(lock_handle)
        lock_handle.close()
        raise
    # Do not call LOCK_UN: the daemon inherited this same locked open-file
    # description. Closing only the parent's descriptor leaves the singleton
    # lock held until the daemon exits.
    lock_handle.close()
    return proc.pid


def _atomic_write_text(path: Path, text: str) -> None:
    """Durably replace a small control file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _scheduler_lock_held(jobs_dir: str | Path) -> bool:
    path = scheduler_lock_file(jobs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        _lock_file(handle, blocking=False)
    except BlockingIOError:
        handle.close()
        return True
    _unlock_file(handle)
    handle.close()
    return False


def _scheduler_lock_owner(jobs_dir: str | Path) -> int | None:
    try:
        text = scheduler_lock_file(jobs_dir).read_text(encoding="ascii").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def stop_scheduler(jobs_dir: str | Path) -> bool:
    """Stop a background scheduler (SIGTERM). Returns True if one was running."""
    pid_file = scheduler_pid_file(jobs_dir)
    if not _scheduler_lock_held(jobs_dir):
        pid_file.unlink(missing_ok=True)
        return False
    if not pid_file.exists():
        return False
    pid_text = pid_file.read_text(encoding="utf-8").strip()
    if not pid_text.isdigit():
        return False
    pid = int(pid_text)
    if _scheduler_lock_owner(jobs_dir) != pid:
        return False
    pid_file.unlink(missing_ok=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    return True


def scheduler_status(jobs_dir: str | Path) -> dict[str, Any]:
    """Whether a background scheduler is alive, paused, and its PID."""
    pid_file = scheduler_pid_file(jobs_dir)
    if not _scheduler_lock_held(jobs_dir) or not pid_file.exists():
        return {"running": False, "pid": None, "paused": queue_paused(jobs_dir)}
    pid_text = pid_file.read_text(encoding="utf-8").strip()
    if not pid_text.isdigit():
        return {"running": False, "pid": None, "paused": queue_paused(jobs_dir)}
    pid = int(pid_text)
    running = _scheduler_lock_owner(jobs_dir) == pid and _pid_alive(pid)
    return {
        "running": running,
        "pid": pid if running else None,
        "paused": queue_paused(jobs_dir),
    }
