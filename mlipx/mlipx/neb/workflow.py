"""Public NEB workflow: provenance, checkpoints, resume, and exports."""

from __future__ import annotations

import hashlib
import os
import platform
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.constraints import FixAtoms

from mlipx.neb.io import (
    NEBCheckpointStore,
    atomic_write_json,
    canonical_fingerprint,
    export_vasp_band,
    load_checkpoint,
    resolve_checkpoint_path,
)
from mlipx.neb.prepare import BandInput, prepare_band
from mlipx.neb.revisions import (
    NEB_CHECKPOINT_SCHEMA_REVISION,
    NEB_SCIENTIFIC_REVISION,
)
from mlipx.neb.schema import NEBOptions, NEBPreparationError
from mlipx.protocols import CancellationRequested
from mlipx.queue import resolve_device_uuid
from mlipx.runners.neb import NEBRunner

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from ase import Atoms

    from mlipx.base_calculator import BaseMLIPCalculator
    from mlipx.config.resolver import ResolvedConfig


_BACKEND_DISTRIBUTIONS = {
    "uma": "fairchem-core",
    "fairchem": "fairchem-core",
    "mace": "mace-torch",
    "dpa": "deepmd-kit",
    "grace": "tensorpotential",
}


def _framework_distributions(
    model_type: str, model_path: str | Path
) -> tuple[str, ...]:
    if model_type == "dpa":
        return (
            ("tensorflow",) if Path(model_path).suffix.lower() == ".pb" else ("torch",)
        )
    if model_type == "grace":
        return ("tensorflow",)
    return ("torch",)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_model(path: str | Path) -> str:
    model = Path(path).expanduser().resolve()
    if not model.exists():
        raise FileNotFoundError(f"Model path not found: {model}")
    digest = hashlib.sha256()
    if model.is_file():
        with model.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not model.is_dir():
        raise NEBPreparationError(f"Unsupported model path type: {model}")
    files = sorted(path for path in model.rglob("*") if path.is_file())
    if not files:
        raise NEBPreparationError(f"Model directory contains no files: {model}")
    for child in files:
        relative = child.relative_to(model).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _model_record(
    calculator: BaseMLIPCalculator,
    resolved: ResolvedConfig,
) -> dict[str, Any]:
    info = dict(calculator.info())
    model_type = resolved.model_type.lower()
    reported_model_type = info.get("model_type")
    if (
        reported_model_type is not None
        and str(reported_model_type).lower() != model_type
    ):
        raise NEBPreparationError(
            f"Calculator reports model_type={reported_model_type!r}, but resolved "
            f"configuration requires {model_type!r}"
        )
    reported_task = info.get("task")
    if (
        reported_task is not None
        and str(reported_task).lower() != resolved.task.lower()
    ):
        raise NEBPreparationError(
            f"Calculator reports task={reported_task!r}, but resolved configuration "
            f"requires {resolved.task!r}"
        )
    reported_path = info.get("model_path")
    if (
        reported_path is not None
        and Path(reported_path).expanduser().resolve()
        != Path(resolved.model_path).expanduser().resolve()
    ):
        raise NEBPreparationError(
            "Calculator model path differs from the resolved model path"
        )
    backend_distribution = _BACKEND_DISTRIBUTIONS[model_type]
    backend_version = info.get("backend_version") or _distribution_version(
        backend_distribution
    )
    if not backend_version:
        raise NEBPreparationError(
            f"Cannot determine exact backend version for {backend_distribution!r}"
        )
    framework_versions = dict(info.get("framework_versions") or {})
    if not framework_versions:
        for distribution in _framework_distributions(model_type, resolved.model_path):
            installed = _distribution_version(distribution)
            if installed is not None:
                framework_versions[distribution] = installed
    if not framework_versions:
        raise NEBPreparationError(
            f"Cannot determine an exact framework version for {model_type!r}"
        )
    calculator_options = dict(resolved.calculator_options)
    dtype_requested = calculator_options.get("default_dtype")
    dtype_effective = info.get("default_dtype")
    if dtype_effective is None and info.get("model_precision"):
        dtype_effective = info["model_precision"]
    head_requested = calculator_options.get("head")
    head_effective = info.get("active_head", info.get("head", head_requested))
    source = resolved.sources.get("model_path")
    model_alias = None
    if source is not None and source.source.startswith("model alias "):
        model_alias = source.source.removeprefix("model alias ").strip("'")
    return {
        "model_type": model_type,
        "actual_device_type": info.get("actual_device_type", "unknown"),
        "actual_device_logical_index": info.get("actual_device_logical_index"),
        "actual_device_uuid": info.get("actual_device_uuid"),
        "inference_mode": info.get(
            "inference_mode",
            calculator.inference_mode
            if hasattr(calculator, "inference_mode")
            else "default",
        ),
        "compile_enabled": info.get("compile_enabled"),
        "direct_forces": info.get("direct_forces"),
        "model_path": str(Path(resolved.model_path).expanduser().resolve()),
        "model_sha256": _hash_model(resolved.model_path),
        "model_alias": model_alias,
        "model_source": source.source if source is not None else None,
        "backend_distribution": backend_distribution,
        "backend_version": str(backend_version),
        "framework_versions": framework_versions,
        "task": resolved.task,
        "head_requested": head_requested,
        "head_effective": head_effective,
        "dtype_requested": dtype_requested,
        "dtype_effective": dtype_effective,
        "backend_reported_device": info.get(
            "actual_device", info.get("device", resolved.device)
        ),
    }


def _restore_checkpoint_model_source(
    model: dict[str, Any], checkpoint: Mapping[str, Any] | None
) -> None:
    """Preserve the original model alias provenance on later attempts."""
    if checkpoint is None:
        return
    recorded = checkpoint.get("resolved_config")
    sources = recorded.get("sources") if isinstance(recorded, Mapping) else None
    source_record = sources.get("model_path") if isinstance(sources, Mapping) else None
    source = source_record.get("source") if isinstance(source_record, Mapping) else None
    if not isinstance(source, str):
        return
    model["model_source"] = source
    if source.startswith("model alias "):
        model["model_alias"] = source.removeprefix("model alias ").strip("'")


def _fixed_indices(atoms: Atoms) -> list[int]:
    fixed: set[int] = set()
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            raise NEBPreparationError(
                f"Unsupported NEB constraint in workflow: {type(constraint).__name__}"
            )
        fixed.update(int(index) for index in constraint.get_indices())
    return sorted(fixed)


def _atom_ids(atoms: Atoms) -> list[Any] | None:
    for key in ("atom_id", "atom_ids"):
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key]).tolist()
        if key in atoms.info:
            return np.asarray(atoms.info[key]).tolist()
    return None


def _identity(atoms: Atoms) -> dict[str, Any]:
    return {
        "atomic_numbers": np.asarray(atoms.numbers, dtype=int).tolist(),
        "symbols": atoms.get_chemical_symbols(),
        "masses_amu": np.asarray(atoms.get_masses(), dtype=float).tolist(),
        "atom_ids": _atom_ids(atoms),
    }


def _original_final_identity(band: BandInput) -> dict[str, Any]:
    """Recover the user-supplied final ordering from the mapped band endpoint."""
    mapped = _identity(band.final)
    inverse = np.argsort(band.atom_map)
    return {
        key: None if values is None else np.asarray(values)[inverse].tolist()
        for key, values in mapped.items()
    }


def _options_dict(options: NEBOptions) -> dict[str, Any]:
    return {item.name: getattr(options, item.name) for item in fields(options)}


def _prepare_endpoint(atoms: Atoms, resolved: ResolvedConfig, *, label: str) -> Atoms:
    """Apply explicit electronic state and enforce task/PBC semantics."""
    prepared = atoms.copy()
    run_options = resolved.run_options
    if "charge" in run_options:
        prepared.info["charge"] = run_options["charge"]
    if "spin" in run_options:
        prepared.info["spin"] = run_options["spin"]

    task = resolved.task.lower()
    pbc = np.asarray(prepared.pbc, dtype=bool)
    periodic_tasks = {"omat", "oc20", "oc25", "odac", "omc", "bulk"}
    molecular_tasks = {"omol", "molecule"}
    if task in periodic_tasks:
        if not pbc.all():
            raise NEBPreparationError(
                f"{label} PBC {pbc.tolist()} is incompatible with periodic "
                f"task {task!r}"
            )
    elif task in molecular_tasks:
        if pbc.any():
            raise NEBPreparationError(
                f"{label} PBC {pbc.tolist()} is incompatible with molecular "
                f"task {task!r}"
            )
        prepared.info.setdefault("charge", 0)
        if task == "omol":
            prepared.info.setdefault("spin", 1)
    else:
        raise NEBPreparationError(
            f"Unknown model task {task!r}; refusing to infer NEB PBC semantics"
        )
    cell = np.asarray(prepared.cell.array, dtype=float)
    if not np.all(np.isfinite(cell)) or abs(float(np.linalg.det(cell))) <= 1.0e-12:
        raise NEBPreparationError(
            f"{label} requires a finite full-rank fixed cell for VASP path export"
        )
    return prepared


def _electronic_state(atoms: Atoms) -> dict[str, Any]:
    return {key: atoms.info.get(key) for key in ("charge", "spin")}


def _prepare_endpoints(
    initial: Atoms, final: Atoms, resolved: ResolvedConfig
) -> tuple[Atoms, Atoms]:
    prepared_initial = _prepare_endpoint(initial, resolved, label="Initial endpoint")
    prepared_final = _prepare_endpoint(final, resolved, label="Final endpoint")
    if _electronic_state(prepared_initial) != _electronic_state(prepared_final):
        raise NEBPreparationError(
            "NEB endpoints have different charge/spin states; absolute energies "
            "from incompatible electronic states cannot define one barrier"
        )
    return prepared_initial, prepared_final


def options_from_resolved(resolved: ResolvedConfig) -> NEBOptions:
    """Translate schema-typed NEB run options into the core option dataclass."""
    values = dict(resolved.run_options)
    settings = dict(resolved.settings)
    return NEBOptions(
        n_intermediate_images=values.get("n_intermediate_images", 7),
        climb=values.get("climb", False),
        method=values.get("neb_method", "improvedtangent"),
        interpolation=values.get("neb_interpolation", "linear"),
        path_convention=values.get("path_convention", "mic"),
        spring_eV_A2=values.get("neb_spring", 0.1),
        optimizer="FIRE",
        fmax_eV_A=values.get("fmax", 0.03),
        max_steps=values.get("max_steps", 1000),
        pre_fmax_eV_A=values.get("neb_pre_fmax", 0.10),
        pre_max_steps=values.get("neb_pre_max_steps", 300),
        maxstep_A=values.get("neb_maxstep", 0.10),
        endpoint_policy=values.get("endpoint_policy", "validate"),
        endpoint_fmax_eV_A=values.get("endpoint_fmax", 0.02),
        endpoint_steps=values.get("endpoint_steps", 500),
        idpp_fmax_eV_A=values.get("idpp_fmax", 0.10),
        idpp_steps=values.get("idpp_steps", 200),
        idpp_mic=values.get("idpp_mic", True),
        min_distance_A=values.get("neb_min_distance", 0.5),
        force_abort_eV_A=values.get("fmax_abort", settings.get("fmax_abort", 20.0)),
    )


def _fingerprint(
    band: BandInput,
    options: NEBOptions,
    model: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    package_version = _distribution_version("mlipx")
    if package_version is None:
        raise NEBPreparationError("Cannot determine mlipx version for resume identity")
    value = {
        "schema": "mlipx.neb-resume-fingerprint/1",
        "software": {
            "mlipx_version": package_version,
            "neb_scientific_revision": NEB_SCIENTIFIC_REVISION,
            "neb_checkpoint_schema_revision": NEB_CHECKPOINT_SCHEMA_REVISION,
        },
        "model": {
            key: model.get(key)
            for key in (
                "model_type",
                "model_sha256",
                "backend_distribution",
                "backend_version",
                "framework_versions",
                "task",
                "head_effective",
                "dtype_effective",
                "inference_mode",
                "compile_enabled",
            )
        },
        "initial_identity": _identity(band.initial),
        "final_identity": _original_final_identity(band),
        "cell_A": np.asarray(band.initial.cell.array, dtype=float).tolist(),
        "pbc": np.asarray(band.initial.pbc, dtype=bool).tolist(),
        "constraints": {"fix_atoms": _fixed_indices(band.initial)},
        "electronic_state": _electronic_state(band.initial),
        "band": {
            "image_count": len(band.images),
            "atom_map": band.atom_map.tolist(),
            "image_shifts": band.image_shifts.tolist(),
            "path_convention": band.path_convention,
            "method": options.method,
            "spring_eV_A2": options.spring_eV_A2,
            "interpolation": options.interpolation,
        },
        "neb_options": _options_dict(options),
    }
    return canonical_fingerprint(value)


def _resume_band(checkpoint: Mapping[str, Any], images: list[Atoms]) -> BandInput:
    fingerprint = checkpoint.get("resume_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise NEBPreparationError("Checkpoint has no resume fingerprint")
    metadata = fingerprint.get("band")
    if not isinstance(metadata, Mapping):
        raise NEBPreparationError("Checkpoint fingerprint has no band identity")
    checkpoint_stage = str(checkpoint.get("stage", ""))
    source = (
        "checkpoint_prepared"
        if checkpoint_stage == "prepared"
        else "checkpoint_optimized_band"
    )
    return BandInput(
        images=tuple(images),
        atom_map=np.asarray(metadata.get("atom_map")),
        image_shifts=np.asarray(metadata.get("image_shifts")),
        path_convention=str(metadata.get("path_convention")),
        source=source,
    )


def _preflight_new_output(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise FileExistsError(f"NEB output path is not a directory: {path}")
    if path.exists() and any(child.name != ".neb-run.lock" for child in path.iterdir()):
        raise FileExistsError(
            f"NEB output directory is not empty: {path}. Choose a new run directory."
        )


@contextmanager
def _run_directory_lock(output: Path):
    """Hold a non-blocking process lock for one NEB run directory."""
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / ".neb-run.lock").open("a+b")
    acquired = False
    try:
        try:
            if os.name == "nt":  # pragma: no cover - POSIX HPC is authoritative
                import msvcrt  # noqa: PLC0415

                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (BlockingIOError, OSError) as exc:
            raise NEBPreparationError(
                f"Another NEB attempt is already writing {output}"
            ) from exc
        yield
    finally:
        if acquired and os.name == "nt":  # pragma: no cover
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def checkpoint_run_directory(path: str | Path) -> Path:
    """Return the original run directory owning a complete checkpoint."""
    checkpoint = resolve_checkpoint_path(path)
    if checkpoint.parent.name != "checkpoints":
        raise NEBPreparationError(
            f"Checkpoint is outside the required RUN/checkpoints layout: {checkpoint}"
        )
    return checkpoint.parent.parent


def _artifacts(
    *,
    status: str,
    run_id: str,
    attempt_id: str,
    latest_checkpoint: Path | None,
    result_path: Path | None = None,
    vasp_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema": "mlipx.neb-artifacts/1",
        "trajectory_kind": "neb_band",
        "status": status,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "artifacts": {
            "checkpoint": (
                None
                if latest_checkpoint is None
                else {
                    "path": str(latest_checkpoint),
                    "kind": "neb_band",
                    "geometry_resume_only": True,
                }
            ),
            "result": None if result_path is None else str(result_path),
            "vasp_path": None if vasp_path is None else str(vasp_path),
        },
    }


def _record_failed_attempt(
    output: Path,
    context: dict[str, Any],
    model: Mapping[str, Any],
    latest_checkpoint: Path | None,
    exc: Exception,
) -> None:
    """Publish a stable failure result without claiming convergence."""
    failure_status = "cancelled" if isinstance(exc, CancellationRequested) else "failed"
    failed_at = _utc_now()
    failure = {
        "schema": "mlipx.neb-results/2",
        "run_id": context["run_id"],
        "attempt_id": context["attempt_id"],
        "trajectory_kind": "neb_band",
        "status": failure_status,
        "converged": False,
        "failure_reason": str(exc),
        "energy_unit": "eV",
        "force_unit": "eV/Angstrom",
        "barrier_unit": "eV",
        "barrier_forward_sampled_eV": None,
        "barrier_reverse_sampled_eV": None,
        "barrier_forward_fitted_eV": None,
        "barrier_reverse_fitted_eV": None,
        "reaction_energy_eV": None,
        "highest_energy_image_index": None,
        "barrier_status": "not_available",
        "climb": False,
        "climbing_image_index": None,
        "max_neb_force_eV_A": None,
        "climbing_physical_fmax_eV_A": None,
        "energies_eV": None,
        "raw_physical_forces_eV_A": None,
        "constraint_applied_forces_eV_A": None,
        "neb_band_forces_eV_A": None,
        "stages": [],
        "model": model,
        "device": context["device"],
        "initial_identity": context["initial_identity"],
        "final_identity": context["final_identity"],
        "cell_A": context["cell_A"],
        "pbc": context["pbc"],
        "constraints": context["constraints"],
        "electronic_state": context["electronic_state"],
        "atom_map": context["atom_map"],
        "image_shifts": context["image_shifts"],
        "path_convention": context["path_convention"],
        "neb_options": context["neb_options"],
        "latest_checkpoint": (
            None if latest_checkpoint is None else str(latest_checkpoint)
        ),
        "vasp_path": None,
        "saddle_validation": "not_performed",
        "failed_at": failed_at,
    }
    result_path = output / "neb_results.json"
    atomic_write_json(result_path, failure)
    context.update(
        {
            "status": failure_status,
            "converged": False,
            "failure_reason": str(exc),
            "failed_at": failed_at,
        }
    )
    atomic_write_json(output / "run_context.json", context)
    atomic_write_json(
        output / "artifacts.json",
        _artifacts(
            status=failure_status,
            run_id=context["run_id"],
            attempt_id=context["attempt_id"],
            latest_checkpoint=latest_checkpoint,
            result_path=result_path,
        ),
    )


def _run_neb_workflow_locked(
    calculator: BaseMLIPCalculator,
    resolved: ResolvedConfig,
    *,
    output_dir: str | Path,
    initial: Atoms | None = None,
    final: Atoms | None = None,
    resume: str | Path | None = None,
    atom_map: Iterable[int] | None = None,
    image_shifts: Iterable[Iterable[int]] | None = None,
    verbose: bool = True,
    progress_callback=None,
    cancel_event=None,
) -> dict[str, Any]:
    """Execute one public NEB run with reproducible output and geometry resume."""
    if resolved.calc_type != "neb":
        raise ValueError("run_neb_workflow requires calc_type='neb'")
    options = options_from_resolved(resolved)
    output = Path(output_dir).expanduser().resolve()
    checkpoint_record: dict[str, Any] | None = None
    if resume is None:
        if initial is None or final is None:
            raise ValueError("Both initial and final structures are required")
        initial, final = _prepare_endpoints(initial, final, resolved)
        band = prepare_band(
            initial,
            final,
            options,
            atom_map=atom_map,
            image_shifts=image_shifts,
        )
        run_id = str(uuid.uuid4())
    else:
        if initial is not None or final is not None:
            raise ValueError("Resume accepts a checkpoint, not new endpoints")
        original_output = checkpoint_run_directory(resume)
        if output != original_output:
            raise NEBPreparationError(
                f"Resume must continue in its original run directory "
                f"{original_output}; got {output}"
            )
        checkpoint_record, images = load_checkpoint(resume)
        band = _resume_band(checkpoint_record, images)
        run_id = str(checkpoint_record.get("run_id"))
        try:
            canonical_run_id = str(uuid.UUID(run_id))
        except ValueError as exc:
            raise NEBPreparationError("Checkpoint run_id is not a UUID") from exc
        if run_id != canonical_run_id:
            raise NEBPreparationError("Checkpoint run_id is not a canonical UUID")
    attempt_id = str(uuid.uuid4())

    if resolved.device != "cpu":
        from mlipx.validation import evaluate

        probe = band.initial.copy()
        probe.calc = calculator.get_calculator()
        evaluate(probe)
    model = _model_record(calculator, resolved)
    from dataclasses import asdict
    from mlipx.capabilities import require_neb_capability, resolve_capabilities

    capability_method = getattr(calculator, "capabilities", None)
    capability = (
        capability_method(model)
        if callable(capability_method)
        else resolve_capabilities(
            model, calculator.get_calculator().implemented_properties
        )
    )
    require_neb_capability(
        capability,
        experimental=resolved.run_options.get("allow_unvalidated_neb", False),
    )
    model["capabilities"] = asdict(capability)
    _restore_checkpoint_model_source(model, checkpoint_record)
    device_uuid = resolve_device_uuid(resolved.device)
    if device_uuid is not None:
        from mlipx.devices import verify_runtime_uuid

        if model["actual_device_type"] != "cuda":
            raise NEBPreparationError("Backend did not confirm actual CUDA execution")
        verify_runtime_uuid(model["actual_device_uuid"], device_uuid)
    fingerprint, fingerprint_sha256 = _fingerprint(band, options, model)
    if checkpoint_record is not None:
        recorded_digest = checkpoint_record.get("resume_fingerprint_sha256")
        recorded_fingerprint = checkpoint_record.get("resume_fingerprint")
        if recorded_digest != fingerprint_sha256 or recorded_fingerprint != fingerprint:
            raise NEBPreparationError(
                "Resume fingerprint is incompatible with this model, backend, "
                "task/head/dtype, cell/PBC, identity, constraints, or band setup; "
                "start a new run instead"
            )

    output.mkdir(parents=True, exist_ok=True)
    mlipx_version = _distribution_version("mlipx")
    if mlipx_version is None:
        raise NEBPreparationError(
            "Cannot determine the exact mlipx package version; install mlipx "
            "before starting a provenance-bearing NEB run"
        )
    context = {
        "schema": "mlipx.neb-run-context/1",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "started_at": _utc_now(),
        "geometry_resume_only": True,
        "resumed_from": (
            None if resume is None else str(resolve_checkpoint_path(resume))
        ),
        "mlipx_version": mlipx_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model": model,
        "device": {
            "requested": resolved.device,
            "effective": "cpu" if device_uuid is None else device_uuid,
            "effective_uuid": device_uuid,
            "actual_device_type": model["actual_device_type"],
            "actual_device_logical_index": model["actual_device_logical_index"],
            "actual_device_uuid": model["actual_device_uuid"],
        },
        "status": "running",
        "converged": None,
        "failure_reason": None,
        "initial_identity": _identity(band.initial),
        "final_identity": _original_final_identity(band),
        "cell_A": np.asarray(band.initial.cell.array, dtype=float).tolist(),
        "pbc": np.asarray(band.initial.pbc, dtype=bool).tolist(),
        "constraints": {"fix_atoms": _fixed_indices(band.initial)},
        "electronic_state": _electronic_state(band.initial),
        "atom_map": band.atom_map.tolist(),
        "image_shifts": band.image_shifts.tolist(),
        "path_convention": band.path_convention,
        "neb_options": _options_dict(options),
        "resume_fingerprint": fingerprint,
        "resume_fingerprint_sha256": fingerprint_sha256,
        "resolved_config": resolved.as_dict(),
    }
    atomic_write_json(output / "run_context.json", context)
    atomic_write_json(output / "resolved_config.json", resolved.as_dict())

    latest_checkpoint: Path | None = None
    try:
        store = NEBCheckpointStore(output)
        latest_checkpoint = store.write(
            band.images,
            run_id=run_id,
            attempt_id=attempt_id,
            stage="prepared",
            stage_step=0,
            resume_fingerprint=fingerprint,
            resume_fingerprint_sha256=fingerprint_sha256,
            resolved_config=resolved.as_dict(),
        )
        atomic_write_json(
            output / "artifacts.json",
            _artifacts(
                status="running",
                run_id=run_id,
                attempt_id=attempt_id,
                latest_checkpoint=latest_checkpoint,
            ),
        )
    except Exception as exc:
        _record_failed_attempt(output, context, model, latest_checkpoint, exc)
        raise

    interval = int(resolved.run_options.get("checkpoint_interval", 10))
    last_checkpoint_key: tuple[str, int] | None = None

    def checkpoint_callback(stage: str, stage_step: int, images) -> None:
        nonlocal latest_checkpoint, last_checkpoint_key
        key = (stage, stage_step)
        if key == last_checkpoint_key:
            return
        if stage_step != 0 and stage_step % interval != 0:
            return
        latest_checkpoint = store.write(
            images,
            run_id=run_id,
            attempt_id=attempt_id,
            stage=stage,
            stage_step=stage_step,
            resume_fingerprint=fingerprint,
            resume_fingerprint_sha256=fingerprint_sha256,
            resolved_config=resolved.as_dict(),
        )
        last_checkpoint_key = key

    runner = NEBRunner(
        calculator,
        options,
        verbose=verbose,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        checkpoint_callback=checkpoint_callback,
    )
    result_path = output / "neb_results.json"
    try:
        result = runner.run(band)
        if runner.last_band is None:
            raise RuntimeError("NEB runner did not retain its final complete band")
        latest_checkpoint = store.write(
            runner.last_band,
            run_id=run_id,
            attempt_id=attempt_id,
            stage="final",
            stage_step=sum(int(stage["steps"]) for stage in result.stages),
            resume_fingerprint=fingerprint,
            resume_fingerprint_sha256=fingerprint_sha256,
            resolved_config=resolved.as_dict(),
        )
        vasp_name = "vasp_path" if resume is None else f"vasp_path_{attempt_id[:8]}"
        vasp_path = export_vasp_band(
            runner.last_band,
            output / vasp_name,
            atom_map=band.atom_map,
            image_shifts=band.image_shifts,
            path_convention=band.path_convention,
        )
        payload = result.to_dict()
        payload.update(
            {
                "run_id": run_id,
                "attempt_id": attempt_id,
                "trajectory_kind": "neb_band",
                "model": model,
                "device": context["device"],
                "initial_identity": context["initial_identity"],
                "final_identity": context["final_identity"],
                "cell_A": context["cell_A"],
                "pbc": context["pbc"],
                "constraints": context["constraints"],
                "electronic_state": context["electronic_state"],
                "atom_map": context["atom_map"],
                "image_shifts": context["image_shifts"],
                "path_convention": context["path_convention"],
                "neb_options": context["neb_options"],
                "saddle_validation": "not_performed",
                "latest_checkpoint": str(latest_checkpoint),
                "vasp_path": str(vasp_path),
                "completed_at": _utc_now(),
            }
        )
        atomic_write_json(result_path, payload)
        context.update(
            {
                "status": result.status,
                "converged": result.converged,
                "failure_reason": payload["failure_reason"],
                "completed_at": payload["completed_at"],
            }
        )
        atomic_write_json(output / "run_context.json", context)
        atomic_write_json(
            output / "artifacts.json",
            _artifacts(
                status=result.status,
                run_id=run_id,
                attempt_id=attempt_id,
                latest_checkpoint=latest_checkpoint,
                result_path=result_path,
                vasp_path=vasp_path,
            ),
        )
        return payload
    except Exception as exc:
        _record_failed_attempt(output, context, model, latest_checkpoint, exc)
        raise


def run_neb_workflow(
    calculator: BaseMLIPCalculator,
    resolved: ResolvedConfig,
    *,
    output_dir: str | Path,
    initial: Atoms | None = None,
    final: Atoms | None = None,
    resume: str | Path | None = None,
    atom_map: Iterable[int] | None = None,
    image_shifts: Iterable[Iterable[int]] | None = None,
    verbose: bool = True,
    progress_callback=None,
    cancel_event=None,
) -> dict[str, Any]:
    """Execute one NEB attempt while exclusively owning its run directory."""
    output = Path(output_dir).expanduser().resolve()
    if resume is None:
        _preflight_new_output(output)
    elif output != checkpoint_run_directory(resume):
        raise NEBPreparationError(
            "Resume output differs from the checkpoint's original run directory"
        )
    with _run_directory_lock(output):
        return _run_neb_workflow_locked(
            calculator,
            resolved,
            output_dir=output,
            initial=initial,
            final=final,
            resume=resume,
            atom_map=atom_map,
            image_shifts=image_shifts,
            verbose=verbose,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )


def checkpoint_resolved_layer(path: str | Path) -> dict[str, Any]:
    """Return a flat typed-resolver layer recorded by a resume checkpoint."""
    checkpoint, _ = load_checkpoint(path)
    recorded = checkpoint.get("resolved_config")
    if not isinstance(recorded, Mapping):
        raise NEBPreparationError("Checkpoint has no resolved configuration")
    layer = {
        key: recorded.get(key)
        for key in ("model_type", "model_path", "task", "device", "inference_mode")
    }
    for section in ("calculator_options", "run_options", "settings"):
        values = recorded.get(section, {})
        if not isinstance(values, Mapping):
            raise NEBPreparationError(
                f"Checkpoint resolved configuration has invalid {section}"
            )
        layer.update(values)
    return {key: value for key, value in layer.items() if value is not None}


__all__ = [
    "checkpoint_resolved_layer",
    "checkpoint_run_directory",
    "options_from_resolved",
    "run_neb_workflow",
]
