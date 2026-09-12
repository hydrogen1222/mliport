"""Analysis task dispatch, result directories, status, and provenance."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import shutil
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mlipx.analysis.dataset import TrajectoryDataset
from mlipx.analysis.schema import AnalysisRequest
from mlipx.analysis.validation import InvalidTrajectoryError, UnsupportedAnalysisError

if TYPE_CHECKING:
    from typing import Any


# Analysis output revisions are part of the request/cache identity. Bump the
# changed scientific tasks while deliberately leaving the native MSD revision
# untouched.
_TASK_OUTPUT_REVISIONS = {"msd": 5, "transport": 4, "electrolyte": 3}
_TASK_SCIENTIFIC_REVISIONS = {"transport": 4, "electrolyte": 3}


def _canonical_npy_sha256(array: np.ndarray) -> str:
    """Deterministic content hash of an array (npy bytes, no zip timestamps)."""
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _store_array(array: np.ndarray, *, base_dir: Path, key: str) -> dict[str, Any]:
    """Write one oversized array to NPZ and return its JSON reference.

    The contract is explicit: ``path`` (relative), ``shape``, ``dtype`` and a
    content ``sha256`` are all recorded, and :func:`load_stored_array` reads
    the exact same array back (review array-export contract).
    """
    arr = np.ascontiguousarray(array)
    digest = _canonical_npy_sha256(arr)
    relative = Path("arrays") / f"{digest[:16]}.npz"
    target = base_dir / relative
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".tmp-{digest[:16]}-{uuid.uuid4().hex[:8]}.npz")
        try:
            np.savez(tmp, array=arr)
            os.replace(tmp, target)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
    return {
        "stored_separately": True,
        "format": "npz",
        "key": "array",
        "path": relative.as_posix(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": digest,
        "sha256_covers": "canonical npy bytes",
        "source_key": key,
    }


def load_stored_array(reference: dict[str, Any], base_dir: Path) -> np.ndarray:
    """Round-trip loader for a ``stored_separately`` array reference."""
    base = Path(base_dir).expanduser().resolve()
    try:
        path = (base / str(reference["path"])).resolve()
    except KeyError as exc:
        raise ValueError("array reference has no path") from exc
    if base != path and base not in path.parents:
        raise ValueError(f"array path escapes the analysis directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"stored array is missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        key = str(reference.get("key", "array"))
        if key not in data:
            raise ValueError(f"stored array {path} has no key {key!r}")
        array = np.asarray(data[key])
    expected_shape = reference.get("shape")
    if expected_shape is not None and list(array.shape) != list(expected_shape):
        raise ValueError(
            f"stored array shape {list(array.shape)} does not match the "
            f"reference {expected_shape}"
        )
    expected_digest = reference.get("sha256")
    if expected_digest is not None and _canonical_npy_sha256(array) != expected_digest:
        raise ValueError(f"stored array content hash mismatch for {path}")
    return array


def _jsonable(
    value: Any,
    *,
    array_limit: int = 2000,
    array_store: Path | None = None,
    key_prefix: str = "array",
) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= array_limit:
            return _jsonable(
                value.tolist(),
                array_limit=array_limit,
                array_store=array_store,
                key_prefix=key_prefix,
            )
        if array_store is None:
            return {
                "stored_separately": False,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "note": "array exceeds the inline limit and no store was available",
            }
        return _store_array(value, base_dir=array_store, key=key_prefix)
    if isinstance(value, np.generic):
        return _jsonable(
            value.item(),
            array_limit=array_limit,
            array_store=array_store,
            key_prefix=key_prefix,
        )
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(
                item,
                array_limit=array_limit,
                array_store=array_store,
                key_prefix=f"{key_prefix}.{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(
                item,
                array_limit=array_limit,
                array_store=array_store,
                key_prefix=f"{key_prefix}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    return value


def _write_json(path: Path, value: Any) -> None:
    """Atomically publish one JSON document (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _jsonable(value, array_store=path.parent),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _read_json_safe(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Sidecar files the loader actually consumes; their content identity is part
#: of the analysis fingerprint (review R06: metadata-only changes must
#: invalidate a cached result).
_RUN_METADATA_FILES = (
    "artifacts.json",
    "resolved_config.json",
    "run.json",
    "mlipx_results.json",
    "neb_results.json",
    "run_context.json",
    "raw/md.csv",
    "md.csv",
)


def _run_dir_for(source: Path) -> Path | None:
    """Directory whose sidecar metadata belongs to this trajectory, if any."""
    if source.is_dir():
        return source
    parent = source.parent
    if parent.name in {"raw", "vasp"}:
        return parent.parent
    if any((parent / name).is_file() for name in _RUN_METADATA_FILES):
        return parent
    return None


def _file_identity(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        return {"path": str(path), "exists": False, "error": str(exc)}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def source_fingerprint(path: Path) -> dict[str, Any]:
    """Content identity of the trajectory plus every analysis-relevant sidecar.

    Path/size/mtime alone let a same-size metadata edit (or an in-place
    rewrite that preserves mtime granularity) reuse a stale result; the
    streamed SHA-256 and the sidecar hashes make the identity content-based
    (review R06).
    """
    resolved = path.expanduser().resolve()
    run_dir = _run_dir_for(resolved)
    if resolved.is_dir():
        for candidate in (
            resolved / "raw" / "trajectory.traj",
            resolved / "trajectory.traj",
            resolved / "vasp" / "XDATCAR",
            resolved / "XDATCAR",
        ):
            if candidate.is_file():
                resolved = candidate
                break
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    fingerprint: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(resolved),
        "method": "path+size+mtime+sha256 plus sidecar content hashes",
    }
    metadata: dict[str, Any] = {}
    if run_dir is not None:
        for name in _RUN_METADATA_FILES:
            candidate = run_dir / name
            if candidate.is_file():
                metadata[name] = _file_identity(candidate)
    fingerprint["run_metadata"] = metadata
    return fingerprint


def _versions() -> dict[str, str]:
    packages = {}
    for distribution in (
        "mlipx",
        "numpy",
        "scipy",
        "matplotlib",
        "ase",
        "kinisi",
        "gemdat",
    ):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    return packages


def _analysis_id(request: dict[str, Any]) -> str:
    canonical = json.dumps(
        _jsonable(request, array_limit=100_000),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _output_root(source: Path) -> Path:
    resolved = source.expanduser().resolve()
    if resolved.is_dir():
        return resolved / "analysis"
    if resolved.parent.name in {"raw", "vasp"}:
        return resolved.parent.parent / "analysis"
    return resolved.parent / "analysis"


@contextmanager
def _index_lock(root: Path):
    """Serialise read-modify-write of index.json across processes."""
    lock_path = root / ".index.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        locked = False
        if os.name != "nt":  # pragma: no cover - POSIX CI hosts
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        try:
            yield
        finally:
            if locked:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _update_index(root: Path, analysis_id: str, record: dict[str, Any]) -> None:
    index_path = root / "index.json"
    with _index_lock(root):
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            index = {"schema": "mlipx.analysis-index/2", "jobs": {}}
        index.setdefault("jobs", {})[analysis_id] = _jsonable(record)
        _write_json(index_path, index)


def _new_attempt_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _publish_attempt(
    output_dir: Path, attempt_id: str, documents: dict[str, Any]
) -> Path:
    """Atomically publish an immutable attempt directory and move the pointer.

    The attempt is fully written under a hidden temporary directory and then
    renamed into place, so a concurrent reader can only ever observe a
    complete attempt (review R06).
    """
    attempts_dir = output_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    final = attempts_dir / attempt_id
    tmp = attempts_dir / f".tmp-{attempt_id}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for name, document in documents.items():
            _write_json(tmp / name, document)
        os.replace(tmp, final)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _write_json(
        attempts_dir / "latest.json",
        {
            "schema": "mlipx.analysis-attempt-pointer/1",
            "attempt_id": attempt_id,
            "status": documents["attempt.json"]["status"],
            "finished_at": documents["attempt.json"]["finished_at"],
        },
    )
    return final


def _write_columns(path: Path, columns: dict[str, Any]) -> None:
    """Write 1-D columns to CSV and every higher-rank array to NPZ + JSON.

    A 2-D coordinate array can never be represented faithfully by a flat CSV
    column, so it is stored through the same NPZ contract as oversized JSON
    arrays and referenced from ``<name>.arrays.json`` (review array export).
    """
    arrays = {key: np.asarray(value) for key, value in columns.items()}
    usable = {key: value for key, value in arrays.items() if value.ndim == 1}
    extra = {key: value for key, value in arrays.items() if value.ndim != 1}
    if usable:
        lengths = {len(value) for value in usable.values()}
        if len(lengths) != 1:
            raise ValueError(f"CSV columns for {path.name} have unequal lengths")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(usable)
            writer.writerows(zip(*usable.values(), strict=True))
    if extra:
        manifest = {
            "schema": "mlipx.analysis-array-manifest/1",
            "source": path.name,
            "arrays": {
                key: _store_array(value, base_dir=path.parent, key=key)
                for key, value in extra.items()
            },
        }
        _write_json(path.with_name(f"{path.name}.arrays.json"), manifest)


def _write_transport_summary(path: Path, result: dict[str, Any]) -> None:
    """Write a one-row human-readable transport summary CSV.

    Collective-conductivity columns are appended only when that analysis ran;
    absent fields are left blank rather than filled with fake zeros.
    """

    tracer = result["tracer_diffusion"]
    lag = tracer["lag_grid"]
    d_post = tracer["D_posterior_m2_s"]
    ne = result["nernst_einstein"]
    sigma = ne["sigma_NE_tracer_posterior_mS_cm"]
    semantics = result["kinisi_position_semantics"]
    fields: dict[str, Any] = {
        "mobile_species": result["mobile_species"],
        "dimensions": result["dimensions"],
        "temperature_K": result["temperature_mean_K"],
        "fit_start_ps": tracer["fit_start_ps"],
        "fit_stop_ps": tracer["fit_stop_ps"],
        "lag_grid_mode": lag["mode"],
        "lag_step_ps": lag["requested_step_ps"],
        "lag_stop_ps": lag["requested_stop_ps"],
        "n_lag_points_total": lag["n_lag_points_total"],
        "n_lag_points_in_fit": lag["n_lag_points_in_fit"],
        "D_mean_m2_s": d_post["mean"],
        "D_std_m2_s": d_post["std"],
        "D_median_m2_s": d_post["median"],
        "D_ci95_low_m2_s": d_post["credible_interval_95"][0],
        "D_ci95_high_m2_s": d_post["credible_interval_95"][1],
        "Dtr_mean_m2_s": d_post["mean"],
        "Dtr_ci95_low_m2_s": d_post["credible_interval_95"][0],
        "Dtr_ci95_high_m2_s": d_post["credible_interval_95"][1],
        "sigma_NE_mean_mS_cm": sigma["mean"],
        "sigma_NE_std_mS_cm": sigma["std"],
        "sigma_NE_median_mS_cm": sigma["median"],
        "sigma_NE_ci95_low_mS_cm": sigma["credible_interval_95"][0],
        "sigma_NE_ci95_high_mS_cm": sigma["credible_interval_95"][1],
        "sigma_NE_tracer_mean_S_m": ne["sigma_NE_tracer_posterior_S_m"]["mean"],
        "sigma_NE_tracer_ci95_low_S_m": ne["sigma_NE_tracer_posterior_S_m"][
            "credible_interval_95"
        ][0],
        "sigma_NE_tracer_ci95_high_S_m": ne["sigma_NE_tracer_posterior_S_m"][
            "credible_interval_95"
        ][1],
        "kinisi_version": tracer["kinisi_version"],
        "random_seed": tracer["random_seed"],
        "positions_convention": semantics["source_positions_convention"],
        "kinisi_backend_reconstruction": semantics["backend_reconstruction"],
    }
    if "collective_conductivity" in result:
        coll = result["collective_conductivity"].get(
            "sigma_collective_mS_cm_posterior",
            result["collective_conductivity"].get(
                "sigma_collective_posterior_mS_cm", {}
            ),
        )
        coll = coll or {}
        fields.update(
            {
                "sigma_collective_mean_mS_cm": coll["mean"],
                "sigma_collective_std_mS_cm": coll["std"],
                "sigma_collective_median_mS_cm": coll["median"],
                "sigma_collective_ci95_low_mS_cm": coll["credible_interval_95"][0],
                "sigma_collective_ci95_high_mS_cm": coll["credible_interval_95"][1],
            }
        )
        coll_s = result["collective_conductivity"].get(
            "sigma_collective_posterior_S_m", {}
        )
        dsigma = result["collective_conductivity"].get("D_sigma_posterior_m2_s", {})
        dsigma_cm = result["collective_conductivity"].get("D_sigma_posterior_cm2_s", {})
        haven = result.get("haven_ratio") or {}
        correlation = result.get("correlation_factor") or {}
        fields.update(
            {
                "sigma_collective_mean_S_m": coll_s.get("mean"),
                "sigma_collective_ci95_low_S_m": (
                    coll_s.get("credible_interval_95") or [None, None]
                )[0],
                "sigma_collective_ci95_high_S_m": (
                    coll_s.get("credible_interval_95") or [None, None]
                )[1],
                "D_sigma_mean_m2_s": dsigma.get("mean"),
                "Dsigma_mean_m2_s": dsigma.get("mean"),
                "D_sigma_ci95_low_m2_s": (
                    dsigma.get("credible_interval_95") or [None, None]
                )[0],
                "D_sigma_ci95_high_m2_s": (
                    dsigma.get("credible_interval_95") or [None, None]
                )[1],
                "Dsigma_mean_cm2_s": dsigma_cm.get("mean"),
                "Haven_point_estimate": haven.get("point_estimate"),
                "Haven": haven.get("point_estimate"),
                "Haven_ci95_low": (haven.get("posterior") or {}).get(
                    "credible_interval_95", [None, None]
                )[0],
                "Haven_ci95_high": (haven.get("posterior") or {}).get(
                    "credible_interval_95", [None, None]
                )[1],
                "correlation_factor_point_estimate": correlation.get("point_estimate"),
                "correlation_factor": correlation.get("point_estimate"),
                "correlation_factor_ci95_low": (correlation.get("posterior") or {}).get(
                    "credible_interval_95", [None, None]
                )[0],
                "correlation_factor_ci95_high": (
                    correlation.get("posterior") or {}
                ).get("credible_interval_95", [None, None])[1],
                "ratio_uncertainty_semantics": (haven.get("uncertainty_semantics")),
            }
        )
    else:
        fields.update(
            {
                "sigma_collective_mean_mS_cm": None,
                "sigma_collective_mean_S_m": None,
                "D_sigma_mean_m2_s": None,
                "Dsigma_mean_m2_s": None,
                "Haven_point_estimate": None,
                "Haven": None,
                "correlation_factor_point_estimate": None,
                "correlation_factor": None,
            }
        )
    if "jump_diffusion" in result:
        jump = result["jump_diffusion"].get("D_J_posterior_m2_s", {})
        fields.update(
            {
                "D_J_mean_m2_s": jump.get("mean"),
                "D_J_ci95_low_m2_s": (jump.get("credible_interval_95") or [None, None])[
                    0
                ],
                "D_J_ci95_high_m2_s": (
                    jump.get("credible_interval_95") or [None, None]
                )[1],
            }
        )
    fields.update(
        {
            "temperature_source": result.get("temperature_source"),
            "drift_reference": (result.get("drift_correction") or {}).get("mode"),
            "dimensions": result.get("dimensions"),
            "fit_start_ps": tracer.get("fit_start_ps"),
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields.keys())
        writer.writerow(["" if value is None else value for value in fields.values()])


def _load_dataset(request: AnalysisRequest) -> TrajectoryDataset:
    parameters = request.parameters
    return TrajectoryDataset.load(
        request.source,
        positions_convention=parameters.pop("positions_convention", None),
        frame_interval_fs=parameters.pop("frame_interval_fs", None),
    )


def _dispatch(request: AnalysisRequest, output_dir: Path) -> tuple[Any, list[str]]:
    task = request.task
    parameters = dict(request.parameters)
    request_copy = AnalysisRequest(task, request.source, parameters, request.force)
    artifacts: list[str] = []
    if task == "arrhenius":
        from mlipx.analysis.arrhenius import fit_arrhenius
        from mlipx.analysis.plots import plot_arrhenius

        result = fit_arrhenius(**parameters)
        _write_columns(
            output_dir / "arrhenius.csv",
            {
                "temperature_K": result["temperatures_K"],
                "inverse_temperature_K^-1": result["inverse_temperature_K^-1"],
                "diffusivity_m2_s": result["diffusivities_m2_s"],
                "ln_diffusivity": result["ln_diffusivity"],
                "ln_diffusivity_fit": result["ln_diffusivity_fit"],
            },
        )
        artifacts.append("arrhenius.csv")
        artifacts.extend(
            path.name for path in plot_arrhenius(result, output_dir / "arrhenius")
        )
        return result, artifacts

    dataset = _load_dataset(request_copy)
    if task == "validate":
        from mlipx.analysis.validation import validate_trajectory

        return validate_trajectory(dataset).to_dict(), artifacts
    if task == "thermo":
        from mlipx.analysis.plots import plot_thermo
        from mlipx.analysis.thermo import thermodynamic_diagnostics

        result = thermodynamic_diagnostics(dataset, **parameters)
        _write_columns(output_dir / "thermo.csv", result["columns"])
        artifacts.append("thermo.csv")
        artifacts.extend(
            path.name for path in plot_thermo(result, output_dir / "thermo")
        )
        return result, artifacts
    if task == "rdf":
        from mlipx.analysis.plots import plot_rdf
        from mlipx.analysis.structure import radial_distribution

        result = radial_distribution(dataset, **parameters)
        _write_columns(
            output_dir / "rdf.csv",
            {
                "r_A": result["r_A"],
                "g_center_neighbor": result["g_center_neighbor"],
                "coordination_number_center_neighbor": result[
                    "coordination_number_center_neighbor"
                ],
                "ordered_neighbor_counts": result["ordered_neighbor_counts"],
            },
        )
        artifacts.append("rdf.csv")
        artifacts.extend(path.name for path in plot_rdf(result, output_dir / "rdf"))
        return result, artifacts
    if task == "rmsd":
        from mlipx.analysis.structure import periodic_rmsd_rmsf

        result = periodic_rmsd_rmsf(dataset, **parameters)
        _write_columns(
            output_dir / "rmsd.csv",
            {
                "time_ps": result["time_ps"],
                "periodic_displacement_rmsd_A": result["periodic_displacement_rmsd_A"],
            },
        )
        _write_columns(
            output_dir / "rmsf.csv",
            {
                "atom_index": result["atom_indices"],
                "periodic_displacement_rmsf_A": result["periodic_displacement_rmsf_A"],
            },
        )
        artifacts.extend(("rmsd.csv", "rmsf.csv"))
        return result, artifacts
    if task == "msd":
        from mlipx.analysis.msd import calculate_msd
        from mlipx.analysis.plots import plot_msd, plot_msd_alpha

        result = calculate_msd(dataset, **parameters)
        columns = {
            "lag_time_ps": result["lag_time_ps"],
            "time_origin_counts": result["time_origin_counts"],
            "msd_x_A2": result["msd_x_A2"],
            "msd_y_A2": result["msd_y_A2"],
            "msd_z_A2": result["msd_z_A2"],
        }
        for axes, values in result["msd_by_axes_A2"].items():
            columns[f"msd_{axes}_A2"] = values
            columns[f"alpha_{axes}"] = result["log_log_alpha_by_axes"][axes]
        _write_columns(output_dir / "msd.csv", columns)
        fits = list(result["diagnostic_linear_diffusion_fits"].values())
        if fits:
            _write_columns(
                output_dir / "diffusion_fits.csv",
                {
                    "axes": [fit["axes"] for fit in fits],
                    "dimensions": [fit["dimensions"] for fit in fits],
                    "fit_start_ps": [fit["fit_start_ps"] for fit in fits],
                    "fit_stop_ps": [fit["fit_stop_ps"] for fit in fits],
                    "actual_fit_start_ps": [fit["actual_fit_start_ps"] for fit in fits],
                    "actual_fit_stop_ps": [fit["actual_fit_stop_ps"] for fit in fits],
                    "fit_points": [fit["fit_points"] for fit in fits],
                    "slope_A2_ps": [fit["slope_A2_ps"] for fit in fits],
                    "intercept_A2": [fit["intercept_A2"] for fit in fits],
                    "r_squared": [fit["r_squared"] for fit in fits],
                    "self_diffusion_coefficient_m2_s": [
                        fit["self_diffusion_coefficient_m2_s"] for fit in fits
                    ],
                    "self_diffusion_coefficient_cm2_s": [
                        fit["self_diffusion_coefficient_cm2_s"] for fit in fits
                    ],
                    "mean_log_log_alpha_in_fit": [
                        fit["mean_log_log_alpha_in_fit"] for fit in fits
                    ],
                    "diffusive_regime_warning": [
                        fit["diffusive_regime_warning"] for fit in fits
                    ],
                    "estimator": [fit["estimator"] for fit in fits],
                    "publication_grade": [fit["publication_grade"] for fit in fits],
                },
            )
        _write_json(
            output_dir / "diagnostics.json",
            {
                "unwrap": result["unwrap_diagnostics"],
                "fits": result["diagnostic_linear_diffusion_fits"],
            },
        )
        artifacts.extend(("msd.csv", "diagnostics.json"))
        if fits:
            artifacts.append("diffusion_fits.csv")
        artifacts.extend(path.name for path in plot_msd(result, output_dir / "msd"))
        artifacts.extend(
            path.name for path in plot_msd_alpha(result, output_dir / "alpha")
        )
        return result, artifacts
    if task == "density":
        from mlipx.analysis.structure import density_map

        result = density_map(dataset, **parameters)
        np.savez(
            output_dir / "density.npz",
            occupancy_probability=result["occupancy_probability"],
            number_density_A3=result["number_density_A^-3"],
            counts_per_voxel=result["counts_per_voxel"],
            cell_A=result["cell_A"],
        )
        artifacts.append("density.npz")
        return result, artifacts
    if task == "transport":
        from mlipx.analysis.plots import (
            plot_transport,
            plot_transport_mscd,
            plot_transport_mstd,
        )
        from mlipx.analysis.transport import kinisi_transport

        result = kinisi_transport(dataset, **parameters)
        arrays = {
            "lag_time_ps": result["lag_time_ps"],
            "msd_A2": result["kinisi_msd_A2"],
            "msd_variance_A4": result["kinisi_msd_variance_A4"],
            "D_tracer_samples_m2_s": result["D_tracer_samples_m2_s"],
            "sigma_NE_samples_S_m": result["sigma_NE_samples_S_m"],
        }
        if "kinisi_mscd" in result:
            arrays.update(
                {
                    "mscd": result["kinisi_mscd"],
                    "mscd_variance": result["kinisi_mscd_variance"],
                    "sigma_collective_samples_S_m": result[
                        "sigma_collective_samples_S_m"
                    ],
                    "D_sigma_samples_m2_s": result["D_sigma_samples_m2_s"],
                }
            )
            if "haven_ratio_samples" in result:
                arrays["haven_ratio_samples"] = result["haven_ratio_samples"]
        if "kinisi_mstd" in result:
            arrays.update(
                {
                    "mstd": result["kinisi_mstd"],
                    "mstd_variance": result["kinisi_mstd_variance"],
                    "D_J_samples_m2_s": result["D_J_samples_m2_s"],
                }
            )
        np.savez_compressed(
            output_dir / "kinisi_arrays.npz",
            **arrays,
        )
        artifacts.append("kinisi_arrays.npz")
        _write_transport_summary(output_dir / "transport_summary.csv", result)
        artifacts.append("transport_summary.csv")
        _write_json(
            output_dir / "diagnostics.json",
            {
                "quality": result.get("quality"),
                "warnings": result.get("warnings", []),
                "kinisi_position_semantics": result.get("kinisi_position_semantics"),
                "kinisi_time_mapping": result.get("kinisi_time_mapping"),
                "kinisi_resource_diagnostics": result.get(
                    "kinisi_resource_diagnostics"
                ),
                "drift_correction": result.get("drift_correction"),
                "uncertainty_semantics": {
                    "tracer": result["nernst_einstein"].get("uncertainty_semantics"),
                    "collective": (result.get("collective_conductivity") or {}).get(
                        "uncertainty_semantics"
                    ),
                    "ratios": (result.get("haven_ratio") or {}).get(
                        "uncertainty_semantics"
                    ),
                },
            },
        )
        artifacts.append("diagnostics.json")
        artifacts.extend(
            path.name for path in plot_transport(result, output_dir / "transport_msd")
        )
        if "kinisi_mscd" in result:
            artifacts.extend(
                path.name
                for path in plot_transport_mscd(result, output_dir / "transport_mscd")
            )
        if "kinisi_mstd" in result:
            artifacts.extend(
                path.name
                for path in plot_transport_mstd(result, output_dir / "transport_mstd")
            )
        return result, artifacts
    if task == "electrolyte":
        from mlipx.analysis.electrolyte import gemdat_electrolyte
        from mlipx.analysis.plots import (
            plot_electrolyte_density,
            plot_electrolyte_distribution,
            plot_electrolyte_paths,
        )

        result = gemdat_electrolyte(dataset, **parameters)
        np.savez_compressed(output_dir / "electrolyte_arrays.npz", **result.arrays)
        artifacts.append("electrolyte_arrays.npz")
        summary_fields = {
            key: value
            for key, value in result.summary.items()
            if np.isscalar(value) and not isinstance(value, (dict, list, tuple))
        }
        _write_columns(
            output_dir / "electrolyte_summary.csv",
            {key: [value] for key, value in summary_fields.items()},
        )
        artifacts.append("electrolyte_summary.csv")
        for matrix_name in ("transition_matrix", "jump_matrix"):
            if matrix_name in result.arrays:
                matrix_path = output_dir / f"{matrix_name}.csv"
                np.savetxt(
                    matrix_path, np.asarray(result.arrays[matrix_name]), delimiter=","
                )
                artifacts.append(matrix_path.name)
        artifacts.extend(
            path.name
            for path in plot_electrolyte_density(
                result.arrays, output_dir / "density_projection"
            )
        )
        artifacts.extend(
            path.name
            for path in plot_electrolyte_paths(
                {"paths": result.paths}, output_dir / "free_energy_paths"
            )
        )
        artifacts.extend(
            path.name
            for path in plot_electrolyte_distribution(
                {"table": result.tables.get("residence_times")},
                output_dir / "residence_time_distribution",
                value_field="residence_time_ps",
                title="Residence-time distribution",
                xlabel="Residence time (ps)",
                warning_sink=result.warnings,
            )
        )
        artifacts.extend(
            path.name
            for path in plot_electrolyte_distribution(
                {"table": result.tables.get("jumps")},
                output_dir / "jump_distance_distribution",
                value_field="jump_distance_A",
                title="Jump-distance distribution",
                xlabel="Jump distance (A)",
                warning_sink=result.warnings,
            )
        )
        artifacts.extend(
            path.name
            for path in plot_electrolyte_distribution(
                {"table": result.tables.get("jump_rates")},
                output_dir / "jump_rate_segments",
                value_field="jump_rate_s^-1",
                title="Jump-rate distribution by site pair",
                xlabel="Jump rate (s^-1)",
                warning_sink=result.warnings,
            )
        )
        for name, table in result.tables.items():
            path = output_dir / f"{name}.csv"
            if hasattr(table, "to_csv"):
                table.to_csv(path, index=False)
            else:
                values = np.asarray(table)
                if values.size == 0:
                    path.write_text("empty\n", encoding="utf-8")
                else:
                    np.savetxt(path, values, delimiter=",")
            artifacts.append(path.name)
        for name, structure in result.structures.items():
            path = output_dir / f"{name}.cif"
            structure.to(filename=str(path))
            artifacts.append(path.name)
        for axis, values in result.paths.items():
            path = output_dir / f"percolation_{axis}.csv"
            _write_columns(path, values)
            artifacts.append(path.name)
        # Explicit discovery artifacts have stable names in addition to the
        # generic structures, making exploratory site generation auditable.
        if "detected_sites" in result.structures:
            path = output_dir / "detected_sites.cif"
            result.structures["detected_sites"].to(filename=str(path))
            artifacts.append(path.name)
        if "occupancy_sites" in result.structures:
            path = output_dir / "occupancy_sites.cif"
            result.structures["occupancy_sites"].to(filename=str(path))
            artifacts.append(path.name)
        _write_json(
            output_dir / "diagnostics.json",
            {"warnings": result.warnings, "summary": result.summary},
        )
        artifacts.append("diagnostics.json")
        return {"summary": result.summary, "warnings": result.warnings}, artifacts
    if task in {"vacf", "spectrum"}:
        from mlipx.analysis.plots import plot_spectrum, plot_vacf
        from mlipx.analysis.spectral import calculate_vacf, velocity_spectrum

        spectrum_parameters = {}
        if task == "spectrum":
            spectrum_parameters = {
                key: parameters.pop(key)
                for key in ("taper", "normalization")
                if key in parameters
            }
        vacf = calculate_vacf(dataset, **parameters)
        _write_columns(
            output_dir / "vacf.csv",
            {
                "lag_time_fs": vacf["lag_time_fs"],
                "vacf_raw_A2_fs2": vacf["vacf_raw_A2_fs2"],
                "vacf_normalized": vacf["vacf_normalized"],
            },
        )
        artifacts.append("vacf.csv")
        artifacts.extend(path.name for path in plot_vacf(vacf, output_dir / "vacf"))
        if task == "vacf":
            return vacf, artifacts
        spectrum = velocity_spectrum(vacf, **spectrum_parameters)
        _write_columns(
            output_dir / "spectrum.csv",
            {
                "frequency_THz": spectrum["frequency_THz"],
                "frequency_cm^-1": spectrum["frequency_cm^-1"],
                "raw_spectrum": spectrum["raw_spectrum"],
                "spectrum": spectrum["spectrum"],
            },
        )
        artifacts.append("spectrum.csv")
        artifacts.extend(
            path.name for path in plot_spectrum(spectrum, output_dir / "spectrum")
        )
        return {"vacf": vacf, "spectrum": spectrum}, artifacts
    raise AssertionError(f"Unhandled analysis task: {task}")


def run_analysis(request: AnalysisRequest) -> dict[str, Any]:
    """Run one analysis job and persist its complete reproducibility record.

    Cache identity is content-based (trajectory SHA-256 plus every sidecar the
    loader consumes) and every attempt is published as an immutable record;
    a failed attempt replaces the published result with a failure document so
    an older success can never be returned as the current answer (review R06).
    """

    fingerprint = source_fingerprint(request.source_path)
    canonical_request = {
        "schema": "mlipx.analysis-request/2",
        "task": request.task,
        "source": str(request.source_path),
        "source_fingerprint": fingerprint,
        "parameters": request.parameters,
        "backend_versions": _versions(),
    }
    if request.task in _TASK_OUTPUT_REVISIONS:
        canonical_request["task_output_revision"] = _TASK_OUTPUT_REVISIONS[request.task]
    if request.task in _TASK_SCIENTIFIC_REVISIONS:
        canonical_request["task_scientific_revision"] = _TASK_SCIENTIFIC_REVISIONS[
            request.task
        ]
    analysis_id = _analysis_id(canonical_request)
    root = _output_root(request.source_path)
    output_dir = root / request.task / analysis_id
    attempts_dir = output_dir / "attempts"
    root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    existing = output_dir / "results.json"

    if existing.is_file() and not request.force:
        value = _read_json_safe(existing)
        pointer = _read_json_safe(attempts_dir / "latest.json")
        if (
            isinstance(value, dict)
            and value.get("status") == "success"
            and value.get("analysis_id") == analysis_id
            and value.get("source_fingerprint") == fingerprint
            and isinstance(pointer, dict)
            and pointer.get("status") == "success"
        ):
            # Cache hit: the published result names exactly this request and
            # fingerprint, and the latest attempt is the success that produced it.
            return {
                "analysis_id": analysis_id,
                "output_dir": str(output_dir),
                "status": "success",
                "reused": True,
                "results": value.get("results"),
            }
        # A stale/unverifiable published document must be recomputed, never
        # reused: fall through to a fresh attempt.

    attempt_id = _new_attempt_id()
    started_at = datetime.now(timezone.utc).isoformat()
    # Preserve whatever was published before as an immutable attempt artefact
    # instead of leaving it in place as a fake success (review R06).
    if existing.is_file():
        _write_json(
            attempts_dir / f"previous-results-{attempt_id}.json",
            _read_json_safe(existing),
        )
        existing.unlink(missing_ok=True)

    provenance = {
        "schema": "mlipx.analysis-provenance/2",
        "analysis_id": analysis_id,
        "request_hash": analysis_id,
        "attempt_id": attempt_id,
        "analysis_timestamp_utc": started_at,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": _versions(),
        "parameters": request.parameters,
        "source_run": str(request.source_path),
        "source_trajectory": fingerprint,
    }
    if request.task in _TASK_SCIENTIFIC_REVISIONS:
        provenance["task_scientific_revision"] = _TASK_SCIENTIFIC_REVISIONS[
            request.task
        ]
    _write_json(output_dir / "request.json", canonical_request)
    _write_json(output_dir / "provenance.json", provenance)
    record = {
        "analysis_id": analysis_id,
        "task": request.task,
        "status": "running",
        "path": str(output_dir.relative_to(root)),
        "attempt_id": attempt_id,
    }
    _update_index(root, analysis_id, record)

    try:
        result, artifacts = _dispatch(request, output_dir)
        if request.task == "transport":
            provenance["transport"] = {
                "fit_start_ps": result["tracer_diffusion"]["fit_start_ps"],
                "fit_stop_ps": result["tracer_diffusion"]["fit_stop_ps"],
                "lag_grid": result["tracer_diffusion"]["lag_grid"],
                "kinisi_position_semantics": result["kinisi_position_semantics"],
                "temperature_source": result.get("temperature_source"),
                "drift_correction": result.get("drift_correction"),
                "dimensions": result.get("dimensions"),
                "random_seed": result["tracer_diffusion"].get("random_seed"),
                "collective_system_particles": (
                    result.get("collective_conductivity") or {}
                ).get("system_particles"),
                "jump_diffusion": "jump_diffusion" in result,
            }
            _write_json(output_dir / "provenance.json", provenance)
        elif request.task == "electrolyte":
            summary = result.get("summary", {})
            provenance["electrolyte"] = {
                "analysis_phase": summary.get("analysis_phase", "production"),
                "time_source": summary.get("time_source"),
                "temperature_source": summary.get("temperature_source"),
                "position_convention": summary.get("position_convention"),
                "drift_correction": summary.get("drift_correction"),
                "site_source": summary.get("site_source"),
                "jump_dimensions": summary.get("jump_dimensions"),
                "percolation_axes": summary.get("percolation_axes"),
            }
            _write_json(output_dir / "provenance.json", provenance)
        finished_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema": "mlipx.analysis-results/2",
            "analysis_id": analysis_id,
            "attempt_id": attempt_id,
            "status": "success",
            "task": request.task,
            "source_fingerprint": fingerprint,
            "results": result,
            "artifacts": sorted(set(artifacts)),
            "finished_at": finished_at,
        }
        _write_json(existing, payload)
        _publish_attempt(
            output_dir,
            attempt_id,
            {
                "attempt.json": {
                    "schema": "mlipx.analysis-attempt/1",
                    "attempt_id": attempt_id,
                    "analysis_id": analysis_id,
                    "task": request.task,
                    "status": "success",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "source_fingerprint": fingerprint,
                    "artifacts": payload["artifacts"],
                    "error": None,
                },
                "request.json": canonical_request,
                "provenance.json": provenance,
                "results.json": payload,
            },
        )
        record.update(status="success", artifacts=payload["artifacts"])
        _update_index(root, analysis_id, record)
        return {
            "analysis_id": analysis_id,
            "output_dir": str(output_dir),
            "status": "success",
            "reused": False,
            "results": _jsonable(result, array_store=output_dir),
        }
    except KeyboardInterrupt as exc:
        _publish_failed_attempt(
            output_dir,
            attempts_dir,
            existing,
            canonical_request,
            analysis_id,
            attempt_id,
            request,
            fingerprint,
            started_at,
            status="cancelled",
            exc=exc,
        )
        record.update(status="cancelled", error=str(exc))
        _update_index(root, analysis_id, record)
        raise
    except Exception as exc:
        status = (
            "unsupported"
            if isinstance(exc, (InvalidTrajectoryError, UnsupportedAnalysisError))
            else "failed"
        )
        _publish_failed_attempt(
            output_dir,
            attempts_dir,
            existing,
            canonical_request,
            analysis_id,
            attempt_id,
            request,
            fingerprint,
            started_at,
            status=status,
            exc=exc,
        )
        record.update(status=status, error=str(exc))
        _update_index(root, analysis_id, record)
        raise


def _publish_failed_attempt(
    output_dir: Path,
    attempts_dir: Path,
    results_path: Path,
    canonical_request: dict[str, Any],
    analysis_id: str,
    attempt_id: str,
    request: AnalysisRequest,
    fingerprint: dict[str, Any],
    started_at: str,
    *,
    status: str,
    exc: BaseException,
) -> None:
    """Replace the published result with the failure and keep the attempt.

    The old success (if any) was already preserved as a ``previous-results``
    artefact, so a later non-force call can never resurrect it (review R06).
    """
    finished_at = datetime.now(timezone.utc).isoformat()
    error = {
        "schema": "mlipx.analysis-error/2",
        "analysis_id": analysis_id,
        "attempt_id": attempt_id,
        "status": status,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "finished_at": finished_at,
    }
    failure = {
        "schema": "mlipx.analysis-results/2",
        "analysis_id": analysis_id,
        "attempt_id": attempt_id,
        "status": status,
        "task": request.task,
        "source_fingerprint": fingerprint,
        "results": None,
        "artifacts": [],
        "error": error["message"],
        "finished_at": finished_at,
    }
    _write_json(results_path, failure)
    _write_json(output_dir / "error.json", error)
    _publish_attempt(
        output_dir,
        attempt_id,
        {
            "attempt.json": {
                "schema": "mlipx.analysis-attempt/1",
                "attempt_id": attempt_id,
                "analysis_id": analysis_id,
                "task": request.task,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "source_fingerprint": fingerprint,
                "artifacts": [],
                "error": error["message"],
            },
            "request.json": canonical_request,
            "error.json": error,
        },
    )
