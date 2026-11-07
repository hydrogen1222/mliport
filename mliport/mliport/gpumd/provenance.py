"""Reproducibility record for a native GPUMD run.

A native run is not reproducible from an energy alone: the exact binary, NEP
model, generated ``run.in``, structure and GPU all matter.  This module builds
one deterministic JSON payload describing them and writes it with sorted keys
and a trailing newline.  Missing optional facts (git, ``nvidia-smi``) are
recorded as ``None`` plus an explicit error string; they are never guessed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mliport.model_record import hash_model_path

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "build_gpumd_provenance",
    "write_provenance",
]

_SCHEMA = "mliport.gpumd-provenance/1"


def _mliport_version() -> str | None:
    """Installed mliport distribution version, or None when not importable."""
    try:
        return version("mliport")
    except PackageNotFoundError:
        return None


def _git_head_sha() -> str | None:
    """Best-effort ``git rev-parse HEAD`` for the source tree; never fails."""
    directory = Path(__file__).resolve().parent
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            directory = candidate
            break
    else:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            ["git", "rev-parse", "HEAD"],
            cwd=str(directory),
            shell=False,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def _nvidia_smi_uuids() -> tuple[list[str] | None, str | None]:
    """Return physical GPU UUIDs in nvidia-smi order, or an error string."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            shell=False,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return None, (
            f"nvidia-smi exited with {completed.returncode}: "
            f"{(completed.stderr or '').strip()}"
        )
    uuids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not uuids:
        return None, "nvidia-smi reported no GPUs"
    return uuids, None


def _logical_device_index(device_requested: str | None) -> tuple[int, str | None]:
    """Map an mliport device string to a logical CUDA index."""
    text = "cuda:0" if device_requested is None else str(device_requested).strip()
    lowered = text.lower()
    if lowered in {"cpu"}:
        return -1, None
    if lowered in {"cuda", "gpu"}:
        return 0, None
    if lowered.startswith("cuda:"):
        suffix = lowered.split(":", 1)[1]
        if suffix.isdigit():
            return int(suffix), None
    return 0, f"cannot interpret device_requested={device_requested!r}"


def _resolve_actual_device_uuid(
    device_requested: str | None,
    cuda_visible_devices: str | None,
) -> tuple[str | None, str | None]:
    """Resolve the visible logical GPU to a physical UUID."""
    logical_index, index_error = _logical_device_index(device_requested)
    if index_error is not None:
        return None, index_error
    if logical_index < 0:
        return None, None
    uuids, query_error = _nvidia_smi_uuids()
    if query_error is not None:
        return None, query_error
    if uuids is None:  # defensive: query_error is None only when uuids is set
        return None, query_error or "nvidia-smi returned no GPU UUIDs"

    visible = (
        cuda_visible_devices
        if cuda_visible_devices is not None
        else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if visible is None or not visible.strip():
        if logical_index >= len(uuids):
            return None, (
                f"logical CUDA index {logical_index} is outside the "
                f"{len(uuids)} physical GPU(s) reported by nvidia-smi"
            )
        return uuids[logical_index], None
    if visible.strip() in {"-1", ""}:
        return None, "CUDA_VISIBLE_DEVICES hides every GPU"
    selectors = [token.strip() for token in visible.split(",") if token.strip()]
    if logical_index >= len(selectors):
        return None, (
            f"logical CUDA index {logical_index} is outside "
            f"CUDA_VISIBLE_DEVICES={visible!r}"
        )
    selector = selectors[logical_index]
    if selector.isdigit():
        physical = int(selector)
        if physical >= len(uuids):
            return None, (
                f"CUDA_VISIBLE_DEVICES physical index {physical} is outside "
                f"the {len(uuids)} GPU(s) reported by nvidia-smi"
            )
        return uuids[physical], None
    if selector.startswith(("GPU-", "MIG-")):
        matches = [uuid for uuid in uuids if uuid.startswith(selector)]
        if len(matches) == 1:
            return matches[0], None
        return None, (
            f"CUDA_VISIBLE_DEVICES selector {selector!r} matched "
            f"{len(matches)} GPUs"
        )
    return None, f"cannot interpret CUDA_VISIBLE_DEVICES selector {selector!r}"


def build_gpumd_provenance(
    *,
    gpumd_path: str | Path,
    nep_path: str | Path,
    run_in_text: str,
    model_xyz_path: str | Path,
    device_requested: str | None = None,
    cuda_visible_devices: str | None = None,
    gpumd_identity_record: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the reproducibility record for one native GPUMD run.

    Args:
        gpumd_path: GPUMD executable that was (or will be) launched.
        nep_path: NEP model file passed to ``potential``.
        run_in_text: Exact ``run.in`` text written to disk.
        model_xyz_path: Exact ``model.xyz`` written to disk.
        device_requested: mliport device string (``cpu``/``cuda``/``cuda:N``).
        cuda_visible_devices: Effective visibility string; defaults to the
            process environment.
        gpumd_identity_record: Pre-computed :func:`gpumd_identity` result, to
            avoid executing the binary a second time.
        extra: Optional additional fields merged into the record.

    Returns:
        JSON-serialisable provenance mapping.
    """
    from mliport.gpumd.input import parse_nep_model  # noqa: PLC0415
    from mliport.gpumd.process import gpumd_identity  # noqa: PLC0415

    identity = (
        dict(gpumd_identity_record)
        if gpumd_identity_record is not None
        else gpumd_identity(gpumd_path)
    )
    nep_path_resolved = Path(nep_path).expanduser().resolve()
    nep_info = parse_nep_model(nep_path_resolved)
    model_xyz_resolved = Path(model_xyz_path).expanduser().resolve()
    effective_visible = (
        cuda_visible_devices
        if cuda_visible_devices is not None
        else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    actual_uuid, device_error = _resolve_actual_device_uuid(
        device_requested, effective_visible
    )
    payload: dict[str, Any] = {
        "schema": _SCHEMA,
        "mliport_version": _mliport_version(),
        "mliport_source_sha": _git_head_sha(),
        "gpumd": identity,
        "nep": {
            "path": str(nep_path_resolved),
            "sha256": hash_model_path(nep_path_resolved, error_class=ValueError),
            "element_types": list(nep_info["element_types"]),
            "model_format": nep_info["model_format"],
            "model_version": nep_info["model_version"],
            "cutoffs": dict(nep_info["cutoffs"]),
            "cutoff_max_A": nep_info["cutoff_max_A"],
            "raw_first_line": nep_info["raw_first_line"],
        },
        "run_in": {
            "text": run_in_text,
            "sha256": hashlib.sha256(run_in_text.encode("utf-8")).hexdigest(),
        },
        "model_xyz": {
            "path": str(model_xyz_resolved),
            "sha256": hash_model_path(model_xyz_resolved, error_class=ValueError),
        },
        "device_requested": device_requested,
        "cuda_visible_devices": effective_visible,
        "actual_device_uuid": actual_uuid,
        "device_query_error": device_error,
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        # GPUMD/V100 inference is float32.  ``precision double`` in run.in
        # only widens the *text* written by dump_xyz; it does not make the
        # model arithmetic float64.  Recorded explicitly so a trajectory is
        # never mistaken for a float64 calculation.
        "numeric_precision": "float32",
        "numeric_precision_note": (
            "GPU inference (NEP/GPUMD on V100) is float32. The 'precision "
            "double' option in dump_xyz only widens the ASCII text output; "
            "it does not change the model arithmetic."
        ),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def write_provenance(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a provenance payload as deterministic JSON.

    Args:
        path: Destination file.
        payload: Mapping produced by :func:`build_gpumd_provenance`.

    Returns:
        The destination path.

    Raises:
        ValueError: If the payload contains a non-finite float (``allow_nan``
            is false so JSON consumers never see ``NaN``).
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )
    destination.write_text(text, encoding="utf-8")
    return destination
