"""Shared infrastructure for the mlipx beta scientific validation suite.

Every result record produced by this suite carries ``mlipx.beta-validation-
result/1`` and identifies the calculation independently of folder names:
model identity and SHA-256, task/head, dtype, backend/framework versions,
GPU identity (hashed UUID), input-structure hash, parameters, metrics and
timing.  The suite revision is bumped whenever formulas, selection rules,
thresholds, status semantics or model identities change, so prior evidence
is never silently overwritten.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from ase import Atoms

BETA_VALIDATION_SUITE_REVISION = 1
RESULT_SCHEMA = "mlipx.beta-validation-result/1"
SUMMARY_SCHEMA = "mlipx.beta-validation-summary/1"
MODEL_MANIFEST_SCHEMA = "mlipx.beta-model-manifest/1"
DATA_MANIFEST_SCHEMA = "mlipx.beta-data-manifest/1"

# Status vocabulary (taskbook section 3).  ``blocked`` is an execution state,
# the rest are result states.
STATUSES = (
    "pass",
    "fail",
    "characterized",
    "insufficient_sampling",
    "unsupported",
    "blocked",
)

EV_A3_TO_GPA = 160.21766208

#: Wrapper classes that are allowed to produce four-backend science evidence.
#: Anything else (EMT, Lennard-Jones, SinglePointCalculator, ...) must never
#: populate backend results -- see taskbook section 47.
ALLOWED_WRAPPER_MODULES = (
    "mlipx.calculator",
    "mlipx.calculators.mace_calc",
    "mlipx.calculators.dpa_calc",
    "mlipx.calculators.grace_calc",
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def structure_id(atoms: Atoms) -> str:
    """Stable hash of a structure (species, positions, cell, pbc).

    Positions and cell are rounded to 1e-6 so that hashing is stable across
    processes while still distinguishing the validation fixtures.
    """
    import numpy as np

    payload = {
        "numbers": [int(z) for z in atoms.numbers],
        "positions": np.asarray(atoms.positions, dtype=float).round(9).tolist(),
        "cell": np.asarray(atoms.cell.array, dtype=float).round(9).tolist(),
        "pbc": [bool(p) for p in atoms.pbc],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def package_version(dist: str) -> str:
    try:
        return _pkg_version(dist)
    except PackageNotFoundError:
        return "not-installed"


def synchronize(device: str) -> None:
    """Block until queued work on ``device`` has completed (torch/TensorFlow)."""
    if "cuda" in str(device):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                return
        except ImportError:
            pass
        try:
            import tensorflow as tf

            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.synchronize(gpu)
        except (ImportError, AttributeError):
            pass


def peak_vram_mib() -> int | None:
    """Peak allocated VRAM on the current process' default GPU, in MiB."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated() / 2**20)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        info = tf.config.experimental.get_memory_info("GPU:0")
        return int(info["peak"] / 2**20)
    except (ImportError, RuntimeError, KeyError):
        return None


def reset_vram_counter() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def device_record(requested: str) -> dict[str, Any]:
    """Actual GPU identity for the current process (UUID hashed).

    Reads the identity from the framework that actually runs inference:
    torch when installed, otherwise tensorflow (GRACE/tensorpotential
    environments carry no torch).  Never substitutes the requested value
    for missing runtime evidence.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        torch = None

    if torch is not None:
        idx = 0
        dev = str(requested)
        if dev.startswith("cuda:") and dev[5:].isdigit():
            idx = int(dev[5:])
        if not torch.cuda.is_available():
            return {
                "requested": requested,
                "actual": "cpu",
                "gpu_name": None,
                "gpu_uuid_hash": None,
            }
        props = torch.cuda.get_device_properties(idx)
        uuid = getattr(props, "uuid", None)
        uuid_hash = (
            hashlib.sha256(str(uuid).encode("utf-8")).hexdigest()[:16]
            if uuid is not None
            else None
        )
        return {
            "requested": requested,
            "actual": f"cuda:{idx}",
            "gpu_name": props.name,
            "gpu_uuid_hash": uuid_hash,
        }

    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError:
        return {
            "requested": requested,
            "actual": "cpu",
            "gpu_name": None,
            "gpu_uuid_hash": None,
        }
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return {
            "requested": requested,
            "actual": "cpu",
            "gpu_name": None,
            "gpu_uuid_hash": None,
        }
    idx = 0
    dev = str(requested)
    if dev.startswith("cuda:") and dev[5:].isdigit():
        idx = int(dev[5:])
    idx = min(idx, len(gpus) - 1)
    try:
        details = tf.config.experimental.get_device_details(gpus[idx])
    except (AttributeError, ValueError, RuntimeError):
        details = {}
    uuid = details.get("uuid")
    uuid_hash = (
        hashlib.sha256(str(uuid).encode("utf-8")).hexdigest()[:16]
        if uuid
        else None
    )
    name = details.get("device_name")
    return {
        "requested": requested,
        "actual": f"cuda:{idx}",
        "gpu_name": name,
        "gpu_uuid_hash": uuid_hash,
    }

def timed(fn: Callable[[], Any], device: str = "cpu") -> tuple[Any, float]:
    """Run ``fn`` once and return ``(result, wall_seconds)`` with GPU sync."""
    synchronize(device)
    t0 = time.monotonic()
    out = fn()
    synchronize(device)
    return out, time.monotonic() - t0


def calculator_identity(wrapper: Any, calculator: Any) -> dict[str, Any]:
    """Prove which calculator performed inference (taskbook section 47)."""
    wrapper_mod = type(wrapper).__module__
    if wrapper_mod not in ALLOWED_WRAPPER_MODULES:
        msg = (
            f"refusing to record evidence from {wrapper_mod}.{type(wrapper).__name__}: "
            "only real mlipx backend wrappers may populate validation results"
        )
        raise RuntimeError(msg)
    return {
        "wrapper_module": wrapper_mod,
        "wrapper_class": type(wrapper).__name__,
        "calculator_module": type(calculator).__module__,
        "calculator_class": type(calculator).__name__,
    }


def result_record(
    *,
    case_id: str,
    test_id: str,
    status: str,
    engine: str,
    model_identity: str,
    model_sha256: str,
    task: str | None,
    head: str | None,
    dtype: str,
    device: dict[str, Any],
    input_structure_id: str | None,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    wall_seconds: float = 0.0,
    peak_vram: int | None = None,
    exception: str | None = None,
) -> dict[str, Any]:
    """Build one ``mlipx.beta-validation-result/1`` record."""
    if status not in STATUSES:
        msg = f"invalid status {status!r}; must be one of {STATUSES}"
        raise ValueError(msg)
    try:
        mlipx_version = package_version("mlipx")
    except Exception:  # noqa: BLE001 - version probe must never abort a run
        mlipx_version = "unknown"
    record: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "suite_revision": BETA_VALIDATION_SUITE_REVISION,
        "git_commit": _git_commit(),
        "case_id": case_id,
        "test_id": test_id,
        "status": status,
        "engine": engine,
        "model_identity": model_identity,
        "model_sha256": model_sha256,
        "task": task,
        "head": head,
        "dtype": dtype,
        "python": platform.python_version(),
        "mlipx_version": mlipx_version,
        "backend_version": None,
        "framework_version": None,
        "device": device,
        "input_structure_id": input_structure_id,
        "parameters": parameters,
        "metrics": metrics,
        "diagnostics": diagnostics or {},
        "wall_seconds": round(wall_seconds, 3),
        "peak_vram_mib": peak_vram,
        "exception": exception,
    }
    return record


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


def write_result(
    record: dict[str, Any], out_dir: str | Path, tag: str | None = None
) -> Path:
    """Write one result record as JSON under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{record['case_id']}__{record['test_id']}__{record['engine']}"
    dtype = record.get("dtype")
    if dtype and dtype not in ("native", "upstream/model-defined"):
        name += f"-{dtype}"
    if tag:
        name += f"-{tag}"
    path = out / f"{name}.json"
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_model_manifest(path: str | Path) -> dict[str, Any]:
    """Load and sanity-check the committed model manifest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != MODEL_MANIFEST_SCHEMA:
        msg = f"unexpected model manifest schema {manifest.get('schema')!r}"
        raise ValueError(msg)
    for pid, profile in manifest.get("profiles", {}).items():
        for key in ("engine", "upstream_model_id", "model_sha256"):
            if not profile.get(key):
                msg = f"model manifest profile {pid!r} missing {key!r}"
                raise ValueError(msg)
    return manifest


def resolve_profile(
    manifest: dict[str, Any], profile_id: str, model_path: str | Path
) -> dict[str, Any]:
    """Return a manifest profile after verifying the local artifact hash.

    The harness refuses to run when the actual file hash differs from the
    manifest entry (taskbook section 5); a new artifact requires a new
    manifest identity, not silent acceptance.

    Directory artifacts (e.g. extracted GRACE checkpoints) cannot match the
    manifest hash of the distributed archive; they are verified file-by-file
    via the profile's ``secondary_hashes`` entries, whose keys are the file
    names inside the extracted directory (optionally prefixed with
    ``extracted_`` to distinguish them from archive-level hashes).
    """
    profile = manifest["profiles"][profile_id]
    model_path = Path(model_path)
    if model_path.is_dir():
        secondary = profile.get("secondary_hashes") or {}
        if not secondary:
            msg = (
                f"directory artifact for profile {profile_id!r} has no "
                "secondary_hashes to verify"
            )
            raise RuntimeError(msg)
        for key, expected in sorted(secondary.items()):
            rel = key.removeprefix("extracted_")
            artifact = model_path / rel
            if not artifact.is_file():
                msg = (
                    f"directory artifact for profile {profile_id!r} is "
                    f"missing expected file {rel!r} ({artifact})"
                )
                raise RuntimeError(msg)
            actual = sha256_file(artifact)
            if actual != expected:
                msg = (
                    f"model hash mismatch for profile {profile_id!r} file "
                    f"{rel!r}: manifest {expected[:12]}... != actual "
                    f"{actual[:12]}... ({artifact}). Update the manifest "
                    "identity deliberately instead."
                )
                raise RuntimeError(msg)
        return profile
    actual = sha256_file(model_path)
    if actual != profile["model_sha256"]:
        msg = (
            f"model hash mismatch for profile {profile_id!r}: manifest "
            f"{profile['model_sha256'][:12]}... != actual {actual[:12]}... "
            f"({model_path}). Update the manifest identity deliberately instead."
        )
        raise RuntimeError(msg)
    return profile


def load_data_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != DATA_MANIFEST_SCHEMA:
        msg = f"unexpected data manifest schema {manifest.get('schema')!r}"
        raise ValueError(msg)
    return manifest
