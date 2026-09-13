"""Shared infrastructure for the mliport beta scientific validation suite.

Every result record produced by this suite carries ``mliport.beta-validation-
result/2`` (migrated in memory from ``/1`` by the canonical evidence loader)
and identifies the calculation independently of folder names:
model identity and SHA-256, task/head, dtype, backend/framework versions,
GPU identity (hashed UUID), input-structure hash, parameters, metrics and
timing.  The suite revision is bumped whenever formulas, selection rules,
thresholds, status semantics or model identities change, so prior evidence
is never silently overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from ase import Atoms

# The canonical evidence layer lives in validation/science/evidence/.  Every
# suite script imports common.py first, so adding the science root here makes
# `evidence` importable everywhere without duplicating identity or schema
# logic (task book PR-A: one interpretation entry point).
_SCIENCE_ROOT = Path(__file__).resolve().parent.parent
if str(_SCIENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCIENCE_ROOT))

from evidence.constants import (  # noqa: E402
    ALLOWED_WRAPPER_MODULES,
    BETA_VALIDATION_SUITE_REVISION,
    DATA_MANIFEST_SCHEMA,
    MODEL_MANIFEST_SCHEMA,
    RESULT_SCHEMA,
    RESULT_SCHEMA_V1,
    RESULT_SCHEMAS,
    STATUSES,
    SUMMARY_SCHEMA,
    SUMMARY_SCHEMA_V1,
    VOLATILE_RECORD_FIELDS,
)
from evidence.identity import (  # noqa: E402
    canonical_json,
    device_class,
    migrate_record,
    new_run_id,
    profile_id_for,
    profile_identity,
    record_fingerprint,
    record_id_for,
    seed_from,
)

EV_A3_TO_GPA = 160.21766208


def _canonical_json(payload: Any) -> str:
    """Backward-compatible alias of the canonical evidence encoder."""
    return canonical_json(payload)


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

        gpus = tf.config.list_logical_devices("GPU")
        if not gpus:
            return None  # CPU-only process: no VRAM to report
        info = tf.config.experimental.get_memory_info(gpus[0].name)
        return int(info["peak"] / 2**20)
    except (ImportError, RuntimeError, KeyError, ValueError):
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
        if idx == 0 and not dev.startswith("cuda"):
            # CPU inference was requested; the process may still see GPUs
            # but they did not run the calculator.  Keep the record honest.
            return {
                "requested": requested,
                "actual": "cpu",
                "gpu_name": None,
                "gpu_uuid_hash": None,
            }
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
    if idx == 0 and not dev.startswith("cuda"):
        # CPU inference was requested; keep the record honest.
        return {
            "requested": requested,
            "actual": "cpu",
            "gpu_name": None,
            "gpu_uuid_hash": None,
        }
    if dev.startswith("cuda:") and dev[5:].isdigit():
        idx = int(dev[5:])
    idx = min(idx, len(gpus) - 1)
    try:
        details = tf.config.experimental.get_device_details(gpus[idx])
    except (AttributeError, ValueError, RuntimeError):
        details = {}
    uuid = details.get("uuid")
    uuid_hash = (
        hashlib.sha256(str(uuid).encode("utf-8")).hexdigest()[:16] if uuid else None
    )
    name = details.get("device_name")
    return {
        "requested": requested,
        "actual": f"cuda:{idx}",
        "gpu_name": name,
        "gpu_uuid_hash": uuid_hash,
    }


class InferenceProbe:
    """Force one real backend inference per repetition and count the calls.

    ASE caches results on the calculator: ``Atoms.get_potential_energy()``
    or ``get_forces()`` on unchanged coordinates is a cache hit and performs
    no new inference, so a repeatability loop built on them measures nothing
    (review R02).  The probe clears ``calculator.results`` before every
    repetition and evaluates through the calculator's own ASE ``calculate``
    entry point, which forces a real backend evaluation without perturbing
    any coordinate.  It wraps that entry point (when the calculator permits
    instance attributes) so the number of real calls is recorded and a cache
    hit cannot masquerade as a zero-noise floor.

    Backend-internal caches (e.g. the GRACE neighbour list) are *not*
    invalidated here -- that is the backend's own, separately recorded
    setting, and this class never claims otherwise.
    """

    def __init__(self, atoms: Atoms) -> None:
        if getattr(atoms, "calc", None) is None:
            msg = "InferenceProbe requires an attached calculator"
            raise ValueError(msg)
        self.atoms = atoms
        self.calculator = atoms.calc
        self.calls = 0
        self.wrapped_calculate = False
        self._original_calculate: Any = None
        self._had_instance_calculate = False

    def _counting_calculate(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._original_calculate(*args, **kwargs)

    def start(self) -> InferenceProbe:
        calculate = getattr(self.calculator, "calculate", None)
        if callable(calculate):
            namespace = getattr(self.calculator, "__dict__", None)
            self._had_instance_calculate = isinstance(namespace, dict) and (
                "calculate" in namespace
            )
            try:
                self.calculator.calculate = self._counting_calculate
            except (AttributeError, TypeError):
                # Slotted/immutable calculator: fall back to counting our own
                # evaluations, and report that the count is not verified.
                self.wrapped_calculate = False
            else:
                self._original_calculate = calculate
                self.wrapped_calculate = True
        return self

    def stop(self) -> None:
        if self.wrapped_calculate and self._original_calculate is not None:
            try:
                if self._had_instance_calculate:
                    self.calculator.calculate = self._original_calculate
                else:
                    del self.calculator.calculate
            except (AttributeError, TypeError):  # pragma: no cover - defensive
                pass
            self.wrapped_calculate = False
            self._original_calculate = None

    def __enter__(self) -> InferenceProbe:
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def run(self) -> tuple[float, np.ndarray, np.ndarray | None]:
        """Perform exactly one real inference and return copied observables."""
        import numpy as np

        results = getattr(self.calculator, "results", None)
        if isinstance(results, dict):
            results.clear()
        energy = float(self.atoms.get_potential_energy())
        forces = np.array(self.atoms.get_forces(), dtype=float, copy=True)
        try:
            stress = np.array(self.atoms.get_stress(), dtype=float, copy=True)
        except Exception:  # noqa: BLE001 - stress may be unsupported
            stress = None
        if not self.wrapped_calculate:
            self.calls += 1
        return energy, forces, stress


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
            "only real mliport backend wrappers may populate validation results"
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
    inference_mode: str | None = None,
    seed: Any = None,
    run_id: str | None = None,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Build one ``mliport.beta-validation-result/2`` record.

    Identity is explicit and additive: ``profile_id`` names the model
    profile, ``record_id`` the intended computation and ``run_id`` this
    execution attempt.  ``campaign_id`` (env ``MLIPORT_VALIDATION_CAMPAIGN``)
    groups attempts that belong to one validation campaign when the caller
    provides one.
    """
    if status not in STATUSES:
        msg = f"invalid status {status!r}; must be one of {STATUSES}"
        raise ValueError(msg)
    try:
        mliport_version = package_version("mliport")
    except Exception:  # noqa: BLE001 - version probe must never abort a run
        mliport_version = "unknown"
    resolved_seed = seed if seed is not None else seed_from({"parameters": parameters})
    resolved_campaign = campaign_id or os.environ.get("MLIPORT_VALIDATION_CAMPAIGN")
    software_commit, software_commit_source = _resolved_software_commit()
    record: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "suite_revision": BETA_VALIDATION_SUITE_REVISION,
        "git_commit": software_commit,
        "software_commit_source": software_commit_source,
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
        "mliport_version": mliport_version,
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
        "inference_mode": inference_mode,
        "seed": resolved_seed,
        "campaign_id": resolved_campaign,
    }
    record["profile_id"] = profile_id_for(record)
    record["record_id"] = record_id_for(record)
    record["run_id"] = run_id or new_run_id()
    return record


#: Explicit software-commit override.  A validation harness run at commit X
#: may legitimately analyse trajectories produced by product software at an
#: earlier commit Y; the record must then carry Y (the validated software
#: revision) while the report records X as validation_code_commit.  The
#: override is explicit, validated and recorded, never a silent fallback.
SOFTWARE_COMMIT_ENV = "MLIPORT_SOFTWARE_COMMIT"


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


def _resolved_software_commit() -> tuple[str, str]:
    """Return ``(commit, source)`` for the software revision being validated."""
    override = os.environ.get(SOFTWARE_COMMIT_ENV)
    if override:
        value = override.strip().lower()
        if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(
                f"{SOFTWARE_COMMIT_ENV} must be a 40-character hex commit sha, "
                f"got {override!r}"
            )
        return value, "env_override"
    commit = _git_commit()
    return commit, "unknown" if commit == "unknown" else "git_head"


class ResultCollisionError(RuntimeError):
    """A different record already occupies the target result path."""


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def record_json(record: dict[str, Any]) -> str:
    """Serialise a record, refusing to emit NaN/Infinity as JSON numbers."""
    try:
        return json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as exc:
        msg = f"refusing to write a result record with non-finite numbers: {exc}"
        raise ValueError(msg) from exc


def write_result(
    record: dict[str, Any],
    out_dir: str | Path,
    tag: str | None = None,
    *,
    version_on_collision: bool = True,
) -> Path:
    """Write one result record as JSON under ``out_dir``.

    Existing records are immutable: when a *different* record would land on
    an occupied name (same case/test/engine/dtype/tag), the new attempt is
    published as an explicitly versioned ``...__run-<id>.json`` sibling and
    the conflict is reported on stderr.  Identical content is written only
    once, so deterministic reruns do not multiply files.
    """
    if record.get("schema") != RESULT_SCHEMA:
        msg = (
            f"write_result only writes {RESULT_SCHEMA} records; got "
            f"{record.get('schema')!r} (migrate legacy records first)"
        )
        raise ValueError(msg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{record['case_id']}__{record['test_id']}__{record['engine']}"
    dtype = record.get("dtype")
    if dtype and dtype not in ("native", "upstream/model-defined"):
        name += f"-{dtype}"
    if tag:
        name += f"-{tag}"
    # Path-safety: slugs are built from free-form fields; an embedded path
    # separator would otherwise make the record unwritable (or misplace it).
    name = name.replace("/", "_").replace(os.sep, "_")
    path = out / f"{name}.json"
    payload = record_json(record)
    if not path.exists():
        _atomic_write_text(path, payload)
        return path

    existing_text = path.read_text(encoding="utf-8")
    if existing_text == payload:
        return path
    try:
        existing = json.loads(existing_text)
    except json.JSONDecodeError:
        existing = None
    if isinstance(existing, dict) and record_fingerprint(
        existing
    ) == record_fingerprint(record):
        return path
    if not version_on_collision:
        msg = (
            f"result collision at {path}: a different record already exists "
            f"(existing record_id={existing.get('record_id') if isinstance(existing, dict) else 'unreadable'}); "
            f"refusing to overwrite"
        )
        raise ResultCollisionError(msg)

    run = str(record.get("run_id") or new_run_id())[:8]
    for _ in range(8):
        versioned = out / f"{name}__run-{run}.json"
        if not versioned.exists():
            _atomic_write_text(versioned, payload)
            print(
                f"[mliport-validation] result collision at {path.name}: kept the "
                f"existing record and published this attempt as {versioned.name}",
                file=sys.stderr,
            )
            return versioned
        if versioned.read_text(encoding="utf-8") == payload:
            return versioned
        run = new_run_id()[:8]
    msg = f"could not publish a versioned result next to {path} after 8 attempts"
    raise ResultCollisionError(msg)


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
