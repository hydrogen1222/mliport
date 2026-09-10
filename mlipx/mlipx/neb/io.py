"""Atomic full-band checkpoints and VASP path export for NEB workflows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory

from mlipx.neb.revisions import NEB_CHECKPOINT_SCHEMA_REVISION
from mlipx.neb.schema import NEBPreparationError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from ase import Atoms


_CHECKPOINT_SCHEMA = f"mlipx.neb-checkpoint/{NEB_CHECKPOINT_SCHEMA_REVISION}"
_BAND_KIND = "neb_band"
_STEP_PATTERN = re.compile(r"step_(\d{6})$")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("JSON metadata cannot contain NaN or Inf")
    return value


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - POSIX is authoritative on HPC
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _checkpoint_root_lock(root: Path):
    """Serialize sequence allocation and latest publication across processes."""
    lock_path = root / ".checkpoint.lock"
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":  # pragma: no cover - POSIX is authoritative on HPC
            import msvcrt  # noqa: PLC0415

            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl  # noqa: PLC0415

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":  # pragma: no cover
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace one JSON document without exposing partial content."""
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
            json.dump(
                _jsonable(value),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fingerprint(value: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Return JSON-normalized fingerprint data and its stable SHA-256."""
    normalized = _jsonable(value)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return normalized, hashlib.sha256(encoded).hexdigest()


def _calculator_free_band(images: Iterable[Atoms]) -> list[Atoms]:
    snapshots = []
    for index, image in enumerate(images):
        snapshot = image.copy()
        snapshot.calc = None
        snapshot.info["trajectory_kind"] = _BAND_KIND
        snapshot.info["mlipx_neb_image_index"] = index
        snapshots.append(snapshot)
    if len(snapshots) < 3:
        raise NEBPreparationError("A checkpoint requires a complete NEB band")
    return snapshots


def _constraint_indices(atoms: Atoms) -> list[int]:
    """Return frozen indices from a POSCAR-compatible constraint set."""
    fixed: set[int] = set()
    for constraint in atoms.constraints:
        getter = getattr(constraint, "get_indices", None)
        if not callable(getter):
            raise NEBPreparationError(
                "VASP path export cannot round-trip constraint "
                f"{type(constraint).__name__}"
            )
        fixed.update(int(index) for index in getter())
    return sorted(fixed)


def _validate_loaded_band(images: list[Atoms], metadata: Mapping[str, Any]) -> None:
    expected = metadata.get("image_count")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 3:
        raise NEBPreparationError("Checkpoint has an invalid image_count")
    if len(images) != expected:
        raise NEBPreparationError(
            f"Checkpoint band has {len(images)} images; metadata requires {expected}"
        )
    reference = images[0]
    numbers = np.asarray(reference.numbers, dtype=int)
    masses = np.asarray(reference.get_masses(), dtype=float)
    cell = np.asarray(reference.cell.array, dtype=float)
    pbc = np.asarray(reference.pbc, dtype=bool)
    fixed = _constraint_indices(reference)
    for index, image in enumerate(images):
        if image.info.get("trajectory_kind") != _BAND_KIND:
            raise NEBPreparationError(
                f"Checkpoint image {index} is not marked trajectory_kind={_BAND_KIND!r}"
            )
        if image.info.get("mlipx_neb_image_index") != index:
            raise NEBPreparationError(f"Checkpoint image {index} has a wrong index tag")
        if not np.array_equal(image.numbers, numbers):
            raise NEBPreparationError(
                f"Checkpoint image {index} changes atom identity/order"
            )
        if not np.allclose(image.get_masses(), masses, rtol=0.0, atol=1.0e-8):
            raise NEBPreparationError(f"Checkpoint image {index} changes atom masses")
        if not np.array_equal(np.asarray(image.pbc, bool), pbc):
            raise NEBPreparationError(f"Checkpoint image {index} changes PBC")
        if not np.allclose(image.cell.array, cell, rtol=0.0, atol=1.0e-8):
            raise NEBPreparationError(f"Checkpoint image {index} changes the cell")
        if _constraint_indices(image) != fixed:
            raise NEBPreparationError(
                f"Checkpoint image {index} changes frozen constraints"
            )
        if not np.all(np.isfinite(image.positions)):
            raise NEBPreparationError(
                f"Checkpoint image {index} positions contain NaN or Inf"
            )


def _read_checkpoint_directory(path: Path) -> tuple[dict[str, Any], list[Atoms]]:
    metadata_path = path / "checkpoint.json"
    band_path = path / "band.traj"
    if not metadata_path.is_file() or not band_path.is_file():
        raise NEBPreparationError(
            f"Incomplete checkpoint {path}: checkpoint.json and band.traj are required"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NEBPreparationError(
            f"Invalid checkpoint metadata: {metadata_path}"
        ) from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != _CHECKPOINT_SCHEMA:
        raise NEBPreparationError(f"Unsupported checkpoint schema in {metadata_path}")
    if metadata.get("complete") is not True:
        raise NEBPreparationError(f"Checkpoint is not marked complete: {path}")
    if metadata.get("trajectory_kind") != _BAND_KIND:
        raise NEBPreparationError(f"Checkpoint is not a NEB band: {path}")
    if file_sha256(band_path) != metadata.get("band_sha256"):
        raise NEBPreparationError(f"Checkpoint band checksum mismatch: {band_path}")
    try:
        with Trajectory(band_path, mode="r") as trajectory:
            images = list(trajectory)
    except Exception as exc:
        raise NEBPreparationError(
            f"Could not read checkpoint band: {band_path}"
        ) from exc
    _validate_loaded_band(images, metadata)
    return metadata, images


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a checkpoint directory, a checkpoints root, or a run root."""
    candidate = Path(path).expanduser().resolve()
    if (candidate / "checkpoint.json").is_file():
        return candidate
    for nested in (candidate / "latest", candidate / "checkpoints" / "latest"):
        if nested.exists() or nested.is_symlink():
            return nested.resolve()
    raise NEBPreparationError(f"No complete NEB checkpoint found at {candidate}")


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], list[Atoms]]:
    return _read_checkpoint_directory(resolve_checkpoint_path(path))


class NEBCheckpointStore:
    """Write immutable complete-band checkpoints and atomically move ``latest``."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.root = self.run_dir / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        sequences = []
        for path in self.root.iterdir():
            match = _STEP_PATTERN.fullmatch(path.name)
            if match and path.is_dir():
                sequences.append(int(match.group(1)))
        self._next_sequence = max(sequences, default=-1) + 1

    def write(
        self,
        images: Iterable[Atoms],
        *,
        run_id: str,
        attempt_id: str,
        stage: str,
        stage_step: int,
        resume_fingerprint: Mapping[str, Any],
        resume_fingerprint_sha256: str,
        resolved_config: Mapping[str, Any],
    ) -> Path:
        with _checkpoint_root_lock(self.root):
            return self._write_locked(
                images,
                run_id=run_id,
                attempt_id=attempt_id,
                stage=stage,
                stage_step=stage_step,
                resume_fingerprint=resume_fingerprint,
                resume_fingerprint_sha256=resume_fingerprint_sha256,
                resolved_config=resolved_config,
            )

    def _write_locked(
        self,
        images: Iterable[Atoms],
        *,
        run_id: str,
        attempt_id: str,
        stage: str,
        stage_step: int,
        resume_fingerprint: Mapping[str, Any],
        resume_fingerprint_sha256: str,
        resolved_config: Mapping[str, Any],
    ) -> Path:
        sequences = [
            int(match.group(1))
            for path in self.root.iterdir()
            if (match := _STEP_PATTERN.fullmatch(path.name)) and path.is_dir()
        ]
        sequence = max(self._next_sequence, max(sequences, default=-1) + 1)
        self._next_sequence = sequence + 1
        destination = self.root / f"step_{sequence:06d}"
        if destination.exists():
            raise FileExistsError(
                f"Checkpoint destination already exists: {destination}"
            )
        snapshots = _calculator_free_band(images)
        temporary = self.root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        temporary_link: Path | None = None
        temporary.mkdir()
        try:
            band_path = temporary / "band.traj"
            with Trajectory(band_path, mode="w") as trajectory:
                for snapshot in snapshots:
                    trajectory.write(snapshot)
            with band_path.open("rb") as handle:
                os.fsync(handle.fileno())
            metadata = {
                "schema": _CHECKPOINT_SCHEMA,
                "trajectory_kind": _BAND_KIND,
                "complete": True,
                "geometry_resume_only": True,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "checkpoint_sequence": sequence,
                "stage": stage,
                "stage_step": int(stage_step),
                "image_count": len(snapshots),
                "band_sha256": file_sha256(band_path),
                "resume_fingerprint": resume_fingerprint,
                "resume_fingerprint_sha256": resume_fingerprint_sha256,
                "resolved_config": resolved_config,
            }
            atomic_write_json(temporary / "checkpoint.json", metadata)
            _read_checkpoint_directory(temporary)
            temporary.replace(destination)
            _fsync_directory(self.root)

            latest = self.root / "latest"
            if (latest.exists() or latest.is_symlink()) and not latest.is_symlink():
                raise NEBPreparationError(
                    f"Refusing to replace non-symlink checkpoint pointer: {latest}"
                )
            temporary_link = self.root / f".latest.{uuid.uuid4().hex}.tmp"
            temporary_link.symlink_to(destination.name, target_is_directory=True)
            temporary_link.replace(latest)
            _fsync_directory(self.root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            if temporary_link is not None:
                temporary_link.unlink(missing_ok=True)
            raise
        return destination


def export_vasp_band(
    images: Iterable[Atoms],
    output_dir: str | Path,
    *,
    atom_map: Iterable[int],
    image_shifts: Iterable[Iterable[int]],
    path_convention: str,
) -> Path:
    """Atomically export ``00/POSCAR ... NN/POSCAR`` without MLIP energies."""
    snapshots = _calculator_free_band(images)
    mapping = [int(index) for index in atom_map]
    shifts = [[int(component) for component in row] for row in image_shifts]
    if len(mapping) != len(snapshots[0]) or sorted(mapping) != list(
        range(len(snapshots[0]))
    ):
        raise NEBPreparationError("VASP path export atom_map is not a permutation")
    if np.asarray(shifts).shape != (len(snapshots[0]), 3):
        raise NEBPreparationError(
            "VASP path export image_shifts must have shape (atoms, 3)"
        )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"VASP path export already exists: {destination}")
    if any(
        abs(float(np.linalg.det(image.cell.array))) <= 1.0e-12 for image in snapshots
    ):
        raise NEBPreparationError(
            "VASP path export requires a finite full-rank cell for every image"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    width = max(2, len(str(len(snapshots) - 1)))
    try:
        for index, snapshot in enumerate(snapshots):
            image_dir = temporary / f"{index:0{width}d}"
            image_dir.mkdir()
            poscar = image_dir / "POSCAR"
            write(poscar, snapshot, format="vasp", direct=True, sort=False, vasp5=True)
            with poscar.open("rb") as handle:
                os.fsync(handle.fileno())
            roundtrip = read(poscar, format="vasp")
            if not np.array_equal(roundtrip.numbers, snapshot.numbers):
                raise NEBPreparationError("VASP path export changed atom order")
            if not np.allclose(
                roundtrip.positions, snapshot.positions, rtol=0.0, atol=1.0e-7
            ):
                raise NEBPreparationError("VASP path export changed image coordinates")
            if not np.allclose(
                roundtrip.cell.array, snapshot.cell.array, rtol=0.0, atol=1.0e-8
            ):
                raise NEBPreparationError("VASP path export changed the cell")
            if _constraint_indices(roundtrip) != _constraint_indices(snapshot):
                raise NEBPreparationError(
                    "VASP path export changed Selective Dynamics/frozen indices"
                )
        band_metadata = {
            "schema": "mlipx.vasp-neb-path/1",
            "trajectory_kind": _BAND_KIND,
            "energy_source": None,
            "image_count": len(snapshots),
            "atom_map": mapping,
            "image_shifts": shifts,
            "path_convention": path_convention,
        }
        atomic_write_json(
            temporary / "band.json",
            band_metadata,
        )
        if json.loads((temporary / "band.json").read_text(encoding="utf-8")) != (
            band_metadata
        ):
            raise NEBPreparationError("VASP path metadata failed round-trip validation")
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "NEBCheckpointStore",
    "atomic_write_json",
    "canonical_fingerprint",
    "export_vasp_band",
    "file_sha256",
    "load_checkpoint",
    "resolve_checkpoint_path",
]
