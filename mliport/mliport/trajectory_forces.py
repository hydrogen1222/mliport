"""Raw (unprojected) MD forces inside an ASE ``.traj`` frame.

``mliport md`` keeps two force views per saved frame: the constraint-applied
forces (the frame calculator's ``forces``, what ASE ``get_forces()`` returns)
and the raw, unprojected forces. By default the raw forces are written as
``atoms.info["forces_raw_eV_A"]``; ASE serialises ``info`` as JSON text, which
costs roughly 2.6x the size of the same array in binary.

With ``raw_forces_storage="binary"`` the raw forces are instead written as one
extra float64 ULM field (``mliport_raw_forces_v1``, shape ``(natoms, 3)``,
eV/angstrom, bit-exact) in the *same* frame, through ASE's own frame writer and
sync. Plain ``ase.io.read`` still returns structure, momenta and the
constraint-applied forces; the raw forces are available through
:func:`read_raw_forces` / :func:`read_frames_with_raw_forces` here, which accept
both the legacy JSON field and the binary field.

The storage mode is explicit and never silently changed: appending to an
existing trajectory with a different mode raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.io import ulm
from ase.io.trajectory import Trajectory, TrajectoryWriter as AseTrajectoryWriter

if TYPE_CHECKING:
    from ase import Atoms

#: Versioned name of the binary field. A layout change gets a new name.
RAW_FORCES_ULM_FIELD = "mliport_raw_forces_v1"
#: Legacy/JSON location, also the name restored by the reader.
RAW_FORCES_INFO_KEY = "forces_raw_eV_A"
RAW_FORCES_STORAGE_CHOICES = ("json", "binary")
RAW_FORCES_STORAGE_DEFAULT = "json"


class RawForcesStorageError(ValueError):
    """Raised when the raw-force storage mode is invalid or inconsistent."""


def validate_raw_forces_storage(value: str) -> str:
    mode = str(value).lower()
    if mode not in RAW_FORCES_STORAGE_CHOICES:
        raise RawForcesStorageError(
            f"raw_forces_storage must be one of {RAW_FORCES_STORAGE_CHOICES}, "
            f"got {value!r}"
        )
    return mode


class RawForcesTrajectoryWriter(AseTrajectoryWriter):
    """ASE trajectory writer that stores raw forces as a binary ULM field."""

    def _write_atoms(
        self, atoms: Atoms, raw_forces: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        if raw_forces is None:
            raise RawForcesStorageError(
                "raw_forces_storage='binary' requires raw forces for every frame"
            )
        array = np.ascontiguousarray(raw_forces, dtype=np.dtype("<f8"))
        if array.shape != (len(atoms), 3):
            raise RawForcesStorageError(
                f"raw forces must have shape ({len(atoms)}, 3), got {array.shape}"
            )
        # Written into the current ULM frame; ASE's own ``_write_atoms`` adds the
        # structure/calculator/info and performs the single frame ``sync``.
        self.backend.write(**{RAW_FORCES_ULM_FIELD: array})
        super()._write_atoms(atoms, **kwargs)


def detect_raw_forces_storage(path: str | Path) -> str | None:
    """Storage mode of an existing trajectory.

    Returns ``"binary"``, ``"json"``, ``None`` (no frames or no raw forces) or
    ``"mixed"`` when the first and last frames disagree.
    """

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with ulm.open(str(path), "r") as reader:
        n = len(reader)
        if n == 0:
            return None
        modes = {_frame_mode(reader[0]), _frame_mode(reader[n - 1])}
    modes.discard(None)
    if not modes:
        return None
    return modes.pop() if len(modes) == 1 else "mixed"


def _frame_mode(frame: Any) -> str | None:
    if RAW_FORCES_ULM_FIELD in frame:
        return "binary"
    info = frame.get("info") if hasattr(frame, "get") else None
    if info is not None and RAW_FORCES_INFO_KEY in info:
        return "json"
    return None


def open_trajectory_writer(
    path: str | Path, *, mode: str = "w", raw_forces_storage: str = "json"
) -> AseTrajectoryWriter:
    """Open the MD trajectory writer for an explicit raw-force storage mode."""

    storage = validate_raw_forces_storage(raw_forces_storage)
    if mode == "a":
        existing = detect_raw_forces_storage(path)
        if existing is not None and existing != storage:
            raise RawForcesStorageError(
                f"cannot append raw_forces_storage={storage!r} frames to "
                f"{Path(path).name}, which holds {existing!r} raw forces"
            )
    cls = RawForcesTrajectoryWriter if storage == "binary" else AseTrajectoryWriter
    return cls(str(path), mode=mode)


def read_raw_forces(path: str | Path, index: int) -> np.ndarray | None:
    """Raw forces (eV/angstrom, float64) of one frame, or ``None`` if absent.

    Reads the binary field when present, otherwise the legacy JSON ``info``
    field. Negative indices count from the end.
    """

    with ulm.open(str(path), "r") as reader:
        n = len(reader)
        i = int(index)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"frame index {index} out of range for {n} frames")
        return _frame_raw_forces(reader[i])


def _frame_raw_forces(frame: Any) -> np.ndarray | None:
    if RAW_FORCES_ULM_FIELD in frame:
        return np.array(getattr(frame, RAW_FORCES_ULM_FIELD), dtype=float)
    info = frame.get("info") if hasattr(frame, "get") else None
    if info is not None and RAW_FORCES_INFO_KEY in info:
        return np.asarray(info[RAW_FORCES_INFO_KEY], dtype=float)
    return None


def read_frames_with_raw_forces(
    path: str | Path, index: int | slice | str | None = None
) -> list[Atoms]:
    """Read frames with ``info["forces_raw_eV_A"]`` restored for either storage.

    ``index`` is a frame number, a slice, an ASE-style slice string such as
    ``"::10"``, or ``None`` for every frame.
    """

    if index is None:
        index = slice(None)
    with Trajectory(str(path), "r") as trajectory:
        total = len(trajectory)
    if isinstance(index, str):
        index = _parse_slice(index)
    indices = (
        list(range(total))[index]
        if isinstance(index, slice)
        else [int(index) % total if total else int(index)]
    )
    frames: list[Atoms] = []
    with Trajectory(str(path), "r") as trajectory, ulm.open(str(path), "r") as raw:
        for i in indices:
            atoms = trajectory[i]
            forces = _frame_raw_forces(raw[i])
            if forces is not None:
                atoms.info[RAW_FORCES_INFO_KEY] = forces
            frames.append(atoms)
    return frames


def _parse_slice(text: str) -> slice:
    parts = [int(p) if p.strip() else None for p in text.split(":")]
    if len(parts) > 3:
        raise ValueError(f"invalid frame slice {text!r}")
    return slice(*parts)


__all__ = [
    "RAW_FORCES_INFO_KEY",
    "RAW_FORCES_STORAGE_CHOICES",
    "RAW_FORCES_STORAGE_DEFAULT",
    "RAW_FORCES_ULM_FIELD",
    "RawForcesStorageError",
    "RawForcesTrajectoryWriter",
    "detect_raw_forces_storage",
    "open_trajectory_writer",
    "read_frames_with_raw_forces",
    "read_raw_forces",
    "validate_raw_forces_storage",
]
