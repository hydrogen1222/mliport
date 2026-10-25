"""Input loading, start preparation and validation (spec section 5).

Mode A takes user-prepared starts plus ``cs_adsorbate_indices``; mode B places a
molecule (one or more conformers) above a slab with the section 5.2 algorithm.
Both modes end with the same shared validation (section 5.3) and produce
:class:`~mliport.confsearch.engine.PreparedStart` objects for the engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read as ase_read

from mliport.confsearch.errors import ConfsearchPreparationError
from mliport.confsearch.engine import PreparedStart
from mliport.confsearch.index_spec import parse_index_spec
from mliport.confsearch.placement import place_molecule
from mliport.confsearch.regions import resolve_regions, validate_all
from mliport.neb.io import file_sha256

if TYPE_CHECKING:
    from ase import Atoms

    from mliport.confsearch.options import ConfsearchOptions

#: Structure file suffixes accepted when a directory is given.
STRUCTURE_SUFFIXES = (
    ".xyz",
    ".extxyz",
    ".traj",
    ".vasp",
    ".poscar",
    ".contcar",
    ".cif",
    ".pdb",
    ".json",
)


def load_frames(path: str | Path, *, what: str) -> list[Atoms]:
    """Read every frame of a structure file, or every file in a directory."""
    location = Path(path).expanduser()
    if not location.exists():
        raise ConfsearchPreparationError(f"{what}: path not found: {location}")
    if location.is_dir():
        files = sorted(
            child
            for child in location.iterdir()
            if child.is_file() and child.suffix.lower() in STRUCTURE_SUFFIXES
        )
        if not files:
            raise ConfsearchPreparationError(
                f"{what}: directory {location} contains no structure files "
                f"({', '.join(STRUCTURE_SUFFIXES)})"
            )
        frames: list[Atoms] = []
        for child in files:
            frames.extend(_read_frames(child, what=f"{what} ({child.name})"))
        return frames
    return _read_frames(location, what=what)


def _read_frames(path: Path, *, what: str) -> list[Atoms]:
    try:
        frames = ase_read(path, index=":")
    except Exception as exc:  # pragma: no cover - depends on the file format
        raise ConfsearchPreparationError(f"{what}: cannot read {path}: {exc}") from exc
    if not frames:
        raise ConfsearchPreparationError(f"{what}: {path} contains no frames")
    return [frame.copy() for frame in frames]


def _fixed_indices(atoms: Atoms) -> tuple[int, ...]:
    indices: set[int] = set()
    for constraint in atoms.constraints:
        if type(constraint) is not FixAtoms:  # noqa: E721 - FixScaled subclasses it
            raise ConfsearchPreparationError(
                f"confsearch accepts only FixAtoms constraints, got "
                f"{type(constraint).__name__}"
            )
        indices.update(int(index) for index in constraint.get_indices())
    return tuple(sorted(indices))


def _electronic_state(atoms: Atoms) -> tuple:
    return (
        atoms.info.get("charge", None),
        atoms.info.get("spin", None),
    )


def validate_start_identity(frames: list[Atoms], *, what: str) -> None:
    """Every start must be the same system (spec section 5.1), exactly."""
    reference = frames[0]
    for index, frame in enumerate(frames[1:], start=1):
        if len(frame) != len(reference):
            raise ConfsearchPreparationError(
                f"{what}: start {index} has {len(frame)} atoms, start 0 has "
                f"{len(reference)}"
            )
        if not np.array_equal(
            np.asarray(frame.numbers, dtype=int),
            np.asarray(reference.numbers, dtype=int),
        ):
            raise ConfsearchPreparationError(
                f"{what}: start {index} has a different element order than start 0"
            )
        if not np.array_equal(
            np.asarray(frame.cell, dtype=float), np.asarray(reference.cell, dtype=float)
        ):
            raise ConfsearchPreparationError(
                f"{what}: start {index} has a different cell than start 0 "
                "(the comparison is exact; no tolerance)"
            )
        if not np.array_equal(
            np.asarray(frame.pbc, dtype=bool), np.asarray(reference.pbc, dtype=bool)
        ):
            raise ConfsearchPreparationError(
                f"{what}: start {index} has different PBC than start 0"
            )
        if _fixed_indices(frame) != _fixed_indices(reference):
            raise ConfsearchPreparationError(
                f"{what}: start {index} has a different fixed-atom set than start 0"
            )
        if _electronic_state(frame) != _electronic_state(reference):
            raise ConfsearchPreparationError(
                f"{what}: start {index} has a different charge/spin than start 0"
            )


def input_hashes(options: ConfsearchOptions) -> dict:
    """Content hashes of every declared input (used by the resume fingerprint)."""
    hashes: dict[str, object] = {}
    for key in ("cs_starts", "cs_slab", "cs_molecule"):
        value = getattr(options, key)
        if value is None:
            continue
        paths = value if isinstance(value, tuple) else (value,)
        entries = {}
        for path in paths:
            location = Path(path).expanduser()
            if location.is_file():
                entries[str(location)] = file_sha256(location)
            elif location.is_dir():
                entries[str(location)] = {
                    child.name: file_sha256(child)
                    for child in sorted(location.iterdir())
                    if child.is_file()
                }
            else:
                entries[str(location)] = None
        hashes[key] = entries
    return hashes


def prepare_starts(
    options: ConfsearchOptions, *, placement_rng: np.random.Generator
) -> list[PreparedStart]:
    """Build every validated start for the resolved mode."""
    if options.mode == "A":
        return _prepare_mode_a(options)
    return _prepare_mode_b(options, placement_rng=placement_rng)


def _prepare_mode_a(options: ConfsearchOptions) -> list[PreparedStart]:
    frames: list[Atoms] = []
    for path in options.cs_starts or ():
        frames.extend(load_frames(path, what="cs_starts"))
    if not frames:
        raise ConfsearchPreparationError("cs_starts produced no structures")
    validate_start_identity(frames, what="cs_starts")
    indices = parse_index_spec(
        options.cs_adsorbate_indices or "",
        n_atoms=len(frames[0]),
        label="cs_adsorbate_indices",
    )
    starts: list[PreparedStart] = []
    for index, frame in enumerate(frames):
        regions = resolve_regions(frame, options, adsorbate_indices=np.asarray(indices))
        surface = validate_all(frame, regions, options)
        starts.append(
            PreparedStart(
                index=index,
                atoms=frame,
                regions=regions,
                surface=surface,
                label=f"start_{index:03d}",
            )
        )
    return starts


def _prepare_mode_b(
    options: ConfsearchOptions, *, placement_rng: np.random.Generator
) -> list[PreparedStart]:
    slab_frames = load_frames(options.cs_slab or "", what="cs_slab")
    if len(slab_frames) != 1:
        raise ConfsearchPreparationError(
            f"cs_slab must contain exactly one structure, got {len(slab_frames)}"
        )
    slab = slab_frames[0]
    conformers = load_frames(options.cs_molecule or "", what="cs_molecule")
    n_starts = int(options.cs_n_starts or 0)
    if n_starts < 1:
        raise ConfsearchPreparationError("cs_n_starts must be >= 1")

    z_top = float(np.max(np.asarray(slab.get_positions(), dtype=float)[:, 2]))
    slab_indices = np.arange(len(slab))
    starts: list[PreparedStart] = []
    for index in range(n_starts):
        placement = place_molecule(
            slab,
            conformers,
            options,
            rng=placement_rng,
            z_top=z_top,
            surface_validator=None,
        )
        adsorbate = np.arange(len(slab), len(slab) + len(conformers[0]))
        regions = resolve_regions(placement.atoms, options, adsorbate_indices=adsorbate)
        surface = validate_all(placement.atoms, regions, options)
        starts.append(
            PreparedStart(
                index=index,
                atoms=placement.atoms,
                regions=regions,
                surface=surface,
                label=f"placed_{index:03d}",
                placement=placement,
            )
        )
    validate_start_identity(
        [start.atoms for start in starts], what="cs_slab/cs_molecule"
    )
    del slab_indices
    return starts


__all__ = [
    "STRUCTURE_SUFFIXES",
    "input_hashes",
    "load_frames",
    "prepare_starts",
    "validate_start_identity",
]
