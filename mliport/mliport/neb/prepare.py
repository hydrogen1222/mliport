"""Fail-closed endpoint and band preparation for fixed-cell NEB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.mep import NEB, idpp_interpolate
from ase.mep.neb import IDPP
from ase.optimize import FIRE

from mliport.neb.schema import NEBOptions, NEBPreparationError


def _integer_array(value, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or not np.issubdtype(raw.dtype, np.integer):
        raise NEBPreparationError(f"{name} must have integer shape {shape}")
    return raw.astype(int, copy=True)


def _atom_ids(atoms: Atoms) -> np.ndarray | None:
    for key in ("atom_id", "atom_ids"):
        if key in atoms.arrays:
            values = np.asarray(atoms.arrays[key])
            if values.ndim != 1 or len(values) != len(atoms):
                raise NEBPreparationError(f"atoms.arrays[{key!r}] has invalid shape")
            return values
        if key in atoms.info:
            values = np.asarray(atoms.info[key])
            if values.ndim != 1 or len(values) != len(atoms):
                raise NEBPreparationError(f"atoms.info[{key!r}] has invalid shape")
            return values
    return None


def _fixed_indices(atoms: Atoms) -> np.ndarray:
    fixed: set[int] = set()
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            raise NEBPreparationError(
                "Only FixAtoms is supported for NEB; refusing constraint "
                f"{type(constraint).__name__}"
            )
        fixed.update(int(index) for index in constraint.get_indices())
    return np.asarray(sorted(fixed), dtype=int)


def _validate_common_geometry(images: list[Atoms], options: NEBOptions) -> None:
    if len(images) < 3:
        raise NEBPreparationError("A NEB band requires at least one intermediate image")
    reference = images[0]
    if len(reference) == 0:
        raise NEBPreparationError("NEB structures must contain at least one atom")
    reference_symbols = tuple(reference.get_chemical_symbols())
    reference_masses = np.asarray(reference.get_masses(), dtype=float)
    reference_atom_ids = _atom_ids(reference)
    reference_pbc = np.asarray(reference.pbc, dtype=bool)
    if bool(reference_pbc.any()) and not bool(reference_pbc.all()):
        raise NEBPreparationError(
            "Partial PBC is unsupported for fixed-cell NEB; use explicit 3-D "
            "periodicity or a non-periodic molecule"
        )
    reference_cell = np.asarray(reference.cell.array, dtype=float)
    if not np.all(np.isfinite(reference_cell)):
        raise NEBPreparationError("NEB cell contains NaN or Inf")
    if reference_pbc.all() and abs(float(np.linalg.det(reference_cell))) <= 1.0e-12:
        raise NEBPreparationError("Periodic NEB cell must be finite and full rank")
    fixed = _fixed_indices(reference)
    for index, image in enumerate(images):
        if len(image) != len(reference):
            raise NEBPreparationError(f"Image {index} changes atom count")
        if tuple(image.get_chemical_symbols()) != reference_symbols:
            raise NEBPreparationError(f"Image {index} changes atom identity/order")
        image_masses = np.asarray(image.get_masses(), dtype=float)
        if not np.allclose(image_masses, reference_masses, rtol=0.0, atol=1.0e-8):
            raise NEBPreparationError(
                f"Image {index} changes atomic masses; element identity alone "
                "does not preserve the physical system"
            )
        image_atom_ids = _atom_ids(image)
        if (reference_atom_ids is None) != (image_atom_ids is None):
            raise NEBPreparationError(
                f"Image {index} changes atom_id metadata presence; per-atom "
                "mapping identity must be carried by every image"
            )
        if reference_atom_ids is not None and not np.array_equal(
            image_atom_ids, reference_atom_ids
        ):
            raise NEBPreparationError(
                f"Image {index} changes per-atom mapping identity (atom_id)"
            )
        if not np.array_equal(np.asarray(image.pbc, dtype=bool), reference_pbc):
            raise NEBPreparationError(f"Image {index} changes PBC flags")
        cell = np.asarray(image.cell.array, dtype=float)
        if not np.all(np.isfinite(cell)):
            raise NEBPreparationError(f"Image {index} cell contains NaN or Inf")
        if not np.allclose(cell, reference_cell, rtol=0.0, atol=1.0e-8):
            raise NEBPreparationError(
                f"Image {index} changes the fixed cell; variable-cell NEB is unsupported"
            )
        if not np.all(np.isfinite(image.positions)):
            raise NEBPreparationError(f"Image {index} positions contain NaN or Inf")
        if not np.array_equal(_fixed_indices(image), fixed):
            raise NEBPreparationError(f"Image {index} changes FixAtoms constraints")
        if len(fixed):
            # Explicit provided-band contract: a frozen atom must keep the
            # exact same raw coordinate in EVERY image.  An equivalent lattice
            # image is NOT accepted here, because the raw-coordinate spring
            # convention would turn it into a spurious segment (task book
            # PR-D / section 6.2).
            movement = image.positions[fixed] - reference.positions[fixed]
            if not np.allclose(movement, 0.0, rtol=0.0, atol=1.0e-8):
                raise NEBPreparationError(
                    f"Image {index} moves FixAtoms atom(s) {fixed.tolist()} by "
                    f"up to {float(np.abs(movement).max()):g} A; every image "
                    "must keep frozen atoms at the exact same coordinates "
                    "(equivalent lattice images are rejected by contract)"
                )


def _validate_identity(initial: Atoms, final: Atoms, atom_map: np.ndarray) -> Atoms:
    if len(initial) != len(final):
        raise NEBPreparationError("NEB endpoints have different atom counts")
    if not np.array_equal(np.sort(atom_map), np.arange(len(initial))):
        raise NEBPreparationError(
            "atom_map must be a 0-based permutation in "
            "final_index_for_initial direction"
        )
    mapped = final[atom_map]
    if tuple(initial.get_chemical_symbols()) != tuple(mapped.get_chemical_symbols()):
        raise NEBPreparationError("atom_map changes element identity")
    if not np.allclose(
        initial.get_masses(), mapped.get_masses(), rtol=0.0, atol=1.0e-8
    ):
        raise NEBPreparationError("atom_map changes atomic masses")
    initial_ids = _atom_ids(initial)
    final_ids = _atom_ids(final)
    if (initial_ids is None) != (final_ids is None):
        raise NEBPreparationError("atom_id metadata must be present on both endpoints")
    if initial_ids is not None and not np.array_equal(initial_ids, final_ids[atom_map]):
        raise NEBPreparationError("atom_map does not preserve atom_id identity")
    return mapped


def _mapped_fixed_indices(final: Atoms, atom_map: np.ndarray) -> np.ndarray:
    final_fixed = set(_fixed_indices(final).tolist())
    return np.asarray(
        sorted(
            index
            for index, final_index in enumerate(atom_map)
            if final_index in final_fixed
        ),
        dtype=int,
    )


def _check_collisions(images: Iterable[Atoms], minimum_distance_A: float) -> None:
    for index, image in enumerate(images):
        if len(image) < 2:
            continue
        distances = image.get_all_distances(mic=bool(np.asarray(image.pbc).all()))
        pair_distances = distances[np.triu_indices(len(image), k=1)]
        if len(pair_distances) and np.min(pair_distances) < minimum_distance_A:
            raise NEBPreparationError(
                f"Image {index} contains an atom pair closer than "
                f"min_distance_A={minimum_distance_A:g}"
            )


def _validate_adjacent_periodic_segments(images: Iterable[Atoms]) -> None:
    """Ensure every segment agrees with ASE's periodic spring image choice."""

    image_list = list(images)
    if not image_list or not bool(np.asarray(image_list[0].pbc, dtype=bool).all()):
        return
    cell = np.asarray(image_list[0].cell.array, dtype=float)
    pbc = np.asarray(image_list[0].pbc, dtype=bool)
    for index, (left, right) in enumerate(zip(image_list, image_list[1:])):
        raw = np.asarray(right.positions - left.positions, dtype=float)
        mic, _ = find_mic(raw, cell=cell, pbc=pbc)
        if not np.allclose(raw, mic, rtol=0.0, atol=1.0e-8):
            raise NEBPreparationError(
                f"Adjacent images {index} and {index + 1} cross more than one "
                "periodic MIC segment; add images or provide a denser band"
            )


def _validate_fixed_coordinates(initial: Atoms, final: Atoms) -> None:
    fixed = _fixed_indices(initial)
    if len(fixed) and not np.allclose(
        initial.positions[fixed], final.positions[fixed], rtol=0.0, atol=1.0e-8
    ):
        raise NEBPreparationError(
            "Fixed atom coordinates differ between endpoints; refusing to "
            "release or silently move the constraint"
        )


@dataclass(slots=True)
class EndpointDisplacement:
    """Calculator-free resolution of the intended NEB endpoint displacement.

    ``intended_displacement`` is the per-atom displacement the band is
    prepared for: the mapped final endpoint minus the initial endpoint,
    resolved through the general triclinic minimum-image convention
    (``path_convention='mic'``) or lifted by explicit per-atom lattice
    winding shifts (``path_convention='unwrapped'``).
    """

    mapped_final: Atoms
    atom_map: np.ndarray
    image_shifts: np.ndarray
    intended_displacement: np.ndarray
    lifted_final_positions: np.ndarray
    winding_present: bool
    path_convention: str


def prepare_endpoint_geometry(
    initial: Atoms,
    final: Atoms,
    *,
    atom_map: Iterable[int] | None = None,
    image_shifts: Iterable[Iterable[int]] | None = None,
    path_convention: str,
) -> EndpointDisplacement:
    """Map the final endpoint and resolve the intended displacement.

    Shared by the production band preparation (:func:`prepare_band`) and the
    TUI endpoint preview so both report the same displacement semantics.
    This performs the endpoint mapping (identity validation included), the
    explicit image-shift handling and the general triclinic MIC resolution.
    It uses no calculator and mutates no input.
    """
    if path_convention not in {"mic", "unwrapped"}:
        raise NEBPreparationError("path_convention must be 'mic' or 'unwrapped'")
    if len(initial) != len(final):
        raise NEBPreparationError("NEB endpoints have different atom counts")
    if not np.allclose(initial.cell.array, final.cell.array, rtol=0.0, atol=1.0e-8):
        raise NEBPreparationError("NEB endpoints have different cells")
    if not np.array_equal(initial.pbc, final.pbc):
        raise NEBPreparationError("NEB endpoints have different PBC flags")
    mapping = (
        np.arange(len(initial), dtype=int)
        if atom_map is None
        else _integer_array(atom_map, shape=(len(initial),), name="atom_map")
    )
    mapped_final = _validate_identity(initial, final, mapping)
    pbc = np.asarray(initial.pbc, dtype=bool)
    cell = np.asarray(initial.cell.array, dtype=float)
    if image_shifts is not None:
        if path_convention != "unwrapped":
            raise NEBPreparationError(
                "image_shifts is only valid with path_convention='unwrapped'"
            )
        shifts = _integer_array(
            image_shifts, shape=(len(initial), 3), name="image_shifts"
        )
        if np.any(shifts[:, ~pbc] != 0):
            raise NEBPreparationError(
                "image_shifts in non-periodic directions must be 0"
            )
    else:
        shifts = np.zeros((len(initial), 3), dtype=int)

    displacement = np.asarray(mapped_final.positions - initial.positions, dtype=float)
    if path_convention == "mic":
        displacement, _ = find_mic(displacement, cell=cell, pbc=pbc)
    elif image_shifts is not None:
        displacement = displacement + shifts @ cell
    if not np.all(np.isfinite(displacement)):
        raise NEBPreparationError("Endpoint displacement contains NaN or Inf")
    return EndpointDisplacement(
        mapped_final=mapped_final,
        atom_map=mapping,
        image_shifts=shifts,
        intended_displacement=displacement,
        lifted_final_positions=initial.positions + displacement,
        winding_present=bool(np.any(shifts)),
        path_convention=path_convention,
    )


def _set_fix_atoms(image: Atoms, fixed: np.ndarray) -> None:
    if len(fixed):
        image.set_constraint(FixAtoms(indices=fixed.tolist()))
    else:
        image.set_constraint([])


def _linear_band(
    initial: Atoms,
    final_positions: np.ndarray,
    n_intermediate: int,
    fixed: np.ndarray,
) -> list[Atoms]:
    images = [initial.copy()]
    for index in range(1, n_intermediate + 1):
        image = initial.copy()
        fraction = index / (n_intermediate + 1)
        image.positions = initial.positions + fraction * (
            final_positions - initial.positions
        )
        _set_fix_atoms(image, fixed)
        images.append(image)
    final = initial.copy()
    final.positions = final_positions
    _set_fix_atoms(final, fixed)
    images.append(final)
    return images


def interpolate_lifted_band(
    initial: Atoms, final: Atoms, options: NEBOptions
) -> list[Atoms]:
    """Build a band between already identity-matched Cartesian endpoints.

    This is used after endpoint relaxation. It interpolates the already
    lifted coordinates directly, so an explicit periodic winding is not
    silently replaced by a new MIC choice.
    """

    if len(initial) != len(final):
        raise NEBPreparationError("Relaxed endpoints have different atom counts")
    if tuple(initial.get_chemical_symbols()) != tuple(final.get_chemical_symbols()):
        raise NEBPreparationError("Relaxed endpoints change atom identity/order")
    if not np.array_equal(initial.pbc, final.pbc) or not np.allclose(
        initial.cell.array, final.cell.array, rtol=0.0, atol=1.0e-8
    ):
        raise NEBPreparationError("Relaxed endpoints change cell or PBC")
    fixed = _fixed_indices(initial)
    if not np.array_equal(_fixed_indices(final), fixed):
        raise NEBPreparationError("Relaxed endpoints change FixAtoms constraints")
    if len(fixed) and not np.allclose(
        initial.positions[fixed], final.positions[fixed], rtol=0.0, atol=1.0e-8
    ):
        raise NEBPreparationError(
            "Fixed atom coordinates differ after endpoint relaxation"
        )
    images = _linear_band(
        initial, final.positions, options.n_intermediate_images, fixed
    )
    if options.interpolation == "idpp":
        _apply_idpp(images, options)
    _validate_common_geometry(images, options)
    _validate_adjacent_periodic_segments(images)
    _check_collisions(images, options.min_distance_A)
    return images


def _apply_idpp(images: list[Atoms], options: NEBOptions) -> None:
    neb = NEB(
        images,
        parallel=False,
        method="improvedtangent",
        allow_shared_calculator=False,
    )
    max_force = 0.0
    try:
        idpp_interpolate(
            neb,
            traj=None,
            log=None,
            fmax=options.idpp_fmax_eV_A,
            optimizer=FIRE,
            mic=options.idpp_mic,
            steps=options.idpp_steps,
        )
        for index, image in enumerate(images[1:-1], start=1):
            target = images[0].get_all_distances(mic=options.idpp_mic) + index * (
                images[-1].get_all_distances(mic=options.idpp_mic)
                - images[0].get_all_distances(mic=options.idpp_mic)
            ) / (len(images) - 1)
            image.calc = IDPP(target, mic=options.idpp_mic)
            force = image.get_forces()
            if not np.all(np.isfinite(force)):
                raise NEBPreparationError("IDPP produced a non-finite force")
            max_force = max(max_force, float(np.linalg.norm(force, axis=1).max()))
    except NEBPreparationError:
        raise
    except Exception as exc:
        raise NEBPreparationError(
            "IDPP interpolation failed; refusing an unmarked linear fallback"
        ) from exc
    finally:
        for image in images:
            image.calc = None
    if max_force > options.idpp_fmax_eV_A * (1.0 + 1.0e-6):
        raise NEBPreparationError(
            "IDPP did not reach idpp_fmax_eV_A within idpp_steps; "
            "refusing an unmarked fallback to linear interpolation"
        )


@dataclass(slots=True)
class BandInput:
    """A validated, calculator-free NEB band and its identity metadata."""

    images: tuple[Atoms, ...]
    atom_map: np.ndarray
    image_shifts: np.ndarray
    path_convention: str
    source: str = "endpoints"

    def __post_init__(self) -> None:
        self.images = tuple(image.copy() for image in self.images)
        self.atom_map = np.asarray(self.atom_map, dtype=int).copy()
        self.image_shifts = np.asarray(self.image_shifts, dtype=int).copy()
        if len(self.images) < 3:
            raise NEBPreparationError(
                "A NEB band requires at least one intermediate image"
            )
        if self.path_convention not in {"mic", "unwrapped"}:
            raise NEBPreparationError("Band path_convention must be mic or unwrapped")
        natoms = len(self.images[0])
        if self.atom_map.shape != (natoms,) or not np.array_equal(
            np.sort(self.atom_map), np.arange(natoms)
        ):
            raise NEBPreparationError("Band atom_map must be a 0-based permutation")
        if self.image_shifts.shape != (natoms, 3):
            raise NEBPreparationError("Band image_shifts must have shape (natoms, 3)")
        pbc = np.asarray(self.images[0].pbc, dtype=bool)
        if np.any(self.image_shifts[:, ~pbc] != 0):
            raise NEBPreparationError(
                "Band image_shifts in non-periodic directions must be 0"
            )
        if self.path_convention == "mic" and np.any(self.image_shifts):
            raise NEBPreparationError(
                "MIC bands cannot also carry explicit image_shifts"
            )
        _validate_common_geometry(
            list(self.images), NEBOptions(n_intermediate_images=len(self.images) - 2)
        )
        _validate_adjacent_periodic_segments(self.images)

    @property
    def initial(self) -> Atoms:
        return self.images[0]

    @property
    def final(self) -> Atoms:
        return self.images[-1]

    @property
    def n_intermediate_images(self) -> int:
        return len(self.images) - 2


def prepare_band(
    initial: Atoms,
    final: Atoms,
    options: NEBOptions,
    *,
    atom_map: Iterable[int] | None = None,
    image_shifts: Iterable[Iterable[int]] | None = None,
) -> BandInput:
    """Validate endpoints and construct a linear or IDPP initial band."""

    initial_copy = initial.copy()
    final_copy = final.copy()
    endpoint = prepare_endpoint_geometry(
        initial_copy,
        final_copy,
        atom_map=atom_map,
        image_shifts=None if image_shifts is None else np.asarray(image_shifts),
        path_convention=options.path_convention,
    )
    mapping = endpoint.atom_map
    fixed_initial = _fixed_indices(initial_copy)
    fixed_final = _mapped_fixed_indices(final_copy, mapping)
    if not np.array_equal(fixed_initial, fixed_final):
        raise NEBPreparationError("atom_map changes the FixAtoms constraint")
    lifted_positions = endpoint.lifted_final_positions
    shifts = endpoint.image_shifts
    # For an explicit winding path, fixed atoms must also remain at the same
    # lifted position; this catches a nonzero shift on a frozen atom while
    # still allowing a wrapped endpoint representation of the same image.
    if len(fixed_initial) and not np.allclose(
        initial_copy.positions[fixed_initial],
        lifted_positions[fixed_initial],
        atol=1.0e-8,
        rtol=0.0,
    ):
        raise NEBPreparationError("image_shifts move a fixed atom between endpoints")
    lifted_final = initial_copy.copy()
    lifted_final.positions = lifted_positions
    _set_fix_atoms(lifted_final, fixed_initial)
    images = interpolate_lifted_band(initial_copy, lifted_final, options)
    return BandInput(
        images=tuple(images),
        atom_map=mapping,
        image_shifts=shifts,
        path_convention=options.path_convention,
    )


def validate_band_images(images: Iterable[Atoms], options: NEBOptions) -> BandInput:
    """Validate an existing complete band without re-interpolating it.

    Every image is checked against the first one for atom count, species
    order, masses, atom_id mapping metadata, cell/PBC, FixAtoms indices and
    frozen-atom coordinates; adjacent-image displacement and collision
    contracts are checked as well.  Comparing only the endpoints is
    explicitly not sufficient (task book PR-D).
    """

    copied = tuple(image.copy() for image in images)
    if len(copied) != options.n_intermediate_images + 2:
        raise NEBPreparationError(
            "Existing band image count conflicts with n_intermediate_images: "
            f"got {len(copied) - 2}, expected {options.n_intermediate_images}"
        )
    _validate_common_geometry(list(copied), options)
    _validate_fixed_coordinates(copied[0], copied[-1])
    _validate_adjacent_periodic_segments(copied)
    _check_collisions(copied, options.min_distance_A)
    return BandInput(
        images=copied,
        atom_map=np.arange(len(copied[0]), dtype=int),
        image_shifts=np.zeros((len(copied[0]), 3), dtype=int),
        path_convention="unwrapped",
        source="provided_band",
    )
