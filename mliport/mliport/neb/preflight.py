"""Runtime energy-force consistency pre-flight for NEB and confsearch.

NEB feeds the model's forces to the optimizer as if they were ``-grad(E)``.
That holds by construction for models that obtain forces by automatic
differentiation of the energy, but it is a property of the exact model,
framework build and device, not of the checkpoint alone.  Instead of looking
the property up in static evidence recorded on other software stacks, this
module measures it on the band that is about to be optimized.

For a few structures (NEB: both endpoints and interpolated images; confsearch:
the first prepared start) and a few displacement directions ``u`` (the force
direction and random directions on the free atoms) it compares the model's
force projection ``-F.u`` with a central finite difference of the model's
energy.  Two step sizes are combined by Richardson extrapolation, which
removes the leading truncation error, and the acceptance tolerance includes
the energy rounding noise implied by the model's floating-point precision (the
same noise-aware reasoning as ``validation/science/scripts/finite_difference.py``).

This is a spot check, not a proof of consistency over the whole potential
energy surface.  It fails closed: an inconsistency, a non-finite value or a
measurement that the energy resolution cannot resolve raises the caller's
error class before any dynamics or optimization step is taken.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mliport.neb.schema import NEBGradientConsistencyError

if TYPE_CHECKING:
    from ase import Atoms

PREFLIGHT_SCHEMA = "mliport.neb-gradient-preflight/1"

#: Deterministic sampling so a repeated run measures the same directions.
RNG_SEED = 20260601
#: Structures sampled from the band (endpoints plus evenly spaced interior).
MAX_SAMPLED_IMAGES = 4
#: Random directions per structure (the force direction is added when it
#: is meaningful).
RANDOM_DIRECTIONS = 2
#: Minimum force norm (eV/A) for the force direction to be informative.
MIN_FORCE_NORM_EV_A = 1e-3

#: Total displacement (A, norm of the displacement vector) of the finer
#: central difference.  Raised for noisy energies, never lowered below the min.
LAMBDA_MIN_A = 0.01
LAMBDA_MAX_A = 0.1
#: Force-derivative noise (eV/A) the step choice aims for.
TARGET_NOISE_EV_A = 1e-3

#: Absolute and relative acceptance tolerance on the directional derivative
#: (eV/A), before the noise allowance.
ABS_TOLERANCE_EV_A = 2e-3
REL_TOLERANCE = 0.01
#: Multiple of the energy-rounding-implied derivative noise that is allowed,
#: as in the validation campaign's finite-difference criterion.
NOISE_MULTIPLIER = 10.0
#: If the noise allowance alone exceeds this, the check cannot discriminate a
#: meaningful inconsistency and the run is refused as unresolved.
MAX_NOISE_ALLOWANCE_EV_A = 0.02

_FLOAT32_EPS = float(np.finfo(np.float32).eps)
_FLOAT64_EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class PreflightNaming:
    """Caller-specific wording and error class for one pre-flight.

    The defaults reproduce the historical NEB messages exactly, so NEB call
    sites are unchanged.  confsearch passes its own naming (and error class)
    so its refusals never claim to be NEB refusals.
    """

    context: str = "NEB"
    item_label: str = "Image"
    item_label_lower: str = "image"
    subject: str = "band"
    schema: str = PREFLIGHT_SCHEMA
    sampled_key: str = "sampled_images"
    error_class: type[BaseException] = NEBGradientConsistencyError
    scope: str = (
        "spot check on the sampled band structures and directions; not a "
        "proof of consistency over the whole potential energy surface"
    )
    consequence: str = (
        "NEB optimizes with these forces, so the band and any barrier would be "
        "meaningless. Use a conservative (autograd-force) model or precision."
    )


#: The historical NEB wording; kept as the default for ``run_gradient_preflight``.
NEB_PREFLIGHT_NAMING = PreflightNaming()


def energy_noise_floor_eV(total_energy_eV: float, dtype: str | None) -> float:
    """Energy resolution ``eps * |E|`` for the model's floating-point precision.

    float64 and float32 use their machine epsilon.  An unknown precision is
    treated as float32, the conservative choice for a fail-closed check.
    """
    kind = str(dtype or "").lower()
    eps = _FLOAT64_EPS if "float64" in kind or kind == "double" else _FLOAT32_EPS
    return eps * abs(float(total_energy_eV))


def directional_check(
    energy_fn: Callable[[np.ndarray], float],
    positions: np.ndarray,
    forces: np.ndarray,
    direction: np.ndarray,
    *,
    noise_floor_eV: float,
) -> dict[str, Any]:
    """Compare ``-F.u`` with a Richardson-extrapolated central difference.

    ``direction`` is a unit vector in the ``(N, 3)`` displacement space.
    ``energy_fn`` maps positions to the model energy in eV.  The returned
    record carries every number the verdict is based on.
    """
    analytic = -float(np.sum(forces * direction))
    lam = min(
        LAMBDA_MAX_A,
        max(LAMBDA_MIN_A, noise_floor_eV / (2.0 * TARGET_NOISE_EV_A)),
    )

    def central(step: float) -> float:
        plus = energy_fn(positions + step * direction)
        minus = energy_fn(positions - step * direction)
        return (plus - minus) / (2.0 * step)

    fine = central(lam)
    coarse = central(2.0 * lam)
    richardson = (4.0 * fine - coarse) / 3.0
    noise_fine = noise_floor_eV / (2.0 * lam)
    noise_coarse = noise_floor_eV / (4.0 * lam)
    noise_extrapolated = (4.0 * noise_fine + noise_coarse) / 3.0
    noise_allowance = NOISE_MULTIPLIER * noise_extrapolated
    tolerance = ABS_TOLERANCE_EV_A + REL_TOLERANCE * abs(analytic) + noise_allowance
    values = (analytic, fine, coarse, richardson)
    finite = all(np.isfinite(value) for value in values)
    error = abs(richardson - analytic) if finite else float("inf")
    resolved = noise_allowance <= MAX_NOISE_ALLOWANCE_EV_A
    return {
        "analytic_derivative_eV_A": analytic,
        "fd_fine_eV_A": fine,
        "fd_coarse_eV_A": coarse,
        "fd_richardson_eV_A": richardson,
        "step_fine_A": lam,
        "step_coarse_A": 2.0 * lam,
        "abs_error_eV_A": float(error),
        "noise_allowance_eV_A": float(noise_allowance),
        "tolerance_eV_A": float(tolerance),
        "finite": bool(finite),
        "resolved": bool(resolved),
        "passed": bool(finite and resolved and error <= tolerance),
    }


def _sample_indices(n_images: int) -> list[int]:
    count = min(MAX_SAMPLED_IMAGES, n_images)
    picks = np.linspace(0, n_images - 1, count).round().astype(int)
    return sorted({int(index) for index in picks})


def _free_mask(atoms: Atoms) -> np.ndarray:
    from ase.constraints import FixAtoms

    mask = np.ones(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            mask[np.asarray(constraint.get_indices(), dtype=int)] = False
    return mask


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return vector / norm


def run_gradient_preflight(
    calculator: Any,
    structures: Sequence[Atoms],
    *,
    dtype: str | None,
    minimum_structures: int = 2,
    naming: PreflightNaming = NEB_PREFLIGHT_NAMING,
) -> dict[str, Any]:
    """Check energy-force consistency on a sample of ``structures``.

    ``calculator`` is the ASE calculator the caller will use.  The structures
    are not modified.  Returns the evidence record when every check passes and
    raises ``naming.error_class`` otherwise.

    NEB calls this with its defaults (at least two images, NEB wording).
    confsearch calls it with ``minimum_structures=1`` and its own naming, on the
    first prepared start.
    """
    if minimum_structures < 1:
        raise ValueError("minimum_structures must be >= 1")
    if len(structures) < minimum_structures:
        raise naming.error_class(
            f"{naming.context} gradient pre-flight needs a {naming.subject}"
        )
    checks: list[dict[str, Any]] = []
    for image_index in _sample_indices(len(structures)):
        atoms = structures[image_index]
        free = _free_mask(atoms)
        if not free.any():
            continue
        work = atoms.copy()
        work.set_constraint()  # raw model forces, no constraint masking
        work.calc = calculator
        base_positions = np.asarray(work.get_positions(), dtype=float).copy()

        def energy(positions: np.ndarray, work: Atoms = work) -> float:
            work.set_positions(positions)
            return float(work.get_potential_energy())

        energy0 = energy(base_positions)
        forces = np.asarray(work.get_forces(), dtype=float).copy()
        if not (np.isfinite(energy0) and np.all(np.isfinite(forces))):
            raise naming.error_class(
                f"{naming.item_label} {image_index}: the model returned a "
                f"non-finite energy or force; refusing to start {naming.context}"
            )
        noise = energy_noise_floor_eV(energy0, dtype)
        rng = np.random.default_rng(RNG_SEED + image_index)
        directions: list[tuple[str, np.ndarray]] = []
        masked_force = np.where(free[:, None], forces, 0.0)
        if float(np.linalg.norm(masked_force)) >= MIN_FORCE_NORM_EV_A:
            unit = _unit(masked_force)
            if unit is not None:
                directions.append(("force", unit))
        for _ in range(RANDOM_DIRECTIONS):
            draw = np.where(
                free[:, None], rng.standard_normal(base_positions.shape), 0.0
            )
            unit = _unit(draw)
            if unit is not None:
                directions.append(("random", unit))
        try:
            for kind, direction in directions:
                record = directional_check(
                    energy,
                    base_positions,
                    forces,
                    direction,
                    noise_floor_eV=noise,
                )
                record.update(
                    {
                        "image_index": image_index,
                        "direction": kind,
                        "energy_eV": energy0,
                        "energy_noise_floor_eV": noise,
                    }
                )
                checks.append(record)
        finally:
            work.set_positions(base_positions)
    if not checks:
        raise naming.error_class(
            f"{naming.context} gradient pre-flight found no free atoms to displace"
        )
    failed = [check for check in checks if not check["passed"]]
    errors = [check["abs_error_eV_A"] for check in checks if check["finite"]]
    summary = {
        "schema": naming.schema,
        "status": "failed" if failed else "passed",
        "method": (
            "central finite difference of the model energy along the force "
            "direction and random directions, two step sizes combined by "
            "Richardson extrapolation, compared with -F.u"
        ),
        "scope": naming.scope,
        "dtype_assumed": str(dtype) if dtype else "float32 (unknown)",
        naming.sampled_key: sorted({check["image_index"] for check in checks}),
        "n_checks": len(checks),
        "n_failed": len(failed),
        "max_abs_error_eV_A": max(errors) if errors else None,
        "parameters": {
            "rng_seed": RNG_SEED,
            "abs_tolerance_eV_A": ABS_TOLERANCE_EV_A,
            "rel_tolerance": REL_TOLERANCE,
            "noise_multiplier": NOISE_MULTIPLIER,
            "max_noise_allowance_eV_A": MAX_NOISE_ALLOWANCE_EV_A,
            "step_range_A": [LAMBDA_MIN_A, LAMBDA_MAX_A],
        },
        "checks": checks,
    }
    if failed:
        worst = max(failed, key=lambda check: check["abs_error_eV_A"])
        if not worst["finite"]:
            reason = "a non-finite value"
        elif not worst["resolved"]:
            reason = (
                "an energy resolution too coarse to discriminate the "
                f"derivative (noise allowance {worst['noise_allowance_eV_A']:.3g} "
                "eV/A); use float64 or a smaller cell"
            )
        else:
            reason = (
                f"|FD - (-F.u)| = {worst['abs_error_eV_A']:.3g} eV/A > "
                f"tolerance {worst['tolerance_eV_A']:.3g} eV/A"
            )
        error = naming.error_class(
            f"{naming.context} refused: the model's forces are not verifiably "
            f"the gradient of its energy ({len(failed)}/{len(checks)} checks "
            f"failed; worst: {naming.item_label_lower} {worst['image_index']}, "
            f"{worst['direction']} direction, {reason}). {naming.consequence}"
        )
        error.summary = summary  # type: ignore[attr-defined]
        raise error
    return summary


__all__ = [
    "NEB_PREFLIGHT_NAMING",
    "NEBGradientConsistencyError",
    "PREFLIGHT_SCHEMA",
    "PreflightNaming",
    "directional_check",
    "energy_noise_floor_eV",
    "run_gradient_preflight",
]
