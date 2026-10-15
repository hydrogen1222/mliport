"""Runtime energy-force consistency pre-flight for NEB.

NEB feeds the model's forces to the optimizer as if they were ``-grad(E)``.
That holds by construction for models that obtain forces by automatic
differentiation of the energy, but it is a property of the exact model,
framework build and device, not of the checkpoint alone.  Instead of looking
the property up in static evidence recorded on other software stacks, this
module measures it on the band that is about to be optimized.

For a few structures of the band (both endpoints and interpolated images) and
a few displacement directions ``u`` (the force direction and random
directions on the free atoms) it compares the model's force projection
``-F.u`` with a central finite difference of the model's energy.  Two step
sizes are combined by Richardson extrapolation, which removes the leading
truncation error, and the acceptance tolerance includes the energy rounding
noise implied by the model's floating-point precision (the same noise-aware
reasoning as ``validation/science/scripts/finite_difference.py``).

This is a spot check, not a proof of consistency over the whole potential
energy surface.  It fails closed: an inconsistency, a non-finite value or a
measurement that the energy resolution cannot resolve raises
:class:`NEBGradientConsistencyError` before any NEB step is taken.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    images: Sequence[Atoms],
    *,
    dtype: str | None,
) -> dict[str, Any]:
    """Check energy-force consistency on a sample of the band.

    ``calculator`` is the ASE calculator the NEB will use.  The images are not
    modified.  Returns the evidence record when every check passes and raises
    :class:`NEBGradientConsistencyError` otherwise.
    """
    if len(images) < 2:
        raise NEBGradientConsistencyError("NEB gradient pre-flight needs a band")
    checks: list[dict[str, Any]] = []
    for image_index in _sample_indices(len(images)):
        atoms = images[image_index]
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
            raise NEBGradientConsistencyError(
                f"Image {image_index}: the model returned a non-finite energy "
                "or force; refusing to start NEB"
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
        raise NEBGradientConsistencyError(
            "NEB gradient pre-flight found no free atoms to displace"
        )
    failed = [check for check in checks if not check["passed"]]
    errors = [check["abs_error_eV_A"] for check in checks if check["finite"]]
    summary = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "failed" if failed else "passed",
        "method": (
            "central finite difference of the model energy along the force "
            "direction and random directions, two step sizes combined by "
            "Richardson extrapolation, compared with -F.u"
        ),
        "scope": (
            "spot check on the sampled band structures and directions; not a "
            "proof of consistency over the whole potential energy surface"
        ),
        "dtype_assumed": str(dtype) if dtype else "float32 (unknown)",
        "sampled_images": sorted({check["image_index"] for check in checks}),
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
        error = NEBGradientConsistencyError(
            "NEB refused: the model's forces are not verifiably the gradient of "
            f"its energy ({len(failed)}/{len(checks)} checks failed; worst: image "
            f"{worst['image_index']}, {worst['direction']} direction, {reason}). "
            "NEB optimizes with these forces, so the band and any barrier would be "
            "meaningless. Use a conservative (autograd-force) model or precision."
        )
        error.summary = summary  # type: ignore[attr-defined]
        raise error
    return summary


__all__ = [
    "NEBGradientConsistencyError",
    "directional_check",
    "energy_noise_floor_eV",
    "run_gradient_preflight",
]
