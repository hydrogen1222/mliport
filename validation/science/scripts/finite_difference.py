"""T2: force-energy and stress-energy finite differences (taskbook 12/13).

Run inside a backend environment::

    python finite_difference.py --engine mace --profile-id mace_omat \
        --model ... --manifest ... --out .validation-work/t2-fd \
        [--displacements 5e-4,1e-3,2e-3,5e-3] [--strains 1e-4,3e-4,1e-3,2e-3]

Forces: central energy differences via
``ase.calculators.fd.calculate_numerical_forces`` on 6-12 DOFs chosen across
species, force magnitude and Cartesian direction (deterministic selection).
Stress: ``ase.calculators.fd.calculate_numerical_stress`` (ASE Voigt order
xx,yy,zz,yz,xz,xy), normal and shear components reported separately.  The
acceptance question is whether a *stable convergence region* exists: at
least two ADJACENT displacement/strain values whose per-component error
stays within the bound implied by the total-energy noise over that scale
(``max(1e-3, 10 * eps_dtype*|E|/(2h))`` for forces, the analogous
``noise/(V*eps)`` for stress).  A single lucky point is never a region, and
no single best scale is selected (review R11).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402
import engines  # noqa: E402
import fixtures  # noqa: E402
from ase.calculators.fd import (
    calculate_numerical_forces,  # noqa: E402
    calculate_numerical_stress,  # noqa: E402
)

DISPLACEMENTS_A = (5e-4, 1e-3, 2e-3, 5e-3)
STRAINS = (1e-4, 3e-4, 1e-3, 2e-3)
MAX_FORCE_DOFS = 8
VOIGT_NORMAL = (0, 1, 2)
VOIGT_SHEAR = (3, 4, 5)

#: Absolute acceptance bound for the per-component FD error relative to the
#: analytic derivative (eV/A for forces, eV/A^3 for stress).  A NEB/phonon
#: consumer needs the analytic first derivative to about this accuracy.
ABSOLUTE_FD_FORCE_BOUND_EV_A = 1e-3
ABSOLUTE_FD_STRESS_BOUND_EV_A3 = 1e-3
#: How far above the energy-noise-implied error scale a scale may sit before
#: it stops being explainable by total-energy resolution.
NOISE_SCALE_MULTIPLIER = 10.0
#: A stable region requires at least this many ADJACENT scales to qualify;
#: a single lucky point is never a region (review R11).
MIN_ADJACENT_SCALES = 2
#: Two adjacent noise-explained scales count as a noise-limited plateau only
#: when their errors agree within this factor (guards against a lucky point
#: next to a wildly different neighbour).
NOISE_PLATEAU_FACTOR = 10.0
FLOAT32_EPS = float(np.finfo(np.float32).eps)
FLOAT64_EPS = float(np.finfo(np.float64).eps)


def energy_cancellation_floor_eV(
    total_energy_eV: float,
    dtype: str | None,
    measured_floor_eV: float | None = None,
) -> float:
    """Total-energy resolution floor in eV (documented units and basis).

    A total energy stored in float32 with magnitude |E| is resolved only to
    ``eps32 * |E|`` (~1e-4 eV for a 1 keV cell), and a central difference
    over displacement h turns that into a force error of roughly
    ``eps32 * |E| / (2h)``.  A measured repeatability floor, when available,
    is combined conservatively (larger wins); unknown precision contributes
    nothing beyond the measured floor.
    """
    kind = str(dtype or "")
    if "float32" in kind:
        machine = FLOAT32_EPS
    elif "float64" in kind:
        machine = FLOAT64_EPS
    else:
        machine = 0.0
    cancellation = machine * abs(float(total_energy_eV))
    if measured_floor_eV is not None and np.isfinite(measured_floor_eV):
        return max(cancellation, max(0.0, float(measured_floor_eV)))
    return cancellation


def _adjacent_pairs(indices: list[int]) -> list[list[int]]:
    """Consecutive index pairs from a sorted list."""
    return [[a, b] for a, b in zip(indices, indices[1:], strict=False) if b == a + 1]


def _region_report(
    entries: list[dict[str, Any]],
    strict_bound: float,
    *,
    noise_scale_key: str,
) -> dict[str, Any]:
    """Whole-sweep region verdict: never a single minimum (review R11).

    Two admissible region kinds:

    * *measured* -- at least two ADJACENT scales whose max component error
      is within the absolute accuracy bound the consumer needs;
    * *noise-limited* -- no measured region exists, but at least two ADJACENT
      scales stay within the total-energy-noise-implied error scale and are
      mutually consistent (a plateau within ``NOISE_PLATEAU_FACTOR``).  That
      is an unresolved measurement, reported as ``insufficient_sampling``,
      never as a pass and never as a backend failure.

    Anything else fails.  A lone lucky point can satisfy neither definition
    because both require an adjacent partner with comparable quality.
    """
    measured_idx = [
        i for i, m in enumerate(entries) if m["max_abs_error"] <= strict_bound
    ]
    noise_idx = [
        i
        for i, m in enumerate(entries)
        if m["noise_limited"] and m["noise_explained"]
    ]
    adjacent_measured = _adjacent_pairs(measured_idx)
    adjacent_noise = []
    for a, b in _adjacent_pairs(noise_idx):
        ea, eb = entries[a]["max_abs_error"], entries[b]["max_abs_error"]
        hi, lo = max(ea, eb), max(min(ea, eb), 1e-30)
        if hi <= NOISE_PLATEAU_FACTOR * lo:
            adjacent_noise.append([a, b])

    errors = [m["max_abs_error"] for m in entries]
    status = "fail"
    kind = None
    if adjacent_measured:
        status = "characterized"
        a, b = adjacent_measured[0]
        kind = "plateau" if errors[b] >= 0.5 * errors[a] else "convergence"
    elif adjacent_noise:
        status = "insufficient_sampling"
        kind = "noise_limited_plateau"
    return {
        "status": status,
        "min_adjacent_scales": MIN_ADJACENT_SCALES,
        "bound": strict_bound,
        "noise_plateau_factor": NOISE_PLATEAU_FACTOR,
        "noise_scale_key": noise_scale_key,
        "n_qualifying_scales": len(measured_idx),
        "qualifying_scales": [entries[i]["scale"] for i in measured_idx],
        "adjacent_qualifying_pairs": [
            [entries[a]["scale"], entries[b]["scale"]] for a, b in adjacent_measured
        ],
        "adjacent_noise_limited_pairs": [
            [entries[a]["scale"], entries[b]["scale"]] for a, b in adjacent_noise
        ],
        "stable_region_found": status != "fail",
        "stable_region_kind": kind,
        "per_scale_qualified": [
            {
                "scale": m["scale"],
                "qualifies": m["qualifies"],
                "noise_limited": m["noise_limited"],
                "noise_explained": m["noise_explained"],
                "bound": strict_bound,
                "noise_scale": m[noise_scale_key],
            }
            for m in entries
        ],
        "max_abs_error_by_scale": errors,
        "biased_scales": [
            m["scale"] for m in entries if not m["noise_explained"] and not m["qualifies"]
        ],
    }


def _stress_is_unsupported(exc: Exception) -> bool:
    """Only a genuine 'property not available' is 'unsupported' (R11).

    A backend execution failure (CUDA error, OOM, shape mismatch, ...) must
    never be relabelled as an unsupported property.
    """
    from ase.calculators.calculator import PropertyNotImplementedError

    if isinstance(exc, (PropertyNotImplementedError, NotImplementedError)):
        return True
    text = str(exc).lower()
    return "not implemented" in text or "not supported" in text


def force_fd_case(
    ctx: engines.EngineContext,
    system: str,
    atoms,
    displacements: tuple[float, ...],
    device: str,
    measured_energy_floor_eV: float | None = None,
) -> dict[str, Any]:
    atoms.calc = ctx.calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    dofs = select_force_dofs(atoms, forces, MAX_FORCE_DOFS)
    atoms_positions0 = atoms.positions.copy()

    noise_floor = energy_cancellation_floor_eV(
        energy, ctx.dtype, measured_energy_floor_eV
    )

    per_disp: list[dict[str, Any]] = []
    for h in displacements:
        errors: list[float] = []
        rel_errors: list[float] = []
        for iatom, icart in dofs:
            f_num = calculate_numerical_forces(
                atoms, eps=h, iatoms=[iatom], icarts=[icart]
            )
            analytic = forces[iatom, icart]
            # ASE 3.29 returns the selected (iatoms, icarts) sub-block only
            err = float(f_num[0, 0] - analytic)
            errors.append(err)
            denom = abs(analytic)
            if denom > 1e-6:  # relative error only where denominator meaningful
                rel_errors.append(abs(err) / denom)
            atoms.positions[:] = atoms_positions0
        max_abs = float(np.max(np.abs(errors)))
        noise_scale = noise_floor / (2.0 * h) if h > 0 else float("inf")
        per_disp.append(
            {
                "displacement_A": h,
                "scale": h,
                "max_abs_error_eV_A": max_abs,
                "max_abs_error": max_abs,
                "rms_error_eV_A": float(np.sqrt(np.mean(np.square(errors)))),
                "max_rel_error": (float(np.max(rel_errors)) if rel_errors else None),
                "mean_abs_error_eV_A": float(np.mean(np.abs(errors))),
                "noise_force_scale_eV_A": noise_scale,
                "noise_limited": bool(noise_scale > ABSOLUTE_FD_FORCE_BOUND_EV_A),
                "noise_explained": bool(max_abs <= NOISE_SCALE_MULTIPLIER * noise_scale),
                "bound": ABSOLUTE_FD_FORCE_BOUND_EV_A,
                "qualifies": bool(max_abs <= ABSOLUTE_FD_FORCE_BOUND_EV_A),
            }
        )
        atoms.positions[:] = atoms_positions0

    region = _region_report(
        per_disp,
        ABSOLUTE_FD_FORCE_BOUND_EV_A,
        noise_scale_key="noise_force_scale_eV_A",
    )
    acceptance = {
        "stable_region_bound_eV_A": ABSOLUTE_FD_FORCE_BOUND_EV_A,
        "noise_scale_multiplier": NOISE_SCALE_MULTIPLIER,
        "noise_plateau_factor": NOISE_PLATEAU_FACTOR,
        "energy_noise_floor_eV": noise_floor,
        "rule": (
            "measured region: >= 2 ADJACENT displacements with max |FD - "
            "analytic| <= 1e-3 eV/A. If none exists, >= 2 adjacent "
            "displacements may still be a noise-limited plateau when "
            "error <= 10 * noise_floor/(2h) at both and the two errors agree "
            "within 10x -- reported as insufficient_sampling, not a pass. "
            "An isolated qualifying point is never a region and no single "
            "best displacement is selected."
        ),
    }
    return {
        "acceptance": acceptance,
        "case_id": f"t2fd_{system}",
        "test_id": "t2_force_fd",
        "status": region["status"],
        "energy_eV": energy,
        "dofs": [
            {
                "atom": int(i),
                "component": int(c),
                "analytic_force_eV_A": float(forces[i, c]),
            }
            for i, c in dofs
        ],
        "per_displacement": per_disp,
        "region": region,
    }


def stress_fd_case(
    ctx: engines.EngineContext,
    system: str,
    atoms,
    strains: tuple[float, ...],
    device: str,
    measured_energy_floor_eV: float | None = None,
) -> dict[str, Any]:
    atoms.calc = ctx.calculator
    try:
        stress = np.asarray(atoms.get_stress(), dtype=float)
    except Exception as exc:  # noqa: BLE001 - classified below
        if _stress_is_unsupported(exc):
            return {
                "case_id": f"t2fd_{system}",
                "test_id": "t2_stress_fd",
                "status": "unsupported",
                "reason": f"model profile does not provide stress: {exc}",
            }
        # A real backend failure must surface as an execution failure, not
        # as an unsupported property (review R11).
        msg = f"stress evaluation failed: {type(exc).__name__}: {exc}"
        raise RuntimeError(msg) from exc
    if not np.all(np.isfinite(stress)):
        raise ValueError("analytic stress contains NaN or Inf")
    energy = float(atoms.get_potential_energy())
    cell0 = atoms.cell.array.copy()
    positions0 = atoms.positions.copy()
    volume = atoms.get_volume()
    noise_floor = energy_cancellation_floor_eV(
        energy, ctx.dtype, measured_energy_floor_eV
    )

    per_strain: list[dict[str, Any]] = []
    for eps in strains:
        num = calculate_numerical_stress(atoms, eps=eps, voigt=True)
        errors = num - stress
        atoms.set_cell(cell0, scale_atoms=False)
        atoms.positions[:] = positions0
        normal_max = float(np.max(np.abs(errors[list(VOIGT_NORMAL)])))
        shear_max = float(np.max(np.abs(errors[list(VOIGT_SHEAR)])))
        noise_scale = noise_floor / (volume * eps) if eps > 0 else float("inf")
        entry = {
            "strain_eps": eps,
            "scale": eps,
            "normal": {
                "max_abs_error_eV_A3": normal_max,
                "mae_eV_A3": float(np.mean(np.abs(errors[list(VOIGT_NORMAL)]))),
                "mae_GPa": float(
                    np.mean(np.abs(errors[list(VOIGT_NORMAL)])) * common.EV_A3_TO_GPA
                ),
            },
            "shear": {
                "max_abs_error_eV_A3": shear_max,
                "mae_eV_A3": float(np.mean(np.abs(errors[list(VOIGT_SHEAR)]))),
                "mae_GPa": float(
                    np.mean(np.abs(errors[list(VOIGT_SHEAR)])) * common.EV_A3_TO_GPA
                ),
            },
            "noise_stress_scale_eV_A3": noise_scale,
            "noise_limited": bool(noise_scale > ABSOLUTE_FD_STRESS_BOUND_EV_A3),
            "noise_explained": bool(
                max(normal_max, shear_max) <= NOISE_SCALE_MULTIPLIER * noise_scale
            ),
            "bound": ABSOLUTE_FD_STRESS_BOUND_EV_A3,
            "max_abs_error": max(normal_max, shear_max),
            "qualifies": bool(
                normal_max <= ABSOLUTE_FD_STRESS_BOUND_EV_A3
                and shear_max <= ABSOLUTE_FD_STRESS_BOUND_EV_A3
            ),
            "analytic_stress_voigt_eV_A3": stress.tolist(),
            "numerical_stress_voigt_eV_A3": num.tolist(),
            "volume_A3": volume,
        }
        per_strain.append(entry)
        atoms.set_cell(cell0, scale_atoms=False)
        atoms.positions[:] = positions0

    region = _region_report(
        per_strain,
        ABSOLUTE_FD_STRESS_BOUND_EV_A3,
        noise_scale_key="noise_stress_scale_eV_A3",
    )
    acceptance = {
        "stable_region_bound_eV_A3": ABSOLUTE_FD_STRESS_BOUND_EV_A3,
        "noise_scale_multiplier": NOISE_SCALE_MULTIPLIER,
        "noise_plateau_factor": NOISE_PLATEAU_FACTOR,
        "energy_noise_floor_eV": noise_floor,
        "note": (
            "normal and shear components are judged separately; ASE Voigt order"
            " xx,yy,zz,yz,xz,xy; no manual pressure sign flip applied. A strain"
            " qualifies when both components satisfy max |FD - analytic| <="
            " 1e-3 eV/A^3 and at least two ADJACENT qualifying strains are"
            " required. A noise-limited plateau (error <= 10 * noise_floor/"
            "(V*eps) at two adjacent strains, agreeing within 10x) is reported"
            " as insufficient_sampling, never as a pass."
        ),
    }
    return {
        "acceptance": acceptance,
        "case_id": f"t2fd_{system}",
        "test_id": "t2_stress_fd",
        "status": region["status"],
        "analytic_stress_voigt_eV_A3": stress.tolist(),
        "analytic_stress_voigt_GPa": (stress * common.EV_A3_TO_GPA).tolist(),
        "per_strain": per_strain,
        "region": region,
    }


def select_force_dofs(
    atoms, forces: np.ndarray, max_dofs: int
) -> list[tuple[int, int]]:
    """Deterministic DOF spread: strongest per-species atoms first, then
    weakest atoms, alternating Cartesian directions."""
    natoms = len(atoms)
    order_by_force = np.argsort(-np.abs(forces).max(axis=1), kind="stable")
    species: dict[int, list[int]] = {}
    for i in range(natoms):
        species.setdefault(int(atoms.numbers[i]), []).append(i)

    chosen: list[tuple[int, int]] = []
    used_species_iters = {z: iter(idx) for z, idx in species.items()}
    # interleave strongest atoms across species
    seen_species: list[int] = []
    while len(chosen) < max_dofs:
        progressed = False
        for z in sorted(used_species_iters):
            for i in used_species_iters[z]:
                if i in {c[0] for c in chosen}:
                    continue
                comp = int(np.argmax(np.abs(forces[i])))
                chosen.append((i, comp))
                seen_species.append(z)
                progressed = True
                break
            if len(chosen) >= max_dofs:
                break
        if not progressed:
            break
    # fill remaining DOFs from weakest-force atoms, rotating directions
    if len(chosen) < max_dofs:
        weak_order = order_by_force[::-1]
        dir_cycle = 0
        for i in weak_order:
            for _ in range(3):
                comp = dir_cycle % 3
                dir_cycle += 1
                if (int(i), comp) not in chosen:
                    chosen.append((int(i), comp))
                    if len(chosen) >= max_dofs:
                        return chosen
                    break
    return chosen[:max_dofs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=engines._ENGINES)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--neighbor-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="GRACE only: toggle the mliport neighbor-list cache",
    )
    parser.add_argument("--head", default=None)
    parser.add_argument("--dtype", default=None, help="MACE only")
    parser.add_argument("--tag", default=None, help="filename suffix")
    parser.add_argument("--systems", default=",".join(fixtures.INVARIANCE_SYSTEMS))
    parser.add_argument("--displacements", default=",".join(map(str, DISPLACEMENTS_A)))
    parser.add_argument("--strains", default=",".join(map(str, STRAINS)))
    parser.add_argument(
        "--energy-noise-floor-eV",
        type=float,
        default=None,
        help=(
            "measured total-energy repeatability floor (eV); combined with "
            "the dtype cancellation scale eps*|E| to derive the noise-implied "
            "FD error bound per scale"
        ),
    )
    args = parser.parse_args()

    manifest = common.load_model_manifest(args.manifest)
    profile = common.resolve_profile(manifest, args.profile_id, args.model)
    out_dir = Path(args.out)
    disps = tuple(float(x) for x in args.displacements.split(",") if x)
    strains = tuple(float(x) for x in args.strains.split(",") if x)

    try:
        ctx = engines.build_engine(
            args.engine,
            args.profile_id,
            profile,
            args.model,
            device=args.device,
            dtype=args.dtype,
            head=args.head,
            neighbor_cache=args.neighbor_cache,
        )
    except Exception as exc:  # noqa: BLE001
        rec = common.result_record(
            case_id="t2fd_setup",
            test_id="t2_fd",
            status="fail",
            engine=args.engine,
            model_identity=profile.get("identity", args.profile_id),
            model_sha256=profile["model_sha256"],
            task=profile.get("task"),
            head=args.head,
            dtype=args.dtype or "upstream/model-defined",
            device=common.device_record(args.device),
            input_structure_id=None,
            parameters={"displacements_A": disps, "strains": strains},
            metrics={},
            diagnostics={"traceback": traceback.format_exc()},
            exception=f"{type(exc).__name__}: {exc}",
        )
        common.write_result(rec, out_dir, tag=args.tag)
        return 1

    exit_code = 0
    for system in [s for s in args.systems.split(",") if s]:
        atoms = (
            fixtures.build_fixture(system)[0]
            if system in fixtures.FIXTURES
            else fixtures.build_extra(system)[0]
        )
        try:
            payload = {
                "force": force_fd_case(
                    ctx,
                    system,
                    atoms,
                    disps,
                    args.device,
                    measured_energy_floor_eV=args.energy_noise_floor_eV,
                ),
            }
            payload["stress"] = stress_fd_case(
                ctx,
                system,
                atoms,
                strains,
                args.device,
                measured_energy_floor_eV=args.energy_noise_floor_eV,
            )
            for key, case in payload.items():
                device = common.device_record(args.device)
                record = common.result_record(
                    case_id=case["case_id"],
                    test_id=case["test_id"],
                    status=case["status"],
                    engine=ctx.engine,
                    model_identity=ctx.model_identity,
                    model_sha256=ctx.model_sha256,
                    task=ctx.task,
                    head=ctx.head,
                    dtype=ctx.dtype,
                    device=device,
                    input_structure_id=common.structure_id(atoms),
                    parameters={
                        "displacements_A": disps,
                        "strains": strains,
                        "method": "ase.calculators.fd central differences",
                        "energy_noise_floor_eV": args.energy_noise_floor_eV,
                        "noise_scale_multiplier": NOISE_SCALE_MULTIPLIER,
                    },
                    metrics=case,
                    diagnostics={
                        **ctx.diagnostics,
                        "backend_version": ctx.backend_version, "framework_version": ctx.framework_version,
                    },
                    peak_vram=common.peak_vram_mib(),
                )
                common.write_result(record, out_dir, tag=args.tag)
                print(
                    f"[t2fd] {ctx.engine} {system}/{key}: {case['status']} "
                    + json.dumps(
                        case.get("per_displacement", case.get("per_strain", []))[:1],
                        default=str,
                    )[:160]
                )
                if case["status"] == "fail":
                    exit_code = 1
        except Exception as exc:  # noqa: BLE001
            record = common.result_record(
                case_id=f"t2fd_{system}",
                test_id="t2_fd_error",
                status="fail",
                engine=ctx.engine,
                model_identity=ctx.model_identity,
                model_sha256=ctx.model_sha256,
                task=ctx.task,
                head=ctx.head,
                dtype=ctx.dtype,
                device=common.device_record(args.device),
                input_structure_id=common.structure_id(atoms),
                parameters={},
                metrics={},
                diagnostics={"traceback": traceback.format_exc()},
                exception=f"{type(exc).__name__}: {exc}",
            )
            common.write_result(record, out_dir, tag=args.tag)
            print(f"[t2fd] {ctx.engine} {system}: ERROR {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
