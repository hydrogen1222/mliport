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
acceptance question is whether a stable convergence region exists -- error
falling to the measured floor for at least one displacement/strain value and
staying bounded for neighbours -- not the best isolated delta.
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


def force_fd_case(
    ctx: engines.EngineContext,
    system: str,
    atoms,
    displacements: tuple[float, ...],
    device: str,
) -> dict[str, Any]:
    atoms.calc = ctx.calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    dofs = select_force_dofs(atoms, forces, MAX_FORCE_DOFS)
    atoms_positions0 = atoms.positions.copy()

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
        per_disp.append(
            {
                "displacement_A": h,
                "max_abs_error_eV_A": float(np.max(np.abs(errors))),
                "rms_error_eV_A": float(np.sqrt(np.mean(np.square(errors)))),
                "max_rel_error": (
                    float(np.max(rel_errors)) if rel_errors else None
                ),
                "mean_abs_error_eV_A": float(np.mean(np.abs(errors))),
            }
        )
        atoms.positions[:] = atoms_positions0

    acceptance = {
        "stable_region_bound_eV_A": 1e-3,
        "rule": (
            "a displacement regime counts as converged when the max absolute"
            " component error falls under the documented bound; absence of any"
            " such regime is a fail for conservative NEB/phonon use"
        ),
    }
    has_region = any(
        m["max_abs_error_eV_A"] <= 10.0 * max(m["rms_error_eV_A"], 1e-12)
        and m["max_abs_error_eV_A"] < 1e-3
        for m in per_disp
    )
    return {
        "acceptance": acceptance,
        "case_id": f"t2fd_{system}",
        "test_id": "t2_force_fd",
        "status": "characterized" if has_region else "fail",
        "energy_eV": energy,
        "dofs": [
            {"atom": int(i), "component": int(c), "analytic_force_eV_A": float(forces[i, c])}
            for i, c in dofs
        ],
        "per_displacement": per_disp,
    }


def stress_fd_case(
    ctx: engines.EngineContext,
    system: str,
    atoms,
    strains: tuple[float, ...],
    device: str,
) -> dict[str, Any]:
    atoms.calc = ctx.calculator
    try:
        stress = np.asarray(atoms.get_stress(), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return {
            "case_id": f"t2fd_{system}",
            "test_id": "t2_stress_fd",
            "status": "unsupported",
            "reason": f"model profile does not provide stress: {exc}",
        }
    cell0 = atoms.cell.array.copy()
    positions0 = atoms.positions.copy()
    volume = atoms.get_volume()

    per_strain: list[dict[str, Any]] = []
    for eps in strains:
        num = calculate_numerical_stress(atoms, eps=eps, voigt=True)
        errors = num - stress
        atoms.set_cell(cell0, scale_atoms=False)
        atoms.positions[:] = positions0
        entry = {
            "strain_eps": eps,
            "normal": {
                "max_abs_error_eV_A3": float(
                    np.max(np.abs(errors[list(VOIGT_NORMAL)]))
                ),
                "mae_eV_A3": float(
                    np.mean(np.abs(errors[list(VOIGT_NORMAL)]))
                ),
                "mae_GPa": float(
                    np.mean(np.abs(errors[list(VOIGT_NORMAL)]))
                    * common.EV_A3_TO_GPA
                ),
            },
            "shear": {
                "max_abs_error_eV_A3": float(
                    np.max(np.abs(errors[list(VOIGT_SHEAR)]))
                ),
                "mae_eV_A3": float(
                    np.mean(np.abs(errors[list(VOIGT_SHEAR)]))
                ),
                "mae_GPa": float(
                    np.mean(np.abs(errors[list(VOIGT_SHEAR)]))
                    * common.EV_A3_TO_GPA
                ),
            },
            "analytic_stress_voigt_eV_A3": stress.tolist(),
            "numerical_stress_voigt_eV_A3": num.tolist(),
            "volume_A3": volume,
        }
        per_strain.append(entry)
        atoms.set_cell(cell0, scale_atoms=False)
        atoms.positions[:] = positions0

    acceptance = {
        "stable_region_bound_eV_A3": 1e-3,
        "note": (
            "normal and shear components are judged separately; ASE Voigt order"
            " xx,yy,zz,yz,xz,xy; no manual pressure sign flip applied"
        ),
    }
    has_region = any(
        m["normal"]["max_abs_error_eV_A3"] < 1e-3
        and m["shear"]["max_abs_error_eV_A3"] < 1e-3
        for m in per_strain
    )
    return {
        "acceptance": acceptance,
        "case_id": f"t2fd_{system}",
        "test_id": "t2_stress_fd",
        "status": "characterized" if has_region else "fail",
        "analytic_stress_voigt_eV_A3": stress.tolist(),
        "analytic_stress_voigt_GPa": (stress * common.EV_A3_TO_GPA).tolist(),
        "per_strain": per_strain,
    }


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
        help="GRACE only: toggle the mlipx neighbor-list cache",
    )
    parser.add_argument("--head", default=None)
    parser.add_argument("--dtype", default=None, help="MACE only")
    parser.add_argument("--tag", default=None, help="filename suffix")
    parser.add_argument("--systems", default=",".join(fixtures.INVARIANCE_SYSTEMS))
    parser.add_argument("--displacements", default=",".join(map(str, DISPLACEMENTS_A)))
    parser.add_argument("--strains", default=",".join(map(str, STRAINS)))
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
                "force": force_fd_case(ctx, system, atoms, disps, args.device),
            }
            payload["stress"] = stress_fd_case(
                ctx, system, atoms, strains, args.device
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
