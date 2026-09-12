"""T2: invariance and state-hygiene tests on real backend models (taskbook
section 11).

Run inside a backend environment::

    python invariance.py --engine mace --profile-id mace_omat \
        --model /path/to/mace-omat-0-medium.model \
        --manifest validation/science/model_manifest.json \
        --out .validation-work/t2 [--device cuda:0] [--head N] [--dtype N]
        [--neighbor-cache on|off] [--tag cache-on]

Order of operations follows section 11.1: measure native repeatability
first, then use a documented multiplier of the measured floor (plus an
absolute floor) as the tolerance for the transformed-structure comparisons.
The policy is recorded verbatim in every result record.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402
import engines  # noqa: E402
import fixtures  # noqa: E402
import transforms  # noqa: E402

REPEAT_CALLS = 5
TOLERANCE_MULTIPLIER = 10.0
ABS_FLOOR_E = 1e-10  # eV
ABS_FLOOR_F = 1e-9  # eV/A
ABS_FLOOR_S = 1e-9  # eV/A^3
PERM_SEED = 20260911
PERTURB_SEED = 20260911
DISPLACEMENT_A = 0.03
TRANSLATION_A = (0.013, 0.077, 0.031)
ROTATION_ANGLE_RAD = 0.37
ROTATION_AXIS = (1.0, 1.0, 1.0)

FLOOR_POLICY = {
    "repeatability_calls": REPEAT_CALLS,
    "tolerance_multiplier": TOLERANCE_MULTIPLIER,
    "absolute_floor_energy_eV": ABS_FLOOR_E,
    "absolute_floor_force_eV_A": ABS_FLOOR_F,
    "absolute_floor_stress_eV_A3": ABS_FLOOR_S,
    "rule": (
        "tolerance = max(multiplier * measured floor, absolute floor); "
        "floors measured per system/profile before transformed comparisons"
    ),
}


def _system_atoms(name: str):
    if name in fixtures.FIXTURES:
        return fixtures.build_fixture(name)[0]
    return fixtures.build_extra(name)[0]


def _full_eval(atoms) -> tuple[float, np.ndarray, np.ndarray | None]:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    try:
        stress = np.asarray(atoms.get_stress(), dtype=float)
    except Exception:  # noqa: BLE001 - stress may be unsupported
        stress = None
    return energy, forces, stress


def measure_floor(atoms, device: str) -> dict[str, Any]:
    """Section 11.1: repeated identical inference after a warm-up call.

    Repetitions are forced through :class:`common.InferenceProbe`: the ASE
    result cache is invalidated before every call and the number of real
    ``calculate`` calls is recorded, so a cache hit can never be reported as
    a zero-noise floor (review R02).  Coordinates are never perturbed.
    """
    with common.InferenceProbe(atoms) as probe:
        probe.run()  # warm-up, counted separately
        warmup_calls = probe.calls
        energies, forces_list, stresses = [], [], []
        for _ in range(REPEAT_CALLS):
            (e, f, s), _seconds = common.timed(probe.run, device)
            energies.append(e)
            forces_list.append(f)
            if s is not None:
                stresses.append(s)
        floor: dict[str, Any] = {
            "energy_spread_eV": float(np.ptp(energies)),
            "force_component_spread_eV_A": float(
                max(np.ptp(np.stack([f.ravel() for f in forces_list]), axis=0))
            ),
            "repeat_calls_requested": REPEAT_CALLS,
            "warmup_calculate_calls": int(warmup_calls),
            "real_calculate_calls": int(probe.calls),
            "calculate_call_count_verified": bool(probe.wrapped_calculate),
            "ase_result_cache_invalidated_per_repeat": True,
            "backend_internal_caches_invalidated": False,
        }
        if stresses:
            floor["stress_component_spread_eV_A3"] = float(
                max(np.ptp(np.stack([s.ravel() for s in stresses]), axis=0))
            )
    return floor


def _tol(floor_val: float | None, abs_floor: float) -> float:
    if floor_val is None or not np.isfinite(floor_val):
        return abs_floor
    return max(TOLERANCE_MULTIPLIER * floor_val, abs_floor)


def check_repeatability(floor: dict[str, float]) -> tuple[str, dict[str, Any]]:
    """Section 11.1: this is a measurement, so record it as characterized."""
    return "characterized", dict(floor)

def check_ab_reversibility(atoms, floor: dict[str, float]) -> tuple[str, dict[str, Any]]:
    e_a1, f_a1, s_a1 = _full_eval(atoms)
    rng = np.random.default_rng(PERTURB_SEED)
    displaced = atoms.copy()
    disp = rng.normal(scale=DISPLACEMENT_A, size=(len(atoms), 3))
    displaced.positions = atoms.positions + disp
    displaced.calc = atoms.calc
    _full_eval(displaced)
    # A2: the original Atoms object still shares the calculator; this
    # re-evaluation is what exposes stale caches and state leakage.
    e_a2, f_a2, s_a2 = _full_eval(atoms)
    tol_e = _tol(floor["energy_spread_eV"], ABS_FLOOR_E)
    tol_f = _tol(floor["force_component_spread_eV_A"], ABS_FLOOR_F)
    de = abs(e_a2 - e_a1)
    df = float(np.abs(f_a2 - f_a1).max()) if len(atoms) else 0.0
    metrics: dict[str, Any] = {
        "energy_delta_eV": de,
        "force_delta_max_eV_A": df,
        "tolerance_energy_eV": tol_e,
        "tolerance_force_eV_A": tol_f,
    }
    ok = de <= tol_e and df <= tol_f
    if s_a1 is not None and s_a2 is not None:
        ds = float(np.abs(s_a2 - s_a1).max())
        tol_s = _tol(floor.get("stress_component_spread_eV_A3"), ABS_FLOOR_S)
        metrics["stress_delta_max_eV_A3"] = ds
        metrics["tolerance_stress_eV_A3"] = tol_s
        ok = ok and ds <= tol_s
    return ("pass" if ok else "fail"), metrics


def check_permutation(atoms, floor: dict[str, float]) -> tuple[str, dict[str, Any]]:
    rng = np.random.default_rng(PERM_SEED)
    perm = rng.permutation(len(atoms))
    e0, f0, s0 = _full_eval(atoms)
    permuted = atoms[perm]
    permuted.calc = atoms.calc
    e1, f1, s1 = _full_eval(permuted)
    tol_e = _tol(floor["energy_spread_eV"], ABS_FLOOR_E)
    tol_f = _tol(floor["force_component_spread_eV_A"], ABS_FLOOR_F)
    de = abs(e1 - e0)
    df = float(np.abs(f1 - f0[perm]).max()) if len(atoms) else 0.0
    metrics = {
        "energy_delta_eV": de,
        "force_delta_max_eV_A": df,
        "tolerance_energy_eV": tol_e,
        "tolerance_force_eV_A": tol_f,
    }
    ok = de <= tol_e and df <= tol_f
    if s0 is not None and s1 is not None:
        ds = float(np.abs(s1 - s0).max())
        tol_s = _tol(floor.get("stress_component_spread_eV_A3"), ABS_FLOOR_S)
        metrics["stress_delta_max_eV_A3"] = ds
        metrics["tolerance_stress_eV_A3"] = tol_s
        ok = ok and ds <= tol_s
    return ("pass" if ok else "fail"), metrics


def check_pbc_wrap(atoms, floor: dict[str, float]) -> tuple[str, dict[str, Any]]:
    e0, f0, s0 = _full_eval(atoms)
    wrapped = atoms.copy()
    rng = np.random.default_rng(PERM_SEED)
    lattice = np.asarray(atoms.cell.array, dtype=float)
    # translate a deterministic half of the atoms by random integer lattice vectors
    n_int = rng.integers(-2, 3, size=(len(atoms), 3))
    mask = np.arange(len(atoms)) % 2 == 0
    n_int[~mask] = 0
    wrapped.positions = atoms.positions + n_int @ lattice
    wrapped.wrap()
    wrapped.calc = atoms.calc
    e1, f1, s1 = _full_eval(wrapped)
    tol_e = _tol(floor["energy_spread_eV"], ABS_FLOOR_E)
    tol_f = _tol(floor["force_component_spread_eV_A"], ABS_FLOOR_F)
    metrics = {
        "energy_delta_eV": abs(e1 - e0),
        "force_delta_max_eV_A": float(np.abs(f1 - f0).max()) if len(atoms) else 0.0,
        "tolerance_energy_eV": tol_e,
        "tolerance_force_eV_A": tol_f,
        "atoms_translated": int(mask.sum()),
    }
    ok = metrics["energy_delta_eV"] <= tol_e and metrics["force_delta_max_eV_A"] <= tol_f
    if s0 is not None and s1 is not None:
        ds = float(np.abs(s1 - s0).max())
        tol_s = _tol(floor.get("stress_component_spread_eV_A3"), ABS_FLOOR_S)
        metrics["stress_delta_max_eV_A3"] = ds
        metrics["tolerance_stress_eV_A3"] = tol_s
        ok = ok and ds <= tol_s
    return ("pass" if ok else "fail"), metrics


def check_translation(atoms, floor: dict[str, float]) -> tuple[str, dict[str, Any]]:
    e0, f0, s0 = _full_eval(atoms)
    shifted = atoms.copy()
    shifted.positions = atoms.positions + np.asarray(TRANSLATION_A)
    shifted.wrap()
    shifted.calc = atoms.calc
    e1, f1, s1 = _full_eval(shifted)
    tol_e = _tol(floor["energy_spread_eV"], ABS_FLOOR_E)
    tol_f = _tol(floor["force_component_spread_eV_A"], ABS_FLOOR_F)
    metrics = {
        "energy_delta_eV": abs(e1 - e0),
        "force_delta_max_eV_A": float(np.abs(f1 - f0).max()) if len(atoms) else 0.0,
        "translation_A": list(TRANSLATION_A),
        "tolerance_energy_eV": tol_e,
        "tolerance_force_eV_A": tol_f,
    }
    ok = metrics["energy_delta_eV"] <= tol_e and metrics["force_delta_max_eV_A"] <= tol_f
    if s0 is not None and s1 is not None:
        ds = float(np.abs(s1 - s0).max())
        tol_s = _tol(floor.get("stress_component_spread_eV_A3"), ABS_FLOOR_S)
        metrics["stress_delta_max_eV_A3"] = ds
        metrics["tolerance_stress_eV_A3"] = tol_s
        ok = ok and ds <= tol_s
    return ("pass" if ok else "fail"), metrics


def check_rotation(atoms, floor: dict[str, float]) -> tuple[str, dict[str, Any]]:
    e0, f0, s0 = _full_eval(atoms)
    rot = transforms.proper_rotation(ROTATION_AXIS, ROTATION_ANGLE_RAD)
    rotated = transforms.rotate_cell_and_positions(atoms, rot)
    rotated.calc = atoms.calc
    e1, f1, s1 = _full_eval(rotated)
    tol_e = _tol(floor["energy_spread_eV"], ABS_FLOOR_E)
    tol_f = _tol(floor["force_component_spread_eV_A"], ABS_FLOOR_F)
    f_expect = transforms.rotate_vectors(f0, rot)
    metrics = {
        "energy_delta_eV": abs(e1 - e0),
        "force_delta_max_eV_A": (
            float(np.abs(f1 - f_expect).max()) if len(atoms) else 0.0
        ),
        "rotation_axis": list(ROTATION_AXIS),
        "rotation_angle_rad": ROTATION_ANGLE_RAD,
        "tolerance_energy_eV": tol_e,
        "tolerance_force_eV_A": tol_f,
    }
    ok = metrics["energy_delta_eV"] <= tol_e and metrics["force_delta_max_eV_A"] <= tol_f
    if s0 is not None and s1 is not None:
        s_expect = transforms.rotate_stress_voigt(s0, rot)
        ds = float(np.abs(s1 - s_expect).max())
        tol_s = _tol(floor.get("stress_component_spread_eV_A3"), ABS_FLOOR_S)
        metrics["stress_delta_max_eV_A3"] = ds
        metrics["tolerance_stress_eV_A3"] = tol_s
        ok = ok and ds <= tol_s
    return ("pass" if ok else "fail"), metrics


CHECKS = (
    ("repeatability", None),
    ("ab_reversibility", check_ab_reversibility),
    ("permutation", check_permutation),
    ("pbc_wrap", check_pbc_wrap),
    ("global_translation", check_translation),
    ("global_rotation", check_rotation),
)


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
    parser.add_argument(
        "--systems", default=",".join(fixtures.INVARIANCE_SYSTEMS)
    )
    args = parser.parse_args()

    manifest = common.load_model_manifest(args.manifest)
    profile = common.resolve_profile(manifest, args.profile_id, args.model)
    out_dir = Path(args.out)

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
            case_id="t2_setup",
            test_id="t2_invariance",
            status="fail",
            engine=args.engine,
            model_identity=profile.get("identity", args.profile_id),
            model_sha256=profile["model_sha256"],
            task=profile.get("task"),
            head=args.head,
            dtype=args.dtype or "upstream/model-defined",
            device=common.device_record(args.device),
            input_structure_id=None,
            parameters={"floor_policy": FLOOR_POLICY},
            metrics={},
            diagnostics={"traceback": traceback.format_exc()},
            exception=f"{type(exc).__name__}: {exc}",
        )
        common.write_result(rec, out_dir, tag=args.tag)
        return 1

    exit_code = 0
    for system in [s for s in args.systems.split(",") if s]:
        atoms = _system_atoms(system)
        atoms.calc = ctx.calculator
        try:
            floor = measure_floor(atoms, args.device)
            for check_name, fn in CHECKS:
                if fn is None:
                    status, metrics = check_repeatability(floor)
                else:
                    status, metrics = fn(atoms, floor)
                rec = common.result_record(
                    case_id=f"t2_{system}",
                    test_id=f"t2_invariance_{check_name}",
                    status=status,
                    engine=ctx.engine,
                    model_identity=ctx.model_identity,
                    model_sha256=ctx.model_sha256,
                    task=ctx.task,
                    head=ctx.head,
                    dtype=ctx.dtype,
                    device=common.device_record(args.device),
                    input_structure_id=common.structure_id(atoms),
                    parameters={
                        "floor_policy": FLOOR_POLICY,
                        "measured_floor": floor,
                        "neighbor_cache": args.neighbor_cache,
                    },
                    metrics=metrics,
                    diagnostics={**ctx.diagnostics, "backend_version": ctx.backend_version, "framework_version": ctx.framework_version},
                    wall_seconds=0.0,
                    peak_vram=common.peak_vram_mib(),
                )
                common.write_result(rec, out_dir, tag=args.tag)
                print(
                    f"[t2] {ctx.engine} {system}/{check_name}: {status} "
                    f"dE={metrics.get('energy_delta_eV', floor.get('energy_spread_eV'))}"
                )
                if status not in ("pass", "characterized"):
                    exit_code = 1
        except Exception as exc:  # noqa: BLE001
            rec = common.result_record(
                case_id=f"t2_{system}",
                test_id="t2_invariance_error",
                status="fail",
                engine=ctx.engine,
                model_identity=ctx.model_identity,
                model_sha256=ctx.model_sha256,
                task=ctx.task,
                head=ctx.head,
                dtype=ctx.dtype,
                device=common.device_record(args.device),
                input_structure_id=common.structure_id(atoms),
                parameters={"floor_policy": FLOOR_POLICY},
                metrics={},
                diagnostics={"traceback": traceback.format_exc()},
                exception=f"{type(exc).__name__}: {exc}",
            )
            common.write_result(rec, out_dir, tag=args.tag)
            print(f"[t2] {ctx.engine} {system}: ERROR {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
