"""T1: single-system real-model inference on one small periodic bulk cell.

Run inside a backend environment::

    python inference.py --engine uma --profile-id uma_omat \
        --model /path/to/uma-s-1p2.pt \
        --manifest validation/science/model_manifest.json \
        --out .validation-work/t1 [--device cuda:0] [--head NAME]

For each fixture the script records energies/forces/stress, first-call and
warmed-call latency, peak VRAM and the calculator identity proof.  The
calculator identity is recorded from the live wrapper object (taskbook
section 47); nothing about the result may claim an engine the wrapper does
not belong to.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402
import engines  # noqa: E402
import fixtures  # noqa: E402

FIRST_CALL_REPS = 1
WARM_REPS = 3


def _record_for_system(
    ctx: engines.EngineContext,
    name: str,
    atoms,
    device: str,
) -> dict[str, Any]:
    atoms.calc = ctx.calculator

    common.reset_vram_counter()

    def _energy():
        return float(atoms.get_potential_energy())

    energy, first_call_s = common.timed(_energy, device)
    warm_times: list[float] = []
    for _ in range(WARM_REPS):
        _, warm_times_s = common.timed(_energy, device)
        warm_times.append(warm_times_s)

    forces = atoms.get_forces()
    stress_supported = True
    stress_voigt = None
    try:
        stress_voigt = [float(x) for x in atoms.get_stress()]
    except Exception:  # noqa: BLE001 - missing stress is a property, not a crash
        stress_supported = False

    metrics = {
        "energy_eV": energy,
        "forces_min_abs_eV_A": float(abs(forces).min()) if len(forces) else 0.0,
        "forces_max_abs_eV_A": float(abs(forces).max()) if len(forces) else 0.0,
        "forces_norm_rms_eV_A": float(
            (forces**2).sum(axis=1).mean() ** 0.5
        ),
        "stress_supported": stress_supported,
        "first_call_seconds": round(first_call_s, 4),
        "warm_call_median_seconds": round(statistics.median(warm_times), 4),
        "warm_call_spread_seconds": round(
            max(warm_times) - min(warm_times), 4
        ),
    }
    if stress_voigt is not None:
        metrics["stress_voigt_eV_A3"] = stress_voigt

    finite = math.isfinite(energy) and bool(np.isfinite(forces).all())
    if stress_voigt is not None:
        finite = finite and bool(np.isfinite(np.asarray(stress_voigt)).all())
    status = "pass" if finite else "fail"
    return common.result_record(
        case_id=f"t1_{name}",
        test_id="t1_real_inference",
        status=status,
        engine=ctx.engine,
        model_identity=ctx.model_identity,
        model_sha256=ctx.model_sha256,
        task=ctx.task,
        head=ctx.head,
        dtype=ctx.dtype,
        device=common.device_record(device),
        input_structure_id=common.structure_id(atoms),
        parameters={
            "fixture": name,
            "first_call_reps": FIRST_CALL_REPS,
            "warm_reps": WARM_REPS,
        },
        metrics=metrics,
        diagnostics={**ctx.diagnostics, "backend_version": ctx.backend_version, "framework_version": ctx.framework_version},
        wall_seconds=sum(warm_times) + first_call_s,
        peak_vram=common.peak_vram_mib(),
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
    parser.add_argument("--tag", default=None, help="filename suffix")
    parser.add_argument("--dtype", default=None, help="MACE only")
    args = parser.parse_args()

    manifest = common.load_model_manifest(args.manifest)
    profile = common.resolve_profile(manifest, args.profile_id, args.model)
    out_dir = Path(args.out)

    records: list[dict[str, Any]] = []
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
    except Exception as exc:  # noqa: BLE001 - record and re-raise as failure
        rec = common.result_record(
            case_id="t1_setup",
            test_id="t1_real_inference",
            status="fail",
            engine=args.engine,
            model_identity=profile.get("identity", args.profile_id),
            model_sha256=profile["model_sha256"],
            task=profile.get("task"),
            head=args.head,
            dtype=args.dtype or "upstream/model-defined",
            device=common.device_record(args.device),
            input_structure_id=None,
            parameters={"fixtures": list(fixtures.FIXTURES)},
            metrics={},
            diagnostics={"traceback": traceback.format_exc()},
            exception=f"{type(exc).__name__}: {exc}",
        )
        common.write_result(rec, out_dir, tag=args.tag)
        return 1

    for name in fixtures.FIXTURES:
        atoms, _desc = fixtures.build_fixture(name)
        try:
            records.append(_record_for_system(ctx, name, atoms, args.device))
        except Exception as exc:  # noqa: BLE001
            records.append(
                common.result_record(
                    case_id=f"t1_{name}",
                    test_id="t1_real_inference",
                    status="fail",
                    engine=ctx.engine,
                    model_identity=ctx.model_identity,
                    model_sha256=ctx.model_sha256,
                    task=ctx.task,
                    head=ctx.head,
                    dtype=ctx.dtype,
                    device=common.device_record(args.device),
                    input_structure_id=common.structure_id(atoms),
                    parameters={"fixture": name},
                    metrics={},
                    diagnostics={"traceback": traceback.format_exc()},
                    exception=f"{type(exc).__name__}: {exc}",
                )
            )

    failures = sum(1 for r in records if r["status"] != "pass")
    for rec in records:
        common.write_result(rec, out_dir, tag=args.tag)
        print(
            f"[t1] {rec['engine']} {rec['case_id']}: {rec['status']} "
            f"E={rec['metrics'].get('energy_eV')} "
            f"first={rec['metrics'].get('first_call_seconds')}s "
            f"warm={rec['metrics'].get('warm_call_median_seconds')}s"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
