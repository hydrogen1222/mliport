"""T8: GPU performance characterization (taskbook section 33).

Run inside a backend environment::

    python performance_suite.py --engine mace --profile-id mace_mpa0 \
        --model /path/to/model.model \
        --manifest validation/science/model_manifest.json \
        --out .validation-work/t8 [--device cuda:0] [--tag NAME]

Two measurement families, per engine:

- ``t8_sp_scaling``: single-point latency/VRAM on alpha-Na3PS4 supercells of
  32, 128 and 512 atoms (first call, warmed median + spread, atoms/s, peak
  VRAM).  Scaling is measured against the same pinned geometry fixture the
  transport tier uses, so the sizes are physically meaningful compositions of
  one crystal, not arbitrary point clouds.
- ``t8_md_throughput``: NVE steps/s and atom-steps/s through the ASE upstream
  VelocityVerlet integrator driven by the real calculator (no synthetic
  physics; the integrator is ASE's), with explicit device synchronization and
  a warmup phase excluded from the timed window.

A size that does not fit GPU memory is reported as ``characterized`` with an
``out_of_memory`` reason -- that is a hardware/size fact, not a product
failure -- and never reported as ``pass``.  Any other exception is ``fail``.
All records carry the calculator identity of the live wrapper (taskbook
section 47); results are never attributed to an engine the wrapper does not
belong to.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402
import engines  # noqa: E402
import fixtures  # noqa: E402

#: Na3PS4 supercell multiples of the 16-atom pinned unit cell.
SP_SIZES: dict[int, tuple[int, int, int]] = {
    32: (2, 1, 1),
    128: (2, 2, 2),
    512: (4, 4, 2),
}

WARM_REPS = 3
MD_WARMUP_STEPS = 5


def _build_supercell(natoms: int):
    base, _meta = fixtures.build_extra("na3ps4")
    if natoms not in SP_SIZES:
        raise ValueError(f"unsupported SP size {natoms}")
    repeat = SP_SIZES[natoms]
    atoms = base.repeat(repeat)
    # Tiny seeded displacement so forces are non-trivial (no frozen ideal
    # lattice shortcuts in the timed force evaluation).
    rng = np.random.default_rng(20260911)
    disp = rng.normal(scale=0.01, size=(len(atoms), 3))
    disp -= disp.mean(axis=0)
    atoms.positions += disp
    return atoms, repeat


def _is_oom(exc: Exception) -> bool:
    """Classify GPU out-of-memory failures without swallowing anything else."""
    names = {type(exc).__name__}
    if any(
        name in names
        for name in ("OutOfMemoryError", "ResourceExhaustedError")
    ):
        return True
    import re

    text = str(exc).lower()
    return "out of memory" in text or re.search(r"\boom\b", text) is not None


def _sp_record(
    ctx: engines.EngineContext,
    natoms: int,
    device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    atoms, repeat = _build_supercell(natoms)
    atoms.calc = ctx.calculator
    common.reset_vram_counter()

    # Deterministic micro-perturbations between timed calls: reusing
    # identical positions would hit the ASE calculator result cache and
    # "measure" a dictionary lookup (~0.2 ms) instead of model inference.
    # 0.01 A keeps the neighbour topology realistic (steady-state latency).
    # These are TIMING samples only -- the reported E/F/stress are measured
    # separately on the restored pristine structure (review R10).
    rng = np.random.default_rng(20260911)
    pristine_positions = atoms.positions.copy()

    def _energy():
        atoms.positions = atoms.positions + rng.normal(
            0.0, 0.01, size=atoms.positions.shape
        )
        return float(atoms.get_potential_energy())

    try:
        _timing_energy, first_call_s = common.timed(_energy, device)
        warm_times: list[float] = []
        for _ in range(WARM_REPS):
            _, warm_s = common.timed(_energy, device)
            warm_times.append(warm_s)

        # Timing is over. Restore the pinned geometry and take the reported
        # observables from a real inference on *that* structure, so energy,
        # forces, stress and the recorded structure hash are one identity.
        atoms.positions = pristine_positions
        with common.InferenceProbe(atoms) as probe:
            (energy, forces, stress), observable_s = common.timed(probe.run, device)
            observable_calls = probe.calls
    except Exception as exc:  # noqa: BLE001 - classified below
        if _is_oom(exc):
            return common.result_record(
                case_id=f"t8_sp_scaling_{natoms}",
                test_id="t8_performance",
                status="characterized",
                engine=ctx.engine,
                model_identity=ctx.model_identity,
                model_sha256=ctx.model_sha256,
                task=ctx.task,
                head=ctx.head,
                dtype=ctx.dtype,
                device=common.device_record(device),
                input_structure_id=common.structure_id(atoms),
                parameters={"natoms": natoms, "repeat": list(repeat)},
                metrics={},
                diagnostics={
                    **ctx.diagnostics,
                    "reason": "out_of_memory",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
        raise
    finite = (
        bool(np.isfinite(forces).all())
        and np.isfinite(energy)
        and (stress is None or bool(np.isfinite(stress).all()))
    )
    throughput = natoms / statistics.median(warm_times)
    metrics = {
        "natoms": natoms,
        "energy_eV": energy,
        "forces_max_abs_eV_A": float(np.abs(forces).max()) if len(atoms) else 0.0,
        "stress_supported": stress is not None,
        "first_call_seconds": round(first_call_s, 4),
        "warm_call_median_seconds": round(statistics.median(warm_times), 4),
        "warm_call_spread_seconds": round(max(warm_times) - min(warm_times), 4),
        "atoms_per_second_warm": round(throughput, 1),
        "energy_finite": bool(finite),
        "timing_perturbation_scale_A": 0.01,
        "timing_samples_perturb_geometry": True,
        "observable_structure_restored": True,
        "observable_structure_id": common.structure_id(atoms),
        "observable_inference_calls": int(observable_calls),
        "observable_seconds": round(observable_s, 4),
    }
    if stress is not None:
        metrics["stress_max_abs_eV_A3"] = float(np.abs(stress).max())
    return common.result_record(
        case_id=f"t8_sp_scaling_{natoms}",
        test_id="t8_performance",
        status="pass" if finite else "fail",
        engine=ctx.engine,
        model_identity=ctx.model_identity,
        model_sha256=ctx.model_sha256,
        task=ctx.task,
        head=ctx.head,
        dtype=ctx.dtype,
        device=common.device_record(device),
        input_structure_id=common.structure_id(atoms),
        parameters={
            "natoms": natoms,
            "repeat": list(repeat),
            "warm_reps": WARM_REPS,
            "timing_perturbation_scale_A": 0.01,
        },
        metrics=metrics,
        diagnostics={**ctx.diagnostics},
        wall_seconds=first_call_s + sum(warm_times) + observable_s,
        peak_vram=common.peak_vram_mib(),
    )


def _md_record(
    ctx: engines.EngineContext,
    natoms: int,
    md_steps: int,
    device: str,
) -> dict[str, Any]:
    from ase import units
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
    )
    from ase.md.verlet import VelocityVerlet

    atoms, repeat = _build_supercell(natoms)
    atoms.calc = ctx.calculator
    timestep_fs = 1.0
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.default_rng(20260911)
    )
    Stationary(atoms)
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)

    common.reset_vram_counter()
    try:
        # Warmup: force kernels, cudnn autotuning, allocator pools.
        for _ in range(MD_WARMUP_STEPS):
            dyn.run(1)
        started = time.perf_counter()
        dyn.run(md_steps)
        common.synchronize(device)
        wall_s = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001 - classified below
        if _is_oom(exc):
            return common.result_record(
                case_id=f"t8_md_throughput_{natoms}",
                test_id="t8_performance",
                status="characterized",
                engine=ctx.engine,
                model_identity=ctx.model_identity,
                model_sha256=ctx.model_sha256,
                task=ctx.task,
                head=ctx.head,
                dtype=ctx.dtype,
                device=common.device_record(device),
                input_structure_id=common.structure_id(atoms),
                parameters={"natoms": natoms, "repeat": list(repeat)},
                metrics={},
                diagnostics={
                    **ctx.diagnostics,
                    "reason": "out_of_memory",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
        raise

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    finite = bool(np.isfinite(forces).all()) and np.isfinite(energy)
    metrics = {
        "natoms": natoms,
        "timestep_fs": timestep_fs,
        "warmup_steps": MD_WARMUP_STEPS,
        "timed_steps": md_steps,
        "wall_seconds": round(wall_s, 3),
        "steps_per_second": round(md_steps / wall_s, 3),
        "atom_steps_per_second": round(md_steps * natoms / wall_s, 1),
        "final_energy_eV": energy,
        "energy_finite": bool(finite),
    }
    return common.result_record(
        case_id=f"t8_md_throughput_{natoms}",
        test_id="t8_performance",
        status="pass" if finite else "fail",
        engine=ctx.engine,
        model_identity=ctx.model_identity,
        model_sha256=ctx.model_sha256,
        task=ctx.task,
        head=ctx.head,
        dtype=ctx.dtype,
        device=common.device_record(device),
        input_structure_id=common.structure_id(atoms),
        parameters={
            "natoms": natoms,
            "repeat": list(repeat),
            "integrator": "ase.md.verlet.VelocityVerlet (NVE)",
            "temperature_K": 300,
        },
        metrics=metrics,
        diagnostics={**ctx.diagnostics},
        wall_seconds=round(wall_s, 3),
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
    parser.add_argument("--head", default=None)
    parser.add_argument("--dtype", default=None, help="MACE only")
    parser.add_argument(
        "--neighbor-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="GRACE only: toggle the mlipx neighbor-list cache",
    )
    parser.add_argument(
        "--md-steps",
        type=int,
        default=30,
        help="timed NVE steps for the 128-atom throughput case",
    )
    parser.add_argument("--tag", default=None, help="filename suffix")
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
    except Exception as exc:  # noqa: BLE001 - record and re-raise as failure
        rec = common.result_record(
            case_id="t8_setup",
            test_id="t8_performance",
            status="fail",
            engine=args.engine,
            model_identity=profile.get("identity", args.profile_id),
            model_sha256=profile["model_sha256"],
            task=profile.get("task"),
            head=args.head,
            dtype=args.dtype or "upstream/model-defined",
            device=common.device_record(args.device),
            input_structure_id=None,
            parameters={},
            metrics={},
            diagnostics={"traceback": traceback.format_exc()},
            exception=f"{type(exc).__name__}: {exc}",
        )
        common.write_result(rec, out_dir, tag=args.tag)
        return 1

    md_steps_512 = max(10, args.md_steps // 3)
    plan: list[tuple[str, int, int]] = [
        ("sp", natoms, 0) for natoms in sorted(SP_SIZES)
    ]
    plan += [("md", 128, args.md_steps), ("md", 512, md_steps_512)]

    records: list[dict[str, Any]] = []
    failures = 0
    for kind, natoms, steps in plan:
        case = (
            f"t8_{kind}_scaling_{natoms}"
            if kind == "sp"
            else f"t8_{kind}_throughput_{natoms}"
        )
        try:
            if kind == "sp":
                rec = _sp_record(ctx, natoms, args.device, args)
            else:
                rec = _md_record(ctx, natoms, steps, args.device)
        except Exception as exc:  # noqa: BLE001 - one bad case must not kill the sweep
            rec = common.result_record(
                case_id=case,
                test_id="t8_performance",
                status="fail",
                engine=ctx.engine,
                model_identity=ctx.model_identity,
                model_sha256=ctx.model_sha256,
                task=ctx.task,
                head=ctx.head,
                dtype=ctx.dtype,
                device=common.device_record(args.device),
                input_structure_id=None,
                parameters={"natoms": natoms, "family": kind},
                metrics={},
                diagnostics={"traceback": traceback.format_exc()},
                exception=f"{type(exc).__name__}: {exc}",
            )
        records.append(rec)
        common.write_result(rec, out_dir, tag=args.tag)
        summary = {
            key: rec["metrics"].get(key)
            for key in (
                "warm_call_median_seconds",
                "atoms_per_second_warm",
                "steps_per_second",
                "atom_steps_per_second",
            )
        }
        print(
            f"[t8] {rec['engine']} {rec['case_id']}: {rec['status']} "
            + " ".join(f"{k}={v}" for k, v in summary.items() if v is not None)
        )
        if rec["status"] == "fail":
            failures += 1

    print(f"[t8] engine={args.engine} cases={len(records)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
