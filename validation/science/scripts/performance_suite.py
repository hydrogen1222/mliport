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
import sys
import time
import traceback
from itertools import pairwise
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

# PERF-01 contract (task book sections 21-24): startup is measured
# separately, at least five untimed warmups (extended while the last three
# calls are unstable, up to WARMUP_MAX), then 20+ timed repetitions whose
# full spread is recorded.  Three noisy samples must never masquerade as a
# "warm median".
WARMUP_TARGET = 5
WARMUP_MAX = 10
WARMUP_STABILITY = 0.05  # relative max-min of the last three warm calls
TIMED_REPS = 20
WARM_REPS = WARMUP_TARGET  # backward-compatible name
ANOMALY_RATIO = 5.0
#: A larger cell must not be faster than a smaller one by more than this
#: factor; smaller inversions are measurement noise, not a scaling warning.
SCALING_TOLERANCE = 0.8
MD_WARMUP_STEPS = 5


def _relative_spread(times: list[float]) -> float:
    if len(times) < 2:
        return float("inf")
    median = float(np.median(times))
    if median <= 0:
        return float("inf")
    return (max(times) - min(times)) / median


def summarize_timings(times: list[float]) -> dict[str, float]:
    """Full latency distribution in milliseconds (PERF-01 section 22)."""
    arr = np.asarray(times, dtype=float) * 1000.0
    if arr.size == 0:
        raise ValueError("summarize_timings needs at least one sample")
    return {
        "n_timed": int(arr.size),
        "median_ms": round(float(np.median(arr)), 4),
        "mean_ms": round(float(arr.mean()), 4),
        "p05_ms": round(float(np.percentile(arr, 5)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
    }


def classify_scaling(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag anomalous SP scaling; never silently pass a non-monotonic sweep.

    ``benchmark_anomaly`` marks a smallest-size median more than
    ``ANOMALY_RATIO`` slower than the next size.  Any non-monotonic median
    (a larger cell faster than a smaller one) downgrades the affected sweep
    to ``characterized_with_warning``.
    """
    medians: dict[int, float] = {}
    by_natoms: dict[int, dict[str, Any]] = {}
    for record in records:
        metrics = record.get("metrics", {})
        natoms = metrics.get("natoms")
        median = metrics.get("median_ms")
        if natoms is None or median is None:
            continue
        medians[int(natoms)] = float(median)
        by_natoms[int(natoms)] = record
    if not medians:
        return records

    sizes = sorted(medians)
    anomaly = bool(
        32 in medians and 128 in medians and medians[32] > ANOMALY_RATIO * medians[128]
    )
    non_monotonic = any(
        medians[later] < SCALING_TOLERANCE * medians[earlier]
        for earlier, later in pairwise(sizes)
    )
    if not anomaly and not non_monotonic:
        return records

    reason = (
        "smallest-size median exceeds 5x the next size"
        if anomaly
        else "non-monotonic scaling: a larger cell is faster than a smaller one"
    )
    if anomaly:
        target = by_natoms.get(32)
        if target is not None:
            target.setdefault("diagnostics", {})["benchmark_anomaly"] = True
    for record in records:
        diagnostics = record.setdefault("diagnostics", {})
        diagnostics["scaling_verdict"] = "characterized_with_warning"
        diagnostics["scaling_reason"] = reason
        if record.get("status") == "pass":
            record["status"] = "characterized"
    return records


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
    if any(name in names for name in ("OutOfMemoryError", "ResourceExhaustedError")):
        return True
    import re

    text = str(exc).lower()
    return "out of memory" in text or re.search(r"\boom\b", text) is not None


def _sp_record(
    ctx: engines.EngineContext,
    natoms: int,
    device: str,
    args: argparse.Namespace,
    model_load_s: float | None = None,
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
        for _ in range(WARMUP_MAX):
            _, warm_s = common.timed(_energy, device)
            warm_times.append(warm_s)
            if (
                len(warm_times) >= WARMUP_TARGET
                and _relative_spread(warm_times[-3:]) <= WARMUP_STABILITY
            ):
                break
        timed_times = [common.timed(_energy, device)[1] for _ in range(TIMED_REPS)]
        stats = summarize_timings(timed_times)

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
    median_s = stats["median_ms"] / 1000.0
    throughput = natoms / median_s if median_s > 0 else float("nan")
    metrics = {
        "natoms": natoms,
        "energy_eV": energy,
        "forces_max_abs_eV_A": float(np.abs(forces).max()) if len(atoms) else 0.0,
        "stress_supported": stress is not None,
        "model_load_s": None if model_load_s is None else round(model_load_s, 4),
        "first_inference_ms": round(first_call_s * 1000.0, 4),
        "first_call_seconds": round(first_call_s, 4),
        "warmup_count": len(warm_times),
        "warmup_target": WARMUP_TARGET,
        "n_timed": stats["n_timed"],
        "median_ms": stats["median_ms"],
        "mean_ms": stats["mean_ms"],
        "p05_ms": stats["p05_ms"],
        "p95_ms": stats["p95_ms"],
        "min_ms": stats["min_ms"],
        "max_ms": stats["max_ms"],
        "warm_call_median_seconds": round(median_s, 4),
        "warm_call_spread_seconds": round(
            (stats["max_ms"] - stats["min_ms"]) / 1000.0, 4
        ),
        "atoms_per_second_warm": round(throughput, 1),
        "peak_memory_mib": common.peak_vram_mib(),
        "device": device,
        "dtype": ctx.dtype,
        "model_hash": ctx.model_sha256,
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
            "warmup_target": WARMUP_TARGET,
            "warmup_max": WARMUP_MAX,
            "timed_reps": TIMED_REPS,
            "timing_perturbation_scale_A": 0.01,
            "neighbor_cache": bool(getattr(args, "neighbor_cache", True)),
            "harness_commit": _harness_commit(),
        },
        metrics=metrics,
        diagnostics={
            **ctx.diagnostics,
            "neighbor_cache": bool(getattr(args, "neighbor_cache", True)),
        },
        wall_seconds=first_call_s + sum(warm_times) + observable_s,
        peak_vram=common.peak_vram_mib(),
    )


def _harness_commit() -> str:
    """Revision of the validation harness that produced a record."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def _md_record(
    ctx: engines.EngineContext,
    natoms: int,
    md_steps: int,
    device: str,
    model_load_s: float | None = None,
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
        "model_load_s": None if model_load_s is None else round(model_load_s, 4),
        "warmup_count": MD_WARMUP_STEPS,
        "n_timed": md_steps,
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
            "warmup_steps": MD_WARMUP_STEPS,
            "timed_steps": md_steps,
            "harness_commit": _harness_commit(),
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
        help="GRACE only: toggle the mliport neighbor-list cache",
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

    load_started = time.perf_counter()
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
        model_load_s = time.perf_counter() - load_started
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
                rec = _sp_record(
                    ctx, natoms, args.device, args, model_load_s=model_load_s
                )
            else:
                rec = _md_record(
                    ctx, natoms, steps, args.device, model_load_s=model_load_s
                )
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

    sp_records = [
        record
        for record in records
        if str(record.get("case_id", "")).startswith("t8_sp_scaling_")
    ]
    classify_scaling(sp_records)
    for record in records:
        common.write_result(record, out_dir, tag=args.tag)
    flagged = [
        record["case_id"]
        for record in sp_records
        if record.get("diagnostics", {}).get("benchmark_anomaly")
        or record.get("diagnostics", {}).get("scaling_verdict")
    ]
    if flagged:
        print(f"[t8] scaling flags: {', '.join(sorted(flagged))}")

    print(f"[t8] engine={args.engine} cases={len(records)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
