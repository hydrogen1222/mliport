#!/usr/bin/env python
"""T6 - MD integration and ensembles (taskbook section 27).

Runs the product MD pipeline (``mlipx.runners.md.MDRunner`` driven exactly
as the public API drives it) per engine profile on a deterministic 32-atom
Cu fixture with identical initial coordinates and a seeded velocity set:

- t6a NVE timestep sweep 0.25/0.50/1.00/2.00 fs (2 ps each): total-energy
  drift slope (eV/atom/ps), max instantaneous deviation, temperature
  statistics, force max, NaN/abort state, timing.  No global drift
  threshold is invented; the recorded expectation is that drift improves
  as the timestep decreases.
- t6b NVT Langevin: temperature statistics, thermostat metadata, COM
  policy, same-seed reproducibility (two full runs compared stepwise).
- t6c NVT Bussi/CSVR: target-temperature behaviour, degrees-of-freedom
  handling, and FixAtoms constraint compatibility.
- t6d Nose-Hoover chain: capability-matrix-allowed combination only
  (unconstrained, com_policy='none') plus the rejection matrix: the
  software must refuse NHC with constraints/COM removal and compound
  constraint combinations instead of silently integrating the wrong DOF.
- t6e Constraints: FixAtoms positions stay fixed exactly (bitwise) and
  the FixCom projector keeps the centre of mass fixed; DOF counts are
  asserted against 3N - 3*n_fixed and 3N - 3.
- t6f Force-safety abort: a controlled deterministic displacement creates
  a high-force configuration; the configured threshold (derived from a
  measured pre-flight force, recorded in the record) must abort cleanly,
  checkpoint the unsafe frame, write an 'aborted' artifacts manifest, and
  never continue into NaN/Inf territory.

Trajectories diverge chaotically between backends and are never compared
atom-by-atom; only ensemble/integration diagnostics are compared.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import engines  # noqa: E402
import fixtures  # noqa: E402
from static_suite import record  # noqa: E402

# --------------------------------------------------------------------------
# pinned suite parameters (documented, not tuned per engine)
# --------------------------------------------------------------------------

MD_SEED = 20260911
TEMPERATURE_K = 300.0
CU32_REPEAT = (2, 2, 2)  # 4-atom conventional cell -> 32 atoms (section 27)

NVE_TIMESTEPS_FS = (0.25, 0.50, 1.00, 2.00)
NVE_RUN_PS = 2.0  # bounded drift-slope window (taskbook: 2-5 ps)

LANGEVIN_DT_FS = 1.0
LANGEVIN_STEPS = 5000  # 5 ps
LANGEVIN_FRICTION_FS = 0.01  # fs^-1; damping time ~100 ps

BUSSI_DT_FS = 1.0
BUSSI_STEPS = 5000
BUSSI_TAU_FS = 100.0  # CSVR coupling time

NHC_DT_FS = 1.0
NHC_STEPS = 5000
NHC_TDAMP_FS = 100.0
NHC_TCHAIN = 3
NHC_TLOOP = 1

CONSTRAINT_DT_FS = 1.0
CONSTRAINT_STEPS = 500  # exactness checks do not need long runs

#: same-seed reproducibility: two identical runs must agree to float64
#: roundoff; anything above this is recorded as 'characterized' (the
#: taskbook makes seed reproducibility conditional on the runtime).
REPRO_MAX_DELTA_EV = 1e-6
#: FixAtoms exactness / COM projector exactness (float64 output precision)
EXACTNESS_TOL_A = 1e-12
COM_DRIFT_TOL_A = 1e-10

FORCE_SAFETY_DISPLACEMENT_A = 0.30
#: threshold configured at half the measured pre-flight force so the abort
#: is deterministic for every engine; both numbers go into the record
FORCE_SAFETY_FRACTION = 0.5
FORCE_SAFETY_STEPS = 200

_SAVE_INTERVAL = 100  # trajectory frames; the CSV logs every step anyway


def cu32() -> Any:
    """Deterministic 32-atom Cu fixture (identical across backends)."""
    base, _ = fixtures.build_fixture("cu_fcc")
    return base.repeat(CU32_REPEAT)


def _fixed_layer_indices(atoms) -> list[int]:
    """Deterministic bottom-layer selection for FixAtoms (z below 0.4*a)."""
    a = float(fixtures.FIXTURES["cu_fcc"]["a"])
    return [int(i) for i in np.where(atoms.positions[:, 2] < 0.4 * a)[0]]


def _md_run_dir(out_dir: Path, tag: str | None, name: str) -> Path:
    d = out_dir / "md_runs" / (tag or "default") / name
    if d.exists():
        shutil.rmtree(d)  # transient run directories; records stay out of git
    d.mkdir(parents=True, exist_ok=True)
    return d


_MD_NUM_COLS = (
    "step",
    "time_fs",
    "potential_energy_eV",
    "kinetic_energy_eV",
    "total_energy_eV",
    "temperature_K",
    "max_force_raw_eV_A",
    "max_force_applied_eV_A",
)


def _parse_md_csv(path: Path) -> dict[str, np.ndarray]:
    """Parse the product per-step thermodynamics CSV into float arrays."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, np.ndarray] = {}
    for col in _MD_NUM_COLS:
        out[col] = np.array([float(r[col]) for r in rows], dtype=float)
    return out


def _drift_slope_eV_per_atom_ps(t_fs: np.ndarray, e_tot: np.ndarray, natoms: int) -> float:
    """Least-squares total-energy drift slope in eV/atom/ps."""
    if len(t_fs) < 2:
        return float("nan")
    slope_eV_per_fs = float(np.polyfit(t_fs, e_tot, 1)[0])
    return slope_eV_per_fs * 1000.0 / natoms


def _finite_all(csv: dict[str, np.ndarray]) -> bool:
    return all(bool(np.all(np.isfinite(v))) for v in csv.values())


def _temperature_stats(csv: dict[str, np.ndarray]) -> dict[str, float]:
    t = csv["temperature_K"]
    return {
        "temperature_mean_K": float(np.mean(t)),
        "temperature_std_K": float(np.std(t)),
        "temperature_min_K": float(np.min(t)),
        "temperature_max_K": float(np.max(t)),
    }


def _run_md(
    ctx,
    run_dir: Path,
    *,
    ensemble: str,
    timestep_fs: float,
    steps: int,
    thermostat: str | None = None,
    temperature_k: float = TEMPERATURE_K,
    seed: int | None = MD_SEED,
    com_policy: str = "auto",
    velocity_policy: str = "initialize",
    friction_fs: float | None = None,
    bussi_tau_fs: float | None = None,
    nhc_tdamp_fs: float | None = None,
    nhc_tchain: int | None = None,
    nhc_tloop: int | None = None,
    fmax_abort: float | None = None,
    atoms=None,
    pre_relax: bool = False,
) -> dict[str, Any]:
    """Drive the product MDRunner exactly once and parse its CSV."""
    from mlipx.runners.md import MDRunner

    if atoms is None:
        atoms = cu32()
    kwargs: dict[str, Any] = {
        "ensemble": ensemble,
        "timestep": timestep_fs,
        "steps": steps,
        "temperature": temperature_k,
        "seed": seed,
        "com_policy": com_policy,
        "velocity_policy": velocity_policy,
        "pre_relax": pre_relax,
        "output_dir": run_dir,
        "save_interval": _SAVE_INTERVAL,
        "verbose": False,
    }
    if thermostat is not None:
        kwargs["thermostat"] = thermostat
    if friction_fs is not None:
        kwargs["friction"] = friction_fs
    if bussi_tau_fs is not None:
        kwargs["bussi_tau"] = bussi_tau_fs
    if nhc_tdamp_fs is not None:
        kwargs["nhc_tdamp"] = nhc_tdamp_fs
    if nhc_tchain is not None:
        kwargs["nhc_tchain"] = nhc_tchain
    if nhc_tloop is not None:
        kwargs["nhc_tloop"] = nhc_tloop
    if fmax_abort is not None:
        kwargs["fmax_abort"] = fmax_abort

    runner = MDRunner(ctx.wrapper, **kwargs)
    results = runner.run(atoms)
    csv_data = _parse_md_csv(run_dir / "raw" / "md.csv")
    return {"results": results, "csv": csv_data, "run_dir": run_dir}


# --------------------------------------------------------------------------
# t6a NVE timestep sweep
# --------------------------------------------------------------------------


def workflow_nve_dt_sweep(ctx, args, out_dir) -> int:
    """t6a: NVE drift diagnostics vs timestep (taskbook section 27.1)."""
    natoms = len(cu32())
    t0 = time.perf_counter()
    per_dt: dict[str, dict[str, Any]] = {}
    failure: str | None = None
    try:
        for dt in NVE_TIMESTEPS_FS:
            steps = int(round(NVE_RUN_PS * 1000.0 / dt))
            run = _run_md(
                ctx,
                _md_run_dir(Path(args.out), args.tag, f"t6a_nve_dt{dt:g}"),
                ensemble="nve",
                timestep_fs=dt,
                steps=steps,
            )
            csv = run["csv"]
            finite = _finite_all(csv)
            if not finite:
                failure = f"non-finite trajectory at dt={dt:g} fs"
            slope = _drift_slope_eV_per_atom_ps(csv["time_fs"], csv["total_energy_eV"], natoms)
            e0 = float(csv["total_energy_eV"][0])
            per_dt[f"{dt:g}"] = {
                "steps": steps,
                "rows": int(len(csv["step"])),
                "drift_slope_eV_atom_ps": slope,
                "max_total_energy_deviation_eV": float(np.max(np.abs(csv["total_energy_eV"] - e0))),
                "total_energy_first_eV": e0,
                "initial_kinetic_energy_eV": float(csv["kinetic_energy_eV"][0]),
                "max_force_raw_eV_A": float(np.max(csv["max_force_raw_eV_A"])),
                "all_finite": finite,
                **_temperature_stats(csv),
            }
    except Exception as exc:  # noqa: BLE001
        record(
            ctx,
            args,
            "t6_nve",
            "t6a_cu32_nve_timestep_sweep",
            "fail",
            input_atoms=cu32(),
            parameters={
                "timesteps_fs": list(NVE_TIMESTEPS_FS),
                "run_ps": NVE_RUN_PS,
                "seed": MD_SEED,
                "velocity_policy": "initialize",
                "com_policy": "auto",
            },
            metrics={},
            wall=time.perf_counter() - t0,
            exception=f"{type(exc).__name__}: {exc}",
        )
        return 1

    slopes = {k: v["drift_slope_eV_atom_ps"] for k, v in per_dt.items()}
    all_finite = all(v["all_finite"] for v in per_dt.values())
    ke0 = {k: v["initial_kinetic_energy_eV"] for k, v in per_dt.items()}
    # taskbook section 27.1: drift should generally improve as the timestep
    # decreases.  No global cutoff is invented; the gate is the broad trend
    # between the extreme timesteps plus finiteness everywhere.
    trend_ok = abs(slopes[f"{NVE_TIMESTEPS_FS[0]:g}"]) < abs(slopes[f"{NVE_TIMESTEPS_FS[-1]:g}"])
    status = "pass" if (all_finite and failure is None and trend_ok) else "fail"
    record(
        ctx,
        args,
        "t6_nve",
        "t6a_cu32_nve_timestep_sweep",
        status,
        input_atoms=cu32(),
        parameters={
            "timesteps_fs": list(NVE_TIMESTEPS_FS),
            "run_ps": NVE_RUN_PS,
            "seed": MD_SEED,
            "velocity_policy": "initialize",
            "com_policy": "auto",
            "drift_gate": "|slope(dt_min)| < |slope(dt_max)| (eV/atom/ps); no global cutoff",
            "natoms": natoms,
        },
        metrics={
            "per_timestep": per_dt,
            "drift_slopes_eV_atom_ps": slopes,
            "initial_kinetic_energy_eV_by_dt": ke0,
            "drift_improves_with_timestep": bool(trend_ok),
            "all_finite": bool(all_finite),
            "abort_state": failure or "none",
        },
        wall=time.perf_counter() - t0,
    )
    return 0 if status == "pass" else 1


# --------------------------------------------------------------------------
# t6b NVT Langevin
# --------------------------------------------------------------------------


def workflow_langevin(ctx, args, out_dir) -> int:
    """t6b: Langevin thermostat behaviour + same-seed reproducibility."""
    t0 = time.perf_counter()
    try:
        run_a = _run_md(
            ctx,
            _md_run_dir(Path(args.out), args.tag, "t6b_langevin_a"),
            ensemble="nvt",
            thermostat="langevin",
            timestep_fs=LANGEVIN_DT_FS,
            steps=LANGEVIN_STEPS,
            friction_fs=LANGEVIN_FRICTION_FS,
        )
        run_b = _run_md(
            ctx,
            _md_run_dir(Path(args.out), args.tag, "t6b_langevin_b"),
            ensemble="nvt",
            thermostat="langevin",
            timestep_fs=LANGEVIN_DT_FS,
            steps=LANGEVIN_STEPS,
            friction_fs=LANGEVIN_FRICTION_FS,
        )
    except Exception as exc:  # noqa: BLE001
        record(
            ctx,
            args,
            "t6_langevin",
            "t6b_cu32_nvt_langevin",
            "fail",
            input_atoms=cu32(),
            parameters={"seed": MD_SEED},
            metrics={},
            wall=time.perf_counter() - t0,
            exception=f"{type(exc).__name__}: {exc}",
        )
        return 1

    csv_a, csv_b = run_a["csv"], run_b["csv"]
    finite = _finite_all(csv_a) and _finite_all(csv_b)
    n = min(len(csv_a["step"]), len(csv_b["step"]))
    repro_delta = float(np.max(np.abs(csv_a["total_energy_eV"][:n] - csv_b["total_energy_eV"][:n])))
    res = run_a["results"]
    prov = res["md_provenance"]
    status = "fail"
    if not finite:
        status = "fail"
    elif repro_delta <= REPRO_MAX_DELTA_EV:
        status = "pass"
    else:
        # same-seed reproducibility is conditional on the runtime (taskbook
        # section 27.2); a backend that cannot reproduce its own trajectory
        # is recorded honestly instead of failing the engine.
        status = "characterized"
    record(
        ctx,
        args,
        "t6_langevin",
        "t6b_cu32_nvt_langevin",
        status,
        input_atoms=cu32(),
        parameters={
            "timestep_fs": LANGEVIN_DT_FS,
            "steps": LANGEVIN_STEPS,
            "friction_fs": LANGEVIN_FRICTION_FS,
            "temperature_K": TEMPERATURE_K,
            "seed": MD_SEED,
            "repro_gate_eV": REPRO_MAX_DELTA_EV,
        },
        metrics={
            **_temperature_stats(csv_a),
            "initial_kinetic_energy_eV": float(csv_a["kinetic_energy_eV"][0]),
            "max_force_raw_eV_A": float(np.max(csv_a["max_force_raw_eV_A"])),
            "all_finite": bool(finite),
            "same_seed_max_energy_delta_eV": repro_delta,
            "thermostat_metadata": prov,
            "seed_recorded": res["seed"],
        },
        wall=time.perf_counter() - t0,
    )
    return 0 if status == "pass" else (0 if status == "characterized" else 1)


# --------------------------------------------------------------------------
# t6c NVT Bussi/CSVR
# --------------------------------------------------------------------------


def workflow_bussi(ctx, args, out_dir) -> int:
    """t6c: Bussi target-T behaviour, DOF handling, FixAtoms compatibility."""
    from ase.constraints import FixAtoms

    t0 = time.perf_counter()
    try:
        run = _run_md(
            ctx,
            _md_run_dir(Path(args.out), args.tag, "t6c_bussi"),
            ensemble="nvt",
            thermostat="bussi",
            timestep_fs=BUSSI_DT_FS,
            steps=BUSSI_STEPS,
            bussi_tau_fs=BUSSI_TAU_FS,
        )
        # constraint compatibility: FixAtoms must be accepted by the product
        atoms = cu32()
        fixed = _fixed_layer_indices(atoms)
        atoms.set_constraint(FixAtoms(indices=fixed))
        run_fix = _run_md(
            ctx,
            _md_run_dir(Path(args.out), args.tag, "t6c_bussi_fixatoms"),
            ensemble="nvt",
            thermostat="bussi",
            timestep_fs=BUSSI_DT_FS,
            steps=CONSTRAINT_STEPS,
            bussi_tau_fs=BUSSI_TAU_FS,
            atoms=atoms,
        )
    except Exception as exc:  # noqa: BLE001
        record(
            ctx,
            args,
            "t6_bussi",
            "t6c_cu32_nvt_bussi",
            "fail",
            input_atoms=cu32(),
            parameters={"seed": MD_SEED},
            metrics={},
            wall=time.perf_counter() - t0,
            exception=f"{type(exc).__name__}: {exc}",
        )
        return 1

    csv = run["csv"]
    csv_fix = run_fix["csv"]
    finite = _finite_all(csv) and _finite_all(csv_fix)
    prov = run["results"]["md_provenance"]
    prov_fix = run_fix["results"]["md_provenance"]
    status = "pass" if finite else "fail"
    record(
        ctx,
        args,
        "t6_bussi",
        "t6c_cu32_nvt_bussi",
        status,
        input_atoms=cu32(),
        parameters={
            "timestep_fs": BUSSI_DT_FS,
            "steps": BUSSI_STEPS,
            "bussi_tau_fs": BUSSI_TAU_FS,
            "temperature_K": TEMPERATURE_K,
            "seed": MD_SEED,
            "fixatoms_indices": fixed,
        },
        metrics={
            **_temperature_stats(csv),
            "initial_kinetic_energy_eV": float(csv["kinetic_energy_eV"][0]),
            "max_force_raw_eV_A": float(np.max(csv["max_force_raw_eV_A"])),
            "all_finite": bool(finite),
            "degrees_of_freedom": prov["degrees_of_freedom"],
            "com_policy_effective": prov["com_policy_effective"],
            "fixatoms_run_dof": prov_fix["degrees_of_freedom"],
            "fixatoms_run_constraints": prov_fix["constraints"],
            "fixatoms_run_max_force_raw_eV_A": float(np.max(csv_fix["max_force_raw_eV_A"])),
            "thermostat_metadata": prov,
        },
        wall=time.perf_counter() - t0,
    )
    return 0 if status == "pass" else 1


# --------------------------------------------------------------------------
# t6d Nose-Hoover chain + rejection matrix
# --------------------------------------------------------------------------


def _expect_rejection(
    ctx,
    *,
    ensemble: str,
    thermostat: str | None,
    com_policy: str,
    atoms,
    label: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Assert the product refuses an unsupported constraint/COM combination.

    The runner validates the combination before any dynamics or model work,
    so this is a cheap negative test (no integration is attempted).
    """
    from mlipx.runners.md import MDRunner

    try:
        kwargs: dict[str, Any] = {
            "ensemble": ensemble,
            "timestep": CONSTRAINT_DT_FS,
            "steps": 1,
            "temperature": TEMPERATURE_K,
            "seed": MD_SEED,
            "com_policy": com_policy,
            "velocity_policy": "initialize",
            "pre_relax": False,
            "output_dir": out_dir / "_rejection_probe",
            "verbose": False,
        }
        if thermostat is not None:
            kwargs["thermostat"] = thermostat
        MDRunner(ctx.wrapper, **kwargs).run(atoms)
    except Exception as exc:  # noqa: BLE001
        return {"case": label, "rejected": True, "exception": type(exc).__name__, "message": str(exc)[:300]}
    return {"case": label, "rejected": False, "exception": None, "message": None}


def workflow_nhc(ctx, args, out_dir) -> int:
    """t6d: NHC (allowed combo only) + unsupported-combination rejections."""
    from ase.constraints import FixAtoms, FixCom

    t0 = time.perf_counter()
    rejections: list[dict[str, Any]] = []
    try:
        # positive run: NHC is only validated unconstrained with com_policy='none'
        run = _run_md(
            ctx,
            _md_run_dir(Path(args.out), args.tag, "t6d_nhc"),
            ensemble="nvt",
            thermostat="nhc",
            timestep_fs=NHC_DT_FS,
            steps=NHC_STEPS,
            nhc_tdamp_fs=NHC_TDAMP_FS,
            nhc_tchain=NHC_TCHAIN,
            nhc_tloop=NHC_TLOOP,
            com_policy="none",
        )
        # rejection matrix (section 27.4): wrong DOF counts must be refused
        rejections.append(
            _expect_rejection(
                ctx,
                ensemble="nvt",
                thermostat="nhc",
                com_policy="none",
                atoms=_atoms_with(cu32(), FixAtoms(indices=_fixed_layer_indices(cu32()))),
                label="nhc_fixatoms",
                out_dir=Path(args.out),
            )
        )
        rejections.append(
            _expect_rejection(
                ctx,
                ensemble="nvt",
                thermostat="nhc",
                com_policy="auto",
                atoms=cu32(),
                label="nhc_auto_com",
                out_dir=Path(args.out),
            )
        )
        rejections.append(
            _expect_rejection(
                ctx,
                ensemble="nvt",
                thermostat="nhc",
                com_policy="constraint",
                atoms=cu32(),
                label="nhc_com_constraint",
                out_dir=Path(args.out),
            )
        )
        compound = cu32()
        compound.set_constraint(
            [FixAtoms(indices=_fixed_layer_indices(compound)), FixCom()]
        )
        rejections.append(
            _expect_rejection(
                ctx,
                ensemble="nvt",
                thermostat="langevin",
                com_policy="constraint",
                atoms=compound,
                label="langevin_compound_constraint",
                out_dir=Path(args.out),
            )
        )
    except Exception as exc:  # noqa: BLE001
        record(
            ctx,
            args,
            "t6_nhc",
            "t6d_cu32_nvt_nhc",
            "fail",
            input_atoms=cu32(),
            parameters={"seed": MD_SEED},
            metrics={},
            wall=time.perf_counter() - t0,
            exception=f"{type(exc).__name__}: {exc}",
        )
        return 1

    csv = run["csv"]
    finite = _finite_all(csv)
    prov = run["results"]["md_provenance"]
    all_rejected = all(r["rejected"] for r in rejections)
    status = "pass" if (finite and all_rejected) else "fail"
    record(
        ctx,
        args,
        "t6_nhc",
        "t6d_cu32_nvt_nhc",
        status,
        input_atoms=cu32(),
        parameters={
            "timestep_fs": NHC_DT_FS,
            "steps": NHC_STEPS,
            "nhc_tdamp_fs": NHC_TDAMP_FS,
            "nhc_tchain": NHC_TCHAIN,
            "nhc_tloop": NHC_TLOOP,
            "temperature_K": TEMPERATURE_K,
            "seed": MD_SEED,
            "com_policy": "none",
        },
        metrics={
            **_temperature_stats(csv),
            "initial_kinetic_energy_eV": float(csv["kinetic_energy_eV"][0]),
            "max_force_raw_eV_A": float(np.max(csv["max_force_raw_eV_A"])),
            "all_finite": bool(finite),
            "degrees_of_freedom": prov["degrees_of_freedom"],
            "rejection_matrix": rejections,
            "all_unsupported_combinations_rejected": bool(all_rejected),
            "thermostat_metadata": prov,
        },
        wall=time.perf_counter() - t0,
    )
    return 0 if status == "pass" else 1


def _atoms_with(atoms, constraint):
    work = atoms.copy()
    work.set_constraint(constraint)
    return work


# --------------------------------------------------------------------------
# t6e Constraints
# --------------------------------------------------------------------------


def workflow_constraints(ctx, args, out_dir) -> int:
    """t6e: FixAtoms exactness, COM projector, DOF counts."""
    from ase.constraints import FixAtoms


    t0 = time.perf_counter()
    try:
        # FixAtoms: frozen atoms must stay exactly fixed
        atoms = cu32()
        fixed = _fixed_layer_indices(atoms)
        atoms.set_constraint(FixAtoms(indices=fixed))
        run_fix = _run_md(
            ctx,
            _md_run_dir(Path(args.out), args.tag, "t6e_fixatoms"),
            ensemble="nve",
            timestep_fs=CONSTRAINT_DT_FS,
            steps=CONSTRAINT_STEPS,
            atoms=atoms,
        )
        # COM constraint: centre of mass must stay fixed by the projector
        run_com = _run_md(
            ctx,
            _md_run_dir(Path(args.out), args.tag, "t6e_com"),
            ensemble="nve",
            timestep_fs=CONSTRAINT_DT_FS,
            steps=CONSTRAINT_STEPS,
            com_policy="constraint",
        )
        # unconstrained (no COM removal) for the 3N DOF reference
        run_free = _run_md(
            ctx,
            _md_run_dir(Path(args.out), args.tag, "t6e_free"),
            ensemble="nve",
            timestep_fs=CONSTRAINT_DT_FS,
            steps=200,
            com_policy="none",
        )
    except Exception as exc:  # noqa: BLE001
        record(
            ctx,
            args,
            "t6_constraints",
            "t6e_cu32_constraint_exactness",
            "fail",
            input_atoms=cu32(),
            parameters={"seed": MD_SEED},
            metrics={},
            wall=time.perf_counter() - t0,
            exception=f"{type(exc).__name__}: {exc}",
        )
        return 1

    initial = cu32().positions
    traj_path = run_fix["run_dir"] / "raw" / "trajectory.traj"
    fixed_max_dev = _fixed_atom_max_deviation(traj_path, initial, fixed)
    com_drift = _com_max_drift(run_com["run_dir"] / "raw" / "trajectory.traj", initial)
    prov_fix = run_fix["results"]["md_provenance"]
    prov_com = run_com["results"]["md_provenance"]
    prov_free = run_free["results"]["md_provenance"]
    natoms = len(cu32())
    dof_fix_ok = prov_fix["degrees_of_freedom"] == 3 * natoms - 3 * len(fixed)
    dof_com_ok = prov_com["degrees_of_freedom"] == 3 * natoms - 3
    dof_free_ok = prov_free["degrees_of_freedom"] == 3 * natoms
    finite = (
        _finite_all(run_fix["csv"]) and _finite_all(run_com["csv"]) and _finite_all(run_free["csv"])
    )
    exact_ok = (
        fixed_max_dev is not None
        and fixed_max_dev <= EXACTNESS_TOL_A
        and com_drift is not None
        and com_drift <= COM_DRIFT_TOL_A
    )
    status = "pass" if (finite and exact_ok and dof_fix_ok and dof_com_ok and dof_free_ok) else "fail"
    record(
        ctx,
        args,
        "t6_constraints",
        "t6e_cu32_constraint_exactness",
        status,
        input_atoms=cu32(),
        parameters={
            "timestep_fs": CONSTRAINT_DT_FS,
            "steps": CONSTRAINT_STEPS,
            "fixatoms_indices": fixed,
            "exactness_tol_A": EXACTNESS_TOL_A,
            "com_drift_tol_A": COM_DRIFT_TOL_A,
            "seed": MD_SEED,
        },
        metrics={
            "fixatoms_max_position_deviation_A": fixed_max_dev,
            "com_max_drift_A": com_drift,
            "fixatoms_dof": prov_fix["degrees_of_freedom"],
            "fixatoms_dof_expected": 3 * natoms - 3 * len(fixed),
            "com_dof": prov_com["degrees_of_freedom"],
            "com_dof_expected": 3 * natoms - 3,
            "free_dof": prov_free["degrees_of_freedom"],
            "free_dof_expected": 3 * natoms,
            "all_finite": bool(finite),
        },
        wall=time.perf_counter() - t0,
    )
    return 0 if status == "pass" else 1


def _fixed_atom_max_deviation(traj_path: Path, initial: np.ndarray, fixed: list[int]) -> float | None:
    """Max deviation of fixed atoms across all trajectory frames (bitwise 0 expected)."""
    try:
        from ase.io import read

        frames = read(traj_path, index=":")
    except Exception:  # noqa: BLE001
        return None
    if not frames:
        return None
    dev = 0.0
    for frame in frames:
        pos = frame.get_positions()
        for i in fixed:
            dev = max(dev, float(np.max(np.abs(pos[i] - initial[i]))))
    return dev


def _com_max_drift(traj_path: Path, initial: np.ndarray) -> float | None:
    """Max COM drift across frames relative to the initial COM."""
    try:
        from ase.io import read

        frames = read(traj_path, index=":")
    except Exception:  # noqa: BLE001
        return None
    if not frames:
        return None
    masses = frames[0].get_masses()
    com0 = masses @ initial / masses.sum()
    dev = 0.0
    for frame in frames:
        pos = frame.get_positions()
        com = masses @ pos / masses.sum()
        dev = max(dev, float(np.max(np.abs(com - com0))))
    return dev


# --------------------------------------------------------------------------
# t6f Force-safety abort
# --------------------------------------------------------------------------


def workflow_force_safety(ctx, args, out_dir) -> int:
    """t6f: controlled high-force abort with checkpoint/output consistency."""
    from ase.geometry import find_mic

    from mlipx.runners.md import ForceSafetyAbort

    t0 = time.perf_counter()
    try:
        # deterministic stress case: pull one atom toward its nearest
        # neighbour by a fixed displacement (bounded, no atomic overlap)
        atoms = cu32()
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        partner = int(np.argmin(d[0]))
        vec, _ = find_mic(
            atoms.positions[partner] - atoms.positions[0], atoms.cell, atoms.pbc
        )
        unit = vec / np.linalg.norm(vec)
        displaced = atoms.copy()
        displaced.positions[0] = atoms.positions[0] + FORCE_SAFETY_DISPLACEMENT_A * unit

        # pre-flight: measure the force this configuration produces, then
        # configure the safety threshold at half of it so the abort is
        # deterministic for every engine (both numbers recorded)
        probe = displaced.copy()
        probe.calc = ctx.calculator
        forces = np.asarray(probe.get_forces(), dtype=float)
        measured_force = float(np.max(np.linalg.norm(forces, axis=1)))
        threshold = FORCE_SAFETY_FRACTION * measured_force

        run_dir = _md_run_dir(Path(args.out), args.tag, "t6f_force_safety")
        aborted: ForceSafetyAbort | None = None
        try:
            _run_md(
                ctx,
                run_dir,
                ensemble="nve",
                timestep_fs=1.0,
                steps=FORCE_SAFETY_STEPS,
                fmax_abort=threshold,
                atoms=displaced,
            )
        except ForceSafetyAbort as exc:
            aborted = exc
    except Exception as exc:  # noqa: BLE001
        record(
            ctx,
            args,
            "t6_force_safety",
            "t6f_cu32_force_safety_abort",
            "fail",
            input_atoms=cu32(),
            parameters={"seed": MD_SEED},
            metrics={},
            wall=time.perf_counter() - t0,
            exception=f"{type(exc).__name__}: {exc}",
        )
        return 1

    diagnostics: dict[str, Any] = {}
    if aborted is not None:
        manifest_path = run_dir / "artifacts.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        csv_ok = (run_dir / "raw" / "md.csv").exists()
        diagnostics = {
            "abort_raised": True,
            "abort_step": aborted.step,
            "abort_atom_index": aborted.atom_index,
            "abort_max_force_raw_eV_A": aborted.max_force,
            "threshold_configured_eV_A": aborted.threshold,
            "artifacts_manifest_status": manifest.get("status"),
            "artifacts_error": manifest.get("error"),
            "unsafe_frame_checkpointed": bool(csv_ok),
            "contcar_written": (run_dir / "vasp" / "CONTCAR").exists(),
        }
        clean = (
            diagnostics["artifacts_manifest_status"] == "aborted"
            and diagnostics["artifacts_error"] is not None
            and diagnostics["artifacts_error"].get("type") == "force_safety_abort"
            and diagnostics["unsafe_frame_checkpointed"]
            and diagnostics["contcar_written"]
            and aborted.max_force > aborted.threshold
        )
    else:
        diagnostics = {
            "abort_raised": False,
            "threshold_configured_eV_A": threshold,
            "measured_preflight_force_eV_A": measured_force,
        }
        clean = False
    status = "pass" if clean else "fail"
    record(
        ctx,
        args,
        "t6_force_safety",
        "t6f_cu32_force_safety_abort",
        status,
        input_atoms=displaced,
        parameters={
            "displacement_A": FORCE_SAFETY_DISPLACEMENT_A,
            "displacement_direction": "toward nearest neighbour (MIC)",
            "threshold_fraction_of_measured": FORCE_SAFETY_FRACTION,
            "steps_before_abort": FORCE_SAFETY_STEPS,
            "seed": MD_SEED,
        },
        metrics={
            "measured_preflight_force_eV_A": measured_force,
            "partner_atom_index": partner,
            **diagnostics,
        },
        wall=time.perf_counter() - t0,
    )
    return 0 if status == "pass" else 1


WORKFLOW_FUNCS = {
    "nve_dt_sweep": workflow_nve_dt_sweep,
    "langevin": workflow_langevin,
    "bussi": workflow_bussi,
    "nhc": workflow_nhc,
    "constraints": workflow_constraints,
    "force_safety": workflow_force_safety,
}

DEFAULT_ORDER = (
    "nve_dt_sweep",
    "langevin",
    "bussi",
    "nhc",
    "constraints",
    "force_safety",
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
        "--workflows",
        default=",".join(DEFAULT_ORDER),
        help=f"comma list from {sorted(WORKFLOW_FUNCS)}",
    )
    parser.add_argument("--tag", default=None, help="filename suffix")
    args = parser.parse_args()

    requested = [w.strip() for w in args.workflows.split(",") if w.strip()]
    unknown = [w for w in requested if w not in WORKFLOW_FUNCS]
    if unknown:
        parser.error(f"unknown workflows: {unknown}")

    manifest = common.load_model_manifest(args.manifest)
    profile = common.resolve_profile(manifest, args.profile_id, args.model)
    out_dir = Path(args.out)

    failures = 0
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
    except Exception as exc:  # noqa: BLE001 - record and return failure
        rec = common.result_record(
            case_id="t6_setup",
            test_id="t6_md_suite_setup",
            status="fail",
            engine=args.engine,
            model_identity=profile.get("identity", args.profile_id),
            model_sha256=profile["model_sha256"],
            task=profile.get("task"),
            head=args.head,
            dtype=args.dtype or "upstream/model-defined",
            device=common.device_record(args.device),
            input_structure_id=None,
            parameters={"workflows": requested},
            metrics={},
            diagnostics={"traceback": traceback.format_exc()},
            exception=f"{type(exc).__name__}: {exc}",
        )
        common.write_result(rec, out_dir, tag=args.tag)
        return 1

    for wf in requested:
        failures += WORKFLOW_FUNCS[wf](ctx, args, out_dir)

    print(f"md_suite: engine={ctx.engine} workflows={requested} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
