#!/usr/bin/env python
"""T7: physical transport analysis through the product pipeline (PR5 GPU part).

For the public alpha-Na3PS4 fixture (Materials Project mp-28782 geometry,
pinned in ``fixtures.py``), run a bounded high-temperature MD trajectory per
backend and push it through the product analysis pipeline:

- product MDRunner NVT (NHC thermostat) -> raw/trajectory.traj (unwrapped)
- ``TrajectoryDataset.load`` (the product's own loading path, provenance kept)
- thermo stability record from the runner CSV
- kinisi tracer diffusion + Nernst-Einstein conductivity (transport authority)
- native MSD diagnostic crosscheck + diffusion sufficiency gate
- GEMDAT site mapping / jumps / residence (diagnostic crosscheck) with an
  explicit reproducible site source (the pinned crystallographic Na sites)
- optional Arrhenius fit when >= 3 temperatures all pass the sufficiency gate

Literature-motivated protocol (recorded per taskbook section 30):

- Na3PS4 undergoes a beta -> alpha transition at ~533 K; the alpha phase is
  the superionic regime (Krauskopf et al. 2017; ACS Mater. Lett. 2019 shows
  the material stays solid well above 500 C).
- T = 700 K sits ~170 K above the transition, safely superionic and far from
  melting; 600 K / 800 K bracket it for the optional Arrhenius stage.
- dt = 2 fs (stable in the t6 NVE sweep for all four engines at comparable
  force magnitudes; energy drift is monitored in the record).

The result is a *demonstration*, not a claim of converged bulk conductivity.

No silent fallbacks: if the engine cannot be loaded under its profile the
case fails (taskbook section 47).
"""

from __future__ import annotations

import csv
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
from engines import EngineContext  # noqa: E402

MD_SEED = 20260911
DEFAULT_TEMPERATURES_K = (700.0,)
EQUILIBRATION_PS = 1.5
PRODUCTION_PS = 15.0
TIMESTEP_FS = 2.0
SAVE_INTERVAL = 25  # 50 fs frames; ~300 production frames for 15 ps
NHC_TDAMP_FS = 100.0
NHC_TCHAIN = 3
NHC_TLOOP = 1
MOBILE_SPECIES = "Na"
IONIC_CHARGE_E = 1.0
# Explicitly justified kinisi fit window: start 2 ps into production (40
# frames) so the earliest, most correlated interval is excluded.
KINISI_FIT_START_PS = 2.0
# Sufficiency gate: the native-MSD fit must see a strictly positive
# Einstein-region slope; otherwise sampling is insufficient.
MIN_MSD_SLOPE_M2_S = 1.0e-11
MIN_GEMDAT_EVENTS = 24


def na3ps4_supercell() -> Any:
    """128-atom 2x2x2 supercell of the pinned alpha-Na3PS4 fixture."""
    atoms, _ = fixtures.build_extra("na3ps4")
    return atoms.repeat((2, 2, 2))


def write_na_sites_vasp(path: Path) -> str:
    """Write the pinned crystallographic Na sites as an explicit site source."""
    from ase.io import write

    atoms = na3ps4_supercell()
    symbols = atoms.get_chemical_symbols()
    na_idx = [i for i, s in enumerate(symbols) if s == MOBILE_SPECIES]
    sites = atoms[na_idx]
    write(str(path), sites, format="vasp", direct=True)
    return common.structure_id(sites)


def _structure_id(atoms) -> str:
    return common.structure_id(atoms)


def _run_dir(out_dir: Path, tag: str | None, name: str) -> Path:
    d = out_dir / "md_runs" / (tag or "default") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _md_run_dir_for(args, temperature_k: float, engine: str) -> Path:
    """Resolve the trajectory run directory for one engine+temperature.

    Run directories are engine-qualified: a completed run belongs to exactly
    one calculator and must never be silently reused by another backend.

    ``--run-dir`` (analysis-only phase) applies to a single temperature.
    """
    if getattr(args, "run_dir", None):
        if len([t for t in args.temperatures.split(",") if t.strip()]) > 1:
            raise ValueError(
                "--run-dir applies to a single temperature; omit it or pass "
                "one temperature"
            )
        return Path(args.run_dir)
    name = f"t7_na3ps4_{engine}_T{temperature_k:g}K"
    return _run_dir(Path(args.out), args.tag, name)


def _md_run_complete(run_dir: Path, equil_steps: int, prod_steps: int) -> bool:
    """A previous MD run in this directory already covers the full protocol."""
    csv_path = run_dir / "raw" / "md.csv"
    if not csv_path.exists():
        return False
    try:
        with csv_path.open(newline="") as fh:
            steps = [int(row["step"]) for row in csv.DictReader(fh)]
    except (OSError, KeyError, ValueError):
        return False
    return bool(steps) and steps[-1] >= equil_steps + prod_steps - 1


def _load_md_dataset(args, temperature_k: float, engine: str):
    """Load one completed MD run through the product loading path."""
    from mliport.analysis.dataset import TrajectoryDataset

    run_dir = _md_run_dir_for(args, temperature_k, engine)
    equil_steps = int(round(EQUILIBRATION_PS * 1000.0 / TIMESTEP_FS))
    prod_steps = int(round(args.production_ps * 1000.0 / TIMESTEP_FS))
    if not _md_run_complete(run_dir, equil_steps, prod_steps):
        raise FileNotFoundError(
            f"MD run for T={temperature_k:g} K is missing or incomplete under "
            f"{run_dir}; run --cases md first in the engine environment"
        )
    return run_dir, TrajectoryDataset.load(run_dir)


def _production_csv_rows(run_dir: Path, equil_steps: int) -> list[dict[str, str]]:
    with (run_dir / "raw" / "md.csv").open(newline="") as fh:
        return [row for row in csv.DictReader(fh) if int(row["step"]) >= equil_steps]


def _analysis_backend_version(args) -> str:
    """Backend package version for the analysis-only context (no MLIP load).

    The analysis phase runs in the analysis venv where the engine package may
    not be importable; the version string is informational provenance and must
    never abort the setup (regression: engines.package_version did not exist).
    """
    if args.backend_version:
        return str(args.backend_version)
    return common.package_version(engines._backend_dist(args.engine))


def _kinisi_transport(dataset, temperature_k: float) -> dict[str, Any]:
    from mliport.analysis.transport import kinisi_transport

    result = kinisi_transport(
        dataset,
        mobile_species=MOBILE_SPECIES,
        ionic_charge_e=IONIC_CHARGE_E,
        temperature_K=temperature_k,
        fit_start_ps=KINISI_FIT_START_PS,
        random_seed=MD_SEED,
    )
    tracer = result.get("tracer_diffusion", {})
    ne = result.get("nernst_einstein", {})
    return {
        "kinisi_version": tracer.get("kinisi_version"),
        "fit_start_ps": KINISI_FIT_START_PS,
        "D_posterior_m2_s": tracer.get("D_posterior_m2_s"),
        "D_posterior_cm2_s": tracer.get("D_posterior_cm2_s"),
        "sigma_NE_S_m": ne.get("sigma_NE_tracer_S_m"),
        "sigma_NE_mS_cm": ne.get("sigma_NE_tracer_mS_cm"),
        "sigma_NE_posterior_mS_cm": ne.get("sigma_NE_tracer_posterior_mS_cm"),
        "nernst_einstein_definition": ne.get("definition"),
        "drift_semantics": {
            "drift_mode": (result.get("drift_correction") or {}).get("drift_mode"),
            "drift_reference": (result.get("drift_correction") or {}).get(
                "drift_reference"
            ),
            "center_definition": (result.get("drift_correction") or {}).get(
                "center_definition"
            ),
        },
        "displacement_input_class": result.get("displacement_input_class"),
        "publication_grade": result.get("publication_grade"),
        "kinisi_position_semantics": result.get("kinisi_position_semantics"),
    }


def _native_msd_diagnostic(dataset) -> dict[str, Any]:
    """Native-MSD diagnostic via the product's explicit-range OLS fit.

    The sufficiency gate uses the product's own diagnostic diffusion fit over
    the last half of the lag range (Einstein region).  The log-log alpha and
    the product's diffusive-regime warning are reported but do not gate:
    high-temperature demonstration trajectories are expected to be noisy.
    """
    from mliport.analysis.msd import calculate_msd, diagnostic_linear_diffusion_fit

    result = calculate_msd(dataset, mobile_species=MOBILE_SPECIES)
    lag_ps = result["lag_time_ps"]
    msd_xyz = np.asarray(result["msd_by_axes_A2"]["xyz"], dtype=float)
    half_start = float(lag_ps[len(lag_ps) // 2])
    fit = diagnostic_linear_diffusion_fit(
        lag_ps,
        msd_xyz,
        axes="xyz",
        fit_start_ps=half_start,
        fit_stop_ps=float(lag_ps[-1]),
    )
    drift = result.get("drift_semantics") or result.get("drift_correction") or {}
    return {
        "publication_grade": fit["publication_grade"],
        "drift_semantics": {
            "drift_mode": drift.get("drift_mode") or drift.get("mode"),
            "drift_reference": drift.get("drift_reference"),
            "center_definition": drift.get("center_definition"),
        },
        "diagnostic_fit": fit,
        "D_diagnostic_m2_s": fit["D_diagnostic_m2_s"],
        "mean_log_log_alpha_in_fit": fit["mean_log_log_alpha_in_fit"],
        "diffusive_regime_warning": fit["diffusive_regime_warning"],
        "msd_final_A2": float(msd_xyz[-1]),
    }


def _gemdat_diagnostic(dataset, args, temperature_k: float) -> dict[str, Any]:
    from mliport.analysis.electrolyte import gemdat_electrolyte

    sites_path = Path(args.sites)
    result = gemdat_electrolyte(
        dataset,
        mobile_species=MOBILE_SPECIES,
        sites_path=str(sites_path),
        temperature_K=temperature_k,
    )
    summary = result.summary
    transition_status = summary.get("gemdat_transition_status")
    # Legacy records may predate the status field; a recorded backend_error or
    # invalid_input must never be downgraded to "0 transition events" (PR-C).
    if transition_status not in (None, "success", "no_events"):
        error = summary.get("gemdat_transition_error") or {}
        raise RuntimeError(
            "GEMDAT transition detection did not produce a physical result "
            f"({transition_status}): "
            f"{error.get('type', 'unknown')}: {error.get('message', '')}"
        )
    tables = result.tables
    events = tables.get("transition_events")
    jumps = tables.get("jumps")
    n_events = int(len(events)) if events is not None else 0
    jump_distances = None
    if jumps is not None and "jump_distance_A" in jumps:
        jd = jumps["jump_distance_A"].to_numpy()
        if len(jd):
            jump_distances = {
                "mean_A": float(np.mean(jd)),
                "max_A": float(np.max(jd)),
            }
    return {
        "site_source": str(sites_path.resolve()),
        "site_source_kind": "pinned_crystallographic_na_sites",
        "summary": summary,
        "gemdat_transition_status": transition_status or "legacy_unknown",
        "warnings": list(result.warnings),
        "n_transition_events": n_events,
        "n_jumps": int(len(jumps)) if jumps is not None else 0,
        "jump_distances": jump_distances,
        "n_residence_rows": int(len(tables["residence_times"]))
        if tables.get("residence_times") is not None
        else 0,
        "percolation": result.summary.get("percolation"),
        "n_paths": len(result.paths),
        "diffusion_diagnostic": result.summary.get("diffusion"),
    }


def _arrhenius_fit(
    temperatures: list[float],
    diffusivities: list[float],
    uncertainties: list[float] | None,
) -> dict[str, Any]:
    from mliport.analysis.arrhenius import fit_arrhenius

    result = fit_arrhenius(
        temperatures_K=np.asarray(temperatures, dtype=float),
        diffusivities_m2_s=np.asarray(diffusivities, dtype=float),
        diffusivity_std_m2_s=None
        if uncertainties is None
        else np.asarray(uncertainties, dtype=float),
    )
    return {
        "activation_energy_eV": result.get("activation_energy_eV"),
        "preexponential_factor_m2_s": result.get("preexponential_factor_m2_s"),
        "r_squared": result.get("r_squared"),
        "warnings": list(result.get("warnings", [])),
        "n_temperature_runs": result.get("number_of_independent_temperature_runs"),
    }


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain JSON types."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(
    ctx,
    case_id,
    test_id,
    status,
    parameters,
    metrics,
    diagnostics=None,
    structure_id=None,
    wall=0.0,
    peak_vram=None,
    exception=None,
) -> int:
    rec = common.result_record(
        case_id=case_id,
        test_id=test_id,
        status=status,
        engine=ctx.engine,
        model_identity=ctx.model_identity,
        model_sha256=ctx.model_sha256,
        task=ctx.task,
        head=ctx.head,
        dtype=ctx.dtype,
        device=common.device_record(ctx.device),
        input_structure_id=structure_id,
        parameters=_jsonable(parameters),
        metrics=_jsonable(metrics),
        diagnostics=diagnostics,
        wall_seconds=wall,
        peak_vram=peak_vram,
        exception=exception,
    )
    common.write_result(rec, Path(ctx.args.out), tag=ctx.args.tag)
    # Honest 'unsupported' classifications (insufficient_sampling, upstream
    # limitations, product refusals) are successful suite outcomes; only a
    # 'fail' (product error) propagates a non-zero exit code.
    return 0 if status in ("pass", "characterized", "unsupported") else 1


def run_md_case(ctx, args) -> int:
    """t7-md: product MDRunner trajectory for the Na3PS4 fixture.

    Runs in the engine environment only; analysis cases later load the
    completed run directory through the product loading path.
    """
    from mliport.runners.md import MDRunner

    temperatures = [float(t) for t in args.temperatures.split(",")]
    atoms = na3ps4_supercell()
    natoms = len(atoms)
    atoms_id = _structure_id(atoms)
    failures = 0
    equil_steps = int(round(EQUILIBRATION_PS * 1000.0 / TIMESTEP_FS))
    prod_steps = int(round(args.production_ps * 1000.0 / TIMESTEP_FS))

    common_params = {
        "fixture": "na3ps4_mp28782_2x2x2",
        "natoms": natoms,
        "ensemble": "nvt_nhc",
        "timestep_fs": TIMESTEP_FS,
        "equilibration_ps": EQUILIBRATION_PS,
        "production_ps": args.production_ps,
        "save_interval_steps": SAVE_INTERVAL,
        "seed": MD_SEED,
        "com_policy": "none",
        "velocity_policy": "initialize",
    }
    for temperature in temperatures:
        t0 = time.perf_counter()
        try:
            run_dir = _md_run_dir_for(args, temperature, ctx.engine)
            if _md_run_complete(run_dir, equil_steps, prod_steps):
                rows = _production_csv_rows(run_dir, equil_steps)
            else:
                runner = MDRunner(
                    ctx.wrapper,
                    ensemble="nvt",
                    thermostat="nhc",
                    nhc_tdamp=NHC_TDAMP_FS,
                    nhc_tchain=NHC_TCHAIN,
                    nhc_tloop=NHC_TLOOP,
                    timestep=TIMESTEP_FS,
                    steps=prod_steps,
                    equilibration_steps=equil_steps,
                    temperature=temperature,
                    seed=MD_SEED,
                    com_policy="none",  # NHC forbids automatic COM removal
                    velocity_policy="initialize",
                    pre_relax=False,
                    output_dir=run_dir,
                    save_interval=SAVE_INTERVAL,
                    verbose=False,
                )
                runner.run(atoms)
                rows = _production_csv_rows(run_dir, equil_steps)
            t_vals = np.array([float(r["temperature_K"]) for r in rows])
            e_tot = np.array([float(r["total_energy_eV"]) for r in rows])
            finite = bool(np.all(np.isfinite(t_vals)) and np.all(np.isfinite(e_tot)))
            metrics = {
                "temperature_target_K": temperature,
                "temperature_mean_K": float(np.mean(t_vals)),
                "temperature_std_K": float(np.std(t_vals)),
                "temperature_min_K": float(np.min(t_vals)),
                "temperature_max_K": float(np.max(t_vals)),
                "all_finite": finite,
                "production_rows": int(len(rows)),
                "total_energy_drift_eV_atom_ps": float(
                    np.polyfit(
                        np.array([float(r["time_fs"]) for r in rows]) / 1000.0,
                        e_tot / natoms,
                        1,
                    )[0]
                ),
            }
            status = "pass" if finite else "fail"
            diagnostics = (
                {}
                if finite
                else {"reason": "non_finite_trajectory", "instability": True}
            )
            failures += _write(
                ctx,
                "t7_md",
                f"t7m0_na3ps4_T{temperature:g}K",
                status,
                common_params,
                metrics,
                diagnostics,
                structure_id=atoms_id,
                wall=time.perf_counter() - t0,
                peak_vram=common.peak_vram_mib(),
            )
        except Exception as exc:  # noqa: BLE001, PERF203
            failures += _write(
                ctx,
                "t7_md",
                f"t7m0_na3ps4_T{temperature:g}K",
                "fail",
                common_params,
                {},
                {"traceback": traceback.format_exc()},
                structure_id=atoms_id,
                wall=time.perf_counter() - t0,
                exception=f"{type(exc).__name__}: {exc}",
            )
    return failures


def _classify_transport_failure(exc: Exception) -> tuple[str, dict[str, Any]]:
    """Honest classification for transport case failures.

    A trajectory whose saved interval cannot be uniquely unwrapped is
    'unsupported' with the explicit insufficient_trajectory_information
    reason, not a generic product failure (PR-E section 7.4).
    """
    from mliport.analysis.validation import (
        InsufficientTrajectoryInformationError,
        UnsupportedAnalysisError,
    )

    if isinstance(exc, InsufficientTrajectoryInformationError):
        return "unsupported", {
            "reason": "insufficient_trajectory_information",
            "exception": str(exc),
        }
    if isinstance(exc, UnsupportedAnalysisError):
        return "unsupported", {"reason": "product_refused", "exception": str(exc)}
    return "fail", {}


def _drift_comparability(
    native: dict[str, Any], kinisi: dict[str, Any]
) -> dict[str, Any]:
    """Cross-backend drift-definition comparison (PR-F section 8.3).

    A native-MSD/kinisi numerical comparison is only meaningful when both
    backends used the same centre definition and reference selection; this
    records the comparison instead of silently claiming agreement.
    """
    from mliport.analysis.drift import drift_definitions_match

    native_drift = native.get("drift_semantics") or {}
    kinisi_drift = kinisi.get("drift_semantics") or {}
    return {
        "definitions_match": drift_definitions_match(native_drift, kinisi_drift),
        "native_msd": native_drift,
        "kinisi_transport": kinisi_drift,
        "note": (
            "cross-backend numerical agreement may only be claimed when "
            "definitions_match is true; the sampling sufficiency gate is "
            "independent of this comparison"
        ),
    }


def run_transport_case(ctx, args) -> int:
    """t7a: kinisi + NE + native-MSD sufficiency per temperature (loads MD runs)."""
    temperatures = [float(t) for t in args.temperatures.split(",")]
    atoms = na3ps4_supercell()
    natoms = len(atoms)
    atoms_id = _structure_id(atoms)
    failures = 0
    equil_steps = int(round(EQUILIBRATION_PS * 1000.0 / TIMESTEP_FS))
    prod_steps = int(round(args.production_ps * 1000.0 / TIMESTEP_FS))

    common_params = {
        "fixture": "na3ps4_mp28782_2x2x2",
        "natoms": natoms,
        "ensemble": "nvt_nhc",
        "timestep_fs": TIMESTEP_FS,
        "equilibration_ps": EQUILIBRATION_PS,
        "production_ps": args.production_ps,
        "save_interval_steps": SAVE_INTERVAL,
        "seed": MD_SEED,
        "mobile_species": MOBILE_SPECIES,
        "ionic_charge_e": IONIC_CHARGE_E,
        "protocol": "demonstration_not_converged",
    }

    diffusive_T: list[float] = []
    D_values: list[float] = []
    D_sigmas: list[float] = []
    t0 = time.perf_counter()
    for temperature in temperatures:
        peak_vram = None
        try:  # noqa: PERF203
            run_dir, dataset = _load_md_dataset(args, temperature, ctx.engine)
            peak_vram = common.peak_vram_mib()
            # Thermo stability from the runner CSV (production rows only).
            rows = _production_csv_rows(run_dir, equil_steps)
            t_vals = np.array([float(r["temperature_K"]) for r in rows])
            e_tot = np.array([float(r["total_energy_eV"]) for r in rows])
            finite = bool(np.all(np.isfinite(t_vals)) and np.all(np.isfinite(e_tot)))

            msd = _native_msd_diagnostic(dataset)
            sufficient = bool(finite and msd["D_diagnostic_m2_s"] >= MIN_MSD_SLOPE_M2_S)

            metrics: dict[str, Any] = {
                "temperature_target_K": temperature,
                "temperature_mean_K": float(np.mean(t_vals)),
                "temperature_std_K": float(np.std(t_vals)),
                "all_finite": finite,
                "total_energy_drift_eV_atom_ps": float(
                    np.polyfit(
                        np.array([float(r["time_fs"]) for r in rows]) / 1000.0,
                        e_tot / natoms,
                        1,
                    )[0]
                ),
                "native_msd": msd,
                "sampling_sufficient": sufficient,
            }
            status = "characterized"
            diagnostics: dict[str, Any] = {}
            if sufficient:
                kin = _kinisi_transport(dataset, temperature)
                metrics["kinisi_transport"] = kin
                metrics["cross_backend_drift"] = _drift_comparability(msd, kin)
                post = kin["D_posterior_m2_s"]
                d_mean = post["mean"] if isinstance(post, dict) else post
                d_std = post.get("std", 0.0) if isinstance(post, dict) else 0.0
                diffusive_T.append(temperature)
                D_values.append(float(d_mean))
                D_sigmas.append(float(d_std))
            else:
                status = "unsupported"
                diagnostics["reason"] = "insufficient_sampling"
                diagnostics["gate"] = {
                    "min_d_diagnostic_m2_s": MIN_MSD_SLOPE_M2_S,
                    "all_finite": finite,
                }
            failures += _write(
                ctx,
                "t7_transport",
                f"t7a_na3ps4_T{temperature:g}K",
                status,
                common_params,
                metrics,
                diagnostics,
                structure_id=atoms_id,
                wall=time.perf_counter() - t0,
                peak_vram=peak_vram,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            status, reason = _classify_transport_failure(exc)
            failures += _write(
                ctx,
                "t7_transport",
                f"t7a_na3ps4_T{temperature:g}K",
                status,
                common_params,
                {},
                {**reason, "traceback": traceback.format_exc()},
                structure_id=atoms_id,
                wall=time.perf_counter() - t0,
                exception=f"{type(exc).__name__}: {exc}",
            )

    # Optional Arrhenius stage: only if >= 3 temperatures all sampled.
    if len(diffusive_T) >= 3:
        try:
            sigmas = D_sigmas if all(s > 0 for s in D_sigmas) else None
            fit = _arrhenius_fit(diffusive_T, D_values, sigmas)
            _write(
                ctx,
                "t7_arrhenius",
                "t7b_na3ps4_arrhenius",
                "characterized",
                {
                    **common_params,
                    "temperatures_K": diffusive_T,
                    "demonstration_only": True,
                },
                {"fit": fit, "D_values_m2_s": D_values, "D_std_m2_s": D_sigmas},
                structure_id=atoms_id,
                wall=time.perf_counter() - t0,
            )
        except Exception as exc:  # noqa: BLE001
            failures += _write(
                ctx,
                "t7_arrhenius",
                "t7b_na3ps4_arrhenius",
                "fail",
                common_params,
                {"temperatures_K": diffusive_T},
                {"traceback": traceback.format_exc()},
                structure_id=atoms_id,
                exception=f"{type(exc).__name__}: {exc}",
            )
    return failures


def _classify_gemdat_failure(exc: Exception) -> tuple[str, dict[str, Any]]:
    """Honest classification for GEMDAT case failures.

    The product's own UnsupportedAnalysisError refusals and the known upstream
    pymatgen occupancy limitation are 'unsupported' (diagnostic unavailable),
    not 'fail' (which is reserved for product errors).
    """
    from mliport.analysis.validation import UnsupportedAnalysisError

    if isinstance(exc, UnsupportedAnalysisError):
        return "unsupported", {"reason": "product_refused", "exception": str(exc)}
    if "occupancies sum to more than 1" in str(exc):
        return "unsupported", {
            "reason": "upstream_pymatgen_occupancy_limitation",
            "exception": str(exc),
        }
    return "fail", {}


def run_gemdat_case(ctx, args) -> int:
    """t7c: GEMDAT diagnostic on the same physical trajectory."""
    temperatures = [float(t) for t in args.temperatures.split(",")]
    atoms = na3ps4_supercell()
    atoms_id = _structure_id(atoms)
    failures = 0
    for temperature in temperatures:
        try:  # noqa: PERF203
            _, dataset = _load_md_dataset(args, temperature, ctx.engine)
            diag = _gemdat_diagnostic(dataset, args, temperature)
            sufficient = diag["n_transition_events"] >= MIN_GEMDAT_EVENTS
            status = "characterized" if sufficient else "unsupported"
            diagnostics = {}
            if not sufficient:
                diagnostics["reason"] = "insufficient_sampling"
                diagnostics["min_transition_events"] = MIN_GEMDAT_EVENTS
            failures += _write(
                ctx,
                "t7_gemdat",
                f"t7c_na3ps4_T{temperature:g}K",
                status,
                {
                    "fixture": "na3ps4_mp28782_2x2x2",
                    "mobile_species": MOBILE_SPECIES,
                    "role": "diagnostic_crosscheck",
                },
                diag,
                diagnostics,
                structure_id=atoms_id,
                peak_vram=common.peak_vram_mib(),
            )
        except Exception as exc:  # noqa: BLE001, PERF203
            status, reason = _classify_gemdat_failure(exc)
            failures += _write(
                ctx,
                "t7_gemdat",
                f"t7c_na3ps4_T{temperature:g}K",
                status,
                {"fixture": "na3ps4_mp28782_2x2x2"},
                {},
                {**reason, "traceback": traceback.format_exc()},
                structure_id=atoms_id,
                exception=f"{type(exc).__name__}: {exc}",
            )
    return failures


def main() -> int:
    import argparse

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
        "--temperatures",
        default=",".join(f"{t:g}" for t in DEFAULT_TEMPERATURES_K),
        help="comma list of temperatures in K (Arrhenius needs >= 3)",
    )
    parser.add_argument("--production-ps", type=float, default=PRODUCTION_PS)
    parser.add_argument("--sites", default=None, help="explicit Na sites VASP file")
    parser.add_argument(
        "--cases",
        default="md,transport,gemdat",
        help="comma list from ['md', 'transport', 'gemdat']",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="existing MD run dir (analysis-only phase; single temperature)",
    )
    parser.add_argument(
        "--backend-version",
        default=None,
        help="backend distribution version override (analysis-only phase: "
        "the engine venv, not this interpreter, holds the package)",
    )
    parser.add_argument("--tag", default=None, help="filename suffix")
    args = parser.parse_args()

    if args.sites is None:
        sites = Path(args.out) / "na3ps4_sites.vasp"
        sites.parent.mkdir(parents=True, exist_ok=True)
        write_na_sites_vasp(sites)
        args.sites = str(sites)

    manifest = common.load_model_manifest(args.manifest)
    profile = common.resolve_profile(manifest, args.profile_id, args.model)

    requested = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in requested if c not in ("md", "transport", "gemdat")]
    if unknown:
        parser.error(f"unknown cases: {unknown}")

    need_engine = "md" in requested
    try:
        if need_engine:
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
        else:
            # Analysis-only phase: metadata-only context, no MLIP loading.
            ctx = EngineContext(
                engine=args.engine,
                profile_id=args.profile_id,
                profile=profile,
                wrapper=None,
                calculator=None,
                device=args.device,
                dtype="engine-defined",  # MD ran in the engine venv
                task=profile.get("task"),
                head=args.head,
                backend_version=_analysis_backend_version(args),
                framework_version="n/a",
            )
    except Exception as exc:  # noqa: BLE001 - record and return failure
        rec = common.result_record(
            case_id="t7_setup",
            test_id="t7_analysis_suite_setup",
            status="fail",
            engine=args.engine,
            model_identity=profile.get("identity", args.profile_id),
            model_sha256=profile["model_sha256"],
            task=profile.get("task"),
            head=args.head,
            dtype=args.dtype or "upstream/model-defined",
            device=common.device_record(args.device),
            input_structure_id=None,
            parameters={"cases": requested},
            metrics={},
            diagnostics={"traceback": traceback.format_exc()},
            exception=f"{type(exc).__name__}: {exc}",
        )
        common.write_result(rec, Path(args.out), tag=args.tag)
        return 1
    ctx.args = args

    failures = 0
    if "md" in requested:
        failures += run_md_case(ctx, args)
    if "transport" in requested:
        failures += run_transport_case(ctx, args)
    if "gemdat" in requested:
        failures += run_gemdat_case(ctx, args)

    print(f"analysis_suite: engine={ctx.engine} cases={requested} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
