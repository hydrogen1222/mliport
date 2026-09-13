"""T4 static DFT-style workflow matrix (taskbook sections 15-24).

Runs relax / cell relax / EOS / elastic / phonon / harmonic thermodynamics /
vacancy / surface / conditional formation-energetics workflows with the same
workflow definition on every real backend.  Every record proves calculator
identity through the mliport CalculatorFactory; no synthetic calculator may
appear here (taskbook section 47).

Systems: Cu fcc, Si diamond, MgO rocksalt (mandatory common) and alpha-Na3PS4
(P-42_1c, pinned public fixture from Materials Project mp-28782 -- geometry
fixture only, never compared against MP energies).

Workflow quality gates (status semantics):

- ``pass``          workflow completed and its internal convergence/fit
                    criteria were met
- ``characterized`` workflow completed; the observable is a model
                    characterization without an external reference (defect
                    energies, surface energies) or an internal criterion that
                    describes the model rather than the workflow
- ``fail``          non-convergence, bad fit, or robust imaginary modes
- ``unsupported``   conditional workflows the taskbook forbids without pinned
                    references (cohesive/formation energy)
"""

from __future__ import annotations

import argparse
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

KB_EV_PER_K = 8.617333262145e-5  # ase.units.kB
EV_TO_CM1 = 8065.543937

SEED = 20260911
RELAX_FMAX = 0.01  # eV/A
RELAX_MAX_STEPS = 500
VACANCY_RELAX_FMAX = 0.02
SURFACE_RELAX_FMAX = 0.02
INTERNAL_RELAX_FMAX = 1e-4  # relaxed-ion elastic internal relaxation
CONV_SYMPREC = 1e-3

EOS_RATIOS = (0.94, 0.955, 0.97, 0.985, 1.00, 1.015, 1.03, 1.045, 1.06)
EOS_RESIDUAL_MAX_EV = 0.005  # eV/atom max |E_fit - E_sampled| for a good fit

ELASTIC_STRAINS = (0.0025, 0.005, 0.010)
ELASTIC_RESIDUAL_MAX = 1e-3  # eV/A^3 max |sigma_fit - sigma_sampled|

PHONON_DELTAS = (0.005, 0.010, 0.020)
PHONON_SUPERCELLS = ((2, 2, 2), (3, 3, 3))
THERMO_DOS_GRID = (8, 8, 8)  # Monkhorst-Pack q-grid for thermal sums
THERMO_TEMPERATURES = (100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0)
IMAGINARY_TOL_EV = 1e-4  # 0.1 meV: below this, imaginary modes are numerical
SUM_EXCLUSION_TOL_EV = 1e-6  # modes below this never enter thermo F/S/Cv sums
ACOUSTIC_TOL_EV = 3e-3  # Gamma acoustic modes must be below 3 meV (ASR)

VACANCY_SUPERCELLS = ((2, 2, 2), (3, 3, 3), (4, 4, 4))
SURFACE_LAYERS = (4, 6, 8)
SURFACE_VACUUMS = (10.0, 15.0)

RELAX_SYSTEMS = ("cu_fcc", "si_diamond", "mgo_rocksalt", "na3ps4")
EOS_SYSTEMS = ("cu_fcc", "si_diamond", "mgo_rocksalt")
ELASTIC_SYSTEMS = ("cu_fcc", "si_diamond")
PHONON_SYSTEMS = ("cu_fcc_prim", "si_diamond_prim")
T4A_SYSTEMS = ("cu_fcc", "si_diamond", "mgo_rocksalt", "na3ps4")
T4A_VARIANTS = (
    "equilibrium",
    "disp_0.01A",
    "disp_0.05A",
    "compressed_0.98",
    "expanded_1.02",
)


# --------------------------------------------------------------------------
# deterministic geometry helpers
# --------------------------------------------------------------------------


def seeded_displacements(natoms: int, magnitude: float, seed: int = SEED) -> np.ndarray:
    """Per-atom random unit vectors scaled to exactly ``magnitude`` Angstrom."""
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(natoms, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec * magnitude


def apply_strain(atoms, kind: str, delta: float):
    """Return a copy of ``atoms`` with cell strained by ``eps(kind, delta)``.

    Row-vector lattice convention: ``new_cell = old_cell @ (I + eps).T``.
    Fractional coordinates are kept fixed (clamped-ion deformations).
    """
    eps = strain_eps(kind, delta)
    new = atoms.copy()
    new.set_cell(atoms.cell.array @ (np.eye(3) + eps).T, scale_atoms=True)
    return new


def strain_eps(kind: str, delta: float) -> np.ndarray:
    """Explicit 3x3 strain tensor.  ``xy`` uses the engineering convention.

    Voigt order is [xx, yy, zz, yz, xz, xy]; the engineering shear gamma_xy
    equals the tensor off-diagonal sum, so eps[0,1] = eps[1,0] = gamma/2.
    """
    eps = np.zeros((3, 3))
    if kind == "xx":
        eps[0, 0] = delta
    elif kind == "yy":
        eps[1, 1] = delta
    elif kind == "zz":
        eps[2, 2] = delta
    elif kind == "xy":
        eps[0, 1] = delta / 2.0
        eps[1, 0] = delta / 2.0
    else:
        msg = f"unknown strain kind {kind!r}"
        raise ValueError(msg)
    return eps


def spacegroup_label(atoms, symprec: float = CONV_SYMPREC) -> str | None:
    try:
        import spglib

        cell = (
            atoms.cell.array,
            atoms.get_scaled_positions(wrap=True),
            atoms.numbers,
        )
        return spglib.get_spacegroup(cell, symprec=symprec)
    except Exception:  # noqa: BLE001 - symmetry is diagnostic, never fatal
        return None


def stress_max_abs(voigt) -> float | None:
    return None if voigt is None else float(np.max(np.abs(voigt)))


def finite_energy(value: float) -> bool:
    return value is not None and np.isfinite(value)


# --------------------------------------------------------------------------
# engine access
# --------------------------------------------------------------------------


def single_point(ctx: engines.EngineContext, atoms) -> dict[str, Any]:
    """One guarded single point; returns energy/forces/stress/wall."""
    work = atoms.copy()
    work.calc = ctx.calculator

    def _calc():
        energy = float(work.get_potential_energy())
        forces = np.asarray(work.get_forces(), dtype=float)
        try:
            stress = np.asarray(work.get_stress(voigt=True), dtype=float)
        except Exception:  # noqa: BLE001 - stress is optional per profile
            stress = None
        return energy, forces, stress

    (energy, forces, stress), wall = common.timed(_calc, ctx.device)
    return {
        "energy_eV": energy,
        "forces_eV_A": forces,
        "stress_voigt": stress,
        "wall_seconds": wall,
    }


def relax_positions(atoms, fmax: float, steps: int, optimizer: str):
    """Fixed-cell position relaxation; returns (atoms, info)."""
    from ase.optimize import BFGS, FIRE, LBFGS  # noqa: PLC0415

    opt_cls = {"fire": FIRE, "lbfgs": LBFGS, "bfgs": BFGS}[optimizer.lower()]
    work = atoms.copy()
    work.calc = atoms.calc
    opt = opt_cls(work, logfile=None)
    t0 = time.monotonic()
    converged = opt.run(fmax=fmax, steps=steps)
    forces = work.get_forces() if work.calc is not None else None
    fmax_final = float(np.max(np.linalg.norm(forces, axis=1)))
    return work, {
        "optimizer": opt_cls.__name__,
        "converged": bool(converged),
        "steps": int(opt.get_number_of_steps()),
        "fmax_final": fmax_final,
        "wall_seconds": time.monotonic() - t0,
    }


def relax_cell_and_ions(atoms, fmax: float, steps: int, optimizer: str = "fire"):
    """Full cell + ions relaxation through ASE FrechetCellFilter."""
    from ase.filters import FrechetCellFilter  # noqa: PLC0415
    from ase.optimize import FIRE, LBFGS  # noqa: PLC0415

    opt_cls = {"fire": FIRE, "lbfgs": LBFGS}[optimizer.lower()]
    work = atoms.copy()
    work.calc = atoms.calc
    filt = FrechetCellFilter(work)
    opt = opt_cls(filt, logfile=None)
    t0 = time.monotonic()
    converged = opt.run(fmax=fmax, steps=steps)
    forces = work.get_forces()
    fmax_final = float(np.max(np.linalg.norm(forces, axis=1)))
    try:
        stress = np.asarray(work.get_stress(voigt=True), dtype=float)
    except Exception:  # noqa: BLE001
        stress = None
    return work, {
        "optimizer": opt_cls.__name__,
        "converged": bool(converged),
        "steps": int(opt.get_number_of_steps()),
        "fmax_final": fmax_final,
        "stress_voigt": stress,
        "wall_seconds": time.monotonic() - t0,
    }


def record(
    ctx: engines.EngineContext,
    args,
    case_id: str,
    test_id: str,
    status: str,
    input_atoms,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    wall: float = 0.0,
    exception: str | None = None,
) -> dict[str, Any]:
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
        input_structure_id=None
        if input_atoms is None
        else common.structure_id(input_atoms),
        parameters=parameters,
        metrics=metrics,
        diagnostics=dict(ctx.diagnostics),
        wall_seconds=wall,
        peak_vram=common.peak_vram_mib(),
        exception=exception,
    )
    rec["backend_version"] = ctx.backend_version
    rec["framework_version"] = ctx.framework_version
    common.write_result(rec, args.out, tag=args.tag)
    return rec


# --------------------------------------------------------------------------
# T4-A single points on equilibrium and displaced structures
# --------------------------------------------------------------------------


def workflow_t4a(ctx, args, out_dir) -> int:
    failures = 0
    for system in T4A_SYSTEMS:
        atoms, desc = (
            fixtures.build_fixture(system)
            if system in fixtures.FIXTURES
            else fixtures.build_extra(system)
        )
        for variant in T4A_VARIANTS:
            work = atoms.copy()
            if variant.startswith("disp_"):
                magnitude = float(variant.split("_")[1].rstrip("A"))
                work.positions = work.positions + seeded_displacements(
                    len(atoms), magnitude
                )
            elif variant == "compressed_0.98":
                work = apply_strain(work, "xx", -0.02)
                work = apply_strain(work, "yy", -0.02)
                work = apply_strain(work, "zz", -0.02)
            elif variant == "expanded_1.02":
                work = apply_strain(work, "xx", 0.02)
                work = apply_strain(work, "yy", 0.02)
                work = apply_strain(work, "zz", 0.02)
            try:
                sp = single_point(ctx, work)
                energy = sp["energy_eV"]
                forces = sp["forces_eV_A"]
                stress = sp["stress_voigt"]
                ok = finite_energy(energy) and bool(np.isfinite(forces).all())
                if stress is not None:
                    ok = ok and bool(np.isfinite(stress).all())
                status = "pass" if ok else "fail"
                if not ok:
                    failures += 1
                record(
                    ctx,
                    args,
                    case_id="t4_singlepoint",
                    test_id=f"t4a_{system}_{variant}",
                    status=status,
                    input_atoms=work,
                    parameters={
                        "fixture": desc,
                        "variant": variant,
                        "displacement_seed": SEED
                        if variant.startswith("disp_")
                        else None,
                    },
                    metrics={
                        "energy_per_atom_eV": energy / len(work),
                        "forces_rms_eV_A": float(np.sqrt(np.mean(forces**2))),
                        "forces_max_abs_eV_A": float(np.max(np.abs(forces))),
                        "stress_max_abs_eV_A3": stress_max_abs(stress),
                        "stress_voigt_eV_A3": None
                        if stress is None
                        else [float(v) for v in stress],
                        "natoms": len(work),
                    },
                    wall=sp["wall_seconds"],
                )
            except Exception as exc:  # noqa: BLE001 - record and continue
                failures += 1
                record(
                    ctx,
                    args,
                    case_id="t4_singlepoint",
                    test_id=f"t4a_{system}_{variant}",
                    status="fail",
                    input_atoms=work,
                    parameters={"fixture": desc, "variant": variant},
                    metrics={},
                    exception=f"{type(exc).__name__}: {exc}",
                )
    return failures


# --------------------------------------------------------------------------
# T4-B geometry relaxation
# --------------------------------------------------------------------------


def workflow_relax(ctx, args, out_dir) -> int:
    failures = 0
    for system in RELAX_SYSTEMS:
        atoms, desc = (
            fixtures.build_fixture(system)
            if system in fixtures.FIXTURES
            else fixtures.build_extra(system)
        )
        perturbed = atoms.copy()
        perturbed.calc = ctx.calculator
        perturbed.positions = perturbed.positions + seeded_displacements(
            len(atoms), 0.05
        )
        e_start = float(perturbed.get_potential_energy())
        start_pos = perturbed.get_positions()
        results: dict[str, dict[str, Any]] = {}
        for optimizer in ("fire", "lbfgs"):
            work = perturbed.copy()
            work.calc = ctx.calculator
            try:
                relaxed, info = relax_positions(
                    work, RELAX_FMAX, RELAX_MAX_STEPS, optimizer
                )
                e_final = float(relaxed.get_potential_energy())
                info.update(
                    {
                        "energy_start_eV": e_start,
                        "energy_final_eV": e_final,
                        "energy_drop_eV": e_start - e_final,
                        "max_atom_displacement_A": float(
                            np.max(
                                np.linalg.norm(
                                    relaxed.get_positions() - start_pos, axis=1
                                )
                            )
                        ),
                        "final_structure_id": common.structure_id(relaxed),
                    }
                )
                results[optimizer] = info
                record(
                    ctx,
                    args,
                    case_id="t4_relax",
                    test_id=f"t4b_{system}_{optimizer}",
                    status="pass" if info["converged"] else "fail",
                    input_atoms=perturbed,
                    parameters={
                        "fixture": desc,
                        "fmax": RELAX_FMAX,
                        "max_steps": RELAX_MAX_STEPS,
                    },
                    metrics=info,
                    wall=info["wall_seconds"],
                )
                if not info["converged"]:
                    failures += 1
            except Exception as exc:  # noqa: BLE001
                failures += 1
                record(
                    ctx,
                    args,
                    case_id="t4_relax",
                    test_id=f"t4b_{system}_{optimizer}",
                    status="fail",
                    input_atoms=perturbed,
                    parameters={"fixture": desc},
                    metrics={},
                    exception=f"{type(exc).__name__}: {exc}",
                )
        if "fire" in results and "lbfgs" in results:
            agree = abs(
                results["fire"]["energy_final_eV"] - results["lbfgs"]["energy_final_eV"]
            )
            both_conv = results["fire"]["converged"] and results["lbfgs"]["converged"]
            status = "pass" if both_conv and agree < 1e-3 else "fail"
            if status == "fail":
                failures += 1
            record(
                ctx,
                args,
                case_id="t4_relax",
                test_id=f"t4b_{system}_optimizer_agreement",
                status=status,
                input_atoms=perturbed,
                parameters={"fixture": desc, "tolerance_eV": 1e-3},
                metrics={
                    "fire_final_eV": results["fire"]["energy_final_eV"],
                    "lbfgs_final_eV": results["lbfgs"]["energy_final_eV"],
                    "energy_difference_eV": agree,
                    "both_converged": both_conv,
                },
            )
    return failures


def workflow_cellrelax(ctx, args, out_dir) -> int:
    failures = 0
    for system in RELAX_SYSTEMS:
        atoms, desc = (
            fixtures.build_fixture(system)
            if system in fixtures.FIXTURES
            else fixtures.build_extra(system)
        )
        try:
            start = atoms.copy()
            start.calc = ctx.calculator
            # start from a deliberately expanded cell to exercise the filter
            start.set_cell(start.cell.array * 1.03, scale_atoms=True)
            sym_before = spacegroup_label(start)
            relaxed, info = relax_cell_and_ions(
                start, RELAX_FMAX, RELAX_MAX_STEPS, "fire"
            )
            sym_after = spacegroup_label(relaxed)
            smax = stress_max_abs(info["stress_voigt"])
            status = (
                "pass"
                if info["converged"] and smax is not None and smax < 1e-3
                else ("fail" if not info["converged"] else "characterized")
            )
            if status == "fail":
                failures += 1
            record(
                ctx,
                args,
                case_id="t4_cellrelax",
                test_id=f"t4b_cell_{system}",
                status=status,
                input_atoms=start,
                parameters={
                    "fixture": desc,
                    "filter": "FrechetCellFilter",
                    "fmax": RELAX_FMAX,
                    "target_pressure_GPa": 0.0,
                    "stress_pass_threshold_eV_A3": 1e-3,
                },
                metrics={
                    **{k: v for k, v in info.items() if k != "stress_voigt"},
                    "initial_volume_A3": float(start.get_volume()),
                    "final_volume_A3": float(relaxed.get_volume()),
                    "final_cell": [list(map(float, row)) for row in relaxed.cell.array],
                    "stress_max_abs_eV_A3": smax,
                    "spacegroup_before": sym_before,
                    "spacegroup_after": sym_after,
                },
                wall=info["wall_seconds"],
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            record(
                ctx,
                args,
                case_id="t4_cellrelax",
                test_id=f"t4b_cell_{system}",
                status="fail",
                input_atoms=atoms,
                parameters={"fixture": desc},
                metrics={},
                exception=f"{type(exc).__name__}: {exc}",
            )
    return failures


# --------------------------------------------------------------------------
# T4-C equation of state
# --------------------------------------------------------------------------


def _fit_eos(volumes, energies, name: str) -> dict[str, Any]:
    """Fit E(V) with an analytic 4-parameter form via least squares.

    ASE 3.29 ``EquationOfState.fit()`` returns only (v0, e0, B0) and its
    polynomial 'sj' form has no B0', so both fits are done directly on the
    published analytic functions (Birch-Murnaghan and Pourier-Tarantola)
    with scipy least_squares.  B is in eV/A^3 (x 160.21766208 -> GPa).
    """
    import ase.eos as ase_eos  # noqa: PLC0415
    from scipy.optimize import least_squares  # noqa: PLC0415

    model = getattr(ase_eos, name)
    v = np.asarray(volumes, dtype=float)
    e = np.asarray(energies, dtype=float)

    # seed from a parabola around the minimum
    i_min = int(np.argmin(e))
    v0_seed = float(v[i_min])
    e0_seed = float(e[i_min])
    # parabolic B0 estimate: B = V d2E/dV2
    if len(v) > 2:
        d2 = float(np.polyfit(v - v[i_min], e - e[i_min], 2)[0]) * 2.0
        b0_seed = max(d2 * v0_seed, 1e-3)
    else:
        b0_seed = 0.5

    def residual(p):
        e0, b0, bp, v0 = p
        return model(v, e0, b0, bp, v0) - e

    sol = least_squares(
        residual,
        x0=np.array([e0_seed, b0_seed, 4.0, v0_seed]),
        bounds=(
            np.array([-np.inf, 1e-6, 0.0, 0.5 * v.min()]),
            np.array([np.inf, np.inf, 20.0, 2.0 * v.max()]),
        ),
    )
    e0, b0, bp, v0 = (float(x) for x in sol.x)
    residuals = residual(sol.x)
    return {
        "form": name,
        "v0_A3": v0,
        "e0_eV": e0,
        "b0_eV_A3": b0,
        "b0_GPa": b0 * common.EV_A3_TO_GPA,
        "b0_prime": bp,
        "max_abs_residual_eV": float(np.max(np.abs(residuals))),
        "residuals_eV": [float(r) for r in residuals],
        "minimum_inside_sampled_range": bool(v.min() < v0 < v.max()),
        "fit_converged": bool(sol.success),
    }

def workflow_eos(ctx, args, out_dir) -> int:
    failures = 0
    for system in EOS_SYSTEMS:
        atoms, desc = fixtures.build_fixture(system)
        atoms.calc = ctx.calculator
        try:
            relaxed, _info = relax_cell_and_ions(
                atoms, RELAX_FMAX, RELAX_MAX_STEPS, "fire"
            )
            v0 = float(relaxed.get_volume())
            volumes: list[float] = []
            energies: list[float] = []
            for ratio in EOS_RATIOS:
                s = ratio ** (1.0 / 3.0)
                work = relaxed.copy()
                work.calc = ctx.calculator
                work.set_cell(relaxed.cell.array * s, scale_atoms=True)
                sp = single_point(ctx, work)
                volumes.append(float(work.get_volume()))
                energies.append(sp["energy_eV"])
            fits = {
                name: _fit_eos(volumes, energies, name)
                for name in ("birchmurnaghan", "pouriertarantola")
            }
            good = all(
                f["fit_converged"]
                and f["max_abs_residual_eV"] / len(relaxed) < EOS_RESIDUAL_MAX_EV
                and f["minimum_inside_sampled_range"]
                and f["b0_prime"] > 0.0
                for f in fits.values()
            )
            status = "pass" if good else "fail"
            if status == "fail":
                failures += 1
            record(
                ctx,
                args,
                case_id="t4_eos",
                test_id=f"t4c_{system}",
                status=status,
                input_atoms=relaxed,
                parameters={
                    "fixture": desc,
                    "volume_ratios": list(EOS_RATIOS),
                    "scaling": "s = (V/V0)^(1/3), fractional coordinates fixed",
                    "residual_max_eV_per_atom": EOS_RESIDUAL_MAX_EV,
                },
                metrics={
                    "relaxed_volume_A3": v0,
                    "volumes_A3": volumes,
                    "energies_eV": energies,
                    "energy_per_atom_eV": [e / len(relaxed) for e in energies],
                    "fits": fits,
                    "fit_to_fit_b0_spread_GPa": abs(
                        fits["birchmurnaghan"]["b0_GPa"]
                        - fits["pouriertarantola"]["b0_GPa"]
                    ),
                    "b0_reportable": good,
                },
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            record(
                ctx,
                args,
                case_id="t4_eos",
                test_id=f"t4c_{system}",
                status="fail",
                input_atoms=atoms,
                parameters={"fixture": desc},
                metrics={},
                exception=f"{type(exc).__name__}: {exc}",
            )
    return failures


# --------------------------------------------------------------------------
# T4-D elastic constants
# --------------------------------------------------------------------------


def _linear_fit(x: list[float], y: list[float]) -> dict[str, float]:
    slope, intercept = np.polyfit(np.array(x), np.array(y), 1)
    residuals = np.array(y) - (slope * np.array(x) + intercept)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


def _cubic_elastic(ctx, atoms, relaxed_ions: bool) -> dict[str, Any]:
    """Cubic clamped/relaxed-ion constants from symmetric +-strains."""

    equilibrium = single_point(ctx, atoms)
    e0 = equilibrium["energy_eV"]
    volume = float(atoms.get_volume())

    samples: dict[str, list[tuple[float, np.ndarray]]] = {
        "xx": [],
        "yy": [],
        "xy": [],
    }
    energy_curve: dict[str, list[tuple[float, float]]] = {"xx": [], "xy": []}
    for kind in samples:
        for delta in ELASTIC_STRAINS:
            for sign in (+1.0, -1.0):
                work = apply_strain(atoms, kind, sign * delta)
                work.calc = ctx.calculator
                if relaxed_ions:
                    # relax_positions returns a relaxed COPY; it must be
                    # propagated or the "relaxed-ion" constants silently
                    # degenerate to the clamped-ion values (regression:
                    # test_elastic_relaxed_ions_differ_from_clamped).
                    work, _ = relax_positions(
                        work, INTERNAL_RELAX_FMAX, 200, "lbfgs"
                    )
                sp = single_point(ctx, work)
                stress = sp["stress_voigt"]
                if stress is None:
                    msg = f"{ctx.engine} profile does not expose stress; "
                    "elastic constants require it"
                    raise RuntimeError(msg)
                samples[kind].append((sign * delta, stress))
                if kind in energy_curve:
                    energy_curve[kind].append((sign * delta, sp["energy_eV"]))

    def voigt_component(stress: np.ndarray, index: int) -> float:
        return float(stress[index])

    fits = {
        "C11": _linear_fit(
            [d for d, _ in samples["xx"]],
            [voigt_component(s, 0) for _, s in samples["xx"]],
        ),
        "C12_from_yy": _linear_fit(
            [d for d, _ in samples["yy"]],
            [voigt_component(s, 0) for _, s in samples["yy"]],
        ),
        "C12_from_xx": _linear_fit(
            [d for d, _ in samples["xx"]],
            [voigt_component(s, 1) for _, s in samples["xx"]],
        ),
        "C44": _linear_fit(
            [d for d, _ in samples["xy"]],
            [voigt_component(s, 5) for _, s in samples["xy"]],
        ),
    }
    # energy-curvature cross-checks: E(d) = E0 + V C d^2 / 2
    curvature = {}
    for kind, key in (("xx", "C11_energy"), ("xy", "C44_energy")):
        arr = np.array(energy_curve[kind])
        quad = np.polyfit(arr[:, 0], arr[:, 1] - e0, 2)
        curvature[key] = float(2.0 * quad[0] / volume)

    c11 = fits["C11"]["slope"]
    c12 = 0.5 * (fits["C12_from_yy"]["slope"] + fits["C12_from_xx"]["slope"])
    c44 = fits["C44"]["slope"]
    max_res = max(
        fits["C11"]["max_abs_residual"],
        fits["C12_from_yy"]["max_abs_residual"],
        fits["C12_from_xx"]["max_abs_residual"],
        fits["C44"]["max_abs_residual"],
    )
    born = {
        "C11_minus_C12_positive": bool(c11 - c12 > 0),
        "C11_plus_2C12_positive": bool(c11 + 2 * c12 > 0),
        "C44_positive": bool(c44 > 0),
    }
    return {
        "fit_mode": "relaxed-ion" if relaxed_ions else "clamped-ion",
        "strains": list(ELASTIC_STRAINS),
        "C11_eV_A3": c11,
        "C12_eV_A3": c12,
        "C44_eV_A3": c44,
        "B_GPa": (c11 + 2 * c12) / 3.0 * common.EV_A3_TO_GPA,
        "C11_GPa": c11 * common.EV_A3_TO_GPA,
        "C12_GPa": c12 * common.EV_A3_TO_GPA,
        "C44_GPa": c44 * common.EV_A3_TO_GPA,
        "fits": fits,
        "energy_curvature": curvature,
        "born_stability": born,
        "max_fit_residual_eV_A3": max_res,
        "stress_intercept_max_eV_A3": max(abs(f["intercept"]) for f in fits.values()),
    }


def workflow_elastic(ctx, args, out_dir) -> int:
    failures = 0
    for system in ELASTIC_SYSTEMS:
        atoms, desc = fixtures.build_fixture(system)
        try:
            result = _cubic_elastic(ctx, atoms, relaxed_ions=False)
            fit_ok = result["max_fit_residual_eV_A3"] < ELASTIC_RESIDUAL_MAX
            born_ok = all(result["born_stability"].values())
            status = (
                "pass"
                if fit_ok and born_ok
                else ("characterized" if fit_ok else "fail")
            )
            if status == "fail":
                failures += 1
            record(
                ctx,
                args,
                case_id="t4_elastic",
                test_id=f"t4d_{system}_clamped",
                status=status,
                input_atoms=atoms,
                parameters={
                    "fixture": desc,
                    "strain_kinds": ["xx", "yy", "xy"],
                    "shear_convention": "engineering gamma_xy, eps_xy = gamma/2",
                    "residual_max_eV_A3": ELASTIC_RESIDUAL_MAX,
                },
                metrics=result,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            record(
                ctx,
                args,
                case_id="t4_elastic",
                test_id=f"t4d_{system}_clamped",
                status="fail",
                input_atoms=atoms,
                parameters={"fixture": desc},
                metrics={},
                exception=f"{type(exc).__name__}: {exc}",
            )
        try:
            result = _cubic_elastic(ctx, atoms, relaxed_ions=True)
            fit_ok = result["max_fit_residual_eV_A3"] < ELASTIC_RESIDUAL_MAX
            born_ok = all(result["born_stability"].values())
            status = (
                "pass"
                if fit_ok and born_ok
                else ("characterized" if fit_ok else "fail")
            )
            if status == "fail":
                failures += 1
            record(
                ctx,
                args,
                case_id="t4_elastic",
                test_id=f"t4d_{system}_relaxed",
                status=status,
                input_atoms=atoms,
                parameters={
                    "fixture": desc,
                    "internal_relax_fmax": INTERNAL_RELAX_FMAX,
                    "shear_convention": "engineering gamma_xy, eps_xy = gamma/2",
                    "residual_max_eV_A3": ELASTIC_RESIDUAL_MAX,
                },
                metrics=result,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            record(
                ctx,
                args,
                case_id="t4_elastic",
                test_id=f"t4d_{system}_relaxed",
                status="fail",
                input_atoms=atoms,
                parameters={"fixture": desc},
                metrics={},
                exception=f"{type(exc).__name__}: {exc}",
            )
    return failures


# --------------------------------------------------------------------------
# T4-E phonons + T4-F harmonic thermodynamics
# --------------------------------------------------------------------------


def _thermo_from_modes(
    freqs_ev: np.ndarray,
    temperatures,
    n_qpoints: int | None = None,
) -> dict[str, Any]:
    """Harmonic thermal sums over a q-grid (phonopy-equivalent formulation).

    ``freqs_ev`` is the (n_qpoints, n_branches) grid of mode energies
    (any layout; it is ravelled).  Following the phonopy mesh
    formulation the thermal sums are **q-averaged**: every sum is
    divided by ``n_qpoints`` so the result is per unit cell of the
    phonon calculation, independent of grid density.  Callers passing a
    plain mode list (no grid) may omit ``n_qpoints`` (defaults to 1).

    Every mode contributes the standard harmonic expression.  Three
    documented tolerances apply:

    - SUM_EXCLUSION_TOL_EV (1e-6 eV): modes with |omega| below this are
      excluded from the entropy/free-energy/Cv sums.  The log terms
      diverge as omega -> 0, so Gamma acoustic modes (even grid has no
      Gamma; raw-FC Gamma modes live at ~1e-8 eV) and float-noise modes
      must not enter the sums.  Physically these are zero-measure.
    - IMAGINARY_TOL_EV (1e-4 eV): modes below -IMAGINARY_TOL_EV are
      robust imaginary modes and gate the workflow status.
    - 1e-9 eV: counting threshold for exactly-zero modes (informational).
    """
    w = np.asarray(freqs_ev, dtype=float).ravel()
    total_modes = w.size
    n_q = int(n_qpoints) if n_qpoints else 1
    if total_modes % n_q != 0:
        raise ValueError(
            f"mode count {total_modes} not divisible by n_qpoints {n_q}"
        )
    tiny = np.abs(w) < 1e-9
    robust_imag = w < -IMAGINARY_TOL_EV
    summed = w > SUM_EXCLUSION_TOL_EV
    zpe = 0.5 * float(np.sum(np.abs(w))) / n_q
    out: dict[str, Any] = {
        "n_modes": int(total_modes),
        "n_exactly_zero_modes": int(np.sum(tiny)),
        "n_negative_modes": int(np.sum(w < 0.0)),
        "n_robust_imaginary_modes": int(np.sum(robust_imag)),
        "min_frequency_eV": float(np.min(w)),
        "zero_point_energy_eV": zpe,
        "per_temperature": [],
    }
    for temp in temperatures:
        x = w[summed] / (KB_EV_PER_K * temp)
        efac = np.exp(-x)
        denom = 1.0 - efac
        cv_per_mode = x * x * efac / (denom * denom)
        s_per_mode = x * efac / denom - np.log(denom)
        f_per_mode = KB_EV_PER_K * temp * np.log(2.0 * np.sinh(x / 2.0))
        cv = KB_EV_PER_K * float(np.sum(cv_per_mode)) / n_q
        s = KB_EV_PER_K * float(np.sum(s_per_mode)) / n_q
        f = float(np.sum(f_per_mode)) / n_q
        out["per_temperature"].append(
            {
                "T_K": temp,
                "cv_eV_per_K": cv,
                "cv_kB_per_cell": cv / KB_EV_PER_K,
                "entropy_eV_per_K": s,
                "free_energy_eV": zpe + f,
            }
        )
    return out


def workflow_phonon(ctx, args, out_dir) -> int:
    """Phonon + harmonic thermodynamics workflows (they share the FC run)."""
    from ase.dft.kpoints import monkhorst_pack  # noqa: PLC0415
    from ase.phonons import Phonons  # noqa: PLC0415

    failures = 0
    for system in PHONON_SYSTEMS:
        if system == "cu_fcc_prim":
            from ase.build import bulk  # noqa: PLC0415

            atoms = bulk("Cu", "fcc", a=3.615, cubic=False)
            desc = {
                "system": system,
                "natoms_prim": len(atoms),
                "source": "fixture cu_fcc primitive",
            }
        else:
            atoms, desc = fixtures.build_extra("si_diamond_prim")
            desc = dict(desc)
            desc["system"] = system
        masses = atoms.get_masses()
        for sc in PHONON_SUPERCELLS:
            for delta in PHONON_DELTAS:
                sc_label = f"{sc[0]}x{sc[1]}x{sc[2]}"
                test_id = f"t4e_{system}_{sc_label}_d{delta:.3f}"
                cache_name = str(
                    out_dir
                    / "cache"
                    / "phonon"
                    / ctx.engine
                    / ctx.profile_id
                    / f"{system}_{sc_label}_d{delta:.3f}"
                )
                try:
                    ph = Phonons(
                        atoms.copy(),
                        ctx.calculator,
                        supercell=sc,
                        delta=delta,
                        name=cache_name,
                    )
                    ph.run()

                    # raw force constants (no acoustic sum rule)
                    ph.read(method="Frederiksen", symmetrize=3, acoustic=False)
                    c_raw = ph.get_force_constant().copy()
                    gamma_raw = np.asarray(
                        ph.band_structure(np.array([[0.0, 0.0, 0.0]])), dtype=float
                    ).ravel()

                    # acoustic-sum-rule treated
                    ph.read(method="Frederiksen", symmetrize=3, acoustic=True)
                    c_asr = ph.get_force_constant().copy()
                    gamma_asr = np.asarray(
                        ph.band_structure(np.array([[0.0, 0.0, 0.0]])), dtype=float
                    ).ravel()

                    asr_row_error = float(np.max(np.abs(np.sum(c_asr, axis=1))))
                    acoustic_max = float(np.max(np.abs(np.sort(gamma_asr)[:3])))
                    robust_imaginary = bool(gamma_asr.min() < -IMAGINARY_TOL_EV)

                    # standard fcc high-symmetry path in primitive reciprocal coords
                    path = np.array(
                        [
                            [0.0, 0.0, 0.0],  # Gamma
                            [0.5, 0.0, 0.0],  # X
                            [0.5, 0.25, 0.5],  # W
                            [0.5, 0.5, 0.5],  # L
                            [0.0, 0.0, 0.0],  # Gamma
                            [0.375, 0.375, 0.75],  # K
                            [0.5, 0.0, 0.0],  # X
                        ]
                    )
                    interp = [
                        path[i] * (1 - t) + path[i + 1] * t
                        for i in range(len(path) - 1)
                        for t in np.linspace(
                            0,
                            1,
                            20 if i not in (0, 3) else 15,
                            endpoint=(i == len(path) - 2),
                        )
                    ]
                    interp = np.array(interp)
                    omega_path = np.asarray(ph.band_structure(interp), dtype=float)

                    status = (
                        "fail"
                        if robust_imaginary
                        else (
                            "pass"
                            if acoustic_max < ACOUSTIC_TOL_EV
                            else "characterized"
                        )
                    )
                    if status == "fail":
                        failures += 1
                    record(
                        ctx,
                        args,
                        case_id="t4_phonon",
                        test_id=test_id,
                        status=status,
                        input_atoms=atoms,
                        parameters={
                            "fixture": desc,
                            "supercell": list(sc),
                            "delta_A": delta,
                            "method": "ASE finite displacement (Frederiksen, symmetrize=3)",
                            "asr": "ase Phonons read(acoustic=True)",
                            "imaginary_tolerance_eV": IMAGINARY_TOL_EV,
                            "acoustic_tolerance_eV": ACOUSTIC_TOL_EV,
                            "masses_amu": [float(m) for m in masses],
                        },
                        metrics={
                            "gamma_frequencies_eV_raw": [
                                float(v) for v in np.sort(gamma_raw)
                            ],
                            "gamma_frequencies_eV_asr": [
                                float(v) for v in np.sort(gamma_asr)
                            ],
                            "gamma_frequencies_cm1_asr": [
                                float(v) * EV_TO_CM1 for v in np.sort(gamma_asr)
                            ],
                            "min_frequency_eV_asr": float(np.min(gamma_asr)),
                            "acoustic_gamma_max_abs_eV": acoustic_max,
                            "asr_max_row_residual_eV_A2": asr_row_error,
                            "robust_imaginary_mode": robust_imaginary,
                            "dispersion_path_points": len(interp),
                            "dispersion_min_frequency_eV": float(np.min(omega_path)),
                            "dispersion_max_frequency_eV": float(np.max(omega_path)),
                            "dispersion_path_kpts": [
                                list(map(float, k)) for k in interp[:6]
                            ],
                            "dispersion_omegas_sampled_eV": [
                                float(v)
                                for v in omega_path[
                                    :, : min(6, omega_path.shape[1])
                                ].ravel()[:96]
                            ],
                            "force_constants_shape": list(c_asr.shape),
                        },
                    )

                    # T4-F harmonic thermodynamics from the q-grid (same FC)
                    grid = monkhorst_pack(THERMO_DOS_GRID)
                    omega_grid = np.asarray(ph.band_structure(grid), dtype=float)
                    thermo = _thermo_from_modes(
                        omega_grid,
                        THERMO_TEMPERATURES,
                        n_qpoints=len(grid),
                    )
                    thermo_status = (
                        "fail"
                        if thermo["n_robust_imaginary_modes"] > 0
                        or thermo["n_exactly_zero_modes"] > 3
                        else "pass"
                    )
                    if thermo_status == "fail":
                        failures += 1
                    record(
                        ctx,
                        args,
                        case_id="t4_thermo",
                        test_id=f"t4f_{system}_{sc_label}_d{delta:.3f}",
                        status=thermo_status,
                        input_atoms=atoms,
                        parameters={
                            "fixture": desc,
                            "supercell": list(sc),
                            "delta_A": delta,
                            "q_grid": list(THERMO_DOS_GRID),
                            "formulation": "q-averaged harmonic sums over MP q-grid (1/N_q normalized, per phonon unit cell)",
                            "zero_mode_policy": (
                                "modes below SUM_EXCLUSION_TOL_EV=1e-6 eV excluded ",
                                "from F/S/Cv sums (log divergence at omega->0); ",
                                "ZPE uses |omega| over all modes",
                            ),
                            "robust_imaginary_tolerance_eV": IMAGINARY_TOL_EV,
                            "temperatures_K": list(THERMO_TEMPERATURES),
                            "label": "harmonic (MLIP PES), not DFT thermochemistry",
                        },
                        metrics=thermo,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    record(
                        ctx,
                        args,
                        case_id="t4_phonon",
                        test_id=test_id,
                        status="fail",
                        input_atoms=atoms,
                        parameters={
                            "fixture": desc,
                            "supercell": list(sc),
                            "delta_A": delta,
                        },
                        metrics={},
                        exception=f"{type(exc).__name__}: {exc}",
                    )
    return failures


# --------------------------------------------------------------------------
# T4-G vacancy
# --------------------------------------------------------------------------


def workflow_vacancy(ctx, args, out_dir) -> int:
    from ase.build import bulk  # noqa: PLC0415

    failures = 0
    base = bulk("Cu", "fcc", a=3.615, cubic=True)
    for sc in VACANCY_SUPERCELLS:
        label = f"{sc[0]}x{sc[1]}x{sc[2]}"
        try:
            bulk_cell = base.repeat(sc)
            n_sites = len(bulk_cell)
            bulk_work = bulk_cell.copy()
            bulk_work.calc = ctx.calculator
            e_bulk = float(bulk_work.get_potential_energy())

            defect = bulk_cell.copy()
            del defect[0]
            defect_work = defect.copy()
            defect_work.calc = ctx.calculator
            e_def_unrelaxed = float(defect_work.get_potential_energy())
            ev_unrelaxed = e_def_unrelaxed - (n_sites - 1) / n_sites * e_bulk

            defect_rel = defect.copy()
            defect_rel.calc = ctx.calculator
            relaxed, info = relax_positions(
                defect_rel, VACANCY_RELAX_FMAX, 300, "lbfgs"
            )
            e_def_relaxed = float(relaxed.get_potential_energy())
            ev_relaxed = e_def_relaxed - (n_sites - 1) / n_sites * e_bulk

            converged = info["converged"]
            record(
                ctx,
                args,
                case_id="t4_vacancy",
                test_id=f"t4g_cu_{label}",
                status="pass" if converged else "fail",
                input_atoms=bulk_cell,
                parameters={
                    "host": "Cu fcc conventional cell",
                    "supercell": list(sc),
                    "n_sites": n_sites,
                    "formula": "E_vac = E(N-1) - (N-1)/N * E(N)",
                    "reference_policy": "same model identity for both terms",
                    "relax": "fixed cell, all ions, LBFGS",
                },
                metrics={
                    "e_bulk_eV": e_bulk,
                    "e_defect_unrelaxed_eV": e_def_unrelaxed,
                    "e_defect_relaxed_eV": e_def_relaxed,
                    "evac_unrelaxed_eV": ev_unrelaxed,
                    "evac_relaxed_eV": ev_relaxed,
                    "evac_unrelaxed_per_vacancy_eV": ev_unrelaxed,
                    "relax": info,
                },
            )
            if not converged:
                failures += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            record(
                ctx,
                args,
                case_id="t4_vacancy",
                test_id=f"t4g_cu_{label}",
                status="fail",
                input_atoms=base,
                parameters={"supercell": list(sc)},
                metrics={},
                exception=f"{type(exc).__name__}: {exc}",
            )
    return failures


# --------------------------------------------------------------------------
# T4-H surface energy
# --------------------------------------------------------------------------


def workflow_surface(ctx, args, out_dir) -> int:
    from ase.build import bulk, fcc111  # noqa: PLC0415

    failures = 0
    prim = bulk("Cu", "fcc", a=3.615, cubic=False)
    prim.calc = ctx.calculator
    e_per_atom = float(prim.get_potential_energy())
    for layers in SURFACE_LAYERS:
        for vacuum in SURFACE_VACUUMS:
            test_id = f"t4h_cu111_{layers}L_{vacuum:.0f}A"
            try:
                slab = fcc111("Cu", size=(2, 2, layers), a=3.615, vacuum=vacuum)
                area = float(abs(np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))))
                n_atoms = len(slab)

                slab_work = slab.copy()
                slab_work.calc = ctx.calculator
                e_unrelaxed = float(slab_work.get_potential_energy())
                gamma_unrelaxed = (
                    (e_unrelaxed - n_atoms * e_per_atom) / (2.0 * area) * 16.021766208
                )  # eV/A^2 -> J/m^2 (1 eV/A^2 = 16.021766208 J/m^2)

                slab_rel = slab.copy()
                slab_rel.calc = ctx.calculator
                relaxed, info = relax_positions(
                    slab_rel, SURFACE_RELAX_FMAX, 300, "lbfgs"
                )
                e_relaxed = float(relaxed.get_potential_energy())
                gamma_relaxed = (
                    (e_relaxed - n_atoms * e_per_atom) / (2.0 * area) * 16.021766208
                )

                record(
                    ctx,
                    args,
                    case_id="t4_surface",
                    test_id=test_id,
                    status="pass" if info["converged"] else "fail",
                    input_atoms=slab,
                    parameters={
                        "host": "Cu(111) symmetric two-sided slab",
                        "slab_size": [2, 2, layers],
                        "vacuum_A": vacuum,
                        "natoms": n_atoms,
                        "formula": "gamma = (E_slab - N E_bulk_atom) / (2A)",
                        "reference_policy": "same model, prim bulk per-atom energy",
                        "relax_policy": "fixed cell, all ions relaxed (symmetric slab)",
                        "unit_conversion": "1 eV/A^2 = 16.021766208 J/m^2",
                    },
                    metrics={
                        "area_A2": area,
                        "e_slab_unrelaxed_eV": e_unrelaxed,
                        "e_slab_relaxed_eV": e_relaxed,
                        "e_bulk_per_atom_eV": e_per_atom,
                        "gamma_unrelaxed_J_m2": gamma_unrelaxed,
                        "gamma_relaxed_J_m2": gamma_relaxed,
                        "relax": info,
                    },
                )
                if not info["converged"]:
                    failures += 1
            except Exception as exc:  # noqa: BLE001
                failures += 1
                record(
                    ctx,
                    args,
                    case_id="t4_surface",
                    test_id=test_id,
                    status="fail",
                    input_atoms=None,
                    parameters={"slab_size": [2, 2, layers], "vacuum_A": vacuum},
                    metrics={},
                    exception=f"{type(exc).__name__}: {exc}",
                )
    return failures


# --------------------------------------------------------------------------
# T4-I conditional cohesive / formation energetics
# --------------------------------------------------------------------------


def workflow_energetics(ctx, args, out_dir) -> int:
    """Conditional logic per taskbook section 24: without pinned isolated-atom
    references for every engine and pinned OMat compatibility corrections,
    these workflows must be recorded as unsupported rather than fabricated."""
    reason = (
        "unsupported_for_common_comparison: a four-engine comparison requires "
        "a common absolute reference level. DPA/GRACE do not expose documented "
        "isolated-atom references (E0) and no OMat compatibility-correction "
        "table is pinned in the manifest, so cohesive/formation energies "
        "cannot be reconstructed without ad-hoc references (taskbook 24)."
    )
    for kind in ("cohesive", "formation"):
        record(
            ctx,
            args,
            case_id="t4_energetics",
            test_id=f"t4i_{kind}",
            status="unsupported",
            input_atoms=None,
            parameters={
                "workflow": kind,
                "reason": reason,
                "would_require": (
                    "pinned per-element isolated-atom energies for every engine "
                    "plus, for formation energies, the official OMat "
                    "compatibility/correction scheme"
                ),
            },
            metrics={"support": "unsupported_for_common_comparison"},
        )
    return 0


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


WORKFLOW_FUNCS = {
    "t4a": workflow_t4a,
    "relax": workflow_relax,
    "cellrelax": workflow_cellrelax,
    "eos": workflow_eos,
    "elastic": workflow_elastic,
    "phonon": workflow_phonon,  # emits both t4_phonon and t4_thermo records
    "thermo": workflow_phonon,
    "vacancy": workflow_vacancy,
    "surface": workflow_surface,
    "energetics": workflow_energetics,
}


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
        "--workflows",
        default=",".join(WORKFLOW_FUNCS),
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
            case_id="t4_setup",
            test_id="t4_static_suite_setup",
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

    # deduplicate while preserving order (phonon/thermo share one pass)
    seen: dict[Any, str] = {}
    ordered: list[str] = []
    for wf in requested:
        func = WORKFLOW_FUNCS[wf]
        if func not in seen:
            seen[func] = wf
            ordered.append(wf)

    for wf in ordered:
        failures += WORKFLOW_FUNCS[wf](ctx, args, out_dir)

    print(f"static_suite: engine={ctx.engine} workflows={ordered} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
