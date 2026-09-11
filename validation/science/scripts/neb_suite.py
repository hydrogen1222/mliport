"""T5 NEB / CI-NEB validation suite (taskbook sections 25-26).

Runs the fixed-cell NEB / CI-NEB product workflow on two documented paths:

- ``cu_vacancy``    31-atom Cu fcc supercell (2x2x2 conventional) with one
                    vacancy and a nearest-neighbour hop into it.  Endpoints
                    are symmetry-related, fixed composition, no chemical
                    potential term, small enough for a full FD Hessian.
- ``na3ps4_vacancy`` alpha-Na3PS4 (pinned mp-28782 geometry fixture) with one
                    Na vacancy and the nearest Na hop into it.  This is a
                    domain/transferability characterisation: the four model
                    barriers are NOT independent DFT references (taskbook
                    section 25.2).

For every backend the suite runs the public product pipeline only
(``resolve_config`` -> ``run_neb_workflow``): endpoint relaxation, explicit
mapping/winding, linear-vs-IDPP initialisation, ordinary NEB warm-up,
CI-NEB at 5/7(/9) total images, checkpoint/resume identity, and the section
26 FD-Hessian saddle check on the climbing image.  No synthetic calculator
may appear here (taskbook section 47).

Status semantics (consistent with static_suite):

- ``pass``          workflow converged / criteria met
- ``characterized`` workflow completed; the observable characterises the
                    model (e.g. domain barriers without a reference) or the
                    criterion describes the saddle candidate itself
- ``fail``          non-convergence within the bounded budget, broken
                    mapping, or a bad saddle candidate (>= 2 robust
                    negative Hessian modes)
"""

from __future__ import annotations

import argparse
import itertools
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
from static_suite import record, relax_positions  # noqa: E402

# --------------------------------------------------------------------------
# budgets and documented thresholds
# --------------------------------------------------------------------------

CI_FMAX = 0.03  # eV/A, taskbook section 25.1 documented final target
NEB_PRE_FMAX = 0.10  # eV/A, product two-stage warm-up target
ENDPOINT_FMAX = 0.02  # eV/A, product endpoint validation target
NEB_MAX_STEPS = 800  # bounded iteration budget (taskbook section 25.1)
NEB_PRE_MAX_STEPS = 300
ENDPOINT_STEPS = 500
IDPP_STEPS = 200
#: total images per CI-NEB run: n_intermediate_images + 2 endpoints
IMAGE_COUNTS = (3, 5, 7)  # -> 5, 7, 9 total images
IMAGE_STABILIZATION_EV = 0.05  # 5->7 barrier delta below this = stabilized
SADDLE_DELTAS = (0.005, 0.010, 0.020)  # Angstrom, taskbook section 26
#: FD-Hessian eigenvalue scale below which a mode is treated as a numerical
#: or physical zero mode (translations of the periodic cell).  Section 26
#: forbids a universal imaginary-frequency threshold, so robustness is
#: judged as: the negative eigenvalue must exceed the zero-mode scale by
#: 10x AND persist at every displacement amplitude.
ZERO_SCALE_EV_A2 = 1e-3
NEG_ROBUST_EV_A2 = -10.0 * ZERO_SCALE_EV_A2
NEG_STABILITY_FRACTION = 0.5  # spread/|mean| bound across deltas
TANGENT_OVERLAP_MIN = 0.8  # |<v_negative, tangent>| for a first-order TS
RESUME_PRE_STEPS = 15
RESUME_MAIN_STEPS = 50
RESUME_CHECKPOINT_INTERVAL = 10

_CU_SUPERCELL = (2, 2, 2)
_NA3PS4 = "na3ps4"

#: per-process cross-workflow state (relaxed endpoints, CI-NEB reference
#: runs).  Keyed by (engine, profile_id); never persisted to git.
_STATE: dict[tuple[str, str], dict[str, Any]] = {}


# --------------------------------------------------------------------------
# geometry builders (deterministic, explicitly documented mapping)
# --------------------------------------------------------------------------


def _nearest_atom(atoms, index: int, symbols: str | None = None) -> int:
    """Index of the closest (MIC) neighbour of ``index``, same symbol filter."""
    from ase.geometry import find_mic

    dists = []
    for j in range(len(atoms)):
        if j == index:
            dists.append(np.inf)
            continue
        if symbols is not None and atoms.symbols[j] != symbols:
            dists.append(np.inf)
            continue
        vec, _ = find_mic(
            (atoms.positions[j] - atoms.positions[index])[None, :],
            atoms.cell,
            pbc=True,
        )
        dists.append(float(np.linalg.norm(vec[0])))
    return int(np.argmin(dists))


def build_vacancy_pair(base, *, migrating_symbol: str) -> dict[str, Any]:
    """Build the vacancy-migration endpoint pair from a perfect supercell.

    The initial endpoint is ``base`` with atom 0 (which must sit on the
    migrating sublattice) removed.  The final endpoint is the *same* 31/15
    atom structure with the nearest same-sublattice neighbour moved into the
    vacancy site.  The atom mapping is the identity permutation (each atom
    maps onto itself; the hop is expressed purely in coordinates) and the
    image shifts are all zero because the hop vector is shorter than half
    the shortest cell diagonal (asserted below).  Both facts are returned
    for the record instead of being silently assumed.
    """
    assert base.symbols[0] == migrating_symbol
    hop_index = _nearest_atom(base, 0, symbols=migrating_symbol)
    hop_vec = base.positions[hop_index] - base.positions[0]
    from ase.geometry import find_mic

    hop_mic, _ = find_mic(hop_vec[None, :], base.cell, pbc=True)
    hop_distance = float(np.linalg.norm(hop_mic[0]))
    # explicit winding check: the hop must not wrap a periodic boundary
    # explicit winding check: the raw hop vector must not have a materially
    # shorter periodic image.  Residual mic/raw differences at the half-cell
    # boundary are coordinate dust from the pinned OPTIMADE positions; the
    # product layer records the explicit per-atom image shifts either way.
    hop_mic_delta_A = float(np.linalg.norm(hop_mic[0]) - np.linalg.norm(hop_vec))
    assert abs(hop_mic_delta_A) < 1e-3, (
        "hop vector has a materially shorter periodic image; the documented "
        "simple path must be reviewed"
    )

    initial = base.copy()
    del initial[0]
    final = initial.copy()
    final.positions[hop_index - 1] = base.positions[0]
    return {
        "initial": initial,
        "final": final,
        "vacancy_site": np.asarray(base.positions[0], dtype=float).tolist(),
        "hop_atom_index_in_endpoint": int(hop_index - 1),
        "hop_distance_A": hop_distance,
        "hop_mic_delta_A": hop_mic_delta_A,
        "atom_map": list(range(len(initial))),
        "image_shifts": [[0, 0, 0]] * len(initial),
        "natoms": len(initial),
    }


def cu_vacancy_pair() -> dict[str, Any]:
    """31-atom Cu vacancy hop (2x2x2 conventional fcc, a = 3.615 A)."""
    atoms, desc = fixtures.build_fixture("cu_fcc")
    base = atoms.repeat(_CU_SUPERCELL)
    pair = build_vacancy_pair(base, migrating_symbol="Cu")
    pair["desc"] = {
        "system": "cu_vacancy",
        "fixture": desc,
        "supercell": list(_CU_SUPERCELL),
        "a_A": 3.615,
    }
    return pair


def na3ps4_vacancy_pair() -> dict[str, Any]:
    """15-atom alpha-Na3PS4 Na-vacancy hop on the pinned mp-28782 fixture.

    Geometry fixture only (taskbook section 15); the barrier is a model
    transferability observable, never compared with Materials Project
    energies.
    """
    atoms, desc = fixtures.build_extra(_NA3PS4)
    pair = build_vacancy_pair(atoms, migrating_symbol="Na")
    pair["desc"] = {
        "system": "na3ps4_vacancy",
        "fixture": desc,
        "supercell": [1, 1, 1],
        "note": "pinned mp-28782 geometry; geometry fixture only",
    }
    return pair


# --------------------------------------------------------------------------
# shared per-engine state
# --------------------------------------------------------------------------


def _repeat_floor(ctx, atoms) -> dict[str, float]:
    """Measured in-process repeatability floor for the symmetry gate."""
    work = atoms.copy()
    work.calc = ctx.calculator
    energies = [float(work.get_potential_energy()) for _ in range(3)]
    forces = [np.asarray(work.get_forces(), dtype=float) for _ in range(3)]
    return {
        "energy_floor_eV": float(max(abs(energies[i] - energies[0]) for i in (1, 2))),
        "force_floor_eV_A": float(
            max(np.abs(forces[i] - forces[0]).max() for i in (1, 2))
        ),
    }


def _ensure_endpoints(ctx, args, out_dir) -> dict[str, Any]:
    """Relax both endpoints once per engine and cache them for the suite."""
    key = (ctx.engine, ctx.profile_id)
    state = _STATE.get(key)
    if state is not None and "endpoints" in state:
        return state["endpoints"]

    pair = cu_vacancy_pair()
    init = pair["initial"].copy()
    init.calc = ctx.calculator
    fin = pair["final"].copy()
    fin.calc = ctx.calculator
    relaxed_init, info_i = relax_positions(init, ENDPOINT_FMAX, ENDPOINT_STEPS, "fire")
    relaxed_fin, info_f = relax_positions(fin, ENDPOINT_FMAX, ENDPOINT_STEPS, "fire")
    e_i = float(relaxed_init.get_potential_energy())
    e_f = float(relaxed_fin.get_potential_energy())
    floor = _repeat_floor(ctx, relaxed_init)
    # symmetry-related endpoints: identical minima up to translation
    sym_tol = max(10.0 * floor["energy_floor_eV"], 1e-3)
    entry = {
        "pair": pair,
        "initial": relaxed_init,
        "final": relaxed_fin,
        "energy_initial_eV": e_i,
        "energy_final_eV": e_f,
        "endpoint_fmax_initial": info_i["fmax_final"],
        "endpoint_fmax_final": info_f["fmax_final"],
        "symmetry_delta_eV": abs(e_i - e_f),
        "symmetry_tol_eV": sym_tol,
        "floor": floor,
        "converged": bool(info_i["converged"] and info_f["converged"]),
    }
    st = _STATE.setdefault(key, {})
    st["endpoints"] = entry
    return entry


# --------------------------------------------------------------------------
# product NEB invocation helpers
# --------------------------------------------------------------------------


def _neb_resolved(ctx, args, **run_options):
    """Typed ResolvedConfig for calc_type='neb' through the product resolver."""
    from mlipx.config import resolve_config

    cli: dict[str, Any] = {
        "model_type": ctx.engine,
        "model_path": str(Path(args.model).resolve()),
        "task": ctx.task or "bulk",
        "device": ctx.device,
    }
    cli.update(run_options)
    return resolve_config(calc_type="neb", cli=cli)


def _run_dir(out_dir: Path, tag: str | None, name: str) -> Path:
    d = out_dir / "neb_runs" / (tag or "default") / name
    if d.exists():
        shutil.rmtree(d)  # transient run directories; records stay out of git
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_product_neb(ctx, resolved, run_dir, init, fin):
    from mlipx.neb.workflow import run_neb_workflow

    t0 = time.perf_counter()
    payload = run_neb_workflow(
        ctx.wrapper,
        resolved,
        output_dir=run_dir,
        initial=init,
        final=fin,
        verbose=False,
    )
    payload["_wall_seconds"] = time.perf_counter() - t0
    return payload


# --------------------------------------------------------------------------
# band / saddle analysis helpers
# --------------------------------------------------------------------------


def band_max_segment_A(images) -> float:
    """Longest MIC nearest-neighbour segment along the band (path strain)."""
    from ase.geometry import find_mic

    worst = 0.0
    for a, b in itertools.pairwise(images):
        d = b.positions - a.positions
        mic, _ = find_mic(d, a.cell, pbc=True)
        worst = max(worst, float(np.linalg.norm(mic, axis=1).max()))
    return worst


def band_energies(ctx, images) -> list[float]:
    vals = []
    for image in images:
        work = image.copy()
        work.calc = ctx.calculator
        vals.append(float(work.get_potential_energy()))
    return vals


def band_tangent_unit(images, energies_eV, ci: int) -> np.ndarray:
    """ASE improved-tangent direction at interior image ``ci`` (3N, MIC).

    Mirrors ``ase.mep.neb.ImprovedTangentMethod.get_tangent``: monotone
    segments take the downhill spring; at an extremum the two adjacent
    segments are weighted by the energy differences (paper I Eqs. 8-11).
    Both spring vectors point toward increasing image index, like ASE.
    """
    from ase.geometry import find_mic

    if not 0 < ci < len(images) - 1:
        raise ValueError("tangent is defined for interior images only")
    if len(energies_eV) != len(images):
        raise ValueError("one energy per image is required")
    cell, pbc = images[ci].cell, images[ci].pbc

    def _seg(a, b):
        d = b.positions - a.positions
        mic, _ = find_mic(d, cell, pbc)
        return mic

    spring1 = _seg(images[ci - 1], images[ci])  # (ci-1 -> ci)
    spring2 = _seg(images[ci], images[ci + 1])  # (ci -> ci+1)
    e_prev, e_cur, e_next = (
        float(energies_eV[ci - 1]),
        float(energies_eV[ci]),
        float(energies_eV[ci + 1]),
    )
    if e_next > e_cur > e_prev:
        tangent = spring2
    elif e_next < e_cur < e_prev:
        tangent = spring1
    else:
        dv_next = abs(e_next - e_cur)
        dv_prev = abs(e_prev - e_cur)
        dv_max = max(dv_next, dv_prev)
        dv_min = min(dv_next, dv_prev)
        if e_next > e_prev:
            tangent = spring2 * dv_max + spring1 * dv_min
        else:
            tangent = spring2 * dv_min + spring1 * dv_max
    flat = tangent.ravel()
    n = float(np.linalg.norm(flat))
    if n == 0.0:
        raise ValueError("degenerate band tangent (zero length)")
    return flat / n


def fd_hessian(atoms, delta: float) -> np.ndarray:
    """Central finite-difference positional Hessian (eV/A^2), 3N x 3N.

    H[i, j] = d2E/dx_i dx_j = -(F_i(+d_j) - F_i(-d_j)) / (2 delta).
    The calculator is attached once; positions are restored afterwards.
    """
    n = len(atoms)
    work = atoms.copy()
    work.calc = atoms.calc
    hessian = np.zeros((3 * n, 3 * n))
    for c in range(3 * n):
        atom, comp = divmod(c, 3)
        p0 = work.positions[atom, comp]
        work.positions[atom, comp] = p0 + delta
        f_plus = np.asarray(work.get_forces(), dtype=float).ravel()
        work.positions[atom, comp] = p0 - delta
        f_minus = np.asarray(work.get_forces(), dtype=float).ravel()
        work.positions[atom, comp] = p0
        hessian[:, c] = -(f_plus - f_minus) / (2.0 * delta)
    return 0.5 * (hessian + hessian.T)


def negative_mode_analysis(evals: np.ndarray) -> dict[str, Any]:
    """Count robust negative modes under the documented section 26 policy.

    A mode is robustly negative when its eigenvalue is below
    ``NEG_ROBUST_EV_A2`` (-0.01 eV/A^2 = -10x the 1e-3 zero-mode scale).
    Near-zero modes (|lambda| <= ZERO_SCALE_EV_A2) are reported separately
    and never counted as imaginary chemistry.
    """
    negative = [float(v) for v in evals if v < NEG_ROBUST_EV_A2]
    zero_like = int(np.sum(np.abs(evals) <= ZERO_SCALE_EV_A2))
    return {
        "min_eigenvalue_eV_A2": float(evals[0]),
        "n_robust_negative_modes": len(negative),
        "negative_eigenvalues_eV_A2": negative,
        "n_zero_like_modes": zero_like,
    }


def saddle_verdict(per_delta: list[dict], overlap: float) -> dict[str, Any]:
    """Section 26 decision: one robust negative mode persisting at every
    displacement amplitude with substantial overlap with the band tangent
    supports ``validated_first_order_candidate``; anything less is
    ``ci_neb_candidate_only``; two or more robust negative modes is a bad
    candidate and fails.
    """
    n_modes = [d["n_robust_negative_modes"] for d in per_delta]
    lam = np.array([d["min_eigenvalue_eV_A2"] for d in per_delta])
    persist = all(n == 1 for n in n_modes)
    any_robust = any(n >= 1 for n in n_modes)
    mean = float(np.mean(lam))
    spread_ok = mean < 0 and (
        float(np.max(lam) - np.min(lam)) <= NEG_STABILITY_FRACTION * abs(mean)
    )
    if any(n >= 2 for n in n_modes):
        verdict = "bad_saddle_candidate"
    elif not any_robust:
        verdict = "no_robust_negative_mode"
    elif persist and spread_ok and overlap >= TANGENT_OVERLAP_MIN:
        verdict = "validated_first_order_candidate"
    else:
        # robust negative mode exists but is not persistent/aligned enough:
        # the CI-NEB image remains only a candidate
        verdict = "ci_neb_candidate_only"
    spread_ratio = (
        float(np.max(lam) - np.min(lam)) / abs(mean) if mean != 0 else float("inf")
    )
    return {
        "saddle_validation": verdict,
        "negative_mode_persists": bool(persist),
        "n_persistent_negative_deltas": int(sum(1 for n in n_modes if n == 1)),
        "n_deltas": len(n_modes),
        "min_eigenvalue_eV_A2": float(lam.min()),
        "mean_negative_eigenvalue_eV_A2": mean,
        "min_eigenvalue_spread_ok": bool(spread_ok),
        "spread_ratio": spread_ratio,
        "spread_tolerance": NEG_STABILITY_FRACTION,
        "tangent_overlap": float(overlap),
        "tangent_overlap_threshold": TANGENT_OVERLAP_MIN,
    }


# --------------------------------------------------------------------------
# T5 workflows
# --------------------------------------------------------------------------


def workflow_endpoints(ctx, args, out_dir) -> int:
    """t5a: per-backend endpoint relaxation + symmetry consistency."""
    t0 = time.perf_counter()
    try:
        entry = _ensure_endpoints(ctx, args, out_dir)
    except Exception as exc:  # noqa: BLE001
        record(
            ctx,
            args,
            "t5_endpoints",
            "t5a_cu_vacancy_endpoints",
            "fail",
            input_atoms=None,
            parameters={"system": "cu_vacancy"},
            metrics={},
            wall=time.perf_counter() - t0,
            exception=f"{type(exc).__name__}: {exc}",
        )
        return 1
    ok = entry["converged"] and entry["symmetry_delta_eV"] <= entry["symmetry_tol_eV"]
    record(
        ctx,
        args,
        "t5_endpoints",
        "t5a_cu_vacancy_endpoints",
        "pass" if ok else "fail",
        input_atoms=entry["initial"],
        parameters={
            "system": "cu_vacancy",
            "endpoint_fmax": ENDPOINT_FMAX,
            "endpoint_steps": ENDPOINT_STEPS,
            "mapping": "identity permutation; hop expressed in coordinates",
            "image_shifts": entry["pair"]["image_shifts"],
            "hop_atom_index": entry["pair"]["hop_atom_index_in_endpoint"],
            "hop_distance_A": entry["pair"]["hop_distance_A"],
            "symmetry_gate": "|E_initial - E_final| <= max(10x repeat floor, 1e-3 eV)",
        },
        metrics={
            "energy_initial_eV": entry["energy_initial_eV"],
            "energy_final_eV": entry["energy_final_eV"],
            "symmetry_delta_eV": entry["symmetry_delta_eV"],
            "symmetry_tol_eV": entry["symmetry_tol_eV"],
            "endpoint_fmax_initial": entry["endpoint_fmax_initial"],
            "endpoint_fmax_final": entry["endpoint_fmax_final"],
            "repeat_floor": entry["floor"],
            "endpoints_converged": entry["converged"],
            "natoms": entry["pair"]["natoms"],
        },
        wall=time.perf_counter() - t0,
    )
    return 0 if ok else 1


def workflow_path_init(ctx, args, out_dir) -> int:
    """t5b: explicit mapping/winding + linear vs IDPP initial band quality."""
    from mlipx.neb.prepare import prepare_band
    from mlipx.neb.schema import NEBOptions

    t0 = time.perf_counter()
    entry = _ensure_endpoints(ctx, args, out_dir)
    pair = entry["pair"]
    metrics: dict[str, Any] = {}
    failures = 0
    try:
        bands: dict[str, Any] = {}
        for interp in ("linear", "idpp"):
            options = NEBOptions(
                n_intermediate_images=3,
                interpolation=interp,
                path_convention="unwrapped",
                fmax_eV_A=CI_FMAX,
                pre_fmax_eV_A=NEB_PRE_FMAX,
                max_steps=NEB_MAX_STEPS,
                pre_max_steps=NEB_PRE_MAX_STEPS,
                endpoint_fmax_eV_A=ENDPOINT_FMAX,
                endpoint_steps=ENDPOINT_STEPS,
                idpp_steps=IDPP_STEPS,
            )
            band = prepare_band(
                entry["initial"],
                entry["final"],
                options,
                atom_map=np.asarray(pair["atom_map"]),
                # explicit zero winding shifts (in-cell hop) under the
                # unwrapped convention, as required by the product
                image_shifts=np.asarray(pair["image_shifts"]),
            )
            energies = band_energies(ctx, band.images)
            bands[interp] = band
            metrics[f"{interp}_band_max_segment_A"] = band_max_segment_A(
                list(band.images)
            )
            metrics[f"{interp}_band_peak_minus_initial_eV"] = (
                max(energies) - energies[0]
            )
        metrics.update(
            {
                "atom_map": pair["atom_map"],
                "image_shifts": pair["image_shifts"],
                "hop_distance_A": pair["hop_distance_A"],
                "natoms": pair["natoms"],
                "mapping_check": (
                    "identity permutation; single unwrap-free hop verified "
                    "in build_vacancy_pair"
                ),
                "supercell": pair["desc"]["supercell"],
            }
        )
        status = "pass"
    except Exception as exc:  # noqa: BLE001
        status = "fail"
        metrics["exception"] = f"{type(exc).__name__}: {exc}"
        failures = 1
    record(
        ctx,
        args,
        "t5_path_init",
        "t5b_cu_vacancy_linear_vs_idpp",
        status,
        input_atoms=entry["initial"],
        parameters={
            "n_intermediate_images": 3,
            "interpolations": ["linear", "idpp"],
            "idpp_steps": IDPP_STEPS,
        },
        metrics=metrics,
        wall=time.perf_counter() - t0,
    )
    return failures


def _ci_neb_record(
    ctx,
    args,
    out_dir,
    *,
    name: str,
    n_intermediate: int,
    climb: bool,
    interpolation: str = "idpp",
    extra_run_options: dict[str, Any] | None = None,
    extra_parameters: dict[str, Any] | None = None,
    domain: str = "cu_vacancy",
) -> tuple[int, dict[str, Any] | None, Path | None]:
    """Shared CI-NEB runner for t5c/t5d/t5e/t5f/t5i records."""
    entry = _ensure_endpoints(ctx, args, out_dir)
    run_options: dict[str, Any] = {
        "n_intermediate_images": n_intermediate,
        "climb": climb,
        "neb_interpolation": interpolation,
        "fmax": CI_FMAX,
        "neb_pre_fmax": NEB_PRE_FMAX,
        "max_steps": NEB_MAX_STEPS,
        "neb_pre_max_steps": NEB_PRE_MAX_STEPS,
        "endpoint_fmax": ENDPOINT_FMAX,
        "endpoint_steps": ENDPOINT_STEPS,
        "endpoint_policy": "validate",
        # the harness proves calculator identity itself (common.calculator_identity);
        # this only skips the engine-evidence table check for these profiles
        "allow_unvalidated_neb": True,
    }
    if extra_run_options:
        run_options.update(extra_run_options)
    resolved = _neb_resolved(ctx, args, **run_options)
    run_dir = _run_dir(Path(args.out), args.tag, name)
    t0 = time.perf_counter()
    try:
        payload = _run_product_neb(
            ctx, resolved, run_dir, entry["initial"], entry["final"]
        )
    except Exception as exc:  # noqa: BLE001
        record(
            ctx,
            args,
            "t5_neb",
            name,
            "fail",
            input_atoms=entry["initial"],
            parameters={
                "domain": domain,
                "n_intermediate_images": n_intermediate,
                "climb": climb,
                "interpolation": interpolation,
                **(extra_parameters or {}),
            },
            metrics={},
            wall=time.perf_counter() - t0,
            exception=f"{type(exc).__name__}: {exc}",
        )
        return 1, None, run_dir
    converged = bool(payload["converged"])
    record(
        ctx,
        args,
        "t5_neb",
        name,
        "pass" if converged else "fail",
        input_atoms=entry["initial"],
        parameters={
            "domain": domain,
            "total_images": n_intermediate + 2,
            "climb": climb,
            "interpolation": interpolation,
            "two_stage": {"pre_fmax": NEB_PRE_FMAX, "fmax": CI_FMAX},
            "max_steps": NEB_MAX_STEPS,
            "run_dir": str(run_dir),
            "path_convention": payload["path_convention"],
            **(extra_parameters or {}),
        },
        metrics={
            "status": payload["status"],
            "converged": converged,
            "barrier_forward_sampled_eV": payload["barrier_forward_sampled_eV"],
            "barrier_reverse_sampled_eV": payload["barrier_reverse_sampled_eV"],
            "reaction_energy_eV": payload["reaction_energy_eV"],
            "barrier_status": payload["barrier_status"],
            "highest_energy_image_index": payload["highest_energy_image_index"],
            "max_neb_force_eV_A": payload["max_neb_force_eV_A"],
            "climbing_image_index": payload["climbing_image_index"],
            "climbing_physical_fmax_eV_A": payload["climbing_physical_fmax_eV_A"],
            "energies_eV": payload["energies_eV"],
            "stages": payload["stages"],
            "run_id": payload["run_id"],
            "wall_seconds": payload["_wall_seconds"],
        },
        wall=payload["_wall_seconds"],
    )
    return 0 if converged else 1, payload, run_dir


def workflow_neb_warmup(ctx, args, out_dir) -> int:
    """t5c: ordinary NEB (climb=False) warm-up through the two-stage path."""
    failures, _, _ = _ci_neb_record(
        ctx,
        args,
        out_dir,
        name="t5c_cu_vacancy_neb5_warmup",
        n_intermediate=3,
        climb=False,
    )
    return failures


def workflow_ci_neb5(ctx, args, out_dir) -> int:
    """t5d: CI-NEB, 5 total images, reference run for the saddle check."""
    failures, payload, run_dir = _ci_neb_record(
        ctx,
        args,
        out_dir,
        name="t5d_cu_vacancy_ci_neb5",
        n_intermediate=3,
        climb=True,
    )
    key = (ctx.engine, ctx.profile_id)
    _STATE.setdefault(key, {})["ci_neb5"] = {
        "payload": payload,
        "run_dir": run_dir,
        "failed": failures,
    }
    return failures


def workflow_ci_neb7(ctx, args, out_dir) -> int:
    """t5e: CI-NEB, 7 total images + 5->7 image-convergence verdict.

    If the sampled forward barrier moved by more than
    IMAGE_STABILIZATION_EV between 5 and 7 images, a 9-image run (t5f) is
    executed in the same workflow -- more images are not assumed to rescue
    a bad path, the delta is recorded either way.
    """
    state = _STATE.setdefault((ctx.engine, ctx.profile_id), {})
    ref = state.get("ci_neb5") or {}
    failures, payload, run_dir = _ci_neb_record(
        ctx,
        args,
        out_dir,
        name="t5e_cu_vacancy_ci_neb7",
        n_intermediate=5,
        climb=True,
    )
    t0 = time.perf_counter()
    image_convergence: dict[str, Any] = {"stabilized_5_to_7": None}
    if payload is not None and ref.get("payload") is not None:
        b5 = ref["payload"]["barrier_forward_sampled_eV"]
        b7 = payload["barrier_forward_sampled_eV"]
        delta = abs(b7 - b5)
        stabilized = delta <= IMAGE_STABILIZATION_EV
        image_convergence = {
            "barrier_5_images_eV": b5,
            "barrier_7_images_eV": b7,
            "delta_eV": delta,
            "stabilization_tol_eV": IMAGE_STABILIZATION_EV,
            "stabilized_5_to_7": stabilized,
        }
        extra_failures = 0
        if not stabilized:
            f9, payload9, _ = _ci_neb_record(
                ctx,
                args,
                out_dir,
                name="t5f_cu_vacancy_ci_neb9",
                n_intermediate=7,
                climb=True,
                extra_parameters={"trigger": "5->7 barrier delta beyond tol"},
            )
            extra_failures = f9
            if payload9 is not None:
                b9 = payload9["barrier_forward_sampled_eV"]
                image_convergence.update(
                    {
                        "barrier_9_images_eV": b9,
                        "delta_7_to_9_eV": abs(b9 - b7),
                    }
                )
            failures += extra_failures
        record(
            ctx,
            args,
            "t5_neb",
            "t5e_image_convergence",
            "pass",
            input_atoms=state["endpoints"]["initial"] if "endpoints" in state else None,
            parameters={
                "policy": "5 -> 7 total images; 9 only if 5->7 not stabilized",
                **image_convergence,
            },
            metrics=dict(image_convergence),
            wall=time.perf_counter() - t0,
        )
    state["ci_neb7"] = {"payload": payload, "run_dir": run_dir, "failed": failures}
    return failures


def workflow_resume_identity(ctx, args, out_dir) -> int:
    """t5g: checkpoint/resume identity (taskbook section 25.1 item 7).

    Run A uses a deliberately tiny budget and stops unconverged with a
    final checkpoint.  Run B resumes from that checkpoint in its original
    run directory with the *same* options (the product resume fingerprint
    covers the full option set, so a budget change is correctly rejected).
    Identity evidence: preserved run_id, endpoint identities, option
    fingerprint, mapping, and bitwise-continuous band geometry.
    """
    entry = _ensure_endpoints(ctx, args, out_dir)
    base_run: dict[str, Any] = {
        "n_intermediate_images": 3,
        "climb": True,
        "neb_interpolation": "idpp",
        "fmax": CI_FMAX,
        "neb_pre_fmax": NEB_PRE_FMAX,
        "max_steps": RESUME_MAIN_STEPS,
        "neb_pre_max_steps": RESUME_PRE_STEPS,
        "endpoint_fmax": ENDPOINT_FMAX,
        "endpoint_steps": ENDPOINT_STEPS,
        "endpoint_policy": "validate",
        "allow_unvalidated_neb": True,  # harness proves identity itself
        "checkpoint_interval": RESUME_CHECKPOINT_INTERVAL,
    }
    run_dir = _run_dir(Path(args.out), args.tag, "t5g_resume")
    t0 = time.perf_counter()
    metrics: dict[str, Any] = {}
    try:
        from mlipx.neb.io import load_checkpoint

        resolved_a = _neb_resolved(ctx, args, **base_run)
        run_a = _run_product_neb(
            ctx, resolved_a, run_dir, entry["initial"], entry["final"]
        )
        checkpoint = run_a["latest_checkpoint"]
        meta_a, _ = load_checkpoint(checkpoint)
        _, images_final_a = load_checkpoint(checkpoint)
        resolved_b = _neb_resolved(ctx, args, **base_run)
        from mlipx.neb.workflow import run_neb_workflow

        run_b = run_neb_workflow(
            ctx.wrapper,
            resolved_b,
            output_dir=run_dir,
            resume=checkpoint,
            verbose=False,
        )
        context_b = json_load(run_dir / "run_context.json")
        run_b_prepared = checkpoint_stage_positions(run_dir, "prepared")
        geometry_continuous = (
            run_b_prepared is not None
            and len(run_b_prepared) == len(images_final_a)
            and all(
                np.array_equal(p, q.positions)
                for p, q in zip(run_b_prepared, images_final_a, strict=False)
            )
        )
        metrics = {
            "run_id_preserved": run_b["run_id"] == run_a["run_id"],
            "run_id": run_a["run_id"],
            "attempt_ids": [run_a["attempt_id"], run_b["attempt_id"]],
            "endpoint_identity_preserved": (
                run_b["initial_identity"] == run_a["initial_identity"]
                and run_b["final_identity"] == run_a["final_identity"]
            ),
            "atom_map_preserved": run_b["atom_map"] == run_a["atom_map"],
            "image_shifts_preserved": run_b["image_shifts"] == run_a["image_shifts"],
            "resolved_options_fingerprint_preserved": (
                context_b.get("resume_fingerprint_sha256")
                == meta_a.get("resume_fingerprint_sha256")
            ),
            "band_geometry_bitwise_continuous": geometry_continuous,
            "run_a_converged": bool(run_a["converged"]),
            "run_b_converged": bool(run_b["converged"]),
            "run_b_status": run_b["status"],
            "checkpoint_stage": meta_a.get("stage"),
            "note": (
                "resume restores band geometry, not optimizer state "
                "(documented product semantics)"
            ),
        }
        identity_ok = all(
            metrics[k]
            for k in (
                "run_id_preserved",
                "endpoint_identity_preserved",
                "atom_map_preserved",
                "image_shifts_preserved",
                "resolved_options_fingerprint_preserved",
                "band_geometry_bitwise_continuous",
            )
        )
        status = "pass" if identity_ok else "fail"
    except Exception as exc:  # noqa: BLE001
        status = "fail"
        metrics["exception"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    record(
        ctx,
        args,
        "t5_resume",
        "t5g_checkpoint_resume_identity",
        status,
        input_atoms=entry["initial"],
        parameters={
            "total_images": 5,
            "run_a_budget": {
                "pre_max_steps": RESUME_PRE_STEPS,
                "max_steps": RESUME_MAIN_STEPS,
            },
            "checkpoint_interval": RESUME_CHECKPOINT_INTERVAL,
        },
        metrics=metrics,
        wall=time.perf_counter() - t0,
    )
    return 0 if status == "pass" else 1


def workflow_saddle_hessian(ctx, args, out_dir) -> int:
    """t5h: section 26 FD-Hessian saddle check on the CI-NEB reference image."""
    state = _STATE.setdefault((ctx.engine, ctx.profile_id), {})
    ref = state.get("ci_neb5")
    t0 = time.perf_counter()
    if ref is None or ref["payload"] is None:
        record(
            ctx,
            args,
            "t5_saddle",
            "t5h_saddle_hessian",
            "fail",
            input_atoms=None,
            parameters={},
            metrics={},
            wall=time.perf_counter() - t0,
            exception="t5d CI-NEB reference run is required and did not complete",
        )
        return 1
    payload = ref["payload"]
    try:
        from mlipx.neb.io import load_checkpoint

        _, images = load_checkpoint(payload["latest_checkpoint"])
        energies = payload["energies_eV"]
        ci = payload["climbing_image_index"]
        if ci is None:
            ci = int(payload["highest_energy_image_index"])
        if ci == 0 or ci == len(images) - 1:
            raise ValueError("peak image is an endpoint; no internal saddle sampled")
        ci_image = images[ci].copy()
        ci_image.calc = ctx.calculator
        tangent = band_tangent_unit(images, energies, ci)
        per_delta = []
        overlap = None
        for delta in SADDLE_DELTAS:
            hessian = fd_hessian(ci_image, delta)
            evals, vecs = np.linalg.eigh(hessian)
            per_delta.append(negative_mode_analysis(evals))
            overlap = abs(float(np.dot(vecs[:, 0], tangent)))
        verdict = saddle_verdict(per_delta, overlap)
        if verdict["saddle_validation"] == "bad_saddle_candidate":
            status = "fail"
        elif verdict["saddle_validation"] == "validated_first_order_candidate":
            status = "pass"
        else:
            status = "characterized"
        metrics = {
            "ci_image_index": ci,
            "ci_converged": bool(payload["converged"]),
            "barrier_forward_sampled_eV": payload["barrier_forward_sampled_eV"],
            "deltas_A": list(SADDLE_DELTAS),
            "zero_scale_eV_A2": ZERO_SCALE_EV_A2,
            "robust_negative_threshold_eV_A2": NEG_ROBUST_EV_A2,
            "hessian_size": "full positional (3N x 3N)",
            **verdict,
            "per_delta": per_delta,
        }
    except Exception as exc:  # noqa: BLE001
        status = "fail"
        metrics = {"exception": f"{type(exc).__name__}: {exc}"}
        traceback.print_exc()
    record(
        ctx,
        args,
        "t5_saddle",
        "t5h_saddle_hessian",
        status,
        input_atoms=None,
        parameters={
            "deltas_A": list(SADDLE_DELTAS),
            "method": "central FD Hessian, full 3N, eigen decomposition",
            "criteria": (
                "exactly one eigenvalue < -1e-2 eV/A^2 (10x zero-mode scale) "
                "at every delta; spread <= 50% of |mean|; |<v,tangent>| >= 0.8"
            ),
        },
        metrics=metrics,
        wall=time.perf_counter() - t0,
    )
    return 0 if status != "fail" else 1


def json_load(path: Path) -> dict[str, Any]:
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_stage_positions(run_dir: Path, stage: str) -> list | None:
    """Positions of the newest checkpoint with ``stage`` == the given value."""
    found = None
    for meta_path in sorted((run_dir / "checkpoints").glob("step_*/checkpoint.json")):
        meta = json_load(meta_path)
        if meta.get("stage") == stage:
            found = meta_path.parent
    if found is None:
        return None
    _, images = load_checkpoint_dir(found)
    return [img.positions.copy() for img in images]


def load_checkpoint_dir(path: Path):
    from mlipx.neb.io import load_checkpoint

    return load_checkpoint(path)


def workflow_na3ps4_hop(ctx, args, out_dir) -> int:
    """t5i: alpha-Na3PS4 Na-vacancy hop (domain/transferability case).

    Barrier is recorded as a model characterisation: no DFT reference is
    pinned for this system, and the four model values must never be read as
    four independent references (taskbook section 25.2).
    """
    t0 = time.perf_counter()
    pair = None
    try:
        pair = na3ps4_vacancy_pair()
        init = pair["initial"].copy()
        init.calc = ctx.calculator
        fin = pair["final"].copy()
        fin.calc = ctx.calculator
        relaxed_i, info_i = relax_positions(init, ENDPOINT_FMAX, ENDPOINT_STEPS, "fire")
        relaxed_f, info_f = relax_positions(fin, ENDPOINT_FMAX, ENDPOINT_STEPS, "fire")
        run_options: dict[str, Any] = {
            "n_intermediate_images": 3,
            "climb": True,
            "neb_interpolation": "idpp",
            "fmax": CI_FMAX,
            "neb_pre_fmax": NEB_PRE_FMAX,
            "max_steps": NEB_MAX_STEPS,
            "neb_pre_max_steps": NEB_PRE_MAX_STEPS,
            "endpoint_fmax": ENDPOINT_FMAX,
            "endpoint_steps": ENDPOINT_STEPS,
            "endpoint_policy": "validate",
            "allow_unvalidated_neb": True,  # harness proves identity itself
        }
        resolved = _neb_resolved(ctx, args, **run_options)
        run_dir = _run_dir(Path(args.out), args.tag, "t5i_na3ps4_hop")
        payload = _run_product_neb(ctx, resolved, run_dir, relaxed_i, relaxed_f)
        converged = bool(payload["converged"])
        metrics = {
            "natoms": pair["natoms"],
            "hop_distance_A": pair["hop_distance_A"],
            "vacancy_site_A": pair["vacancy_site"],
            "mapping_check": "identity permutation; zero image shifts (unwrapped hop)",
            "endpoint_fmax_initial": info_i["fmax_final"],
            "endpoint_fmax_final": info_f["fmax_final"],
            "barrier_forward_sampled_eV": payload["barrier_forward_sampled_eV"],
            "barrier_reverse_sampled_eV": payload["barrier_reverse_sampled_eV"],
            "reaction_energy_eV": payload["reaction_energy_eV"],
            "barrier_status": payload["barrier_status"],
            "max_neb_force_eV_A": payload["max_neb_force_eV_A"],
            "converged": converged,
            "reference": "none (transferability characterization)",
        }
        status = "characterized" if converged else "fail"
    except Exception as exc:  # noqa: BLE001
        status = "fail"
        metrics = {"exception": f"{type(exc).__name__}: {exc}"}
        traceback.print_exc()
    record(
        ctx,
        args,
        "t5_neb",
        "t5i_na3ps4_na_hop_ci_neb5",
        status,
        input_atoms=pair["initial"] if pair is not None else None,
        parameters={
            "domain": "na3ps4_vacancy",
            "fixture": "mp-28782 geometry (pinned)",
            "total_images": 5,
            "note": "barrier characterizes the model; not a DFT reference",
        },
        metrics=metrics,
        wall=time.perf_counter() - t0,
    )
    return 0 if status != "fail" else 1


WORKFLOW_FUNCS = {
    "endpoints": workflow_endpoints,
    "path_init": workflow_path_init,
    "neb_warmup": workflow_neb_warmup,
    "ci_neb5": workflow_ci_neb5,
    "ci_neb7": workflow_ci_neb7,
    "resume_identity": workflow_resume_identity,
    "saddle_hessian": workflow_saddle_hessian,
    "na3ps4_hop": workflow_na3ps4_hop,
}

#: execution order; dependencies are resolved lazily via _STATE caches but
#: the default order makes every cache hit warm.
DEFAULT_ORDER = (
    "endpoints",
    "path_init",
    "neb_warmup",
    "ci_neb5",
    "ci_neb7",
    "resume_identity",
    "saddle_hessian",
    "na3ps4_hop",
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
            case_id="t5_setup",
            test_id="t5_neb_suite_setup",
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

    print(f"neb_suite: engine={ctx.engine} workflows={requested} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
