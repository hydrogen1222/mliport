"""T3: common OMat24 held-out accuracy (taskbook section 14).

Run inside a backend environment::

    python omat_eval.py --engine mace --profile-id mace_omat \
        --model ... --manifest ... --out .validation-work/t3 \
        --subset .validation-work/omat24/omat24_subset.extxyz

Evaluates the pinned ~256-structure OMat24 validation subset (built by
``omat_subset.py`` with an immutable seed) through the same mlipx
calculator path the product exposes, and reports energy/force/stress
error statistics against the dataset's own reference labels.  Metrics:

- energy per atom: MAE / RMSE / median / p95 of |dE|
- forces: component MAE/RMSE, vector-norm error, cosine statistics for
  non-negligible reference forces, error stratified by reference force
  magnitude
- stress (profiles that support it): component MAE/RMSE in eV/A^3 and
  GPa, hydrostatic and deviatoric errors
- strata: source subdataset, atom-count range, low/high reference force

Per-structure errors are written to ``t3_errors_<engine>.csv`` next to
the records so the figures regenerate from machine-readable data.

No elemental offsets are fitted or subtracted (taskbook section 14);
the OMat24 PBE(+U) reference level is self-consistent for this
comparison and is never mixed with Materials Project energies.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import common  # noqa: E402
import engines  # noqa: E402

EV_A3_TO_GPA = common.EV_A3_TO_GPA

#: cosine statistics are only meaningful for non-negligible forces
COS_MIN_REF_FORCE_EV_A = 0.1
#: reference force magnitude bins (eV/A)
FORCE_BINS = (0.1, 0.5, 1.0, 2.0)
#: atom-count strata edges
NATOMS_BINS = (20, 50)
#: low/high reference-force stratum edge (eV/A)
REF_FORCE_STRATUM_EV_A = 0.5


def _abs_stats(values: np.ndarray) -> dict[str, Any]:
    """Statistics of |values| with a shared RMSE over signed values."""
    v = np.asarray(values, dtype=float)
    a = np.abs(v)
    return {
        "n": int(a.size),
        "mae": float(a.mean()),
        "rmse": float(np.sqrt((v**2).mean())),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _force_metrics(
    f_ref_all: np.ndarray, f_model_all: np.ndarray
) -> dict[str, Any]:
    diff = f_model_all - f_ref_all
    ref_norm = np.linalg.norm(f_ref_all, axis=1)
    metrics: dict[str, Any] = {
        "component_mae_eV_A": float(np.abs(diff).mean()),
        "component_rmse_eV_A": float(np.sqrt((diff**2).mean())),
        "norm_error_mean_eV_A": float(np.linalg.norm(diff, axis=1).mean()),
    }
    mask = ref_norm >= COS_MIN_REF_FORCE_EV_A
    if mask.sum() > 0:
        fr = f_ref_all[mask]
        fm = f_model_all[mask]
        cos = (fr * fm).sum(axis=1) / (
            np.linalg.norm(fr, axis=1) * np.linalg.norm(fm, axis=1)
        )
        metrics["cosine"] = {
            "min_ref_force_eV_A": COS_MIN_REF_FORCE_EV_A,
            "n": int(mask.sum()),
            "mean": float(cos.mean()),
            "median": float(np.median(cos)),
            "min": float(cos.min()),
            "frac_below_0p9": float((cos < 0.9).mean()),
        }
    edges = [0.0, *FORCE_BINS, float("inf")]
    strata = []
    for lo, hi in itertools.pairwise(edges):
        sel = (ref_norm >= lo) & (ref_norm < hi)
        if sel.sum() == 0:
            continue
        strata.append(
            {
                "ref_force_lo_eV_A": lo,
                "ref_force_hi_eV_A": None if np.isinf(hi) else hi,
                "n_atoms": int(sel.sum()),
                "component_mae_eV_A": float(np.abs(diff[sel]).mean()),
                "component_rmse_eV_A": float(
                    np.sqrt((diff[sel] ** 2).mean())
                ),
            }
        )
    metrics["by_ref_force_magnitude"] = strata
    return metrics


def _stress_metrics(
    s_ref_all: np.ndarray, s_model_all: np.ndarray
) -> dict[str, Any]:
    """Voigt order xx,yy,zz,yz,xz,xy in eV/A^3 (ASE convention)."""
    diff = s_model_all - s_ref_all
    hydro_ref = s_ref_all[:, :3].mean(axis=1)
    hydro_model = s_model_all[:, :3].mean(axis=1)
    dev_ref = s_ref_all.copy()
    dev_model = s_model_all.copy()
    dev_ref[:, :3] -= hydro_ref[:, None]
    dev_model[:, :3] -= hydro_model[:, None]
    return {
        "n_structures": int(diff.shape[0]),
        "component_mae_eV_A3": float(np.abs(diff).mean()),
        "component_rmse_eV_A3": float(np.sqrt((diff**2).mean())),
        "component_mae_GPa": float(np.abs(diff).mean() * EV_A3_TO_GPA),
        "component_rmse_GPa": float(
            np.sqrt((diff**2).mean()) * EV_A3_TO_GPA
        ),
        "hydrostatic_error_mean_eV_A3": float(
            (hydro_model - hydro_ref).mean()
        ),
        "hydrostatic_error_mae_GPa": float(
            (np.abs(hydro_model - hydro_ref).mean()) * EV_A3_TO_GPA
        ),
        "deviatoric_component_mae_eV_A3": float(
            np.abs(dev_model - dev_ref).mean()
        ),
    }


def _natoms_stratum(n: int) -> str:
    if n < NATOMS_BINS[0]:
        return f"atoms_lt_{NATOMS_BINS[0]}"
    if n < NATOMS_BINS[1]:
        return f"atoms_{NATOMS_BINS[0]}_to_{NATOMS_BINS[1]}"
    return f"atoms_ge_{NATOMS_BINS[1]}"


def _breakdown(
    groups: dict[str, list[int]],
    frames_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, idxs in sorted(groups.items()):
        d = np.array(
            [
                (frames_meta[i]["e_model_eV"] - frames_meta[i]["e_ref_eV"])
                / frames_meta[i]["natoms"]
                for i in idxs
            ]
        )
        out[name] = _abs_stats(d)
    return out


def run_engine(args: argparse.Namespace) -> int:
    from ase.io import read as ase_read

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    profile = manifest["profiles"][args.profile_id]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = ase_read(args.subset, index=":", format="extxyz")
    print(f"[t3] {args.engine}: {len(frames)} OMat24 structures")

    ctx = engines.build_engine(
        args.engine,
        args.profile_id,
        profile,
        args.model,
        device=args.device,
        dtype=args.dtype,
        head=args.head,
    )

    strata_seen: dict[str, int] = {}
    frames_meta: list[dict[str, Any]] = []
    f_ref_list: list[np.ndarray] = []
    f_model_list: list[np.ndarray] = []
    s_ref_list: list[np.ndarray] = []
    s_model_list: list[np.ndarray] = []
    failures = 0
    t0 = time.perf_counter()
    common.reset_vram_counter()

    for i, atoms in enumerate(frames):
        ref_calc = atoms.calc
        if ref_calc is None:
            raise RuntimeError(
                f"subset frame {i} has no reference labels; the extxyz "
                "must carry OMat24 reference energy/forces"
            )
        e_ref = float(ref_calc.get_potential_energy())
        f_ref = np.asarray(ref_calc.get_forces(), dtype=float)
        try:
            s_ref = np.asarray(ref_calc.get_stress(), dtype=float)
        except Exception:  # noqa: BLE001 - subset stress may be absent
            s_ref = None
        meta: dict[str, Any] = {
            "stratum": atoms.info.get("omat24_stratum", "unknown"),
            "index": int(atoms.info.get("omat24_index", i)),
            "natoms": len(atoms),
            "f_ref_max": float(np.linalg.norm(f_ref, axis=1).max()),
            "f_ref_rms": float(np.sqrt((f_ref**2).sum(axis=1).mean())),
            "e_ref_eV": e_ref,
            "ok": False,
            "e_model_eV": None,
            "f_comp_mae": None,
            "f_cos_mean": None,
            "s_component_mae": None,
            "s_hydro_error": None,
        }
        strata_seen[meta["stratum"]] = strata_seen.get(meta["stratum"], 0) + 1
        try:
            # fresh Atoms each time; the calculator is the product path
            work = atoms.copy()
            work.calc = ctx.calculator
            e_model = float(work.get_potential_energy())
            f_model = np.asarray(work.get_forces(), dtype=float)
            try:
                s_model = np.asarray(work.get_stress(), dtype=float)
            except Exception:  # noqa: BLE001 - stress support is a property
                s_model = None
            if np.isfinite(e_model) and np.isfinite(f_model).all():
                meta["e_model_eV"] = e_model
                meta["f_comp_mae"] = float(np.abs(f_model - f_ref).mean())
                ref_norm = np.linalg.norm(f_ref, axis=1)
                mask = ref_norm >= COS_MIN_REF_FORCE_EV_A
                if mask.sum():
                    fr = f_ref[mask]
                    fm = f_model[mask]
                    cos = (fr * fm).sum(axis=1) / (
                        np.linalg.norm(fr, axis=1)
                        * np.linalg.norm(fm, axis=1)
                    )
                    meta["f_cos_mean"] = float(cos.mean())
                f_ref_list.append(f_ref)
                f_model_list.append(f_model)
                if s_ref is not None and s_model is not None:
                    s_ref_list.append(s_ref)
                    s_model_list.append(s_model)
                    meta["s_component_mae"] = float(
                        np.abs(s_model - s_ref).mean()
                    )
                    meta["s_hydro_error"] = float(
                        s_model[:3].mean() - s_ref[:3].mean()
                    )
                meta["ok"] = True
            else:
                meta["error"] = "non-finite energy or forces"
        except Exception as exc:  # noqa: BLE001 - record and continue
            meta["error"] = f"{type(exc).__name__}: {exc}"[:200]
        if not meta["ok"]:
            failures += 1
        frames_meta.append(meta)

    wall = time.perf_counter() - t0
    ok_meta = [m for m in frames_meta if m["ok"]]
    if not ok_meta:
        status = "fail"
        reason = "all_structures_failed"
    elif failures:
        status = "characterized"
        reason = "partial_execution"
    else:
        status = "pass"
        reason = None

    de = np.array(
        [
            (m["e_model_eV"] - m["e_ref_eV"]) / m["natoms"]
            for m in ok_meta
        ]
    )
    metrics: dict[str, Any] = {
        "n_structures": len(frames),
        "n_failed": failures,
        "strata_seen": strata_seen,
        "energy_per_atom_eV": _abs_stats(de),
    }
    if f_ref_list:
        metrics["forces"] = _force_metrics(
            np.vstack(f_ref_list), np.vstack(f_model_list)
        )
    if s_ref_list:
        metrics["stress"] = _stress_metrics(
            np.vstack(s_ref_list), np.vstack(s_model_list)
        )
    else:
        # the official OMat24 validation LMDB carries no stress labels --
        # record that absence instead of silently skipping the section
        metrics["stress"] = {
            "reference_stress_available": False,
            "note": (
                "OMat24 validation reference stress labels are absent in "
                "the official LMDB; stress parity versus DFT is not "
                "computable for this reference level"
            ),
        }

    # energy-per-atom strata breakdowns
    for label, key_fn in (
        ("stratum", lambda m: m["stratum"]),
        ("natoms", lambda m: _natoms_stratum(m["natoms"])),
        (
            "ref_force",
            lambda m: "low_force"
            if m["f_ref_max"] < REF_FORCE_STRATUM_EV_A
            else "high_force",
        ),
    ):
        groups: dict[str, list[int]] = {}
        for i, m in enumerate(frames_meta):
            if m["ok"]:
                groups.setdefault(key_fn(m), []).append(i)
        metrics[f"energy_by_{label}"] = _breakdown(groups, frames_meta)

    # machine-readable per-structure errors for the figure generator
    csv_path = out_dir / f"t3_errors_{args.engine}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "stratum",
                "omat24_index",
                "natoms",
                "e_ref_eV",
                "e_model_eV",
                "de_per_atom_eV",
                "f_ref_rms_eV_A",
                "f_ref_max_eV_A",
                "f_component_mae_eV_A",
                "f_cosine_mean",
                "s_component_mae_eV_A3",
                "s_hydro_error_eV_A3",
                "ok",
            ]
        )
        for m in frames_meta:
            writer.writerow(
                [
                    m["stratum"],
                    m["index"],
                    m["natoms"],
                    f"{m['e_ref_eV']:.6f}",
                    f"{m['e_model_eV']:.6f}" if m["ok"] else "",
                    f"{(m['e_model_eV'] - m['e_ref_eV']) / m['natoms']:.6e}"
                    if m["ok"]
                    else "",
                    f"{m['f_ref_rms']:.6f}",
                    f"{m['f_ref_max']:.6f}",
                    f"{m['f_comp_mae']:.6e}" if m["f_comp_mae"] is not None else "",
                    f"{m['f_cos_mean']:.4f}" if m["f_cos_mean"] is not None else "",
                    f"{m['s_component_mae']:.6e}" if m["s_component_mae"] is not None else "",
                    f"{m['s_hydro_error']:.6e}" if m["s_hydro_error"] is not None else "",
                    int(m["ok"]),
                ]
            )

    diagnostics: dict[str, Any] = {**ctx.diagnostics}
    if reason:
        diagnostics["reason"] = reason
        diagnostics["failed_structures"] = [
            {
                "stratum": m["stratum"],
                "index": m["index"],
                "error": m.get("error", ""),
            }
            for m in frames_meta
            if not m["ok"]
        ][:20]

    record = common.result_record(
        case_id="t3_omat24",
        test_id="t3_omat24",
        status=status,
        engine=ctx.engine,
        model_identity=ctx.model_identity,
        model_sha256=ctx.model_sha256,
        task=ctx.task,
        head=ctx.head,
        dtype=ctx.dtype,
        device=common.device_record(args.device),
        input_structure_id=common.sha256_file(Path(args.subset)),
        parameters={
            "subset": str(args.subset),
            "n_structures": len(frames),
            "cos_min_ref_force_eV_A": COS_MIN_REF_FORCE_EV_A,
            "force_bins_eV_A": list(FORCE_BINS),
            "natoms_bins": list(NATOMS_BINS),
            "ref_force_stratum_eV_A": REF_FORCE_STRATUM_EV_A,
        },
        metrics=metrics,
        diagnostics=diagnostics,
        wall_seconds=round(wall, 2),
        peak_vram=common.peak_vram_mib(),
    )
    path = common.write_result(record, out_dir, tag=args.tag)
    print(
        f"[t3] {args.engine}: {status} structures={len(ok_meta)}"
        f"/{len(frames)} E/atom MAE={metrics['energy_per_atom_eV']['mae']:.4f}"
        f" eV wall={wall:.1f}s -> {path.name}"
    )
    return 0 if status in ("pass", "characterized") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=engines._ENGINES)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--tag", default="t3beta")
    args = parser.parse_args()
    try:
        return run_engine(args)
    except Exception:  # noqa: BLE001 - record the crash and fail loudly
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
