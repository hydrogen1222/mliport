"""T9: literature-referenced DFT accuracy (LiPS-25, LiPS20, OMat24, JARVIS).

Run inside a backend environment, against an *imported* reference file::

    python literature_eval.py --engine mace --profile-id mace_omat \
        --model ... --manifest validation/science/model_manifest.json \
        --dataset lips25 --split test \
        --subset .validation-work/lit/lips25/test.extxyz \
        --out .validation-work/t9

The import step (``validation/science/scripts/benchmark.py`` or
``literature.importers``) is deliberately separate, exactly like
``omat_subset.py`` + ``omat_eval.py``: the model environment never downloads
anything and every evaluation can be reproduced offline from a hash-verified
reference file.

Scientific contract
-------------------

* **Software correctness and model accuracy are separate.** ``status=pass``
  means every requested structure was evaluated and produced finite values.
  Accuracy is *recorded*, never converted into a pass/fail claim, because no
  universal error threshold exists for MLIPs.
* **Reference levels are never mixed.** Each record stores the reference
  level (functional/software) and the comparison mode from the registry.
  ``reference_matched`` datasets may be read as absolute-energy accuracy;
  ``cross_method`` datasets are reported as forces, within-group relative
  energies and rankings, and the absolute energy block is explicitly labelled
  non-comparable.
* **No fabricated labels.** A missing force/stress label is reported as
  unavailable for that section; a virial array is never silently converted
  into stress.
* **No hard-coded accuracy thresholds.** The only numeric gates are execution
  integrity (finite values, all structures attempted).
"""

from __future__ import annotations

import argparse
import csv
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

_SCIENCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SCIENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCIENCE_ROOT))

from literature import (  # noqa: E402
    datasets as literature_datasets,
    metrics as literature_metrics,
)
from literature.importers import (  # noqa: E402
    LIPS25_ENDPOINT_HIGH,
    LIPS25_ENDPOINT_LOW,
    LIPS25_TIE_LINE,
)

EV_A3_TO_GPA = common.EV_A3_TO_GPA


# ---------------------------------------------------------------------------
# reference label inspection
# ---------------------------------------------------------------------------


def _labels_available(frames: list[Any]) -> dict[str, bool]:
    """Probe the reference labels on the first few frames (never assume)."""
    energy = forces = stress = True
    for atoms in frames[: min(8, len(frames))]:
        calc = atoms.calc
        if calc is None:
            return {"energy": False, "forces": False, "stress": False}
        try:
            calc.get_potential_energy(atoms)
        except (NotImplementedError, KeyError, ValueError):
            energy = False
        try:
            calc.get_forces(atoms)
        except (NotImplementedError, KeyError, ValueError):
            forces = False
        try:
            calc.get_stress(atoms)
        except (NotImplementedError, KeyError, ValueError):
            stress = False
    return {"energy": energy, "forces": forces, "stress": stress}


def _reference_energy(atoms: Any) -> float | None:
    if atoms.calc is None:
        return None
    try:
        value = float(atoms.calc.get_potential_energy(atoms))
    except Exception:  # noqa: BLE001 - unlabelled reference input
        return None
    return value if np.isfinite(value) else None


def _reference_forces(atoms: Any) -> np.ndarray | None:
    if atoms.calc is None:
        return None
    try:
        forces = np.asarray(atoms.calc.get_forces(atoms), dtype=float)
    except Exception:  # noqa: BLE001 - forces are an optional label
        return None
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        return None
    return forces


def _reference_stress(atoms: Any) -> np.ndarray | None:
    if atoms.calc is None:
        return None
    try:
        stress = np.asarray(atoms.calc.get_stress(atoms), dtype=float)
    except Exception:  # noqa: BLE001 - stress is an optional label
        return None
    if stress.shape != (6,) or not np.isfinite(stress).all():
        return None
    return stress


def _load_selection(subset: Path) -> dict[str, Any]:
    candidate = subset.parent / "selection.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _tie_line_key(label: str) -> str | None:
    if label in LIPS25_TIE_LINE:
        return label
    prefixed = f"dhf_{label}"
    if prefixed in LIPS25_TIE_LINE:
        return prefixed
    return None


# ---------------------------------------------------------------------------
# relaxation (ASE upstream algorithms only)
# ---------------------------------------------------------------------------


def _relax_structure(
    atoms: Any, calculator: Any, fmax: float, max_steps: int
) -> dict[str, Any]:
    from ase.filters import ExpCellFilter  # noqa: PLC0415
    from ase.optimize import BFGS  # noqa: PLC0415

    work = atoms.copy()
    work.calc = calculator
    filtered = ExpCellFilter(work)
    optimizer = BFGS(filtered, logfile=None)
    converged = False
    error = None
    try:
        optimizer.run(fmax=fmax, steps=max_steps)
        converged = bool(optimizer.converged())
    except Exception as exc:  # noqa: BLE001 - record the failure and continue
        error = f"{type(exc).__name__}: {exc}"[:200]
    return {
        "atoms": work,
        "converged": converged,
        "steps": int(optimizer.nsteps),
        "fmax_target": float(fmax),
        "max_steps": int(max_steps),
        "error": error,
    }


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------


def run_engine(args: argparse.Namespace) -> int:
    from ase.io import read as ase_read  # noqa: PLC0415

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    profile = manifest["profiles"][args.profile_id]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = literature_datasets.load_registry()
    entry = literature_datasets.dataset_entry(registry, args.dataset)
    if args.split not in entry["splits"]:
        msg = (
            f"dataset {args.dataset!r} has no split {args.split!r}; expected "
            f"one of {sorted(entry['splits'])}"
        )
        raise SystemExit(msg)
    split_entry = entry["splits"][args.split]
    subset = Path(args.subset)
    if not subset.exists():
        msg = (
            f"reference file not found: {subset}. Import the dataset first: "
            "python validation/science/scripts/benchmark.py --tier core "
            f"--datasets {args.dataset} --engine ... --import-only"
        )
        raise SystemExit(msg)
    subset_sha = common.sha256_file(subset)
    selection = _load_selection(subset)
    expected_sha = (
        (selection.get("splits", {}).get(args.split, {}) or {})
        .get("source", {})
        .get("sha256")
    )
    frames = ase_read(subset, index=":", format="extxyz")
    if args.max_structures:
        frames = frames[: int(args.max_structures)]
    labels = _labels_available(frames)
    print(f"[t9] {args.dataset}/{args.split}: {len(frames)} structures, labels={labels}")

    ctx = engines.build_engine(
        args.engine,
        args.profile_id,
        profile,
        args.model,
        device=args.device,
        dtype=args.dtype,
        head=args.head,
    )

    e_ref: list[float] = []
    model_energies: list[float] = []
    atom_counts: list[int] = []
    groups: list[str] = []
    f_ref_all: list[np.ndarray] = []
    f_model_all: list[np.ndarray] = []
    symbols_all: list[str] = []
    s_ref_all: list[np.ndarray] = []
    s_model_all: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    relaxation: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    common.reset_vram_counter()
    for index, reference in enumerate(frames):
        record_id = str(
            reference.info.get("literature_record_id", f"{args.dataset}:{index}")
        )
        group = str(reference.info.get("literature_group", "all"))
        phase = str(
            reference.info.get("literature_phase", reference.info.get("label", group))
        )
        natoms_ref = len(reference)
        row: dict[str, Any] = {
            "record_id": record_id,
            "group": group,
            "phase": phase,
            "natoms": natoms_ref,
            "ok": False,
        }
        try:
            scored = reference
            if args.relax:
                relaxed = _relax_structure(
                    reference, ctx.calculator, args.fmax, args.max_steps
                )
                scored = relaxed["atoms"]
                relaxation.append(
                    {
                        "record_id": record_id,
                        "group": group,
                        "phase": phase,
                        "converged": relaxed["converged"],
                        "steps": relaxed["steps"],
                        "error": relaxed["error"],
                        "cell": np.asarray(scored.cell.array).tolist(),
                        "volume_A3": float(scored.get_volume()),
                        "energy_eV": float(scored.get_potential_energy()),
                        "fmax_eV_A": float(
                            np.linalg.norm(scored.get_forces(), axis=1).max()
                        ),
                    }
                )
            else:
                # never attach the model calculator to the reference frame:
                # that would overwrite the reference labels in place
                scored = reference.copy()
                scored.calc = ctx.calculator
            e_model = float(scored.get_potential_energy())
            f_model = np.asarray(scored.get_forces(), dtype=float)
            if not (np.isfinite(e_model) and np.isfinite(f_model).all()):
                raise ValueError("non-finite energy or forces from the model")
            row["e_model_eV"] = e_model
            if args.relax:
                row["converged"] = relaxation[-1]["converged"]

            ref_energy = _reference_energy(reference)
            if ref_energy is not None:
                e_ref.append(ref_energy)
                model_energies.append(e_model)
                atom_counts.append(natoms_ref)
                groups.append(group)
                row["e_ref_eV"] = ref_energy
                row["de_per_atom_eV"] = (e_model - ref_energy) / natoms_ref

            ref_forces = _reference_forces(reference)
            if ref_forces is not None:
                f_ref_all.append(ref_forces)
                f_model_all.append(f_model)
                symbols_all.extend(reference.get_chemical_symbols())
                diff = f_model - ref_forces
                ref_norm = np.linalg.norm(ref_forces, axis=1)
                row["f_component_mae_eV_A"] = float(np.abs(diff).mean())
                row["f_vector_norm_p95_eV_A"] = float(
                    np.percentile(np.linalg.norm(diff, axis=1), 95)
                )
                row["f_ref_norm_max_eV_A"] = float(ref_norm.max())
                mask = ref_norm >= literature_metrics.COS_MIN_REF_FORCE_EV_A
                if mask.any():
                    fr = ref_forces[mask]
                    fm = f_model[mask]
                    cos = (fr * fm).sum(axis=1) / (
                        np.linalg.norm(fr, axis=1) * np.linalg.norm(fm, axis=1)
                    )
                    row["f_cosine_mean"] = float(np.clip(cos, -1, 1).mean())

            ref_stress = _reference_stress(reference)
            if ref_stress is not None:
                try:
                    model_stress = np.asarray(scored.get_stress(), dtype=float)
                except Exception:  # noqa: BLE001 - backend may not support stress
                    model_stress = None
                if model_stress is not None and model_stress.shape == ref_stress.shape:
                    s_ref_all.append(ref_stress)
                    s_model_all.append(model_stress)
                    row["s_component_mae_eV_A3"] = float(
                        np.abs(model_stress - ref_stress).mean()
                    )
            row["ok"] = True
        except Exception as exc:  # noqa: BLE001 - record and continue
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            failures.append({"record_id": record_id, "error": row["error"]})
        rows.append(row)

    wall = time.perf_counter() - t0
    n_ok = sum(1 for row in rows if row["ok"])
    comparison_mode = entry["comparison_mode"]

    metrics: dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "comparison_mode": comparison_mode,
        "reference_level": entry["reference_level"],
        "reference_dft": entry["dft"],
        "reference_units": entry["units"],
        "data_overlap": entry.get("data_overlap"),
        "labels_available_in_reference": labels,
        "n_structures": len(frames),
        "n_evaluated": n_ok,
        "n_failed": len(failures),
    }

    if e_ref:
        metrics["energy_per_atom_eV"] = literature_metrics.energy_per_atom_stats(
            e_ref, model_energies, atom_counts
        )
        metrics["energy_per_atom_eV"]["interpretation"] = (
            "absolute total-energy accuracy; comparable because the reference "
            "level matches the model's training level"
            if comparison_mode == "reference_matched"
            else "diagnostic only: the reference level differs from the model "
            "training level, so this absolute difference includes an arbitrary "
            "cross-method offset and is NOT an accuracy score"
        )
    else:
        metrics["energy_per_atom_eV"] = {
            "available": False,
            "reason": "the reference frames carry no energy labels",
        }

    if e_ref and split_entry.get("group_key"):
        try:
            metrics["relative_energy"] = (
                literature_metrics.group_relative_energy_stats(
                    e_ref, model_energies, atom_counts, groups
                )
            )
            metrics["relative_energy"]["group_key"] = split_entry["group_key"]
        except ValueError as exc:
            metrics["relative_energy"] = {"available": False, "reason": str(exc)}
    elif e_ref:
        metrics["relative_energy"] = {
            "available": False,
            "reason": (
                "this split has no within-composition grouping, so removing a "
                "group mean would be physically meaningless"
            ),
        }

    if f_ref_all:
        metrics["forces"] = literature_metrics.force_stats(
            np.vstack(f_ref_all),
            np.vstack(f_model_all),
            symbols=symbols_all,
        )
        metrics["forces"]["interpretation"] = (
            "force errors are comparable across reference levels at first "
            "order; residual cross-method bias remains and is recorded"
        )
    else:
        metrics["forces"] = {
            "available": False,
            "reason": (
                "the reference frames carry no force labels"
                if not labels["forces"]
                else "no frame produced a force comparison"
            ),
        }

    if s_ref_all:
        metrics["stress"] = literature_metrics.stress_stats(
            np.vstack(s_ref_all), np.vstack(s_model_all)
        )
    else:
        metrics["stress"] = {
            "available": False,
            "reference_stress_available": bool(labels["stress"]),
            "reason": (
                "this dataset ships no verified stress label"
                if not labels["stress"]
                else "stress comparison unavailable"
            ),
            "virial_policy": (
                "a stored virial is provenance only and is never converted to "
                "stress without a documented sign/normalisation convention"
            ),
        }

    if args.relax:
        metrics["relaxation"] = {
            "fmax_eV_A": args.fmax,
            "max_steps": args.max_steps,
            "n_structures": len(relaxation),
            "n_converged": sum(1 for r in relaxation if r["converged"]),
            "records": relaxation,
        }
        reference_split = split_entry.get("relaxed_reference_split")
        reference_relaxed = (
            subset.parent / f"{reference_split}.extxyz"
            if reference_split
            else None
        )
        if reference_relaxed is not None and reference_relaxed.exists():
            metrics["relaxation"]["lattice_vs_reference"] = _lattice_comparison(
                relaxation, reference_relaxed
            )
        else:
            metrics["relaxation"]["lattice_vs_reference"] = {
                "available": False,
                "reason": "no DFT-relaxed geometry reference imported",
            }

    if args.dataset == "lips25" and args.split in (
        "task1_relaxed",
        "task1_unrelaxed",
    ):
        metrics["formation_energy"] = _formation_energy_block(
            subset, args.split, rows, relaxation if args.relax else None
        )

    if labels["forces"] and not f_ref_all:
        metrics.setdefault("limitations", []).append(
            "force labels were reported available but no frame could be compared"
        )
    metrics.setdefault("limitations", []).extend(
        _limitations(args, entry, split_entry)
    )

    formation_ok = bool(
        metrics.get("formation_energy", {}).get("available") is True
    )
    if not rows:
        status, reason = "fail", "no_reference_structures"
    elif n_ok == 0:
        status, reason = "fail", "all_structures_failed"
    elif failures:
        status, reason = "characterized", "partial_execution"
    elif labels["energy"] or formation_ok:
        status, reason = "pass", None
    else:
        status, reason = "unsupported", "no_reference_labels"

    metrics["sampling"] = {
        "tier": (selection or {}).get("tier"),
        "n_structures": len(frames),
        "n_source_records": (
            (selection or {}).get("splits", {}).get(args.split, {})
        ).get("selected_records"),
        "statistical_convergence_claimed": False,
        "note": (
            "the reported errors describe the evaluated structures only; the "
            "core/standard selections are deterministic screening samples and "
            "no statistical convergence or error bar is claimed for them"
        ),
    }
    metrics["software_verification"] = {
        "status": "pass" if n_ok == len(frames) else "incomplete",
        "definition": (
            "all requested structures were evaluated through the mliport "
            "calculator and produced finite energies/forces; model accuracy is "
            "recorded separately and never decides this verdict"
        ),
        "n_evaluated": n_ok,
        "n_requested": len(frames),
    }

    csv_path = out_dir / f"t9_{args.dataset}_{args.split}_{args.engine}.csv"
    _write_csv(csv_path, rows)

    diagnostics: dict[str, Any] = {**ctx.diagnostics}
    if reason:
        diagnostics["reason"] = reason
        diagnostics["failures"] = failures[:20]
    diagnostics["subset_sha256"] = subset_sha
    if selection:
        diagnostics["selection_record"] = {
            "schema": selection.get("schema"),
            "tier": selection.get("tier"),
            "seed": selection.get("seed"),
            "upstream_commit": selection.get("upstream_commit"),
            "upstream_doi": selection.get("upstream_doi"),
        }

    record = common.result_record(
        case_id=f"t9_{args.dataset}",
        test_id=f"t9_{args.dataset}_{args.split}",
        status=status,
        engine=ctx.engine,
        model_identity=ctx.model_identity,
        model_sha256=ctx.model_sha256,
        task=ctx.task,
        head=ctx.head,
        dtype=ctx.dtype,
        device=common.device_record(args.device),
        input_structure_id=subset_sha,
        parameters={
            "dataset": args.dataset,
            "split": args.split,
            "subset": str(subset),
            "selection_record": str(subset.parent / "selection.json"),
            "expected_source_sha256": expected_sha,
            "comparison_mode": comparison_mode,
            "reference_level": entry["reference_level"],
            "relax": bool(args.relax),
            "fmax_eV_A": args.fmax if args.relax else None,
            "max_steps": args.max_steps if args.relax else None,
            "max_structures": args.max_structures,
            "cos_min_ref_force_eV_A": literature_metrics.COS_MIN_REF_FORCE_EV_A,
            "force_bins_eV_A": list(literature_metrics.FORCE_BINS),
            "metric_definitions": {
                "energy_per_atom": "(E_model - E_ref)/N, then MAE/RMSE/median/p95",
                "relative_energy": "per-atom energy minus group mean on both sides",
                "formation_energy": "LiPS-25 Eq. 1 along the Li2S-P2S5 tie line",
                "forces": "component and per-atom vector-norm errors, eV/Angstrom",
                "stress": "ASE Voigt 6-vector, eV/Angstrom^3 and GPa",
            },
        },
        metrics=metrics,
        diagnostics=diagnostics,
        wall_seconds=round(wall, 2),
        peak_vram=common.peak_vram_mib(),
    )
    path = common.write_result(record, out_dir, tag=args.tag)
    energy_block = metrics.get("energy_per_atom_eV", {})
    mae = energy_block.get("mae") if isinstance(energy_block, dict) else None
    mae_text = f"{mae:.4f}" if isinstance(mae, float) else "n/a"
    print(
        f"[t9] {args.dataset}/{args.split} {args.engine}: {status} "
        f"{n_ok}/{len(frames)} E/atom MAE={mae_text} eV "
        f"wall={wall:.1f}s -> {path.name}"
    )
    return 0 if status in ("pass", "characterized") else 1


def _limitations(
    args: argparse.Namespace, entry: dict[str, Any], split: dict[str, Any]
) -> list[str]:
    notes = []
    if entry["comparison_mode"] == "cross_method":
        notes.append(
            "cross-method reference: absolute energy errors are diagnostics; "
            "only forces, within-group relative energies, formation energies "
            "and rankings are interpreted"
        )
    if split.get("labels", {}).get("stress") is False:
        notes.append(
            "the reference dataset provides no verified stress label; stress "
            "accuracy is reported as unavailable rather than substituted by a "
            "virial tensor"
        )
    if split.get("mobile_elements"):
        notes.append(
            "mobile-atom force statistics are reported per element only; "
            "per-atom migration identity is not tracked because the reference "
            "frames are independent snapshots without persistent atom ids"
        )
    return notes


# ---------------------------------------------------------------------------
# LiPS-25 Task-1 formation energies
# ---------------------------------------------------------------------------


def _tie_line_reference(subset: Path, split: str) -> dict[str, dict[str, Any]]:
    """DFT-relaxed total energy (and N, x) per tie-line phase."""
    from ase.io import read as ase_read  # noqa: PLC0415

    candidate = subset.parent / "task1_relaxed.extxyz"
    if split == "task1_relaxed":
        candidate = subset
    if not candidate.exists():
        return {}
    frames = ase_read(candidate, index=":", format="extxyz")
    out: dict[str, dict[str, Any]] = {}
    for atoms in frames:
        label = str(
            atoms.info.get("literature_phase", atoms.info.get("label", ""))
        )
        key = _tie_line_key(label)
        if key is None:
            continue
        energy = atoms.info.get("dft_energy")
        if energy is None:
            energy = _reference_energy(atoms)
        if energy is None:
            continue
        out[key] = {
            "energy_eV": float(energy),
            "natoms": len(atoms),
            "x": LIPS25_TIE_LINE[key],
        }
    return out


def _formation_energy_block(
    subset: Path,
    split: str,
    rows: list[dict[str, Any]],
    relaxation: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    reference = _tie_line_reference(subset, split)
    relaxed_by_group = {
        str(r["phase"]): r for r in (relaxation or []) if "energy_eV" in r
    }
    model_energies: dict[str, float] = {}
    for row in rows:
        if not row["ok"]:
            continue
        key = _tie_line_key(str(row.get("phase", row["group"])))
        if key is None:
            continue
        if relaxation:
            relaxed = relaxed_by_group.get(str(row.get("phase", row["group"])))
            if relaxed is None:
                continue
            model_energies[key] = float(relaxed["energy_eV"])
        else:
            model_energies[key] = float(row["e_model_eV"])
    keys = sorted(set(reference) & set(model_energies), key=lambda k: LIPS25_TIE_LINE[k])
    block: dict[str, Any] = {
        "protocol": (
            "Ef/atom = (E_cell/(7-4x) - x E_Li2S^fu - (1-x) E_P2S5^fu)/(7-4x); "
            "both end points are taken from the same energy set"
        ),
        "protocol_variant": (
            "model-relaxed structures (ExpCellFilter+BFGS); the reference "
            "energies are the DFT-relaxed labels of the same phases"
            if relaxation
            else "single point on the fixed DFT-relaxed structures"
        ),
        "split": split,
        "n_structures": len(keys),
        "phases": keys,
    }
    if len(keys) < 3:
        block["available"] = False
        block["reason"] = (
            "formation energies need both end points plus at least one interior "
            "composition from the same energy set; available phases: "
            f"{keys}"
        )
        return block
    try:
        ef_ref = literature_metrics.formation_energies(
            {k: reference[k]["energy_eV"] for k in keys},
            {k: reference[k]["natoms"] for k in keys},
            {k: reference[k]["x"] for k in keys},
            LIPS25_ENDPOINT_LOW,
            LIPS25_ENDPOINT_HIGH,
        )
        ef_model = literature_metrics.formation_energies(
            {k: model_energies[k] for k in keys},
            {k: reference[k]["natoms"] for k in keys},
            {k: reference[k]["x"] for k in keys},
            LIPS25_ENDPOINT_LOW,
            LIPS25_ENDPOINT_HIGH,
        )
    except ValueError as exc:
        block["available"] = False
        block["reason"] = str(exc)
        return block
    delta = np.array([ef_model[k] - ef_ref[k] for k in keys])
    block.update(
        {
            "available": True,
            "structures": [
                {
                    "phase": k,
                    "x": LIPS25_TIE_LINE[k],
                    "n_atoms": reference[k]["natoms"],
                    "reference_eV_per_atom": ef_ref[k],
                    "model_eV_per_atom": ef_model[k],
                    "error_eV_per_atom": ef_model[k] - ef_ref[k],
                }
                for k in keys
            ],
            "rmse_eV_per_atom": float(np.sqrt((delta**2).mean())),
            "mae_eV_per_atom": float(np.abs(delta).mean()),
            "max_abs_error_eV_per_atom": float(np.abs(delta).max()),
            "rank_agreement": literature_metrics.rank_agreement(
                [ef_ref[k] for k in keys], [ef_model[k] for k in keys]
            ),
            "hull": literature_metrics.hull_agreement(
                ef_ref, ef_model, {k: LIPS25_TIE_LINE[k] for k in keys}
            ),
        }
    )
    return block


def _lattice_comparison(
    relaxation: list[dict[str, Any]], reference_relaxed: Path
) -> dict[str, Any]:
    from ase.io import read as ase_read  # noqa: PLC0415

    references = ase_read(reference_relaxed, index=":", format="extxyz")
    by_key: dict[str, Any] = {}
    for atoms in references:
        key = _tie_line_key(
            str(atoms.info.get("literature_phase", atoms.info.get("label", "")))
        )
        if key is not None:
            by_key[key] = atoms
    per_structure: list[dict[str, Any]] = []
    for relaxed in relaxation:
        key = _tie_line_key(str(relaxed.get("phase", relaxed["group"])))
        reference = by_key.get(key) if key else None
        if reference is None:
            per_structure.append(
                {
                    "record_id": relaxed["record_id"],
                    "available": False,
                    "reason": "no matching DFT-relaxed reference structure",
                }
            )
            continue
        stats = literature_metrics.lattice_stats(
            np.asarray(reference.cell.array), np.asarray(relaxed["cell"])
        )
        stats.update(
            {
                "record_id": relaxed["record_id"],
                "phase": key,
                "converged": relaxed["converged"],
                "fmax_eV_A": relaxed["fmax_eV_A"],
            }
        )
        per_structure.append(stats)
    available = [s for s in per_structure if "volume_rel_error" in s]
    return {
        "available": bool(available),
        "reference": reference_relaxed.name,
        "n_compared": len(available),
        "volume_rel_error_mae": float(
            np.mean([abs(s["volume_rel_error"]) for s in available])
        )
        if available
        else None,
        "lattice_length_mae_A": float(
            np.mean([s["lattice_length_mae_A"] for s in available])
        )
        if available
        else None,
        "structures": per_structure,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "record_id",
        "group",
        "natoms",
        "ok",
        "e_ref_eV",
        "e_model_eV",
        "de_per_atom_eV",
        "f_component_mae_eV_A",
        "f_vector_norm_p95_eV_A",
        "f_cosine_mean",
        "f_ref_norm_max_eV_A",
        "s_component_mae_eV_A3",
        "converged",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_failure_record(args: argparse.Namespace, exc: BaseException) -> None:
    """Record an execution crash as ``blocked`` instead of losing it.

    A model that cannot be loaded, a backend that raises, or a missing
    capability are *execution* problems: they must appear in the evidence as
    ``blocked`` with the reason, never as a silent absence or a fake
    scientific failure.  This is best effort and never masks the exit code.
    """
    try:
        profile = json.loads(Path(args.manifest).read_text(encoding="utf-8"))[
            "profiles"
        ].get(args.profile_id, {})
    except Exception:  # noqa: BLE001 - the manifest may itself be broken
        profile = {}
    reason = f"{type(exc).__name__}: {exc}"[:400]
    try:
        record = common.result_record(
            case_id=f"t9_{args.dataset}",
            test_id=f"t9_{args.dataset}_{args.split}",
            status="blocked",
            engine=args.engine,
            model_identity=profile.get(
                "identity", profile.get("upstream_model_id", "unknown")
            ),
            model_sha256=profile.get("model_sha256", "0" * 64),
            task=profile.get("task"),
            head=args.head or profile.get("head"),
            dtype=args.dtype or profile.get("dtype", "unknown"),
            device=common.device_record(args.device),
            input_structure_id=None,
            parameters={
                "dataset": args.dataset,
                "split": args.split,
                "subset": str(args.subset),
            },
            metrics={},
            diagnostics={"reason": reason, "profile_id": args.profile_id},
            exception=reason,
            inference_mode="literature",
        )
        common.write_result(record, Path(args.out), tag=args.tag)
    except Exception:  # noqa: BLE001 - never mask the original failure
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=engines._ENGINES)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(literature_datasets.load_registry()["datasets"]),
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--tag", default="t9")
    parser.add_argument("--max-structures", type=int, default=None)
    parser.add_argument(
        "--relax",
        action="store_true",
        help=(
            "relax each structure with ExpCellFilter+BFGS before scoring "
            "(method-specific relaxation protocol)"
        ),
    )
    parser.add_argument(
        "--fmax", type=float, default=0.02, help="relaxation fmax, eV/Angstrom"
    )
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    try:
        return run_engine(args)
    except Exception as exc:  # noqa: BLE001 - record the crash and fail loudly
        traceback.print_exc()
        _write_failure_record(args, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
