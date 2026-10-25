"""Ranking, saturation diagnostics, refinement and DFT exports (spec section 10).

Everything here is a *screening* product: the outputs say "lowest found", never
"global minimum", and every number carries the provenance needed to re-rank the
candidates with DFT (spec sections 10.2, 15).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write as ase_write

from mliport.confsearch import status as cs_status
from mliport.confsearch.graphs import covalent_bonds, classify_minimum, ionic_mask
from mliport.confsearch.quench import refine
from mliport.confsearch.registry import MinimumRecord, MinimumRegistry, UniqueMinimum
from mliport.writers.contcar import ContcarWriter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mliport.confsearch.graphs import ClassificationReference
    from mliport.confsearch.options import ConfsearchOptions
    from mliport.confsearch.regions import Regions

RESULTS_SCHEMA = "mliport.confsearch-results/1"
#: Saturation window for the "new low minima in the last quarter" diagnostic.
NEW_LOW_WINDOW_EV = 0.2


@dataclass(frozen=True, slots=True)
class RefinementOutcome:
    """Refinement result for one representative."""

    unique_id: int
    status: str
    energy_eV: float
    fmax_eV_A: float
    steps: int
    atoms: Atoms
    classification: Any
    converged: bool

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "energy_eV": float(self.energy_eV),
            "fmax_eV_A": float(self.fmax_eV_A),
            "steps": int(self.steps),
            "converged": bool(self.converged),
            "classification": self.classification.to_dict(),
        }


def _index_ranges(indices: Sequence[int]) -> str:
    """Compress 1-based indices into CP2K ``a..b`` ranges."""
    ordered = sorted(set(int(index) for index in indices))
    if not ordered:
        return ""
    parts: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        parts.append(str(start) if start == previous else f"{start}..{previous}")
        start = previous = index
    parts.append(str(start) if start == previous else f"{start}..{previous}")
    return " ".join(parts)


def make_whole_adsorbate(
    atoms: Atoms, regions: Regions, options: ConfsearchOptions
) -> Atoms:
    """Wrap the slab and make the adsorbate whole (spec section 10.5).

    Slab atoms are wrapped into the cell.  The non-ionic adsorbate is rebuilt by
    a breadth-first walk over the covalent graph under MIC, each new atom placed
    at the image closest to its parent; ionic adsorbate atoms are then placed at
    the image closest to their nearest non-ionic adsorbate atom.
    """
    work = atoms.copy()
    work.wrap()
    cell = np.asarray(work.cell, dtype=float)
    inverse = np.linalg.inv(cell)
    positions = np.asarray(work.get_positions(), dtype=float)
    adsorbate = np.asarray(regions.adsorbate, dtype=int)
    ionic = ionic_mask(work, options)
    non_ionic = [int(index) for index in adsorbate if not ionic[index]]

    if non_ionic:
        graph = covalent_bonds(
            work,
            np.asarray(non_ionic, dtype=int),
            bond_scale=options.cs_bond_scale,
            ionic_symbols=options.ionic_symbols,
        )
        adjacency: dict[int, set[int]] = {index: set() for index in non_ionic}
        for first, second in graph.bonds:
            adjacency[int(first)].add(int(second))
            adjacency[int(second)].add(int(first))
        start = non_ionic[0]
        visited = {start}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbour in sorted(adjacency[current]):
                if neighbour in visited:
                    continue
                delta = positions[neighbour] - positions[current]
                fractional = delta @ inverse
                fractional -= np.round(fractional)
                positions[neighbour] = positions[current] + fractional @ cell
                visited.add(neighbour)
                queue.append(neighbour)
        reference = np.asarray(non_ionic, dtype=int)
    else:
        reference = adsorbate

    for index in adsorbate:
        if not ionic[index]:
            continue
        deltas = positions[reference] - positions[index]
        fractional = deltas @ inverse
        fractional -= np.round(fractional)
        candidates = positions[index] + fractional @ cell
        nearest = int(np.argmin(np.linalg.norm(candidates - positions[index], axis=1)))
        positions[index] = candidates[nearest]
    work.positions = positions
    return work


def _ranking_entry(
    unique: UniqueMinimum,
    *,
    best_energy: float,
    n_atoms: int,
    refinement: RefinementOutcome | None,
    total_quenches: int,
) -> dict:
    representative = unique.representative
    energy = refinement.energy_eV if refinement is not None else unique.energy_eV
    first_found_fraction = (
        representative.quench_index / total_quenches if total_quenches else 0.0
    )
    return {
        "id": unique.unique_id,
        "status": unique.status,
        "energy_eV": float(energy),
        "delta_energy_eV": float(energy - best_energy),
        "delta_energy_per_atom_eV": float((energy - best_energy) / max(1, n_atoms)),
        "hit_count": unique.hit_count,
        "replicas": sorted(unique.replicas),
        "n_replicas": len(unique.replicas),
        "first_found_md_ps": float(unique.first_found_md_ps),
        "first_found_fraction": float(first_found_fraction),
        "flags": list(unique.flags),
        "chemisorbed": representative.classification.chemisorbed,
        "refined": refinement is not None,
        "refinement_status": None if refinement is None else refinement.status,
        "quench_energy_eV": float(unique.energy_eV),
        "fmax_eV_A": float(
            refinement.fmax_eV_A if refinement is not None else representative.fmax_eV_A
        ),
        "classification": representative.classification.to_dict(),
        "origin": representative.origin.to_dict(),
    }


def _export_candidate(
    directory: Path,
    *,
    unique: UniqueMinimum,
    atoms: Atoms,
    options: ConfsearchOptions,
    regions: Regions,
    entry: dict,
    energy: float,
) -> dict:
    """Write the DFT-facing files for one exported rank (spec section 10.5)."""
    directory.mkdir(parents=True, exist_ok=True)
    whole = make_whole_adsorbate(atoms, regions, options)

    constrained = whole.copy()
    fixed = np.asarray(regions.fixed, dtype=int)
    constrained.set_constraint(FixAtoms(indices=fixed) if fixed.size else None)
    ContcarWriter().write(
        constrained,
        directory / "POSCAR",
        comment=f"confsearch rank {unique.unique_id} E={energy:.6f} eV",
        direct=True,
    )

    export_atoms = whole.copy()
    export_atoms.info["confsearch_id"] = int(unique.unique_id)
    export_atoms.info["confsearch_status"] = unique.status
    export_atoms.info["energy_eV"] = float(energy)
    export_atoms.info["flags"] = list(unique.flags)
    ase_write(directory / "structure.extxyz", export_atoms, format="extxyz")
    ase_write(directory / "cp2k_coord.xyz", export_atoms, format="xyz")

    cell = np.asarray(whole.cell, dtype=float)
    (directory / "cp2k_cell.inc").write_text(
        "".join(
            f"{name} {vector[0]:.12f} {vector[1]:.12f} {vector[2]:.12f}\n"
            for name, vector in zip("ABC", cell, strict=True)
        ),
        encoding="utf-8",
    )
    (directory / "cp2k_fixed_atoms.inc").write_text(
        "# CP2K &MOTION/&CONSTRAINT/&FIXED_ATOMS list, 1-based indices\n"
        f"LIST {_index_ranges([int(index) + 1 for index in fixed])}\n".lstrip()
        if fixed.size
        else "# CP2K &MOTION/&CONSTRAINT/&FIXED_ATOMS list: no fixed atoms\n",
        encoding="utf-8",
    )

    payload = unique.to_dict()
    payload["export"] = {
        "rank_energy_eV": float(energy),
        "ranking": entry,
        "coordinate_convention": (
            "slab wrapped into the cell; adsorbate made whole under MIC"
        ),
    }
    (directory / "minimum.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = [
        "POSCAR",
        "structure.extxyz",
        "cp2k_coord.xyz",
        "cp2k_cell.inc",
        "cp2k_fixed_atoms.inc",
        "minimum.json",
    ]
    return {
        "rank": unique.unique_id,
        "path": directory.name,
        "energy_eV": float(energy),
        "files": files,
    }


def _saturation(
    registry: MinimumRegistry,
    *,
    rankable: list[tuple[UniqueMinimum, float]],
    total_quenches: int,
) -> dict:
    if not rankable:
        return {
            "best_hit_count": 0,
            "best_replicas": [],
            "best_first_found_fraction": None,
            "best_repeatedly_found": False,
            "new_low_minima_last_quarter": 0,
            "lowest_found_language": "lowest found (no global-minimum claim)",
        }
    best = rankable[0][0]
    representative = best.representative
    fraction = representative.quench_index / total_quenches if total_quenches else 0.0
    cutoff = int(math.ceil(0.75 * total_quenches))
    recent = 0
    for unique, energy in rankable:
        if energy - rankable[0][1] > NEW_LOW_WINDOW_EV:
            continue
        if unique.first_found_quench_index > cutoff:
            recent += 1
    return {
        "best_id": best.unique_id,
        "best_energy_eV": float(rankable[0][1]),
        "best_hit_count": best.hit_count,
        "best_replicas": sorted(best.replicas),
        "best_first_found_md_ps": float(best.first_found_md_ps),
        "best_first_found_fraction": float(fraction),
        "best_repeatedly_found": bool(best.hit_count >= 3 and len(best.replicas) >= 2),
        "new_low_minima_last_quarter": int(recent),
        "lowest_found_language": "lowest found (no global-minimum claim)",
    }


def _status_warnings(
    registry: MinimumRegistry, options: ConfsearchOptions, ranked: list[dict]
) -> list[str]:
    warnings: list[str] = []
    all_records = registry.total_records()
    if not all_records:
        return warnings
    classes = registry.status_classes()
    reacted = sum(unique.hit_count for unique in classes.get(cs_status.REACTED, []))
    if reacted / all_records > 0.10:
        warnings.append(
            f"{100 * reacted / all_records:.1f}% of quenches are reacted; the "
            "temperature may be too high or the model unreliable for this chemistry"
        )
    unusable = sum(
        unique.hit_count
        for status in (cs_status.DESORBED, cs_status.IMAGE_CONTACT)
        for unique in classes.get(status, [])
    )
    if unusable / all_records > 0.10:
        warnings.append(
            f"{100 * unusable / all_records:.1f}% of quenches are desorbed or in "
            "image contact; check the cell vacuum and the placement scales"
        )
    not_converged = sum(
        unique.hit_count for unique in classes.get(cs_status.QUENCH_NOT_CONVERGED, [])
    )
    if not_converged / all_records > 0.05:
        warnings.append(
            f"{100 * not_converged / all_records:.1f}% of quenches did not converge "
            "within cs_quench_steps"
        )
    for entry in ranked:
        if "far_field_slab_changed" in entry["flags"]:
            warnings.append(
                f"top-{options.cs_top_k} minimum {entry['id']} has "
                "far_field_slab_changed: its energy difference may reflect slab "
                "changes rather than adsorption"
            )
    return warnings


def build_results(
    *,
    registry: MinimumRegistry,
    options: ConfsearchOptions,
    replicas: Sequence[Any],
    discovery: list[dict],
    baseline: dict,
    warnings: list[str],
    output_dir: Path,
    model_record: dict,
    gradient_preflight: dict,
    cumulative_md_ps: float,
    seed: int,
    starts: list[dict],
    regions: Regions | None = None,
    references: Sequence[ClassificationReference] | None = None,
    calculator=None,
) -> dict:
    """Write every result artifact and return the ``search_results.json`` body."""
    output_dir = Path(output_dir)
    (output_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    _write_discovery_csv(output_dir / "diagnostics" / "discovery.csv", discovery)

    rankable_statuses = cs_status.rankable_statuses(options.cs_reaction_policy)
    rankable = sorted(
        (
            (unique, unique.energy_eV)
            for status in rankable_statuses
            for unique in registry.uniques(status)
        ),
        key=lambda item: (item[1], item[0].unique_id),
    )

    refinements: dict[int, RefinementOutcome] = {}
    if (
        options.cs_refine
        and rankable
        and calculator is not None
        and regions is not None
    ):
        refinements = _refine_top_k(
            rankable,
            options=options,
            regions=regions,
            references=references or [],
            replicas=replicas,
            calculator=calculator,
        )

    ranked_energies = {
        unique.unique_id: (
            refinements[unique.unique_id].energy_eV
            if unique.unique_id in refinements
            else energy
        )
        for unique, energy in rankable
    }
    best_energy = min(ranked_energies.values()) if ranked_energies else 0.0
    total_quenches = registry.total_records()
    n_atoms = len(rankable[0][0].representative.atoms) if rankable else 0
    entries = [
        _ranking_entry(
            unique,
            best_energy=best_energy,
            n_atoms=n_atoms,
            refinement=refinements.get(unique.unique_id),
            total_quenches=total_quenches,
        )
        for unique, _energy in rankable
    ]
    refined_entries = [entry for entry in entries if entry["refined"]]
    unrefined_entries = [entry for entry in entries if not entry["refined"]]

    baseline_report = _baseline_report(baseline, entries, options)

    exports: list[dict] = []
    if regions is not None:
        window = [
            (unique, ranked_energies[unique.unique_id])
            for unique, _energy in rankable
            if ranked_energies[unique.unique_id] - best_energy
            <= options.cs_export_window
        ][: int(options.cs_top_k)]
        for position, (unique, energy) in enumerate(window, start=1):
            atoms = (
                refinements[unique.unique_id].atoms
                if unique.unique_id in refinements
                else unique.representative.atoms
            )
            exports.append(
                _export_candidate(
                    output_dir / "top" / f"rank_{position:03d}",
                    unique=unique,
                    atoms=atoms,
                    options=options,
                    regions=regions,
                    entry=entries[
                        next(
                            index
                            for index, item in enumerate(entries)
                            if item["id"] == unique.unique_id
                        )
                    ],
                    energy=energy,
                )
            )

    saturation = _saturation(
        registry,
        rankable=[
            (unique, ranked_energies[unique.unique_id]) for unique, _ in rankable
        ],
        total_quenches=total_quenches,
    )
    warnings = list(warnings) + _status_warnings(registry, options, entries)
    warnings = list(dict.fromkeys(warnings))

    status = "completed"
    if not any(getattr(replica, "status", None) == "active" for replica in replicas):
        status = "failed"

    results = {
        "schema": RESULTS_SCHEMA,
        "status": status,
        "scientific_revision": _scientific_revision(),
        "seed": int(seed),
        "settings": {
            "cs_mode": options.cs_mode,
            "cs_reaction_policy": options.cs_reaction_policy,
            "cs_refine": bool(options.cs_refine),
            "cs_dedup_energy_tol": float(options.cs_dedup_energy_tol),
            "cs_dedup_geom_tol": float(options.cs_dedup_geom_tol),
            "cs_dedup_fp_tol": float(options.cs_dedup_fp_tol),
            "cs_fp_sigma": float(options.cs_fp_sigma),
        },
        "ranking": {
            "best_energy_eV": float(best_energy) if rankable else None,
            "n_rankable": len(entries),
            "refined": refined_entries,
            "unrefined": unrefined_entries,
        },
        "diagnostics": {
            "discovery_curve": "diagnostics/discovery.csv",
            "n_quenches": total_quenches,
            "cumulative_md_ps": float(cumulative_md_ps),
            "saturation": saturation,
            "registry": registry.to_dict(),
        },
        "baseline": baseline_report,
        "exports": exports,
        "warnings": warnings,
        "starts": starts,
        "model": model_record,
        "gradient_preflight": gradient_preflight,
        "refinement": {
            "enabled": bool(options.cs_refine),
            "fmax_eV_A": float(options.cs_refine_fmax),
            "steps": int(options.cs_refine_steps),
            "n_refined": len(refinements),
        },
    }
    (output_dir / "search_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_artifacts(output_dir, results)
    return results


def _scientific_revision() -> int:
    from mliport.confsearch.revisions import CS_SCIENTIFIC_REVISION  # noqa: PLC0415

    return CS_SCIENTIFIC_REVISION


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _write_discovery_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "cumulative_md_ps",
        "quench_index",
        "unique_valid",
        "unique_valid_within_0p2eV",
        "current_best_eV",
        "energy_eV",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _refine_top_k(
    rankable: list[tuple[UniqueMinimum, float]],
    *,
    options: ConfsearchOptions,
    regions: Regions,
    references: Sequence[ClassificationReference],
    replicas: Sequence[Any],
    calculator,
) -> dict[int, RefinementOutcome]:
    best = rankable[0][1]
    window = [
        (unique, energy)
        for unique, energy in rankable
        if energy - best <= options.cs_export_window
    ][: int(options.cs_top_k)]
    outcomes: dict[int, RefinementOutcome] = {}
    for unique, _energy in window:
        representative = unique.representative
        result = refine(
            representative.atoms,
            calculator,
            options,
            fixed_indices=regions.fixed,
        )
        reference = _reference_for_record(representative, references, replicas)
        classification = classify_minimum(
            result.atoms, regions=regions, options=options, reference=reference
        )
        outcomes[unique.unique_id] = RefinementOutcome(
            unique_id=unique.unique_id,
            status=classification.status if result.usable else result.status,
            energy_eV=float(result.energy_eV),
            fmax_eV_A=float(result.fmax_eV_A),
            steps=int(result.steps),
            atoms=result.atoms,
            classification=classification,
            converged=bool(result.converged),
        )
    return outcomes


def _reference_for_record(
    record: MinimumRecord,
    references: Sequence[ClassificationReference],
    replicas: Sequence[Any],
) -> ClassificationReference:
    """Pick the reference of the replica the record came from."""
    if references and len(references) == len(replicas):
        index = int(record.origin.replica)
        if 0 <= index < len(references):
            return references[index]
    if references:
        return references[0]
    raise ValueError("No classification reference available for refinement")


def _baseline_report(
    baseline: dict, entries: list[dict], options: ConfsearchOptions
) -> dict:
    best_baseline = baseline.get("best_energy_eV")
    report = dict(baseline)
    if best_baseline is None or not entries:
        report["best_energy_eV"] = best_baseline
        report["top_k_never_reached_by_start_quench"] = None
        return report
    top_k = entries[: int(options.cs_top_k)]
    never = [
        entry["id"]
        for entry in top_k
        if entry["origin"]["source"] != "start"
        and entry["energy_eV"] < best_baseline - 1e-12
    ]
    report["top_k_never_reached_by_start_quench"] = never
    return report


def _write_artifacts(output_dir: Path, results: dict) -> None:
    manifest: dict[str, dict] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifacts.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        manifest[relative] = {
            "bytes": path.stat().st_size,
            "format": _format_for(path),
            "units": "eV, Angstrom, fs, K",
        }
    payload = {
        "schema": "mliport.confsearch-artifacts/1",
        "status": results["status"],
        "files": manifest,
    }
    (output_dir / "artifacts.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _format_for(path: Path) -> str:
    return {
        ".json": "json",
        ".csv": "csv",
        ".extxyz": "extxyz",
        ".xyz": "xyz",
        ".traj": "ase-traj",
        ".inc": "cp2k-include",
        ".log": "text",
    }.get(path.suffix, "text")


__all__ = [
    "RESULTS_SCHEMA",
    "RefinementOutcome",
    "build_results",
    "make_whole_adsorbate",
]
