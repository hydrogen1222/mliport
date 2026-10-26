"""Ranking, saturation diagnostics, refinement and DFT exports (spec section 10).

Everything here is a *screening* product: the outputs say "lowest found", never
"global minimum", and every number carries the provenance needed to re-rank the
candidates with DFT (spec sections 10.2, 15).

Ranking rules enforced here (spec 10.1/10.3 revision 4, review of PR #64):

* a refinement replaces the quench result only when it converged **and** its
  classification is still rankable; an unconverged refinement keeps the quench
  result and is flagged, and a refinement that changed to a non-rankable status
  leaves the ranking with a record saying why;
* the refined set is re-deduplicated with the section 9.3 rule, and the merged
  entry keeps the summed hit counts of its sources;
* refined and unrefined energies are never interleaved: each table has its own
  best energy and its own export window, and the exports come from the refined
  table whenever it exists.
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
from mliport.confsearch.registry import (
    MinimumRecord,
    MinimumRegistry,
    UniqueMinimum,
)
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


# ---------------------------------------------------------------------------
# ranking entries
# ---------------------------------------------------------------------------


def _refined_entry(
    group: dict, *, best_energy: float, n_atoms: int, total_quenches: int
) -> dict:
    """Ranking entry for a re-deduplicated refined minimum."""
    energy = float(group["energy_eV"])
    return {
        "id": int(group["ids"][0]),
        "merged_ids": [int(value) for value in group["ids"]],
        "status": str(group["status"]),
        "energy_eV": energy,
        "delta_energy_eV": float(energy - best_energy),
        "delta_energy_per_atom_eV": float((energy - best_energy) / max(1, n_atoms)),
        "hit_count": int(group["hit_count"]),
        "replicas": sorted(int(value) for value in group["replicas"]),
        "n_replicas": len(group["replicas"]),
        "first_found_md_ps": float(group["first_found_md_ps"]),
        "first_found_fraction": float(
            group["first_found_quench_index"] / total_quenches
            if total_quenches
            else 0.0
        ),
        "first_found_quench_index": int(group["first_found_quench_index"]),
        "flags": list(group["flags"]),
        "chemisorbed": bool(group["classification"].chemisorbed),
        "refined": True,
        "refinement_status": str(group["status"]),
        "refinement_failed": False,
        "quench_energy_eV": float(group["quench_energy_eV"]),
        "fmax_eV_A": float(group["fmax_eV_A"]),
        "classification": group["classification"].to_dict(),
        "origin": group["origin"].to_dict(),
    }


def _unrefined_entry(
    unique: UniqueMinimum,
    *,
    best_energy: float,
    n_atoms: int,
    total_quenches: int,
    refinement: RefinementOutcome | None = None,
) -> dict:
    """Ranking entry for a quench result (no usable refinement)."""
    representative = unique.representative
    energy = float(unique.energy_eV)
    return {
        "id": int(unique.unique_id),
        "merged_ids": [int(unique.unique_id)],
        "status": unique.status,
        "energy_eV": energy,
        "delta_energy_eV": float(energy - best_energy),
        "delta_energy_per_atom_eV": float((energy - best_energy) / max(1, n_atoms)),
        "hit_count": int(unique.hit_count),
        "replicas": sorted(int(value) for value in unique.replicas),
        "n_replicas": len(unique.replicas),
        "first_found_md_ps": float(unique.first_found_md_ps),
        "first_found_fraction": float(
            unique.first_found_quench_index / total_quenches if total_quenches else 0.0
        ),
        "first_found_quench_index": int(unique.first_found_quench_index),
        "flags": list(unique.flags),
        "chemisorbed": representative.classification.chemisorbed,
        "refined": False,
        "refinement_status": None if refinement is None else refinement.status,
        "refinement_failed": bool(refinement is not None and not refinement.converged),
        "quench_energy_eV": energy,
        "fmax_eV_A": float(representative.fmax_eV_A),
        "classification": representative.classification.to_dict(),
        "origin": representative.origin.to_dict(),
    }


def _dedup_refined(
    accepted: dict[int, RefinementOutcome],
    *,
    quench_rankable: list[tuple[UniqueMinimum, float]],
    options: ConfsearchOptions,
    regions: Regions,
) -> list[dict]:
    """Re-deduplicate the refined set with the section 9.3 rule (spec 10.3).

    A temporary registry applies the same ``dE`` AND (``G`` OR ``D``) matching to
    the refined representatives in ascending refined energy; the merged group
    keeps the summed hit counts, the union of replicas and the earliest
    first-found record of its sources.
    """
    by_id = {unique.unique_id: unique for unique, _energy in quench_rankable}
    scratch = MinimumRegistry(
        options=options, adsorbate=regions.adsorbate, fixed=regions.fixed
    )
    groups: dict[int, dict] = {}
    ordered = sorted(accepted.items(), key=lambda item: (item[1].energy_eV, item[0]))
    for unique_id, outcome in ordered:
        source = by_id[unique_id]
        record = MinimumRecord(
            record_id=scratch.new_record_id(),
            energy_eV=float(outcome.energy_eV),
            atoms=outcome.atoms,
            classification=outcome.classification,
            origin=source.representative.origin,
            quench_index=int(source.first_found_quench_index),
            cumulative_md_ps=float(source.first_found_md_ps),
            fmax_eV_A=float(outcome.fmax_eV_A),
            steps=int(outcome.steps),
        )
        registered = scratch.register(record)
        group = groups.get(registered.unique_id)
        if group is None:
            groups[registered.unique_id] = {
                "energy_eV": float(outcome.energy_eV),
                "quench_energy_eV": float(source.energy_eV),
                "atoms": outcome.atoms,
                "classification": outcome.classification,
                "fmax_eV_A": float(outcome.fmax_eV_A),
                "steps": int(outcome.steps),
                "ids": [int(unique_id)],
                "hit_count": int(source.hit_count),
                "replicas": {int(value) for value in source.replicas},
                "first_found_md_ps": float(source.first_found_md_ps),
                "first_found_quench_index": int(source.first_found_quench_index),
                "status": str(outcome.status),
                # Spec 10.3 revision 6: a refined entry's flags come from the
                # refined classification (chemisorbed/physisorbed and the
                # far-field slab flag can all change during refinement).
                "flags": tuple(outcome.classification.flags),
                "origin": source.representative.origin,
            }
            continue
        group["ids"].append(int(unique_id))
        group["hit_count"] += int(source.hit_count)
        group["replicas"] |= {int(value) for value in source.replicas}
        group["first_found_md_ps"] = min(
            group["first_found_md_ps"], float(source.first_found_md_ps)
        )
        group["first_found_quench_index"] = min(
            group["first_found_quench_index"], int(source.first_found_quench_index)
        )
    return [groups[key] for key in sorted(groups)]


def _refined_payload(group: dict, energy: float) -> dict:
    """``minimum.json`` body for a refined (possibly merged) entry."""
    atoms = group["atoms"]
    return {
        "id": int(group["ids"][0]),
        "merged_ids": [int(value) for value in group["ids"]],
        "status": str(group["status"]),
        "energy_eV": float(energy),
        "flags": list(group["flags"]),
        "hit_count": int(group["hit_count"]),
        "replicas": sorted(int(value) for value in group["replicas"]),
        "first_found_md_ps": float(group["first_found_md_ps"]),
        "first_found_quench_index": int(group["first_found_quench_index"]),
        "representative": {
            "record_id": None,
            "energy_eV": float(energy),
            "fmax_eV_A": float(group["fmax_eV_A"]),
            "steps": int(group["steps"]),
            "origin": group["origin"].to_dict(),
            "numbers": [int(z) for z in atoms.numbers],
            "symbols": atoms.get_chemical_symbols(),
            "cell_A": np.asarray(atoms.cell, dtype=float).tolist(),
            "pbc": [bool(value) for value in atoms.pbc],
            "positions_A": np.asarray(atoms.get_positions(), dtype=float).tolist(),
            "classification": group["classification"].to_dict(),
        },
        "members": [],
    }


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------


def _export_candidate(
    directory: Path,
    *,
    unique_id: int,
    payload: dict,
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
        comment=f"confsearch rank {unique_id} E={energy:.6f} eV",
        direct=True,
    )

    export_atoms = whole.copy()
    export_atoms.info["confsearch_id"] = int(unique_id)
    export_atoms.info["confsearch_status"] = entry["status"]
    export_atoms.info["energy_eV"] = float(energy)
    export_atoms.info["flags"] = list(entry["flags"])
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

    body = dict(payload)
    body["export"] = {
        "rank_energy_eV": float(energy),
        "ranking": entry,
        "coordinate_convention": (
            "slab wrapped into the cell; adsorbate made whole under MIC"
        ),
    }
    (directory / "minimum.json").write_text(
        json.dumps(body, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
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
        "rank": int(unique_id),
        "path": directory.name,
        "energy_eV": float(energy),
        "refined": bool(entry.get("refined")),
        "refinement_failed": bool(entry.get("refinement_failed")),
        "files": files,
    }


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def _saturation(entries: list[dict], *, total_quenches: int) -> dict:
    if not entries:
        return {
            "best_hit_count": 0,
            "best_replicas": [],
            "best_first_found_fraction": None,
            "best_repeatedly_found": False,
            "new_low_minima_last_quarter": 0,
            "lowest_found_language": "lowest found (no global-minimum claim)",
        }
    best = entries[0]
    best_energy = float(best["energy_eV"])
    fraction = (
        float(best["first_found_quench_index"]) / total_quenches
        if total_quenches
        else 0.0
    )
    cutoff = int(math.ceil(0.75 * total_quenches))
    recent = sum(
        1
        for entry in entries
        if float(entry["energy_eV"]) - best_energy <= NEW_LOW_WINDOW_EV
        and int(entry["first_found_quench_index"]) > cutoff
    )
    return {
        "best_id": int(best["id"]),
        "best_energy_eV": best_energy,
        "best_hit_count": int(best["hit_count"]),
        "best_replicas": list(best["replicas"]),
        "best_first_found_md_ps": float(best["first_found_md_ps"]),
        "best_first_found_fraction": fraction,
        "best_repeatedly_found": bool(
            int(best["hit_count"]) >= 3 and len(best["replicas"]) >= 2
        ),
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
    for entry in ranked[: int(options.cs_top_k)]:
        if "far_field_slab_changed" in entry["flags"]:
            warnings.append(
                f"top-{options.cs_top_k} minimum {entry['id']} has "
                "far_field_slab_changed: its energy difference may reflect slab "
                "changes rather than adsorption"
            )
    return warnings


def _baseline_report(
    baseline: dict,
    top_entries: list[dict],
    *,
    uniques_by_id: dict[int, UniqueMinimum],
    options: ConfsearchOptions,
) -> dict:
    """Baseline comparison (spec 10.4).

    ``top_k_never_reached_by_start_quench`` lists the top-K minima that have **no
    member** whose origin is a start quench -- not merely a representative that
    did not come from a start.
    """
    report = dict(baseline)
    best_baseline = baseline.get("best_energy_eV")
    if best_baseline is None or not top_entries:
        report["top_k_never_reached_by_start_quench"] = None
        return report
    never: list[int] = []
    for entry in top_entries[: int(options.cs_top_k)]:
        ids = entry.get("merged_ids") or [entry["id"]]
        reached = False
        for unique_id in ids:
            unique = uniques_by_id.get(int(unique_id))
            if unique is None:
                continue
            if any(member.origin.source == "start" for member in unique.members):
                reached = True
                break
        if not reached:
            never.append(int(entry["id"]))
    report["top_k_never_reached_by_start_quench"] = never
    return report


# ---------------------------------------------------------------------------
# results assembly
# ---------------------------------------------------------------------------


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
    _write_minima(output_dir, registry)

    rankable_statuses = cs_status.rankable_statuses(options.cs_reaction_policy)
    quench_rankable = sorted(
        (
            (unique, unique.energy_eV)
            for status in rankable_statuses
            for unique in registry.uniques(status)
        ),
        key=lambda item: (item[1], item[0].unique_id),
    )
    total_quenches = registry.total_records()
    n_atoms = len(quench_rankable[0][0].representative.atoms) if quench_rankable else 0
    uniques_by_id = {unique.unique_id: unique for unique, _ in quench_rankable}

    refinements: dict[int, RefinementOutcome] = {}
    if (
        options.cs_refine
        and quench_rankable
        and calculator is not None
        and regions is not None
    ):
        refinements = _refine_top_k(
            quench_rankable,
            options=options,
            regions=regions,
            references=references or [],
            replicas=replicas,
            calculator=calculator,
        )

    # Spec 10.1/10.3: only a converged refinement whose status is still rankable
    # replaces the quench result.  Everything else is recorded with its reason.
    accepted: dict[int, RefinementOutcome] = {}
    removed: set[int] = set()
    refinement_notes: list[dict] = []
    for unique, _quench_energy in quench_rankable:
        outcome = refinements.get(unique.unique_id)
        if outcome is None:
            continue
        if not outcome.converged:
            refinement_notes.append(
                {
                    "id": unique.unique_id,
                    "reason": "refinement_not_converged",
                    "refinement_status": outcome.status,
                    "kept": "quench_result",
                }
            )
            continue
        if outcome.status not in rankable_statuses:
            # The refinement is the tighter classification: an entry that is no
            # longer rankable leaves the ranking entirely (spec 10.1).
            removed.add(unique.unique_id)
            refinement_notes.append(
                {
                    "id": unique.unique_id,
                    "reason": "refinement_changed_status",
                    "refinement_status": outcome.status,
                    "kept": "none (no longer rankable)",
                    "energy_eV": float(outcome.energy_eV),
                }
            )
            continue
        accepted[unique.unique_id] = outcome

    refined_groups = (
        _dedup_refined(
            accepted,
            quench_rankable=quench_rankable,
            options=options,
            regions=regions,
        )
        if accepted and regions is not None
        else []
    )

    refined_best = min((group["energy_eV"] for group in refined_groups), default=None)
    unrefined_candidates = [
        (unique, energy)
        for unique, energy in quench_rankable
        if unique.unique_id not in accepted and unique.unique_id not in removed
    ]
    unrefined_best = min(
        (energy for _unique, energy in unrefined_candidates), default=None
    )

    refined_entries = [
        _refined_entry(
            group,
            best_energy=float(refined_best),
            n_atoms=n_atoms,
            total_quenches=total_quenches,
        )
        for group in sorted(
            refined_groups, key=lambda group: (group["energy_eV"], group["ids"][0])
        )
    ]
    unrefined_entries = [
        _unrefined_entry(
            unique,
            best_energy=float(unrefined_best),
            n_atoms=n_atoms,
            total_quenches=total_quenches,
            refinement=refinements.get(unique.unique_id),
        )
        for unique, _energy in sorted(
            unrefined_candidates, key=lambda item: (item[1], item[0].unique_id)
        )
    ]

    # Never interleave energies at different fmax: exports come from the refined
    # table when it exists, and the window is relative to that table's best.
    export_groups: dict[int, dict] = {}
    if refined_entries:
        export_source = refined_entries
        export_groups = {int(group["ids"][0]): group for group in refined_groups}
        best_export: float | None = float(refined_best)
    else:
        export_source = unrefined_entries
        best_export = float(unrefined_best) if unrefined_best is not None else None

    exports: list[dict] = []
    if regions is not None and export_source and best_export is not None:
        window = [
            entry
            for entry in export_source
            if float(entry["energy_eV"]) - best_export <= options.cs_export_window
        ][: int(options.cs_top_k)]
        for position, entry in enumerate(window, start=1):
            unique_id = int(entry["id"])
            if unique_id in export_groups:
                group = export_groups[unique_id]
                atoms = group["atoms"]
                payload = _refined_payload(group, float(entry["energy_eV"]))
            else:
                unique = uniques_by_id[unique_id]
                atoms = unique.representative.atoms
                payload = unique.to_dict()
            exports.append(
                _export_candidate(
                    output_dir / "top" / f"rank_{position:03d}",
                    unique_id=unique_id,
                    payload=payload,
                    atoms=atoms,
                    options=options,
                    regions=regions,
                    entry=entry,
                    energy=float(entry["energy_eV"]),
                )
            )

    # Spec 10.3 revision 6: a candidate in the refinement window whose
    # refinement did not converge must still reach the DFT deliverable.  It is
    # exported with its quench geometry and energy as top/unrefined_NNN/ and a
    # warning names it -- a failed refinement must never silently remove a top
    # candidate.
    unrefined_by_id = {int(entry["id"]): entry for entry in unrefined_entries}
    unconverged_ids = sorted(
        (
            int(unique_id)
            for unique_id, outcome in refinements.items()
            if not outcome.converged and int(unique_id) in unrefined_by_id
        ),
        key=lambda unique_id: (
            float(unrefined_by_id[unique_id]["energy_eV"]),
            unique_id,
        ),
    )
    if regions is not None:
        for position, unique_id in enumerate(unconverged_ids, start=1):
            unique = uniques_by_id[unique_id]
            entry = unrefined_by_id[unique_id]
            exports.append(
                _export_candidate(
                    output_dir / "top" / f"unrefined_{position:03d}",
                    unique_id=unique_id,
                    payload=unique.to_dict(),
                    atoms=unique.representative.atoms,
                    options=options,
                    regions=regions,
                    entry=entry,
                    energy=float(entry["energy_eV"]),
                )
            )

    ranked_for_diagnostics = refined_entries or unrefined_entries
    saturation = _saturation(ranked_for_diagnostics, total_quenches=total_quenches)
    baseline_report = _baseline_report(
        baseline,
        ranked_for_diagnostics,
        uniques_by_id=uniques_by_id,
        options=options,
    )
    warnings = list(warnings)
    if unconverged_ids:
        warnings.append(
            "refinement did not converge for candidate(s) "
            + ", ".join(str(unique_id) for unique_id in unconverged_ids)
            + "; they are exported as top/unrefined_NNN/ with their quench "
            "geometry and energy (spec 10.3)"
        )
    warnings += _status_warnings(registry, options, ranked_for_diagnostics)
    warnings = list(dict.fromkeys(warnings))

    # A merge has no replicas of its own; only a search whose every replica
    # died (no surviving start) is a failure (spec 7.1).
    status = "completed"
    if replicas and not any(
        getattr(replica, "status", None) == "active" for replica in replicas
    ):
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
            "best_energy_eV": (
                float(refined_best)
                if refined_entries
                else (float(unrefined_best) if unrefined_best is not None else None)
            ),
            "best_energy_source": "refined" if refined_entries else "quench",
            "n_rankable": len(refined_entries) + len(unrefined_entries),
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
            "n_refined": len(accepted),
            "n_merged_into_existing": int(
                sum(len(group["ids"]) - 1 for group in refined_groups)
            ),
            "notes": refinement_notes,
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


def _finite_info(info: dict) -> dict:
    """Drop non-finite floats: extxyz cannot carry NaN/Inf."""
    cleaned: dict = {}
    for key, value in info.items():
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, (list, tuple)):
            cleaned[key] = list(value)
            continue
        cleaned[key] = value
    return cleaned


def _write_minima(output_dir: Path, registry: MinimumRegistry) -> None:
    """Write ``minima/all_minima.extxyz`` and ``minima/registry.json`` (spec 11.1).

    Every quench result is kept, whatever its status, so rejected, failed and
    ``reacted`` geometries can be inspected after the run.
    """
    directory = output_dir / "minima"
    directory.mkdir(parents=True, exist_ok=True)
    frames: list[Atoms] = []
    for status in sorted(registry.status_classes()):
        for unique in sorted(registry.uniques(status), key=lambda item: item.unique_id):
            for member in sorted(unique.members, key=lambda item: item.record_id):
                atoms = member.atoms.copy()
                atoms.info.update(
                    _finite_info(
                        {
                            "confsearch_unique_id": int(unique.unique_id),
                            "confsearch_record_id": int(member.record_id),
                            "confsearch_status": member.status,
                            "confsearch_flags": list(member.flags),
                            "energy_eV": float(member.energy_eV),
                            "fmax_eV_A": float(member.fmax_eV_A),
                            "quench_steps": int(member.steps),
                            "quench_index": int(member.quench_index),
                            "cumulative_md_ps": float(member.cumulative_md_ps),
                            "origin_replica": int(member.origin.replica),
                            "origin_cycle": int(member.origin.cycle),
                            "origin_source": member.origin.source,
                        }
                    )
                )
                frames.append(atoms)
    if frames:
        ase_write(directory / "all_minima.extxyz", frames, format="extxyz")
    (directory / "registry.json").write_text(
        json.dumps(registry.to_dict(), indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


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
