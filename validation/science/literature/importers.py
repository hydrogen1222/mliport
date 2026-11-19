"""Deterministic importers: upstream literature files -> normalised extxyz.

Every importer returns a small provenance document (written next to the
imported extxyz as ``selection.json``) that records the upstream commit, the
SHA-256 of every contributing file, the selection rule, the seed and the
chosen record ids.  The evaluation script only ever reads the normalised
extxyz, so a model run never needs network access and the same imported file
can be re-evaluated by any engine.

Normalised conventions (identical for every dataset):

* ``SinglePointCalculator`` energy = total energy of the cell in eV;
* ``forces`` = eV / Angstrom, one row per atom;
* ``atoms.info`` carries ``literature_dataset``, ``literature_split``,
  ``literature_record_id``, ``literature_group``, ``reference_level``,
  ``comparison_mode`` and the dataset-specific fields documented below.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import numpy as np

_LITERATURE_ROOT = Path(__file__).resolve().parent
_SCIENCE_ROOT = _LITERATURE_ROOT.parent
if str(_SCIENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCIENCE_ROOT))

from literature import datasets as registry_layer  # noqa: E402

#: LiPS-25 Task-1 tie-line composition x in (Li2S)_x (P2S5)_(1-x).
LIPS25_TIE_LINE = {
    "dhf_p2s5": 0.0,
    "dhf_li2p2s6": 0.5,
    "dhf_li4p2s7": 2.0 / 3.0,
    "dhf_li7p3s11": 0.7,
    "dhf_li3ps4_alpha": 0.75,
    "dhf_li3ps4_beta": 0.75,
    "dhf_li3ps4_gamma": 0.75,
    "dhf_li7ps6_f-43m": 0.875,
    "dhf_li7ps6_pna21": 0.875,
    "dhf_li2s": 1.0,
}
LIPS25_ENDPOINT_LOW = "dhf_p2s5"
LIPS25_ENDPOINT_HIGH = "dhf_li2s"


# ---------------------------------------------------------------------------
# selection helpers (pure, deterministic, unit-tested)
# ---------------------------------------------------------------------------


def evenly_spaced_indices(n: int, count: int) -> list[int]:
    """``count`` indices spread evenly over ``[0, n-1]`` (endpoints included)."""
    if n <= 0:
        msg = "evenly_spaced_indices needs n > 0"
        raise ValueError(msg)
    count = max(1, min(int(count), n))
    if count == 1:
        return [0]
    raw = np.linspace(0, n - 1, count)
    return sorted({round(float(value)) for value in raw})


def stratified_indices(
    keys: Iterable[str], count: int, seed: int
) -> list[int]:
    """Proportional stratified sample with at least one record per stratum.

    Allocation uses the largest-remainder method so the realised sample size
    is exactly ``min(count, n)``.  Randomness is confined to the per-stratum
    shuffle, which uses ``numpy.random.default_rng(seed)`` derived from the
    registry seed and the sorted stratum names, so the result is reproducible
    across machines and NumPy versions that keep the Generator contract.
    """
    key_list = [str(k) for k in keys]
    n = len(key_list)
    if n == 0:
        return []
    target = max(1, min(int(count), n))
    order = sorted(set(key_list))
    index_by_key: dict[str, list[int]] = {key: [] for key in order}
    for i, key in enumerate(key_list):
        index_by_key[key].append(i)
    total = float(n)
    quotas: dict[str, float] = {
        key: target * len(index_by_key[key]) / total for key in order
    }
    allocation = {key: int(quotas[key]) for key in order}
    if target >= len(order):
        allocation = {key: max(1, allocation[key]) for key in order}
    assigned = sum(allocation.values())
    remainders = sorted(
        order,
        key=lambda key: (-(quotas[key] - int(quotas[key])), key),
    )
    cursor = 0
    while assigned < target and remainders:
        key = remainders[cursor % len(remainders)]
        if allocation[key] < len(index_by_key[key]):
            allocation[key] += 1
            assigned += 1
        cursor += 1
        if cursor > 10 * len(remainders) + target:
            break
    while assigned > target:
        # remove from the largest allocation that can spare one record
        candidates = [k for k in order if allocation[k] > 1]
        if not candidates:
            break
        key = max(candidates, key=lambda k: (allocation[k], k))
        allocation[key] -= 1
        assigned -= 1
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for key in order:
        take = allocation[key]
        pool = list(index_by_key[key])
        rng.shuffle(pool)
        chosen.extend(pool[:take])
    return sorted(chosen)


# ---------------------------------------------------------------------------
# extxyz writing
# ---------------------------------------------------------------------------


def write_reference_frames(frames: list[Any], path: Path) -> Path:
    """Write labelled frames as extxyz and verify the round trip."""
    from ase.io import write as ase_write  # noqa: PLC0415

    if not frames:
        msg = "refusing to write an empty reference extxyz"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(path, frames, format="extxyz")
    return path


def _single_point(atoms: Any, energy: float, forces: np.ndarray | None) -> Any:
    from ase.calculators.singlepoint import SinglePointCalculator  # noqa: PLC0415

    atoms = atoms.copy()
    atoms.calc = SinglePointCalculator(
        atoms, energy=float(energy), forces=forces
    )
    return atoms


def _frame_labels(atoms: Any) -> tuple[float | None, np.ndarray | None]:
    """Read energy/forces from either the ASE calculator or the raw extxyz props.

    ASE >= 3.23 turns the standard ``energy``/``forces`` extxyz keys into a
    ``SinglePointCalculator`` while dataset-specific keys such as
    ``dft_energy``/``dft_forces`` stay in ``info``/``arrays``.  Both layouts are
    first class here; a virial is never returned as stress.
    """
    energy: float | None = None
    forces: np.ndarray | None = None
    if atoms.calc is not None:
        try:
            value = float(atoms.calc.get_potential_energy(atoms))
            energy = value if np.isfinite(value) else None
        except Exception:  # noqa: BLE001 - label may be absent
            energy = None
        try:
            candidate = np.asarray(atoms.calc.get_forces(atoms), dtype=float)
            forces = candidate if np.isfinite(candidate).all() else None
        except Exception:  # noqa: BLE001 - label may be absent
            forces = None
    if energy is None:
        for key in ("dft_energy", "energy", "total_energy"):
            if key in atoms.info:
                value = float(atoms.info[key])
                energy = value if np.isfinite(value) else None
                break
    if energy is None and "energy_per_at" in atoms.info:
        energy = float(atoms.info["energy_per_at"]) * len(atoms)
    if forces is None:
        for key in ("dft_forces", "forces"):
            if key in atoms.arrays:
                candidate = np.asarray(atoms.arrays[key], dtype=float)
                forces = candidate if np.isfinite(candidate).all() else None
                break
    return energy, forces


def _nan_count(atoms: Any) -> tuple[bool, bool]:
    """Return (energy_finite, forces_finite) for a labelled frame."""
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return bool(np.isfinite(energy)), bool(np.isfinite(forces).all())


# ---------------------------------------------------------------------------
# LiPS-25
# ---------------------------------------------------------------------------


def import_lips25(
    entry: dict[str, Any],
    root: Path,
    out_dir: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Import the LiPS-25 Task-1/Task-2/test splits."""
    from ase.io import read as ase_read  # noqa: PLC0415

    provenance = entry["provenance"]
    seed = int(selection.get("seed") or 0)
    rules = selection["rules"]
    split_meta: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    def split_files(split_id: str) -> list[dict[str, Any]]:
        return entry["splits"][split_id].get("files", [])

    def import_labelled_split(split_id: str, rule: dict[str, Any]) -> list[Any]:
        source = split_files(split_id)[0]
        source_path = root / source["path"]
        raw = ase_read(source_path, index=":", format="extxyz")
        groups = [
            str(
                frame.info.get(
                    "label",
                    frame.info.get("literature_group", f"{split_id}_frame"),
                )
            )
            for frame in raw
        ]
        if rule["rule"] == "all":
            chosen = list(range(len(raw)))
        elif rule["rule"] == "evenly_spaced":
            chosen = evenly_spaced_indices(len(raw), int(rule["count"]))
        elif rule["rule"] == "stratified":
            chosen = stratified_indices(groups, int(rule["count"]), seed)
        else:
            msg = f"unknown LiPS-25 selection rule {rule['rule']!r}"
            raise registry_layer.RegistryError(msg)
        frames: list[Any] = []
        for index in chosen:
            atoms = raw[index]
            energy, forces = _frame_labels(atoms)
            if energy is None:
                msg = f"{split_id} frame {index} carries no energy label"
                raise registry_layer.DatasetAccessError(msg)
            if forces is None:
                msg = f"{split_id} frame {index} carries no force labels"
                raise registry_layer.DatasetAccessError(msg)
            frame = _single_point(atoms, energy, forces)
            label = str(atoms.info.get("label", ""))
            if split_id == "task1_relaxed":
                # one physical group: the Li2S-P2S5 tie line.  Each frame is a
                # different phase, so per-phase means would be degenerate.
                group = "li2s-p2s5-tie-line"
            elif split_id == "task2_mq":
                group = "mq_li7p3s11_672"
            else:
                group = label
            record_id = f"lips25:{split_id}:{index}"
            frame.info.update(
                {
                    "literature_dataset": "lips25",
                    "literature_split": split_id,
                    "literature_record_id": record_id,
                    "literature_group": group,
                    "literature_phase": label,
                    "literature_source_file": source["path"],
                    "literature_source_index": int(index),
                    "reference_level": entry["reference_level"],
                    "comparison_mode": entry["comparison_mode"],
                    "reference_dft_software": entry["dft"]["software"],
                    "reference_dft_functional": entry["dft"]["functional"],
                    "reference_label_source": "dft_energy/dft_forces (PBEsol)",
                }
            )
            mobile = entry["splits"][split_id].get("mobile_elements") or []
            if mobile:
                frame.info["mobile_elements"] = ",".join(mobile)
            if split_id == "task1_relaxed":
                label_key = label
                if label_key not in LIPS25_TIE_LINE:
                    msg = (
                        f"unknown LiPS-25 tie-line label {label_key!r}; the "
                        "Task-1 formation-energy protocol needs its x value"
                    )
                    raise registry_layer.DatasetAccessError(msg)
                x = LIPS25_TIE_LINE[label_key]
                frame.info.update(
                    {
                        "li_ps_x": float(x),
                        "n_formula_units": float(len(frame)) / (7.0 - 4.0 * x),
                        "tie_line_endpoint_low": LIPS25_ENDPOINT_LOW,
                        "tie_line_endpoint_high": LIPS25_ENDPOINT_HIGH,
                    }
                )
            if "virial" in atoms.info:
                frame.info["reference_virial_present"] = True
                frame.info["reference_virial_used_as_stress"] = False
            frames.append(frame)
        energy_ok = all(_nan_count(frame)[0] for frame in frames)
        forces_ok = all(_nan_count(frame)[1] for frame in frames)
        split_meta[split_id] = {
            "source": {
                "path": source["path"],
                "sha256": source["sha256"],
                "bytes": source["bytes"],
            },
            "available_records": len(raw),
            "selected_records": len(frames),
            "rule": rule,
            "selection_rule": (
                "all" if rule["rule"] == "all" else rule["rule"]
            ),
            "selected_source_indices": chosen,
            "labels": entry["splits"][split_id]["labels"],
            "energy_finite": energy_ok,
            "forces_finite": forces_ok,
            "output": f"{split_id}.extxyz",
        }
        records.append(split_meta[split_id])
        return frames

    written: list[str] = []
    for split_id, rule in rules.items():
        if split_id not in entry["splits"]:
            continue
        if not entry["splits"][split_id].get("files"):
            continue
        # unlabelled splits (relaxation inputs) are copied below, not labelled
        if not entry["splits"][split_id]["labels"].get("energy"):
            continue
        frames = import_labelled_split(split_id, rule)
        target = out_dir / f"{split_id}.extxyz"
        write_reference_frames(frames, target)
        written.append(target.name)

    # Unlabelled relaxation inputs are copied verbatim (geometry only) so the
    # evaluator can relax them without inventing labels.
    if "task1_unrelaxed" in rules and "task1_unrelaxed" in entry["splits"]:
        source = split_files("task1_unrelaxed")[0]
        raw = ase_read(root / source["path"], index=":", format="extxyz")
        frames = []
        for index, atoms in enumerate(raw):
            frame = atoms.copy()
            frame.info.update(
                {
                    "literature_dataset": "lips25",
                    "literature_split": "task1_unrelaxed",
                    "literature_record_id": f"lips25:task1_unrelaxed:{index}",
                    "literature_group": "li2s-p2s5-tie-line",
                    "literature_phase": str(atoms.info.get("label", "")),
                    "reference_level": entry["reference_level"],
                    "comparison_mode": entry["comparison_mode"],
                    "reference_label_source": "unlabelled DFT relaxation input",
                }
            )
            frames.append(frame)
        target = out_dir / "task1_unrelaxed.extxyz"
        write_reference_frames(frames, target)
        written.append(target.name)
        split_meta["task1_unrelaxed"] = {
            "source": {
                "path": source["path"],
                "sha256": source["sha256"],
                "bytes": source["bytes"],
            },
            "available_records": len(raw),
            "selected_records": len(frames),
            "rule": rules.get("task1_unrelaxed", {"rule": "all"}),
            "labels": entry["splits"]["task1_unrelaxed"]["labels"],
            "output": target.name,
        }
        records.append(split_meta["task1_unrelaxed"])

    document = {
        "schema": "mliport.literature-selection/1",
        "dataset": "lips25",
        "tier": selection["tier"],
        "seed": seed,
        "registry_revision": 1,
        "upstream_commit": provenance["commit"],
        "upstream_repo": provenance["repo"],
        "comparison_mode": entry["comparison_mode"],
        "reference_level": entry["reference_level"],
        "license": entry["license"],
        "units": entry["units"],
        "splits": split_meta,
        "outputs": sorted(written),
    }
    (out_dir / "selection.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return document


# ---------------------------------------------------------------------------
# LiPS20 (DeePMD .npy arrays)
# ---------------------------------------------------------------------------


def _read_int_list(path: Path) -> list[int]:
    return [int(tok) for tok in path.read_text(encoding="utf-8").split()]


def import_lips20(
    entry: dict[str, Any],
    root: Path,
    out_dir: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Import a deterministic frame subset of every LiPS20 composition."""
    from ase import Atoms  # noqa: PLC0415

    split_id = "lips20_3000k"
    split = entry["splits"][split_id]
    rule = selection["rules"][split_id]
    per_composition = int(rule["count"])
    frames: list[Any] = []
    composition_meta: dict[str, Any] = {}
    for comp, meta in split["compositions"].items():
        base = root / meta["path_prefix"]
        set_dir = base / "set.000"
        type_map = (base / "type_map.raw").read_text(encoding="utf-8").split()
        types = _read_int_list(base / "type.raw")
        energies = np.load(set_dir / "energy.npy")
        coords = np.load(set_dir / "coord.npy")
        forces = np.load(set_dir / "force.npy")
        boxes = np.load(set_dir / "box.npy")
        virials = np.load(set_dir / "virial.npy")
        n_frames, flat = coords.shape
        natoms = flat // 3
        if not (natoms == len(types) == forces.shape[1] // 3):
            msg = (
                f"LiPS20 {comp}: inconsistent atom counts between coord "
                f"({flat // 3}), type.raw ({len(types)}) and force "
                f"({forces.shape[1] // 3})"
            )
            raise registry_layer.DatasetAccessError(msg)
        if energies.shape[0] != n_frames or boxes.shape[0] != n_frames:
            msg = f"LiPS20 {comp}: frame counts differ between arrays"
            raise registry_layer.DatasetAccessError(msg)
        if virials.shape != (n_frames, 9):
            msg = f"LiPS20 {comp}: unexpected virial shape {virials.shape}"
            raise registry_layer.DatasetAccessError(msg)
        symbols = [type_map[t] for t in types]
        chosen = evenly_spaced_indices(n_frames, per_composition)
        for index in chosen:
            atoms = Atoms(
                symbols=symbols,
                positions=coords[index].reshape(natoms, 3),
                cell=boxes[index].reshape(3, 3),
                pbc=True,
            )
            frame = _single_point(
                atoms,
                float(energies[index]),
                np.asarray(forces[index], dtype=float).reshape(natoms, 3),
            )
            frame.info.update(
                {
                    "literature_dataset": "lips20",
                    "literature_split": split_id,
                    "literature_record_id": f"lips20:{comp}:{index}",
                    "literature_group": comp,
                    "literature_composition": comp,
                    "literature_formula": meta["formula"],
                    "literature_temperature_K": int(meta["temperature_K"]),
                    "literature_source_frame": int(index),
                    "reference_level": entry["reference_level"],
                    "comparison_mode": entry["comparison_mode"],
                    "reference_dft_software": entry["dft"]["software"],
                    "reference_dft_functional": entry["dft"]["functional"],
                    "reference_dft_dispersion": entry["dft"]["dispersion"],
                    "reference_virial_present": True,
                    "reference_virial_used_as_stress": False,
                    "mobile_elements": ",".join(
                        split.get("mobile_elements") or []
                    ),
                }
            )
            energy_ok, forces_ok = _nan_count(frame)
            if not (energy_ok and forces_ok):
                msg = f"LiPS20 {comp} frame {index} contains non-finite labels"
                raise registry_layer.DatasetAccessError(msg)
            frames.append(frame)
        composition_meta[comp] = {
            "formula": meta["formula"],
            "natoms": natoms,
            "source_frames": n_frames,
            "selected_frames": len(chosen),
            "selected_source_indices": chosen,
            "sha256": meta["sha256"],
            "type_map_sha256": meta["type_map_sha256"],
            "type_sha256": meta["type_sha256"],
            "virial_sha256": meta["sha256"]["virial.npy"],
            "virial_used_as_stress": False,
        }
    target = out_dir / f"{split_id}.extxyz"
    write_reference_frames(frames, target)
    document = {
        "schema": "mliport.literature-selection/1",
        "dataset": "lips20",
        "tier": selection["tier"],
        "seed": int(selection.get("seed") or 0),
        "registry_revision": 1,
        "upstream_commit": entry["provenance"]["commit"],
        "upstream_repo": entry["provenance"]["repo"],
        "comparison_mode": entry["comparison_mode"],
        "reference_level": entry["reference_level"],
        "license": entry["license"],
        "units": entry["units"],
        "splits": {
            split_id: {
                "rule": rule,
                "output": target.name,
                "selected_records": len(frames),
                "compositions": composition_meta,
                "labels": split["labels"],
            }
        },
        "outputs": [target.name],
    }
    (out_dir / "selection.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return document


# ---------------------------------------------------------------------------
# JARVIS-DFT (single JSON dump of the 3D database)
# ---------------------------------------------------------------------------


def _jarvis_formula_key(formula: str) -> str:
    """Composition key independent of the element order in ``formula``."""
    tokens = re.findall(r"([A-Z][a-z]?)([0-9.]*)", formula)
    parts = []
    for element, count in tokens:
        number = float(count) if count else 1.0
        parts.append((element, number))
    return "".join(f"{el}{num:g}" for el, num in sorted(parts))


def import_jarvis(
    entry: dict[str, Any],
    root: Path,
    out_dir: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Import polymorph groups from the JARVIS-DFT 3D JSON dump."""
    from ase import Atoms  # noqa: PLC0415

    split_id = "polymorphs"
    rule = selection["rules"][split_id]
    filters = entry["selection"]["filters"]
    archive = root / entry["provenance"]["file"]["name"]
    inner_name = entry["provenance"]["inner_file"]["name"]
    with zipfile.ZipFile(archive) as zf, zf.open(inner_name) as handle:
        database = json.load(handle)
    if not isinstance(database, list) or not database:
        msg = "JARVIS dump is not a non-empty list of records"
        raise registry_layer.DatasetAccessError(msg)

    groups: dict[str, list[dict[str, Any]]] = {}
    rejected = {"natoms": 0, "energy": 0, "structure": 0}
    for record in database:
        atoms_block = record.get("atoms") or {}
        elements = atoms_block.get("elements")
        coords = atoms_block.get("coords")
        lattice = atoms_block.get("lattice_mat")
        energy = record.get("optb88vdw_total_energy")
        if not (elements and coords and lattice):
            rejected["structure"] += 1
            continue
        natoms = len(elements)
        if natoms < filters["natoms_min"] or natoms > filters["natoms_max"]:
            rejected["natoms"] += 1
            continue
        if not isinstance(energy, (int, float)) or not np.isfinite(energy):
            rejected["energy"] += 1
            continue
        key = _jarvis_formula_key(str(record.get("formula", "")))
        groups.setdefault(key, []).append(record)

    candidates = []
    for key, records in groups.items():
        unique_energies = {
            round(float(r["optb88vdw_total_energy"]), 6) for r in records
        }
        if len(records) >= int(filters["min_polymorphs"]) and len(unique_energies) >= 2:
            natoms = int(len(records[0]["atoms"]["elements"]))
            candidates.append((natoms, key, records))
    candidates.sort(key=lambda item: (item[0], item[1]))
    seed = int(selection.get("seed") or 0)
    rng = np.random.default_rng(seed)
    order = np.arange(len(candidates))
    rng.shuffle(order)
    wanted_groups = int(rule["groups"])
    per_group = int(rule["per_group"])
    chosen_groups = [
        candidates[i] for i in order[: min(wanted_groups, len(candidates))]
    ]
    frames: list[Any] = []
    group_meta: dict[str, Any] = {}
    for _, key, records in sorted(chosen_groups, key=lambda item: item[1]):
        ordered = sorted(
            records,
            key=lambda r: (float(r["optb88vdw_total_energy"]), str(r["jid"])),
        )
        indices = evenly_spaced_indices(len(ordered), per_group)
        for index in indices:
            record = ordered[index]
            atoms_block = record["atoms"]
            positions = np.asarray(atoms_block["coords"], dtype=float)
            cell = np.asarray(atoms_block["lattice_mat"], dtype=float)
            if not atoms_block.get("cartesian", True):
                # fractional coordinates -> Cartesian; the published OptB88vdW
                # records are Cartesian, but the flag is honoured when present
                positions = positions @ cell
            atoms = Atoms(
                symbols=atoms_block["elements"], positions=positions, cell=cell, pbc=True
            )
            natoms = len(atoms)
            energy_total = float(record["optb88vdw_total_energy"]) * natoms
            frame = _single_point(atoms, energy_total, None)
            frame.info.update(
                {
                    "literature_dataset": "jarvis",
                    "literature_split": split_id,
                    "literature_record_id": f"jarvis:{record['jid']}",
                    "literature_group": key,
                    "literature_formula": record.get("formula"),
                    "jarvis_jid": record["jid"],
                    "jarvis_encut_eV": record.get("encut"),
                    "jarvis_func": record.get("func"),
                    "jarvis_formation_energy_peratom_eV": record.get(
                        "formation_energy_peratom"
                    ),
                    "reference_level": entry["reference_level"],
                    "comparison_mode": entry["comparison_mode"],
                    "reference_dft_software": entry["dft"]["software"],
                    "reference_dft_functional": entry["dft"]["functional"],
                    "reference_label_source": "optb88vdw_total_energy (eV/atom)",
                }
            )
            frames.append(frame)
            group_meta.setdefault(
                key,
                {
                    "formula": record.get("formula"),
                    "available_records": len(records),
                    "selected_records": 0,
                    "jids": [],
                },
            )
            group_meta[key]["selected_records"] += 1
            group_meta[key]["jids"].append(record["jid"])
    if not frames:
        msg = (
            "JARVIS selection produced no structures; the filters "
            f"{filters} rejected every formula group (rejected={rejected})"
        )
        raise registry_layer.DatasetAccessError(msg)
    target = out_dir / f"{split_id}.extxyz"
    write_reference_frames(frames, target)
    document = {
        "schema": "mliport.literature-selection/1",
        "dataset": "jarvis",
        "tier": selection["tier"],
        "seed": seed,
        "registry_revision": 1,
        "upstream_doi": entry["provenance"]["doi"],
        "upstream_file": entry["provenance"]["file"],
        "comparison_mode": entry["comparison_mode"],
        "reference_level": entry["reference_level"],
        "license": entry["license"],
        "units": entry["units"],
        "filters": filters,
        "splits": {
            split_id: {
                "rule": rule,
                "output": target.name,
                "selected_records": len(frames),
                "selected_formula_groups": sorted(group_meta),
                "groups": group_meta,
                "total_records_in_database": len(database),
                "candidate_formula_groups": len(candidates),
                "rejected": rejected,
                "labels": entry["splits"][split_id]["labels"],
            }
        },
        "outputs": [target.name],
    }
    (out_dir / "selection.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return document


# ---------------------------------------------------------------------------
# MS25 / user-supplied references (local extxyz manifest)
# ---------------------------------------------------------------------------


def import_local_extxyz(
    entry: dict[str, Any],
    root: Path,
    out_dir: Path,
    selection: dict[str, Any],
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Import a user-supplied reference through an explicit manifest.

    The manifest is mandatory because MS25 (and similar datasets) cannot be
    downloaded by this harness and their DFT settings are not machine-readable
    upstream.  Required fields::

        {
          "schema": "mliport.literature-local-manifest/1",
          "dataset": "ms25",
          "subset": "mgo_surfaces",
          "license": {"id": "...", "redistributable": false},
          "reference_level": "PBE (VASP 6.x)",
          "dft": {"software": "...", "functional": "...", ...},
          "comparison_mode": "cross_method",
          "units": {"energy": "eV", "forces": "eV/Angstrom", "stress": null},
          "files": [{"path": "mgo.xyz", "sha256": "<64 hex>", ...}]
        }

    Every file is hash-verified before a single frame is read; a mismatch
    aborts instead of silently importing different data.
    """
    from ase.io import read as ase_read  # noqa: PLC0415

    if manifest_path is None:
        msg = (
            "dataset 'ms25' requires --reference-manifest PATH: the data is "
            "not automatically downloadable (Materials Data Facility / Globus) "
            "and its DFT settings are not machine-readable upstream"
        )
        raise registry_layer.DatasetAccessError(msg)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    problems = []
    for field in ("dataset", "subset", "license", "reference_level", "dft",
                  "comparison_mode", "units", "files"):
        if field not in manifest:
            problems.append(field)
    if problems:
        msg = f"local reference manifest is missing fields: {problems}"
        raise registry_layer.RegistryError(msg)
    if manifest["dataset"] != entry["id"]:
        msg = (
            f"manifest dataset {manifest['dataset']!r} does not match the "
            f"requested dataset {entry['id']!r}"
        )
        raise registry_layer.RegistryError(msg)
    if manifest["comparison_mode"] not in registry_layer.COMPARISON_MODES:
        msg = f"invalid comparison_mode {manifest['comparison_mode']!r}"
        raise registry_layer.RegistryError(msg)
    base = Path(manifest_path).resolve().parent
    frames: list[Any] = []
    file_meta: list[dict[str, Any]] = []
    for spec in manifest["files"]:
        path = (base / spec["path"]).resolve()
        if not path.exists():
            msg = f"local reference file not found: {path}"
            raise registry_layer.DatasetAccessError(msg)
        actual = registry_layer.sha256_file(path)
        if actual != spec.get("sha256"):
            msg = (
                f"local reference file {path.name} does not match its manifest "
                f"sha256 (expected {spec.get('sha256')}, got {actual})"
            )
            raise registry_layer.DatasetAccessError(msg)
        frames_in_file = ase_read(path, index=":", format="extxyz")
        for index, atoms in enumerate(frames_in_file):
            if atoms.calc is None:
                msg = (
                    f"{path.name} frame {index} has no SinglePointCalculator; "
                    "reference extxyz files must carry energy (and optionally "
                    "forces/stress) labels"
                )
                raise registry_layer.DatasetAccessError(msg)
            try:
                energy = float(atoms.get_potential_energy())
            except Exception as exc:  # noqa: BLE001 - classify as data problem
                msg = f"{path.name} frame {index} has no energy label: {exc}"
                raise registry_layer.DatasetAccessError(msg) from exc
            try:
                forces = np.asarray(atoms.get_forces(), dtype=float)
            except Exception:  # noqa: BLE001 - forces are optional
                forces = None
            frame = _single_point(atoms, energy, forces)
            frame.info.update(
                {
                    "literature_dataset": entry["id"],
                    "literature_split": manifest["subset"],
                    "literature_record_id": f"{entry['id']}:{spec['path']}:{index}",
                    "literature_group": str(
                        atoms.info.get("literature_group", manifest["subset"])
                    ),
                    "reference_level": manifest["reference_level"],
                    "comparison_mode": manifest["comparison_mode"],
                    "reference_dft_software": manifest["dft"].get("software"),
                    "reference_dft_functional": manifest["dft"].get("functional"),
                }
            )
            frames.append(frame)
        file_meta.append(
            {
                "path": spec["path"],
                "sha256": spec["sha256"],
                "bytes": path.stat().st_size,
                "frames": len(frames_in_file),
            }
        )
    split_id = str(manifest["subset"])
    target = out_dir / f"{split_id}.extxyz"
    write_reference_frames(frames, target)
    document = {
        "schema": "mliport.literature-selection/1",
        "dataset": entry["id"],
        "tier": selection["tier"],
        "seed": int(selection.get("seed") or 0),
        "comparison_mode": manifest["comparison_mode"],
        "reference_level": manifest["reference_level"],
        "dft": manifest["dft"],
        "license": manifest["license"],
        "units": manifest["units"],
        "manifest": str(Path(manifest_path).resolve()),
        "splits": {
            split_id: {
                "rule": {"rule": "local_manifest"},
                "output": target.name,
                "selected_records": len(frames),
                "files": file_meta,
            }
        },
        "outputs": [target.name],
    }
    (out_dir / "selection.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return document


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def import_dataset(
    dataset_id: str,
    tier: str = "core",
    cache: str | Path | None = None,
    out_dir: str | Path | None = None,
    registry: dict[str, Any] | None = None,
    fetch: bool = True,
    offline: bool = False,
    reference_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch (unless cached/offline) and import one dataset for one tier."""
    registry = registry or registry_layer.load_registry()
    entry = registry_layer.dataset_entry(registry, dataset_id)
    selection = registry_layer.tier_selection(registry, dataset_id, tier)
    root = registry_layer.dataset_cache_dir(registry, dataset_id, cache)
    target = Path(out_dir) if out_dir else root / "imported" / tier
    target.mkdir(parents=True, exist_ok=True)

    if dataset_id == "omat24":
        return _import_omat24(entry, registry, target, selection)
    if fetch:
        report = registry_layer.fetch_dataset(
            dataset_id, cache=cache, offline=offline, registry=registry
        )
        registry_layer.write_fetch_record(report, target)
    if dataset_id == "lips25":
        return import_lips25(entry, root, target, selection)
    if dataset_id == "lips20":
        return import_lips20(entry, root, target, selection)
    if dataset_id == "jarvis":
        return import_jarvis(entry, root, target, selection)
    if dataset_id == "ms25":
        return import_local_extxyz(
            entry, root, target, selection, manifest_path=reference_manifest
        )
    msg = f"no importer registered for dataset {dataset_id!r}"
    raise registry_layer.RegistryError(msg)


def _import_omat24(
    entry: dict[str, Any],
    registry: dict[str, Any],
    out_dir: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Reuse the already-committed 256-structure OMat24 subset."""
    from ase.io import read as ase_read  # noqa: PLC0415

    repo_root = _SCIENCE_ROOT.parents[1]
    split = entry["splits"]["val256"]
    source = repo_root / split["path"]
    if not source.exists():
        msg = (
            f"the committed OMat24 subset is missing: {source}. It is pinned "
            "by validation/science/data_manifest.json and should be present in "
            "a complete checkout; do not regenerate it with a different seed"
        )
        raise registry_layer.DatasetAccessError(msg)
    actual = registry_layer.sha256_file(source)
    if actual != split["sha256"]:
        msg = (
            f"the committed OMat24 subset hash changed ({actual} != "
            f"{split['sha256']}); refusing to benchmark against a different "
            "subset than the one pinned in data_manifest.json"
        )
        raise registry_layer.DatasetAccessError(msg)
    rule = selection["rules"]["val256"]
    frames_in = ase_read(source, index=":", format="extxyz")
    if rule["rule"] == "all":
        chosen = list(range(len(frames_in)))
    elif rule["rule"] == "first_n":
        chosen = list(range(min(int(rule["count"]), len(frames_in))))
    else:
        msg = f"unknown OMat24 selection rule {rule['rule']!r}"
        raise registry_layer.RegistryError(msg)
    frames = []
    for index in chosen:
        atoms = frames_in[index]
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        frame = _single_point(atoms, energy, forces)
        frame.info.update(
            {
                "literature_dataset": "omat24",
                "literature_split": "val256",
                "literature_record_id": f"omat24:val256:{index}",
                "literature_group": str(
                    atoms.info.get("omat24_stratum", "unknown")
                ),
                "reference_level": entry["reference_level"],
                "comparison_mode": entry["comparison_mode"],
                "reference_label_source": "OMat24 validation LMDB labels",
            }
        )
        frames.append(frame)
    target = out_dir / "val256.extxyz"
    write_reference_frames(frames, target)
    document = {
        "schema": "mliport.literature-selection/1",
        "dataset": "omat24",
        "tier": selection["tier"],
        "seed": selection.get("seed"),
        "registry_revision": 1,
        "source": {
            "path": split["path"],
            "sha256": split["sha256"],
            "selection_script": entry["provenance"]["selection_script"],
            "selection_seed": entry["provenance"]["selection_seed"],
        },
        "comparison_mode": entry["comparison_mode"],
        "reference_level": entry["reference_level"],
        "license": entry["license"],
        "units": entry["units"],
        "splits": {
            "val256": {
                "rule": rule,
                "output": target.name,
                "available_records": len(frames_in),
                "selected_records": len(frames),
                "selected_source_indices": chosen,
            }
        },
        "outputs": [target.name],
    }
    (out_dir / "selection.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return document
