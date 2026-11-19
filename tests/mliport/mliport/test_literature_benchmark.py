"""CPU-only tests for the literature benchmark suite (tier ``t9``).

Covers, without any backend or network:

* known-answer checks for every metric (including the LiPS-25 Task-1
  formation-energy protocol, Voigt stress order and the eV/A^3 -> GPa
  conversion);
* the registry contract (licences, units, checksums, tiers) and its
  fail-closed behaviour on tampering;
* cache semantics (offline reuse, hash mismatch, no silent empty dataset);
* deterministic selection helpers;
* the importers (LiPS-25 extxyz labels in both ASE layouts, LiPS20 DeePMD
  arrays with the virial kept out of stress, JARVIS polymorph groups, the
  MS25 local manifest, and the committed OMat24 subset);
* the evaluation helpers (formation energy, tie-line keys, lattice metrics);
* the tiered orchestrator's configuration and its ``--import-only`` path;
* the Markdown/JSON report and its handling of blocked records.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCIENCE = REPO / "validation" / "science"
SCRIPTS = SCIENCE / "scripts"
LITERATURE = SCIENCE / "literature"
for path in (str(LITERATURE), str(SCIENCE), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import common  # noqa: E402
import literature_eval  # noqa: E402
import literature_report  # noqa: E402
from literature import datasets, importers, metrics  # noqa: E402

# ---------------------------------------------------------------------------
# metrics: known answers
# ---------------------------------------------------------------------------


def test_abs_stats_known_answer():
    values = np.array([-1.0, 2.0, -2.0, 0.0])
    stats = metrics.abs_stats(values)
    assert stats["n"] == 4
    assert stats["mae"] == pytest.approx(1.25)
    assert stats["rmse"] == pytest.approx(np.sqrt((1 + 4 + 4 + 0) / 4))
    assert stats["median"] == pytest.approx(1.5)
    assert stats["max"] == pytest.approx(2.0)
    assert stats["bias"] == pytest.approx(-0.25)
    assert stats["p95"] <= stats["max"]


def test_energy_per_atom_stats_known_answer():
    stats = metrics.energy_per_atom_stats(
        e_ref=[-10.0, -20.0], e_model=[-10.5, -19.0], natoms=[2, 4]
    )
    # per-atom errors: -0.25 and +0.25 eV/atom
    assert stats["mae"] == pytest.approx(0.25)
    assert stats["rmse"] == pytest.approx(0.25)
    assert stats["bias"] == pytest.approx(0.0)


def test_force_stats_direction_known_answers():
    ref = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    model = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.05, 0.0, 0.0]])
    stats = metrics.force_stats(ref, model)
    # component errors: atom1 0, atom2 1, atom3 0.1 -> mean (0+1+1+0.1+... )
    component = np.abs(model - ref)
    assert stats["component_mae_eV_A"] == pytest.approx(component.mean())
    # only the first two atoms exceed the 0.1 eV/A cosine threshold
    direction = stats["direction"]
    assert direction["n_atoms"] == 2
    assert direction["angle_deg_mean"] == pytest.approx(45.0)
    assert direction["frac_angle_above_30deg"] == pytest.approx(0.5)
    # the third atom (0.05 eV/A) must be excluded from direction statistics
    strata = stats["by_ref_force_magnitude"]
    assert strata[0]["n_atoms"] == 1
    assert strata[1]["n_atoms"] == 2


def test_force_stats_per_element_breakdown():
    ref = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    model = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    stats = metrics.force_stats(ref, model, symbols=["Li", "S"])
    assert stats["by_element"]["Li"]["component_mae_eV_A"] == pytest.approx(0.0)
    assert stats["by_element"]["S"]["component_mae_eV_A"] == pytest.approx(1 / 3)
    assert stats["by_element"]["Li"]["n_atoms"] == 1


def test_stress_stats_voigt_order_and_units():
    """A single +1 eV/A^3 xx offset must be reported in the xx component only."""
    ref = np.zeros((1, 6))
    model = np.zeros((1, 6))
    model[0, 0] = 1.0
    stats = metrics.stress_stats(ref, model)
    assert stats["component_mae_GPa"] == pytest.approx(1.0 / 6.0 * metrics.EV_A3_TO_GPA)
    assert stats["hydrostatic_error_bias_eV_A3"] == pytest.approx(1.0 / 3.0)
    # deviatoric part of a single-axis offset is (2/3, -1/3, -1/3)
    assert stats["deviatoric_component_mae_eV_A3"] == pytest.approx(
        (2 / 3 + 1 / 3 + 1 / 3) / 6
    )
    assert stats["voigt_order"] == ["xx", "yy", "zz", "yz", "xz", "xy"]


def test_stress_stats_hydrostatic_sign_is_not_flipped():
    ref = np.zeros((1, 6))
    model = np.full((1, 6), 0.01)
    assert metrics.stress_stats(ref, model)["hydrostatic_error_bias_eV_A3"] == pytest.approx(
        0.01
    )
    assert metrics.stress_stats(model, ref)["hydrostatic_error_bias_eV_A3"] == pytest.approx(
        -0.01
    )


def test_group_relative_energy_removes_only_the_offset():
    # a constant cross-method offset must vanish; real structure errors remain
    e_ref = [10.0, 12.0, 20.0, 24.0]
    e_model = [x + 5.0 for x in e_ref]
    e_model[1] += 0.4  # +0.2 eV/atom for a 2-atom cell
    natoms = [2, 2, 2, 2]
    groups = ["a", "a", "b", "b"]
    stats = metrics.group_relative_energy_stats(e_ref, e_model, natoms, groups)
    assert stats["relative_energy_eV_per_atom"]["mae"] == pytest.approx(0.05)
    assert stats["by_group"]["a"]["offset_eV_per_atom"] == pytest.approx(5.0 / 2 + 0.1)
    assert stats["by_group"]["a"]["relative_mae_eV_per_atom"] == pytest.approx(0.1)
    assert stats["by_group"]["b"]["relative_mae_eV_per_atom"] == pytest.approx(0.0)


def test_group_relative_energy_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        metrics.group_relative_energy_stats([1.0], [1.0], [1], ["a", "b"])


def test_formation_energy_end_points_are_exactly_zero():
    e = {"p2s5": -100.0, "li2s": -60.0, "mid": -81.0}
    n = {"p2s5": 28, "li2s": 12, "mid": 20}
    x = {"p2s5": 0.0, "li2s": 1.0, "mid": 0.5}
    ef = metrics.formation_energies(e, n, x, "p2s5", "li2s")
    assert ef["p2s5"] == pytest.approx(0.0, abs=1e-12)
    assert ef["li2s"] == pytest.approx(0.0, abs=1e-12)
    # hand-computed mid value: formula units 20/(7-2) = 4
    low = e["p2s5"] / (28 / 7)
    high = e["li2s"] / (12 / 3)
    expected = ((e["mid"] / 4) - 0.5 * high - 0.5 * low) / 5
    assert ef["mid"] == pytest.approx(expected)


def test_formation_energy_requires_both_end_points():
    with pytest.raises(ValueError, match="end point"):
        metrics.formation_energies(
            {"p2s5": -100.0, "mid": -81.0},
            {"p2s5": 28, "mid": 20},
            {"p2s5": 0.0, "mid": 0.5},
            "p2s5",
            "li2s",
        )


def test_rank_agreement_known_answers():
    perfect = metrics.rank_agreement([1, 2, 3, 4], [10, 20, 30, 40])
    assert perfect["spearman_rho"] == pytest.approx(1.0)
    assert perfect["max_rank_shift"] == pytest.approx(0.0)
    reversed_order = metrics.rank_agreement([1, 2, 3, 4], [4, 3, 2, 1])
    assert reversed_order["spearman_rho"] == pytest.approx(-1.0)
    swapped = metrics.rank_agreement([1, 2, 3], [1, 3, 2])
    assert swapped["spearman_rho"] == pytest.approx(0.5)
    assert swapped["max_rank_shift"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics.rank_agreement([1.0], [1.0])


def test_convex_hull_known_answers():
    x = {"a": 0.0, "b": 0.5, "c": 1.0}
    # 'b' lies below the a-c line and is therefore on the hull
    below = {"a": 0.0, "b": -1.0, "c": 0.0}
    assert metrics.convex_hull_keys(below, x) == ["a", "b", "c"]
    # 'b' lies above the line and must not be on the hull
    above = {"a": 0.0, "b": 1.0, "c": 0.0}
    assert metrics.convex_hull_keys(above, x) == ["a", "c"]
    agreement = metrics.hull_agreement(below, above, x)
    assert agreement["agreement"] == ["a", "c"]
    assert agreement["only_reference"] == ["b"]


def test_lattice_stats_known_answer():
    ref = np.eye(3) * 3.0
    model = np.eye(3) * 3.3
    stats = metrics.lattice_stats(ref, model)
    assert stats["volume_ref_A3"] == pytest.approx(27.0)
    assert stats["volume_rel_error"] == pytest.approx(1.1**3 - 1.0)
    assert stats["lattice_length_mae_A"] == pytest.approx(0.3)


def test_metrics_reject_non_finite_values():
    with pytest.raises(ValueError):
        metrics.abs_stats([1.0, float("nan")])
    with pytest.raises(ValueError):
        metrics.abs_stats([])


# ---------------------------------------------------------------------------
# registry contract
# ---------------------------------------------------------------------------


def test_committed_registry_is_valid_and_complete():
    registry = datasets.load_registry()
    assert registry["schema"] == datasets.REGISTRY_SCHEMA
    for expected in ("lips25", "lips20", "omat24", "ms25", "jarvis"):
        assert expected in registry["datasets"]
    for dataset_id, entry in registry["datasets"].items():
        assert entry["id"] == dataset_id
        assert entry["comparison_mode"] in datasets.COMPARISON_MODES
        assert "redistributable" in entry["license"]["data"]
        assert "stress" in entry["units"]
        assert entry["data_overlap"]["status"]
        for tier in datasets.TIERS:
            assert tier in entry["selection"]["tiers"]
    # datasets without a confirmed data licence must not be redistributable
    assert registry["datasets"]["lips25"]["license"]["data"]["redistributable"] is False
    assert registry["datasets"]["lips20"]["license"]["data"]["redistributable"] is False
    assert registry["datasets"]["jarvis"]["license"]["data"]["redistributable"] is True


def test_registry_document_matches_its_committed_json_schema():
    from evidence import schema as schema_module

    registry = json.loads(datasets.registry_path().read_text(encoding="utf-8"))
    schema = json.loads(
        (LITERATURE / "registry.schema.json").read_text(encoding="utf-8")
    )
    problems = schema_module.validate(registry, schema)
    assert problems == [], problems


def test_registry_rejects_a_broken_checksum():
    registry = json.loads(datasets.registry_path().read_text(encoding="utf-8"))
    registry["datasets"]["lips25"]["splits"]["test"]["files"][0]["sha256"] = "abc"
    with pytest.raises(datasets.RegistryError, match="sha256"):
        datasets.validate_registry(registry)


def test_registry_rejects_missing_licence_and_units():
    registry = json.loads(datasets.registry_path().read_text(encoding="utf-8"))
    del registry["datasets"]["lips25"]["license"]["data"]["redistributable"]
    del registry["datasets"]["lips25"]["units"]["stress"]
    with pytest.raises(datasets.RegistryError) as excinfo:
        datasets.validate_registry(registry)
    message = str(excinfo.value)
    assert "redistributable" in message
    assert "units.stress" in message


def test_registry_rejects_unknown_comparison_mode():
    registry = json.loads(datasets.registry_path().read_text(encoding="utf-8"))
    registry["datasets"]["jarvis"]["comparison_mode"] = "definitely-comparable"
    with pytest.raises(datasets.RegistryError, match="comparison_mode"):
        datasets.validate_registry(registry)


def test_tier_selection_is_complete_and_rejects_unknown_tiers():
    registry = datasets.load_registry()
    for tier in datasets.TIERS:
        selection = datasets.tier_selection(registry, "lips25", tier)
        assert selection["tier"] == tier
        assert selection["seed"] == registry["datasets"]["lips25"]["selection"]["seed"]
        assert selection["rules"]
    with pytest.raises(datasets.RegistryError, match="unknown tier"):
        datasets.tier_selection(registry, "lips25", "ultra")


def test_file_specs_cover_every_pinned_lips25_file():
    registry = datasets.load_registry()
    specs = datasets.file_specs(registry, "lips25")
    paths = {spec.relpath for spec in specs}
    assert "benchmarking/task_1/crystals_relaxed.extxyz" in paths
    assert "benchmarking/task_2/mq_errors.extxyz" in paths
    assert "data/lips-25/test.extxyz" in paths
    for spec in specs:
        assert spec.url.startswith("https://")
        assert len(spec.sha256) == 64


def test_file_specs_for_lips20_include_npy_and_type_files():
    registry = datasets.load_registry()
    specs = datasets.file_specs(registry, "lips20")
    relpaths = {spec.relpath for spec in specs}
    composition = registry["datasets"]["lips20"]["splits"]["lips20_3000k"][
        "compositions"
    ]["Li2S"]
    base = composition["path_prefix"]
    assert base + "set.000/energy.npy" in relpaths
    assert base + "set.000/virial.npy" in relpaths
    assert base + "type_map.raw" in relpaths
    assert len(specs) == 20 * 7


def test_cache_root_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv(datasets.CACHE_ENV, raising=False)
    assert datasets.cache_root() == datasets.DEFAULT_CACHE.expanduser()
    monkeypatch.setenv(datasets.CACHE_ENV, str(tmp_path / "env"))
    assert datasets.cache_root() == (tmp_path / "env").resolve()
    assert datasets.cache_root(tmp_path / "arg") == (tmp_path / "arg").resolve()


# ---------------------------------------------------------------------------
# cache / fetch semantics
# ---------------------------------------------------------------------------


def _tmp_registry(tmp_path: Path, sha256: str, size: int, name: str = "file.txt"):
    registry = json.loads(datasets.registry_path().read_text(encoding="utf-8"))
    entry = registry["datasets"]["lips25"]
    entry["provenance"]["raw_url_prefix"] = "https://invalid.invalid/"
    entry["splits"] = {
        "test": {
            "description": "synthetic",
            "files": [
                {
                    "id": "test",
                    "path": name,
                    "sha256": sha256,
                    "bytes": size,
                    "records": 1,
                    "labels": ["energy"],
                }
            ],
            "labels": {"energy": True, "forces": True, "stress": False},
        }
    }
    return registry


def test_offline_fetch_fails_loudly_when_the_cache_is_empty(tmp_path):
    registry = datasets.load_registry()
    with pytest.raises(datasets.DatasetAccessError, match="offline"):
        datasets.fetch_dataset(
            "lips25", cache=tmp_path, offline=True, registry=registry
        )


def test_offline_fetch_reuses_a_verified_file(tmp_path):
    payload = b"literature payload\n"
    import hashlib

    sha = hashlib.sha256(payload).hexdigest()
    registry = _tmp_registry(tmp_path, sha, len(payload))
    target = tmp_path / "lips25" / "file.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    report = datasets.fetch_dataset(
        "lips25", cache=tmp_path, offline=True, registry=registry
    )
    assert report.reused == ["file.txt"]
    assert report.downloaded == []
    assert report.entries[0]["status"] == "reused"


def test_offline_fetch_deletes_and_rejects_a_corrupted_file(tmp_path):
    import hashlib

    payload = b"correct\n"
    sha = hashlib.sha256(payload).hexdigest()
    registry = _tmp_registry(tmp_path, sha, len(payload))
    target = tmp_path / "lips25" / "file.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"tampered\n")
    with pytest.raises(datasets.DatasetAccessError, match="offline"):
        datasets.fetch_dataset("lips25", cache=tmp_path, offline=True, registry=registry)
    assert not target.exists(), "a hash-mismatched cache entry must be removed"


def test_fetch_writes_a_provenance_record(tmp_path):
    import hashlib

    payload = b"payload\n"
    sha = hashlib.sha256(payload).hexdigest()
    registry = _tmp_registry(tmp_path, sha, len(payload))
    target = tmp_path / "lips25" / "file.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    report = datasets.fetch_dataset(
        "lips25", cache=tmp_path, offline=True, registry=registry
    )
    out = datasets.write_fetch_record(report, tmp_path / "work")
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["schema"] == "mliport.literature-fetch/1"
    assert document["files"] == 1


# ---------------------------------------------------------------------------
# selection helpers
# ---------------------------------------------------------------------------


def test_evenly_spaced_indices_cover_endpoints():
    assert importers.evenly_spaced_indices(1000, 4) == [0, 333, 666, 999]
    assert importers.evenly_spaced_indices(3, 10) == [0, 1, 2]
    assert importers.evenly_spaced_indices(10, 1) == [0]
    with pytest.raises(ValueError):
        importers.evenly_spaced_indices(0, 1)


def test_stratified_indices_are_deterministic_and_balanced():
    keys = ["a"] * 10 + ["b"] * 5 + ["c"] * 2
    first = importers.stratified_indices(keys, 6, seed=7)
    second = importers.stratified_indices(keys, 6, seed=7)
    assert first == second
    assert len(first) == len(set(first)) == 6
    chosen = [keys[i] for i in first]
    assert set(chosen) == {"a", "b", "c"}
    # a different seed changes the sample, proving the seed is really used
    assert importers.stratified_indices(keys, 6, seed=8) != first
    assert importers.stratified_indices(keys, 100, seed=7) == list(range(len(keys)))


def test_stratified_indices_handles_empty_input():
    assert importers.stratified_indices([], 5, seed=1) == []


# ---------------------------------------------------------------------------
# importers
# ---------------------------------------------------------------------------


def _lips25_registry(tmp_path: Path, relaxed_path: Path, unrelaxed_path: Path):
    registry = json.loads(datasets.registry_path().read_text(encoding="utf-8"))
    entry = registry["datasets"]["lips25"]
    entry["splits"]["task1_relaxed"]["files"] = [
        {
            "id": "task1_relaxed",
            "path": str(relaxed_path),
            "sha256": datasets.sha256_file(relaxed_path),
            "bytes": relaxed_path.stat().st_size,
            "records": 3,
            "labels": ["energy", "forces"],
        }
    ]
    entry["splits"]["task1_unrelaxed"]["files"] = [
        {
            "id": "task1_unrelaxed",
            "path": str(unrelaxed_path),
            "sha256": datasets.sha256_file(unrelaxed_path),
            "bytes": unrelaxed_path.stat().st_size,
            "records": 3,
            "labels": [],
        }
    ]
    entry["splits"].pop("task2_mq", None)
    entry["splits"].pop("test", None)
    entry["selection"]["tiers"]["core"] = {
        "task1_relaxed": {"rule": "all"},
        "task1_unrelaxed": {"rule": "all"},
    }
    return registry


def _write_lips25_tie_line(tmp_path: Path) -> tuple[Path, Path]:
    """Three labelled structures in the two extxyz layouts ASE can produce."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    frames = []
    for label, x, energy in (
        ("dhf_p2s5", 0.0, -100.0),
        ("dhf_li7p3s11", 0.7, -200.0),
        ("dhf_li2s", 1.0, -50.0),
    ):
        natoms = {0.0: 28, 0.7: 42, 1.0: 12}[x]
        atoms = Atoms(
            "Li" * natoms, positions=np.zeros((natoms, 3)), cell=np.eye(3) * 10.0,
            pbc=True,
        )
        if x == 0.0:
            # dataset-specific keys, read back into info/arrays
            atoms.info["dft_energy"] = energy
            atoms.arrays["dft_forces"] = np.zeros((natoms, 3))
        else:
            atoms.calc = SinglePointCalculator(
                atoms, energy=energy, forces=np.zeros((natoms, 3))
            )
        atoms.info["label"] = label
        atoms.info["virial"] = np.zeros(9)  # provenance only, never stress
        frames.append(atoms)
    labelled = tmp_path / "crystals_relaxed.extxyz"
    write(labelled, frames, format="extxyz")
    unlabelled_frames = []
    for atoms in frames:
        copy = atoms.copy()
        copy.calc = None
        copy.info.pop("dft_energy", None)
        copy.arrays.pop("dft_forces", None)
        unlabelled_frames.append(copy)
    unlabelled = tmp_path / "crystals_unrelaxed.extxyz"
    write(unlabelled, unlabelled_frames, format="extxyz")
    return labelled, unlabelled


def test_import_lips25_reads_both_label_layouts_and_annotates_tie_line(tmp_path):
    labelled, unlabelled = _write_lips25_tie_line(tmp_path)
    registry = _lips25_registry(tmp_path, labelled, unlabelled)
    entry = registry["datasets"]["lips25"]
    selection = datasets.tier_selection(registry, "lips25", "core")
    out = tmp_path / "out"
    document = importers.import_lips25(entry, tmp_path, out, selection)

    from ase.io import read

    frames = read(out / "task1_relaxed.extxyz", index=":", format="extxyz")
    assert len(frames) == 3
    assert frames[1].info["li_ps_x"] == pytest.approx(0.7)
    assert frames[1].info["literature_group"] == "li2s-p2s5-tie-line"
    assert frames[1].info["literature_phase"] == "dhf_li7p3s11"
    assert frames[1].info["reference_virial_used_as_stress"] is False
    assert frames[1].calc.get_potential_energy(frames[1]) == pytest.approx(-200.0)
    assert document["splits"]["task1_relaxed"]["selected_records"] == 3
    assert document["comparison_mode"] == "cross_method"
    assert (out / "selection.json").is_file()

    unlabelled_frames = read(
        out / "task1_unrelaxed.extxyz", index=":", format="extxyz"
    )
    assert len(unlabelled_frames) == 3
    assert unlabelled_frames[0].calc is None


def test_import_lips20_builds_atoms_and_keeps_virial_out_of_stress(tmp_path):
    registry = json.loads(datasets.registry_path().read_text(encoding="utf-8"))
    entry = registry["datasets"]["lips20"]
    base = tmp_path / "cache" / "Dataset"
    set_dir = base / "set.000"
    set_dir.mkdir(parents=True)
    natoms, nframes = 4, 3
    symbols = ["Li", "Li", "S", "S"]
    np.save(set_dir / "energy.npy", np.array([-10.0, -11.0, -12.0]))
    coords = np.zeros((nframes, natoms * 3))
    coords[:, 0] = [0.0, 0.1, 0.2]
    np.save(set_dir / "coord.npy", coords)
    np.save(set_dir / "force.npy", np.full((nframes, natoms * 3), 0.5))
    np.save(set_dir / "box.npy", np.tile(np.eye(3).ravel() * 10.0, (nframes, 1)))
    np.save(set_dir / "virial.npy", np.full((nframes, 9), 1.5))
    (base / "type_map.raw").write_text("Li S\n", encoding="utf-8")
    (base / "type.raw").write_text("0 0 1 1\n", encoding="utf-8")
    entry["splits"]["lips20_3000k"]["compositions"] = {
        "Li2S": {
            "composition": "Li2S",
            "formula": "Li2S",
            "temperature_K": 3000,
            "path_prefix": "Dataset/",
            "sha256": {
                name: datasets.sha256_file(set_dir / name)
                for name in (
                    "box.npy",
                    "coord.npy",
                    "energy.npy",
                    "force.npy",
                    "virial.npy",
                )
            },
            "bytes": {
                name: (set_dir / name).stat().st_size
                for name in (
                    "box.npy",
                    "coord.npy",
                    "energy.npy",
                    "force.npy",
                    "virial.npy",
                )
            },
            "type_map": ["Li", "S"],
            "type_map_sha256": datasets.sha256_file(base / "type_map.raw"),
            "type_sha256": datasets.sha256_file(base / "type.raw"),
            "frames": nframes,
            "natoms": natoms,
        }
    }
    entry["selection"]["tiers"]["core"]["lips20_3000k"] = {
        "rule": "evenly_spaced_per_composition",
        "count": 2,
    }
    selection = datasets.tier_selection(registry, "lips20", "core")
    out = tmp_path / "out"
    document = importers.import_lips20(entry, tmp_path / "cache", out, selection)

    from ase.io import read

    frames = read(out / "lips20_3000k.extxyz", index=":", format="extxyz")
    assert len(frames) == 2
    assert frames[0].get_chemical_symbols() == symbols
    assert frames[0].calc.get_potential_energy(frames[0]) == pytest.approx(-10.0)
    assert frames[0].info["reference_virial_present"] is True
    assert frames[0].info["reference_virial_used_as_stress"] is False
    assert document["splits"]["lips20_3000k"]["compositions"]["Li2S"][
        "virial_used_as_stress"
    ] is False
    assert frames[0].cell[0, 0] == pytest.approx(10.0)


def test_import_lips20_rejects_inconsistent_atom_counts(tmp_path):
    registry = json.loads(datasets.registry_path().read_text(encoding="utf-8"))
    entry = registry["datasets"]["lips20"]
    base = tmp_path / "cache" / "Dataset"
    set_dir = base / "set.000"
    set_dir.mkdir(parents=True)
    np.save(set_dir / "energy.npy", np.array([-10.0]))
    np.save(set_dir / "coord.npy", np.zeros((1, 6)))
    np.save(set_dir / "force.npy", np.zeros((1, 6)))
    np.save(set_dir / "box.npy", np.eye(3).ravel().reshape(1, 9))
    np.save(set_dir / "virial.npy", np.zeros((1, 9)))
    (base / "type_map.raw").write_text("Li S\n", encoding="utf-8")
    (base / "type.raw").write_text("0 0 0\n", encoding="utf-8")  # 3 types, 2 atoms
    entry["splits"]["lips20_3000k"]["compositions"] = {
        "Li2S": {
            "formula": "Li2S",
            "temperature_K": 3000,
            "path_prefix": "Dataset/",
            "sha256": {"energy.npy": "0" * 64},
            "type_map": ["Li", "S"],
            "type_map_sha256": "0" * 64,
            "type_sha256": "0" * 64,
            "frames": 1,
            "natoms": 2,
        }
    }
    selection = datasets.tier_selection(registry, "lips20", "core")
    with pytest.raises(datasets.DatasetAccessError, match="inconsistent atom counts"):
        importers.import_lips20(entry, tmp_path / "cache", tmp_path / "out", selection)


def test_import_jarvis_selects_polymorph_groups(tmp_path):
    records = []
    for group in ("Li2O", "NaCl"):
        for index, energy in enumerate((-2.0, -1.5, -1.0)):
            records.append(
                {
                    "jid": f"JVASP-{group}-{index}",
                    "formula": group,
                    "natoms": 4,
                    "func": "OptB88vdW",
                    "encut": 650,
                    "formation_energy_peratom": -0.5,
                    "optb88vdw_total_energy": energy,
                    "atoms": {
                        "lattice_mat": np.eye(3).tolist(),
                        "coords": [[0, 0, 0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.7, 0.7, 0.7]],
                        "elements": ["Li", "Li", "O", "O"],
                        "cartesian": True,
                    },
                }
            )
    archive = tmp_path / "jdft.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("jdft.json", json.dumps(records))
    registry = json.loads(datasets.registry_path().read_text(encoding="utf-8"))
    entry = registry["datasets"]["jarvis"]
    entry["provenance"]["file"] = {
        "name": "jdft.zip",
        "download_url": "https://invalid.invalid/jdft.zip",
    }
    entry["provenance"]["inner_file"] = {"name": "jdft.json"}
    entry["selection"]["filters"]["natoms_min"] = 2
    entry["selection"]["filters"]["natoms_max"] = 6
    entry["selection"]["filters"]["min_polymorphs"] = 3
    entry["selection"]["tiers"]["core"]["polymorphs"] = {
        "rule": "seeded_formula_groups",
        "groups": 2,
        "per_group": 3,
    }
    selection = datasets.tier_selection(registry, "jarvis", "core")
    out = tmp_path / "out"
    document = importers.import_jarvis(entry, tmp_path, out, selection)
    from ase.io import read

    frames = read(out / "polymorphs.extxyz", index=":", format="extxyz")
    assert len(frames) == 6
    # energies are stored as totals (eV/atom * N) so the reported metric is
    # eV/atom independent of the cell size
    assert frames[0].calc.get_potential_energy(frames[0]) / len(frames[0]) == pytest.approx(
        -2.0
    )
    assert frames[0].info["jarvis_encut_eV"] == 650
    assert len(document["splits"]["polymorphs"]["groups"]) == 2


def test_import_local_extxyz_requires_and_verifies_the_manifest(tmp_path):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    atoms = Atoms("Li2", positions=[[0, 0, 0], [1, 0, 0]], cell=np.eye(3) * 8, pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-5.0, forces=np.zeros((2, 3)))
    write(tmp_path / "ref.xyz", atoms, format="extxyz")
    manifest = {
        "schema": "mliport.literature-local-manifest/1",
        "dataset": "ms25",
        "subset": "mgo_surfaces",
        "license": {"id": "user-supplied", "redistributable": False},
        "reference_level": "PBE (VASP)",
        "dft": {"software": "VASP", "functional": "PBE"},
        "comparison_mode": "cross_method",
        "units": {"energy": "eV", "forces": "eV/Angstrom", "stress": None},
        "files": [
            {
                "path": "ref.xyz",
                "sha256": datasets.sha256_file(tmp_path / "ref.xyz"),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registry = datasets.load_registry()
    entry = registry["datasets"]["ms25"]
    selection = datasets.tier_selection(registry, "ms25", "core")
    document = importers.import_local_extxyz(
        entry, tmp_path, tmp_path / "out", selection, manifest_path=manifest_path
    )
    assert document["splits"]["mgo_surfaces"]["selected_records"] == 1
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(datasets.DatasetAccessError, match="sha256"):
        importers.import_local_extxyz(
            entry, tmp_path, tmp_path / "out2", selection, manifest_path=manifest_path
        )


def test_import_ms25_without_a_manifest_is_blocked():
    registry = datasets.load_registry()
    entry = registry["datasets"]["ms25"]
    selection = datasets.tier_selection(registry, "ms25", "core")
    with pytest.raises(datasets.DatasetAccessError, match="reference-manifest"):
        importers.import_local_extxyz(entry, Path("."), Path("."), selection)


def test_import_omat24_reuses_the_committed_pinned_subset(tmp_path):
    registry = datasets.load_registry()
    document = importers.import_dataset(
        "omat24",
        tier="core",
        out_dir=tmp_path,
        registry=registry,
        fetch=False,
        offline=True,
    )
    from ase.io import read

    frames = read(tmp_path / "val256.extxyz", index=":", format="extxyz")
    assert len(frames) == 32
    assert document["source"]["sha256"] == registry["datasets"]["omat24"]["splits"][
        "val256"
    ]["sha256"]
    assert frames[0].info["comparison_mode"] == "reference_matched"


def test_import_omat24_refuses_a_changed_subset(tmp_path):
    registry = json.loads(datasets.registry_path().read_text(encoding="utf-8"))
    registry["datasets"]["omat24"]["splits"]["val256"]["sha256"] = "1" * 64
    with pytest.raises(datasets.DatasetAccessError, match="hash changed"):
        importers.import_dataset(
            "omat24",
            tier="core",
            out_dir=tmp_path,
            registry=registry,
            fetch=False,
            offline=True,
        )


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------


def test_tie_line_key_accepts_prefixed_and_raw_labels():
    assert literature_eval._tie_line_key("dhf_li2s") == "dhf_li2s"
    assert literature_eval._tie_line_key("li2s") == "dhf_li2s"
    assert literature_eval._tie_line_key("not-a-phase") is None


def test_formation_energy_block_matches_a_known_offset(tmp_path):
    """A model that is offset by +c eV/atom everywhere has a known Ef error."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    phases = [
        ("dhf_p2s5", "P2S5", 28, -100.0),
        ("dhf_li7p3s11", "Li7P3S11", 42, -200.0),
        ("dhf_li2s", "Li2S", 12, -50.0),
    ]
    frames = []
    for label, _formula, natoms, energy in phases:
        atoms = Atoms("Li" * natoms, cell=np.eye(3) * 10, pbc=True)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((natoms, 3)))
        atoms.info["label"] = label
        atoms.info["literature_phase"] = label
        atoms.info["literature_group"] = "li2s-p2s5-tie-line"
        frames.append(atoms)
    path = tmp_path / "task1_relaxed.extxyz"
    write(path, frames, format="extxyz")

    offset = 0.5  # eV/atom
    rows = []
    for atoms in frames:
        natoms = len(atoms)
        rows.append(
            {
                "ok": True,
                "group": "li2s-p2s5-tie-line",
                "phase": atoms.info["literature_phase"],
                "natoms": natoms,
                "e_model_eV": atoms.calc.get_potential_energy(atoms) + offset * natoms,
            }
        )
    block = literature_eval._formation_energy_block(
        path, "task1_relaxed", rows, relaxation=None
    )
    assert block["available"] is True
    # Ef is normalised per atom and both end points are subtracted, so a
    # uniform +0.5 eV/atom offset cancels exactly.
    assert block["rmse_eV_per_atom"] == pytest.approx(0.0, abs=1e-12)
    assert block["rank_agreement"]["spearman_rho"] == pytest.approx(1.0)


def test_formation_energy_block_reports_missing_phases(tmp_path):
    frame_path = tmp_path / "empty.extxyz"
    from ase import Atoms
    from ase.io import write

    write(frame_path, Atoms("Li"), format="extxyz")
    block = literature_eval._formation_energy_block(
        tmp_path / "task1_relaxed.extxyz", "task1_relaxed", [], None
    )
    assert block["available"] is False
    assert "end points" in block["reason"]


def test_labels_available_detects_unlabelled_reference():
    from ase import Atoms

    atoms = Atoms("Li", positions=[[0, 0, 0]], cell=np.eye(3) * 6, pbc=True)
    assert literature_eval._labels_available([atoms]) == {
        "energy": False,
        "forces": False,
        "stress": False,
    }


def test_dataset_split_parsing_matches_the_test_id_convention():
    assert literature_report._dataset_split("t9_lips25_task1_relaxed") == (
        "lips25",
        "task1_relaxed",
    )
    assert literature_report._dataset_split("t9_omat24_val256") == ("omat24", "val256")


# ---------------------------------------------------------------------------
# orchestrators
# ---------------------------------------------------------------------------


def test_run_suite_exposes_t9_and_requires_dataset_and_split():
    sys.path.insert(0, str(SCRIPTS))
    import run_suite

    assert run_suite.TIER_COMMANDS["t9"] == ["literature_eval.py"]
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_suite.py"),
            "--tiers",
            "t9",
            "--engine",
            "mace",
            "--profile-id",
            "mace_omat",
            "--model",
            "unused",
            "--manifest",
            str(SCIENCE / "model_manifest.json"),
            "--out",
            str(Path("/tmp") / "mliport-t9-argcheck"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--dataset and --split" in result.stderr


def test_literature_engines_are_the_four_pinned_profiles():
    import engines

    manifest = json.loads(
        (SCIENCE / "model_manifest.json").read_text(encoding="utf-8")
    )
    assert set(engines._ENGINES) == {"uma", "mace", "dpa", "grace"}
    manifest_engines = {profile["engine"] for profile in manifest["profiles"].values()}
    assert manifest_engines == set(engines._ENGINES)
    # NEP is a separate product runtime; t9 must refuse it before loading
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "literature_eval.py"),
            "--engine",
            "nep",
            "--profile-id",
            "nep",
            "--model",
            "x",
            "--manifest",
            str(SCIENCE / "model_manifest.json"),
            "--dataset",
            "lips25",
            "--split",
            "test",
            "--subset",
            "x",
            "--out",
            "x",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_benchmark_tier_plan_is_monotone():
    sys.path.insert(0, str(SCRIPTS))
    import benchmark

    assert set(benchmark.TIER_PLAN) == set(datasets.TIERS)
    assert benchmark.TIER_PLAN["core"]["tiers"] == []
    assert "t4" in benchmark.TIER_PLAN["standard"]["tiers"]
    assert set(benchmark.TIER_PLAN["standard"]["tiers"]) <= set(
        benchmark.TIER_PLAN["research"]["tiers"]
    )
    assert {"task1_unrelaxed"} == benchmark.RELAX_SPLITS


def test_benchmark_import_only_writes_resolved_config(tmp_path):
    """``--import-only`` must work with no backend and no network."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "benchmark.py"),
            "--tier",
            "core",
            "--datasets",
            "omat24",
            "--engine",
            "mace",
            "--profile-id",
            "mace_omat",
            "--model",
            str(tmp_path / "not-a-model.model"),
            "--manifest",
            str(SCIENCE / "model_manifest.json"),
            "--out",
            str(tmp_path / "work"),
            "--import-only",
            "--offline",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    resolved = json.loads((tmp_path / "work" / "resolved.json").read_text(encoding="utf-8"))
    assert resolved["schema"] == "mliport.literature-benchmark-config/1"
    assert resolved["tier"] == "core"
    assert resolved["datasets"] == ["omat24"]
    assert resolved["selections"]["omat24"]["splits"]["val256"]["selected_records"] == 32
    assert (tmp_path / "work" / "data" / "omat24" / "val256.extxyz").is_file()


def test_benchmark_rejects_an_unknown_dataset(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "benchmark.py"),
            "--datasets",
            "made-up",
            "--engine",
            "mace",
            "--profile-id",
            "mace_omat",
            "--model",
            "x",
            "--manifest",
            str(SCIENCE / "model_manifest.json"),
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unknown dataset" in result.stderr


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _synthetic_record(status: str, **metric_overrides) -> dict:
    metrics = {
        "dataset": "lips25",
        "split": "test",
        "comparison_mode": "cross_method",
        "reference_level": "PBEsol",
        "n_structures": 4,
        "n_evaluated": 4,
        "energy_per_atom_eV": {"mae": 0.3, "rmse": 0.35, "p95": 0.4},
        "relative_energy": {
            "available": True,
            "relative_energy_eV_per_atom": {"mae": 0.05, "rmse": 0.06, "p95": 0.08},
        },
        "forces": {
            "available": True,
            "component_mae_eV_A": 0.1,
            "component_rmse_eV_A": 0.12,
            "vector_norm_error_eV_A": {"p95": 0.4, "p99": 0.5},
            "direction": {"angle_deg_mean": 4.0, "angle_deg_median": 3.0,
                          "angle_deg_p95": 10.0, "frac_angle_above_10deg": 0.1},
            "by_element": {"Li": {"component_mae_eV_A": 0.05}},
        },
        "stress": {"available": False, "reason": "no verified stress label"},
        "limitations": ["cross-method reference"],
        "labels_available_in_reference": {"energy": True, "forces": True, "stress": False},
    }
    metrics.update(metric_overrides)
    return common.result_record(
        case_id="t9_lips25",
        test_id="t9_lips25_test",
        status=status,
        engine="mace",
        model_identity="synthetic",
        model_sha256="a" * 64,
        task="bulk",
        head=None,
        dtype="float64",
        device={"requested": "cpu", "actual": "cpu", "gpu_name": None, "gpu_uuid_hash": None},
        input_structure_id="b" * 64,
        parameters={},
        metrics=metrics,
        diagnostics={},
    )


def test_report_summary_and_markdown_are_generated(tmp_path):
    common.write_result(_synthetic_record("pass"), tmp_path / "t9")
    records, bundle = literature_report._t9_records(tmp_path)
    summary = literature_report.build_summary(
        records, datasets.load_registry(), bundle, tier="core"
    )
    assert summary["schema"] == literature_report.REPORT_SCHEMA
    assert len(summary["entries"]) == 1
    entry = summary["entries"][0]
    assert entry["dataset"] == "lips25"
    assert entry["forces"]["component_mae_eV_A"] == pytest.approx(0.1)
    markdown = literature_report.render_markdown(summary, "en")
    assert re.search(r"\| lips25 \| test \| [^|]+ \| pass \| cross_method \|", markdown)
    assert "Formation energy" in markdown or "no verified stress" in markdown
    chinese = literature_report.render_markdown(summary, "cn")
    assert "文献基准测试报告" in chinese


def test_report_lists_blocked_records_separately(tmp_path):
    common.write_result(
        _synthetic_record(
            "blocked",
            energy_per_atom_eV={"available": False, "reason": "no data"},
            relative_energy={"available": False},
            forces={"available": False},
        ),
        tmp_path / "t9",
    )
    records, bundle = literature_report._t9_records(tmp_path)
    summary = literature_report.build_summary(
        records, datasets.load_registry(), bundle
    )
    markdown = literature_report.render_markdown(summary, "en")
    assert "Not run / blocked" in markdown
    assert "blocked" in markdown
