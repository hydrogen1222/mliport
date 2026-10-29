"""End-to-end search-engine tests on the synthetic egg-crate model (spec 13.2).

The model has three known sites with exactly known energies:

* hollow (1/2, 1/2) -> -0.60 eV
* bridge (1/2, 0)   -> -0.35 eV
* top    (0, 0)     -> -0.20 eV

The internal terms vanish at their minimum geometry, so a relaxed molecule on a
site has exactly the site energy.  The slab is fixed, which enables the
geometric identity rule G of spec section 9.3 revision 4.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read

from mliport.confsearch.checkpoint import ConfsearchCheckpointStore
from mliport.confsearch.engine import PreparedStart, run_confsearch
from mliport.confsearch.options import ConfsearchOptions
from mliport.confsearch.regions import resolve_regions, validate_all
from tests.mliport.mliport.confsearch.calculators import (
    EggCrateCalculator,
    MorseBreakCalculator,
    NonFiniteCalculator,
)

SLAB_ATOMS = 16
MOLECULE_ATOMS = 3
SITE_ENERGIES = {(0.5, 0.5): -0.60, (0.5, 0.0): -0.35, (0.0, 0.0): -0.20}


def _molecule(angle_deg: float = 109.47) -> Atoms:
    angle = np.deg2rad(angle_deg)
    return Atoms(
        "CSO",
        positions=[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.5 - 1.5 * np.cos(angle), 1.5 * np.sin(angle), 0.0],
        ],
        pbc=False,
    )


def _slab() -> Atoms:
    spacing = 3.0
    return Atoms(
        "S" * SLAB_ATOMS,
        positions=[
            [0.5 + spacing * i, 0.5 + spacing * j, 1.0]
            for i in range(4)
            for j in range(4)
        ],
        cell=[12.0, 12.0, 20.0],
        pbc=True,
    )


def _start(site: tuple[float, float], *, fixed: bool = True) -> Atoms:
    slab = _slab()
    molecule = _molecule()
    shift = np.array([site[0] * 12.0, site[1] * 12.0, 2.0])
    atoms = Atoms(
        numbers=np.concatenate([slab.numbers, molecule.numbers]),
        positions=np.concatenate(
            [slab.get_positions(), molecule.get_positions() + shift]
        ),
        cell=slab.cell,
        pbc=slab.pbc,
    )
    if fixed:
        atoms.set_constraint(FixAtoms(indices=list(range(SLAB_ATOMS))))
    return atoms


def _options(**overrides) -> ConfsearchOptions:
    base = {
        "cs_starts": ("starts.xyz",),
        "cs_adsorbate_indices": "16-18",
        "cs_cycles": 2,
        "cs_hold_ps": 0.05,
        "cs_cool_ps": 0.05,
        "cs_snapshot_interval_ps": 0.02,
        "timestep": 0.5,
        "cs_temperature_high": 800.0,
        "cs_temperature_low": 300.0,
        "cs_quench_fmax": 0.01,
        "cs_quench_steps": 2000,
        "cs_refine": True,
        "cs_refine_fmax": 0.005,
        "cs_refine_steps": 2000,
    }
    base.update(overrides)
    return ConfsearchOptions(**base)


def _prepared(sites, *, fixed: bool = True, options=None):
    options = options or _options()
    starts = []
    for index, site in enumerate(sites):
        atoms = _start(site, fixed=fixed)
        regions = resolve_regions(
            atoms, options, adsorbate_indices=np.arange(SLAB_ATOMS, len(atoms))
        )
        surface = validate_all(atoms, regions, options)
        starts.append(
            PreparedStart(
                index=index,
                atoms=atoms,
                regions=regions,
                surface=surface,
                label=f"start_{index:03d}",
            )
        )
    return starts


def _run(tmp_path, sites, *, seed=7, calculator=None, options=None, name="run"):
    options = options or _options()
    starts = _prepared(sites, options=options)
    output = tmp_path / name
    results = run_confsearch(
        calculator or EggCrateCalculator(),
        options,
        starts=starts,
        output_dir=output,
        seed=seed,
        model_record={"model_type": "synthetic", "task": "bulk"},
        gradient_preflight={"status": "passed"},
    )
    return output, results


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_search_finds_every_known_minimum_with_its_exact_energy(tmp_path):
    sites = list(SITE_ENERGIES)
    output, results = _run(tmp_path, sites)
    assert results["status"] == "completed"
    entries = results["ranking"]["refined"] + results["ranking"]["unrefined"]
    energies = sorted(entry["energy_eV"] for entry in entries)
    for expected in sorted(SITE_ENERGIES.values()):
        assert any(abs(energy - expected) < 1e-6 for energy in energies), (
            expected,
            energies,
        )
    # Every known minimum was reached, and the baseline (start quench only)
    # already reports the deepest one.
    assert results["baseline"]["best_energy_eV"] == pytest.approx(-0.60, abs=1e-6)
    assert results["diagnostics"]["n_quenches"] >= len(sites)
    assert (output / "search_results.json").is_file()
    payload = json.loads((output / "search_results.json").read_text())
    assert payload["schema"] == "mliport.confsearch-results/1"


def test_same_basin_quenches_merge_through_the_geometry_rule(tmp_path):
    # Three replicas start at the same site: every quench must merge into one
    # unique minimum with hit_count >= 3 (G is enabled by the fixed slab).
    sites = [(0.5, 0.5)] * 3
    _output, results = _run(tmp_path, sites)
    entries = results["ranking"]["refined"] + results["ranking"]["unrefined"]
    deepest = [entry for entry in entries if abs(entry["energy_eV"] + 0.60) < 1e-6]
    assert deepest
    assert max(entry["hit_count"] for entry in deepest) >= 3
    assert max(entry["n_replicas"] for entry in deepest) >= 2


def test_fixed_atoms_never_move_and_slab_is_reported(tmp_path):
    output, results = _run(tmp_path, [(0.5, 0.5)], name="fixed")
    assert results["starts"][0]["regions"]["fixed"] == list(range(SLAB_ATOMS))
    registry = results["diagnostics"]["registry"]
    representative = registry["classes"]["valid"][0]["representative"]
    reference = read(output / "top" / "rank_001" / "structure.extxyz")
    slab_positions = np.asarray(reference.get_positions())[:SLAB_ATOMS]
    assert np.allclose(slab_positions[:, 2], 1.0)
    assert len(representative["positions_A"]) == SLAB_ATOMS + MOLECULE_ATOMS


def test_exports_re_read_equal_to_the_minimum_up_to_lattice_translations(tmp_path):
    output, results = _run(tmp_path, [(0.5, 0.5)], name="exports")
    assert results["exports"]
    export_dir = output / "top" / "rank_001"
    for name in (
        "POSCAR",
        "structure.extxyz",
        "cp2k_coord.xyz",
        "cp2k_cell.inc",
        "cp2k_fixed_atoms.inc",
        "minimum.json",
    ):
        assert (export_dir / name).is_file(), name

    payload = json.loads((export_dir / "minimum.json").read_text())
    representative = payload["representative"]
    reference = Atoms(
        numbers=np.asarray(representative["numbers"], dtype=int),
        positions=np.asarray(representative["positions_A"], dtype=float),
        cell=np.asarray(representative["cell_A"], dtype=float),
        pbc=np.asarray(representative["pbc"], dtype=bool),
    )
    exported = read(export_dir / "structure.extxyz")
    cell = np.asarray(exported.cell, dtype=float)
    fractional = (
        np.asarray(exported.get_positions(), dtype=float)
        - np.asarray(reference.get_positions(), dtype=float)
    ) @ np.linalg.inv(cell)
    residual = fractional - np.round(fractional)
    assert np.all(np.abs(residual) < 1e-8), np.abs(residual).max()
    assert exported.get_pbc().all()

    fixed_inc = (export_dir / "cp2k_fixed_atoms.inc").read_text()
    assert fixed_inc.startswith("# CP2K")
    assert "LIST 1..16" in fixed_inc
    cell_inc = (export_dir / "cp2k_cell.inc").read_text().splitlines()
    assert len(cell_inc) == 3
    assert all(line.startswith(("A ", "B ", "C ")) for line in cell_inc)

    poscar = (export_dir / "POSCAR").read_text()
    assert "Selective dynamics" in poscar
    flag_lines = [
        line.split()[-3:] for line in poscar.splitlines() if len(line.split()) >= 6
    ]
    assert sum(1 for flags in flag_lines if flags == ["F", "F", "F"]) >= SLAB_ATOMS


def test_search_is_bitwise_reproducible_for_the_same_seed(tmp_path):
    sites = list(SITE_ENERGIES)
    first_dir, first = _run(tmp_path, sites, seed=11, name="first")
    second_dir, second = _run(tmp_path, sites, seed=11, name="second")
    assert first["ranking"] == second["ranking"]
    assert first["diagnostics"]["n_quenches"] == second["diagnostics"]["n_quenches"]
    raw_first = sorted(
        (path.relative_to(first_dir).as_posix(), path.read_bytes())
        for path in first_dir.rglob("*")
        if path.is_file() and "checkpoints" not in path.parts
    )
    raw_second = sorted(
        (path.relative_to(second_dir).as_posix(), path.read_bytes())
        for path in second_dir.rglob("*")
        if path.is_file() and "checkpoints" not in path.parts
    )
    assert [name for name, _ in raw_first] == [name for name, _ in raw_second]


def test_temperature_schedule_is_linear_and_recorded(tmp_path):
    options = _options(cs_ramp_interval=10)
    output, _results = _run(tmp_path, [(0.5, 0.5)], options=options, name="ramp")
    cycle = json.loads(
        (output / "raw" / "replica_000" / "cycle_001" / "cycle.json").read_text()
    )
    schedule = cycle["temperature_schedule"]
    assert schedule
    assert schedule[0]["target_temperature_K"] == pytest.approx(800.0)
    n_cool = options.n_cool
    for entry in schedule:
        fraction = (entry["step"] - options.n_hold) / n_cool
        expected = 800.0 + (300.0 - 800.0) * fraction
        assert entry["target_temperature_K"] == pytest.approx(expected)
    assert cycle["status"] == "completed"
    assert cycle["snapshot_steps"] == [40, 80]


def test_raw_md_output_uses_the_shared_csv_header(tmp_path):
    output, _results = _run(tmp_path, [(0.5, 0.5)], name="raw")
    csv_path = output / "raw" / "replica_000" / "cycle_001" / "md.csv"
    header = csv_path.read_text().splitlines()[0]
    from mliport.runners.md_output import MD_CSV_HEADER

    assert header == ",".join(MD_CSV_HEADER)
    assert (output / "raw" / "replica_000" / "cycle_001" / "trajectory.traj").is_file()


def test_force_abort_is_recorded_and_the_search_continues(tmp_path):
    options = _options(cs_temperature_high=1200.0, fmax_abort=2.0)
    output, results = _run(tmp_path, [(0.5, 0.5)], options=options, name="abort")
    cycle = json.loads(
        (output / "raw" / "replica_000" / "cycle_001" / "cycle.json").read_text()
    )
    assert cycle["status"] == "md_force_abort"
    assert "fmax_abort" in cycle["message"]
    assert results["status"] == "completed"


def test_non_finite_energy_is_recorded_as_an_abort(tmp_path):
    options = _options(cs_temperature_high=1200.0)
    output, _results = _run(
        tmp_path,
        [(0.5, 0.5)],
        calculator=NonFiniteCalculator(fail_after=6),
        options=options,
        name="nonfinite",
    )
    cycle = json.loads(
        (output / "raw" / "replica_000" / "cycle_001" / "cycle.json").read_text()
    )
    assert cycle["status"] == "md_nonfinite_abort"


def test_morse_model_reports_an_integrity_abort_or_reacted_minimum(tmp_path):
    options = _options(
        cs_temperature_high=1500.0,
        cs_temperature_low=300.0,
        cs_hold_ps=0.1,
        cs_cool_ps=0.05,
        cs_snapshot_interval_ps=0.02,
    )
    output, results = _run(
        tmp_path,
        [(0.5, 0.5)],
        calculator=MorseBreakCalculator(D_e=0.03),
        options=options,
        name="morse",
    )
    statuses = [
        json.loads(
            (
                output / "raw" / "replica_000" / f"cycle_{cycle:03d}" / "cycle.json"
            ).read_text()
        )["status"]
        for cycle in (1, 2)
    ]
    reacted = results["diagnostics"]["registry"]["classes"].get("reacted", [])
    assert "md_integrity_abort" in statuses or reacted


# ---------------------------------------------------------------------------
# checkpoint / resume
# ---------------------------------------------------------------------------


def test_resume_reproduces_the_uninterrupted_search(tmp_path, monkeypatch):
    from mliport.confsearch.engine import SearchEngine

    sites = list(SITE_ENERGIES)
    reference_dir, reference = _run(tmp_path, sites, seed=23, name="reference")

    # Simulate a crash right after the second checkpointed unit.
    options = _options()
    starts = _prepared(sites, options=options)
    interrupted_dir = tmp_path / "interrupted"
    original = SearchEngine._write_checkpoint
    counter = {"n": 0}

    def crash_after_two(self):
        original(self)
        counter["n"] += 1
        if counter["n"] >= 2:
            raise KeyboardInterrupt("simulated crash")

    monkeypatch.setattr(SearchEngine, "_write_checkpoint", crash_after_two)
    with pytest.raises(KeyboardInterrupt):
        run_confsearch(
            EggCrateCalculator(),
            options,
            starts=starts,
            output_dir=interrupted_dir,
            seed=23,
            model_record={"model_type": "synthetic", "task": "bulk"},
            gradient_preflight={"status": "passed"},
        )
    monkeypatch.undo()

    store = ConfsearchCheckpointStore(interrupted_dir)
    data, fingerprint = store.load()
    assert 0 < data.unit_index < 1 + len(sites) * options.cs_cycles

    resumed = run_confsearch(
        EggCrateCalculator(),
        options,
        starts=starts,
        output_dir=interrupted_dir,
        seed=23,
        model_record={"model_type": "synthetic", "task": "bulk"},
        gradient_preflight={"status": "passed"},
        resume=data,
        resume_fingerprint=fingerprint,
    )
    assert resumed["ranking"] == reference["ranking"]
    assert (
        resumed["diagnostics"]["n_quenches"] == reference["diagnostics"]["n_quenches"]
    )


def test_resume_fingerprint_refuses_a_changed_option(tmp_path):
    from mliport.confsearch.checkpoint import (
        build_resume_fingerprint,
        compare_fingerprints,
    )

    options = _options()
    first, _sha = build_resume_fingerprint(
        options=options,
        model={"model_type": "synthetic"},
        input_hashes={"starts": "abc"},
        regions={"adsorbate": [16, 17, 18]},
        seed=1,
        mliport_version="2.0.0b4",
    )
    second, _sha = build_resume_fingerprint(
        options=replace(options, cs_temperature_high=900.0),
        model={"model_type": "synthetic"},
        input_hashes={"starts": "abc"},
        regions={"adsorbate": [16, 17, 18]},
        seed=1,
        mliport_version="2.0.0b4",
    )
    differences = compare_fingerprints(first, second)
    assert differences
    assert any("cs_temperature_high" in item for item in differences)


def test_checkpoint_round_trip_restores_the_registry(tmp_path):
    output, results = _run(tmp_path, [(0.5, 0.5)], name="roundtrip")
    store = ConfsearchCheckpointStore(output)
    data, _fingerprint = store.load()
    assert data.registry["classes"]
    assert data.rng_states["replicas"]
    assert data.unit_index == 1 + 1 * 2  # start stage + two cycles
    assert data.registry["schema"] == "mliport.confsearch-registry/1"
    assert (
        results["diagnostics"]["registry"]["schema"] == "mliport.confsearch-registry/1"
    )


def test_interrupted_raw_directories_are_archived_not_deleted(tmp_path):
    store = ConfsearchCheckpointStore(tmp_path / "run")
    source = tmp_path / "run" / "raw" / "replica_000" / "cycle_001"
    source.mkdir(parents=True)
    (source / "md.csv").write_text("partial\n")
    archived = store.archive_incomplete(replica=0, cycle=1)
    assert archived is not None
    assert archived.name == "incomplete_attempt_1"
    assert (archived / "md.csv").read_text() == "partial\n"
    assert not source.exists()


def test_log_reports_cycle_outcomes_md_heartbeat_and_progress(tmp_path):
    """A long search must not be silent between cycles (V100 T6 feedback)."""
    options = _options()
    starts = _prepared([(0.5, 0.5), (0.25, 0.25)], options=options)
    lines: list[str] = []
    run_confsearch(
        EggCrateCalculator(),
        options,
        starts=starts,
        output_dir=tmp_path / "log",
        seed=7,
        model_record={"model_type": "synthetic", "task": "bulk"},
        gradient_preflight={"status": "passed"},
        log=lines.append,
    )
    text = "\n".join(lines)
    # Every cycle unit announces its MD, reports its quenches and decision.
    for replica in (0, 1):
        for cycle in (1, 2):
            prefix = f"replica {replica} cycle {cycle}:"
            assert f"{prefix} MD " in text
            assert any(
                line.startswith(prefix) and "quenched" in line for line in lines
            ), prefix
    assert "steps/s" in text
    progress = [line for line in lines if line.startswith("progress:")]
    assert len(progress) == 4
    assert progress[-1].startswith("progress: 4/4 cycles done")
    assert "ETA 0:00:00" in progress[-1]
    assert all("best" in line and "elapsed" in line for line in progress)
